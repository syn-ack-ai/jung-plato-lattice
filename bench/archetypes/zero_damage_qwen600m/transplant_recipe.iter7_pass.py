"""MUTABLE TARGET FOR AUTORESEARCH (`exp256_qwen600m_autoresearch/transplant_recipe.py`).
Iteration 7 (`H7_native_js_protect_readout`):
  - Hypothesis: Iter 6's Native-Frame Dual-Stratum Whole-Model Realignment exploded quality to +11.70
    (Code 8->11/40 +7.5pp, Actor +5.0pp, Proc +3.5pp, NLL -0.0091, L8 +6.58pts & L14 +8.84pts, 0 CE!),
    missing `pass: true` ONLY by -3/240 (-1.25pp) on SWE-Smith code syntax NLL because layers L21..L27
    (final token-probability readout) were modified under one-sided forward KL.
  - Fix (still ZERO text CE `ce_w = 0.0`, ZERO external `plato_head`!):
      1. Protect lexical/syntax ingress (`L0..L1`) and final token-probability readout (`L21..L27`)
         by keeping them 100% FROZEN; cooperatively realign the interior 19-layer reasoning stack `L2..L20`
         (`L2..L14` native-frame 30%+50% consensus denoising, `L15..L20` cooperative readout harmonization).
      2. Replace one-sided forward KL with **Symmetric Bidirectional KL** `0.5*KL(p_ref||p_cur) + 0.5*KL(p_cur||p_ref)`:
         the mode-seeking reverse-KL term strictly prevents leaking probability mass onto buggy code tokens!
      3. Double the 768D orthogonal complement lock (`perp_w = 1.20`) protecting OOD syntax subspaces!
"""
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_30 = 8
LAYER_50 = 14


class LoRALinearWrap(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        dev = base_linear.weight.device
        self.scale = alpha / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.active = True

    def forward(self, x):
        out = self.base(x)
        if self.active:
            xf = x.float()
            delta = (xf @ self.lora_A.T) @ self.lora_B.T
            out = out + (delta * self.scale).to(out.dtype)
        return out

    @torch.no_grad()
    def merge_into_base(self):
        dw = (self.lora_B @ self.lora_A) * self.scale
        self.base.weight.data.add_(dw.to(self.base.weight.dtype))
        return self.base


def centered_frob_cos_gram_loss(pred, target):
    p_u = F.normalize(pred, dim=-1, eps=1e-6)
    t_u = F.normalize(target, dim=-1, eps=1e-6)
    cos_l = (1.0 - (p_u * t_u).sum(dim=-1)).mean()
    if pred.shape[0] >= 3:
        pc = pred - pred.mean(dim=0, keepdim=True)
        tc = target - target.mean(dim=0, keepdim=True)
        pn = pc / (torch.norm(pc) + 1e-6)
        tn = tc / (torch.norm(tc) + 1e-6)
        frob_l = torch.sum((pn - tn) ** 2)
        gram_l = F.mse_loss(p_u @ p_u.T, t_u @ t_u.T)
    else:
        frob_l = pred.new_tensor(0.0)
        gram_l = pred.new_tensor(0.0)
    return 0.50 * cos_l + 0.80 * frob_l + 0.40 * gram_l


@torch.no_grad()
def fit_native_consensus_frames(model, d_t, layers_keys, pad_id, dev, n_fit=1000):
    tr_ids_f, tr_ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    tr_pts_f, tr_pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
    n_pass = min(n_fit, len(tr_ids_o) - 1)

    cap = {}
    hooks = []
    for li, _ in layers_keys:
        def _make(idx):
            def _hk(_m, _i, out):
                cap[idx] = out[0] if isinstance(out, tuple) else out
            return _hk
        hooks.append(model.model.layers[li].register_forward_hook(_make(li)))

    Xs = {li: [] for li, _ in layers_keys}
    Ys = {li: [] for li, _ in layers_keys}
    try:
        for s in range(0, n_pass, 32):
            e = min(n_pass, s + 32)
            b_ids = [tr_ids_f[int(tr_ids_o[i]):int(tr_ids_o[i + 1])].astype(np.int64) for i in range(s, e)]
            b_pts = [tr_pts_f[int(tr_pts_o[i]):int(tr_pts_o[i + 1])].astype(np.int64) for i in range(s, e)]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            for li, tkey in layers_keys:
                h = cap[li].float().cpu().numpy()
                tgt_all = d_t[tkey]
                for bi, pts in enumerate(b_pts):
                    Xs[li].append(h[bi, pts, :])
                    i_abs = s + bi
                    Ys[li].append(tgt_all[int(tr_pts_o[i_abs]):int(tr_pts_o[i_abs + 1])])
    finally:
        for hk in hooks:
            hk.remove()

    frames = {}
    for li, _ in layers_keys:
        X = np.concatenate(Xs[li], axis=0).astype(np.float64)
        Y = np.concatenate(Ys[li], axis=0).astype(np.float64)
        mu_x = X.mean(axis=0, keepdims=True)
        Xc = X - mu_x
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        k = Y.shape[1]
        V_q = vt[:k].T
        Zx = Xc @ V_q
        s_x = float(np.sqrt(np.mean(np.sum(Zx ** 2, axis=1)))) + 1e-12
        rms_x = float(np.sqrt(np.mean(Xc ** 2))) + 1e-12

        mu_y = Y.mean(axis=0, keepdims=True)
        Yc = Y - mu_y
        s_y = float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1)))) + 1e-12

        M = (Yc / s_y).T @ (Zx / s_x)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R_star = U @ Vt

        frames[li] = {
            "mu_x": torch.from_numpy(mu_x.squeeze(0).astype(np.float32)).to(dev),
            "V_q": torch.from_numpy(V_q.astype(np.float32)).to(dev),
            "mu_y": torch.from_numpy(mu_y.squeeze(0).astype(np.float32)).to(dev),
            "R_star": torch.from_numpy((R_star * (s_x / s_y)).astype(np.float32)).to(dev),
            "rms_x": rms_x,
        }
    return frames


def run_transplant(model, tok, d_t, seed=256):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Freeze lexical ingress (L0-L1) and final code-syntax readout (L21-L27); realign L2..L20!
    active_layers = list(range(2, 21))
    rank = 16
    steps = 260
    grad_accum = 4
    lr = 2.6e-4
    kl_w = 3.50
    w30 = 1.60
    w50 = 2.60
    perp_w = 1.20  # Tight 768D orthogonal syntax complement lock!

    frames = fit_native_consensus_frames(
        model, d_t, [(LAYER_30, "r3_30_tr"), (LAYER_50, "r3_50_tr")], pad_id, dev, n_fit=1000
    )

    for p in model.parameters():
        p.requires_grad_(False)

    wrappers = []
    lora_params = []
    for li in active_layers:
        layer = model.model.layers[li]
        for parent, attr in ((layer.self_attn, "o_proj"), (layer.mlp, "down_proj")):
            w = LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((parent, attr, w))
            lora_params.extend([w.lora_A, w.lora_B])

    tr_ids_f, tr_ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    tr_pts_f, tr_pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
    tr_r30 = d_t["r3_30_tr"]
    tr_r50 = d_t["r3_50_tr"]
    n_pass = len(tr_ids_o) - 1

    cap = {}
    def _make_hk(idx):
        def _hk(_m, _i, out):
            cap[idx] = out[0] if isinstance(out, tuple) else out
        return _hk
    hooks = [
        model.model.layers[LAYER_30].register_forward_hook(_make_hk(LAYER_30)),
        model.model.layers[LAYER_50].register_forward_hook(_make_hk(LAYER_50)),
    ]

    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.015, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    history = []
    n_proj = 0

    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 15.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul

            if frac < 0.72:
                pw_scale = 1.0
            elif frac < 0.90:
                pw_scale = 1.0 - (frac - 0.72) / 0.18
            else:
                pw_scale = 0.0

            idxs = rng.integers(0, n_pass, size=grad_accum)
            batch = []
            for pi in idxs:
                ids = tr_ids_f[int(tr_ids_o[pi]):int(tr_ids_o[pi + 1])].astype(np.int64)
                pts = tr_pts_f[int(tr_pts_o[pi]):int(tr_pts_o[pi + 1])].astype(np.int64)
                t30 = tr_r30[int(tr_pts_o[pi]):int(tr_pts_o[pi + 1])]
                t50 = tr_r50[int(tr_pts_o[pi]):int(tr_pts_o[pi + 1])]
                if len(ids) >= 4 and len(pts) >= 1:
                    batch.append((ids, pts, t30, t50))
            n_acc = len(batch)

            for _, _, w in wrappers:
                w.active = False
            refs = []
            with torch.no_grad():
                for g_ids, g_pts, _t30, _t50 in batch:
                    x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg_r = model(input_ids=x, use_cache=False).logits[:, :-1, :]
                    t_idx = torch.from_numpy(g_pts).to(dev)
                    perp_b = {}
                    for li in (LAYER_30, LAYER_50):
                        fr = frames[li]
                        hc = cap[li][0, t_idx, :].float() - fr["mu_x"]
                        z = hc @ fr["V_q"]
                        perp_b[li] = (hc - z @ fr["V_q"].T).detach()
                    refs.append((F.log_softmax(lg_r.float(), dim=-1), perp_b))
            for _, _, w in wrappers:
                w.active = True

            g_kl_acc = [torch.zeros_like(p) for p in lora_params]
            g_pl_acc = [torch.zeros_like(p) for p in lora_params]
            sum_kl = sum_p30 = sum_p50 = 0.0

            for (g_ids, g_pts, g_t30, g_t50), (lp_ref, perp_b) in zip(batch, refs):
                x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(input_ids=x, use_cache=False).logits[:, :-1, :]
                lp_cur = F.log_softmax(lg.float(), dim=-1)
                p_ref = lp_ref.exp()
                p_cur = lp_cur.exp()
                # Symmetric bidirectional KL (Jeffreys / mode-seeking + mass-covering!)
                kl_fwd = (p_ref * (lp_ref - lp_cur)).sum(dim=-1).mean()
                kl_rev = (p_cur * (lp_cur - lp_ref)).sum(dim=-1).mean()
                kl_sym = 0.5 * (kl_fwd + kl_rev)
                sum_kl += float(kl_sym.item()) / n_acc

                if pw_scale > 0.0:
                    t_idx = torch.from_numpy(g_pts).to(dev)
                    loss_geo = lg.new_tensor(0.0, dtype=torch.float32)
                    for li, g_tgt, lw in ((LAYER_30, g_t30, w30), (LAYER_50, g_t50, w50)):
                        fr = frames[li]
                        hc = cap[li][0, t_idx, :].float() - fr["mu_x"]
                        z_cur = hc @ fr["V_q"]
                        h_perp = hc - z_cur @ fr["V_q"].T

                        y_t = torch.from_numpy(g_tgt).to(dev, dtype=torch.float32)
                        z_star = (y_t - fr["mu_y"]) @ fr["R_star"]

                        l_sub = centered_frob_cos_gram_loss(z_cur, z_star)
                        l_perp = F.mse_loss(h_perp, perp_b[li]) / (fr["rms_x"] ** 2)
                        loss_geo = loss_geo + lw * l_sub + perp_w * l_perp
                        if li == LAYER_30:
                            sum_p30 += float(l_sub.item()) / n_acc
                        else:
                            sum_p50 += float(l_sub.item()) / n_acc

                    grads_p = torch.autograd.grad(
                        (pw_scale / n_acc) * loss_geo, lora_params, retain_graph=True, allow_unused=True
                    )
                    for i, g in enumerate(grads_p):
                        if g is not None:
                            g_pl_acc[i].add_(g)

                grads_k = torch.autograd.grad((kl_w / n_acc) * kl_sym, lora_params, retain_graph=False, allow_unused=True)
                for i, g in enumerate(grads_k):
                    if g is not None:
                        g_kl_acc[i].add_(g)
            del refs

            opt.zero_grad(set_to_none=True)
            for i, p in enumerate(lora_params):
                gk = g_kl_acc[i]
                gp = g_pl_acc[i]
                if pw_scale > 0.0:
                    dot = torch.sum(gp * gk)
                    if dot < 0.0:
                        denom = torch.sum(gk * gk).clamp_min(1e-12)
                        gp = gp - (dot / denom) * gk
                        n_proj += 1
                p.grad = gk + gp

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()

            if s % 50 == 0 or s == steps - 1:
                history.append({"step": s, "kl": round(sum_kl, 5), "p30": round(sum_p30, 4), "p50": round(sum_p50, 4), "s": round(pw_scale, 3)})
    finally:
        for hk in hooks:
            hk.remove()
        for parent, attr, w in wrappers:
            setattr(parent, attr, w.merge_into_base())
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        del frames, wrappers
        gc.collect()
        torch.cuda.empty_cache()

    return {"recipe": "iter7_H7_native_js_protect_readout", "rank": rank, "active_layers": [2, 20], "steps": steps, "pcgrad_projections": n_proj, "train_sec": round(time.time() - t0, 1), "history": history}
