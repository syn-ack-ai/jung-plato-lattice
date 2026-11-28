"""MUTABLE TARGET FOR AUTORESEARCH (`exp256_qwen600m_autoresearch/transplant_recipe.py`).
Iteration 14 (`H14_global_frob_L2_20_frozen` — Global Inter-Passage Batched Frobenius + Strictly Frozen L0-1/L21-27!):
  - Hypothesis: Iter 13's Global Inter-Passage Batched Frobenius achieved record alignment (+7.72 / +10.23 pts)
    and 10/40 (85/163) coding, missing `pass: TRUE` by only -2/240 SWE because L23..L27 were unfrozen.
    Strictly freezing L0-1 and L21-27 (as proven in Iter 7) while scaling Global Batched Frobenius to
    `bs = 8` (~45 cross-domain boundary points per step!) with `rank = 24` on `L2..L20` (`o_proj + down_proj`,
    smooth Forward KL `kl_w = 3.20`, ZERO text CE `ce_w = 0.0`!) should hit `pass: TRUE` with high coding/reasoning lift!
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
    def __init__(self, base_linear: nn.Linear, r: int = 24, alpha: float = 24.0):
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


def global_frob_cos_gram_loss(pred, target):
    p_u = F.normalize(pred, dim=-1, eps=1e-6)
    t_u = F.normalize(target, dim=-1, eps=1e-6)
    cos_l = (1.0 - (p_u * t_u).sum(dim=-1)).mean()
    if pred.shape[0] >= 4:
        pc = pred - pred.mean(dim=0, keepdim=True)
        tc = target - target.mean(dim=0, keepdim=True)
        pn = pc / (torch.norm(pc) + 1e-6)
        tn = tc / (torch.norm(tc) + 1e-6)
        frob_l = torch.sum((pn - tn) ** 2)
        gram_l = F.mse_loss(p_u @ p_u.T, t_u @ t_u.T)
    else:
        frob_l = pred.new_tensor(0.0)
        gram_l = pred.new_tensor(0.0)
    return 0.45 * cos_l + 0.90 * frob_l + 0.45 * gram_l


@torch.no_grad()
def fit_native_consensus_frames(model, d_t, layers_keys, pad_id, dev, n_fit=1200):
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

    active_layers = list(range(2, 21))  # L0-1 & L21-27 100% FROZEN for 0 SWE damage!
    rank = 24
    steps = 320
    batch_size = 8  # ~45 global cross-domain points per step!
    lr = 2.4e-4
    kl_w = 3.20
    w30 = 1.65
    w50 = 2.75
    perp_w = 0.85

    frames = fit_native_consensus_frames(
        model, d_t, [(LAYER_30, "r3_30_tr"), (LAYER_50, "r3_50_tr")], pad_id, dev, n_fit=1200
    )

    for p in model.parameters():
        p.requires_grad_(False)

    wrappers = []
    all_lora = []
    for li in active_layers:
        layer = model.model.layers[li]
        for parent, attr in ((layer.self_attn, "o_proj"), (layer.mlp, "down_proj")):
            w = LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((parent, attr, w))
            all_lora.extend([w.lora_A, w.lora_B])

    tr_ids_f, tr_ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    tr_pts_f, tr_pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
    tr_r30, tr_r50 = d_t["r3_30_tr"], d_t["r3_50_tr"]
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

    opt = torch.optim.AdamW(all_lora, lr=lr, weight_decay=0.012, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pass)
    cursor = 0
    history = []
    n_proj = 0

    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 20.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul

            if frac < 0.72:
                pw_scale = 1.0
            elif frac < 0.90:
                pw_scale = 1.0 - (frac - 0.72) / 0.18
            else:
                pw_scale = 0.0

            batch = []
            while len(batch) < batch_size:
                if cursor >= n_pass:
                    perm = rng.permutation(n_pass)
                    cursor = 0
                pi = int(perm[cursor])
                cursor += 1
                ids = tr_ids_f[int(tr_ids_o[pi]):int(tr_ids_o[pi + 1])].astype(np.int64)
                pts = tr_pts_f[int(tr_pts_o[pi]):int(tr_pts_o[pi + 1])].astype(np.int64)
                t30 = tr_r30[int(tr_pts_o[pi]):int(tr_pts_o[pi + 1])]
                t50 = tr_r50[int(tr_pts_o[pi]):int(tr_pts_o[pi + 1])]
                if len(ids) >= 4 and len(pts) >= 1:
                    batch.append((ids, pts, t30, t50))

            ml = max(len(b[0]) for b in batch)
            B = len(batch)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((B, ml), dtype=torch.long, device=dev)
            valid_tok = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
            b_idx_list, t_idx_list, t30_list, t50_list = [], [], [], []
            for bi, (ids, pts, t30, t50) in enumerate(batch):
                L = len(ids)
                inp[bi, :L] = torch.from_numpy(ids).to(dev)
                attn[bi, :L] = 1
                valid_tok[bi, : L - 1] = 1.0
                b_idx_list.append(torch.full((len(pts),), bi, dtype=torch.long, device=dev))
                t_idx_list.append(torch.from_numpy(pts).to(dev))
                t30_list.append(t30)
                t50_list.append(t50)

            b_idx = torch.cat(b_idx_list, dim=0)
            t_idx = torch.cat(t_idx_list, dim=0)
            y30_t = torch.from_numpy(np.concatenate(t30_list, axis=0)).to(dev, dtype=torch.float32)
            y50_t = torch.from_numpy(np.concatenate(t50_list, axis=0)).to(dev, dtype=torch.float32)
            v_sum = valid_tok.sum().clamp_min(1.0)

            for _, _, w in wrappers:
                w.active = False
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_r = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
                lp_ref = F.log_softmax(lg_r.float(), dim=-1)
                p_ref = lp_ref.exp()
                perp_b = {}
                for li in (LAYER_30, LAYER_50):
                    fr = frames[li]
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z = hc @ fr["V_q"]
                    perp_b[li] = (hc - z @ fr["V_q"].T).detach()
            for _, _, w in wrappers:
                w.active = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            lp_cur = F.log_softmax(lg.float(), dim=-1)
            tok_kl = (p_ref * (lp_ref - lp_cur)).sum(dim=-1)
            kl = (tok_kl * valid_tok).sum() / v_sum

            g_pl = None
            p30_val = p50_val = 0.0
            if pw_scale > 0.0:
                loss_geo = lg.new_tensor(0.0, dtype=torch.float32)
                for li, y_t, lw in ((LAYER_30, y30_t, w30), (LAYER_50, y50_t, w50)):
                    fr = frames[li]
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z_cur = hc @ fr["V_q"]
                    h_perp = hc - z_cur @ fr["V_q"].T
                    z_star = (y_t - fr["mu_y"]) @ fr["R_star"]

                    l_sub = global_frob_cos_gram_loss(z_cur, z_star)
                    l_perp = F.mse_loss(h_perp, perp_b[li]) / (fr["rms_x"] ** 2)
                    loss_geo = loss_geo + lw * l_sub + perp_w * l_perp
                    if li == LAYER_30:
                        p30_val = float(l_sub.item())
                    else:
                        p50_val = float(l_sub.item())

                g_pl = torch.autograd.grad(pw_scale * loss_geo, all_lora, retain_graph=True, allow_unused=True)

            g_kl = torch.autograd.grad(kl_w * kl, all_lora, retain_graph=False, allow_unused=True)

            opt.zero_grad(set_to_none=True)
            for i, p in enumerate(all_lora):
                gk = g_kl[i] if g_kl[i] is not None else torch.zeros_like(p)
                gp = (g_pl[i] if (g_pl is not None and g_pl[i] is not None) else torch.zeros_like(p))
                if pw_scale > 0.0 and g_pl is not None and g_pl[i] is not None:
                    dot = torch.sum(gp * gk)
                    if dot < 0.0:
                        denom = torch.sum(gk * gk).clamp_min(1e-12)
                        gp = gp - (dot / denom) * gk
                        n_proj += 1
                p.grad = gk + gp

            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()

            if s % 80 == 0 or s == steps - 1:
                history.append({"step": s, "kl": round(float(kl.item()), 5), "p30": round(p30_val, 4), "p50": round(p50_val, 4), "s": round(pw_scale, 3)})
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

    return {"recipe": "iter14_H14_global_frob_L2_20_frozen", "rank": rank, "steps": steps, "batch_size": batch_size, "pcgrad_projections": n_proj, "train_sec": round(time.time() - t0, 1), "history": history}
