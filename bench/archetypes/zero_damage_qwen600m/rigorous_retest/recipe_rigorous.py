"""Rigorous Native-Frame Shape Transfer & Full 6-Arm Causal Control Suite (Exp 257).

Fixes all methodological & numerical defects identified in PR #2 audit:
  1. Stratified Native PCA Frame (`F5`): Samples equal shares per domain
     (`400 gut + 400 proc + 400 chat = 1,200`, logging exact counts) instead of taking
     the first 1,200 rows in file order (which was 99.7% Gutenberg!).
  2. Exact FP32 Merge & Exception-Safe Cleanup (`F9`): Base model lives in exact fp32;
     adapters merge ONLY upon clean completion (original modules restored on error!).
     Verifies & logs exact `KL(unmerged || merged)` on 16 held-out passages (~0.0).
  3. Proper 6-Arm Causal Control Ladder (`F3`, `F8`):
       - `kl_only`: w30=0, w50=0, perp_w=0 (verifies ~zero drift)
       - `random_lora_matched`: isotropic rank-24 LoRA on the EXACT same L2..L20
         `o_proj + down_proj` matrices, scaled to match the EXACT per-layer `||ΔW_l||_F`
         of `true_stratified_r3` on the same seed!
       - `shuffle_within_domain`: permutes target rows STRICTLY within each
         `(domain, segment_index)` stratum (preserving domain clustering, segment
         trajectory, and marginal norms while destroying passage correspondence!),
         refitting `R*` on the within-stratum-shuffled targets.
       - `true_gut_only`: audited Project Gutenberg ONLY (`tr_domains == 'gut'` with
         strict regex check confirming zero shell/code/URL lines), PCA fit on `gut` only.
       - `true_stratified_r3`: stratified 3-domain `R3` rival consensus (`Gemma 4 +
         Llama 3.1 + Mistral`, zero Qwen, `ce_w = 0.0`, zero code training).
       - `true_dual_domain`: 75% stratified `R3` prose + 25% 8B rival Gemma 4 Python
         stdlib code geometry (`ce_w = 0.0`).
"""
from collections import Counter, defaultdict
import gc
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_30 = 8
LAYER_50 = 14
ACTIVE_LAYERS = list(range(2, 21))  # L0-1 and L21-27 strictly frozen

# Strict shell / code / technical-markup audit regex for `true_gut_only`
CODE_SHELL_AUDIT_RE = re.compile(
    r"(```|pip install|sudo |#!/|SELECT\s+\w+\s+FROM|\bdef\s+\w+\s*\(|\bimport\s+\w+|"
    r"https?://|\b(?:bash|zsh|chmod|grep|curl|kubectl)\b)",
    flags=re.IGNORECASE,
)


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
    def delta_weight_fp32(self):
        return (self.lora_B @ self.lora_A) * self.scale

    @torch.no_grad()
    def merge_into_base_fp32(self):
        dw = self.delta_weight_fp32()
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


def select_stratified_indices(domains, n_fit: int = 1200, seed: int = 2026):
    """Stratified equal-per-domain passage sampling for Native PCA Frame fitting (`F5` fix!)."""
    rng = np.random.default_rng(seed)
    by_dom = defaultdict(list)
    for i, d in enumerate(domains):
        by_dom[str(d)].append(i)
    doms = sorted(by_dom.keys())
    per_dom = n_fit // max(1, len(doms))
    picked = []
    for d in doms:
        idxs = np.array(by_dom[d], dtype=np.int64)
        rng.shuffle(idxs)
        picked.extend(idxs[:per_dom].tolist())
    # Top up if any domain had fewer than per_dom
    if len(picked) < n_fit:
        picked_set = set(picked)
        rem = [i for i in range(len(domains)) if i not in picked_set]
        rng.shuffle(rem)
        picked.extend(rem[: n_fit - len(picked)])
    picked = sorted(picked[:n_fit])
    dom_counts = dict(Counter(str(domains[i]) for i in picked))
    return picked, dom_counts


def within_domain_segment_shuffle(r30, r50, pts_off, tr_domains, seed: int):
    """Permute target rows STRICTLY within each `(domain, segment_index)` stratum!
    Preserves domain clustering, trajectory position, and marginal norm distribution
    while breaking passage-level correspondence (`F3` fix!)."""
    rng = np.random.default_rng(seed + 7777)
    r30_shuf = r30.copy()
    r50_shuf = r50.copy()
    buckets = defaultdict(list)
    n_pass = len(pts_off) - 1
    for pi in range(n_pass):
        dom = str(tr_domains[pi])
        st, en = int(pts_off[pi]), int(pts_off[pi + 1])
        for seg_idx, row_i in enumerate(range(st, en)):
            buckets[(dom, seg_idx)].append(row_i)
    bucket_sizes = {}
    for key, idxs in sorted(buckets.items()):
        arr = np.array(idxs, dtype=np.int64)
        perm = rng.permutation(arr)
        r30_shuf[arr] = r30[perm]
        r50_shuf[arr] = r50[perm]
        bucket_sizes[f"{key[0]}_s{key[1]}"] = len(idxs)
    return r30_shuf, r50_shuf, bucket_sizes


def get_audited_gut_indices(d_t, tok):
    """Return passage indices in `d_t` that are strictly `gut` AND pass the zero-code/shell regex audit (`F8`)."""
    tr_domains = d_t["tr_domains"]
    ids_f, ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    kept, rejected = [], 0
    for pi in range(len(tr_domains)):
        if str(tr_domains[pi]) != "gut":
            continue
        ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].tolist()
        txt = tok.decode(ids, skip_special_tokens=True)
        if CODE_SHELL_AUDIT_RE.search(txt):
            rejected += 1
        else:
            kept.append(pi)
    return kept, rejected


@torch.no_grad()
def fit_native_frames_indexed(model, ids_f, ids_o, pts_f, pts_o, tgts_dict, fit_indices, pad_id, dev):
    layers = list(tgts_dict.keys())
    cap = {}
    hooks = []
    for li in layers:
        def _make(idx):
            def _hk(_m, _i, out):
                cap[idx] = out[0] if isinstance(out, tuple) else out
            return _hk
        hooks.append(model.model.layers[li].register_forward_hook(_make(li)))

    Xs = {li: [] for li in layers}
    Ys = {li: [] for li in layers}
    try:
        for s in range(0, len(fit_indices), 32):
            sub_idx = fit_indices[s : s + 32]
            b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64) for i in sub_idx]
            b_pts = [pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64) for i in sub_idx]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            for li in layers:
                h = cap[li].float().cpu().numpy()
                tgt_all = tgts_dict[li]
                for bi, (i_abs, pts) in enumerate(zip(sub_idx, b_pts)):
                    Xs[li].append(h[bi, pts, :])
                    Ys[li].append(tgt_all[int(pts_o[i_abs]):int(pts_o[i_abs + 1])])
    finally:
        for hk in hooks:
            hk.remove()

    frames = {}
    for li in layers:
        X = np.concatenate(Xs[li], axis=0).astype(np.float64)
        Y = np.concatenate(Ys[li], axis=0).astype(np.float64)
        mu_x = X.mean(axis=0, keepdims=True)
        Xc = X - mu_x
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(Y.shape[1], vt.shape[0])
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


@torch.no_grad()
def _measure_merge_kl_fp32(model, wrappers, d_t, pad_id, dev, n_check=16, mini_bs=4):
    """Mechanically verify KL(unmerged_fp32 || merged_fp32) after exact fp32 merge (`F9`)."""
    ev_f, ev_o = d_t["ev_ids_flat"], d_t["ev_ids_off"]
    batches = []
    for s in range(0, n_check, mini_bs):
        b_ids = [ev_f[int(ev_o[i]):int(ev_o[i + 1])].astype(np.int64) for i in range(s, min(n_check, s + mini_bs))]
        ml = max(len(x) for x in b_ids)
        inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long)
        attn = torch.zeros((len(b_ids), ml), dtype=torch.long)
        valid = torch.zeros((len(b_ids), ml - 1), dtype=torch.float32)
        for bi, ids in enumerate(b_ids):
            inp[bi, : len(ids)] = torch.from_numpy(ids)
            attn[bi, : len(ids)] = 1
            valid[bi, : len(ids) - 1] = 1.0
        batches.append((inp, attn, valid))

    # Capture unmerged logits on CPU (tiny: 16 short passages!) before merging
    unmerged_cpu = []
    for inp_c, attn_c, _ in batches:
        lg_u = model(input_ids=inp_c.to(dev), attention_mask=attn_c.to(dev), use_cache=False).logits[:, :-1, :].float()
        unmerged_cpu.append(lg_u.cpu())
        del lg_u
    torch.cuda.empty_cache()

    # Record per-layer Frobenius norms and merge cleanly in fp32
    fro_norms = {}
    for li_tag, parent, attr, w in wrappers:
        fro_norms[li_tag] = float(torch.norm(w.delta_weight_fp32(), p="fro").item())
        setattr(parent, attr, w.merge_into_base_fp32())

    tot_kl = 0.0
    tot_tok = 0.0
    max_logit_diff = 0.0
    for (inp_c, attn_c, valid_c), lg_u_c in zip(batches, unmerged_cpu):
        lg_m = model(input_ids=inp_c.to(dev), attention_mask=attn_c.to(dev), use_cache=False).logits[:, :-1, :].float()
        lg_u = lg_u_c.to(dev)
        max_logit_diff = max(max_logit_diff, float((lg_u - lg_m).abs().max().item()))
        lp_u = F.log_softmax(lg_u, dim=-1)
        lp_m = F.log_softmax(lg_m, dim=-1)
        tok_kl = (lp_u.exp() * (lp_u - lp_m)).sum(dim=-1)
        v = valid_c.to(dev)
        tot_kl += float((tok_kl * v).sum().item())
        tot_tok += float(v.sum().item())
        del lg_m, lg_u, lp_u, lp_m, tok_kl, v
    del unmerged_cpu
    torch.cuda.empty_cache()
    return tot_kl / max(1.0, tot_tok), max_logit_diff, fro_norms


@torch.no_grad()
def apply_random_lora_matched(model, target_fro_norms: dict, seed: int, rank: int = 24):
    """Matched-Norm Random LoRA Control (`F3`): applies an isotropic rank-24 random
    update to EVERY L2..L20 `o_proj + down_proj` matrix scaled to match the EXACT
    per-layer Frobenius norm `||ΔW_l||_F` of `true_stratified_r3` on the same seed."""
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed + 90909)
    applied_norms = {}
    for li in ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mod_name, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            tag = f"L{li}.{mod_name}"
            tgt_norm = float(target_fro_norms.get(tag, 0.0))
            lin = getattr(parent, attr)
            out_f, in_f = lin.weight.shape
            A = torch.randn((rank, in_f), generator=gen, dtype=torch.float32)
            B = torch.randn((out_f, rank), generator=gen, dtype=torch.float32)
            dw = B @ A
            cur_norm = float(torch.norm(dw, p="fro").item())
            if cur_norm > 0.0 and tgt_norm > 0.0:
                dw = dw * (tgt_norm / cur_norm)
            lin.weight.data.add_(dw.to(device=lin.weight.device, dtype=lin.weight.dtype))
            applied_norms[tag] = float(torch.norm(dw, p="fro").item())
    model.eval()
    return {
        "arm": "random_lora_matched",
        "seed": seed,
        "rank": rank,
        "delta_fro_norms": applied_norms,
        "total_fro_norm": round(float(np.sqrt(sum(v * v for v in applied_norms.values()))), 6),
        "train_sec": round(time.time() - t0, 2),
    }


def run_arm(model, tok, d_t, arm: str, seed: int, code_targets_npz: Path = None, matched_fro_norms: dict = None):
    """Execute one causal-ladder arm cleanly on an fp32 base model."""
    if arm == "random_lora_matched":
        assert matched_fro_norms is not None, "random_lora_matched requires target per-layer Frobenius norms!"
        return apply_random_lora_matched(model, matched_fro_norms, seed=seed, rank=24)

    assert arm in ("kl_only", "shuffle_within_domain", "true_gut_only", "true_stratified_r3", "true_dual_domain"), f"Unknown arm: {arm}"

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rank = 24
    steps = 360
    batch_size = 8
    lr = 2.4e-4
    kl_w = 3.20
    w30 = 0.0 if arm == "kl_only" else 1.65
    w50 = 0.0 if arm == "kl_only" else 2.75
    perp_w = 0.0 if arm == "kl_only" else 0.85

    tr_domains = d_t["tr_domains"]
    r30_tr = d_t["r3_30_tr"]
    r50_tr = d_t["r3_50_tr"]
    extra_meta = {}

    if arm == "shuffle_within_domain":
        r30_tr, r50_tr, b_sizes = within_domain_segment_shuffle(r30_tr, r50_tr, d_t["tr_pts_off"], tr_domains, seed=seed)
        extra_meta["shuffle_strata_count"] = len(b_sizes)
        train_pool = list(range(len(tr_domains)))
        fit_idx, fit_dom_counts = select_stratified_indices(tr_domains, n_fit=1200, seed=2026)
    elif arm == "true_gut_only":
        train_pool, n_rej = get_audited_gut_indices(d_t, tok)
        extra_meta["gut_audited_kept"] = len(train_pool)
        extra_meta["gut_regex_rejected"] = n_rej
        fit_idx = list(train_pool)
        fit_dom_counts = {"gut": len(fit_idx)}
    else:
        train_pool = list(range(len(tr_domains)))
        fit_idx, fit_dom_counts = select_stratified_indices(tr_domains, n_fit=1200, seed=2026)

    extra_meta["pca_domain_counts"] = fit_dom_counts
    prose_frames = fit_native_frames_indexed(
        model, d_t["tr_ids_flat"], d_t["tr_ids_off"], d_t["tr_pts_flat"], d_t["tr_pts_off"],
        {LAYER_30: r30_tr, LAYER_50: r50_tr}, fit_idx, pad_id, dev,
    )

    d_c, code_frames, code_pool = None, None, None
    if arm == "true_dual_domain":
        d_c = np.load(code_targets_npz, allow_pickle=True)
        n_code = len(d_c["code_ids_off"]) - 1
        code_pool = list(range(n_code))
        code_frames = fit_native_frames_indexed(
            model, d_c["code_ids_flat"], d_c["code_ids_off"], d_c["code_pts_flat"], d_c["code_pts_off"],
            {LAYER_30: d_c["g4_code_30"], LAYER_50: d_c["g4_code_50"]}, code_pool, pad_id, dev,
        )
        extra_meta["code_n_snippets"] = n_code

    for p in model.parameters():
        p.requires_grad_(False)

    wrappers = []
    all_lora = []
    for li in ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mod_name, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            w = LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((f"L{li}.{mod_name}", parent, attr, w))
            all_lora.extend([w.lora_A, w.lora_B])

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

    pool_arr = np.array(train_pool, dtype=np.int64)
    perm_p, cur_p = rng.permutation(pool_arr), 0
    if code_pool is not None:
        code_arr = np.array(code_pool, dtype=np.int64)
        perm_c, cur_c = rng.permutation(code_arr), 0

    history = []
    n_proj = 0
    completed = False
    merge_kl = max_logit_diff = 0.0
    fro_norms = {}

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

            is_code_step = (arm == "true_dual_domain") and (s % 4 == 3)
            if not is_code_step:
                ids_f, ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
                pts_f, pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
                t30_all, t50_all = r30_tr, r50_tr
                frames = prose_frames
                batch = []
                while len(batch) < batch_size:
                    if cur_p >= len(perm_p):
                        perm_p, cur_p = rng.permutation(pool_arr), 0
                    pi = int(perm_p[cur_p]); cur_p += 1
                    ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                    pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                    t30 = t30_all[int(pts_o[pi]):int(pts_o[pi + 1])]
                    t50 = t50_all[int(pts_o[pi]):int(pts_o[pi + 1])]
                    if len(ids) >= 4 and len(pts) >= 1:
                        batch.append((ids, pts, t30, t50))
            else:
                ids_f, ids_o = d_c["code_ids_flat"], d_c["code_ids_off"]
                pts_f, pts_o = d_c["code_pts_flat"], d_c["code_pts_off"]
                t30_all, t50_all = d_c["g4_code_30"], d_c["g4_code_50"]
                frames = code_frames
                batch = []
                while len(batch) < batch_size:
                    if cur_c >= len(perm_c):
                        perm_c, cur_c = rng.permutation(code_arr), 0
                    pi = int(perm_c[cur_c]); cur_c += 1
                    ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                    pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                    t30 = t30_all[int(pts_o[pi]):int(pts_o[pi + 1])]
                    t50 = t50_all[int(pts_o[pi]):int(pts_o[pi + 1])]
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

            for _, _, _, w in wrappers:
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
            for _, _, _, w in wrappers:
                w.active = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            lp_cur = F.log_softmax(lg.float(), dim=-1)
            tok_kl = (p_ref * (lp_ref - lp_cur)).sum(dim=-1)
            kl = (tok_kl * valid_tok).sum() / v_sum

            g_pl = None
            p30_val = p50_val = 0.0
            if pw_scale > 0.0 and (w30 > 0.0 or w50 > 0.0 or perp_w > 0.0):
                loss_geo = lg.new_tensor(0.0, dtype=torch.float32)
                dom_mul = 0.75 if is_code_step else 1.0
                for li, y_t, lw in ((LAYER_30, y30_t, w30 * dom_mul), (LAYER_50, y50_t, w50 * dom_mul)):
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
                if g_pl is not None and g_pl[i] is not None:
                    dot = torch.sum(gp * gk)
                    if dot < 0.0:
                        denom = torch.sum(gk * gk).clamp_min(1e-12)
                        gp = gp - (dot / denom) * gk
                        n_proj += 1
                p.grad = gk + gp

            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()

            if s % 90 == 0 or s == steps - 1:
                history.append({
                    "step": s,
                    "dom": "code" if is_code_step else "prose",
                    "kl": round(float(kl.item()), 6),
                    "p30": round(p30_val, 4),
                    "p50": round(p50_val, 4),
                    "s": round(pw_scale, 3),
                })
        completed = True
    finally:
        for hk in hooks:
            hk.remove()
        cap.clear()
        opt.zero_grad(set_to_none=True)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if completed:
            merge_kl, max_logit_diff, fro_norms = _measure_merge_kl_fp32(model, wrappers, d_t, pad_id, dev)
        else:
            # Exception-safe rollback: restore unmodified base modules (`F9` fix!)
            for _, parent, attr, w in wrappers:
                setattr(parent, attr, w.base)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        del prose_frames, code_frames, wrappers
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "arm": arm,
        "seed": seed,
        "ce_w": 0.0,
        "rank": rank,
        "steps": steps,
        "batch_size": batch_size,
        "pcgrad_projections": n_proj,
        "merge_kl_fp32": round(merge_kl, 9),
        "merge_max_logit_diff_fp32": round(max_logit_diff, 8),
        "total_fro_norm": round(float(np.sqrt(sum(v * v for v in fro_norms.values()))), 6),
        "delta_fro_norms": {k: round(v, 6) for k, v in fro_norms.items()},
        "train_sec": round(time.time() - t0, 1),
        "history": history,
        **extra_meta,
    }
