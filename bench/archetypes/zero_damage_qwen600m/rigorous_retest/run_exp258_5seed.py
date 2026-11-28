"""Experiment 258 (5-Seed Rigorous Replication): Python-Coding Platonic Lattice vs.
Matched-Norm Random LoRA, Within-Segment Shuffled Control, and Plain Next-Token CE.

Addresses external audit follow-up #2:
  - Runs 5 fresh paired seeds {101, 102, 103, 104, 105} on exact FP32 Qwen/Qwen3-0.6B
    across 4 arms (plus unmodified Base):
      1. `H2_r3_tri_trustkl`: Tri-stratum 30/50/70% 3-family code GPA (`ce_w=0.0`).
      2. `random_lora_matched`: Isotropic rank-24 LoRA on exact same L2..L24
         `o_proj + down_proj` matched per-layer to `H2_r3_tri_trustkl`'s `||ΔW_l||_F`.
      3. `H5_tri_SHUFFLED_ctrl`: Within-segment-index shuffled target control of H2.
      4. `plain_ce_code`: Plain next-token CE (`ce_w=0.50`) + exact same trust-region
         hinge KL on the exact same 1,997 decontaminated Python functions (zero geometry).
  - Evaluates strictly on disjoint dev benchmarks (`bwrap --unshare-all` + `prlimit`):
      * `dev_code_nll` on all 399 held-out functions (persisting per-function NLLs)
      * `dev_mbpp_170` (`task_id < 11 or task_id > 510`, persisting per-task bools)
      * Blind `Phi-4-mini` & `OLMo 3 7B` OOF `r2_frob` at 30% (L8), 50% (L14), 70% (L19)
  - Computes sample SDs (`ddof=1`), paired `df=4` 95% t-CIs, and 10,000-resample paired
    bootstraps over the 399 held-out functions and 170 disjoint MBPP dev tasks.
"""
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_dev_code_cuda as base258  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]
T_CRIT_95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706}
ACTIVE_LAYERS = list(range(2, 25))
MODULES = ("o_proj", "down_proj")


def sample_stats(vals):
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    tc = T_CRIT_95.get(n - 1, 1.96) if n > 1 else 0.0
    return {
        "n": n, "mean": round(mean, 6), "sd_ddof1": round(sd, 6),
        "ci95_lo": round(mean - tc * se, 6), "ci95_hi": round(mean + tc * se, 6),
        "values": [round(float(x), 6) for x in arr],
    }


def paired_seed_contrast(va, vb):
    diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
    st = sample_stats(diff)
    n = len(diff)
    st["t_stat"] = round(float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n))), 4) if (n > 1 and st["sd_ddof1"] > 0) else 0.0
    st["df"] = max(0, n - 1)
    st["excludes_zero"] = bool(st["ci95_lo"] > 0.0 or st["ci95_hi"] < 0.0)
    return st


def bootstrap_diff(diffs, n_boot=10000, seed=258):
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    b = arr[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p_two = float(2.0 * min(np.mean(b <= 0.0), np.mean(b >= 0.0)))
    return {
        "n_items": len(arr),
        "mean_diff": round(float(arr.mean()), 6),
        "boot_ci95_lo": round(float(lo), 6),
        "boot_ci95_hi": round(float(hi), 6),
        "boot_p_two_sided": round(min(1.0, p_two), 5),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


@torch.no_grad()
def evaluate_dev_per_item(model, tok, d_c, mbpp_dev_tasks):
    """Exact fp32 eval returning per-function NLLs (399) + per-task MBPP bools (170) + blind OOF."""
    t0 = time.time()
    model.eval()
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    ev_f, ev_o = d_c["ev_ids_flat"], d_c["ev_ids_off"]
    pt_f, pt_o = d_c["ev_pts_flat"], d_c["ev_pts_off"]
    n_ev = len(ev_o) - 1
    cap = {}
    hooks = [
        model.model.layers[li].register_forward_hook(
            lambda _m, _i, out, idx=li: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for li in base258.LAYERS_MAP.values()
    ]
    H = {dep: [] for dep in base258.LAYERS_MAP}
    per_fn_nll = []
    tot_nll, tot_tok = 0.0, 0
    try:
        for s in range(0, n_ev, 8):
            e = min(n_ev, s + 8)
            b_ids = [ev_f[int(ev_o[i]):int(ev_o[i + 1])].astype(np.int64) for i in range(s, e)]
            b_pts = [pt_f[int(pt_o[i]):int(pt_o[i + 1])].astype(np.int64) for i in range(s, e)]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
            for bi, ids in enumerate(b_ids):
                L = len(ids) - 1
                fn_sum = float(F.cross_entropy(logits[bi, :L].float(), inp[bi, 1 : L + 1], reduction="sum").item())
                tot_nll += fn_sum
                tot_tok += int(L)
                per_fn_nll.append({"idx": int(s + bi), "nll": round(fn_sum / max(1, L), 6), "n_tok": int(L)})
            for dep, li in base258.LAYERS_MAP.items():
                arr = cap[li].float().cpu().numpy()
                for bi, pts in enumerate(b_pts):
                    H[dep].append(arr[bi, pts, :])
            del logits, inp, attn
    finally:
        for hk in hooks:
            hk.remove()
    torch.cuda.empty_cache()

    grps = d_c["ev_pt_groups"].tolist()
    uniq = sorted(set(grps))
    rng = random.Random(2588); rng.shuffle(uniq)
    g_fold = {g: (0 if i < len(uniq) // 2 else 1) for i, g in enumerate(uniq)}
    folds = np.array([g_fold[g] for g in grps], dtype=np.int32)
    align = {}
    for dep in (30, 50, 70):
        X = np.concatenate(H[dep], axis=0).astype(np.float64)
        vals = []
        for bk in ("phi4_mini", "olmo3_7b"):
            Y = d_c[f"{bk}_{dep}_ev"].astype(np.float64)
            pa, ta = [], []
            for tf in (0, 1):
                tr, te = folds != tf, folds == tf
                Z_tr, Z_te = base258._pca_proj(X[tr], X[te], d_out=256)
                pa.append(base258._apply_proc(Z_te, base258._fit_proc(Z_tr, Y[tr])))
                ta.append(Y[te])
            pc, tc = np.concatenate(pa), np.concatenate(ta)
            r2 = 1.0 - float(np.sum((tc - pc) ** 2)) / max(1e-12, float(np.sum((tc - tc.mean(0, keepdims=True)) ** 2)))
            vals.append(r2)
        align[f"code_blind_{dep}_r2"] = round(float(np.mean(vals)), 5)

    orig_side = tok.padding_side
    tok.padding_side = "left"
    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    prompts = []
    for t in mbpp_dev_tasks:
        sigs = base258.extract_mbpp_sigs(t["code"])
        imp = ("\nRequired imports:\n" + "\n".join(t["test_imports"])) if t.get("test_imports") else ""
        prompts.append(
            tok.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{sigs}\n```{imp}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )
    gens = []
    try:
        for s in range(0, len(prompts), 24):
            bp = prompts[s : s + 24]
            enc = tok(bp, return_tensors="pt", padding=True).to(dev)
            in_len = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
            for bi in range(len(bp)):
                gens.append(tok.decode(out[bi, in_len:], skip_special_tokens=True))
            del out, enc
            torch.cuda.empty_cache()
    finally:
        tok.padding_side = orig_side

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as ex:
        per_t = list(ex.map(lambda p: base258.grade_mbpp_one(p[0], p[1]), zip(mbpp_dev_tasks, gens)))
    hits = sum(1 for r in per_t if r["ok"])
    return {
        "summary": {
            "dev_code_nll": round(tot_nll / max(1, tot_tok), 6),
            "dev_mbpp_pass1": round(hits / len(per_t), 5),
            "dev_mbpp_hits": f"{hits}/{len(per_t)}",
            "dev_mbpp_hits_int": hits,
            **align,
            "eval_sec": round(time.time() - t0, 1),
        },
        "per_fn_nll": per_fn_nll,
        "mbpp_per_task_bool": {str(r["task_id"]): bool(r["ok"]) for r in per_t},
    }


def train_h2_or_shuffled(model, tok, d_c, shuffled: bool, seed: int):
    """Exact H2 / H5 recipe returning per-layer delta Frobenius norms for exact random matching."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    depths = (30, 50, 70)
    weights = {30: 1.40, 50: 1.80, 70: 2.20}
    rank, steps, batch_size, lr = 24, 300, 8, 2.4e-4
    kl_w, kl_hinge, kl_floor, perp_w = 2.50, 0.015, 0.30, 0.55

    tgts_by_dep = {dep: d_c[f"code_r3_{dep}_tr"] for dep in depths}
    if shuffled:
        tgts_by_dep = base258.shuffle_within_seg_index(tgts_by_dep, d_c["tr_pts_off"], seed=seed)
    tgts_by_layer = {base258.LAYERS_MAP[dep]: tgts_by_dep[dep] for dep in depths}
    frames = base258.fit_native_frames(model, d_c, tgts_by_layer, pad_id, dev, n_fit=1200)

    for p in model.parameters():
        p.requires_grad_(False)
    wrappers, all_lora = [], []
    for li in ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mname, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            w = base258.LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((f"L{li}.{mname}", parent, attr, w))
            all_lora.extend([w.lora_A, w.lora_B])

    cap = {}
    hooks = [
        model.model.layers[base258.LAYERS_MAP[dep]].register_forward_hook(
            lambda _m, _i, out, idx=base258.LAYERS_MAP[dep]: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for dep in depths
    ]
    opt = torch.optim.AdamW(all_lora, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_ex = len(ids_o) - 1
    perm, cur = rng.permutation(n_ex), 0

    completed = False
    fro_norms = {}
    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 20.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul
            pw_scale = 1.0 if frac < 0.78 else (max(0.0, 1.0 - (frac - 0.78) / 0.17) if frac < 0.95 else 0.0)

            batch = []
            while len(batch) < batch_size:
                if cur >= n_ex:
                    perm, cur = rng.permutation(n_ex), 0
                pi = int(perm[cur]); cur += 1
                ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                if len(ids) >= 4 and len(pts) >= 1:
                    ts = {dep: tgts_by_dep[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in depths}
                    batch.append((ids, pts, ts))

            ml = max(len(b[0]) for b in batch)
            B = len(batch)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((B, ml), dtype=torch.long, device=dev)
            valid_tok = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
            b_idx_l, t_idx_l = [], []
            yt_lists = {dep: [] for dep in depths}
            for bi, (ids, pts, ts) in enumerate(batch):
                L = len(ids)
                inp[bi, :L] = torch.from_numpy(ids).to(dev)
                attn[bi, :L] = 1
                valid_tok[bi, : L - 1] = 1.0
                b_idx_l.append(torch.full((len(pts),), bi, dtype=torch.long, device=dev))
                t_idx_l.append(torch.from_numpy(pts).to(dev))
                for dep in depths:
                    yt_lists[dep].append(ts[dep])

            b_idx, t_idx = torch.cat(b_idx_l), torch.cat(t_idx_l)
            y_t = {dep: torch.from_numpy(np.concatenate(yt_lists[dep], axis=0)).to(dev, dtype=torch.float32) for dep in depths}
            v_sum = valid_tok.sum().clamp_min(1.0)

            for _, _, _, w in wrappers:
                w.active = False
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_r = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
                lp_ref = F.log_softmax(lg_r.float(), dim=-1)
                p_ref = lp_ref.exp()
                perp_b = {}
                for dep in depths:
                    li = base258.LAYERS_MAP[dep]
                    fr = frames[li]
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z = hc @ fr["V_q"]
                    perp_b[li] = (hc - z @ fr["V_q"].T).detach()
                del lg_r
            for _, _, _, w in wrappers:
                w.active = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            lp_cur = F.log_softmax(lg.float(), dim=-1)
            tok_kl = (p_ref * (lp_ref - lp_cur)).sum(dim=-1)
            kl = (tok_kl * valid_tok).sum() / v_sum
            kl_loss = kl_w * F.relu(kl - kl_hinge) + kl_floor * kl

            g_pl = None
            if pw_scale > 0.0:
                loss_geo = lg.new_tensor(0.0, dtype=torch.float32)
                for dep in depths:
                    li = base258.LAYERS_MAP[dep]
                    fr = frames[li]
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z_cur = hc @ fr["V_q"]
                    h_perp = hc - z_cur @ fr["V_q"].T
                    z_star = (y_t[dep] - fr["mu_y"]) @ fr["R_star"]
                    l_sub = base258.global_frob_cos_gram_loss(z_cur, z_star)
                    l_perp = F.mse_loss(h_perp, perp_b[li]) / (fr["rms_x"] ** 2)
                    loss_geo = loss_geo + weights[dep] * l_sub + perp_w * l_perp
                g_pl = torch.autograd.grad(pw_scale * loss_geo, all_lora, retain_graph=True, allow_unused=True)

            g_kl = torch.autograd.grad(kl_loss, all_lora, retain_graph=False, allow_unused=True)
            opt.zero_grad(set_to_none=True)
            for i, p in enumerate(all_lora):
                gk = g_kl[i] if g_kl[i] is not None else torch.zeros_like(p)
                gp = (g_pl[i] if (g_pl is not None and g_pl[i] is not None) else torch.zeros_like(p))
                if g_pl is not None and g_pl[i] is not None:
                    dot = torch.sum(gp * gk)
                    if dot < 0.0:
                        gp = gp - (dot / torch.sum(gk * gk).clamp_min(1e-12)) * gk
                p.grad = gk + gp
            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()
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
            for tag, parent, attr, w in wrappers:
                fro_norms[tag] = round(w.delta_fro(), 6)
                setattr(parent, attr, w.merge_into_base())
        else:
            for _, parent, attr, w in wrappers:
                setattr(parent, attr, w.base)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        del frames, wrappers
        gc.collect()
        torch.cuda.empty_cache()

    tot_fro = float(np.sqrt(sum(v * v for v in fro_norms.values())))
    return {
        "arm": "H5_tri_SHUFFLED_ctrl" if shuffled else "H2_r3_tri_trustkl",
        "seed": seed, "ce_w": 0.0,
        "total_fro_norm": round(tot_fro, 6),
        "delta_fro_norms": fro_norms,
        "train_sec": round(time.time() - t0, 1),
    }


@torch.no_grad()
def apply_random_lora_matched_258(model, target_fro_norms: dict, seed: int, rank: int = 24):
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed + 88888)
    applied = {}
    for li in ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mname, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            tag = f"L{li}.{mname}"
            tgt = float(target_fro_norms.get(tag, 0.0))
            lin = getattr(parent, attr)
            out_f, in_f = lin.weight.shape
            A = torch.randn((rank, in_f), generator=gen, dtype=torch.float32)
            B = torch.randn((out_f, rank), generator=gen, dtype=torch.float32)
            dw = B @ A
            cn = float(torch.norm(dw, p="fro").item())
            if cn > 0 and tgt > 0:
                dw = dw * (tgt / cn)
            lin.weight.data.add_(dw.to(device=lin.weight.device, dtype=lin.weight.dtype))
            applied[tag] = round(float(torch.norm(dw, p="fro").item()), 6)
    model.eval()
    return {
        "arm": "random_lora_matched", "seed": seed, "ce_w": 0.0,
        "total_fro_norm": round(float(np.sqrt(sum(v * v for v in applied.values()))), 6),
        "delta_fro_norms": applied, "train_sec": round(time.time() - t0, 2),
    }


def train_plain_ce_code(model, tok, d_c, seed: int, ce_w: float = 0.50):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rank, steps, batch_size, lr = 24, 300, 8, 2.4e-4
    kl_w, kl_hinge, kl_floor = 2.50, 0.015, 0.30

    for p in model.parameters():
        p.requires_grad_(False)
    wrappers, all_lora = [], []
    for li in ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mname, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            w = base258.LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((f"L{li}.{mname}", parent, attr, w))
            all_lora.extend([w.lora_A, w.lora_B])

    opt = torch.optim.AdamW(all_lora, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    n_ex = len(ids_o) - 1
    perm, cur = rng.permutation(n_ex), 0
    completed = False
    fro_norms = {}
    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 20.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul
            batch = []
            while len(batch) < batch_size:
                if cur >= n_ex:
                    perm, cur = rng.permutation(n_ex), 0
                pi = int(perm[cur]); cur += 1
                ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                if len(ids) >= 4:
                    batch.append(ids)
            ml = max(len(x) for x in batch)
            B = len(batch)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((B, ml), dtype=torch.long, device=dev)
            valid_tok = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
            for bi, ids in enumerate(batch):
                L = len(ids)
                inp[bi, :L] = torch.from_numpy(ids).to(dev)
                attn[bi, :L] = 1
                valid_tok[bi, : L - 1] = 1.0
            v_sum = valid_tok.sum().clamp_min(1.0)

            for _, _, _, w in wrappers:
                w.active = False
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_r = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
                lp_ref = F.log_softmax(lg_r.float(), dim=-1)
                p_ref = lp_ref.exp()
                del lg_r
            for _, _, _, w in wrappers:
                w.active = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            lp_cur = F.log_softmax(lg.float(), dim=-1)
            del lg
            kl = ((p_ref * (lp_ref - lp_cur)).sum(dim=-1) * valid_tok).sum() / v_sum
            kl_loss = kl_w * F.relu(kl - kl_hinge) + kl_floor * kl
            tgt = inp[:, 1:].unsqueeze(-1)
            ce = (-lp_cur.gather(-1, tgt).squeeze(-1) * valid_tok).sum() / v_sum
            loss = ce_w * ce + kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            del lp_cur, lp_ref, p_ref
            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()
        completed = True
    finally:
        opt.zero_grad(set_to_none=True)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if completed:
            for tag, parent, attr, w in wrappers:
                fro_norms[tag] = round(w.delta_fro(), 6)
                setattr(parent, attr, w.merge_into_base())
        else:
            for _, parent, attr, w in wrappers:
                setattr(parent, attr, w.base)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        del wrappers
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "arm": "plain_ce_code", "seed": seed, "ce_w": ce_w,
        "total_fro_norm": round(float(np.sqrt(sum(v * v for v in fro_norms.values()))), 6),
        "delta_fro_norms": fro_norms, "train_sec": round(time.time() - t0, 1),
    }


def summarize_258(runs_dir: Path, out_path: Path):
    base = json.loads((runs_dir / "base.json").read_text())
    bsum = base["summary"]
    arms = ["H2_r3_tri_trustkl", "random_lora_matched", "H5_tri_SHUFFLED_ctrl", "plain_ce_code"]
    loaded = {a: [json.loads((runs_dir / f"{a}_seed{s}.json").read_text()) for s in SEEDS if (runs_dir / f"{a}_seed{s}.json").exists()] for a in arms}
    metrics = ["dev_code_nll", "dev_mbpp_pass1", "code_blind_30_r2", "code_blind_50_r2", "code_blind_70_r2"]

    arm_tables = {}
    for arm in arms:
        recs = loaded[arm]
        if not recs:
            continue
        tbl = {"n_seeds": len(recs), "abs": {}, "delta_vs_base": {}}
        for m in metrics:
            vals = [r["summary"][m] for r in recs]
            tbl["abs"][m] = sample_stats(vals)
            tbl["delta_vs_base"][m] = sample_stats([v - bsum[m] for v in vals])
        tbl["total_fro_norm"] = sample_stats([r["train_meta"]["total_fro_norm"] for r in recs])
        arm_tables[arm] = tbl

    contrasts = {}
    for a, b in [
        ("H2_r3_tri_trustkl", "random_lora_matched"),
        ("H2_r3_tri_trustkl", "H5_tri_SHUFFLED_ctrl"),
        ("H2_r3_tri_trustkl", "plain_ce_code"),
        ("plain_ce_code", "random_lora_matched"),
    ]:
        if len(loaded.get(a, [])) == len(SEEDS) and len(loaded.get(b, [])) == len(SEEDS):
            ck = f"{a}_minus_{b}"
            contrasts[ck] = {m: paired_seed_contrast([r["summary"][m] for r in loaded[a]], [r["summary"][m] for r in loaded[b]]) for m in metrics}
            fn_a = np.mean([[x["nll"] for x in r["per_fn_nll"]] for r in loaded[a]], axis=0)
            fn_b = np.mean([[x["nll"] for x in r["per_fn_nll"]] for r in loaded[b]], axis=0)
            tkeys = sorted(loaded[a][0]["mbpp_per_task_bool"].keys(), key=int)
            mb_a = np.mean([[float(r["mbpp_per_task_bool"][k]) for k in tkeys] for r in loaded[a]], axis=0)
            mb_b = np.mean([[float(r["mbpp_per_task_bool"][k]) for k in tkeys] for r in loaded[b]], axis=0)
            contrasts[ck]["item_bootstrap"] = {
                "dev_code_fn_nll": bootstrap_diff(fn_a - fn_b),
                "dev_mbpp_task_pass1": bootstrap_diff(mb_a - mb_b),
            }

    out = {"seeds": SEEDS, "base": bsum, "arms": arm_tables, "paired_contrasts": contrasts}
    out_path.write_text(json.dumps(out, indent=1))
    return out


def main():
    d_c = np.load(base258.TARGETS_NPZ, allow_pickle=True)
    mbpp = json.loads((base258.EXP257_DATA / "sanitized-mbpp.json").read_text())
    mbpp_dev = [x for x in mbpp if int(x["task_id"]) < 11 or int(x["task_id"]) > 510]
    assert len(mbpp_dev) == 170

    runs_dir = ROOT / "results_5seed" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_p = ROOT / "results_5seed" / "exp258_5seed_summary.json"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, device_map="cuda:0")
    model.eval()
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _restore():
        sd = model.state_dict()
        for k, v in base_sd.items():
            sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
        model.eval()

    bp = runs_dir / "base.json"
    if not bp.exists():
        ev_b = evaluate_dev_per_item(model, tok, d_c, mbpp_dev)
        bp.write_text(json.dumps(ev_b, indent=1))
        print("[EXP258_5S] BASE:", json.dumps(ev_b["summary"]), flush=True)

    for seed in SEEDS:
        for arm in ("H2_r3_tri_trustkl", "random_lora_matched", "H5_tri_SHUFFLED_ctrl", "plain_ce_code"):
            rp = runs_dir / f"{arm}_seed{seed}.json"
            if rp.exists():
                print(f"[EXP258_5S] Cached {arm} s={seed}", flush=True)
                continue
            _restore()
            print(f"[EXP258_5S] Running {arm} s={seed}...", flush=True)
            if arm == "H2_r3_tri_trustkl":
                tm = train_h2_or_shuffled(model, tok, d_c, shuffled=False, seed=seed)
            elif arm == "random_lora_matched":
                ref = json.loads((runs_dir / f"H2_r3_tri_trustkl_seed{seed}.json").read_text())
                tm = apply_random_lora_matched_258(model, ref["train_meta"]["delta_fro_norms"], seed=seed)
            elif arm == "H5_tri_SHUFFLED_ctrl":
                tm = train_h2_or_shuffled(model, tok, d_c, shuffled=True, seed=seed)
            else:
                tm = train_plain_ce_code(model, tok, d_c, seed=seed, ce_w=0.50)
            ev_r = evaluate_dev_per_item(model, tok, d_c, mbpp_dev)
            ev_r["train_meta"] = tm
            rp.write_text(json.dumps(ev_r, indent=1))
            s = ev_r["summary"]
            print(
                f"[EXP258_5S] DONE {arm} s={seed} | nll={s['dev_code_nll']:.5f} mbpp={s['dev_mbpp_hits']} "
                f"al={s['code_blind_30_r2']:.4f}/{s['code_blind_50_r2']:.4f}/{s['code_blind_70_r2']:.4f} "
                f"fro={tm['total_fro_norm']:.3f} t={tm['train_sec']}+{s['eval_sec']}s",
                flush=True,
            )
        summarize_258(runs_dir, summary_p)

    out = summarize_258(runs_dir, summary_p)
    print("[EXP258_5S] ALL_DONE!", json.dumps({
        arm: {m: tbl["delta_vs_base"][m]["mean"] for m in ("dev_code_nll", "dev_mbpp_pass1", "code_blind_30_r2", "code_blind_50_r2", "code_blind_70_r2")}
        for arm, tbl in out["arms"].items()
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
