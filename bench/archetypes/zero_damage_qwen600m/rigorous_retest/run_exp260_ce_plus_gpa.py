"""Experiment 260: Unfrozen-Readout Joint Training (`Plain CE` vs. `CE + Standard GPA` vs.
`CE + Whitened GCCA` vs. `CE + Shuffled`) using the Upgraded 3-Family `Gemma 4 31B + Llama 3.1 + Mistral` Lattice.

Directly tests whether fixing the structural bottlenecks of `ce_w = 0` shape transfer unlocks
real downstream coding & OOD-retention gains:
  1. Drops the base-0.6B KL collar (`kl_w = 0.0`) so output logits are free to improve.
  2. Unfreezes ALL upper readout layers (`L2..L27` `o_proj + down_proj` rank-24 LoRA).
  3. Compares TWO cross-family geometric targets from `Gemma 4 31B dense + Llama 3.1 8B + Mistral 7B`:
       - `Standard GPA` (`gpa31_{30,50,70}`: variance-weighted, `0.692 / 0.649 / 0.574` blind r2_frob)
       - `Whitened GCCA` (`gcca31_{30,50,70}`: `90.7% / 90.2% / 83.8%` blind Top-1 retrieval!)
         aligned via Native Whitened-Frame Projection (`W_q = V_q @ diag(1 / sqrt(eig + 0.05*mean))`)
         so Qwen's natural residual-stream variance is preserved at step 0 while all 256
         fine-grained cross-family dimensions get equal gradient weight!

Evaluates 8 arms across 5 fresh paired seeds {101..105} in exact FP32 (`bwrap` OS sandbox):
  Regime A (Full 1,997 decontaminated Python functions, 250 steps = 1.0 epoch):
    - `plain_ce_full`, `ce_plus_gpa_full`, `ce_plus_gcca_full`, `ce_plus_shuf_full`
  Regime B (Low-data 250-function overfitting regime, 250 steps = 8.0 epochs):
    - `plain_ce_low250`, `ce_plus_gpa_low250`, `ce_plus_gcca_low250`, `ce_plus_shuf_low250`

Metrics (exact FP32, persisted per-item for 10,000-resample paired bootstraps + df=4 t-CIs):
  - `dev_code_nll` (399 strictly held-out Python functions)
  - `dev_mbpp_170` (170 disjoint MBPP dev tasks, bwrap sandbox)
  - `humaneval_164` (164 untouched HumanEval tasks, bwrap sandbox)
  - `wikitext2_nll` (62 OOD Wikipedia articles — catastrophic forgetting test!)
  - Blind `Phi-4-mini`, `OLMo 3 7B`, AND `Qwen3.8-27B` OOF `r2_frob` & `Top-1` at 30/50/70%!
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
EXP257_DIR = ROOT.parent / "exp257_rigorous"
for p in (str(ROOT), str(EXP257_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_rigorous as ev257  # noqa: E402
import run_dev_code_cuda as base258  # noqa: E402
import run_exp258_5seed as r258  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]
ACTIVE_LAYERS = list(range(2, 28))  # L2..L27 ALL readout layers unfrozen!
DEPTHS = (30, 50, 70)
GEO_WEIGHTS = {30: 0.25, 50: 0.35, 70: 0.40}
TARGETS_31B = ROOT / "targets_qwen600m_31b_gpa_gcca.npz"


@torch.no_grad()
def fit_native_frames_260(model, d_c, tgts_by_layer, pad_id, dev, whitened: bool = False, n_fit: int = 1200):
    """Fit Native PCA (whitened=False) or Native Whitened-PCA (whitened=True) Procrustes frames."""
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_tot = len(ids_o) - 1
    rng = np.random.default_rng(2601)
    fit_idx = sorted(rng.choice(n_tot, size=min(n_fit, n_tot), replace=False).tolist())
    layers = sorted(tgts_by_layer.keys())
    cap = {}
    hooks = [
        model.model.layers[li].register_forward_hook(
            lambda _m, _i, out, idx=li: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for li in layers
    ]
    Xs, Ys = {li: [] for li in layers}, {li: [] for li in layers}
    try:
        for s in range(0, len(fit_idx), 32):
            sub = fit_idx[s : s + 32]
            b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64) for i in sub]
            b_pts = [pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64) for i in sub]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            for li in layers:
                h = cap[li].float().cpu().numpy()
                tgt = tgts_by_layer[li]
                for bi, (i_abs, pts) in enumerate(zip(sub, b_pts)):
                    Xs[li].append(h[bi, pts, :])
                    Ys[li].append(tgt[int(pts_o[i_abs]):int(pts_o[i_abs + 1])])
    finally:
        for hk in hooks:
            hk.remove()

    frames = {}
    for li in layers:
        X = np.concatenate(Xs[li], axis=0).astype(np.float64)
        Y = np.concatenate(Ys[li], axis=0).astype(np.float64)
        mu_x = X.mean(0, keepdims=True)
        Xc = X - mu_x
        cov = (Xc.T @ Xc) / float(len(Xc))
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1][: Y.shape[1]]
        V_q = vecs[:, idx]
        if whitened:
            ev_top = np.maximum(vals[idx], 1e-8)
            inv_std = 1.0 / np.sqrt(ev_top + 0.05 * float(np.mean(ev_top)))
            P_q = V_q * inv_std[None, :]  # whitened projection preserving base scale at step 0!
        else:
            P_q = V_q
        Zx = Xc @ P_q
        s_x = float(np.sqrt(np.mean(np.sum(Zx ** 2, axis=1)))) + 1e-12

        mu_y = Y.mean(0, keepdims=True)
        Yc = Y - mu_y
        s_y = float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1)))) + 1e-12
        U, _, Vt = np.linalg.svd((Yc / s_y).T @ (Zx / s_x), full_matrices=False)
        R_star = U @ Vt
        frames[li] = {
            "mu_x": torch.from_numpy(mu_x.squeeze(0).astype(np.float32)).to(dev),
            "P_q": torch.from_numpy(P_q.astype(np.float32)).to(dev),
            "mu_y": torch.from_numpy(mu_y.squeeze(0).astype(np.float32)).to(dev),
            "R_star": torch.from_numpy((R_star * (s_x / s_y)).astype(np.float32)).to(dev),
        }
    return frames


@torch.no_grad()
def evaluate_260(model, tok, d_c, mbpp_dev_tasks, data_dir: Path):
    """Exact FP32 eval: code_eval NLL (399), blind 30/50/70 R2 (Phi4+OLMo3+Qwen27B), MBPP-dev-170, HumanEval-164, WikiText-2 (62)."""
    t0 = time.time()
    dev_res = r258.evaluate_dev_per_item(model, tok, d_c, mbpp_dev_tasks)
    he_res = ev257.eval_humaneval(model, tok, data_dir, max_workers=12)
    wt2_res = ev257.eval_wikitext2_nll(model, tok, data_dir)

    summary = dict(dev_res["summary"])
    summary["humaneval_pass1"] = he_res["pass_at_1"]
    summary["humaneval_hits"] = f"{he_res['hits']}/{he_res['total']}"
    summary["humaneval_hits_int"] = he_res["hits"]
    summary["wikitext2_nll"] = wt2_res["nll"]
    summary["eval_sec"] = round(time.time() - t0, 1)

    return {
        "summary": summary,
        "per_fn_nll": dev_res["per_fn_nll"],
        "mbpp_per_task_bool": dev_res["mbpp_per_task_bool"],
        "humaneval_per_task_bool": he_res["per_task_bool"],
        "wikitext2_per_article": wt2_res["per_article"],
    }


def train_260_arm(model, tok, d_c, arm: str, seed: int, train_indices: list, frames_map: dict, tgts_map: dict):
    """Joint CE + optional Standard-GPA / Whitened-GCCA / Shuffled on unfrozen L2..L27 LoRA (zero KL collar!)."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    mode = "none"
    if "gpa" in arm:
        mode = "gpa"
    elif "gcca" in arm:
        mode = "gcca"
    elif "shuf" in arm:
        mode = "shuf"
    use_geo = (mode != "none")
    geo_scale = (0.50 if "low250" in arm else 0.35) if use_geo else 0.0
    rank, steps, batch_size, lr = 24, 250, 8, 1.0e-4

    tgts_by_dep = tgts_map[mode] if use_geo else None
    frames = frames_map[mode] if use_geo else None

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
    hooks = []
    if use_geo:
        hooks = [
            model.model.layers[base258.LAYERS_MAP[dep]].register_forward_hook(
                lambda _m, _i, out, idx=base258.LAYERS_MAP[dep]: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
            )
            for dep in DEPTHS
        ]

    opt = torch.optim.AdamW(all_lora, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    pool = np.array(train_indices, dtype=np.int64)
    perm, cur = rng.permutation(pool), 0

    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]

    history = []
    completed = False
    fro_norms = {}
    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 20.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul

            batch = []
            while len(batch) < batch_size:
                if cur >= len(perm):
                    perm, cur = rng.permutation(pool), 0
                pi = int(perm[cur]); cur += 1
                ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                if len(ids) >= 4 and len(pts) >= 1:
                    ts = {dep: tgts_by_dep[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in DEPTHS} if use_geo else None
                    batch.append((ids, pts, ts))

            ml = max(len(b[0]) for b in batch)
            B = len(batch)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((B, ml), dtype=torch.long, device=dev)
            valid_tok = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
            b_idx_l, t_idx_l = [], []
            yt_lists = {dep: [] for dep in DEPTHS}
            for bi, (ids, pts, ts) in enumerate(batch):
                L = len(ids)
                inp[bi, :L] = torch.from_numpy(ids).to(dev)
                attn[bi, :L] = 1
                valid_tok[bi, : L - 1] = 1.0
                if use_geo:
                    b_idx_l.append(torch.full((len(pts),), bi, dtype=torch.long, device=dev))
                    t_idx_l.append(torch.from_numpy(pts).to(dev))
                    for dep in DEPTHS:
                        yt_lists[dep].append(ts[dep])

            v_sum = valid_tok.sum().clamp_min(1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            tgt = inp[:, 1:].contiguous()
            tok_ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").reshape(B, ml - 1)
            ce_loss = (tok_ce * valid_tok).sum() / v_sum
            del lg, tok_ce

            geo_loss = ce_loss.new_tensor(0.0)
            if use_geo:
                b_idx = torch.cat(b_idx_l)
                t_idx = torch.cat(t_idx_l)
                for dep in DEPTHS:
                    li = base258.LAYERS_MAP[dep]
                    fr = frames[li]
                    y_t = torch.from_numpy(np.concatenate(yt_lists[dep], axis=0)).to(dev, dtype=torch.float32)
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z_cur = hc @ fr["P_q"]
                    z_star = (y_t - fr["mu_y"]) @ fr["R_star"]
                    geo_loss = geo_loss + GEO_WEIGHTS[dep] * base258.global_frob_cos_gram_loss(z_cur, z_star)

            loss = ce_loss + geo_scale * geo_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()

            if s % 125 == 0 or s == steps - 1:
                history.append({"step": s, "ce": round(float(ce_loss.item()), 4), "geo": round(float(geo_loss.item()), 4)})
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
        del wrappers
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "arm": arm, "seed": seed, "mode": mode, "ce_w": 1.0, "geo_scale": geo_scale, "kl_w": 0.0,
        "n_train_functions": len(train_indices),
        "total_fro_norm": round(float(np.sqrt(sum(v * v for v in fro_norms.values()))), 6),
        "train_sec": round(time.time() - t0, 1),
        "history": history,
    }


def summarize_260(runs_dir: Path, out_path: Path):
    base = json.loads((runs_dir / "base.json").read_text())
    bsum = base["summary"]
    arms = [
        "plain_ce_full", "ce_plus_gpa_full", "ce_plus_gcca_full", "ce_plus_shuf_full",
        "plain_ce_low250", "ce_plus_gpa_low250", "ce_plus_gcca_low250", "ce_plus_shuf_low250",
    ]
    loaded = {a: [json.loads((runs_dir / f"{a}_seed{s}.json").read_text()) for s in SEEDS if (runs_dir / f"{a}_seed{s}.json").exists()] for a in arms}
    metrics = ["dev_code_nll", "dev_mbpp_pass1", "humaneval_pass1", "wikitext2_nll", "code_blind_30_r2", "code_blind_50_r2", "code_blind_70_r2"]

    arm_tables = {}
    for arm in arms:
        recs = loaded[arm]
        if not recs:
            continue
        tbl = {"n_seeds": len(recs), "abs": {}, "delta_vs_base": {}}
        for m in metrics:
            vals = [r["summary"][m] for r in recs]
            tbl["abs"][m] = r258.sample_stats(vals)
            tbl["delta_vs_base"][m] = r258.sample_stats([v - bsum[m] for v in vals])
        tbl["total_fro_norm"] = r258.sample_stats([r["train_meta"]["total_fro_norm"] for r in recs])
        arm_tables[arm] = tbl

    contrasts = {}
    pairs = [
        ("ce_plus_gpa_full", "plain_ce_full"),
        ("ce_plus_gcca_full", "plain_ce_full"),
        ("ce_plus_gcca_full", "ce_plus_gpa_full"),
        ("ce_plus_gpa_full", "ce_plus_shuf_full"),
        ("ce_plus_gcca_full", "ce_plus_shuf_full"),
        ("ce_plus_gpa_low250", "plain_ce_low250"),
        ("ce_plus_gcca_low250", "plain_ce_low250"),
        ("ce_plus_gcca_low250", "ce_plus_gpa_low250"),
        ("ce_plus_gpa_low250", "ce_plus_shuf_low250"),
        ("ce_plus_gcca_low250", "ce_plus_shuf_low250"),
    ]
    for a, b in pairs:
        if len(loaded.get(a, [])) == len(SEEDS) and len(loaded.get(b, [])) == len(SEEDS):
            ck = f"{a}_minus_{b}"
            contrasts[ck] = {m: r258.paired_seed_contrast([r["summary"][m] for r in loaded[a]], [r["summary"][m] for r in loaded[b]]) for m in metrics}
            fn_a = np.mean([[x["nll"] for x in r["per_fn_nll"]] for r in loaded[a]], axis=0)
            fn_b = np.mean([[x["nll"] for x in r["per_fn_nll"]] for r in loaded[b]], axis=0)
            wt_a = np.mean([[x["nll"] for x in r["wikitext2_per_article"]] for r in loaded[a]], axis=0)
            wt_b = np.mean([[x["nll"] for x in r["wikitext2_per_article"]] for r in loaded[b]], axis=0)
            mkeys = sorted(loaded[a][0]["mbpp_per_task_bool"].keys(), key=int)
            mb_a = np.mean([[float(r["mbpp_per_task_bool"][k]) for k in mkeys] for r in loaded[a]], axis=0)
            mb_b = np.mean([[float(r["mbpp_per_task_bool"][k]) for k in mkeys] for r in loaded[b]], axis=0)
            hkeys = sorted(loaded[a][0]["humaneval_per_task_bool"].keys())
            he_a = np.mean([[float(r["humaneval_per_task_bool"][k]) for k in hkeys] for r in loaded[a]], axis=0)
            he_b = np.mean([[float(r["humaneval_per_task_bool"][k]) for k in hkeys] for r in loaded[b]], axis=0)
            contrasts[ck]["item_bootstrap"] = {
                "dev_code_fn_nll": r258.bootstrap_diff(fn_a - fn_b),
                "wikitext2_article_nll": r258.bootstrap_diff(wt_a - wt_b),
                "dev_mbpp_task_pass1": r258.bootstrap_diff(mb_a - mb_b),
                "humaneval_task_pass1": r258.bootstrap_diff(he_a - he_b),
            }

    out = {"seeds": SEEDS, "teachers": ["gemma4_31b", "llama31_8b", "mistral7b"], "base": bsum, "arms": arm_tables, "paired_contrasts": contrasts}
    out_path.write_text(json.dumps(out, indent=1))
    return out


def main():
    d_c = np.load(TARGETS_31B, allow_pickle=True)
    data_dir = base258.EXP257_DATA
    mbpp = json.loads((data_dir / "sanitized-mbpp.json").read_text())
    mbpp_dev = [x for x in mbpp if int(x["task_id"]) < 11 or int(x["task_id"]) > 510]
    assert len(mbpp_dev) == 170

    runs_dir = ROOT / "results_260" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_p = ROOT / "results_260" / "exp260_ce_plus_gpa_gcca_summary.json"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, device_map="cuda:0")
    model.eval()
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _restore():
        sd = model.state_dict()
        for k, v in base_sd.items():
            sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
        model.eval()

    bp = runs_dir / "base.json"
    if not bp.exists():
        ev_b = evaluate_260(model, tok, d_c, mbpp_dev, data_dir)
        bp.write_text(json.dumps(ev_b, indent=1))
        print("[EXP260] BASE:", json.dumps(ev_b["summary"]), flush=True)

    n_tot = len(d_c["tr_ids_off"]) - 1
    full_idx = list(range(n_tot))
    low250_idx = sorted(np.random.default_rng(2600).choice(n_tot, size=250, replace=False).tolist())

    gpa_tgts = {dep: d_c[f"gpa31_{dep}_tr"] for dep in DEPTHS}
    gcca_tgts = {dep: d_c[f"gcca31_{dep}_tr"] for dep in DEPTHS}
    frames_gpa = fit_native_frames_260(model, d_c, {base258.LAYERS_MAP[dep]: gpa_tgts[dep] for dep in DEPTHS}, pad_id, dev, whitened=False)
    frames_gcca = fit_native_frames_260(model, d_c, {base258.LAYERS_MAP[dep]: gcca_tgts[dep] for dep in DEPTHS}, pad_id, dev, whitened=True)

    arm_order = [
        "plain_ce_full", "ce_plus_gpa_full", "ce_plus_gcca_full", "ce_plus_shuf_full",
        "plain_ce_low250", "ce_plus_gpa_low250", "ce_plus_gcca_low250", "ce_plus_shuf_low250",
    ]
    for seed in SEEDS:
        shuf_tgts = base258.shuffle_within_seg_index(gpa_tgts, d_c["tr_pts_off"], seed=seed)
        _restore()
        frames_shuf = fit_native_frames_260(model, d_c, {base258.LAYERS_MAP[dep]: shuf_tgts[dep] for dep in DEPTHS}, pad_id, dev, whitened=False)
        tgts_map = {"gpa": gpa_tgts, "gcca": gcca_tgts, "shuf": shuf_tgts}
        frames_map = {"gpa": frames_gpa, "gcca": frames_gcca, "shuf": frames_shuf}

        for arm in arm_order:
            rp = runs_dir / f"{arm}_seed{seed}.json"
            if rp.exists():
                print(f"[EXP260] Cached {arm} s={seed}", flush=True)
                continue
            _restore()
            print(f"[EXP260] Running {arm} s={seed}...", flush=True)
            t_idx = low250_idx if "low250" in arm else full_idx
            tm = train_260_arm(model, tok, d_c, arm, seed, t_idx, frames_map, tgts_map)
            ev_r = evaluate_260(model, tok, d_c, mbpp_dev, data_dir)
            ev_r["train_meta"] = tm
            rp.write_text(json.dumps(ev_r, indent=1))
            s = ev_r["summary"]
            print(
                f"[EXP260] DONE {arm} s={seed} | code_nll={s['dev_code_nll']:.5f} wt2={s['wikitext2_nll']:.5f} "
                f"MBPP={s['dev_mbpp_hits']} HE={s['humaneval_hits']} "
                f"al={s['code_blind_30_r2']:.4f}/{s['code_blind_50_r2']:.4f}/{s['code_blind_70_r2']:.4f} "
                f"fro={tm['total_fro_norm']:.3f} t={tm['train_sec']}+{s['eval_sec']}s",
                flush=True,
            )
        summarize_260(runs_dir, summary_p)

    out = summarize_260(runs_dir, summary_p)
    print("[EXP260] ALL_DONE!", flush=True)
    for arm, tbl in out["arms"].items():
        d = tbl["delta_vs_base"]
        print(
            f"  {arm:22s} | code_nll={d['dev_code_nll']['mean']:+.5f} wt2={d['wikitext2_nll']['mean']:+.5f} "
            f"MBPP={d['dev_mbpp_pass1']['mean']*100:+.2f}pp HE={d['humaneval_pass1']['mean']*100:+.2f}pp "
            f"L8={d['code_blind_30_r2']['mean']:+.4f} L14={d['code_blind_50_r2']['mean']:+.4f} fro={tbl['total_fro_norm']['mean']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
