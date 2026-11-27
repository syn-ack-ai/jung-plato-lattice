"""Aggregate 3-seed (seeds 251, 252, 253) Experiment 251 50%-depth 8B Gemma 4 E4B transplant runs.

Computes exact per-seed AND 3-seed mean +/- sample SD (ddof=1) across:
  - Unseen-family 50%-depth Procrustes alignment (Phi-4-mini, OLMo 3 7B) + 8 domain x model cells
  - Paired across-seed deltas (c3_50 - ctrl50, c3_50 - base, c4_50 - ctrl50, c3_50 - c4_50)
  - Exact L12 invariance (max |Delta| == 0.0 across all 10 checkpoints)
  - LM guardrails (40 held_gut NLL, 400 proc_held MCQ, actor2x2 NLL) & multimodal vision/audio checks
  - Zero-task-label 50%-depth probing (within-native, within-256D, rival->Gemma, Gemma->rival, HOUSE->SELF2)
  - Exact training hyperparameters and per-seed MLP L19/L20 relative L2 weight drift from train logs
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve repo root & scratch/cache paths via environment variables (zero hardcoded user paths in committed copy!)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("JPL_REPO_ROOT", str(SCRIPT_DIR.parents[1]))).resolve()
ARCH_DIR = REPO_ROOT / "bench" / "archetypes"
for p in (ARCH_DIR, ARCH_DIR / "fluent", ARCH_DIR / "aion_8b", SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plato_data as PD
import arch_rot229 as AR
import valence_ctrl234 as V234
import bow_mirror238 as BM238


def _build_valence_erasure_bases(Dv, Dh, sg_doms, house_cats):
    n_sg = len(sg_doms)
    u_sent = AR.unit(Dv[:n_sg].mean(0))
    u_ov = AR.unit(Dv[n_sg:].mean(0))
    Q2 = V234._orthonormal_basis([u_sent, u_ov])
    dom_vecs = [AR.unit(Dv[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
    Q5 = V234._orthonormal_basis(dom_vecs + [u_ov])
    uh = AR.house_dir(Dh, house_cats)
    Q6 = V234._orthonormal_basis(dom_vecs + [u_ov, uh])
    return {"raw": np.zeros((Dv.shape[1], 0)), "val2": Q2, "val5": Q5, "val5_house": Q6}, u_sent, uh


def _sid_and_splits(rows, seed_base):
    sids_str = [str(r["setting_id"]) for r in rows]
    uniq = sorted(set(sids_str))
    sid2i = {s: i for i, s in enumerate(uniq)}
    sid_idx = np.array([sid2i[s] for s in sids_str], dtype=int)
    splits = []
    for s_i in range(20):
        rng = np.random.default_rng(seed_base + s_i)
        perm = rng.permutation(len(uniq))
        f0 = set(int(x) for x in perm[: len(uniq) // 2])
        splits.append(np.array([0 if int(s) in f0 else 1 for s in sid_idx], dtype=int))
    return sid_idx, splits

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = Path(os.environ.get("EXP251_DIR", os.environ.get("JPL_EXP251_DIR", str(SCRATCH_DIR / "exp251_e4b_c4_50")))).resolve()
EVAL_DIR = EXP_DIR / "eval_cache"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(ARCH_DIR / "fluent" / "plato"))).resolve()
DEPTH_DIR = Path(os.environ.get("EXP251_DEPTH_DIR", os.environ.get("JPL_DEPTH_DIR", str(PLATO_CACHE / "depth240")))).resolve()
SELFSHAPE_DIR = ARCH_DIR / "selfshape"
OUT_REPO_JSON = ARCH_DIR / "aion_8b" / "runs" / "e4b_transplant50_exp251.json"


def _targets_path() -> Path:
    p = EXP_DIR / "targets_e4b_50.npz"
    if p.exists():
        return p
    alt = EXP_DIR / "data" / "targets_e4b_50.npz"
    return alt if alt.exists() else p

PLATO50_LAYER = 20
MLP_LAYERS = [19, 20]
C3_KEYS = ("qwen35_9b", "llama31_8b", "mistral7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
BLIND_KEYS = ("phi4_mini", "olmo3_7b")
RIVALS = ("qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
DOMAINS = ("held_gut", "items3_mech", "items3_myth", "proc_held")
SEEDS = (251, 252, 253)
TRAINED_ARMS = ("ctrl50", "c4_50", "c3_50")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def _paired_boot_ci(diff_per_unit, n_boot=2000, seed=2510):
    arr = np.asarray(diff_per_unit, dtype=np.float64)
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    bs = arr[idx].mean(axis=1)
    p_lo, p_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    b_lo, b_hi = 2.0 * obs - p_hi, 2.0 * obs - p_lo
    return {
        "obs": round(obs, 5),
        "ci95_pct": [round(p_lo, 5), round(p_hi, 5)],
        "ci95_basic": [round(b_lo, 5), round(b_hi, 5)],
    }


def _canonical_procrustes_oof(src_ev, tgt_ev, group_ids, domains):
    """Exact domain-stratified Random(2288) 2-fold group-disjoint Procrustes OOF predictions
    matching `plato_data.eval_procrustes_transfer` (strictly train-fold centering & affine map,
    zero test-fold leakage!)."""
    src_ev = np.asarray(src_ev, dtype=np.float64)
    tgt_ev = np.asarray(tgt_ev, dtype=np.float64)
    dom_groups = defaultdict(list)
    seen_g = set()
    for g, d in zip(group_ids, domains):
        if g not in seen_g:
            seen_g.add(g)
            dom_groups[d].append(g)
    g_fold = {}
    rng = random.Random(2288)
    for d in sorted(dom_groups):
        gs = list(dom_groups[d])
        rng.shuffle(gs)
        half = len(gs) // 2
        for i, g in enumerate(gs):
            g_fold[g] = 0 if i < half else 1
    folds = np.array([g_fold[g] for g in group_ids], dtype=np.int32)
    preds = np.zeros_like(tgt_ev)
    for test_f in (0, 1):
        m_tr = folds != test_f
        m_te = folds == test_f
        params = PD.fit_procrustes_map(src_ev[m_tr], tgt_ev[m_tr])
        preds[m_te] = PD.apply_procrustes_map(src_ev[m_te], params)
    pn = preds / np.maximum(1e-12, np.linalg.norm(preds, axis=1, keepdims=True))
    tn = tgt_ev / np.maximum(1e-12, np.linalg.norm(tgt_ev, axis=1, keepdims=True))
    pt_cos = np.sum(pn * tn, axis=1)
    return pt_cos


def _group_cos_stats(pt_cos, group_ids):
    """Return sorted group keys, per-group mean cosine (equal passage weight),
    and per-group (sum_cos, count) for exact cluster-resampled clause-weighted bootstrap."""
    by_g = defaultdict(list)
    for g, c in zip(group_ids, pt_cos):
        by_g[g].append(float(c))
    keys = sorted(by_g.keys())
    g_means = np.array([np.mean(by_g[g]) for g in keys], dtype=np.float64)
    g_sums = np.array([np.sum(by_g[g]) for g in keys], dtype=np.float64)
    g_cnts = np.array([len(by_g[g]) for g in keys], dtype=np.float64)
    return keys, g_means, g_sums, g_cnts


def _paired_cluster_ratio_boot_ci(diff_sums_per_group, cnts_per_group, n_boot=2000, seed=2510):
    """Cluster-resampled bootstrap over 400 passage groups with exact clause-count weighting,
    so the observed point estimate matches the headline clause-weighted mean cosine delta."""
    sums = np.asarray(diff_sums_per_group, dtype=np.float64)
    cnts = np.asarray(cnts_per_group, dtype=np.float64)
    obs = float(sums.sum() / cnts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    bs = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    p_lo, p_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    b_lo, b_hi = 2.0 * obs - p_hi, 2.0 * obs - p_lo
    return {
        "obs": round(obs, 5),
        "ci95_pct": [round(p_lo, 5), round(p_hi, 5)],
        "ci95_basic": [round(b_lo, 5), round(b_hi, 5)],
    }


def _eval_procrustes_fp(src_ev, tgt_ev, group_ids, domains, ret_mask):
    """Exact copy of `PD.eval_procrustes_transfer` returning full-precision floats (zero double-rounding!)."""
    src_ev = np.asarray(src_ev, dtype=np.float64)
    tgt_ev = np.asarray(tgt_ev, dtype=np.float64)
    dom_groups = defaultdict(list)
    seen_g = set()
    for g, d in zip(group_ids, domains):
        if g not in seen_g:
            seen_g.add(g)
            dom_groups[d].append(g)
    g_fold = {}
    rng = random.Random(2288)
    for d in sorted(dom_groups):
        gs = list(dom_groups[d])
        rng.shuffle(gs)
        half = len(gs) // 2
        for i, g in enumerate(gs):
            g_fold[g] = 0 if i < half else 1
    folds = np.array([g_fold[g] for g in group_ids], dtype=np.int32)

    cos_all, pred_all, pred_opt_all, true_all = [], [], [], []
    top1_list, top5_list = [], []
    dom_cos = defaultdict(list)
    for test_f in (0, 1):
        m_tr = folds != test_f
        m_te = folds == test_f
        params = PD.fit_procrustes_map(src_ev[m_tr], tgt_ev[m_tr])
        pred_tr = PD.apply_procrustes_map(src_ev[m_tr], params)
        pred = PD.apply_procrustes_map(src_ev[m_te], params)
        true_tr = tgt_ev[m_tr]
        true = tgt_ev[m_te]
        ym_tr = true_tr.mean(axis=0, keepdims=True)
        pm_tr = pred_tr.mean(axis=0, keepdims=True)
        alpha_f = float(np.sum((true_tr - ym_tr) * (pred_tr - pm_tr)) / max(1e-12, np.sum((pred_tr - pm_tr) ** 2)))
        pred_opt_all.append(ym_tr + alpha_f * (pred - pm_tr))

        pn = pred / np.maximum(1e-12, np.linalg.norm(pred, axis=1, keepdims=True))
        tn = true / np.maximum(1e-12, np.linalg.norm(true, axis=1, keepdims=True))
        cs = np.sum(pn * tn, axis=1)
        cos_all.append(cs)
        pred_all.append(pred)
        true_all.append(true)
        te_doms = [domains[i] for i in np.flatnonzero(m_te)]
        for c_val, d_val in zip(cs, te_doms):
            dom_cos[d_val].append(float(c_val))

        rm = np.asarray(ret_mask, dtype=bool)[m_te]
        pn_r, tn_r = pn[rm], tn[rm]
        sim = pn_r @ tn_r.T
        diag = np.diag(sim)[:, None]
        strictly_better = (sim > diag + 1e-12).sum(axis=1)
        tied = (np.abs(sim - diag) <= 1e-12).sum(axis=1)
        top1_list.append(float(np.mean(np.where(strictly_better == 0, 1.0 / np.maximum(1, tied), 0.0))))
        rem5 = np.maximum(0, 5 - strictly_better)
        top5_list.append(float(np.mean(np.minimum(1.0, rem5 / np.maximum(1, tied)))))

    pred_cat = np.concatenate(pred_all, axis=0)
    pred_opt_cat = np.concatenate(pred_opt_all, axis=0)
    true_cat = np.concatenate(true_all, axis=0)
    ss_res = float(np.sum((true_cat - pred_cat) ** 2))
    ss_tot = float(np.sum((true_cat - true_cat.mean(axis=0, keepdims=True)) ** 2))
    r2_frob = 1.0 - ss_res / max(1e-12, ss_tot)
    r2_opt = 1.0 - float(np.sum((true_cat - pred_opt_cat) ** 2)) / max(1e-12, ss_tot)
    return {
        "cos": float(np.concatenate(cos_all).mean()),
        "r2_frob": float(r2_frob),
        "r2_opt": float(r2_opt),
        "top1": float(np.mean(top1_list)),
        "top5": float(np.mean(top5_list)),
        "by_domain_cos": {d: float(np.mean(v)) for d, v in sorted(dom_cos.items())},
    }


def _stats(vals, ndigits=4):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), ndigits),
        "sd": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, ndigits),
        "min": round(float(arr.min()), ndigits),
        "max": round(float(arr.max()), ndigits),
        "per_seed": [round(float(x), ndigits) for x in arr],
    }


def _tag_for(arm: str, seed: int) -> str:
    if arm == "base":
        return "base"
    return arm if seed == 251 else f"{arm}_s{seed}"


def _eval_cv2_20split(Ds, Dt, Qs, Qt, splits, mask_ben=None, sid_i=None):
    Ds_p = V234._project_out_rows(Ds, Qs) if Qs is not None and Qs.shape[1] > 0 else Ds
    Dt_p = V234._project_out_rows(Dt, Qt) if Qt is not None and Qt.shape[1] > 0 else Dt
    sp_acc, sp_ben, sp_adv, sp_all4 = [], [], [], []
    oof_sum = np.zeros(len(Dt), dtype=np.float64)
    for fl in splits:
        oof = np.zeros(len(Dt), dtype=np.float64)
        for f in (0, 1):
            tr, te = (fl != f), (fl == f)
            u = AR.unit(Ds_p[tr].mean(0))
            if Qt is not None and Qt.shape[1] > 0:
                u = V234._project_out(u, Qt)
            oof[te] = Dt_p[te] @ u
        oof_sum += (oof > 0).astype(np.float64)
        sp_acc.append(float(np.mean(oof > 0)))
        if mask_ben is not None:
            sp_ben.append(float(np.mean(oof[mask_ben] > 0)))
            sp_adv.append(float(np.mean(oof[~mask_ben] > 0)))
        if sid_i is not None:
            u_s = sorted(set(int(x) for x in sid_i))
            a4 = [all(oof[sid_i == s] > 0) for s in u_s]
            sp_all4.append(float(np.mean(a4)))
    item_rate = oof_sum / float(len(splits))
    out = {"acc": round(float(np.mean(sp_acc)), 4), "_item_rate": item_rate}
    if mask_ben is not None:
        out["benign"] = round(float(np.mean(sp_ben)), 4)
        out["adverse"] = round(float(np.mean(sp_adv)), 4)
    if sid_i is not None:
        out["all4"] = round(float(np.mean(sp_all4)), 4)
    return out


def analyze_3seed(write_repo_json: bool = True):
    t0 = time.time()
    d_t = np.load(_targets_path(), allow_pickle=True)
    ev_groups = d_t["ev_pt_groups"].tolist()
    ev_doms = d_t["ev_pt_domains"].tolist()
    ev_ret = d_t["ev_pt_ret"]
    phi4_ev = d_t["phi4_50_ev"].astype(np.float64)
    olmo3_ev = d_t["olmo3_50_ev"].astype(np.float64)
    c3_ev = d_t["c3_50_ev"].astype(np.float64)
    c4_ev = d_t["c4_50_ev"].astype(np.float64)

    # Load all 10 checkpoints: base + 3 arms x 3 seeds (REQUIRE all 9 train logs!)
    tags = ["base"] + [_tag_for(arm, s) for s in SEEDS for arm in TRAINED_ARMS]
    acts = {}
    lms = {}
    tlogs = {}
    for tag in tags:
        acts[tag] = dict(np.load(EVAL_DIR / f"acts_{tag}.npz"))
        lm_raw = json.loads((EVAL_DIR / f"lm_{tag}.json").read_text())
        # Drop quick_blind_* fields so released JSON carries only the single canonical strict Procrustes block
        lm_raw.pop("quick_blind_phi4_50", None)
        lm_raw.pop("quick_blind_olmo3_50", None)
        lms[tag] = lm_raw
    for s in SEEDS:
        for arm in TRAINED_ARMS:
            tag = _tag_for(arm, s)
            tlog_p = EXP_DIR / f"train_{tag}_log.json"
            if not tlog_p.exists():
                raise FileNotFoundError(f"Missing required training log {tlog_p} for arm={arm}, seed={s}")
            tl = json.loads(tlog_p.read_text())
            assert tl.get("plato50_arm") == arm, f"Log arm mismatch in {tag}: {tl.get('plato50_arm')}"
            assert int(tl.get("plato50_seed", -1)) == s, f"Log seed mismatch in {tag}: {tl.get('plato50_seed')}"
            assert int(tl.get("plato50_steps", -1)) == 450, f"Log steps mismatch in {tag}: {tl.get('plato50_steps')}"
            assert list(tl.get("plato50_mlp_layers", [])) == MLP_LAYERS, f"Log MLP layers mismatch in {tag}"
            assert int(tl.get("plato50_layer", -1)) == PLATO50_LAYER, f"Log layer mismatch in {tag}"
            assert isinstance(tl.get("slice_sha256"), str) and len(tl["slice_sha256"]) == 64, f"Invalid slice_sha256 in {tag}"
            assert "plato50_deltas" in tl and "L19" in tl["plato50_deltas"] and "L20" in tl["plato50_deltas"], f"Missing deltas in {tag}"
            tlogs[tag] = tl

    # 1. Exact L12 invariance (exact 0.0 across 9 trained checkpoints + base on 16-passage token-mean probe) & multimodal check
    l12_invariance = {}
    base_l12 = acts["base"]["l12_probe"].astype(np.float64)
    mm_all_ok = True
    for tag in tags:
        diff = float(np.max(np.abs(acts[tag]["l12_probe"].astype(np.float64) - base_l12)))
        l12_invariance[tag] = diff
        assert diff == 0.0, f"L12 exact invariance violated in {tag}: {diff}"
        mm = lms[tag].get("multimodal", {})
        if not (mm.get("vision_ok") is True and mm.get("audio_ok") is True):
            mm_all_ok = False
    assert mm_all_ok, "Multimodal vision/audio fixture failed on at least one checkpoint!"

    # 2. Held-out-family 50%-depth Procrustes alignment (full-precision per-seed + canonical OOF bootstrap)
    blind_by_tag = {}
    blind_fp_by_tag = {}
    grp_cos = {bk: {} for bk in BLIND_KEYS}
    grp_sums = {bk: {} for bk in BLIND_KEYS}
    grp_cnts = None
    for tag in tags:
        _, Z_ev, _ = PD.pca_project(acts[tag]["tr_pts"].astype(np.float64), acts[tag]["ev_pts"].astype(np.float64), 256)
        blind_fp_by_tag[tag] = {
            "blind_phi4_50": _eval_procrustes_fp(Z_ev, phi4_ev, ev_groups, ev_doms, ev_ret),
            "blind_olmo3_50": _eval_procrustes_fp(Z_ev, olmo3_ev, ev_groups, ev_doms, ev_ret),
            "diag_c3_50_ev": _eval_procrustes_fp(Z_ev, c3_ev, ev_groups, ev_doms, ev_ret),
            "diag_c4_50_ev": _eval_procrustes_fp(Z_ev, c4_ev, ev_groups, ev_doms, ev_ret),
        }
        blind_by_tag[tag] = {
            tgt: {
                "cos": round(d["cos"], 4),
                "r2_frob": round(d["r2_frob"], 4),
                "r2_opt": round(d["r2_opt"], 4),
                "top1": round(d["top1"], 4),
                "top5": round(d["top5"], 4),
                "by_domain_cos": {dm: round(v, 4) for dm, v in d["by_domain_cos"].items()},
            }
            for tgt, d in blind_fp_by_tag[tag].items()
        }
        for bk, bk_key, Yt in (("phi4_mini", "blind_phi4_50", phi4_ev), ("olmo3_7b", "blind_olmo3_50", olmo3_ev)):
            pt_cos = _canonical_procrustes_oof(Z_ev, Yt, ev_groups, ev_doms)
            assert abs(float(pt_cos.mean()) - blind_fp_by_tag[tag][bk_key]["cos"]) < 1e-10, "Canonical OOF mismatch!"
            _, g_means, g_s, g_c = _group_cos_stats(pt_cos, ev_groups)
            grp_cos[bk][tag] = g_means
            grp_sums[bk][tag] = g_s
            if grp_cnts is None:
                grp_cnts = g_c

    # 3. Zero-label 50%-depth probing across all 10 checkpoints (identical protocol to e4b_transplant50.py!)
    actor_rows = json.loads((SELFSHAPE_DIR / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    bm120_rows = BM238.load_mirror_data()
    bmrel_rows = json.loads((SELFSHAPE_DIR / "bm_rel242_data.json").read_text())["bm_rel_120"]
    bmneut_rows = json.loads((SELFSHAPE_DIR / "bm_neut245_data.json").read_text())["bm_neut_120"]
    ben_mask = np.array([r["actor_type"] == "benign" for r in actor_rows], dtype=bool)
    adv_mask = ~ben_mask

    sid_actor, sp_actor = _sid_and_splits(actor_rows, 24700)
    _, sp_bm120 = _sid_and_splits(bm120_rows, 23800)
    _, sp_bmrel = _sid_and_splits(bmrel_rows, 24200)
    _, sp_bmneut = _sid_and_splits(bmneut_rows, 24500)

    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    self2_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] == "self2"])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    sg_rows, _, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])

    SLOT50 = 2
    r_d240 = {k: dict(np.load(DEPTH_DIR / f"depth240_{k}.npz")) for k in RIVALS}
    r_actor = {k: dict(np.load(DEPTH_DIR / f"bmactor247_{k}.npz")) for k in RIVALS}
    r_bmrel = {k: dict(np.load(DEPTH_DIR / f"bmrel242_{k}.npz")) for k in RIVALS}
    r_bmneut = {k: dict(np.load(DEPTH_DIR / f"bmneut245_{k}.npz")) for k in RIVALS}

    # Precompute frozen 3-rival reference PCA + GPA anchor ONCE per pool (exact identical frame across all 10 checkpoints!)
    frozen_ref = {}
    for pool_key in ("mean", "last"):
        pca_r = {}
        Y_n = {}
        for k in RIVALS:
            M = r_d240[k][f"neut_{pool_key}"][SLOT50].astype(np.float64)
            mu = M.mean(axis=0, keepdims=True)
            Mc = M - mu
            _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
            V = Vt[:256].T
            Z = Mc @ V
            sc = float(np.sqrt(np.mean(np.sum(Z * Z, axis=1))) + 1e-12)
            Y_n[k] = Z / sc
            pca_r[k] = (mu, V, sc)
        C_ref, _, _, _, _ = PD.generalized_procrustes_consensus([Y_n[k] for k in C3_KEYS], [Y_n[k] for k in C3_KEYS], n_iters=10)
        R_r = {}
        for k in RIVALS:
            U, _, Vt = np.linalg.svd(Y_n[k].T @ C_ref, full_matrices=False)
            R_r[k] = U @ Vt

        def _map_r(M, k):
            mu, V, sc = pca_r[k]
            return (((M.astype(np.float64) - mu) @ V) / sc) @ R_r[k]

        Z_act_r, Z_nt_r, Z_120_r, Z_rel_r, Q5_r = {}, {}, {}, {}, {}
        for k in RIVALS:
            Z_act_r[k] = _map_r(r_actor[k][f"pos_{pool_key}"][SLOT50], k) - _map_r(r_actor[k][f"neg_{pool_key}"][SLOT50], k)
            Z_nt_r[k] = _map_r(r_bmneut[k][f"pos_{pool_key}"][SLOT50], k) - _map_r(r_bmneut[k][f"neg_{pool_key}"][SLOT50], k)
            Z_120_r[k] = _map_r(r_d240[k][f"bm120_pos_{pool_key}"][SLOT50], k) - _map_r(r_d240[k][f"bm120_neg_{pool_key}"][SLOT50], k)
            Z_rel_r[k] = _map_r(r_bmrel[k][f"pos_{pool_key}"][SLOT50], k) - _map_r(r_bmrel[k][f"neg_{pool_key}"][SLOT50], k)
            Dh_k = _map_r(r_d240[k][f"house_pos_{pool_key}"][SLOT50][house_idx], k) - _map_r(r_d240[k][f"house_neg_{pool_key}"][SLOT50][house_idx], k)
            Dv_k = _map_r(r_d240[k][f"val_pos_{pool_key}"][SLOT50], k) - _map_r(r_d240[k][f"val_neg_{pool_key}"][SLOT50], k)
            bk_e, _, _ = _build_valence_erasure_bases(Dv_k, Dh_k, sg_doms, house_cats)
            Q5_r[k] = bk_e["val5"]
        frozen_ref[pool_key] = (C_ref, Z_act_r, Z_nt_r, Z_120_r, Z_rel_r, Q5_r)

    probing_by_tag = {}
    setting_rates_store = {}
    fh = AR.make_stratified_2fold(house_cats)
    fs = AR.make_stratified_2fold(np.array(["self2"] * len(self2_idx)))

    for tag in tags:
        aa = acts[tag]
        probing_by_tag[tag] = {"within_native": {}, "within_256d": {}, "rival_to_gemma": {}, "gemma_to_rival": {}, "house_L20": {}}
        Dh_last = aa["house_pos_last"].astype(np.float64) - aa["house_neg_last"].astype(np.float64)
        rh = AR.eval_disjoint_2fold([Dh_last[house_idx]], Dh_last[house_idx], house_cats, fh, is_house=True, seed=2511)
        rs = AR.eval_disjoint_2fold([Dh_last[self2_idx]], Dh_last[self2_idx], np.array(["self2"] * len(self2_idx)), fs, is_house=False, seed=2512)
        u_h = AR.house_dir(Dh_last[house_idx], house_cats)
        probing_by_tag[tag]["house_L20"] = {
            "house_2f_acc": rh["acc"],
            "self2_2f_acc": rs["acc"],
            "house_to_self2_acc": round(float(np.mean((Dh_last[self2_idx] @ u_h) > 0.0)), 4),
        }

        for pool_key in ("mean", "last"):
            Dh_nat = aa[f"house_pos_{pool_key}"][house_idx].astype(np.float64) - aa[f"house_neg_{pool_key}"][house_idx].astype(np.float64)
            Dv_nat = aa[f"val_pos_{pool_key}"].astype(np.float64) - aa[f"val_neg_{pool_key}"].astype(np.float64)
            b_nat, _, _ = _build_valence_erasure_bases(Dv_nat, Dh_nat, sg_doms, house_cats)
            Q5_nat = b_nat["val5"]
            D_act_nat = aa[f"actor_pos_{pool_key}"].astype(np.float64) - aa[f"actor_neg_{pool_key}"].astype(np.float64)
            D_nt_nat = aa[f"bmneut_pos_{pool_key}"].astype(np.float64) - aa[f"bmneut_neg_{pool_key}"].astype(np.float64)
            D_120_nat = aa[f"bm120_pos_{pool_key}"].astype(np.float64) - aa[f"bm120_neg_{pool_key}"].astype(np.float64)
            D_rel_nat = aa[f"bmrel_pos_{pool_key}"].astype(np.float64) - aa[f"bmrel_neg_{pool_key}"].astype(np.float64)

            wn_act = _eval_cv2_20split(D_act_nat, D_act_nat, Q5_nat, Q5_nat, sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
            probing_by_tag[tag]["within_native"][pool_key] = {
                "bm_actor2x2_247": {k: v for k, v in wn_act.items() if not k.startswith("_")},
                "bm120": _eval_cv2_20split(D_120_nat, D_120_nat, Q5_nat, Q5_nat, sp_bm120)["acc"],
                "bm_rel242": _eval_cv2_20split(D_rel_nat, D_rel_nat, Q5_nat, Q5_nat, sp_bmrel)["acc"],
                "bm_neut245": _eval_cv2_20split(D_nt_nat, D_nt_nat, Q5_nat, Q5_nat, sp_bmneut)["acc"],
            }
            if pool_key == "mean":
                setting_rates_store.setdefault("within_native_actor_mean", {})[tag] = np.array(
                    [wn_act["_item_rate"][sid_actor == s].mean() for s in sorted(set(sid_actor))]
                )

            C_ref, Z_act_r, Z_nt_r, Z_120_r, Z_rel_r, Q5_r = frozen_ref[pool_key]
            Mg = aa[f"neut_{pool_key}"].astype(np.float64)
            mu_g = Mg.mean(axis=0, keepdims=True)
            Mc_g = Mg - mu_g
            _, _, Vt_g = np.linalg.svd(Mc_g, full_matrices=False)
            V_g = Vt_g[:256].T
            Z_g = Mc_g @ V_g
            sc_g = float(np.sqrt(np.mean(np.sum(Z_g * Z_g, axis=1))) + 1e-12)
            Y_g = Z_g / sc_g
            U_r, _, Vt_r = np.linalg.svd(Y_g.T @ C_ref, full_matrices=False)
            R_g = U_r @ Vt_r

            def _map_g(M):
                return (((M.astype(np.float64) - mu_g) @ V_g) / sc_g) @ R_g

            Z_act_g = _map_g(aa[f"actor_pos_{pool_key}"]) - _map_g(aa[f"actor_neg_{pool_key}"])
            Z_nt_g = _map_g(aa[f"bmneut_pos_{pool_key}"]) - _map_g(aa[f"bmneut_neg_{pool_key}"])
            Z_120_g = _map_g(aa[f"bm120_pos_{pool_key}"]) - _map_g(aa[f"bm120_neg_{pool_key}"])
            Z_rel_g = _map_g(aa[f"bmrel_pos_{pool_key}"]) - _map_g(aa[f"bmrel_neg_{pool_key}"])
            Dh_g = _map_g(aa[f"house_pos_{pool_key}"][house_idx]) - _map_g(aa[f"house_neg_{pool_key}"][house_idx])
            Dv_g = _map_g(aa[f"val_pos_{pool_key}"]) - _map_g(aa[f"val_neg_{pool_key}"])
            bg, _, _ = _build_valence_erasure_bases(Dv_g, Dh_g, sg_doms, house_cats)
            Q5_g = bg["val5"]

            w256 = _eval_cv2_20split(Z_act_g, Z_act_g, Q5_g, Q5_g, sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
            probing_by_tag[tag]["within_256d"][pool_key] = {k: v for k, v in w256.items() if not k.startswith("_")}

            r2g = {"per_source": {}, "summary": {}}
            g2r = {"per_target": {}, "summary": {}}
            r2g_cv_items, g2r_cv_items, r2g_zs_items, g2r_zs_items = {}, {}, {}, {}

            Dt_g_p = V234._project_out_rows(Z_act_g, Q5_g)
            u_nt_g = V234._project_out(AR.unit(Z_nt_g.mean(0)), Q5_g)
            for rk in RIVALS:
                cv_in = _eval_cv2_20split(Z_act_r[rk], Z_act_g, Q5_r[rk], Q5_g, sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
                u_nt_r = V234._project_out(V234._project_out(AR.unit(Z_nt_r[rk].mean(0)), Q5_r[rk]), Q5_g)
                zs_in_sc = Dt_g_p @ u_nt_r
                r2g["per_source"][rk] = {
                    "actor_cv2": {k: v for k, v in cv_in.items() if not k.startswith("_")},
                    "zs_bmneut_to_actor": {
                        "acc": round(float(np.mean(zs_in_sc > 0)), 4),
                        "benign": round(float(np.mean(zs_in_sc[ben_mask] > 0)), 4),
                        "adverse": round(float(np.mean(zs_in_sc[adv_mask] > 0)), 4),
                    },
                    "bm120_cv2": _eval_cv2_20split(Z_120_r[rk], Z_120_g, Q5_r[rk], Q5_g, sp_bm120)["acc"],
                    "bmrel_cv2": _eval_cv2_20split(Z_rel_r[rk], Z_rel_g, Q5_r[rk], Q5_g, sp_bmrel)["acc"],
                }
                r2g_cv_items[rk] = cv_in["_item_rate"]
                r2g_zs_items[rk] = (zs_in_sc > 0).astype(np.float64)

                cv_out = _eval_cv2_20split(Z_act_g, Z_act_r[rk], Q5_g, Q5_r[rk], sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
                Dt_r_p = V234._project_out_rows(Z_act_r[rk], Q5_r[rk])
                u_nt_go = V234._project_out(u_nt_g, Q5_r[rk])
                zs_out_sc = Dt_r_p @ u_nt_go
                g2r["per_target"][rk] = {
                    "actor_cv2": {k: v for k, v in cv_out.items() if not k.startswith("_")},
                    "zs_bmneut_to_actor": {
                        "acc": round(float(np.mean(zs_out_sc > 0)), 4),
                        "benign": round(float(np.mean(zs_out_sc[ben_mask] > 0)), 4),
                        "adverse": round(float(np.mean(zs_out_sc[adv_mask] > 0)), 4),
                    },
                }
                g2r_cv_items[rk] = cv_out["_item_rate"]
                g2r_zs_items[rk] = (zs_out_sc > 0).astype(np.float64)

            for grp_nm, keys_sub in (("all5", RIVALS), ("teacher3", C3_KEYS), ("blind2_phi4_olmo", BLIND_KEYS)):
                r2g["summary"][grp_nm] = {
                    "actor_cv2_acc": round(float(np.mean([r2g["per_source"][k]["actor_cv2"]["acc"] for k in keys_sub])), 4),
                    "actor_cv2_adv": round(float(np.mean([r2g["per_source"][k]["actor_cv2"]["adverse"] for k in keys_sub])), 4),
                    "actor_cv2_all4": round(float(np.mean([r2g["per_source"][k]["actor_cv2"]["all4"] for k in keys_sub])), 4),
                    "zs_bmneut_acc": round(float(np.mean([r2g["per_source"][k]["zs_bmneut_to_actor"]["acc"] for k in keys_sub])), 4),
                }
                g2r["summary"][grp_nm] = {
                    "actor_cv2_acc": round(float(np.mean([g2r["per_target"][k]["actor_cv2"]["acc"] for k in keys_sub])), 4),
                    "actor_cv2_adv": round(float(np.mean([g2r["per_target"][k]["actor_cv2"]["adverse"] for k in keys_sub])), 4),
                    "actor_cv2_all4": round(float(np.mean([g2r["per_target"][k]["actor_cv2"]["all4"] for k in keys_sub])), 4),
                    "zs_bmneut_acc": round(float(np.mean([g2r["per_target"][k]["zs_bmneut_to_actor"]["acc"] for k in keys_sub])), 4),
                }
                if pool_key == "mean":
                    avg_in = np.mean([r2g_cv_items[k] for k in keys_sub], axis=0)
                    avg_out = np.mean([g2r_cv_items[k] for k in keys_sub], axis=0)
                    avg_zsin = np.mean([r2g_zs_items[k] for k in keys_sub], axis=0)
                    avg_zsout = np.mean([g2r_zs_items[k] for k in keys_sub], axis=0)
                    for ptag, vec in (
                        (f"r2g_cv2_{grp_nm}", avg_in),
                        (f"g2r_cv2_{grp_nm}", avg_out),
                        (f"r2g_zs_{grp_nm}", avg_zsin),
                        (f"g2r_zs_{grp_nm}", avg_zsout),
                    ):
                        setting_rates_store.setdefault(ptag, {})[tag] = np.array(
                            [vec[sid_actor == s].mean() for s in sorted(set(sid_actor))]
                        )

            probing_by_tag[tag]["rival_to_gemma"][pool_key] = r2g
            probing_by_tag[tag]["gemma_to_rival"][pool_key] = g2r

    # 4. Aggregate 3-seed means +/- SD + paired across-seed deltas (from FULL-PRECISION per-seed values!)
    def _agg_blind_arm(arm: str):
        if arm == "base":
            return {
                tgt: {
                    **blind_by_tag["base"][tgt],
                    "r2_opt_fp": round(blind_fp_by_tag["base"][tgt]["r2_opt"], 6),
                    "top1_fp": round(blind_fp_by_tag["base"][tgt]["top1"], 6),
                }
                for tgt in ("blind_phi4_50", "blind_olmo3_50", "diag_c3_50_ev", "diag_c4_50_ev")
            }
        seed_tags = [_tag_for(arm, s) for s in SEEDS]
        res = {}
        for tgt in ("blind_phi4_50", "blind_olmo3_50", "diag_c3_50_ev", "diag_c4_50_ev"):
            res[tgt] = {
                m: round(float(np.mean([blind_fp_by_tag[t][tgt][m] for t in seed_tags])), 4)
                for m in ("cos", "r2_frob", "r2_opt", "top1", "top5")
            }
            res[tgt]["r2_opt_fp"] = round(float(np.mean([blind_fp_by_tag[t][tgt]["r2_opt"] for t in seed_tags])), 6)
            res[tgt]["top1_fp"] = round(float(np.mean([blind_fp_by_tag[t][tgt]["top1"] for t in seed_tags])), 6)
            res[tgt]["sd"] = {
                m: round(float(np.std([blind_fp_by_tag[t][tgt][m] for t in seed_tags], ddof=1)), 4)
                for m in ("cos", "r2_frob", "r2_opt", "top1", "top5")
            }
            res[tgt]["by_domain_cos"] = {
                d: round(float(np.mean([blind_fp_by_tag[t][tgt]["by_domain_cos"][d] for t in seed_tags])), 4)
                for d in DOMAINS
            }
            res[tgt]["by_domain_cos_sd"] = {
                d: round(float(np.std([blind_fp_by_tag[t][tgt]["by_domain_cos"][d] for t in seed_tags], ddof=1)), 4)
                for d in DOMAINS
            }
            res[tgt]["per_seed"] = {
                str(s): {
                    "cos": blind_by_tag[_tag_for(arm, s)][tgt]["cos"],
                    "r2_frob": blind_by_tag[_tag_for(arm, s)][tgt]["r2_frob"],
                    "r2_opt": blind_by_tag[_tag_for(arm, s)][tgt]["r2_opt"],
                    "top1": blind_by_tag[_tag_for(arm, s)][tgt]["top1"],
                    "top5": blind_by_tag[_tag_for(arm, s)][tgt]["top5"],
                    "by_domain_cos": blind_by_tag[_tag_for(arm, s)][tgt]["by_domain_cos"],
                }
                for s in SEEDS
            }
        return res

    blind_3seed = {arm: _agg_blind_arm(arm) for arm in ("base", "ctrl50", "c4_50", "c3_50")}

    lm_3seed = {"base": lms["base"]}
    for arm in TRAINED_ARMS:
        stags = [_tag_for(arm, s) for s in SEEDS]
        lm_3seed[arm] = {
            "arm": arm,
            "n_seeds": len(SEEDS),
            "seeds": list(SEEDS),
            "held_gut_nll_mean": round(float(np.mean([lms[t]["held_gut_nll_mean"] for t in stags])), 4),
            "held_gut_nll_sd": round(float(np.std([lms[t]["held_gut_nll_mean"] for t in stags], ddof=1)), 4),
            "held_gut_nll_per_seed": {str(s): lms[_tag_for(arm, s)]["held_gut_nll_mean"] for s in SEEDS},
            "held_gut_n": lms[stags[0]]["held_gut_n"],
            "proc_held": {
                "overall": {
                    "n": lms[stags[0]]["proc_held"]["overall"]["n"],
                    "with": round(float(np.mean([lms[t]["proc_held"]["overall"]["with"] for t in stags])), 4),
                    "with_sd": round(float(np.std([lms[t]["proc_held"]["overall"]["with"] for t in stags], ddof=1)), 4),
                    "without": round(float(np.mean([lms[t]["proc_held"]["overall"]["without"] for t in stags])), 4),
                    "gain": round(float(np.mean([lms[t]["proc_held"]["overall"]["gain"] for t in stags])), 4),
                    "gain_sd": round(float(np.std([lms[t]["proc_held"]["overall"]["gain"] for t in stags], ddof=1)), 4),
                    "with_per_seed": {str(s): lms[_tag_for(arm, s)]["proc_held"]["overall"]["with"] for s in SEEDS},
                }
            },
            "actor_nll_whole_acc": round(float(np.mean([lms[t]["actor_nll_whole_acc"] for t in stags])), 4),
            "actor_nll_whole_sd": round(float(np.std([lms[t]["actor_nll_whole_acc"] for t in stags], ddof=1)), 4),
            "actor_nll_changed_acc": round(float(np.mean([lms[t]["actor_nll_changed_acc"] for t in stags])), 4),
            "actor_nll_changed_sd": round(float(np.std([lms[t]["actor_nll_changed_acc"] for t in stags], ddof=1)), 4),
            "multimodal_all_seeds_pass": all(
                lms[t]["multimodal"]["vision_ok"] and lms[t]["multimodal"]["audio_ok"] for t in stags
            ),
            "per_seed": {str(s): lms[_tag_for(arm, s)] for s in SEEDS},
        }

    # Probing 3-seed summary (with top-level compatibility matching 1-seed paths!)
    probing_3seed = {"base": probing_by_tag["base"]}
    for arm in TRAINED_ARMS:
        stags = [_tag_for(arm, s) for s in SEEDS]
        nat_mean = [probing_by_tag[t]["within_native"]["mean"]["bm_actor2x2_247"]["acc"] for t in stags]
        nat_ben = [probing_by_tag[t]["within_native"]["mean"]["bm_actor2x2_247"]["benign"] for t in stags]
        nat_adv = [probing_by_tag[t]["within_native"]["mean"]["bm_actor2x2_247"]["adverse"] for t in stags]
        nat_a4 = [probing_by_tag[t]["within_native"]["mean"]["bm_actor2x2_247"]["all4"] for t in stags]
        probing_3seed[arm] = {
            "within_native": {
                "mean": {
                    "bm_actor2x2_247": {
                        "acc": round(float(np.mean(nat_mean)), 4),
                        "acc_sd": round(float(np.std(nat_mean, ddof=1)), 4),
                        "benign": round(float(np.mean(nat_ben)), 4),
                        "adverse": round(float(np.mean(nat_adv)), 4),
                        "all4": round(float(np.mean(nat_a4)), 4),
                        "per_seed": {str(s): probing_by_tag[_tag_for(arm, s)]["within_native"]["mean"]["bm_actor2x2_247"]["acc"] for s in SEEDS},
                    },
                    "bm120": round(float(np.mean([probing_by_tag[t]["within_native"]["mean"]["bm120"] for t in stags])), 4),
                    "bm_rel242": round(float(np.mean([probing_by_tag[t]["within_native"]["mean"]["bm_rel242"] for t in stags])), 4),
                    "bm_neut245": round(float(np.mean([probing_by_tag[t]["within_native"]["mean"]["bm_neut245"] for t in stags])), 4),
                },
                "last": {
                    "bm_actor2x2_247": {
                        "acc": round(float(np.mean([probing_by_tag[t]["within_native"]["last"]["bm_actor2x2_247"]["acc"] for t in stags])), 4),
                        "acc_sd": round(float(np.std([probing_by_tag[t]["within_native"]["last"]["bm_actor2x2_247"]["acc"] for t in stags], ddof=1)), 4),
                    }
                },
            },
            "rival_to_gemma_mean_summary": {
                grp: {
                    m: _stats([probing_by_tag[t]["rival_to_gemma"]["mean"]["summary"][grp][m] for t in stags], 4)
                    for m in ("actor_cv2_acc", "actor_cv2_adv", "actor_cv2_all4", "zs_bmneut_acc")
                }
                for grp in ("all5", "teacher3", "blind2_phi4_olmo")
            },
            "gemma_to_rival_mean_summary": {
                grp: {
                    m: _stats([probing_by_tag[t]["gemma_to_rival"]["mean"]["summary"][grp][m] for t in stags], 4)
                    for m in ("actor_cv2_acc", "actor_cv2_adv", "actor_cv2_all4", "zs_bmneut_acc")
                }
                for grp in ("all5", "teacher3", "blind2_phi4_olmo")
            },
            "house_L20": {
                m: _stats([probing_by_tag[t]["house_L20"][m] for t in stags], 4)
                for m in ("house_2f_acc", "self2_2f_acc", "house_to_self2_acc")
            },
            "per_seed": {str(s): probing_by_tag[_tag_for(arm, s)] for s in SEEDS},
        }

    # 5. Paired across-seed deltas (exact seed-matched full-precision differences for seeds 251, 252, 253!)
    paired_3seed = {}
    paired_fp_r2f = {}
    for cmp_arm, ref_arm in (("c3_50", "ctrl50"), ("c3_50", "base"), ("c4_50", "ctrl50"), ("c4_50", "base"), ("c3_50", "c4_50"), ("ctrl50", "base")):
        key = f"{cmp_arm}_minus_{ref_arm}"
        paired_3seed[key] = {}
        paired_fp_r2f[key] = {}
        for bk, bk_key in (("phi4_mini", "blind_phi4_50"), ("olmo3_7b", "blind_olmo3_50")):
            d_cos, d_r2f, d_r2o, d_t1 = [], [], [], []
            d_doms = {d: [] for d in DOMAINS}
            gcos_diffs = []
            gsum_diffs = []
            for s in SEEDS:
                tc = _tag_for(cmp_arm, s)
                tr = _tag_for(ref_arm, s)
                bc = blind_fp_by_tag[tc][bk_key]
                br = blind_fp_by_tag[tr][bk_key]
                d_cos.append(bc["cos"] - br["cos"])
                d_r2f.append(bc["r2_frob"] - br["r2_frob"])
                d_r2o.append(bc["r2_opt"] - br["r2_opt"])
                d_t1.append(bc["top1"] - br["top1"])
                for d in DOMAINS:
                    d_doms[d].append(bc["by_domain_cos"][d] - br["by_domain_cos"][d])
                gcos_diffs.append(grp_cos[bk][tc] - grp_cos[bk][tr])
                gsum_diffs.append(grp_sums[bk][tc] - grp_sums[bk][tr])
            paired_fp_r2f[key][bk] = d_r2f
            mean_gcos_diff = np.mean(gcos_diffs, axis=0)
            mean_gsum_diff = np.mean(gsum_diffs, axis=0)
            paired_3seed[key][bk] = {
                "delta_cos_pt": _stats(d_cos, 4),
                "delta_r2_frob": _stats(d_r2f, 4),
                "delta_r2_opt": _stats(d_r2o, 4),
                "delta_top1": _stats(d_t1, 4),
                "wins_3of3_r2_frob": sum(1 for v in d_r2f if v > 0),
                "by_domain_delta_cos": {d: _stats(d_doms[d], 4) for d in DOMAINS},
                "clause_cos_3seed_mean_paired_boot": _paired_cluster_ratio_boot_ci(mean_gsum_diff, grp_cnts, n_boot=2000, seed=2511),
                "group_cos_3seed_mean_paired_boot": _paired_boot_ci(mean_gcos_diff, n_boot=2000, seed=2511),
                "group_cos_per_seed_paired_boot": {
                    str(s): _paired_boot_ci(gcos_diffs[i], n_boot=2000, seed=2511 + i) for i, s in enumerate(SEEDS)
                },
            }
        # LM & Probing paired deltas across seeds
        d_nll = [lms[_tag_for(cmp_arm, s)]["held_gut_nll_mean"] - lms[_tag_for(ref_arm, s)]["held_gut_nll_mean"] for s in SEEDS]
        d_proc = [lms[_tag_for(cmp_arm, s)]["proc_held"]["overall"]["with"] - lms[_tag_for(ref_arm, s)]["proc_held"]["overall"]["with"] for s in SEEDS]
        d_nat = [probing_by_tag[_tag_for(cmp_arm, s)]["within_native"]["mean"]["bm_actor2x2_247"]["acc"]
                 - probing_by_tag[_tag_for(ref_arm, s)]["within_native"]["mean"]["bm_actor2x2_247"]["acc"] for s in SEEDS]
        d_g2r_cv = [probing_by_tag[_tag_for(cmp_arm, s)]["gemma_to_rival"]["mean"]["summary"]["blind2_phi4_olmo"]["actor_cv2_acc"]
                    - probing_by_tag[_tag_for(ref_arm, s)]["gemma_to_rival"]["mean"]["summary"]["blind2_phi4_olmo"]["actor_cv2_acc"] for s in SEEDS]
        d_g2r_zs = [probing_by_tag[_tag_for(cmp_arm, s)]["gemma_to_rival"]["mean"]["summary"]["blind2_phi4_olmo"]["zs_bmneut_acc"]
                    - probing_by_tag[_tag_for(ref_arm, s)]["gemma_to_rival"]["mean"]["summary"]["blind2_phi4_olmo"]["zs_bmneut_acc"] for s in SEEDS]
        d_r2g_all5 = [probing_by_tag[_tag_for(cmp_arm, s)]["rival_to_gemma"]["mean"]["summary"]["all5"]["actor_cv2_acc"]
                      - probing_by_tag[_tag_for(ref_arm, s)]["rival_to_gemma"]["mean"]["summary"]["all5"]["actor_cv2_acc"] for s in SEEDS]
        d_r2g_t3 = [probing_by_tag[_tag_for(cmp_arm, s)]["rival_to_gemma"]["mean"]["summary"]["teacher3"]["actor_cv2_acc"]
                    - probing_by_tag[_tag_for(ref_arm, s)]["rival_to_gemma"]["mean"]["summary"]["teacher3"]["actor_cv2_acc"] for s in SEEDS]
        nll_doc_mean_diff = np.mean(
            [acts[_tag_for(cmp_arm, s)]["gut_nlls"].astype(np.float64) - acts[_tag_for(ref_arm, s)]["gut_nlls"].astype(np.float64) for s in SEEDS],
            axis=0,
        )
        proc_item_mean_diff = np.mean(
            [acts[_tag_for(cmp_arm, s)]["proc_with_items"].astype(np.float64) - acts[_tag_for(ref_arm, s)]["proc_with_items"].astype(np.float64) for s in SEEDS],
            axis=0,
        )
        paired_3seed[key]["lm_and_probing"] = {
            "held_gut_nll_delta": _stats(d_nll, 5),
            "held_gut_nll_passage_boot_3seed_mean": _paired_boot_ci(nll_doc_mean_diff, n_boot=2000, seed=2522),
            "proc_held_with_delta": _stats(d_proc, 4),
            "proc_held_item_boot_3seed_mean": _paired_boot_ci(proc_item_mean_diff, n_boot=2000, seed=2523),
            "within_native_actor_mean_delta": _stats(d_nat, 4),
            "g2r_cv2_blind2_delta": _stats(d_g2r_cv, 4),
            "g2r_zs_blind2_delta": _stats(d_g2r_zs, 4),
            "r2g_cv2_all5_delta": _stats(d_r2g_all5, 4),
            "r2g_cv2_teacher3_delta": _stats(d_r2g_t3, 4),
        }

    # Base-referenced lift ratios computed from FULL-PRECISION deltas (zero intermediate rounding!)
    lift_ratios = {}
    for bk in BLIND_KEYS:
        c3_b = paired_fp_r2f["c3_50_minus_base"][bk]
        ct_b = paired_fp_r2f["ctrl50_minus_base"][bk]
        per_s_rat_fp = [c3_b[i] / ct_b[i] if abs(ct_b[i]) > 1e-9 else None for i in range(len(SEEDS))]
        per_s_rat = [round(r, 2) if r is not None else None for r in per_s_rat_fp]
        m_c3 = float(np.mean(c3_b))
        m_ct = float(np.mean(ct_b))
        lift_ratios[bk] = {
            "c3_50_minus_base_mean": round(m_c3, 4),
            "ctrl50_minus_base_mean": round(m_ct, 4),
            "ratio_of_3seed_means": round(m_c3 / m_ct, 2) if abs(m_ct) > 1e-9 else None,
            "mean_of_per_seed_ratios": round(float(np.mean([r for r in per_s_rat_fp if r is not None])), 2),
            "per_seed_ratios": {str(SEEDS[i]): per_s_rat[i] for i in range(len(SEEDS))},
        }

    # Weight drift & train config block (validated strictly from all 9 committed/local train logs!)
    weight_drift = {}
    for arm in TRAINED_ARMS:
        l19_rel, l20_rel, l19_l2, l20_l2 = [], [], [], []
        by_s = {}
        for s in SEEDS:
            t = _tag_for(arm, s)
            dd = tlogs[t]["plato50_deltas"]
            l19_rel.append(dd["L19"]["rel_delta"])
            l20_rel.append(dd["L20"]["rel_delta"])
            l19_l2.append(dd["L19"]["l2_delta"])
            l20_l2.append(dd["L20"]["l2_delta"])
            by_s[str(s)] = {
                "L19": dd["L19"],
                "L20": dd["L20"],
                "elapsed_sec": tlogs[t].get("plato50_elapsed_sec"),
                "slice_sha256": tlogs[t].get("slice_sha256"),
                "plato50_steps": tlogs[t].get("plato50_steps"),
                "plato50_w": tlogs[t].get("plato50_w"),
            }
        weight_drift[arm] = {
            "L19_rel_delta": _stats(l19_rel, 6),
            "L20_rel_delta": _stats(l20_rel, 6),
            "L19_l2_delta": _stats(l19_l2, 4),
            "L20_l2_delta": _stats(l20_l2, 4),
            "per_seed": by_s,
        }

    train_config = {
        "provenance_note": "Fields under observed_in_train_logs are parsed and asserted directly from all 9 train_*_log.json files; fields under runner_code_defaults reflect the resolved constants in e4b_transplant50_cuda.py.",
        "observed_in_train_logs": {
            "seeds": list(SEEDS),
            "n_seeds": len(SEEDS),
            "steps": int(tlogs["c3_50"]["plato50_steps"]),
            "plato_w_c3_c4": float(tlogs["c3_50"]["plato50_w"]),
            "plato_w_ctrl50": float(tlogs["ctrl50"]["plato50_w"]),
            "mlp_layers": list(tlogs["c3_50"]["plato50_mlp_layers"]),
            "plato50_layer": int(tlogs["c3_50"]["plato50_layer"]),
        },
        "runner_code_defaults": {
            "model_id": "google/gemma-4-E4B-it",
            "precision": "bf16_backbone_fp32_master_mlp",
            "grad_accum": 4,
            "optimizer": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay_mlp": 0.01,
            "weight_decay_head": 0.0,
            "lr_mlp": 1.5e-5,
            "lr_head": 1e-3,
            "lr_schedule": "linear_warmup_10pct_then_cosine",
            "grad_clip_norm": 1.0,
            "ce_w": 0.25,
            "kl_w": 1.5,
            "anneal_start_frac": 0.65,
            "anneal_end_frac": 0.90,
            "train_corpus": "neutral_only (gut+proc+chat, zero myth passages)",
        },
        "eval_wrapper_note": "Evaluated via LivingE4B wrapper (aion_8b/e4b_living.py) with Titans disabled during base-study extraction and enabled on multimodal fixtures; the transplant itself modifies only L19-20 MLPs (0 incremental parameters).",
        "weight_drift_mlp_l19_l20": weight_drift,
    }

    out = {
        "script_sha256": _sha256(Path(__file__)),
        "targets_sha256": _sha256(_targets_path()),
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "plato50_layer": PLATO50_LAYER,
        "mlp_layers": MLP_LAYERS,
        "c3_teacher_families": list(C3_KEYS),
        "c4_teacher_families": list(C4_KEYS),
        "held_out_target_families": list(BLIND_KEYS),
        "bootstrap_scope_note": "Passage/item bootstraps condition on the 3 fitted training-order seeds (251, 252, 253); OOF Procrustes uses canonical domain-stratified Random(2288) 2-fold split with train-fold-only centering and affine map.",
        "l12_invariance_probe_desc": "max |Delta| of token-mean L12 hidden states across first 16 eval passages (9 trained checkpoints vs Base, exact 0.0 by construction)",
        "train_config": train_config,
        "l12_invariance_max_abs_diff": l12_invariance,
        "lm_guardrails": lm_3seed,
        "blind_50pct_alignment": blind_3seed,
        "blind_50pct_paired_deltas_3seed": paired_3seed,
        "base_referenced_lift_ratios_3seed": lift_ratios,
        "probing_50pct": probing_3seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    _atomic_json(EXP_DIR / "exp251_e4b_transplant50_3seed_results.json", out)
    if write_repo_json:
        _atomic_json(OUT_REPO_JSON, out)

    print("\n================ EXP 251 3-SEED SUMMARY (SEEDS 251, 252, 253) ================", flush=True)
    for arm in ("base", "ctrl50", "c4_50", "c3_50"):
        p4 = blind_3seed[arm]["blind_phi4_50"]
        ol = blind_3seed[arm]["blind_olmo3_50"]
        lg = lm_3seed[arm]
        nat = probing_3seed[arm]["within_native"]["mean"]["bm_actor2x2_247"]
        p4_sd = f"±{p4['sd']['r2_frob']:.4f}" if "sd" in p4 else "       "
        ol_sd = f"±{ol['sd']['r2_frob']:.4f}" if "sd" in ol else "       "
        nll_sd = f"±{lg['held_gut_nll_sd']:.4f}" if "held_gut_nll_sd" in lg else "       "
        print(
            f"  {arm:7s} | Phi4: cos={p4['cos']:.4f} R2f={p4['r2_frob']:+.4f}{p4_sd} (R2o={p4['r2_opt_fp']:.3f}, T1={p4['top1_fp']*100:.1f}%) "
            f"| OLMo: cos={ol['cos']:.4f} R2f={ol['r2_frob']:+.4f}{ol_sd} (R2o={ol['r2_opt_fp']:.3f}, T1={ol['top1_fp']*100:.1f}%) "
            f"| NLL={lg['held_gut_nll_mean']:.4f}{nll_sd} | MCQ={lg['proc_held']['overall']['with']*100:.2f}% "
            f"| natAct={nat['acc']*100:.2f}%",
            flush=True,
        )
    print("\n--- Paired 3-Seed Deltas (c3_50 - ctrl50) ---", flush=True)
    d = paired_3seed["c3_50_minus_ctrl50"]
    for bk in BLIND_KEYS:
        print(f"  {bk}: d_cos={d[bk]['delta_cos_pt']} d_r2f={d[bk]['delta_r2_frob']} d_t1={d[bk]['delta_top1']} wins={d[bk]['wins_3of3_r2_frob']}/3")
        print(f"    group_cos_3seed_mean_boot={d[bk]['group_cos_3seed_mean_paired_boot']}")
    print("  lm_and_probing:", json.dumps(d["lm_and_probing"], indent=2))
    print("\n--- Base-Referenced Lift Ratios (3-seed) ---", flush=True)
    print(json.dumps(lift_ratios, indent=2))
    print(f"\nSAVED {OUT_REPO_JSON} in {out['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    analyze_3seed(write_repo_json=True)
