"""Experiment 242 (FROZEN PRE-SPECIFIED PROTOCOL): Cross-Domain Abstract Relational Mirrored Identical-BoW Swaps.

THIS SCRIPT IS WRITTEN AND SHA-256 FROZEN PRIOR TO AUTHORING ANY ITEM OF `bm_rel242_data.json`.

Pre-specified dataset contract (`bm_rel242_data.json`):
  - 120 mirrored pairs = 60 brand-new abstract relational / causal / reciprocity / institutional /
    systems-dynamics settings x 2 clause orders (`orig` and `mirror`), exact 60 `fav_pos == 'first'`
    and 60 `fav_pos == 'second'`.
  - Exact identical bag-of-words on all 120/120 pairs: `Counter(words(pos)) == Counter(words(neg))`
    and `pos != neg`.
  - Zero setting overlap with `bow_mirror238_data.json` (60 physical maintenance settings).

Pre-specified hypotheses & primary metrics (evaluated across all 6 MLX families at 30% & 50% depth,
plus full 5-depth profile via `depth240` caches, using neutral-text Procrustes rotations):
  1. Primary Replication (20 setting-disjoint 2-fold splits, both mirrors kept in same fold):
     - `Gemma 4 -> 5 rival MLX labs` AND `Qwen 3.5 -> 5 rival MLX labs` (mean pool & last token):
       * `cv2_perp_v2_120`, `cv2_perp_v2_both_mirrors` (vs 25% no-signal chance / 0% order-only),
       * setting-clustered sign-flip Holm-adjusted p-value across the 5 targets (`alpha = 0.05`).
  2. Strict Cross-Domain Zero-Shot Transfer (`bow_mirror_120` physical maintenance -> `bm_rel242`
     abstract relational dynamics, ZERO `bm_rel242` labels used in source OR target!):
     - Raw & double-projected `perp VAL2`: accuracy, `both_mirrors_right`, setting-clustered sign-flip
       Holm-adjusted p-value across the 5 rival targets.
  3. On-Valence vs. Off-Valence Decomposition (explicitly reporting ALL four readouts):
     - `SENT -> bm_rel242`, raw `HOUSE -> bm_rel242`, `HOUSE perp VAL2 -> bm_rel242`, `2F perp VAL2`.
  4. Output-Probability (`ΔNLL`) Plausibility Control:
     - Standalone `ΔNLL > 0` accuracy and `both_mirrors_right` on all 6 MLX models,
     - Setting-clustered 2,000-draw bootstrap 95% CI on OLS intercept at `ΔNLL = 0` for `2F perp VAL2`
       and `bm120->bm_rel242 perp VAL2`.
  5. Lexical Baselines: Unigram BoW & Bigram BoW across the exact same 20 setting-disjoint splits.
"""
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path("./local_scratch")
EXP_DIR = SCRATCH_DIR / "exp240_242"
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPO_ARCH = Path("./bench/archetypes")
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", "./bench/archetypes/fluent/plato"))
DEPTH_DIR = PLATO_CACHE / "depth240"

DATA_JSON = EXP_DIR / "bm_rel242_data.json"
OUT_JSON = EXP_DIR / "bm_rel242_results.json"

for p in (str(REPO_ARCH), str(REPO_FLUENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_swap236 as BS236  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402

MLX_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)


def _word_bag(text):
    return Counter(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def merge_and_verify():
    rows = []
    for part in (1, 2):
        p = EXP_DIR / f"bm_rel242_part{part}.json"
        assert p.exists(), f"Missing {p}"
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("bm_rel_120") or data.get("pairs") or data.get("items")
        rows.extend(data)

    rows.sort(key=lambda r: (str(r["setting_id"]), 0 if r["order"] == "orig" else 1))
    assert len(rows) == 120, f"Expected 120 rows, got {len(rows)}"

    bm238_settings = {r["setting"].strip().lower() for r in BM238.load_mirror_data()}
    by_sid = {}
    bad = []
    for r in rows:
        sid = str(r["setting_id"])
        assert r["order"] in ("orig", "mirror"), f"bad order {r}"
        assert r["fav_pos"] in ("first", "second"), f"bad fav_pos {r}"
        p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
        if p_txt == n_txt or _word_bag(p_txt) != _word_bag(n_txt):
            bad.append((r["id"], "bow_mismatch"))
        if r["setting"].strip().lower() in bm238_settings:
            bad.append((r["id"], "setting_overlap_with_bm238"))
        by_sid.setdefault(sid, []).append(r)

    assert not bad, f"Verification failures: {bad[:10]}"
    assert len(by_sid) == 60, f"Expected 60 unique settings, got {len(by_sid)}"
    n_first, n_second = 0, 0
    for sid, grp in by_sid.items():
        assert len(grp) == 2, f"Setting {sid} count={len(grp)}"
        assert {x["order"] for x in grp} == {"orig", "mirror"}, f"Setting {sid} orders"
        assert {x["fav_pos"] for x in grp} == {"first", "second"}, f"Setting {sid} fav_pos"
        n_first += sum(1 for x in grp if x["fav_pos"] == "first")
        n_second += sum(1 for x in grp if x["fav_pos"] == "second")
    assert n_first == 60 and n_second == 60
    DATA_JSON.write_text(json.dumps({"bm_rel_120": rows}, indent=2))
    print(f"MERGED & VERIFIED {DATA_JSON}: 120/120 exact BoW, 60 settings, 60/60 order balance, 0 overlap with bm238", flush=True)
    return rows


def extract_5depths(keys=MLX_KEYS):
    """Extract hidden states at all 5 depths + exact per-sentence NLL on all 6 MLX models."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    rows = json.loads(DATA_JSON.read_text())["bm_rel_120"]
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    for key in keys:
        out_npz = DEPTH_DIR / f"bmrel242_{key}.npz"
        if out_npz.exists():
            print(f"BMREL242 EXTRACT {key}: already exists, skipping", flush=True)
            continue
        spec = PD.MODEL_SPECS[key]
        mx.set_cache_limit(2 << 30)
        t0 = time.time()
        print(f"BMREL242 EXTRACT {key}: loading {spec['path']} ...", flush=True)
        model, tok = load(spec["path"])
        layers = PD._find_layers(model)
        nl_cnt = len(layers)
        layer_idxs = [max(1, min(nl_cnt - 2, int(round(df * (nl_cnt - 1))))) for df in DEPTH_FRACS]
        blk2slot = {id(layers[li]): slot for slot, li in enumerate(layer_idxs)}

        rec_buf = {}
        patched = {}
        for blk in layers:
            cls = type(blk)
            if cls not in patched:
                orig = cls.__call__
                patched[cls] = orig

                def make_hook(orig_fn):
                    def hooked(self, *a, **k):
                        out = orig_fn(self, *a, **k)
                        slot = blk2slot.get(id(self))
                        if slot is not None:
                            h = out[0] if isinstance(out, tuple) else out
                            rec_buf[slot] = h[0]
                        return out
                    return hooked

                cls.__call__ = make_hook(orig)

        n_d = len(DEPTH_FRACS)
        pl = [[] for _ in range(n_d)]
        pm = [[] for _ in range(n_d)]
        nl = [[] for _ in range(n_d)]
        nm = [[] for _ in range(n_d)]
        p_nll_m, n_nll_m, p_nll_s, n_nll_s = [], [], [], []
        try:
            for r in rows:
                for side in ("pos", "neg"):
                    ids = PD._text_ids_mlx(tok, r[side].strip())
                    assert 2 <= len(ids) <= 512
                    rec_buf.clear()
                    x = mx.array([ids])
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    assert len(rec_buf) == n_d
                    ce = nn.losses.cross_entropy(logits[0, :-1, :].astype(mx.float32), x[0, 1:], reduction="none")
                    nll_m = mx.mean(ce)
                    nll_s = mx.sum(ce)
                    eval_list = [nll_m, nll_s]
                    slot_vecs = []
                    for s in range(n_d):
                        h = rec_buf[s].astype(mx.float32)
                        hl = h[-1]
                        hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
                        slot_vecs.append((hl, hm))
                        eval_list.extend([hl, hm])
                    mx.eval(*eval_list)
                    rec_buf.clear()
                    if side == "pos":
                        p_nll_m.append(float(nll_m.item()))
                        p_nll_s.append(float(nll_s.item()))
                        for s, (hl, hm) in enumerate(slot_vecs):
                            pl[s].append(np.array(hl, dtype=np.float32))
                            pm[s].append(np.array(hm, dtype=np.float32))
                    else:
                        n_nll_m.append(float(nll_m.item()))
                        n_nll_s.append(float(nll_s.item()))
                        for s, (hl, hm) in enumerate(slot_vecs):
                            nl[s].append(np.array(hl, dtype=np.float32))
                            nm[s].append(np.array(hm, dtype=np.float32))
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            rec_buf.clear()
            del layers, model, tok
            gc.collect()
            mx.clear_cache()

        pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
        PD._atomic_savez(
            out_npz,
            pos_last=pack(np.stack([np.stack(x, 0) for x in pl], 0)),
            pos_mean=pack(np.stack([np.stack(x, 0) for x in pm], 0)),
            neg_last=pack(np.stack([np.stack(x, 0) for x in nl], 0)),
            neg_mean=pack(np.stack([np.stack(x, 0) for x in nm], 0)),
            pos_nll_mean=np.array(p_nll_m, dtype=np.float32),
            neg_nll_mean=np.array(n_nll_m, dtype=np.float32),
            pos_nll_sum=np.array(p_nll_s, dtype=np.float32),
            neg_nll_sum=np.array(n_nll_s, dtype=np.float32),
            layer_idxs=np.array(layer_idxs, dtype=np.int32),
            num_layers=np.int32(nl_cnt),
            depth_fracs=np.array(DEPTH_FRACS, dtype=np.float32),
        )
        print(f"BMREL242_EXTRACT_DONE {key}: layers={layer_idxs}/{nl_cnt} in {time.time() - t0:.1f}s", flush=True)


def _holm_adjust(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (tk, pv) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * pv))
        adj[tk] = round(running, 4)
    return adj


def _fit_neutral_rot_at_slot(slot, pool_key, d_plato=256):
    """Fit 4-MLX neutral-text GPA consensus & Procrustes rotations at depth slot (strictly neutral gut+proc+chat)."""
    # Use 4 consensus MLX models for C4 consensus; phi4_mini and olmo3_7b mapped strictly via neutral Procrustes to C4!
    c4_keys = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
    pca_params, Z_tr = {}, {}
    for k in MLX_KEYS:
        d = np.load(DEPTH_DIR / f"depth240_{k}.npz", allow_pickle=True)
        X = d[f"neut_{pool_key}"][slot].astype(np.float64)
        mu, V, rms, z_norm = AR.fit_pca_params(X, d_plato)
        pca_params[k] = (mu, V, rms)
        Z_tr[k] = z_norm
    # 10-iteration GPA on C4
    C = Z_tr[c4_keys[0]].copy()
    for _ in range(10):
        aligned = [PD.apply_procrustes_map(Z_tr[k], PD.fit_procrustes_map(Z_tr[k], C)) for k in c4_keys]
        C = np.mean(aligned, axis=0)
        C = C / max(1e-12, float(np.sqrt(np.mean(C ** 2))))
    to_c4 = {k: PD.fit_procrustes_map(Z_tr[k], C) for k in MLX_KEYS}
    return {"pca": pca_params, "to_c4": to_c4}


def _map_mat(X, rot, key):
    mu, V, rms = rot["pca"][key]
    return PD.apply_procrustes_map(AR.apply_pca_params(X.astype(np.float64), mu, V, rms), rot["to_c4"][key])


def analyze_frozen():
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = json.loads(DATA_JSON.read_text())["bm_rel_120"]
    sids_str = [str(r["setting_id"]) for r in rows]
    uniq_sids = sorted(set(sids_str))
    sid2i = {s: i for i, s in enumerate(uniq_sids)}
    sid_idx = np.array([sid2i[s] for s in sids_str], dtype=int)
    is_first = np.array([r["fav_pos"] == "first" for r in rows], dtype=bool)
    is_second = ~is_first

    splits = []
    for s_i in range(20):
        rng = np.random.default_rng(24200 + s_i)
        perm = rng.permutation(len(uniq_sids))
        f0 = set(int(x) for x in perm[:30])
        splits.append(np.array([0 if int(s) in f0 else 1 for s in sid_idx], dtype=int))

    bow_uni = BS236._bow_2fold_pairs(rows, "pos", "neg", BS236._words, splits)
    bow_bi = BS236._bow_2fold_pairs(rows, "pos", "neg", BS236._bigrams, splits)

    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    sg_rows, ov_rows, _ = V234.load_valence_data()
    n_sg = len(sg_rows)

    # Plausibility ΔNLL baseline
    dnll = {}
    plaus = {}
    for k in MLX_KEYS:
        dr = np.load(DEPTH_DIR / f"bmrel242_{k}.npz")
        dm = dr["neg_nll_mean"].astype(np.float64) - dr["pos_nll_mean"].astype(np.float64)
        dnll[k] = dm
        bm_r, bm_n, _ = BM238._both_mirrors_right(dm > 0, sid_idx)
        plaus[k] = {
            "acc_120": round(float(np.mean(dm > 0)), 4),
            "first_acc": round(float(np.mean(dm[is_first] > 0)), 4),
            "second_acc": round(float(np.mean(dm[is_second] > 0)), 4),
            "both_mirrors": round(bm_r, 4),
            "both_mirrors_n": int(bm_n),
        }

    # Evaluate across all 5 depths (mean_pool & last_token) for BOTH Gemma 4 source and Qwen 3.5 source!
    by_depth = {}
    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        by_depth[dkey] = {}
        for pool_key in ("mean", "last"):
            rot = _fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
            Z_rel, Z_bm120, Q2, u_house, u_sent = {}, {}, {}, {}, {}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                dr = np.load(DEPTH_DIR / f"bmrel242_{k}.npz")
                Z_rel[k] = _map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - _map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
                Z_bm120[k] = _map_mat(d240[f"bm120_pos_{pool_key}"][slot], rot, k) - _map_mat(d240[f"bm120_neg_{pool_key}"][slot], rot, k)
                Dh = _map_mat(d240[f"house_pos_{pool_key}"][slot][house_idx], rot, k) - _map_mat(d240[f"house_neg_{pool_key}"][slot][house_idx], rot, k)
                Dv = _map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - _map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
                u_house[k] = AR.house_dir(Dh, house_cats)
                u_sent[k] = AR.unit(Dv[:n_sg].mean(0))
                u_ov = AR.unit(Dv[n_sg:].mean(0))
                Q2[k] = V234._orthonormal_basis([u_sent[k], u_ov])

            pool_out = {}
            for src_k in ("gemma4_e4b", "qwen35_9b"):
                tgts = [k for k in MLX_KEYS if k != src_k]
                Ds = Z_rel[src_k]
                u_bm120_s = AR.unit(Z_bm120[src_k].mean(0))
                u_bm120_p2_s = V234._project_out(u_bm120_s, Q2[src_k])
                uh_p2_s = V234._project_out(u_house[src_k], Q2[src_k])
                src_res = {}
                for tk in tgts:
                    Dt = Z_rel[tk]
                    Dt2 = V234._project_out_rows(Dt, Q2[tk])

                    # 1. On-valence & off-valence zero-shot readouts
                    sc_sent = Dt @ u_sent[src_k]
                    _, p_sent = BM238._setting_clustered_signflip_p(u_sent[src_k], Dt, sid_idx, seed=2421)
                    bm_sent, _, _ = BM238._both_mirrors_right(sc_sent > 0, sid_idx)

                    sc_hr = Dt @ u_house[src_k]
                    _, p_hr = BM238._setting_clustered_signflip_p(u_house[src_k], Dt, sid_idx, seed=2422)
                    bm_hr, _, _ = BM238._both_mirrors_right(sc_hr > 0, sid_idx)

                    uh2 = V234._project_out(uh_p2_s, Q2[tk])
                    sc_hp2 = Dt2 @ uh2
                    c_hp2, p_hp2 = BM238._setting_clustered_signflip_p(uh2, Dt2, sid_idx, seed=2423)
                    bm_hp2, _, _ = BM238._both_mirrors_right(sc_hp2 > 0, sid_idx)

                    # 2. Strict Cross-Domain Zero-Shot: bm120 (physical maintenance) -> bm_rel242 (abstract relational)!
                    ub_r = u_bm120_s
                    sc_xdom_r = Dt @ ub_r
                    _, p_xdom_r = BM238._setting_clustered_signflip_p(ub_r, Dt, sid_idx, seed=2424)
                    bm_xdom_r, _, _ = BM238._both_mirrors_right(sc_xdom_r > 0, sid_idx)

                    ub2 = V234._project_out(u_bm120_p2_s, Q2[tk])
                    sc_xdom_p2 = Dt2 @ ub2
                    c_xdom_p2, p_xdom_p2 = BM238._setting_clustered_signflip_p(ub2, Dt2, sid_idx, seed=2425)
                    bm_xdom_p2, _, _ = BM238._both_mirrors_right(sc_xdom_p2 > 0, sid_idx)

                    # 3. Setting-disjoint 2-fold CV on bm_rel242 (20 splits)
                    p2_accs, p2_boths, p2_f, p2_s, raw_accs = [], [], [], [], []
                    oof_p2_s0 = np.zeros(len(rows), dtype=np.float64)
                    for si, fl in enumerate(splits):
                        oof_r = np.zeros(len(rows), dtype=np.float64)
                        oof_2 = np.zeros(len(rows), dtype=np.float64)
                        for f in (0, 1):
                            tr, te = (fl != f), (fl == f)
                            ut = AR.unit(Ds[tr].mean(0))
                            ut2 = V234._project_out(V234._project_out(ut, Q2[src_k]), Q2[tk])
                            oof_r[te] = Dt[te] @ ut
                            oof_2[te] = Dt2[te] @ ut2
                        if si == 0:
                            oof_p2_s0 = oof_2.copy()
                        raw_accs.append(float(np.mean(oof_r > 0)))
                        p2_accs.append(float(np.mean(oof_2 > 0)))
                        p2_f.append(float(np.mean(oof_2[is_first] > 0)))
                        p2_s.append(float(np.mean(oof_2[is_second] > 0)))
                        br, _, _ = BM238._both_mirrors_right(oof_2 > 0, sid_idx)
                        p2_boths.append(br)

                    # Split-0 setting-clustered p-value
                    sp0 = splits[0]
                    fp2, fc2 = [], []
                    for f in (0, 1):
                        tr, te = (sp0 != f), (sp0 == f)
                        ut2 = V234._project_out(V234._project_out(AR.unit(Ds[tr].mean(0)), Q2[src_k]), Q2[tk])
                        c2, p2 = BM238._setting_clustered_signflip_p(ut2, Dt2[te], sid_idx[te], seed=2426 + f)
                        fc2.append(c2); fp2.append(p2)

                    ols_cv2 = BM238._setting_clustered_ols_intercept_ci(oof_p2_s0, dnll[tk], sid_idx, seed=2428)
                    ols_xdom = BM238._setting_clustered_ols_intercept_ci(sc_xdom_p2, dnll[tk], sid_idx, seed=2429)

                    src_res[tk] = {
                        "zs_sent_120": round(float(np.mean(sc_sent > 0)), 4),
                        "zs_sent_both": round(bm_sent, 4),
                        "zs_sent_p": round(p_sent, 4),
                        "zs_house_raw_120": round(float(np.mean(sc_hr > 0)), 4),
                        "zs_house_raw_both": round(bm_hr, 4),
                        "zs_house_raw_p": round(p_hr, 4),
                        "zs_house_p2_120": round(float(np.mean(sc_hp2 > 0)), 4),
                        "zs_house_p2_both": round(bm_hp2, 4),
                        "zs_house_p2_cos": round(c_hp2, 4),
                        "zs_house_p2_p": round(p_hp2, 4),
                        "xdom_bm120_raw_120": round(float(np.mean(sc_xdom_r > 0)), 4),
                        "xdom_bm120_raw_both": round(bm_xdom_r, 4),
                        "xdom_bm120_raw_p": round(p_xdom_r, 4),
                        "xdom_bm120_p2_120": round(float(np.mean(sc_xdom_p2 > 0)), 4),
                        "xdom_bm120_p2_both": round(bm_xdom_p2, 4),
                        "xdom_bm120_p2_cos": round(c_xdom_p2, 4),
                        "xdom_bm120_p2_p": round(p_xdom_p2, 4),
                        "cv2_raw_120": round(float(np.mean(raw_accs)), 4),
                        "cv2_p2_120": round(float(np.mean(p2_accs)), 4),
                        "cv2_p2_first": round(float(np.mean(p2_f)), 4),
                        "cv2_p2_second": round(float(np.mean(p2_s)), 4),
                        "cv2_p2_both": round(float(np.mean(p2_boths)), 4),
                        "cv2_p2_cos": round(float(np.mean(fc2)), 4),
                        "cv2_p2_p_max": round(float(max(fp2)), 4),
                        "ols_cv2_p2_vs_dnll": ols_cv2,
                        "ols_xdom_p2_vs_dnll": ols_xdom,
                    }

                holm = {
                    "zs_house_p2_holm": _holm_adjust({tk: src_res[tk]["zs_house_p2_p"] for tk in tgts}),
                    "xdom_bm120_p2_holm": _holm_adjust({tk: src_res[tk]["xdom_bm120_p2_p"] for tk in tgts}),
                    "cv2_p2_holm": _holm_adjust({tk: src_res[tk]["cv2_p2_p_max"] for tk in tgts}),
                }
                pool_out[src_k] = {"targets": src_res, "holm": holm}
            by_depth[dkey][pool_key] = pool_out

    report = {
        "frozen_script_sha256": script_sha,
        "lexical_baselines": {
            "bow_unigram_20split_mean": round(float(np.mean(bow_uni)), 4),
            "bow_bigram_20split_mean": round(float(np.mean(bow_bi)), 4),
        },
        "output_prob_nll_baseline": plaus,
        "by_depth": by_depth,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} (script sha256={script_sha[:16]}...)", flush=True)
    return report


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "merge":
        merge_and_verify()
    elif cmd == "extract":
        keys = tuple(sys.argv[2:]) if len(sys.argv) > 2 else MLX_KEYS
        extract_5depths(keys)
    elif cmd == "analyze":
        analyze_frozen()
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
