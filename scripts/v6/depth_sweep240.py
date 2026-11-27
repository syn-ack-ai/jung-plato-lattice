"""Experiment 240: 5-Depth Layer Sweep (15%, 30%, 50%, 70%, 85%) Across All 6 Open MLX Families.

Hooks ALL FIVE relative depths (0.15, 0.30, 0.50, 0.70, 0.85) in a SINGLE forward pass per model across:
  - Google Gemma 4 E4B (multimodal, 42 layers -> L6, L12, L20, L29, L35)
  - Alibaba Qwen3.5 9B (32 layers -> L5, L9, L16, L22, L26)
  - Meta Llama 3.1 8B (32 layers -> L5, L9, L16, L22, L26)
  - Mistral 7B v0.3 (32 layers -> L5, L9, L16, L22, L26)
  - Held-out Microsoft Phi-4-mini (32 layers -> L5, L9, L16, L22, L26)
  - Held-out AllenAI OLMo 3 7B (32 layers -> L5, L9, L16, L22, L26)

Directly tests Nordfors (2026, arXiv:2611.07234) quantitatively with open neutral-text Procrustes:
  1. Matched-depth (d -> d) curves at 15%, 30%, 50%, 70%, 85% for:
     - Held-out neutral C4 -> Phi-4 / OLMo 3 Frobenius R^2 (200 held_gut + proc_held)
     - Zero-shot HOUSE (raw & perp VAL2)
     - Mirrored identical-BoW compositional binding bm120 (mean & last, 2F perp VAL2, HOUSE perp VAL2, SENT, raw HOUSE)
     - Role taxonomies: Jung 8, Jobs 8, Family 8, Real Historical Myths 500 (ATU-disjoint & West->NonWest)
     - Held-out basis test Q_jung7 vs Q_jobs7 vs Q_fam7 on 500 Real Historical Myths across all 5 depths
     - Zero-PN items3 narrative register gap on Phi-4 & OLMo 3 across all 5 depths
  2. Full 5x5 Cross-Depth Procrustes Transfer Matrices (d_src x d_tgt) testing whether multimodal
     Gemma 4 peaks at earlier layers (15-30%) while text-only models (Qwen3.5, Llama 3.1, Mistral,
     Phi-4, OLMo 3) peak deeper (50-70%) on compositional binding (bm120 perp VAL2) and neutral R^2!

Usage:
  python depth_sweep240.py extract [KEY...]
  python depth_sweep240.py analyze
"""
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path("./local_scratch")
FOLK_DIR = SCRATCH_DIR / "folklore_test"
EXP_DIR = SCRATCH_DIR / "exp240_242"
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPO_ARCH = Path("./bench/archetypes")
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", "./bench/archetypes/fluent/plato"))
DEPTH_DIR = PLATO_CACHE / "depth240"
DEPTH_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = EXP_DIR / "depth_sweep240_results.json"

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(FOLK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402

MLX_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)


def _load_all_texts():
    tr_rows = [json.loads(l) for l in open(PLATO_CACHE / "passages.jsonl")]
    neut_rows = [r for r in tr_rows if r["domain"] in ("gut", "proc", "chat")]
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    house_pairs = AR.load_arch_pairs()
    sg, ov, _ = V234.load_valence_data()
    val_pairs = sg + ov
    bm_rows = BM238.load_mirror_data()
    j_rows = NA235.load_named_arch_data()
    n_rows = NJ237.load_nonjung_data()
    f_rows = json.loads(R239.FAM_JSON.read_text())
    a_rows = json.loads(R239.AFT_JSON.read_text())
    return {
        "neut": [r["text"].strip() for r in neut_rows],
        "ev": [r["text"].strip() for r in ev_rows],
        "house_pos": [p["done"].strip() for p in house_pairs],
        "house_neg": [p["neglected"].strip() for p in house_pairs],
        "val_pos": [p["pos"].strip() for p in val_pairs],
        "val_neg": [p["neg"].strip() for p in val_pairs],
        "bm120_pos": [r["pos"].strip() for r in bm_rows],
        "bm120_neg": [r["neg"].strip() for r in bm_rows],
        "jung": [r["text"].strip() for r in j_rows],
        "jobs": [r["text"].strip() for r in n_rows],
        "fam": [r["text"].strip() for r in f_rows],
        "aft": [r["text"].strip() for r in a_rows],
    }


def extract_model_5depths(key, corpus=None):
    out_npz = DEPTH_DIR / f"depth240_{key}.npz"
    if out_npz.exists():
        print(f"DEPTH240 EXTRACT {key}: already exists, skipping", flush=True)
        return
    if corpus is None:
        corpus = _load_all_texts()

    import mlx.core as mx
    from mlx_lm import load

    spec = PD.MODEL_SPECS[key]
    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"DEPTH240 EXTRACT {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl_cnt = len(layers)
    layer_idxs = [max(1, min(nl_cnt - 2, int(round(df * (nl_cnt - 1))))) for df in DEPTH_FRACS]
    blk2slot = {id(layers[li]): slot for slot, li in enumerate(layer_idxs)}
    n_d = len(DEPTH_FRACS)
    print(f"DEPTH240 EXTRACT {key}: L={nl_cnt}, hooking layers {layer_idxs} ({[round(i/(nl_cnt-1), 2) for i in layer_idxs]})", flush=True)

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

    need_last = {"neut", "house_pos", "house_neg", "val_pos", "val_neg", "bm120_pos", "bm120_neg"}
    payload = {
        "layer_idxs": np.array(layer_idxs, dtype=np.int32),
        "num_layers": np.int32(nl_cnt),
        "depth_fracs": np.array(DEPTH_FRACS, dtype=np.float32),
    }
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)

    try:
        for name, texts in corpus.items():
            t1 = time.time()
            do_last = name in need_last
            means = [[] for _ in range(n_d)]
            lasts = [[] for _ in range(n_d)] if do_last else None
            for i, txt in enumerate(texts):
                ids = PD._text_ids_mlx(tok, txt)
                assert 2 <= len(ids) <= 768, f"{key}/{name}[{i}] bad tok len {len(ids)}"
                rec_buf.clear()
                _ = model(mx.array([ids]))
                assert len(rec_buf) == n_d
                eval_list = []
                slot_vecs = []
                for s in range(n_d):
                    h = rec_buf[s].astype(mx.float32)
                    hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
                    if do_last:
                        hl = h[-1]
                        slot_vecs.append((hm, hl))
                        eval_list.extend([hm, hl])
                    else:
                        slot_vecs.append((hm, None))
                        eval_list.append(hm)
                mx.eval(*eval_list)
                rec_buf.clear()
                for s, (hm, hl) in enumerate(slot_vecs):
                    means[s].append(np.array(hm, dtype=np.float32))
                    if do_last:
                        lasts[s].append(np.array(hl, dtype=np.float32))
            payload[f"{name}_mean"] = pack(np.stack([np.stack(x, 0) for x in means], 0))
            if do_last:
                payload[f"{name}_last"] = pack(np.stack([np.stack(x, 0) for x in lasts], 0))
            print(f"  {key} {name:10s} ({len(texts):4d}): {time.time() - t1:.1f}s", flush=True)
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, model, tok
        gc.collect()
        mx.clear_cache()

    PD._atomic_savez(out_npz, **payload)
    print(f"DEPTH240_EXTRACT_DONE {key} in {time.time() - t0:.1f}s ({out_npz.stat().st_size / 1e6:.1f} MB)", flush=True)


def _fit_rot_at_slot(data_map, slot, pool_key, d_plato=256):
    pca_params, Z_tr = {}, {}
    for k in MLX_KEYS:
        X = data_map[k][f"neut_{pool_key}"][slot].astype(np.float64)
        mu, V, rms, z_norm = AR.fit_pca_params(X, d_plato)
        pca_params[k] = (mu, V, rms)
        Z_tr[k] = z_norm
    C = Z_tr[C4_KEYS[0]].copy()
    for _ in range(10):
        aligned = [PD.apply_procrustes_map(Z_tr[k], PD.fit_procrustes_map(Z_tr[k], C)) for k in C4_KEYS]
        C = np.mean(aligned, axis=0)
        C = C / max(1e-12, float(np.sqrt(np.mean(C ** 2))))
    to_c4 = {k: PD.fit_procrustes_map(Z_tr[k], C) for k in MLX_KEYS}
    return {"pca": pca_params, "to_c4": to_c4, "Z_neut_c4": C}


def _map(X, rot, k):
    mu, V, rms = rot["pca"][k]
    return PD.apply_procrustes_map(AR.apply_pca_params(X.astype(np.float64), mu, V, rms), rot["to_c4"][k])


def _frob_r2(Y_true, Y_pred):
    Yc = Y_true - Y_true.mean(0, keepdims=True)
    Pc = Y_pred - Y_pred.mean(0, keepdims=True)
    return float(1.0 - np.sum((Yc - Pc) ** 2) / max(1e-12, np.sum(Yc ** 2)))


def analyze_all():
    t0 = time.time()
    print("Loading 5-depth caches for all 6 MLX models ...", flush=True)
    data_map = {k: dict(np.load(DEPTH_DIR / f"depth240_{k}.npz")) for k in MLX_KEYS}

    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    ev_neut_mask = np.array([r["domain"] in ("held_gut", "proc_held") for r in ev_rows], dtype=bool)
    ev_m_no_pn = np.array([r["domain"] == "items3_myth" and r["id"].endswith("_anchor") and not NJ237._has_proper_name(r["text"]) for r in ev_rows])
    ev_c_no_pn = np.array([r["domain"] == "items3_mech" and r["id"].endswith("_anchor") and not NJ237._has_proper_name(r["text"]) for r in ev_rows])

    house_pairs = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(house_pairs) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([house_pairs[i]["cat"] for i in house_idx])
    sg, ov, _ = V234.load_valence_data()
    n_sg = len(sg)

    bm_rows = BM238.load_mirror_data()
    uniq_bs = sorted({r["setting_id"] for r in bm_rows})
    bs2i = {s: i for i, s in enumerate(uniq_bs)}
    bm_sid = np.array([bs2i[r["setting_id"]] for r in bm_rows])
    bm_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23800 + s_i)
        f0 = set(int(x) for x in rng.permutation(len(uniq_bs))[:30])
        bm_splits.append(np.array([0 if int(s) in f0 else 1 for s in bm_sid], dtype=int))

    # Role labels & splits
    j_rows = NA235.load_named_arch_data()
    y_j = np.array([NA235.ARCHETYPES.index(r["archetype"]) for r in j_rows])
    sids_j = np.array([int(r["setting_id"]) for r in j_rows])
    n_rows = NJ237.load_nonjung_data()
    y_n = np.array([NJ237.ROLES.index(r["role"]) for r in n_rows])
    sids_n = np.array([int(r["setting_id"]) for r in n_rows])
    f_rows = json.loads(R239.FAM_JSON.read_text())
    y_f = np.array([R239.FAM_ROLES.index(r["role"]) for r in f_rows])
    sids_f = np.array([int(r["setting_id"]) for r in f_rows])
    a_rows = json.loads(R239.AFT_JSON.read_text())
    y_a = np.array([R239.ATU_DIVS.index(r["division"]) for r in a_rows])
    nw_mask = np.array([bool(r["non_western"]) for r in a_rows])
    atu_ids = [str(r["atu_id"]) for r in a_rows]

    role_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23500 + s_i)
        f0 = set(int(x) for x in rng.permutation(range(1, 41))[:20])
        role_splits.append(np.array([0 if int(s) in f0 else 1 for s in sids_j], dtype=int))

    # ATU-disjoint splits for real myths
    aft_atu_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(239200 + s_i)
        fl = np.zeros(len(y_a), dtype=int)
        for c in range(10):
            ix_c = np.where(y_a == c)[0]
            by_atu = {}
            for i in ix_c:
                by_atu.setdefault(atu_ids[i], []).append(i)
            keys_a = list(by_atu.keys())
            rng.shuffle(keys_a)
            keys_a.sort(key=lambda k: len(by_atu[k]), reverse=True)
            c0, c1 = 0, 0
            for ka in keys_a:
                tgt_f = 0 if c0 <= c1 else 1
                for i in by_atu[ka]:
                    fl[i] = tgt_f
                if tgt_f == 0:
                    c0 += len(by_atu[ka])
                else:
                    c1 += len(by_atu[ka])
        aft_atu_splits.append(fl)

    # Stratified 2-fold splits for HOUSE
    house_splits = []
    for s_i in range(10):
        rng = np.random.default_rng(22900 + s_i)
        fl = np.zeros(len(house_idx), dtype=int)
        for cat in AR.HOUSE_CATS:
            ix = rng.permutation(np.where(house_cats == cat)[0])
            fl[ix[len(ix) // 2:]] = 1
        house_splits.append(fl)

    layer_table = {k: {"num_layers": int(data_map[k]["num_layers"]), "layers": data_map[k]["layer_idxs"].tolist()} for k in MLX_KEYS}

    # Precompute PCA at every (model, slot) for cross-depth tests too!
    matched_depth = {}
    pca_neut_mean = {}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        print(f"Analyzing matched depth {dkey} ({df:.0%}) ...", flush=True)
        rot_m = _fit_rot_at_slot(data_map, slot, "mean", d_plato=256)
        rot_l = _fit_rot_at_slot(data_map, slot, "last", d_plato=256)
        for k in MLX_KEYS:
            pca_neut_mean[(k, slot)] = rot_m["pca"][k]

        # 1. Held-out neutral Frobenius R^2 (C4 -> Phi-4 / OLMo 3) & zero-PN items3 register gap
        Z_ev = {k: _map(data_map[k]["ev_mean"][slot], rot_m, k) for k in MLX_KEYS}
        C4_ev = np.mean([Z_ev[k] for k in C4_KEYS], axis=0)
        neut_r2, items3_reg = {}, {}
        for hk in ("phi4_mini", "olmo3_7b"):
            # Predict held-out target directly from neutral-text-rotated C4 (ZERO eval calibration!)
            neut_r2[f"c4_to_{hk}_zero_cal"] = round(_frob_r2(Z_ev[hk][ev_neut_mask], C4_ev[ev_neut_mask]), 4)
            # Single-teacher comparison (Gemma 4 & Qwen 3.5 -> held-out)
            neut_r2[f"gemma_to_{hk}_zero_cal"] = round(_frob_r2(Z_ev[hk][ev_neut_mask], Z_ev["gemma4_e4b"][ev_neut_mask]), 4)
            neut_r2[f"qwen_to_{hk}_zero_cal"] = round(_frob_r2(Z_ev[hk][ev_neut_mask], Z_ev["qwen35_9b"][ev_neut_mask]), 4)
            # Per-passage cosine for zero-PN items3_myth vs items3_mech anchor
            Zc = Z_ev[hk] - Z_ev[hk].mean(0, keepdims=True)
            Pc = C4_ev - C4_ev.mean(0, keepdims=True)
            p_cos = np.sum(Zc * Pc, 1) / np.maximum(1e-12, np.linalg.norm(Zc, 1) * np.linalg.norm(Pc, 1))
            items3_reg[hk] = {
                "myth_no_pn": round(float(p_cos[ev_m_no_pn].mean()), 4),
                "mech_no_pn": round(float(p_cos[ev_c_no_pn].mean()), 4),
                "diff": round(float(p_cos[ev_m_no_pn].mean() - p_cos[ev_c_no_pn].mean()), 4),
            }

        # 2. HOUSE, VAL234, bm120 (mean & last)
        Dh_m, Q2_m, Q3_m, uh_m, us_m, Zbm_m = {}, {}, {}, {}, {}, {}
        Dh_l, Q2_l, uh_l, us_l, Zbm_l = {}, {}, {}, {}, {}
        for k in MLX_KEYS:
            d = data_map[k]
            Dh_m[k] = _map(d["house_pos_mean"][slot][house_idx], rot_m, k) - _map(d["house_neg_mean"][slot][house_idx], rot_m, k)
            Dv_m = _map(d["val_pos_mean"][slot], rot_m, k) - _map(d["val_neg_mean"][slot], rot_m, k)
            Zbm_m[k] = _map(d["bm120_pos_mean"][slot], rot_m, k) - _map(d["bm120_neg_mean"][slot], rot_m, k)
            uh_m[k] = AR.house_dir(Dh_m[k], house_cats)
            us_m[k] = AR.unit(Dv_m[:n_sg].mean(0))
            uo_m = AR.unit(Dv_m[n_sg:].mean(0))
            Q2_m[k] = V234._orthonormal_basis([us_m[k], uo_m])
            Q3_m[k] = V234._orthonormal_basis([us_m[k], uo_m, uh_m[k]])

            Dh_l[k] = _map(d["house_pos_last"][slot][house_idx], rot_l, k) - _map(d["house_neg_last"][slot][house_idx], rot_l, k)
            Dv_l = _map(d["val_pos_last"][slot], rot_l, k) - _map(d["val_neg_last"][slot], rot_l, k)
            Zbm_l[k] = _map(d["bm120_pos_last"][slot], rot_l, k) - _map(d["bm120_neg_last"][slot], rot_l, k)
            uh_l[k] = AR.house_dir(Dh_l[k], house_cats)
            us_l[k] = AR.unit(Dv_l[:n_sg].mean(0))
            Q2_l[k] = V234._orthonormal_basis([us_l[k], AR.unit(Dv_l[n_sg:].mean(0))])

        # Valence fraction of HOUSE in C4
        uh_c4 = AR.unit(np.mean([uh_m[k] for k in C4_KEYS], 0))
        us_c4 = AR.unit(np.mean([us_m[k] for k in C4_KEYS], 0))
        uo_c4 = AR.unit(np.mean([Q2_m[k][:, 1] for k in C4_KEYS], 0))
        Q2_c4 = V234._orthonormal_basis([us_c4, uo_c4])
        Q3_c4 = V234._orthonormal_basis([us_c4, uo_c4, uh_c4])
        house_val2_share = round(float(np.sum((Q2_c4.T @ uh_c4) ** 2)), 4)

        # Evaluate bm120 & HOUSE for BOTH Gemma 4 source and Qwen 3.5 source!
        bm_out = {}
        house_out = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            # HOUSE 2-fold transfer (raw & perp VAL2)
            h_raw_all, h_p2_all = {}, {}
            for tk in tgts:
                ar_l, ap_l = [], []
                for fl in house_splits:
                    ok_r, ok_p = [], []
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        ut = AR.house_dir(Dh_m[src_k][tr], house_cats[tr])
                        ut2 = V234._project_out(V234._project_out(ut, Q2_m[src_k]), Q2_m[tk])
                        Dt2 = V234._project_out_rows(Dh_m[tk][te], Q2_m[tk])
                        ok_r.extend((Dh_m[tk][te] @ ut > 0).tolist())
                        ok_p.extend((Dt2 @ ut2 > 0).tolist())
                    ar_l.append(float(np.mean(ok_r)))
                    ap_l.append(float(np.mean(ok_p)))
                h_raw_all[tk] = round(float(np.mean(ar_l)), 4)
                h_p2_all[tk] = round(float(np.mean(ap_l)), 4)
            house_out[src_k] = {
                "mean_raw": round(float(np.mean(list(h_raw_all.values()))), 4),
                "mean_p2": round(float(np.mean(list(h_p2_all.values()))), 4),
                "per_target_raw": h_raw_all,
                "per_target_p2": h_p2_all,
            }

            # bm120 mean & last
            bm_out[src_k] = {}
            for mname, Zb, Qd, uhd, usd in [
                ("mean", Zbm_m, Q2_m, uh_m, us_m),
                ("last", Zbm_l, Q2_l, uh_l, us_l),
            ]:
                per_t = {}
                for tk in tgts:
                    Dt = Zb[tk]
                    Dt2 = V234._project_out_rows(Dt, Qd[tk])
                    uh2 = V234._project_out(V234._project_out(uhd[src_k], Qd[src_k]), Qd[tk])
                    sc_hp2 = Dt2 @ uh2
                    bm_hp2, _, _ = BM238._both_mirrors_right(sc_hp2 > 0, bm_sid)
                    sc_s = Dt @ usd[src_k]
                    sc_hr = Dt @ uhd[src_k]

                    p2_accs, p2_boths = [], []
                    for fl in bm_splits:
                        oof = np.zeros(len(bm_rows))
                        for f in (0, 1):
                            tr, te = (fl != f), (fl == f)
                            ut = AR.unit(Zb[src_k][tr].mean(0))
                            ut2 = V234._project_out(V234._project_out(ut, Qd[src_k]), Qd[tk])
                            oof[te] = Dt2[te] @ ut2
                        p2_accs.append(float(np.mean(oof > 0)))
                        br, _, _ = BM238._both_mirrors_right(oof > 0, bm_sid)
                        p2_boths.append(br)
                    per_t[tk] = {
                        "cv2_p2": round(float(np.mean(p2_accs)), 4),
                        "cv2_p2_both": round(float(np.mean(p2_boths)), 4),
                        "house_p2": round(float(np.mean(sc_hp2 > 0)), 4),
                        "house_p2_both": round(bm_hp2, 4),
                        "house_raw": round(float(np.mean(sc_hr > 0)), 4),
                        "sent": round(float(np.mean(sc_s > 0)), 4),
                    }
                bm_out[src_k][mname] = {
                    "mean_cv2_p2": round(float(np.mean([v["cv2_p2"] for v in per_t.values()])), 4),
                    "mean_cv2_p2_both": round(float(np.mean([v["cv2_p2_both"] for v in per_t.values()])), 4),
                    "mean_house_p2": round(float(np.mean([v["house_p2"] for v in per_t.values()])), 4),
                    "mean_house_raw": round(float(np.mean([v["house_raw"] for v in per_t.values()])), 4),
                    "mean_sent": round(float(np.mean([v["sent"] for v in per_t.values()])), 4),
                    "per_target": per_t,
                }

        # 3. Role spaces (Jung 8, Jobs 8, Family 8, Real Myths ATU-disjoint & West->NonWest) + Basis Test on Real Myths!
        Zj = {k: _map(data_map[k]["jung_mean"][slot], rot_m, k) for k in MLX_KEYS}
        Zn = {k: _map(data_map[k]["jobs_mean"][slot], rot_m, k) for k in MLX_KEYS}
        Zf = {k: _map(data_map[k]["fam_mean"][slot], rot_m, k) for k in MLX_KEYS}
        Za = {k: _map(data_map[k]["aft_mean"][slot], rot_m, k) for k in MLX_KEYS}

        roles_out = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            j_t1, n_t1, f_t1, a_atu_t1, a_atu_mauc, a_xc_t1 = [], [], [], [], [], []
            for tk in tgts:
                j_t1.append(R239._transfer_multiclass_splits(Zj[src_k], Zj[tk], y_j, role_splits, 8, Q3_m[src_k], Q3_m[tk])["top1_mean"])
                n_t1.append(R239._transfer_multiclass_splits(Zn[src_k], Zn[tk], y_n, role_splits, 8, Q3_m[src_k], Q3_m[tk])["top1_mean"])
                f_t1.append(R239._transfer_multiclass_splits(Zf[src_k], Zf[tk], y_f, role_splits, 8, Q3_m[src_k], Q3_m[tk])["top1_mean"])
                ra = R239._transfer_multiclass_splits(Za[src_k], Za[tk], y_a, aft_atu_splits, 10, Q3_m[src_k], Q3_m[tk])
                a_atu_t1.append(ra["top1_mean"])
                a_atu_mauc.append(ra["macro_auroc_mean"])
                # West -> NonWest
                tr_w, te_nw = ~nw_mask, nw_mask
                Zs_p = V234._project_out_rows(Za[src_k][tr_w] - Za[src_k][tr_w].mean(0, keepdims=True), Q3_m[src_k])
                Zt_p = V234._project_out_rows(Za[tk][te_nw] - Za[tk][tr_w].mean(0, keepdims=True), Q3_m[tk])
                Up = np.stack([V234._project_out(V234._project_out(u, Q3_m[src_k]), Q3_m[tk]) for u in NA235._ovr_dirs(Zs_p, y_a[tr_w], 10)], 0)
                a_xc_t1.append(float(np.mean(np.argmax(Zt_p @ Up.T, 1) == y_a[te_nw])))
            roles_out[src_k] = {
                "jung8_p3_top1_mean": round(float(np.mean(j_t1)), 4),
                "jobs8_p3_top1_mean": round(float(np.mean(n_t1)), 4),
                "fam8_p3_top1_mean": round(float(np.mean(f_t1)), 4),
                "aft_atu_disj_p3_top1_mean": round(float(np.mean(a_atu_t1)), 4),
                "aft_atu_disj_p3_mauc_mean": round(float(np.mean(a_atu_mauc)), 4),
                "aft_west2nonwest_p3_top1_mean": round(float(np.mean(a_xc_t1)), 4),
            }

        # Held-out basis capture Q_jung7 vs Q_jobs7 vs Q_fam7 on 500 Real Myths in C4 (perp VAL3)
        Zj_c4 = V234._project_out_rows(np.mean([Zj[k] for k in C4_KEYS], 0), Q3_c4)
        Zn_c4 = V234._project_out_rows(np.mean([Zn[k] for k in C4_KEYS], 0), Q3_c4)
        Zf_c4 = V234._project_out_rows(np.mean([Zf[k] for k in C4_KEYS], 0), Q3_c4)
        Za_c4 = V234._project_out_rows(np.mean([Za[k] for k in C4_KEYS], 0), Q3_c4)
        Qj = R239._centroid_basis_k(Zj_c4, y_j, 8, 7)
        Qn = R239._centroid_basis_k(Zn_c4, y_n, 8, 7)
        Qf = R239._centroid_basis_k(Zf_c4, y_f, 8, 7)
        ej = R239._eval_basis_capture(Qj, Za_c4, y_a, 10)
        en = R239._eval_basis_capture(Qn, Za_c4, y_a, 10)
        ef = R239._eval_basis_capture(Qf, Za_c4, y_a, 10)
        nj_sh = R239._shuffled_null_basis_capture(Zj_c4, y_j, sids_j, Za_c4, y_a, 10, n_draws=200, seed=2400 + slot)
        nn_sh = R239._shuffled_null_basis_capture(Zn_c4, y_n, sids_n, Za_c4, y_a, 10, n_draws=200, seed=2410 + slot)
        nf_sh = R239._shuffled_null_basis_capture(Zf_c4, y_f, sids_f, Za_c4, y_a, 10, n_draws=200, seed=2420 + slot)
        basis_aft = {
            "jung7_v_dir": round(ej["v_dir"], 4), "jung7_null": nj_sh["v_dir_null_mean"],
            "jung7_excess": round(ej["v_dir"] - nj_sh["v_dir_null_mean"], 4),
            "jung7_p": round(float((sum(x >= ej["v_dir"] for x in nj_sh["_raw_v_dir"]) + 1) / 201), 4),
            "fam7_v_dir": round(ef["v_dir"], 4), "fam7_null": nf_sh["v_dir_null_mean"],
            "fam7_excess": round(ef["v_dir"] - nf_sh["v_dir_null_mean"], 4),
            "fam7_p": round(float((sum(x >= ef["v_dir"] for x in nf_sh["_raw_v_dir"]) + 1) / 201), 4),
            "jobs7_v_dir": round(en["v_dir"], 4), "jobs7_null": nn_sh["v_dir_null_mean"],
            "jobs7_excess": round(en["v_dir"] - nn_sh["v_dir_null_mean"], 4),
            "jobs7_p": round(float((sum(x >= en["v_dir"] for x in nn_sh["_raw_v_dir"]) + 1) / 201), 4),
        }

        matched_depth[dkey] = {
            "depth_frac": df,
            "neutral_r2": neut_r2,
            "items3_zero_pn_register": items3_reg,
            "house_val2_share_c4": house_val2_share,
            "house_2fold": house_out,
            "bm120": bm_out,
            "roles_perp_val3": roles_out,
            "basis_on_real_myths_c4_p3": basis_aft,
        }

    # =========================================================================
    # 5x5 CROSS-DEPTH PROCRUSTES MATRICES (testing Nordfors Gemma-4 early-shift!)
    # =========================================================================
    print("Computing 5x5 Cross-Depth Procrustes Transfer Matrices ...", flush=True)
    # Precompute PCA-normalized neutral, ev_neut, val, bm120 for every (model, slot)
    Z_pca = {}
    for k in MLX_KEYS:
        d = data_map[k]
        for s in range(len(DEPTH_FRACS)):
            mu, V, rms = pca_neut_mean[(k, s)]
            zn = AR.apply_pca_params(d["neut_mean"][s].astype(np.float64), mu, V, rms)
            ze = AR.apply_pca_params(d["ev_mean"][s][ev_neut_mask].astype(np.float64), mu, V, rms)
            dv = AR.apply_pca_params(d["val_pos_mean"][s].astype(np.float64), mu, V, rms) - AR.apply_pca_params(d["val_neg_mean"][s].astype(np.float64), mu, V, rms)
            db = AR.apply_pca_params(d["bm120_pos_mean"][s].astype(np.float64), mu, V, rms) - AR.apply_pca_params(d["bm120_neg_mean"][s].astype(np.float64), mu, V, rms)
            q2 = V234._orthonormal_basis([AR.unit(dv[:n_sg].mean(0)), AR.unit(dv[n_sg:].mean(0))])
            Z_pca[(k, s)] = (zn, ze, db, q2)

    cross_depth = {}
    pairs_to_test = [
        ("gemma4_e4b", "qwen35_9b"),
        ("gemma4_e4b", "llama31_8b"),
        ("gemma4_e4b", "olmo3_7b"),
        ("qwen35_9b", "llama31_8b"),
        ("qwen35_9b", "olmo3_7b"),
        ("qwen35_9b", "gemma4_e4b"),
        ("llama31_8b", "olmo3_7b"),
    ]
    sp0 = bm_splits[0]
    for sk, tk in pairs_to_test:
        mat_r2 = np.zeros((5, 5))
        mat_bm = np.zeros((5, 5))
        for si in range(5):
            zn_s, ze_s, db_s, q2_s = Z_pca[(sk, si)]
            for ti in range(5):
                zn_t, ze_t, db_t, q2_t = Z_pca[(tk, ti)]
                # Direct pairwise neutral-text Procrustes rotation src(si) -> tgt(ti)!
                p_st = PD.fit_procrustes_map(zn_s, zn_t)
                pred_e = PD.apply_procrustes_map(ze_s, p_st)
                mat_r2[si, ti] = _frob_r2(ze_t, pred_e)

                # Rotate src bm120 & q2 into tgt(ti) space (apply_procrustes_map uses @ R.T!)
                _, _, R, _ = p_st
                Ds_rot = db_s @ R.T
                Q2s_rot = V234._orthonormal_basis([AR.unit(q2_s[:, 0] @ R.T), AR.unit(q2_s[:, 1] @ R.T)])
                Dt2 = V234._project_out_rows(db_t, q2_t)
                accs = []
                for fl in bm_splits[:10]:
                    oof = np.zeros(len(bm_rows))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        ut = AR.unit(Ds_rot[tr].mean(0))
                        ut2 = V234._project_out(V234._project_out(ut, Q2s_rot), q2_t)
                        oof[te] = Dt2[te] @ ut2
                    accs.append(float(np.mean(oof > 0)))
                mat_bm[si, ti] = float(np.mean(accs))
        cross_depth[f"{sk}__to__{tk}"] = {
            "depths": [int(round(df * 100)) for df in DEPTH_FRACS],
            "neutral_r2_5x5": [[round(float(x), 4) for x in row] for row in mat_r2],
            "bm120_p2_5x5": [[round(float(x), 4) for x in row] for row in mat_bm],
            "best_r2_pair": [int(round(DEPTH_FRACS[i] * 100)) for i in np.unravel_index(np.argmax(mat_r2), (5, 5))],
            "best_bm120_pair": [int(round(DEPTH_FRACS[i] * 100)) for i in np.unravel_index(np.argmax(mat_bm), (5, 5))],
            "best_bm120_val": round(float(np.max(mat_bm)), 4),
        }

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "layer_table": layer_table,
        "matched_depth": matched_depth,
        "cross_depth_5x5": cross_depth,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} in {report['elapsed_sec']}s", flush=True)
    return report


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "extract":
        keys = tuple(sys.argv[2:]) if len(sys.argv) > 2 else MLX_KEYS
        corpus = _load_all_texts()
        for k in keys:
            extract_model_5depths(k, corpus=corpus)
    elif cmd == "analyze":
        analyze_all()
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
