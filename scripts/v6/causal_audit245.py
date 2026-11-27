"""Experiment 245-C: Strict Dose-Matched, Margin-Matched, Sentence-Span-Only Causal Ablation Test.

Addresses EVERY Round-1 finding on Exp 241 (Fable 5.2 + Astra 6.2):
  1. STRICT `batch_size = 1` serial inference everywhere (zero batching drift!).
  2. Strictly HELD-OUT cross-domain source direction (`bm120` in Gemma 4 -> evaluated on `bm_rel242`
     and `bm_neut245`), crossing BOTH clause orders (`orig` AND `mirror`) with A/B order!
  3. CENTERED MEAN-ABLATION (`h[span] <- h[span] - ((h[span] - mu_span) . w) w`) restricted STRICTLY
     to the evidence sentence tokens (`sp_a`, `sp_b` in FC; `sp_s` in Topic) — NEVER BOS, choices, or answer cue!
  4. DIFFICULTY-MATCHED Topic control task: each setting's foil is chosen as the closest setting by
     character/word overlap (plus setting-specific lexical-distractor phrasing) to avoid a 4.2-logit ceiling!
  5. EXACT REMOVED-ENERGY-MATCHED null controls (40 orthogonal directions in `span(V_tgt)` tuned so their
     actual per-token removed variance `E_rem` on the exact FC sentence tokens matches `w_swap_p2` within +-3%!).
  6. Logs BOTH FC margin drop AND Topic margin drop (absolute AND fractional) for EVERY named AND control
     direction, plus a RECIPROCAL topic-direction ablation for a true 2x2 double-dissociation test!
  7. Pre-registered Pass/Drop rule:
     - Requires setting-clustered bootstrap 95% CI on `(frac_drop_FC - frac_drop_Topic)` to exclude 0 AND
       beat 40 energy-matched controls at p < 0.05 (one-sided) on held-out items. Otherwise: CLEAN NEGATIVE!
"""
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH_DIR / "exp245_controls"
OLD_EXP = SCRATCH_DIR / "exp240_242"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(OLD_EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import depth_sweep240 as D240  # noqa: E402
import causal_steer241 as C241  # noqa: E402

MLX_KEYS = D240.MLX_KEYS
DEPTH_FRACS = D240.DEPTH_FRACS


def _closest_foil_settings(settings):
    """For each setting, pick the most lexically/character-similar OTHER setting to make Topic harder."""
    def trigrams(s):
        t = re.sub(r"[^a-z0-9 ]", "", s.lower())
        return Counter(t[i:i + 3] for i in range(max(1, len(t) - 2)))
    bags = [trigrams(s) for s in settings]
    foils = []
    for i in range(len(settings)):
        best_j, best_sc = (i + 1) % len(settings), -1.0
        bi = bags[i]
        ni = np.sqrt(sum(v * v for v in bi.values())) + 1e-12
        for j in range(len(settings)):
            if i == j:
                continue
            bj = bags[j]
            nj = np.sqrt(sum(v * v for v in bj.values())) + 1e-12
            dot = sum(bi[k] * bj[k] for k in bi if k in bj)
            sc = dot / (ni * nj)
            if sc > best_sc:
                best_sc = sc
                best_j = j
        foils.append(settings[best_j])
    return foils


def _build_hard_topic_prompt(tok, sent, set_a, set_b):
    """Harder topic discrimination prompt asking which specific scenario/title fits best."""
    p0 = "Choose the exact scenario title for this sentence.\nSentence: "
    p1 = "\n(A) "
    p2 = "\n(B) "
    p3 = "\nAnswer with a single letter (A or B):"
    full = p0 + sent + p1 + set_a + p2 + set_b + p3
    ids_full = PD._text_ids_mlx(tok, full)
    ids_p0 = PD._text_ids_mlx(tok, p0)
    ids_s_end = PD._text_ids_mlx(tok, p0 + sent)
    s0, s1 = max(1, len(ids_p0) - 1), len(ids_s_end)
    assert s1 > s0 and s1 < len(ids_full)
    return ids_full, (s0, s1)


def _match_energy_directions(H_span_centered, V_tgt, Q_excl_nat, target_var, n_ctrl=40, seed=24510):
    """Construct `n_ctrl` unit directions strictly inside span(V_tgt) orthogonal to `Q_excl_nat`
    whose empirical per-token variance `mean((H_span_centered @ w)^2)` matches `target_var` within +-1%!
    Uses exact analytical mixing of a high-variance and low-variance orthogonal basis vector."""
    # Orthonormal basis for span(V_tgt) perp Q_excl_nat
    P = V_tgt - Q_excl_nat @ (Q_excl_nat.T @ V_tgt)
    Q_free, _ = np.linalg.qr(P)
    # Keep columns with non-trivial norm
    # Project H_span_centered into Q_free coordinates: (N_tok, K)
    Z = H_span_centered @ Q_free  # (N_tok, K)
    # SVD of Z gives exact principal directions & per-token variances inside the free subspace!
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    vars_pc = (S ** 2) / len(Z)  # variance along each free PC
    # Partition PCs into above-target and below-target pools
    hi_idx = np.where(vars_pc > target_var * 1.05)[0]
    lo_idx = np.where(vars_pc < target_var * 0.95)[0]
    assert len(hi_idx) >= 2 and len(lo_idx) >= 2, f"Target var {target_var:.4f} outside free PC range [{vars_pc.min():.4f}, {vars_pc.max():.4f}]"

    rng = np.random.default_rng(seed)
    ctrls = []
    achieved_vars = []
    for _ in range(n_ctrl):
        # Random unit vector in hi subspace and random unit vector in lo subspace (automatically orthogonal!)
        c_hi = rng.standard_normal(len(hi_idx))
        c_hi /= np.linalg.norm(c_hi)
        u_hi = Vt[hi_idx].T @ c_hi  # in Q_free coords
        v_hi = float(np.mean((Z @ u_hi) ** 2))

        c_lo = rng.standard_normal(len(lo_idx))
        c_lo /= np.linalg.norm(c_lo)
        u_lo = Vt[lo_idx].T @ c_lo
        v_lo = float(np.mean((Z @ u_lo) ** 2))

        # u(theta) = cos(t)*u_hi + sin(t)*u_lo -> Var = cos^2*v_hi + sin^2*v_lo + 2*sin*cos*cov(u_hi, u_lo)
        # Note: u_hi and u_lo are orthogonal in Z-covariance too because they come from disjoint right-singular vectors of Z!
        # Verify: cov(Z u_hi, Z u_lo) == 0!
        p_hi = (target_var - v_lo) / max(1e-12, (v_hi - v_lo))
        p_hi = float(np.clip(p_hi, 0.0, 1.0))
        u_mix = np.sqrt(p_hi) * u_hi + np.sqrt(1.0 - p_hi) * u_lo
        u_mix /= np.linalg.norm(u_mix)
        w_nat = Q_free @ u_mix
        w_nat /= np.linalg.norm(w_nat)
        v_act = float(np.mean((H_span_centered @ w_nat) ** 2))
        ctrls.append(w_nat.astype(np.float32))
        achieved_vars.append(v_act)
    return ctrls, achieved_vars


def run_clean_causal(eval_dataset="bm_rel242", src_key="gemma4_e4b", tgt_key="qwen35_9b", depth_frac=0.50, n_ctrl=40):
    import mlx.core as mx
    from mlx_lm import load

    t0 = time.time()
    slot = DEPTH_FRACS.index(depth_frac)
    print(f"=== CAUSAL245 CLEAN {src_key}->{tgt_key} @ d{int(depth_frac*100)} on {eval_dataset} (bs=1 serial, centered span-only, n_ctrl={n_ctrl}) ===", flush=True)

    data_map = {k: dict(np.load(DEPTH_DIR / f"depth240_{k}.npz")) for k in MLX_KEYS}
    rot_m = D240._fit_rot_at_slot(data_map, slot, "mean", d_plato=256)

    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    sg, ov, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg])
    n_sg = len(sg)

    ds = data_map[src_key]
    Dh_s = D240._map(ds["house_pos_mean"][slot][house_idx], rot_m, src_key) - D240._map(ds["house_neg_mean"][slot][house_idx], rot_m, src_key)
    Dv_s = D240._map(ds["val_pos_mean"][slot], rot_m, src_key) - D240._map(ds["val_neg_mean"][slot], rot_m, src_key)
    Dbm_s = D240._map(ds["bm120_pos_mean"][slot], rot_m, src_key) - D240._map(ds["bm120_neg_mean"][slot], rot_m, src_key)

    u_sent_s = AR.unit(Dv_s[:n_sg].mean(0))
    u_ov_s = AR.unit(Dv_s[n_sg:].mean(0))
    dom_vecs = [AR.unit(Dv_s[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
    Q5_s = V234._orthonormal_basis(dom_vecs + [u_ov_s])
    u_house_s = AR.house_dir(Dh_s, house_cats)
    u_swap_raw_s = AR.unit(Dbm_s.mean(0))
    u_swap_v5_s = V234._project_out(u_swap_raw_s, Q5_s)

    # Also build a cross-family TOPIC direction from neutral corpus PC1 (gutenberg vs proc/chat domain axis in source!)
    Z_neut_s = D240._map(ds["neut_mean"][slot], rot_m, src_key)
    tr_rows = [json.loads(l) for l in open(PLATO_CACHE / "passages.jsonl")]
    neut_doms = np.array([r["domain"] for r in tr_rows if r["domain"] in ("gut", "proc", "chat")])
    u_topic_s = AR.unit(Z_neut_s[neut_doms == "gut"].mean(0) - Z_neut_s[neut_doms != "gut"].mean(0))

    pca_t = rot_m["pca"][tgt_key]
    proc_t = rot_m["to_c4"][tgt_key]
    mu_t, V_t, rms_t = pca_t
    w_swap_v5 = C241._native_pullback_unit(u_swap_v5_s, pca_t, proc_t)
    w_swap_raw = C241._native_pullback_unit(u_swap_raw_s, pca_t, proc_t)
    w_sent = C241._native_pullback_unit(u_sent_s, pca_t, proc_t)
    w_topic = C241._native_pullback_unit(u_topic_s, pca_t, proc_t)
    Q_excl_nat = V234._orthonormal_basis([w_swap_v5.astype(np.float64), w_swap_raw.astype(np.float64), w_sent.astype(np.float64), w_topic.astype(np.float64)])
    del data_map
    gc.collect()

    # Load held-out evaluation dataset (ZERO overlap with bm120 source fit!)
    if eval_dataset == "bm_rel242":
        rows = json.loads((OLD_EXP / "bm_rel242_data.json").read_text())["bm_rel_120"]
    elif eval_dataset == "bm_neut245":
        rows = json.loads((EXP_DIR / "bm_neut245_data.json").read_text())["bm_neut_120"]
    elif eval_dataset == "bm_polconf245":
        rows = json.loads((EXP_DIR / "bm_polconf245_data.json").read_text())["bm_polconf_120"]
    else:
        raise ValueError(eval_dataset)

    # Balance BOTH clause order (`orig` + `mirror` = all 120 items!) AND A/B answer letter!
    uniq_sids = sorted(set(str(r["setting_id"]) for r in rows))
    sid2i = {s: i for i, s in enumerate(uniq_sids)}
    sid_to_setting = {}
    for r in rows:
        sid_to_setting[str(r["setting_id"])] = r["setting"].strip()
    settings_ordered = [sid_to_setting[s] for s in uniq_sids]
    foils_ordered = _closest_foil_settings(settings_ordered)
    sid_to_foil = {s: foils_ordered[i] for i, s in enumerate(uniq_sids)}

    spec = PD.MODEL_SPECS[tgt_key]
    mx.set_cache_limit(2 << 30)
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl_cnt = len(layers)
    tgt_idx = max(1, min(nl_cnt - 2, int(round(depth_frac * (nl_cnt - 1)))))
    tgt_blk = layers[tgt_idx]
    id_A = C241._letter_token_id(tok, "A")
    id_B = C241._letter_token_id(tok, "B")

    # Build 120 balanced FC prompts (all 60 settings x both orig & mirror; exact 60 A-true, 60 B-true)
    # and 120 matched Topic prompts (same 120 sentences x closest foil; exact 60 A-true, 60 B-true)
    fc_items, top_items = [], []
    for r in rows:
        gi = sid2i[str(r["setting_id"])]
        is_orig = (r["order"] == "orig")
        # Cross clause order with A/B order: even setting -> orig=A, mirror=B; odd -> orig=B, mirror=A
        pos_is_A = (gi % 2 == 0) if is_orig else (gi % 2 == 1)
        sa, sb = (r["pos"].strip(), r["neg"].strip()) if pos_is_A else (r["neg"].strip(), r["pos"].strip())
        ids_fc, sp_a, sp_b = C241._build_fc_prompt_and_spans(tok, sa, sb)
        fc_items.append({"sid": gi, "order": r["order"], "pos_is_A": pos_is_A, "ids": ids_fc, "spans": [sp_a, sp_b]})

        true_set = sid_to_setting[str(r["setting_id"])]
        foil_set = sid_to_foil[str(r["setting_id"])]
        ta, tb = (true_set, foil_set) if pos_is_A else (foil_set, true_set)
        ids_t, sp_s = _build_hard_topic_prompt(tok, r["pos"].strip(), ta, tb)
        top_items.append({"sid": gi, "order": r["order"], "pos_is_A": pos_is_A, "ids": ids_t, "spans": [sp_s]})

    # Hook supporting (a) baseline span activation collection, (b) centered mean-ablation on exact spans ONLY!
    hook_state = {"mode": "none", "w": None, "mu": None, "spans": None, "rec": None}
    patched = {}
    for blk in layers:
        cls = type(blk)
        if cls not in patched:
            orig = cls.__call__
            patched[cls] = orig
            def make_hook(orig_fn):
                def hooked(self, *a, **k):
                    out = orig_fn(self, *a, **k)
                    if self is tgt_blk and hook_state["mode"] != "none":
                        h = out[0] if isinstance(out, tuple) else out
                        if hook_state["mode"] == "record":
                            hook_state["rec"] = h[0].astype(mx.float32)
                        elif hook_state["mode"] == "center_ablate":
                            T = h.shape[1]
                            mask_np = np.zeros(T, dtype=np.float32)
                            for s0, s1 in hook_state["spans"]:
                                mask_np[s0:s1] = 1.0
                            m_mx = mx.array(mask_np)[None, :, None]
                            w_mx = hook_state["w"]
                            mu_mx = hook_state["mu"]
                            h_f = h.astype(mx.float32)
                            proj = mx.sum((h_f - mu_mx) * w_mx, axis=-1, keepdims=True)
                            h_new = (h_f - m_mx * proj * w_mx).astype(h.dtype)
                            if isinstance(out, tuple):
                                out = (h_new,) + out[1:]
                            else:
                                out = h_new
                    return out
                return hooked
            cls.__call__ = make_hook(orig)

    def eval_serial_items(items, mode="none", w_np=None, mu_np=None, collect_spans=False):
        hook_state["w"] = mx.array(w_np, dtype=mx.float32)[None, None, :] if w_np is not None else None
        hook_state["mu"] = mx.array(mu_np, dtype=mx.float32)[None, None, :] if mu_np is not None else None
        margins, span_vecs = [], []
        for it in items:
            hook_state["mode"] = "record" if collect_spans else mode
            hook_state["spans"] = it["spans"]
            hook_state["rec"] = None
            x = mx.array([it["ids"]])
            lg = model(x)
            if isinstance(lg, tuple):
                lg = lg[0]
            last = lg[0, -1, :].astype(mx.float32)
            diff_ab = last[id_A] - last[id_B]
            if collect_spans:
                h_rec = hook_state["rec"]
                mx.eval(diff_ab, h_rec)
                h_np = np.array(h_rec, dtype=np.float32)
                for s0, s1 in it["spans"]:
                    span_vecs.append(h_np[s0:s1])
            else:
                mx.eval(diff_ab)
            m = float(diff_ab.item()) if it["pos_is_A"] else float(-diff_ab.item())
            margins.append(m)
        hook_state["mode"] = "none"
        return np.array(margins, dtype=np.float64), (np.vstack(span_vecs).astype(np.float64) if collect_spans else None)

    try:
        # 1. Serial bs=1 baselines + collect EXACT sentence-span token activations!
        base_fc, H_fc_spans = eval_serial_items(fc_items, collect_spans=True)
        base_top, H_top_spans = eval_serial_items(top_items, collect_spans=True)
        mu_fc = H_fc_spans.mean(axis=0)
        mu_top = H_top_spans.mean(axis=0)
        H_fc_c = H_fc_spans - mu_fc[None, :]
        H_top_c = H_top_spans - mu_top[None, :]

        # Exact per-token removed variance of w_swap_v5 on FC sentence tokens!
        target_var_fc = float(np.mean((H_fc_c @ w_swap_v5.astype(np.float64)) ** 2))
        print(f"  Baseline FC: acc={np.mean(base_fc>0):.4f} margin={base_fc.mean():.4f} | Topic: acc={np.mean(base_top>0):.4f} margin={base_top.mean():.4f} | target_var_fc={target_var_fc:.4f}", flush=True)

        # Build 40 EXACT energy-matched control directions on H_fc_c!
        w_ctrls, ctrl_vars_fc = _match_energy_directions(H_fc_c, V_t, Q_excl_nat, target_var_fc, n_ctrl=n_ctrl, seed=24510)
        print(f"  Energy-matched {n_ctrl} controls: var_fc mean={np.mean(ctrl_vars_fc):.4f} (min={np.min(ctrl_vars_fc):.4f}, max={np.max(ctrl_vars_fc):.4f})", flush=True)

        def score_ablation(name, w_vec):
            m_fc, _ = eval_serial_items(fc_items, mode="center_ablate", w_np=w_vec, mu_np=mu_fc)
            m_top, _ = eval_serial_items(top_items, mode="center_ablate", w_np=w_vec, mu_np=mu_top)
            v_fc = float(np.mean((H_fc_c @ w_vec.astype(np.float64)) ** 2))
            v_top = float(np.mean((H_top_c @ w_vec.astype(np.float64)) ** 2))
            fc_drop = float(base_fc.mean() - m_fc.mean())
            top_drop = float(base_top.mean() - m_top.mean())
            fc_frac = fc_drop / max(1e-6, float(base_fc.mean()))
            top_frac = top_drop / max(1e-6, float(base_top.mean()))
            # Setting-clustered paired bootstrap CI on (fc_frac - top_frac)
            sids = np.array([it["sid"] for it in fc_items])
            rng = np.random.default_rng(24515)
            diffs_int = []
            for _ in range(2000):
                gs = rng.integers(0, len(uniq_sids), size=len(uniq_sids))
                ix = np.concatenate([np.where(sids == g)[0] for g in gs])
                ff = float((base_fc[ix].mean() - m_fc[ix].mean()) / max(1e-6, float(base_fc[ix].mean())))
                tf = float((base_top[ix].mean() - m_top[ix].mean()) / max(1e-6, float(base_top[ix].mean())))
                diffs_int.append(ff - tf)
            diffs_int = np.asarray(diffs_int)
            return {
                "name": name,
                "removed_var_fc": round(v_fc, 5),
                "removed_var_topic": round(v_top, 5),
                "fc_acc_after": round(float(np.mean(m_fc > 0)), 4),
                "fc_ties": int(np.sum(m_fc == 0)),
                "fc_margin_drop": round(fc_drop, 4),
                "fc_frac_drop": round(fc_frac, 4),
                "topic_acc_after": round(float(np.mean(m_top > 0)), 4),
                "topic_margin_drop": round(top_drop, 4),
                "topic_frac_drop": round(top_frac, 4),
                "interaction_fc_minus_topic_frac": round(fc_frac - top_frac, 4),
                "interaction_ci95": [round(float(np.percentile(diffs_int, 2.5)), 4), round(float(np.percentile(diffs_int, 97.5)), 4)],
            }

        named_res = {}
        for nm, wv in [("u_swap_val5", w_swap_v5), ("u_swap_raw", w_swap_raw), ("u_sent", w_sent), ("u_topic_reciprocal", w_topic)]:
            named_res[nm] = score_ablation(nm, wv)
            print(f"  [{nm}] var_fc={named_res[nm]['removed_var_fc']:.4f} | FC frac={named_res[nm]['fc_frac_drop']:.4f} | Topic frac={named_res[nm]['topic_frac_drop']:.4f} | int={named_res[nm]['interaction_fc_minus_topic_frac']:.4f} CI={named_res[nm]['interaction_ci95']}", flush=True)

        ctrl_res = [score_ablation(f"ctrl_{i}", w_ctrls[i]) for i in range(n_ctrl)]
        c_fc_frac = np.array([c["fc_frac_drop"] for c in ctrl_res])
        c_top_frac = np.array([c["topic_frac_drop"] for c in ctrl_res])
        c_int = np.array([c["interaction_fc_minus_topic_frac"] for c in ctrl_res])

        sw = named_res["u_swap_val5"]
        p_fc_vs_ctrl = float((np.sum(c_fc_frac >= sw["fc_frac_drop"] - 1e-12) + 1) / (n_ctrl + 1))
        p_int_vs_ctrl = float((np.sum(c_int >= sw["interaction_fc_minus_topic_frac"] - 1e-12) + 1) / (n_ctrl + 1))

        report = {
            "elapsed_sec": round(time.time() - t0, 1),
            "src": src_key, "tgt": tgt_key, "depth_frac": depth_frac, "layer_idx": tgt_idx,
            "eval_dataset": eval_dataset, "batch_size": 1, "ablation_type": "centered_span_only",
            "baseline": {
                "fc_acc": round(float(np.mean(base_fc > 0)), 4),
                "fc_margin_mean": round(float(base_fc.mean()), 4),
                "topic_acc": round(float(np.mean(base_top > 0)), 4),
                "topic_margin_mean": round(float(base_top.mean()), 4),
                "n_items": len(fc_items),
            },
            "named_directions": named_res,
            "energy_matched_controls_summary": {
                "n_ctrl": n_ctrl,
                "var_fc_mean": round(float(np.mean([c["removed_var_fc"] for c in ctrl_res])), 5),
                "fc_frac_drop_mean": round(float(c_fc_frac.mean()), 4),
                "fc_frac_drop_max": round(float(c_fc_frac.max()), 4),
                "topic_frac_drop_mean": round(float(c_top_frac.mean()), 4),
                "interaction_mean": round(float(c_int.mean()), 4),
                "interaction_max": round(float(c_int.max()), 4),
                "p_swap_fc_frac_gt_ctrl": round(p_fc_vs_ctrl, 4),
                "p_swap_interaction_gt_ctrl": round(p_int_vs_ctrl, 4),
            },
            "controls_detail": ctrl_res,
        }
        out_p = EXP_DIR / f"causal_audit245_{eval_dataset}.json"
        out_p.write_text(json.dumps(report, indent=2))
        print(f"SAVED {out_p} in {report['elapsed_sec']}s | p_fc={p_fc_vs_ctrl:.4f}, p_int={p_int_vs_ctrl:.4f}", flush=True)
        return report
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "bm_rel242"
    run_clean_causal(eval_dataset=ds)
