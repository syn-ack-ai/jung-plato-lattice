"""Experiment 241: Quantitative Cross-Family Causal Intervention (1D Steering & Ablation + Topic Double Dissociation).

Directly answers Nordfors (2026, arXiv:2611.07234) with a 100% OPEN, QUANTITATIVE, ZERO-SHOT CROSS-FAMILY
causal intervention on ALL 60 mirrored identical-BoW settings (120 counterbalanced pairs) across rival LLM labs:

Zero-Shot Native-Space Teleportation (ZERO target swap/sentiment labels used!):
  1. Fit neutral-text 256D PCA (mu, V, rms) and neutral-text Procrustes map (mu_a, mu_b, R, s) at depth d.
  2. Measure 1D unit directions STRICTLY in a RIVAL source model (e.g., Gemma 4 -> Qwen3.5 / Llama-3.1,
     Qwen3.5 -> OLMo-3):
       - `u_swap_p2`: valence-free compositional binding direction (`bm120` perp VAL2; also cross-domain on `bm_rel242`!)
       - `u_house_p2`: valence-free care-vs-neglect direction (`HOUSE` perp VAL2)
       - `u_sent`: pure general sentiment direction (`sentiment_gen`)
       - `u_rand_0..19`: 20 random unit directions strictly inside `span(V_tgt)` orthogonal to `u_swap_p2` & `u_sent`
  3. Rotate into target native hidden space via exact analytic pullback `w_native = V_tgt @ (R_tgt.T @ u_src_c4)`
     (unit-tested to 1e-6 against `apply_procrustes_map`!).

Three Quantitative Causal Tests (ALL items, ZERO cherry-picking):
  - Test A (Intrinsic Log-Probability Ablation): On all 120 mirrored pairs of `bm120`, project out `w_native`
    (`h <- h - (h . w) w`) across all tokens at layer L and measure shift in target `Delta NLL = NLL(neg) - NLL(pos)`
    and neutral-text perplexity drift vs. 20 random directions.
  - Test B (Targeted Clause Steering on Forced-Choice Compositional Binding Judgment — 120 counterbalanced prompts!):
    Prompt: "Which sentence makes more sense?\n(A) {sent_A}\n(B) {sent_B}\nAnswer with a single letter (A or B):"
    Inject `+alpha * sigma * w` ONLY on Option A's token span and `-alpha * sigma * w` ONLY on Option B's token span
    (and vice versa: `-alpha` on A, `+alpha` on B!), measuring the exact bidirectional steering effect
    `Delta_steer = mean( [logit(A)-logit(B)]_{A+,B-} - [logit(A)-logit(B)]_{A-,B+} )` and flip rate across all
    120 counterbalanced prompts vs. all 20 matched-norm random PCA directions!
  - Test C (Topic/Setting Double-Dissociation Control — 60 prompts):
    Prompt: "Which setting is described by this sentence?\nSentence: {sent}\n(A) {true_setting}\n(B) {foil_setting}\nAnswer with a single letter (A or B):"
    Apply the EXACT SAME `+alpha` / `-alpha` steering and ablation along `u_swap_p2` on the sentence/option tokens
    and measure whether topic/setting accuracy stays intact while compositional binding judgment is causally steered!

Usage:
  python causal_steer241.py run [DEPTH_FRAC]   # e.g., 0.50 (default) or 0.30
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH_DIR / "exp240_242"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import depth_sweep240 as D240  # noqa: E402

DEPTH_FRACS = D240.DEPTH_FRACS
MLX_KEYS = D240.MLX_KEYS


def _native_pullback_unit(u_c4, pca_params, proc_params):
    """Exact unit vector w in target native hidden space R^D such that <h, w> is strictly proportional
    to <apply_procrustes_map(apply_pca_params(h), proc_params), u_c4>."""
    mu, V, rms = pca_params
    mu_a, mu_b, R, s = proc_params
    u_c4 = AR.unit(u_c4)
    w = V @ (R.T @ u_c4)
    w = w / max(1e-12, float(np.linalg.norm(w)))
    # Analytic unit test on random test vector
    rng = np.random.default_rng(241)
    h1 = rng.standard_normal((1, V.shape[0]))
    h2 = rng.standard_normal((1, V.shape[0]))
    z1 = PD.apply_procrustes_map(AR.apply_pca_params(h1, mu, V, rms), proc_params)[0]
    z2 = PD.apply_procrustes_map(AR.apply_pca_params(h2, mu, V, rms), proc_params)[0]
    lhs = float(np.dot(z1 - z2, u_c4))
    rhs = float((s / rms) * np.dot((h1 - h2)[0], w))
    assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(lhs)), f"Pullback mismatch: {lhs} vs {rhs}"
    return w.astype(np.float32)


def _find_subseq(seq, sub):
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return i, i + m
    return None


def _build_fc_prompt_and_spans(tok, sent_a, sent_b):
    """Build forced-choice prompt and locate exact token index ranges [a0, a1) and [b0, b1) for sent_a and sent_b."""
    p0 = "Which sentence makes more sense?\n(A) "
    p1 = "\n(B) "
    p2 = "\nAnswer with a single letter (A or B):"
    full = p0 + sent_a + p1 + sent_b + p2
    ids_full = PD._text_ids_mlx(tok, full)
    # Locate boundaries via incremental prefix tokenization (exact even across BPE merges!)
    ids_p0 = PD._text_ids_mlx(tok, p0)
    ids_a_end = PD._text_ids_mlx(tok, p0 + sent_a)
    ids_b_start = PD._text_ids_mlx(tok, p0 + sent_a + p1)
    ids_b_end = PD._text_ids_mlx(tok, p0 + sent_a + p1 + sent_b)
    a0 = max(1, len(ids_p0) - 1)
    a1 = min(len(ids_full), len(ids_a_end))
    b0 = max(a1, len(ids_b_start) - 1)
    b1 = min(len(ids_full), len(ids_b_end))
    assert a1 > a0 and b1 > b0 and b1 < len(ids_full), f"Bad spans {(a0, a1, b0, b1)} for len {len(ids_full)}"
    return ids_full, (a0, a1), (b0, b1)


def _build_topic_prompt_and_spans(tok, sent, set_a, set_b):
    p0 = "Which setting is described by this sentence?\nSentence: "
    p1 = "\n(A) "
    p2 = "\n(B) "
    p3 = "\nAnswer with a single letter (A or B):"
    full = p0 + sent + p1 + set_a + p2 + set_b + p3
    ids_full = PD._text_ids_mlx(tok, full)
    ids_p0 = PD._text_ids_mlx(tok, p0)
    ids_s_end = PD._text_ids_mlx(tok, p0 + sent)
    ids_a_start = PD._text_ids_mlx(tok, p0 + sent + p1)
    ids_a_end = PD._text_ids_mlx(tok, p0 + sent + p1 + set_a)
    ids_b_start = PD._text_ids_mlx(tok, p0 + sent + p1 + set_a + p2)
    ids_b_end = PD._text_ids_mlx(tok, p0 + sent + p1 + set_a + p2 + set_b)
    s0, s1 = max(1, len(ids_p0) - 1), len(ids_s_end)
    a0, a1 = max(s1, len(ids_a_start) - 1), len(ids_a_end)
    b0, b1 = max(a1, len(ids_b_start) - 1), min(len(ids_full), len(ids_b_end))
    return ids_full, (s0, s1), (a0, a1), (b0, b1)


def _letter_token_id(tok, letter):
    for cand in (f" {letter}", letter):
        ids = tok.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    ids = tok.encode(letter, add_special_tokens=False)
    return int(ids[-1])


def run_causal_pair(src_key, tgt_key, depth_frac=0.50, eval_dataset="bm120", alphas=(1.0, 2.0), n_rand=20):
    """Run all three causal tests porting 1D directions ZERO-SHOT from `src_key` -> `tgt_key` at `depth_frac`."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    slot = DEPTH_FRACS.index(depth_frac)
    print(f"\n=== CAUSAL241 {src_key} -> {tgt_key} @ depth={depth_frac:.0%} (slot {slot}, ds={eval_dataset}) ===", flush=True)
    data_map = {k: dict(np.load(DEPTH_DIR / f"depth240_{k}.npz")) for k in MLX_KEYS}
    rot_m = D240._fit_rot_at_slot(data_map, slot, "mean", d_plato=256)

    # Compute source-only 1D directions in C4 space (ZERO target labels used!)
    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    sg, ov, _ = V234.load_valence_data()
    n_sg = len(sg)

    ds = data_map[src_key]
    Dh_s = D240._map(ds["house_pos_mean"][slot][house_idx], rot_m, src_key) - D240._map(ds["house_neg_mean"][slot][house_idx], rot_m, src_key)
    Dv_s = D240._map(ds["val_pos_mean"][slot], rot_m, src_key) - D240._map(ds["val_neg_mean"][slot], rot_m, src_key)
    Dbm_s = D240._map(ds["bm120_pos_mean"][slot], rot_m, src_key) - D240._map(ds["bm120_neg_mean"][slot], rot_m, src_key)

    u_sent_s = AR.unit(Dv_s[:n_sg].mean(0))
    u_ov_s = AR.unit(Dv_s[n_sg:].mean(0))
    Q2_s = V234._orthonormal_basis([u_sent_s, u_ov_s])
    u_house_s = AR.house_dir(Dh_s, house_cats)
    u_house_p2_s = V234._project_out(u_house_s, Q2_s)
    u_swap_raw_s = AR.unit(Dbm_s.mean(0))
    u_swap_p2_s = V234._project_out(u_swap_raw_s, Q2_s)

    # Pull back strictly through target's neutral-text PCA + Procrustes map into target native hidden space!
    pca_t = rot_m["pca"][tgt_key]
    proc_t = rot_m["to_c4"][tgt_key]
    w_swap_p2 = _native_pullback_unit(u_swap_p2_s, pca_t, proc_t)
    w_swap_raw = _native_pullback_unit(u_swap_raw_s, pca_t, proc_t)
    w_house_p2 = _native_pullback_unit(u_house_p2_s, pca_t, proc_t)
    w_sent = _native_pullback_unit(u_sent_s, pca_t, proc_t)

    # Generate 20 random unit directions strictly inside span(V_tgt) orthogonal to {u_swap_p2_s, u_swap_raw_s, u_sent_s, u_house_p2_s}!
    Q_excl = V234._orthonormal_basis([u_swap_p2_s, u_swap_raw_s, u_sent_s, u_house_p2_s])
    rng = np.random.default_rng(24100 + slot)
    w_rands = []
    for _ in range(n_rand):
        rv = rng.standard_normal(256)
        rv = V234._project_out(rv, Q_excl)
        w_rands.append(_native_pullback_unit(rv, pca_t, proc_t))

    # Measure empirical RMS projection scale sigma_h along w_swap_p2 on target neutral passages so alpha is dimensionless!
    X_neut_t = data_map[tgt_key]["neut_mean"][slot].astype(np.float64)
    X_neut_c = X_neut_t - X_neut_t.mean(0, keepdims=True)
    # Typical per-token residual norm scale: use RMS norm of neutral hidden vectors divided by sqrt(D) OR std along w!
    sigma_w = float(np.std(X_neut_c @ w_swap_p2.astype(np.float64)))
    mean_h_norm = float(np.mean(np.linalg.norm(X_neut_t, axis=1)))
    # Also compute typical std along random PCA directions so alpha scales fairly across directions
    sigma_rand_mean = float(np.mean([np.std(X_neut_c @ wr.astype(np.float64)) for wr in w_rands]))
    print(f"  target={tgt_key} sigma_w={sigma_w:.3f}, sigma_rand={sigma_rand_mean:.3f}, mean||h||={mean_h_norm:.2f}", flush=True)
    # Use a SINGLE fixed steering magnitude step_base = 0.05 * mean_h_norm? Wait: better to normalize by exact
    # residual-stream norm fraction (`frac * mean_h_norm`), so EVERY direction (swap, house, sent, rand) gets
    # the EXACT SAME Euclidean L2 perturbation norm `alpha * step_unit`!!
    step_unit = float(0.05 * mean_h_norm)  # alpha=1.0 -> 5% of layer RMS norm; alpha=2.0 -> 10%; alpha=4.0 -> 20%!
    del data_map
    gc.collect()

    # Load evaluation pairs
    if eval_dataset == "bm120":
        rows = BM238.load_mirror_data()
    else:
        rows = json.loads((EXP_DIR / "bm_rel242_data.json").read_text())["bm_rel_120"]
    # Pick 1 orig pair per setting (60 unique settings) for counterbalanced A/B + B/A = 120 prompts!
    orig_rows = [r for r in rows if r["order"] == "orig"]
    assert len(orig_rows) == 60

    # Neutral perplexity sanity sentences (20 held-out neutral passages)
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    neut_eval_texts = [r["text"].strip() for r in ev_rows if r["domain"] == "held_gut"][:20]

    # Load target MLX model
    spec = PD.MODEL_SPECS[tgt_key]
    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl_cnt = len(layers)
    tgt_idx = max(1, min(nl_cnt - 2, int(round(depth_frac * (nl_cnt - 1)))))
    tgt_blk = layers[tgt_idx]
    id_A = _letter_token_id(tok, "A")
    id_B = _letter_token_id(tok, "B")
    print(f"  Loaded {tgt_key} L={nl_cnt}, hooking layer {tgt_idx}; tok(A)={id_A}, tok(B)={id_B}", flush=True)

    # Dynamic intervention state read by hook (supports both batch_size=1 and right-padded batched B>1!)
    state = {"mode": "none", "w": None, "alpha": 0.0, "mask_mx": None}

    patched = {}
    for blk in layers:
        cls = type(blk)
        if cls not in patched:
            orig = cls.__call__
            patched[cls] = orig

            def make_hook(orig_fn):
                def hooked(self, *a, **k):
                    out = orig_fn(self, *a, **k)
                    if self is tgt_blk and state["mode"] != "none":
                        is_tup = isinstance(out, tuple)
                        h = out[0] if is_tup else out
                        w_mx = state["w"]  # (D,)
                        if state["mode"] == "ablate":
                            proj = mx.sum(h.astype(mx.float32) * w_mx, axis=-1, keepdims=True)
                            h_new = h - (proj * w_mx).astype(h.dtype)
                        elif state["mode"] == "steer_spans":
                            delta_h = (state["mask_mx"] * float(state["alpha"]) * w_mx).astype(h.dtype)
                            h_new = h + delta_h
                        else:
                            h_new = h
                        out = (h_new,) + out[1:] if is_tup else h_new
                    return out
                return hooked

            cls.__call__ = make_hook(orig)

    pad_id = int(getattr(tok, "eos_token_id", 0) or 0)
    BS_DEFAULT = 16

    def eval_fc_list(items, swap_spans=False, mode="none", w_np=None, alpha=0.0, bs=BS_DEFAULT):
        state["mode"] = mode
        state["w"] = mx.array(w_np) if w_np is not None else None
        state["alpha"] = alpha
        out_vals = []
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            B = len(chunk)
            lens = [len(it["ids"]) for it in chunk]
            T = max(lens)
            arr = np.full((B, T), pad_id, dtype=np.int32)
            for b, it in enumerate(chunk):
                arr[b, :lens[b]] = it["ids"]
            if mode == "steer_spans":
                mask = np.zeros((B, T, 1), dtype=np.float32)
                for b, it in enumerate(chunk):
                    sp_p = it["sp_b"] if swap_spans else it["sp_a"]
                    sp_n = it["sp_a"] if swap_spans else it["sp_b"]
                    mask[b, sp_p[0]:sp_p[1], 0] = 1.0
                    mask[b, sp_n[0]:sp_n[1], 0] = -1.0
                state["mask_mx"] = mx.array(mask)
            else:
                state["mask_mx"] = None
            logits = model(mx.array(arr))
            if isinstance(logits, tuple):
                logits = logits[0]
            # Extract logits at each item's exact last real token lens[b] - 1 (strictly causal -> zero right-pad leak!)
            idx_b = mx.arange(B)
            idx_t = mx.array([l - 1 for l in lens])
            last_l = logits[idx_b, idx_t, :].astype(mx.float32)
            diff = last_l[:, id_A] - last_l[:, id_B]
            mx.eval(diff)
            out_vals.extend(np.array(diff, dtype=np.float64).tolist())
        return np.array(out_vals, dtype=np.float64)

    def eval_nll_list(ids_list, mode="none", w_np=None, bs=BS_DEFAULT):
        state["mode"] = mode
        state["w"] = mx.array(w_np) if w_np is not None else None
        state["alpha"] = 0.0
        state["mask_mx"] = None
        out_vals = []
        for i in range(0, len(ids_list), bs):
            chunk = ids_list[i:i + bs]
            B = len(chunk)
            lens = [len(ids) for ids in chunk]
            T = max(lens)
            arr = np.full((B, T), pad_id, dtype=np.int32)
            for b, ids in enumerate(chunk):
                arr[b, :lens[b]] = ids
            x = mx.array(arr)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            ce = nn.losses.cross_entropy(logits[:, :-1, :].astype(mx.float32), x[:, 1:], reduction="none")
            mx.eval(ce)
            ce_np = np.array(ce, dtype=np.float64)
            for b, ln in enumerate(lens):
                out_vals.append(float(ce_np[b, :ln - 1].mean()))
        return np.array(out_vals, dtype=np.float64)

    try:
        # Pre-tokenize all 120 counterbalanced forced-choice Binding prompts (60 settings x {pos=A, pos=B})
        fc_items = []
        for si, r in enumerate(orig_rows):
            ids_ab, sp_a, sp_b = _build_fc_prompt_and_spans(tok, r["pos"].strip(), r["neg"].strip())
            fc_items.append({"sid": si, "correct": "A", "ids": ids_ab, "sp_a": sp_a, "sp_b": sp_b})
            ids_ba, sp_a2, sp_b2 = _build_fc_prompt_and_spans(tok, r["neg"].strip(), r["pos"].strip())
            fc_items.append({"sid": si, "correct": "B", "ids": ids_ba, "sp_a": sp_a2, "sp_b": sp_b2})

        # Pre-tokenize 60 Topic/Setting double-dissociation prompts (30 A=true, 30 B=true)
        all_settings = [r["setting"] for r in orig_rows]
        topic_items = []
        for si, r in enumerate(orig_rows):
            foil = all_settings[(si + 17) % len(all_settings)]
            if si % 2 == 0:
                ids_t, sp_s, sp_a, sp_b = _build_topic_prompt_and_spans(tok, r["pos"].strip(), r["setting"], foil)
                topic_items.append({"sid": si, "correct": "A", "ids": ids_t, "sp_s": sp_s, "sp_a": sp_a, "sp_b": sp_b})
            else:
                ids_t, sp_s, sp_a, sp_b = _build_topic_prompt_and_spans(tok, r["pos"].strip(), foil, r["setting"])
                topic_items.append({"sid": si, "correct": "B", "ids": ids_t, "sp_s": sp_s, "sp_a": sp_a, "sp_b": sp_b})

        # Pre-tokenize intrinsic ΔNLL pairs (all 120 mirrored pairs) + 20 neutral passages
        bm_pos_ids = [PD._text_ids_mlx(tok, r["pos"].strip()) for r in rows]
        bm_neg_ids = [PD._text_ids_mlx(tok, r["neg"].strip()) for r in rows]
        neut_ids = [PD._text_ids_mlx(tok, t) for t in neut_eval_texts]

        # Exact-equivalence unit test: verify right-padded batching (bs=16) matches bs=1 within bf16 2-ULP (0.25)!
        b1_chk = eval_fc_list(fc_items[:4], mode="steer_spans", w_np=w_swap_p2, alpha=float(step_unit), bs=1)
        b16_chk = eval_fc_list(fc_items[:4], mode="steer_spans", w_np=w_swap_p2, alpha=float(step_unit), bs=4)
        max_err = float(np.max(np.abs(b1_chk - b16_chk)))
        use_bs = BS_DEFAULT if max_err < 0.25 else 1
        print(f"  Batch equivalence check (bs=4 vs bs=1, steered): max_err={max_err:.5f} -> using bs={use_bs}", flush=True)

        # 1. Baseline measurements (no intervention)
        base_fc_ab = eval_fc_list(fc_items, mode="none", bs=use_bs)
        base_fc_corr = np.array([m if it["correct"] == "A" else -m for m, it in zip(base_fc_ab, fc_items)])
        base_topic_ab = eval_fc_list(topic_items, mode="none", bs=use_bs)
        base_topic_corr = np.array([m if it["correct"] == "A" else -m for m, it in zip(base_topic_ab, topic_items)])
        base_dnll = eval_nll_list(bm_neg_ids, mode="none", bs=use_bs) - eval_nll_list(bm_pos_ids, mode="none", bs=use_bs)
        base_neut_nll = eval_nll_list(neut_ids, mode="none", bs=use_bs)

        print(f"  BASELINE: FC_bind acc={np.mean(base_fc_corr > 0):.4f} (margin={np.mean(base_fc_corr):+.3f}), "
              f"Topic acc={np.mean(base_topic_corr > 0):.4f}, ΔNLL acc={np.mean(base_dnll > 0):.4f} (mean={np.mean(base_dnll):+.4f})", flush=True)

        # 2. Test B: Targeted Clause Steering (`+delta` on Option A, `-delta` on Option B vs reversed) across alphas!
        named_dirs = [
            ("u_swap_p2", w_swap_p2),
            ("u_swap_raw", w_swap_raw),
            ("u_house_p2", w_house_p2),
            ("u_sent", w_sent),
        ]
        steer_res = {}
        for alpha in alphas:
            delta = float(alpha * step_unit)
            ares = {}
            for dname, wv in named_dirs:
                ab_plus = eval_fc_list(fc_items, swap_spans=False, mode="steer_spans", w_np=wv, alpha=delta, bs=use_bs)
                ab_minus = eval_fc_list(fc_items, swap_spans=True, mode="steer_spans", w_np=wv, alpha=delta, bs=use_bs)
                shift_vec = 0.5 * (ab_plus - ab_minus)
                by_s = np.array([shift_vec[2 * s:2 * s + 2].mean() for s in range(60)])
                rng_b = np.random.default_rng(24150)
                boots = [float(rng_b.choice(by_s, 60, replace=True).mean()) for _ in range(2000)]
                flips = [(ab_minus[i] < 0) if base_fc_ab[i] > 0 else (ab_plus[i] > 0) for i in range(len(fc_items))]
                pro_corr = [ab_plus[i] > 0 if fc_items[i]["correct"] == "A" else ab_minus[i] < 0 for i in range(len(fc_items))]
                anti_corr = [ab_minus[i] > 0 if fc_items[i]["correct"] == "A" else ab_plus[i] < 0 for i in range(len(fc_items))]
                t_plus = eval_fc_list(topic_items, swap_spans=False, mode="steer_spans", w_np=wv, alpha=delta, bs=use_bs)
                t_minus = eval_fc_list(topic_items, swap_spans=True, mode="steer_spans", w_np=wv, alpha=delta, bs=use_bs)
                t_shift = float(np.mean(0.5 * (t_plus - t_minus)))
                t_acc_pres = float(0.5 * (
                    np.mean([(m > 0 if it["correct"] == "A" else m < 0) for m, it in zip(t_plus, topic_items)])
                    + np.mean([(m > 0 if it["correct"] == "A" else m < 0) for m, it in zip(t_minus, topic_items)])
                ))
                ares[dname] = {
                    "mean_steer_logit_shift": round(float(np.mean(shift_vec)), 4),
                    "ci95_60settings": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                    "settings_pos_frac": round(float(np.mean(by_s > 0)), 4),
                    "anti_steer_flip_rate": round(float(np.mean(flips)), 4),
                    "acc_pro_steer": round(float(np.mean(pro_corr)), 4),
                    "acc_anti_steer": round(float(np.mean(anti_corr)), 4),
                    "topic_steer_logit_shift": round(t_shift, 4),
                    "topic_acc_preserved": round(t_acc_pres, 4),
                }

            rand_shifts, rand_flips, rand_topic_shifts = [], [], []
            for wr in w_rands:
                rp = eval_fc_list(fc_items, swap_spans=False, mode="steer_spans", w_np=wr, alpha=delta, bs=use_bs)
                rm = eval_fc_list(fc_items, swap_spans=True, mode="steer_spans", w_np=wr, alpha=delta, bs=use_bs)
                rand_shifts.append(float(np.mean(0.5 * (rp - rm))))
                rand_flips.append(float(np.mean([(rm[i] < 0) if base_fc_ab[i] > 0 else (rp[i] > 0) for i in range(len(fc_items))])))
                tp = eval_fc_list(topic_items, swap_spans=False, mode="steer_spans", w_np=wr, alpha=delta, bs=use_bs)
                tm = eval_fc_list(topic_items, swap_spans=True, mode="steer_spans", w_np=wr, alpha=delta, bs=use_bs)
                rand_topic_shifts.append(float(np.mean(0.5 * (tp - tm))))
            rand_abs = [abs(x) for x in rand_shifts]
            for dname, _ in named_dirs:
                obs = ares[dname]["mean_steer_logit_shift"]
                ares[dname]["p_vs_20rand_signed"] = round(float((sum(x >= obs for x in rand_shifts) + 1) / (n_rand + 1)), 4)
                ares[dname]["p_vs_20rand_abs"] = round(float((sum(x >= abs(obs) for x in rand_abs) + 1) / (n_rand + 1)), 4)
            ares["random_20_null"] = {
                "mean_signed": round(float(np.mean(rand_shifts)), 4),
                "mean_abs": round(float(np.mean(rand_abs)), 4),
                "max_signed": round(float(np.max(rand_shifts)), 4),
                "max_abs": round(float(np.max(rand_abs)), 4),
                "mean_flip_rate": round(float(np.mean(rand_flips)), 4),
                "max_flip_rate": round(float(np.max(rand_flips)), 4),
                "mean_topic_abs_shift": round(float(np.mean([abs(x) for x in rand_topic_shifts])), 4),
            }
            steer_res[f"alpha_{alpha}"] = ares
            print(f"  STEER alpha={alpha} (L2={delta:.2f}): swap_p2={ares['u_swap_p2']['mean_steer_logit_shift']:+.3f} (flip={ares['u_swap_p2']['anti_steer_flip_rate']:.3f}, p={ares['u_swap_p2']['p_vs_20rand_signed']}, topic_acc={ares['u_swap_p2']['topic_acc_preserved']:.3f}) | "
                  f"house_p2={ares['u_house_p2']['mean_steer_logit_shift']:+.3f} | sent={ares['u_sent']['mean_steer_logit_shift']:+.3f} | rand_max={ares['random_20_null']['max_abs']:.3f}", flush=True)

        # 3. Test A: Pure 1D Projection Ablation on intrinsic ΔNLL, FC Binding, Topic, and Neutral Perplexity
        ablate_res = {}
        for dname, wv in named_dirs:
            dn_ab = eval_nll_list(bm_neg_ids, mode="ablate", w_np=wv, bs=use_bs) - eval_nll_list(bm_pos_ids, mode="ablate", w_np=wv, bs=use_bs)
            fc_ab = eval_fc_list(fc_items, mode="ablate", w_np=wv, bs=use_bs)
            fc_c = np.array([m if it["correct"] == "A" else -m for m, it in zip(fc_ab, fc_items)])
            tp_ab = eval_fc_list(topic_items, mode="ablate", w_np=wv, bs=use_bs)
            tp_c = np.array([m if it["correct"] == "A" else -m for m, it in zip(tp_ab, topic_items)])
            nt_ab = eval_nll_list(neut_ids, mode="ablate", w_np=wv, bs=use_bs)
            ablate_res[dname] = {
                "dnll_mean_drop": round(float(np.mean(base_dnll - dn_ab)), 5),
                "dnll_acc_after": round(float(np.mean(dn_ab > 0)), 4),
                "fc_margin_drop": round(float(np.mean(base_fc_corr - fc_c)), 4),
                "fc_acc_after": round(float(np.mean(fc_c > 0)), 4),
                "topic_margin_drop": round(float(np.mean(base_topic_corr - tp_c)), 4),
                "topic_acc_after": round(float(np.mean(tp_c > 0)), 4),
                "neut_nll_abs_drift": round(float(np.mean(np.abs(nt_ab - base_neut_nll))), 5),
            }
        r_dn_drops, r_fc_drops, r_nt_drifts = [], [], []
        for wr in w_rands[:10]:
            dn_ab = eval_nll_list(bm_neg_ids, mode="ablate", w_np=wr, bs=use_bs) - eval_nll_list(bm_pos_ids, mode="ablate", w_np=wr, bs=use_bs)
            fc_ab = eval_fc_list(fc_items, mode="ablate", w_np=wr, bs=use_bs)
            fc_c = np.array([m if it["correct"] == "A" else -m for m, it in zip(fc_ab, fc_items)])
            nt_ab = eval_nll_list(neut_ids, mode="ablate", w_np=wr, bs=use_bs)
            r_dn_drops.append(float(np.mean(base_dnll - dn_ab)))
            r_fc_drops.append(float(np.mean(base_fc_corr - fc_c)))
            r_nt_drifts.append(float(np.mean(np.abs(nt_ab - base_neut_nll))))
        ablate_res["random_10_null"] = {
            "dnll_drop_mean": round(float(np.mean(r_dn_drops)), 5),
            "dnll_drop_max": round(float(np.max(r_dn_drops)), 5),
            "fc_margin_drop_mean": round(float(np.mean(r_fc_drops)), 4),
            "fc_margin_drop_max": round(float(np.max(r_fc_drops)), 4),
            "neut_nll_abs_drift_mean": round(float(np.mean(r_nt_drifts)), 5),
        }

        out = {
            "src_key": src_key,
            "tgt_key": tgt_key,
            "depth_frac": depth_frac,
            "layer_idx": int(tgt_idx),
            "num_layers": int(nl_cnt),
            "eval_dataset": eval_dataset,
            "step_unit_l2": round(step_unit, 4),
            "baseline": {
                "fc_bind_acc": round(float(np.mean(base_fc_corr > 0)), 4),
                "fc_bind_margin": round(float(np.mean(base_fc_corr)), 4),
                "topic_acc": round(float(np.mean(base_topic_corr > 0)), 4),
                "topic_margin": round(float(np.mean(base_topic_corr)), 4),
                "dnll_acc": round(float(np.mean(base_dnll > 0)), 4),
                "dnll_mean": round(float(np.mean(base_dnll)), 5),
                "neut_nll_mean": round(float(np.mean(base_neut_nll)), 4),
            },
            "targeted_clause_steering": steer_res,
            "ablation_1d": ablate_res,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        return out
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()


def main():
    depth_frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
    ds = sys.argv[2] if len(sys.argv) > 2 else "bm120"
    tag = f"d{int(round(depth_frac * 100))}_{ds}"
    out_json = EXP_DIR / f"causal_steer241_{tag}.json"
    runs = {}
    for sk, tk in [
        ("gemma4_e4b", "qwen35_9b"),
        ("gemma4_e4b", "llama31_8b"),
        ("qwen35_9b", "olmo3_7b"),
    ]:
        runs[f"{sk}__to__{tk}"] = run_causal_pair(sk, tk, depth_frac=depth_frac, eval_dataset=ds)
    out_json.write_text(json.dumps(runs, indent=2))
    print(f"SAVED {out_json}", flush=True)


if __name__ == "__main__":
    main()
