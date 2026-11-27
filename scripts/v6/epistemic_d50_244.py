"""Experiment 244: 5-Depth Native-Layer Epistemic Probing (CORRECTNESS + HOUSE_PERP vs. SENTIMENT_PERP & Entropy).

Extends Section 4.7 (`audit233_epistemic.py`) across ALL FIVE relative depths (15%, 30%, 50%, 70%, 85%)
on FOUR independent open LLM families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, OLMo 3 7B) using the exact
zero-entity-overlap `CLEAN_FALSE_PREMISE_12` + `CLEAN_TENSE_MATCHED_UNK_12` (24 pairs) under BOTH
`NEUTRAL_SYS` and `BUDDY_SYSTEM_PROMPT`, with strictly calibration-frozen z-scoring, Gram-Schmidt
orthogonalization against `CORRECTNESS` AND `SENTIMENT`, and 500-draw mirrored random-pair nulls.

Directly tests whether the 50%-depth relational peak discovered in Exp 240 strengthens native-layer
`CORRECTNESS + HOUSE_PERP` epistemic ranking and valence-orthogonal `-HOUSE_PERP_BOTH` across 4 labs!
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
OUT_JSON = EXP_DIR / "epistemic_d50_244_results.json"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent")):
    if p not in sys.path:
        sys.path.insert(0, p)

from aion_8b import audit233_mac as am, audit233_epistemic as ae, aion_data230 as bd, sceptic_ctrl230 as sc  # noqa: E402
from fluent import arch_rot229, plato_data, valence_ctrl234 as V234  # noqa: E402

MODELS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "olmo3_7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)


def _score_with_sent(V_cal_tf, V_cal_house, V_cal_sent, V_eval, ents_eval, u_corr, u_house, u_sent, sub_masks, n_rand=500, seed=244):
    base = ae._score_clean_epistemic(V_cal_tf, V_cal_house, V_eval, ents_eval, u_corr, u_house, sub_masks, n_rand=n_rand, seed=seed)
    uc = arch_rot229.unit(u_corr)
    uh = arch_rot229.unit(u_house)
    us = arch_rot229.unit(u_sent)
    us_p = arch_rot229.unit(us - float(np.dot(us, uc)) * uc)
    Q_cs = V234._orthonormal_basis([uc, us_p])
    uh_pb = V234._project_out(uh, Q_cs)

    V_cal = np.concatenate([V_cal_tf, V_cal_house, V_cal_sent], axis=0)
    def cal_z(V_ev, u):
        cp = -(V_cal @ u)
        return (-(V_ev @ u) - float(cp.mean())) / max(1e-6, float(cp.std()))

    zc = cal_z(V_eval, uc)
    zsp = cal_z(V_eval, us_p)
    zhpb = cal_z(V_eval, uh_pb)
    comb_s = 0.5 * zc + 0.5 * zsp
    comb_pb = 0.5 * zc + 0.5 * zhpb

    for sub_name, idx_pairs in sub_masks.items():
        ids_flat = [x for pair in idx_pairs for x in pair]
        y = np.array([0, 1] * len(idx_pairs), dtype=int)
        d = base["subsets"][sub_name]
        d["neg_sent_perp_auroc"] = round(ae._auroc(y, zsp[ids_flat]), 4)
        d["neg_house_perp_both_auroc"] = round(ae._auroc(y, zhpb[ids_flat]), 4)
        d["combined_corr_sent_auroc"] = round(ae._auroc(y, comb_s[ids_flat]), 4)
        d["combined_corr_house_pb_auroc"] = round(ae._auroc(y, comb_pb[ids_flat]), 4)
    return base


def eval_model_5depths(key):
    import mlx.core as mx
    from mlx_lm import load

    t0 = time.time()
    mx.set_cache_limit(2 << 30)
    spec = plato_data.MODEL_SPECS[key]
    print(f"EPI244 {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = plato_data._find_layers(model)
    nl = len(layers)
    layer_idxs = [max(1, min(nl - 2, int(round(df * (nl - 1))))) for df in DEPTH_FRACS]
    blk2slot = {id(layers[li]): slot for slot, li in enumerate(layer_idxs)}
    n_d = len(DEPTH_FRACS)

    rec = {}
    orig_calls = {}
    for blk in layers:
        cls = type(blk)
        if cls not in orig_calls:
            orig_calls[cls] = cls.__call__
            def make_wrap(orig_fn):
                def wrapped(self_blk, *a, **kw):
                    y = orig_fn(self_blk, *a, **kw)
                    slot = blk2slot.get(id(self_blk))
                    if slot is not None:
                        h = y[0] if isinstance(y, tuple) else y
                        rec[slot] = h[0, -1].astype(mx.float32)
                    return y
                return wrapped
            cls.__call__ = make_wrap(orig_calls[cls])

    def get_last(txt, already_templated=False):
        ids = tok.encode(txt, add_special_tokens=False)[:512] if already_templated else plato_data._text_ids_mlx(tok, txt)[:512]
        rec.clear()
        logits = model(mx.array([ids]))
        if isinstance(logits, tuple):
            logits = logits[0]
        lg = logits[0, -1].astype(mx.float32)
        p = mx.softmax(lg, axis=-1)
        ent = -(p * mx.log(mx.maximum(p, 1e-12))).sum()
        ev_list = [rec[s] for s in range(n_d)] + [ent]
        mx.eval(*ev_list)
        vecs = [np.array(rec[s], dtype=np.float32) for s in range(n_d)]
        return vecs, float(ent.item())

    all_pairs = am.CLEAN_FALSE_PREMISE_12 + am.CLEAN_TENSE_MATCHED_UNK_12
    sub_masks = {
        "clean_false_premise_12": [(2 * i, 2 * i + 1) for i in range(12)],
        "clean_tense_matched_unk_12": [(24 + 2 * i, 24 + 2 * i + 1) for i in range(12)],
        "clean_tense_clean_24": [(2 * i, 2 * i + 1) for i in range(24)],
    }

    try:
        diffs_c = [[] for _ in range(n_d)]; cal_tf = [[] for _ in range(n_d)]
        for ts, fs in sc.TRUE_FALSE_20:
            vt, _ = get_last(ts); vf, _ = get_last(fs)
            for s in range(n_d):
                diffs_c[s].append(vt[s] - vf[s])
                cal_tf[s].extend([vt[s], vf[s]])
        u_corr = [arch_rot229.unit(np.stack(diffs_c[s]).mean(0)) for s in range(n_d)]

        pairs = arch_rot229.load_arch_pairs()
        diffs_h = [[] for _ in range(n_d)]; cal_h = [[] for _ in range(n_d)]; cats_h = []
        for p in pairs:
            if p["cat"] in arch_rot229.HOUSE_CATS:
                vd, _ = get_last(p["done"]); vn, _ = get_last(p["neglected"])
                for s in range(n_d):
                    diffs_h[s].append(vd[s] - vn[s])
                    cal_h[s].extend([vd[s], vn[s]])
                cats_h.append(p["cat"])
        u_house = [arch_rot229.house_dir(np.stack(diffs_h[s]), np.array(cats_h)) for s in range(n_d)]

        sg_rows, _, _ = V234.load_valence_data()
        diffs_s = [[] for _ in range(n_d)]; cal_s = [[] for _ in range(n_d)]
        for r in sg_rows:
            vp, _ = get_last(r["pos"]); vn, _ = get_last(r["neg"])
            for s in range(n_d):
                diffs_s[s].append(vp[s] - vn[s])
                cal_s[s].extend([vp[s], vn[s]])
        u_sent = [arch_rot229.unit(np.stack(diffs_s[s]).mean(0)) for s in range(n_d)]

        res_prompts = {}
        for pname, sys_p in [("neutral_prompt", sc.NEUTRAL_SYS), ("buddy_prompt", bd.BUDDY_SYSTEM_PROMPT)]:
            vecs_l = [[] for _ in range(n_d)]; ents = []
            for kq, uq in all_pairs:
                for q in (kq, uq):
                    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": q}]
                    try:
                        pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    vs, ent = get_last(pr, already_templated=True)
                    for s in range(n_d):
                        vecs_l[s].append(vs[s])
                    ents.append(ent)
            res_prompts[pname] = {}
            for s, df in enumerate(DEPTH_FRACS):
                dkey = f"d{int(round(df * 100))}_L{layer_idxs[s]}"
                res_prompts[pname][dkey] = _score_with_sent(
                    np.stack(cal_tf[s]), np.stack(cal_h[s]), np.stack(cal_s[s]),
                    np.stack(vecs_l[s]), ents, u_corr[s], u_house[s], u_sent[s], sub_masks, n_rand=500, seed=244 + s
                )
        print(f"EPI244_DONE {key} in {time.time() - t0:.1f}s", flush=True)
        return {"layers": layer_idxs, "num_layers": nl, **res_prompts}
    finally:
        for cls, orig_fn in orig_calls.items():
            cls.__call__ = orig_fn
        del model, tok, layers
        gc.collect()
        mx.clear_cache()


def main():
    ae._verify_zero_entity_overlap()
    out = {}
    for k in MODELS:
        out[k] = eval_model_5depths(k)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"SAVED {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
