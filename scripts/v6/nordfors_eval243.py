"""Experiment 243: Independent Non-Agent-Authored Validation on Nordfors (2026) 14 Relational Texts Across 6 Labs x 5 Depths.

Tests the 14 independently authored texts released under CC BY 4.0 by Nordfors (arXiv:2611.07234v2,
`anc/reproduce/texts/test_texts.jsonl`): 4 Senge/Gentner relational patterns (`the tipping point`,
`winner takes all`, `spontaneous synchronization`, `precommitment`) across 5 subject domains
(`ecology`, `politics and society`, `networking and cybersecurity`, `cell and molecular biology`,
`science and academia`) and 3 languages (`en` n=12, `zh` n=1, `ja` n=1).

Directly measures at ALL 5 depths (15%, 30%, 50%, 70%, 85%) in frozen neutral-text C4 and per-model spaces:
  1. Same-Pattern-Different-Topic vs. Different-Both pairwise cosine gap & AUROC (all 91 pairs & English-only 66 pairs),
     plus exact 5,000-draw label-permutation p-value!
  2. Same-Topic-Different-Pattern vs. Different-Both pairwise cosine gap & AUROC (directly testing Nordfors's
     "sandwich" claim: do topics peak earlier while cross-domain relational patterns peak deeper in middle layers,
     and does multimodal Gemma 4 peak earlier than text-only Qwen/Llama/Mistral/Phi/OLMo?).
  3. Cross-family Leave-One-Out / Cross-Topic Nearest-Centroid Retrieval in neutral-text C4 (Gemma 4 / Qwen 3.5 -> rival labs).
  4. Lexical baselines: Unigram BoW & TF-IDF pairwise cosine AUROC on the exact same 91 pairs.
"""
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH_DIR / "exp240_242"
NORDFORS_JSONL = SCRATCH_DIR / "refs/nordfors_src/anc/reproduce/texts/test_texts.jsonl"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"
OUT_JSON = EXP_DIR / "nordfors243_results.json"

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import depth_sweep240 as D240  # noqa: E402

MLX_KEYS = D240.MLX_KEYS
C4_KEYS = D240.C4_KEYS
DEPTH_FRACS = D240.DEPTH_FRACS


def load_nordfors():
    return [json.loads(l) for l in open(NORDFORS_JSONL)]


def extract_5depths(keys=MLX_KEYS):
    import mlx.core as mx
    from mlx_lm import load

    rows = load_nordfors()
    texts = [r["text"].strip() for r in rows]
    for key in keys:
        out_npz = DEPTH_DIR / f"nord243_{key}.npz"
        if out_npz.exists():
            print(f"NORD243 EXTRACT {key}: exists, skipping", flush=True)
            continue
        spec = PD.MODEL_SPECS[key]
        mx.set_cache_limit(2 << 30)
        t0 = time.time()
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
        means = [[] for _ in range(len(DEPTH_FRACS))]
        lasts = [[] for _ in range(len(DEPTH_FRACS))]
        try:
            for txt in texts:
                ids = PD._text_ids_mlx(tok, txt)
                rec_buf.clear()
                _ = model(mx.array([ids]))
                ev = []
                sv = []
                for s in range(len(DEPTH_FRACS)):
                    h = rec_buf[s].astype(mx.float32)
                    hm = h[1:].mean(0) if h.shape[0] > 1 else h[0]
                    hl = h[-1]
                    sv.append((hm, hl)); ev.extend([hm, hl])
                mx.eval(*ev)
                rec_buf.clear()
                for s, (hm, hl) in enumerate(sv):
                    means[s].append(np.array(hm, dtype=np.float32))
                    lasts[s].append(np.array(hl, dtype=np.float32))
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            del layers, model, tok
            gc.collect()
            mx.clear_cache()
        PD._atomic_savez(
            out_npz,
            mean=np.stack([np.stack(x, 0) for x in means], 0).astype(np.float32),
            last=np.stack([np.stack(x, 0) for x in lasts], 0).astype(np.float32),
            layer_idxs=np.array(layer_idxs, dtype=np.int32),
            num_layers=np.int32(nl_cnt),
        )
        print(f"NORD243_EXTRACT_DONE {key} in {time.time() - t0:.1f}s", flush=True)


def _pair_metrics(Z, archs, topics, langs, en_only=False, n_perm=5000, seed=243):
    """Centered pairwise cosine analysis comparing Same-Pattern-Diff-Topic, Same-Topic-Diff-Pattern, and Diff-Both."""
    idx = [i for i in range(len(archs)) if (not en_only or langs[i] == "en")]
    Zs = Z[idx]
    Zc = Zs - Zs.mean(0, keepdims=True)
    Zn = Zc / np.maximum(1e-12, np.linalg.norm(Zc, axis=1, keepdims=True))
    S = Zn @ Zn.T
    a_sub = [archs[i] for i in idx]
    t_sub = [topics[i] for i in idx]

    sp_dt, st_dp, diff_b = [], [], []
    pairs_ij = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            sa = (a_sub[i] == a_sub[j])
            st = (t_sub[i] == t_sub[j])
            pairs_ij.append((i, j))
            if sa and not st:
                sp_dt.append(S[i, j])
            elif st and not sa:
                st_dp.append(S[i, j])
            elif not sa and not st:
                diff_b.append(S[i, j])

    auroc_pat = NA235._auroc([1] * len(sp_dt) + [0] * len(diff_b), sp_dt + diff_b)
    auroc_top = NA235._auroc([1] * len(st_dp) + [0] * len(diff_b), st_dp + diff_b)
    auroc_pat_vs_top = NA235._auroc([1] * len(sp_dt) + [0] * len(st_dp), sp_dt + st_dp)

    # Exact permutation p-values (permute archetype labels for pattern, topic labels for topic)
    rng = np.random.default_rng(seed)
    obs_p_gap = float(np.mean(sp_dt) - np.mean(diff_b))
    obs_t_gap = float(np.mean(st_dp) - np.mean(diff_b))
    ge_p, ge_t = 0, 0
    a_arr = np.array(a_sub)
    t_arr = np.array(t_sub)
    for _ in range(n_perm):
        ap = rng.permutation(a_arr)
        s1, s0 = [], []
        for i, j in pairs_ij:
            if t_arr[i] != t_arr[j]:
                (s1 if ap[i] == ap[j] else s0).append(S[i, j])
        if float(np.mean(s1) - np.mean(s0)) >= obs_p_gap:
            ge_p += 1
        tp = rng.permutation(t_arr)
        t1, t0 = [], []
        for i, j in pairs_ij:
            if a_arr[i] != a_arr[j]:
                (t1 if tp[i] == tp[j] else t0).append(S[i, j])
        if float(np.mean(t1) - np.mean(t0)) >= obs_t_gap:
            ge_t += 1

    # Also cross-lingual check: synchronization zh & ja vs synchronization en vs other en!
    xling = None
    if not en_only:
        sync_x = [S[i, j] for i in range(len(idx)) for j in range(len(idx))
                  if langs[idx[i]] != "en" and langs[idx[j]] == "en" and archs[idx[i]] == archs[idx[j]]]
        other_x = [S[i, j] for i in range(len(idx)) for j in range(len(idx))
                   if langs[idx[i]] != "en" and langs[idx[j]] == "en" and archs[idx[i]] != archs[idx[j]]]
        xling = {
            "xling_same_pat_cos": round(float(np.mean(sync_x)), 4),
            "xling_diff_pat_cos": round(float(np.mean(other_x)), 4),
            "xling_auroc": round(NA235._auroc([1] * len(sync_x) + [0] * len(other_x), sync_x + other_x), 4),
        }

    return {
        "n_pairs": {"same_pat_diff_top": len(sp_dt), "same_top_diff_pat": len(st_dp), "diff_both": len(diff_b)},
        "mean_cos_same_pat_diff_top": round(float(np.mean(sp_dt)), 4),
        "mean_cos_same_top_diff_pat": round(float(np.mean(st_dp)), 4),
        "mean_cos_diff_both": round(float(np.mean(diff_b)), 4),
        "auroc_pattern_xtopic": round(auroc_pat, 4),
        "p_pattern_xtopic": round((ge_p + 1) / (n_perm + 1), 4),
        "auroc_topic_xpattern": round(auroc_top, 4),
        "p_topic_xpattern": round((ge_t + 1) / (n_perm + 1), 4),
        "auroc_pat_over_top": round(auroc_pat_vs_top, 4),
        "cross_lingual_zh_ja_to_en": xling,
    }


def _cross_model_loo_xtopic(Z_src, Z_tgt, archs, topics):
    """Zero-shot cross-family Leave-One-Out retrieval: fit 4 archetype centroids on source EXCLUDING test item i
    (and any item sharing i's topic!), score test item i in target!"""
    uniq_a = sorted(set(archs))
    correct, total = 0, 0
    for i in range(len(archs)):
        tr = [j for j in range(len(archs)) if j != i and topics[j] != topics[i]]
        # Ensure all 4 archetypes present in tr
        if len({archs[j] for j in tr}) < len(uniq_a):
            continue
        Zs_c = Z_src[tr] - Z_src[tr].mean(0, keepdims=True)
        Zt_i = Z_tgt[i] - Z_tgt[tr].mean(0)
        C_a = np.stack([AR.unit(Zs_c[[k for k, j in enumerate(tr) if archs[j] == a]].mean(0)) for a in uniq_a], 0)
        pred = uniq_a[int(np.argmax(C_a @ Zt_i))]
        correct += int(pred == archs[i])
        total += 1
    return round(correct / max(1, total), 4), correct, total


def analyze():
    rows = load_nordfors()
    texts = [r["text"].strip() for r in rows]
    archs = [r["archetype"] for r in rows]
    topics = [r["topic"] for r in rows]
    langs = [r["language"] for r in rows]

    # Lexical baselines (binary unigram BoW & TF-IDF)
    def tok_u(t):
        return re.findall(r"[a-z]{2,}", t.lower())
    vocab = sorted({w for t in texts for w in tok_u(t)})
    w2i = {w: j for j, w in enumerate(vocab)}
    X_bow = np.zeros((len(texts), len(vocab)))
    df = {}
    for i, t in enumerate(texts):
        for w in set(tok_u(t)):
            X_bow[i, w2i[w]] = 1.0
            df[w] = df.get(w, 0) + 1
    idf = np.array([math.log((1 + len(texts)) / (1 + df[w])) + 1.0 for w in vocab])
    X_tf = np.zeros_like(X_bow)
    for i, t in enumerate(texts):
        cnt = {}
        for w in tok_u(t):
            cnt[w2i[w]] = cnt.get(w2i[w], 0) + 1
        for j, c in cnt.items():
            X_tf[i, j] = (1.0 + math.log(c)) * idf[j]
    lex_res = {
        "bow_unigram_all14": _pair_metrics(X_bow, archs, topics, langs, en_only=False, n_perm=5000, seed=1),
        "bow_unigram_en12": _pair_metrics(X_bow, archs, topics, langs, en_only=True, n_perm=5000, seed=2),
        "tfidf_all14": _pair_metrics(X_tf, archs, topics, langs, en_only=False, n_perm=5000, seed=3),
        "tfidf_en12": _pair_metrics(X_tf, archs, topics, langs, en_only=True, n_perm=5000, seed=4),
    }

    data_map = {k: dict(np.load(DEPTH_DIR / f"depth240_{k}.npz")) for k in MLX_KEYS}
    nord_map = {k: dict(np.load(DEPTH_DIR / f"nord243_{k}.npz")) for k in MLX_KEYS}

    by_depth = {}
    for slot, df_val in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df_val * 100))}"
        rot_m = D240._fit_rot_at_slot(data_map, slot, "mean", d_plato=256)
        Zn = {k: D240._map(nord_map[k]["mean"][slot], rot_m, k) for k in MLX_KEYS}
        C4_n = np.mean([Zn[k] for k in C4_KEYS], axis=0)
        per_m = {}
        for k in MLX_KEYS:
            # Native space AND C4-projected space
            per_m[k] = {
                "all14": _pair_metrics(Zn[k], archs, topics, langs, en_only=False, n_perm=2000, seed=100 + slot),
                "en12": _pair_metrics(Zn[k], archs, topics, langs, en_only=True, n_perm=2000, seed=200 + slot),
            }
        # Zero-shot cross-family LOO cross-topic retrieval (Gemma 4 -> 5, Qwen 3.5 -> 5)
        xloo = {}
        for src in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src]
            xloo[src] = {tk: _cross_model_loo_xtopic(Zn[src], Zn[tk], archs, topics)[0] for tk in tgts}
            xloo[src]["mean_5labs"] = round(float(np.mean(list(xloo[src].values()))), 4)

        by_depth[dkey] = {
            "c4_all14": _pair_metrics(C4_n, archs, topics, langs, en_only=False, n_perm=5000, seed=300 + slot),
            "c4_en12": _pair_metrics(C4_n, archs, topics, langs, en_only=True, n_perm=5000, seed=400 + slot),
            "per_model": per_m,
            "cross_family_xtopic_loo": xloo,
        }

    report = {"n_texts": len(rows), "lexical_baselines": lex_res, "by_depth": by_depth}
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON}", flush=True)
    return report


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("extract", "all"):
        extract_5depths()
    if cmd in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
