"""Experiment 235: Fair Trial of 8 Canonical Named Jungian Archetypes (Table III Neutral-Text Protocol Across 7 LLM Families).

Tests whether 8 canonical Jungian archetypes (`Hero`, `Shadow`, `Trickster`, `Wise_Old_Man`, `Great_Mother`,
`Persona`, `Anima_Animus`, `Threshold_Guardian`) transfer zero-shot across 6 rival LLM families via rigid
Procrustes rotations fit strictly on neutral `gut + proc + chat` text (`arch_rot229.py`), under four strict
controls:
  1. Setting-balanced design: 40 fresh everyday modern settings x 8 archetypes = 320 stories (`named_arch235_data.json`),
     so every setting appears once in every archetype (zero setting/topic confound, zero overlap with HOUSE/SELF2/VAL234,
     zero archetype-name ban-list leakage, balanced word counts across all 8 archetypes).
  2. 20 deterministic setting-disjoint 2-fold splits (20 train settings = 160 stories / 20 test settings = 160 stories),
     scored with per-archetype One-vs.-Rest (OvR) AUROC, 8-way Top-1/Top-2 accuracy (chance 12.5%/25.0%),
     2-fold direction cosine, and 1,000-draw within-setting label-permutation p-values.
  3. Three strong lexical baselines fit strictly train-only on the exact same 20 setting-disjoint splits:
     unigram binary BoW, unigram+bigram sublinear TF-IDF centroid, and unigram+bigram TF-IDF L2-Ridge OvR.
  4. Double-projected valence & HOUSE controls: projecting out `VALENCE_2D = span(u_sent, u_outval)` AND
     `VAL3 = span(u_sent, u_outval, u_house)` in BOTH source Gemma 4 and each target model, plus fold-local
     token-length residualization and effective dimensionality (participation ratio) inside `C5`.

Usage:
  python fluent/named_arch235.py test
  python fluent/named_arch235.py extract [KEY...]
  python fluent/named_arch235.py analyze
  python fluent/named_arch235.py all
"""
import gc
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO_LOCAL = HERE / "plato"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(PLATO_LOCAL)))
DATA_JSON = ARCH / "selfshape" / "named_arch235_data.json"
OUT_JSON = PLATO_LOCAL / "named_arch235.json"

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402

ARCHETYPES = (
    "Hero",
    "Shadow",
    "Trickster",
    "Wise_Old_Man",
    "Great_Mother",
    "Persona",
    "Anima_Animus",
    "Threshold_Guardian",
)

BAN_WORDS = (
    "hero", "heroic", "heroism", "heroes", "heroine",
    "shadow", "shadows", "shadowy", "shadowed",
    "trickster", "tricksters", "trick", "tricks", "tricked", "tricking", "tricky",
    "prank", "pranks", "prankster", "pranked",
    "wise", "wisely", "wisdom", "wiser", "wisest",
    "sage", "sages", "mentor", "mentors", "mentored", "mentoring", "elder", "elders", "elderly",
    "mother", "mothers", "motherly", "maternal", "nurturing", "nurture", "nurtured",
    "persona", "personas", "mask", "masks", "masked", "masking", "facade", "facades",
    "anima", "animus", "muse", "muses",
    "guardian", "guardians", "gatekeeper", "gatekeepers", "threshold", "thresholds",
    "archetype", "archetypes", "archetypal", "jung", "jungian",
    "myth", "myths", "mythic", "mythical", "fable", "fables",
)
_BAN_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in BAN_WORDS) + r")\b", re.I)


def load_named_arch_data():
    rows = json.loads(DATA_JSON.read_text())
    assert isinstance(rows, list) and len(rows) == 320, f"Expected 320 rows, got {len(rows)}"
    return rows


def verify_dataset_integrity():
    """Verify exact 40 settings x 8 archetypes = 320 stories, zero overlap with existing settings,
    zero ban-list words, balanced word counts across archetypes, and zero 12-gram overlap."""
    rows = load_named_arch_data()
    sids = sorted({r["setting_id"] for r in rows})
    assert len(sids) == 40, f"Expected 40 unique setting_ids, got {len(sids)}"

    seen_pairs = set()
    by_arch_wc = {a: [] for a in ARCHETYPES}
    new_settings = set()
    ban_hits = []
    for i, r in enumerate(rows):
        a = r["archetype"]
        assert a in ARCHETYPES, f"Unknown archetype '{a}' at row {i}"
        sid = int(r["setting_id"])
        key = (sid, a)
        assert key not in seen_pairs, f"Duplicate (setting_id, archetype) {key}"
        seen_pairs.add(key)
        s_norm = r["setting"].strip().lower()
        new_settings.add(s_norm)
        txt = r["text"].strip()
        wc = len(txt.split())
        assert 38 <= wc <= 62, f"Word count {wc} out of [38, 62] at {key}: {txt[:80]}"
        by_arch_wc[a].append(wc)
        m = _BAN_RE.findall(txt)
        if m:
            ban_hits.append((key, m))

    assert len(seen_pairs) == 320, f"Incomplete grid: {len(seen_pairs)}"
    assert not ban_hits, f"Ban-list leakage found: {ban_hits[:10]}"

    # Check zero setting overlap with HOUSE, SELF2, and valence_ctrl234
    existing_settings = {p["setting"].strip().lower() for p in AR.load_arch_pairs() if p.get("setting")}
    _, _, vm = V234.load_valence_data()
    existing_settings |= {x["setting"].strip().lower() for x in vm if x.get("setting")}
    overlap = existing_settings & new_settings
    assert not overlap, f"Setting overlap with existing benchmarks: {overlap}"

    # Check zero 12-gram overlap with existing benchmarks
    def ngrams12(text):
        toks = re.findall(r"[a-z]+", text.lower())
        return {tuple(toks[j:j + 12]) for j in range(max(0, len(toks) - 11))}

    ex_12 = set()
    for p in AR.load_arch_pairs():
        ex_12 |= ngrams12(p["done"]) | ngrams12(p["neglected"])
    sg, ov, vm = V234.load_valence_data()
    for r in sg + ov:
        ex_12 |= ngrams12(r["pos"]) | ngrams12(r["neg"])
    for r in vm:
        ex_12 |= ngrams12(r["integrated"]) | ngrams12(r["neglected"])
    for r in rows:
        hit = ngrams12(r["text"]) & ex_12
        assert not hit, f"12-gram overlap in {r['setting_id']}/{r['archetype']}: {hit}"

    wc_stats = {
        a: {"mean": round(float(np.mean(ws)), 2), "std": round(float(np.std(ws)), 2),
            "min": int(min(ws)), "max": int(max(ws))}
        for a, ws in by_arch_wc.items()
    }
    means = [v["mean"] for v in wc_stats.values()]
    assert max(means) - min(means) <= 5.0, f"Archetype mean word lengths imbalanced: {wc_stats}"
    return {
        "n_stories": len(rows),
        "n_settings": len(sids),
        "n_archetypes": len(ARCHETYPES),
        "n_existing_settings_checked": len(existing_settings),
        "setting_overlap": 0,
        "ban_list_hits": 0,
        "ngram12_overlap": 0,
        "word_count_by_archetype": wc_stats,
    }


# ---------------------------------------------------------------- 7-Model Extraction

def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _save_arch235_npz(out_npz, rows, h_last, h_mean, tok_lens, ok_flags, layer_idx, num_layers):
    hl = np.stack(h_last, axis=0)
    hm = np.stack(h_mean, axis=0)
    assert np.isfinite(hl).all() and np.isfinite(hm).all()
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        out_npz,
        last=pack(hl),
        mean=pack(hm),
        tok_len=np.array(tok_lens, dtype=np.int32),
        ok=np.array(ok_flags, dtype=bool),
        setting_ids=np.array([int(r["setting_id"]) for r in rows], dtype=np.int32),
        archetypes=np.array([r["archetype"] for r in rows]),
        layer_idx=np.int32(layer_idx),
        num_layers=np.int32(num_layers),
    )


def _extract_llamacpp(key, spec, rows, out_npz):
    port = spec["port"]
    srv = str(Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server")))
    if _port_open(port):
        raise RuntimeError(f"port {port} already serving before starting {key} llama-server!")
    cmd = [
        srv, "-m", spec["path"], "--port", str(port), "--host", "127.0.0.1",
        "--embeddings", "--pooling", "none", "--embd-normalize", "-1",
        "--no-webui", "-c", "2048", "-b", "2048", "-ub", "2048", "-ngl", "99",
    ]
    t0 = time.time()
    print(f"ARCH235 EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post(ep, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{ep}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    h_last, h_mean, tok_lens, ok_flags = [], [], [], []
    try:
        ready = False
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                raise RuntimeError(f"{key} llama-server exited early rc={proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)
        assert ready, f"{key} llama-server failed health check"

        for i, r in enumerate(rows):
            txt = r["text"].strip()
            toks = post("/tokenize", {"content": txt, "add_special": True})["tokens"]
            assert 2 <= len(toks) <= 512, f"{key} bad tok len {len(toks)} at {i}"
            emb = post("/embedding", {"content": txt})
            arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[0] == len(toks), f"{key} emb shape mismatch at {i}"
            h_last.append(arr[-1].copy())
            h_mean.append((arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0]).copy())
            tok_lens.append(len(toks))
            ok_flags.append(True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _save_arch235_npz(out_npz, rows, h_last, h_mean, tok_lens, ok_flags, -1, 24)
    print(f"ARCH235_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


def extract_model(key):
    assert key in PD.MODEL_SPECS, f"unknown key {key}"
    out_npz = PLATO_CACHE / f"arch235_{key}.npz"
    if out_npz.exists():
        print(f"ARCH235 EXTRACT {key}: already exists ({out_npz}), skipping", flush=True)
        return
    rows = load_named_arch_data()
    spec = PD.MODEL_SPECS[key]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp(key, spec, rows, out_npz)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"ARCH235 EXTRACT {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl_cnt = len(layers)
    tgt_idx = max(1, min(nl_cnt - 2, int(round(spec["depth_frac"] * (nl_cnt - 1)))))
    tgt_blk = layers[tgt_idx]
    print(f"ARCH235 EXTRACT {key}: L={nl_cnt}, hooking layer {tgt_idx} ({tgt_idx / (nl_cnt - 1):.1%})", flush=True)

    rec_buf = []
    patched = {}
    for blk in layers:
        cls = type(blk)
        if cls not in patched:
            orig = cls.__call__
            patched[cls] = orig

            def make_hook(orig_fn):
                def hooked(self, *a, **k):
                    out = orig_fn(self, *a, **k)
                    if self is tgt_blk:
                        h = out[0] if isinstance(out, tuple) else out
                        rec_buf.append(h[0])
                    return out
                return hooked

            cls.__call__ = make_hook(orig)

    h_last, h_mean, tok_lens, ok_flags = [], [], [], []
    try:
        for i, r in enumerate(rows):
            ids = PD._text_ids_mlx(tok, r["text"].strip())
            assert 2 <= len(ids) <= 512, f"{key} bad tok len {len(ids)} at {i}"
            rec_buf.clear()
            _ = model(mx.array([ids]))
            assert len(rec_buf) == 1
            h = rec_buf[0].astype(mx.float32)
            hl = h[-1]
            hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
            mx.eval(hl, hm)
            rec_buf.clear()
            h_last.append(np.array(hl, dtype=np.float32))
            h_mean.append(np.array(hm, dtype=np.float32))
            tok_lens.append(len(ids))
            ok_flags.append(True)
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()

    _save_arch235_npz(out_npz, rows, h_last, h_mean, tok_lens, ok_flags, tgt_idx, nl_cnt)
    print(f"ARCH235_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Analysis Primitives

def _auroc(y_true, scores):
    """Exact rank-based AUROC handling ties properly (0.5 credit on ties)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_r = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_r
        i = j
    sum_pos_ranks = float(ranks[y == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _map_stories(raw_vecs, rot_bundle, key, frame="to_c5"):
    mu, V, rms = rot_bundle["pca"][key]
    params = rot_bundle[frame][key]
    z = AR.apply_pca_params(raw_vecs, mu, V, rms)
    return PD.apply_procrustes_map(z, params)


def _ovr_dirs(Z, y_idx, n_cls=8):
    """Compute unit One-vs.-Rest difference-of-means directions (n_cls, D)."""
    U = np.zeros((n_cls, Z.shape[1]), dtype=np.float64)
    for c in range(n_cls):
        m = (y_idx == c)
        U[c] = AR.unit(Z[m].mean(axis=0) - Z[~m].mean(axis=0))
    return U


def _within_setting_perm_p(U_src, Z_te, y_te, sids_te, n_perm=1000, seed=235):
    """Exact within-setting label permutation p-value for each archetype's 2-fold OvR cosine.
    Shuffles the 8 archetype labels independently inside each test setting so setting structure is preserved."""
    rng = np.random.default_rng(seed)
    n_cls = U_src.shape[0]
    U_te = _ovr_dirs(Z_te, y_te, n_cls=n_cls)
    obs_cos = np.sum(U_src * U_te, axis=1)  # (n_cls,)

    uniq_s = sorted(set(int(s) for s in sids_te))
    groups = [np.where(sids_te == s)[0] for s in uniq_s]
    # Center Z_te globally so OvR mean difference is proportional to class mean
    Zc = Z_te - Z_te.mean(axis=0, keepdims=True)
    ge_cnt = np.zeros(n_cls, dtype=np.int64)
    for _ in range(n_perm):
        y_perm = np.empty_like(y_te)
        for idx in groups:
            y_perm[idx] = rng.permutation(y_te[idx])
        # Fast OvR means since each class has equal count and Zc sums to 0:
        # mean(c) - mean(~c) is strictly proportional to sum(Zc[y_perm == c])!
        for c in range(n_cls):
            v = Zc[y_perm == c].sum(axis=0)
            nv = float(np.linalg.norm(v))
            cos_p = float(np.dot(U_src[c], v) / max(1e-12, nv))
            if cos_p >= obs_cos[c]:
                ge_cnt[c] += 1
    p_vals = (ge_cnt + 1.0) / float(n_perm + 1)
    return obs_cos, p_vals


def _eval_transfer_splits(Z_src, Z_tgt, y_idx, sids, splits, Q_src=None, Q_tgt=None, tok_len_tgt=None):
    """Evaluate 20 setting-disjoint 2-fold splits from source -> target (ZERO target archetype labels used!).
    When Q_src / Q_tgt are given, double-projects out the nuisance subspace in BOTH source and target."""
    n_cls = len(ARCHETYPES)
    per_arch_auroc = {a: [] for a in ARCHETYPES}
    macro_auroc_splits = []
    top1_splits = []
    top2_splits = []
    within_set_top1_splits = []

    for fl in splits:
        all_scores = np.zeros((len(y_idx), n_cls), dtype=np.float64)
        for f in (0, 1):
            tr, te = (fl != f), (fl == f)
            Z_s_tr = Z_src[tr]
            Z_t_tr = Z_tgt[tr]
            Z_t_te = Z_tgt[te]

            # Optional fold-local length residualization on target (fit strictly on target train fold, zero labels!)
            if tok_len_tgt is not None:
                l_tr = tok_len_tgt[tr].astype(np.float64)
                l_te = tok_len_tgt[te].astype(np.float64)
                X_tr = np.column_stack([np.ones(len(l_tr)), l_tr, np.log(np.maximum(1.0, l_tr))])
                coef, _, _, _ = np.linalg.lstsq(X_tr, Z_t_tr, rcond=None)
                X_tr_l = np.column_stack([l_tr, np.log(np.maximum(1.0, l_tr))])
                X_te_l = np.column_stack([l_te, np.log(np.maximum(1.0, l_te))])
                Z_t_tr = Z_t_tr - X_tr_l @ coef[1:]
                Z_t_te = Z_t_te - X_te_l @ coef[1:]

            # Subtract unlabeled train-fold grand mean so 8-way argmax has zero global offset
            Z_t_te_c = Z_t_te - Z_t_tr.mean(axis=0, keepdims=True)
            Z_s_tr_c = Z_s_tr - Z_s_tr.mean(axis=0, keepdims=True)

            if Q_src is not None:
                Z_s_tr_c = V234._project_out_rows(Z_s_tr_c, Q_src)
            if Q_tgt is not None:
                Z_t_te_c = V234._project_out_rows(Z_t_te_c, Q_tgt)

            U_tr = _ovr_dirs(Z_s_tr_c, y_idx[tr], n_cls=n_cls)
            if Q_src is not None:
                U_tr = np.stack([V234._project_out(u, Q_src) for u in U_tr], axis=0)
            if Q_tgt is not None:
                U_tr = np.stack([V234._project_out(u, Q_tgt) for u in U_tr], axis=0)

            all_scores[te] = Z_t_te_c @ U_tr.T

        # Per-archetype OvR AUROC across all 320 out-of-fold predictions
        aucs = []
        for c, a in enumerate(ARCHETYPES):
            auc_c = _auroc((y_idx == c).astype(int), all_scores[:, c])
            per_arch_auroc[a].append(auc_c)
            aucs.append(auc_c)
        macro_auroc_splits.append(float(np.mean(aucs)))

        # 8-way Top-1 and Top-2 accuracy (single-story independent!)
        pred1 = np.argmax(all_scores, axis=1)
        top2 = np.argsort(all_scores, axis=1)[:, -2:]
        top1_splits.append(float(np.mean(pred1 == y_idx)))
        top2_splits.append(float(np.mean([y_idx[i] in top2[i] for i in range(len(y_idx))])))

        # Within-setting forced choice: for each setting, on each archetype direction c, does true c beat the other 7?
        ws_ok = []
        for s in sorted(set(int(x) for x in sids)):
            idx = np.where(sids == s)[0]
            sub_sc = all_scores[idx]  # (8, 8)
            sub_y = y_idx[idx]
            for c in range(n_cls):
                best_row = int(np.argmax(sub_sc[:, c]))
                ws_ok.append(1.0 if sub_y[best_row] == c else 0.0)
        within_set_top1_splits.append(float(np.mean(ws_ok)))

    # 2-fold direction cosine & within-setting label-permutation p-value on canonical split 0
    fl0 = splits[0]
    cos_folds, p_folds = [], []
    for f in (0, 1):
        tr, te = (fl0 != f), (fl0 == f)
        Zs_tr = Z_src[tr] - Z_src[tr].mean(axis=0, keepdims=True)
        Zt_te = Z_tgt[te] - Z_tgt[te].mean(axis=0, keepdims=True)
        if Q_src is not None:
            Zs_tr = V234._project_out_rows(Zs_tr, Q_src)
        if Q_tgt is not None:
            Zt_te = V234._project_out_rows(Zt_te, Q_tgt)
        U_tr = _ovr_dirs(Zs_tr, y_idx[tr], n_cls=n_cls)
        if Q_src is not None:
            U_tr = np.stack([V234._project_out(u, Q_src) for u in U_tr], axis=0)
        if Q_tgt is not None:
            U_tr = np.stack([V234._project_out(u, Q_tgt) for u in U_tr], axis=0)
        c_f, p_f = _within_setting_perm_p(U_tr, Zt_te, y_idx[te], sids[te], n_perm=1000, seed=2350 + f)
        cos_folds.append(c_f)
        p_folds.append(p_f)
    mean_cos = 0.5 * (cos_folds[0] + cos_folds[1])
    max_p = np.maximum(p_folds[0], p_folds[1])

    return {
        "per_arch_auroc_splits": per_arch_auroc,
        "macro_auroc_splits": macro_auroc_splits,
        "top1_splits": top1_splits,
        "top2_splits": top2_splits,
        "within_set_top1_splits": within_set_top1_splits,
        "cos_2fold": {a: round(float(mean_cos[c]), 4) for c, a in enumerate(ARCHETYPES)},
        "p_perm_max": {a: round(float(max_p[c]), 4) for c, a in enumerate(ARCHETYPES)},
    }


def _lexical_baselines_20splits(texts, y_idx, splits):
    """Run three strictly train-only lexical baselines on the exact same 20 setting-disjoint 2-fold splits:
      1. Unigram Binary BoW OvR difference-of-means (exact Table III baseline)
      2. Unigram+Bigram sublinear TF-IDF OvR centroid
      3. Unigram+Bigram sublinear TF-IDF L2-regularized Ridge OvR classifier (alpha=1.0)
    """
    n = len(texts)
    n_cls = len(ARCHETYPES)
    tok_lists = [re.findall(r"[a-z]+", t.lower()) for t in texts]
    uni_sets = [set(ts) for ts in tok_lists]
    uni_bi_lists = [ts + [f"{ts[j]}_{ts[j + 1]}" for j in range(len(ts) - 1)] for ts in tok_lists]

    out = {
        name: {"per_arch_auroc_splits": {a: [] for a in ARCHETYPES}, "macro_auroc_splits": [], "top1_splits": [], "top2_splits": []}
        for name in ("bow_unigram", "tfidf_centroid", "tfidf_ridge")
    }

    for fl in splits:
        sc_bow = np.zeros((n, n_cls), dtype=np.float64)
        sc_tfc = np.zeros((n, n_cls), dtype=np.float64)
        sc_tfr = np.zeros((n, n_cls), dtype=np.float64)

        for f in (0, 1):
            tr_idx = np.where(fl != f)[0]
            te_idx = np.where(fl == f)[0]
            y_tr = y_idx[tr_idx]

            # --- 1. Unigram Binary BoW (strictly train-only vocab) ---
            vocab_u = sorted({w for i in tr_idx for w in uni_sets[i]})
            w2i_u = {w: j for j, w in enumerate(vocab_u)}
            X_tr_u = np.zeros((len(tr_idx), len(vocab_u)), dtype=np.float64)
            for row_i, i in enumerate(tr_idx):
                for w in uni_sets[i]:
                    X_tr_u[row_i, w2i_u[w]] = 1.0
            X_te_u = np.zeros((len(te_idx), len(vocab_u)), dtype=np.float64)
            for row_i, i in enumerate(te_idx):
                for w in uni_sets[i]:
                    j = w2i_u.get(w)
                    if j is not None:
                        X_te_u[row_i, j] = 1.0
            W_bow = _ovr_dirs(X_tr_u, y_tr, n_cls=n_cls)
            X_te_uc = X_te_u - X_tr_u.mean(axis=0, keepdims=True)
            sc_bow[te_idx] = X_te_uc @ W_bow.T

            # --- 2 & 3. Unigram+Bigram sublinear TF-IDF (strictly train-only vocab & IDF) ---
            df = {}
            for i in tr_idx:
                for tok in set(uni_bi_lists[i]):
                    df[tok] = df.get(tok, 0) + 1
            vocab_tb = sorted(df.keys())
            w2i_tb = {w: j for j, w in enumerate(vocab_tb)}
            n_tr = float(len(tr_idx))
            idf = np.array([math.log((1.0 + n_tr) / (1.0 + df[w])) + 1.0 for w in vocab_tb], dtype=np.float64)

            def build_tfidf(idxs):
                X = np.zeros((len(idxs), len(vocab_tb)), dtype=np.float64)
                for row_i, i in enumerate(idxs):
                    cnt = {}
                    for tok in uni_bi_lists[i]:
                        j = w2i_tb.get(tok)
                        if j is not None:
                            cnt[j] = cnt.get(j, 0) + 1
                    for j, c in cnt.items():
                        X[row_i, j] = (1.0 + math.log(c)) * idf[j]
                norms = np.maximum(1e-12, np.linalg.norm(X, axis=1, keepdims=True))
                return X / norms

            X_tr_tf = build_tfidf(tr_idx)
            X_te_tf = build_tfidf(te_idx)

            # TF-IDF Centroid OvR
            W_tfc = _ovr_dirs(X_tr_tf, y_tr, n_cls=n_cls)
            sc_tfc[te_idx] = (X_te_tf - X_tr_tf.mean(axis=0, keepdims=True)) @ W_tfc.T

            # TF-IDF L2-Ridge OvR via dual kernel trick (n_tr = 160 << V, exact closed form!)
            Y_oh = np.full((len(tr_idx), n_cls), -1.0, dtype=np.float64)
            for row_i, c in enumerate(y_tr):
                Y_oh[row_i, c] = 1.0
            K_tr = X_tr_tf @ X_tr_tf.T
            alpha = 1.0
            # Include intercept via centering
            mu_x = X_tr_tf.mean(axis=0, keepdims=True)
            Xc_tr = X_tr_tf - mu_x
            Xc_te = X_te_tf - mu_x
            mu_y = Y_oh.mean(axis=0, keepdims=True)
            dual = np.linalg.solve(Xc_tr @ Xc_tr.T + alpha * np.eye(len(tr_idx)), Y_oh - mu_y)
            sc_tfr[te_idx] = (Xc_te @ Xc_tr.T) @ dual + mu_y

        for name, sc in (("bow_unigram", sc_bow), ("tfidf_centroid", sc_tfc), ("tfidf_ridge", sc_tfr)):
            aucs = []
            for c, a in enumerate(ARCHETYPES):
                ac = _auroc((y_idx == c).astype(int), sc[:, c])
                out[name]["per_arch_auroc_splits"][a].append(ac)
                aucs.append(ac)
            out[name]["macro_auroc_splits"].append(float(np.mean(aucs)))
            pred1 = np.argmax(sc, axis=1)
            top2 = np.argsort(sc, axis=1)[:, -2:]
            out[name]["top1_splits"].append(float(np.mean(pred1 == y_idx)))
            out[name]["top2_splits"].append(float(np.mean([y_idx[i] in top2[i] for i in range(n)])))

    return out


def _effective_rank_pr(U):
    """Participation ratio effective dimensionality of rows of U: (sum s_i^2)^2 / sum s_i^4."""
    s = np.linalg.svd(U, compute_uv=False)
    s2 = s ** 2
    pr = float((s2.sum() ** 2) / max(1e-12, float(np.sum(s2 ** 2))))
    return pr, [round(float(x), 4) for x in s]


def _summarize_block(res, lex_bow, lex_tfr):
    """Compact JSON summary of a transfer result block with paired split wins vs BoW and TF-IDF Ridge."""
    per_arch = {}
    for a in ARCHETYPES:
        vals = res["per_arch_auroc_splits"][a]
        b_vals = lex_bow["per_arch_auroc_splits"][a]
        r_vals = lex_tfr["per_arch_auroc_splits"][a]
        per_arch[a] = {
            "auroc_20split_mean": round(float(np.mean(vals)), 4),
            "auroc_20split_std": round(float(np.std(vals)), 4),
            "cos_2fold": res["cos_2fold"][a],
            "p_perm_max": res["p_perm_max"][a],
            "wins_over_bow_20": int(sum(x > y for x, y in zip(vals, b_vals))),
            "wins_over_tfidf_ridge_20": int(sum(x > y for x, y in zip(vals, r_vals))),
        }
    return {
        "macro_auroc_mean": round(float(np.mean(res["macro_auroc_splits"])), 4),
        "macro_auroc_std": round(float(np.std(res["macro_auroc_splits"])), 4),
        "macro_wins_over_bow_20": int(sum(x > y for x, y in zip(res["macro_auroc_splits"], lex_bow["macro_auroc_splits"]))),
        "macro_wins_over_tfidf_ridge_20": int(sum(x > y for x, y in zip(res["macro_auroc_splits"], lex_tfr["macro_auroc_splits"]))),
        "top1_8way_mean": round(float(np.mean(res["top1_splits"])), 4),
        "top1_8way_std": round(float(np.std(res["top1_splits"])), 4),
        "top1_wins_over_bow_20": int(sum(x > y for x, y in zip(res["top1_splits"], lex_bow["top1_splits"]))),
        "top1_wins_over_tfidf_ridge_20": int(sum(x > y for x, y in zip(res["top1_splits"], lex_tfr["top1_splits"]))),
        "top2_8way_mean": round(float(np.mean(res["top2_splits"])), 4),
        "within_setting_top1_mean": round(float(np.mean(res["within_set_top1_splits"])), 4),
        "per_archetype": per_arch,
    }


def analyze_7model():
    """Run complete 7-company Table III neutral-text transfer analysis on all 8 canonical named archetypes."""
    integrity = verify_dataset_integrity()
    rows = load_named_arch_data()
    texts = [r["text"].strip() for r in rows]
    arch2idx = {a: i for i, a in enumerate(ARCHETYPES)}
    y_idx = np.array([arch2idx[r["archetype"]] for r in rows], dtype=int)
    sids = np.array([int(r["setting_id"]) for r in rows], dtype=int)

    # 20 deterministic setting-disjoint 2-fold splits (20 train settings / 20 test settings!)
    uniq_sids = sorted(set(int(s) for s in sids))
    splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23500 + s_i)
        perm = rng.permutation(uniq_sids)
        f0 = set(int(x) for x in perm[:20])
        splits.append(np.array([0 if int(s) in f0 else 1 for s in sids], dtype=int))

    # 1. Lexical baselines on exact same 20 splits
    lex = _lexical_baselines_20splits(texts, y_idx, splits)
    lex_summary = {}
    for name in ("bow_unigram", "tfidf_centroid", "tfidf_ridge"):
        lex_summary[name] = {
            "macro_auroc_mean": round(float(np.mean(lex[name]["macro_auroc_splits"])), 4),
            "macro_auroc_std": round(float(np.std(lex[name]["macro_auroc_splits"])), 4),
            "top1_8way_mean": round(float(np.mean(lex[name]["top1_splits"])), 4),
            "top1_8way_std": round(float(np.std(lex[name]["top1_splits"])), 4),
            "top2_8way_mean": round(float(np.mean(lex[name]["top2_splits"])), 4),
            "per_archetype_auroc": {
                a: round(float(np.mean(lex[name]["per_arch_auroc_splits"][a])), 4) for a in ARCHETYPES
            },
        }

    # 2. Fit strictly neutral-text-only (`gut + proc + chat`) 256-dim Procrustes frame
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato

    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])

    Z_arch, tok_lens = {}, {}
    u_house_m, u_sent_m, u_outval_m = {}, {}, {}
    Q_val2_m, Q_val3_m = {}, {}
    U_arch_m = {}

    for k in PD.ALL_KEYS:
        d235 = AR._load_npz_dict(PLATO_CACHE / f"arch235_{k}.npz")
        Z_arch[k] = _map_stories(d235["mean"], rot, k, frame="to_c5")
        tok_lens[k] = d235["tok_len"].astype(np.int32)

        # Load HOUSE + VALENCE_2D directions in the EXACT same neutral-text C5 frame!
        a = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
        v = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")
        _, _, D_h = AR.map_diffs(a["done_mean"][house_idx], a["neg_mean"][house_idx], rot, k, frame="to_c5")
        _, _, D_v = AR.map_diffs(v["pos_mean"], v["neg_mean"], rot, k, frame="to_c5")
        groups = v["groups"]
        u_house_m[k] = AR.house_dir(D_h, house_cats)
        u_sent_m[k] = AR.unit(D_v[groups == "sentiment_gen"].mean(axis=0))
        u_outval_m[k] = AR.unit(D_v[groups == "outcome_valence"].mean(axis=0))
        Q_val2_m[k] = V234._orthonormal_basis([u_sent_m[k], u_outval_m[k]])
        Q_val3_m[k] = V234._orthonormal_basis([u_sent_m[k], u_outval_m[k], u_house_m[k]])

        Zc = Z_arch[k] - Z_arch[k].mean(axis=0, keepdims=True)
        U_arch_m[k] = _ovr_dirs(Zc, y_idx, n_cls=len(ARCHETYPES))

    # 3. C5 geometry: valence/HOUSE overlap per archetype, 8x8 cosine matrix, & effective dimensionality!
    u_house_c5 = AR.unit(np.mean([u_house_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_sent_c5 = AR.unit(np.mean([u_sent_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_outval_c5 = AR.unit(np.mean([u_outval_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    Q2_c5 = V234._orthonormal_basis([u_sent_c5, u_outval_c5])
    Q3_c5 = V234._orthonormal_basis([u_sent_c5, u_outval_c5, u_house_c5])

    U_c5_raw = np.stack([
        AR.unit(np.mean([U_arch_m[k][c] for k in PD.CONSENSUS_KEYS], axis=0))
        for c in range(len(ARCHETYPES))
    ], axis=0)
    U_c5_p3 = np.stack([V234._project_out(U_c5_raw[c], Q3_c5) for c in range(len(ARCHETYPES))], axis=0)

    pr_raw, sv_raw = _effective_rank_pr(U_c5_raw)
    pr_p3, sv_p3 = _effective_rank_pr(U_c5_p3)
    C_raw = U_c5_raw @ U_c5_raw.T
    C_p3 = U_c5_p3 @ U_c5_p3.T
    off_mask = ~np.eye(len(ARCHETYPES), dtype=bool)

    per_arch_geom = {}
    for c, a in enumerate(ARCHETYPES):
        u = U_c5_raw[c]
        per_arch_geom[a] = {
            "cos_sent": round(float(np.dot(u, u_sent_c5)), 4),
            "cos_outval": round(float(np.dot(u, u_outval_c5)), 4),
            "cos_house": round(float(np.dot(u, u_house_c5)), 4),
            "sqnorm_in_val2_span": round(float(np.sum((Q2_c5.T @ u) ** 2)), 4),
            "sqnorm_in_val3_span": round(float(np.sum((Q3_c5.T @ u) ** 2)), 4),
        }

    c5_geom = {
        "effective_rank_pr_raw": round(pr_raw, 3),
        "singular_values_raw": sv_raw,
        "mean_abs_offdiag_cos_raw": round(float(np.mean(np.abs(C_raw[off_mask]))), 4),
        "max_pos_offdiag_cos_raw": round(float(np.max(C_raw[off_mask])), 4),
        "effective_rank_pr_perp_val3": round(pr_p3, 3),
        "singular_values_perp_val3": sv_p3,
        "mean_abs_offdiag_cos_perp_val3": round(float(np.mean(np.abs(C_p3[off_mask]))), 4),
        "per_archetype_valence_house_overlap": per_arch_geom,
        "pairwise_cos_raw": {
            a: {b: round(float(C_raw[i, j]), 3) for j, b in enumerate(ARCHETYPES)}
            for i, a in enumerate(ARCHETYPES)
        },
        "pairwise_cos_perp_val3": {
            a: {b: round(float(C_p3[i, j]), 3) for j, b in enumerate(ARCHETYPES)}
            for i, a in enumerate(ARCHETYPES)
        },
    }

    # 4. Zero-shot neutral-text transfer: Gemma 4 E4B -> all 6 rival companies!
    src_k = "gemma4_e4b"
    targets = [k for k in PD.ALL_KEYS if k != src_k]
    gemma_to_6 = {}
    for tk in targets:
        r_raw = _eval_transfer_splits(Z_arch[src_k], Z_arch[tk], y_idx, sids, splits)
        r_pv2 = _eval_transfer_splits(Z_arch[src_k], Z_arch[tk], y_idx, sids, splits, Q_src=Q_val2_m[src_k], Q_tgt=Q_val2_m[tk])
        r_pv3 = _eval_transfer_splits(Z_arch[src_k], Z_arch[tk], y_idx, sids, splits, Q_src=Q_val3_m[src_k], Q_tgt=Q_val3_m[tk])
        r_lr = _eval_transfer_splits(Z_arch[src_k], Z_arch[tk], y_idx, sids, splits, tok_len_tgt=tok_lens[tk])
        gemma_to_6[tk] = {
            "raw": _summarize_block(r_raw, lex["bow_unigram"], lex["tfidf_ridge"]),
            "perp_val2_dbl": _summarize_block(r_pv2, lex["bow_unigram"], lex["tfidf_ridge"]),
            "perp_val3_dbl": _summarize_block(r_pv3, lex["bow_unigram"], lex["tfidf_ridge"]),
            "lenres_fold_local": {
                "macro_auroc_mean": round(float(np.mean(r_lr["macro_auroc_splits"])), 4),
                "top1_8way_mean": round(float(np.mean(r_lr["top1_splits"])), 4),
            },
        }

    # 5. 5-family C5 consensus average -> held-out Phi-4-mini & OLMo 3 7B!
    Z_c5 = np.mean([Z_arch[k] for k in PD.CONSENSUS_KEYS], axis=0)
    c5_to_heldout = {}
    for tk in ("phi4_mini", "olmo3_7b"):
        rc_raw = _eval_transfer_splits(Z_c5, Z_arch[tk], y_idx, sids, splits)
        rc_pv3 = _eval_transfer_splits(Z_c5, Z_arch[tk], y_idx, sids, splits, Q_src=Q3_c5, Q_tgt=Q_val3_m[tk])
        c5_to_heldout[tk] = {
            "raw": _summarize_block(rc_raw, lex["bow_unigram"], lex["tfidf_ridge"]),
            "perp_val3_dbl": _summarize_block(rc_pv3, lex["bow_unigram"], lex["tfidf_ridge"]),
        }

    report = {
        "dataset_integrity": integrity,
        "lexical_baselines_20splits": lex_summary,
        "c5_geometry": c5_geom,
        "zero_shot_gemma_to_6": gemma_to_6,
        "c5_consensus_to_heldout": c5_to_heldout,
    }
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, OUT_JSON)
    print(f"SAVED {OUT_JSON}", flush=True)
    return report


# ---------------------------------------------------------------- Zero-GPU Unit Test

def self_test():
    rng = np.random.default_rng(235)
    # 1. Exact AUROC sanity
    assert abs(_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
    assert abs(_auroc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12

    # 2. Pure-noise calibration (320 = 40 settings x 8 classes)
    sids = np.repeat(np.arange(1, 41), 8)
    y = np.tile(np.arange(8), 40)
    splits = [np.array([0 if s <= 20 else 1 for s in sids])]
    Z_n1 = rng.normal(size=(320, 64))
    Z_n2 = rng.normal(size=(320, 64))
    rn = _eval_transfer_splits(Z_n1, Z_n2, y, sids, splits)
    assert 0.35 <= rn["macro_auroc_splits"][0] <= 0.65, rn["macro_auroc_splits"]
    assert 0.03 <= rn["top1_splits"][0] <= 0.28, rn["top1_splits"]

    # 3. Synthetic 8-class signal + orthogonal valence nuisance projection!
    proto = rng.normal(size=(8, 64))
    v_nuis = AR.unit(rng.normal(size=64))
    Q_nuis = V234._orthonormal_basis([v_nuis])
    # Embed true orthogonal class prototypes + strong shared valence bias
    Z_s = proto[y] + 3.0 * ((y < 4).astype(float)[:, None] - 0.5) * v_nuis[None, :] + 0.25 * rng.normal(size=(320, 64))
    Z_t = proto[y] + 3.0 * ((y < 4).astype(float)[:, None] - 0.5) * v_nuis[None, :] + 0.25 * rng.normal(size=(320, 64))
    rs = _eval_transfer_splits(Z_s, Z_t, y, sids, splits, Q_src=Q_nuis, Q_tgt=Q_nuis)
    assert rs["macro_auroc_splits"][0] > 0.99 and rs["top1_splits"][0] > 0.95, rs
    assert all(p < 0.01 for p in rs["p_perm_max"].values()), rs["p_perm_max"]

    # 4. Pure-valence fake signal collapses when Q_nuis is projected out!
    Z_fake = 3.0 * ((y < 4).astype(float)[:, None] - 0.5) * v_nuis[None, :] + 0.5 * rng.normal(size=(320, 64))
    rf = _eval_transfer_splits(Z_fake, Z_fake.copy(), y, sids, splits, Q_src=Q_nuis, Q_tgt=Q_nuis)
    assert rf["macro_auroc_splits"][0] < 0.65, rf["macro_auroc_splits"]

    # 5. Effective rank sanity
    pr_rank1, _ = _effective_rank_pr(np.tile(v_nuis, (8, 1)))
    assert abs(pr_rank1 - 1.0) < 1e-6
    pr_eye, _ = _effective_rank_pr(np.eye(7))
    assert abs(pr_eye - 7.0) < 1e-6

    # 6. Lexical baseline zero-leakage check on random tokens
    fake_texts = [" ".join(f"w{rng.integers(0, 80)}" for _ in range(30)) for _ in range(320)]
    lx = _lexical_baselines_20splits(fake_texts, y, splits)
    for k in ("bow_unigram", "tfidf_centroid", "tfidf_ridge"):
        assert 0.35 <= lx[k]["macro_auroc_splits"][0] <= 0.65, (k, lx[k]["macro_auroc_splits"])

    # 7. Real dataset integrity check if present
    if DATA_JSON.exists():
        integ = verify_dataset_integrity()
        assert integ["n_stories"] == 320 and integ["ban_list_hits"] == 0 and integ["setting_overlap"] == 0

    print("NAMED_ARCH235_TEST_OK")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        self_test()
    elif cmd == "extract":
        keys = sys.argv[2:] if len(sys.argv) > 2 else list(PD.ALL_KEYS)
        for k in keys:
            extract_model(k)
    elif cmd == "analyze":
        analyze_7model()
    elif cmd == "all":
        self_test()
        for k in PD.ALL_KEYS:
            extract_model(k)
        analyze_7model()
        print("ALL_EXP235_DONE_OK")
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
