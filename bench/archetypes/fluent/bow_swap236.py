"""Experiment 236: Exact Identical-Bag-of-Words Swap Audit (`bow_swap_60`) & Held-Out Valence Residual Audit.

Directly settles the two sharp methodological questions raised on Experiment 234 (`valence_ctrl234`):
  1. **Exact Identical-Bag-of-Words Care-vs.-Neglect Swaps (`bow_swap_60`, 60 setting-disjoint pairs):**
     Every pair `(pos, neg)` has the EXACT same word multiset (`Counter(tokens(pos)) == Counter(tokens(neg))`),
     so unigram Bag-of-Words accuracy is mathematically `50.0%` by construction (zero lexical cue, zero token-count
     bias, zero VADER word-level difference). We evaluate zero-shot neutral-text Procrustes transfer (`gut+proc+chat`)
     from Gemma 4 E4B into all 6 rival families for:
       - Pure `SENTIMENT` -> `bow_swap_60`
       - Raw `HOUSE` -> `bow_swap_60`
       - Double-projected `HOUSE ⊥ VALENCE_2D` -> `bow_swap_60` (both source Gemma 4 AND target projected!)
       - Strictly disjoint 2-fold CV on `bow_swap_60` (raw & double-projected `⊥ VALENCE_2D`)
       - Unigram BoW (exact 50.0%), Bigram BoW (20 splits), and `HOUSE`-trained Unigram+Bigram BoW -> `bow_swap_60` & `VM_60`!
  2. **Held-Out Valence Residual Audit & Paired-Wins Check on Experiment 234:**
     - Computes exact paired-split wins (`/20`) of double-projected `HOUSE ⊥ VALENCE_2D` against unigram BoW on `HOUSE`!
     - Computes like-for-like cross-dataset unigram BoW (`HOUSE` -> `VM_60` and `HOUSE` -> `bow_swap_60`)!
     - Runs a **held-out valence probe check on the residual**: fits `span(u_sent, u_outval)` strictly on Fold A
       of `sentiment_gen`/`outcome_valence`, projects `HOUSE` orthogonal to Fold-A valence, and tests whether
       `HOUSE ⊥ VALENCE_2D(FoldA)` still classifies **unseen Fold-B `sentiment_gen` and `outcome_valence`**!
       Also tests projecting out a richer **5D multi-domain valence subspace** (`4` `sentiment_gen` domains + `outcome_valence`).

Usage:
  python fluent/bow_swap236.py test
  python fluent/bow_swap236.py extract [KEY...]
  python fluent/bow_swap236.py analyze
  python fluent/bow_swap236.py all
"""
from collections import Counter
import gc
import json
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
DATA_JSON = ARCH / "selfshape" / "bow_swap236_data.json"
OUT_JSON = PLATO_LOCAL / "bow_swap236.json"

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def _bigrams(text):
    ws = _words(text)
    return [f"{ws[i]}_{ws[i + 1]}" for i in range(len(ws) - 1)]


def load_bow_swap_data():
    d = json.loads(DATA_JSON.read_text())
    rows = d["bow_swap_60"]
    assert len(rows) == 60, f"Expected 60 pairs, got {len(rows)}"
    return rows


def verify_dataset_integrity():
    """Verify 60 unique settings, exact identical unigram multiset on 60/60 pairs, and pos != neg."""
    rows = load_bow_swap_data()
    settings = [r["setting"].strip().lower() for r in rows]
    assert len(settings) == len(set(settings)) == 60, "Duplicate settings in bow_swap_60!"
    existing = {p["setting"].strip().lower() for p in AR.load_arch_pairs() if p.get("setting")}
    _, _, vm = V234.load_valence_data()
    existing |= {r["setting"].strip().lower() for r in vm if r.get("setting")}
    overlap = existing & set(settings)
    assert not overlap, f"Setting overlap with existing benchmarks: {overlap}"

    wcs = []
    for r in rows:
        pos, neg = r["pos"].strip(), r["neg"].strip()
        assert pos != neg, f"Identical text in {r['id']}"
        cp, cn = Counter(_words(pos)), Counter(_words(neg))
        assert cp == cn, f"Unigram multiset mismatch in {r['id']}: pos-neg={cp - cn}, neg-pos={cn - cp}"
        wc = len(pos.split())
        assert 16 <= wc <= 50, f"Word count {wc} out of range in {r['id']}"
        wcs.append(wc)

    return {
        "n_pairs": len(rows),
        "identical_unigram_multiset_count": len(rows),
        "setting_overlap": 0,
        "mean_words": round(float(np.mean(wcs)), 2),
        "min_words": int(min(wcs)),
        "max_words": int(max(wcs)),
    }


# ---------------------------------------------------------------- 7-Model Extraction

def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _save_bs_npz(out_npz, rows, pl, pm, nl, nm, p_lens, n_lens, ok_flags, layer_idx, num_layers):
    pl, pm = np.stack(pl, axis=0), np.stack(pm, axis=0)
    nl, nm = np.stack(nl, axis=0), np.stack(nm, axis=0)
    assert np.isfinite(pl).all() and np.isfinite(pm).all() and np.isfinite(nl).all() and np.isfinite(nm).all()
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        out_npz,
        pos_last=pack(pl), pos_mean=pack(pm),
        neg_last=pack(nl), neg_mean=pack(nm),
        pos_len=np.array(p_lens, dtype=np.int32),
        neg_len=np.array(n_lens, dtype=np.int32),
        ok=np.array(ok_flags, dtype=bool),
        ids=np.array([r["id"] for r in rows]),
        swap_types=np.array([r.get("swap_type", "object_swap") for r in rows]),
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
    print(f"BS236 EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post(ep, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{ep}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    pl, pm, nl, nm, p_lens, n_lens, ok_flags = [], [], [], [], [], [], []
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

        def embed_one(text):
            toks = post("/tokenize", {"content": text, "add_special": True})["tokens"]
            assert 2 <= len(toks) <= 512
            emb = post("/embedding", {"content": text})
            arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[0] == len(toks)
            return arr[-1].copy(), (arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0]).copy(), len(toks)

        for r in rows:
            p_l, p_m, p_ln = embed_one(r["pos"].strip())
            n_l, n_m, n_ln = embed_one(r["neg"].strip())
            pl.append(p_l); pm.append(p_m); nl.append(n_l); nm.append(n_m)
            p_lens.append(p_ln); n_lens.append(n_ln); ok_flags.append(True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _save_bs_npz(out_npz, rows, pl, pm, nl, nm, p_lens, n_lens, ok_flags, -1, 24)
    print(f"BS236_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


def extract_model(key):
    assert key in PD.MODEL_SPECS, f"unknown key {key}"
    out_npz = PLATO_CACHE / f"bs236_{key}.npz"
    if out_npz.exists():
        print(f"BS236 EXTRACT {key}: already exists ({out_npz}), skipping", flush=True)
        return
    rows = load_bow_swap_data()
    spec = PD.MODEL_SPECS[key]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp(key, spec, rows, out_npz)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"BS236 EXTRACT {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl_cnt = len(layers)
    tgt_idx = max(1, min(nl_cnt - 2, int(round(spec["depth_frac"] * (nl_cnt - 1)))))
    tgt_blk = layers[tgt_idx]

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

    pl, pm, nl, nm, p_lens, n_lens, ok_flags = [], [], [], [], [], [], []
    try:
        def embed_one(text):
            ids = PD._text_ids_mlx(tok, text)
            assert 2 <= len(ids) <= 512
            rec_buf.clear()
            _ = model(mx.array([ids]))
            assert len(rec_buf) == 1
            h = rec_buf[0].astype(mx.float32)
            hl = h[-1]
            hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
            mx.eval(hl, hm)
            rec_buf.clear()
            return np.array(hl, dtype=np.float32), np.array(hm, dtype=np.float32), len(ids)

        for r in rows:
            p_l, p_m, p_ln = embed_one(r["pos"].strip())
            n_l, n_m, n_ln = embed_one(r["neg"].strip())
            pl.append(p_l); pm.append(p_m); nl.append(n_l); nm.append(n_m)
            p_lens.append(p_ln); n_lens.append(n_ln); ok_flags.append(True)
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()

    _save_bs_npz(out_npz, rows, pl, pm, nl, nm, p_lens, n_lens, ok_flags, tgt_idx, nl_cnt)
    print(f"BS236_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Lexical Helpers

def _pair_bow_fit_score(train_pairs, eval_pairs, pos_k_tr, neg_k_tr, pos_k_ev, neg_k_ev, feat_fn):
    """Difference-of-means binary feature classifier fit on train_pairs, scored on eval_pairs (0.5 on ties)."""
    w = {}
    for r in train_pairs:
        fp, fn = set(feat_fn(r[pos_k_tr])), set(feat_fn(r[neg_k_tr]))
        for t in fp - fn:
            w[t] = w.get(t, 0.0) + 1.0
        for t in fn - fp:
            w[t] = w.get(t, 0.0) - 1.0
    ok = []
    for r in eval_pairs:
        fp, fn = set(feat_fn(r[pos_k_ev])), set(feat_fn(r[neg_k_ev]))
        sc = sum(w.get(t, 0.0) for t in fp) - sum(w.get(t, 0.0) for t in fn)
        ok.append(1.0 if sc > 0 else (0.5 if sc == 0 else 0.0))
    return float(np.mean(ok))


def _bow_2fold_pairs(rows, pos_k, neg_k, feat_fn, splits):
    accs = []
    for fl in splits:
        ok = []
        for f in (0, 1):
            tr = [rows[i] for i in range(len(rows)) if fl[i] != f]
            te = [rows[i] for i in range(len(rows)) if fl[i] == f]
            ok.append(_pair_bow_fit_score(tr, te, pos_k, neg_k, pos_k, neg_k, feat_fn))
        accs.append(float(np.mean(ok)))
    return accs


def analyze_7model():
    """Run complete 7-company analysis on `bow_swap_60`, held-out valence residual erasure,
    multi-domain 5D valence projection, and exact paired-wins counts against unigram/bigram BoW."""
    integrity = verify_dataset_integrity()
    bs_rows = load_bow_swap_data()
    sg_rows, ov_rows, vm_rows = V234.load_valence_data()

    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_pairs = [pairs_arch[i] for i in house_idx]
    house_cats = np.array([p["cat"] for p in house_pairs])

    # Exact 20 deterministic splits for bow_swap_60 (all 60 settings unique)
    bs_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23600 + s_i)
        perm = rng.permutation(len(bs_rows))
        fl = np.zeros(len(bs_rows), dtype=int)
        fl[perm[30:]] = 1
        bs_splits.append(fl)

    # Exact same 20 setting-disjoint splits for HOUSE as audit233_mac / valence_ctrl234!
    h_uniq_sets = sorted({p["setting"].strip().lower() for p in house_pairs})
    h_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23300 + s_i)
        perm = rng.permutation(len(h_uniq_sets))
        f0 = {h_uniq_sets[j] for j in perm[:len(h_uniq_sets) // 2]}
        h_splits.append(np.array([0 if p["setting"].strip().lower() in f0 else 1 for p in house_pairs], dtype=int))

    # Lexical baselines
    bow_house_20 = _bow_2fold_pairs(house_pairs, "done", "neglected", _words, h_splits)
    bow_bs_uni_20 = _bow_2fold_pairs(bs_rows, "pos", "neg", _words, bs_splits)
    bow_bs_bi_20 = _bow_2fold_pairs(bs_rows, "pos", "neg", _bigrams, bs_splits)
    bow_house_to_vm_uni = _pair_bow_fit_score(house_pairs, vm_rows, "done", "neglected", "integrated", "neglected", _words)
    bow_house_to_bs_uni = _pair_bow_fit_score(house_pairs, bs_rows, "done", "neglected", "pos", "neg", _words)
    bow_house_to_bs_bi = _pair_bow_fit_score(house_pairs, bs_rows, "done", "neglected", "pos", "neg", _bigrams)
    assert abs(np.mean(bow_bs_uni_20) - 0.5) < 1e-12 and abs(bow_house_to_bs_uni - 0.5) < 1e-12

    # Fit strictly neutral-text-only (`gut + proc + chat`) 256-dim Procrustes frame
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato

    Z_house_D, Z_sg_D, Z_ov_D, Z_bs_D = {}, {}, {}, {}
    u_house_m, u_sent_m, u_outval_m, u_bs_m = {}, {}, {}, {}
    Q_val2_m, Q_val5_m = {}, {}
    sg_domains = sorted({r["domain"] for r in sg_rows})
    sg_dom_arr = np.array([r["domain"] for r in sg_rows])

    for k in PD.ALL_KEYS:
        a = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
        v = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")
        b = AR._load_npz_dict(PLATO_CACHE / f"bs236_{k}.npz")

        _, _, Z_house_D[k] = AR.map_diffs(a["done_mean"][house_idx], a["neg_mean"][house_idx], rot, k, frame="to_c5")
        _, _, D_v = AR.map_diffs(v["pos_mean"], v["neg_mean"], rot, k, frame="to_c5")
        _, _, Z_bs_D[k] = AR.map_diffs(b["pos_mean"], b["neg_mean"], rot, k, frame="to_c5")

        groups = v["groups"]
        Z_sg_D[k] = D_v[groups == "sentiment_gen"]
        Z_ov_D[k] = D_v[groups == "outcome_valence"]

        u_house_m[k] = AR.house_dir(Z_house_D[k], house_cats)
        u_sent_m[k] = AR.unit(Z_sg_D[k].mean(axis=0))
        u_outval_m[k] = AR.unit(Z_ov_D[k].mean(axis=0))
        u_bs_m[k] = AR.unit(Z_bs_D[k].mean(axis=0))
        Q_val2_m[k] = V234._orthonormal_basis([u_sent_m[k], u_outval_m[k]])

        # Richer 5D multi-domain valence subspace: 4 sentiment domains + outcome_valence!
        dom_vecs = [AR.unit(Z_sg_D[k][sg_dom_arr == d].mean(axis=0)) for d in sg_domains] + [u_outval_m[k]]
        Q_val5_m[k] = V234._orthonormal_basis(dom_vecs)

    # 1. C5 geometry on bow_swap_60 + 5D valence span
    u_house_c5 = AR.unit(np.mean([u_house_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_sent_c5 = AR.unit(np.mean([u_sent_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_outval_c5 = AR.unit(np.mean([u_outval_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_bs_c5 = AR.unit(np.mean([u_bs_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    Q2_c5 = V234._orthonormal_basis([u_sent_c5, u_outval_c5])
    dom_vecs_c5 = [
        AR.unit(np.mean([AR.unit(Z_sg_D[k][sg_dom_arr == d].mean(axis=0)) for k in PD.CONSENSUS_KEYS], axis=0))
        for d in sg_domains
    ] + [u_outval_c5]
    Q5_c5 = V234._orthonormal_basis(dom_vecs_c5)
    u_house_p2_c5 = V234._project_out(u_house_c5, Q2_c5)
    u_house_p5_c5 = V234._project_out(u_house_c5, Q5_c5)

    c5_geom = {
        "cos_bs_house": round(float(np.dot(u_bs_c5, u_house_c5)), 4),
        "cos_bs_house_perp_val2": round(float(np.dot(u_bs_c5, u_house_p2_c5)), 4),
        "cos_bs_house_perp_val5": round(float(np.dot(u_bs_c5, u_house_p5_c5)), 4),
        "cos_bs_sent": round(float(np.dot(u_bs_c5, u_sent_c5)), 4),
        "cos_bs_outval": round(float(np.dot(u_bs_c5, u_outval_c5)), 4),
        "bs_sqnorm_in_val2_span": round(float(np.sum((Q2_c5.T @ u_bs_c5) ** 2)), 4),
        "bs_sqnorm_in_val5_span": round(float(np.sum((Q5_c5.T @ u_bs_c5) ** 2)), 4),
        "house_sqnorm_in_val2_span": round(float(np.sum((Q2_c5.T @ u_house_c5) ** 2)), 4),
        "house_sqnorm_in_val5_span": round(float(np.sum((Q5_c5.T @ u_house_c5) ** 2)), 4),
        "n_sg_domains_in_val5": len(sg_domains),
        "sg_domains": sg_domains,
    }

    # 2. Held-out valence erasure audit on the residual (fit Q_val2 strictly on even pairs, test on odd pairs & vice versa!)
    heldout_val_erasure = {}
    for k in PD.ALL_KEYS:
        sg_accs, ov_accs, sg_cos_list, ov_cos_list = [], [], [], []
        sg_accs_5d, ov_accs_5d = [], []
        for f in (0, 1):
            tr_sg = np.arange(len(sg_rows)) % 2 != f
            te_sg = ~tr_sg
            tr_ov = np.arange(len(ov_rows)) % 2 != f
            te_ov = ~tr_ov
            Q2_tr = V234._orthonormal_basis([AR.unit(Z_sg_D[k][tr_sg].mean(0)), AR.unit(Z_ov_D[k][tr_ov].mean(0))])
            uh_p2_tr = V234._project_out(u_house_m[k], Q2_tr)
            sg_accs.append(float(np.mean((Z_sg_D[k][te_sg] @ uh_p2_tr) > 0)))
            ov_accs.append(float(np.mean((Z_ov_D[k][te_ov] @ uh_p2_tr) > 0)))
            sg_cos_list.append(float(np.dot(uh_p2_tr, AR.unit(Z_sg_D[k][te_sg].mean(0)))))
            ov_cos_list.append(float(np.dot(uh_p2_tr, AR.unit(Z_ov_D[k][te_ov].mean(0)))))

            # 5D held-out check
            d_vecs_tr = [AR.unit(Z_sg_D[k][tr_sg & (sg_dom_arr == d)].mean(0)) for d in sg_domains] + [AR.unit(Z_ov_D[k][tr_ov].mean(0))]
            Q5_tr = V234._orthonormal_basis(d_vecs_tr)
            uh_p5_tr = V234._project_out(u_house_m[k], Q5_tr)
            sg_accs_5d.append(float(np.mean((Z_sg_D[k][te_sg] @ uh_p5_tr) > 0)))
            ov_accs_5d.append(float(np.mean((Z_ov_D[k][te_ov] @ uh_p5_tr) > 0)))
        heldout_val_erasure[k] = {
            "house_perp_val2_on_heldout_sg_acc": round(float(np.mean(sg_accs)), 4),
            "house_perp_val2_on_heldout_sg_cos": round(float(np.mean(sg_cos_list)), 4),
            "house_perp_val2_on_heldout_ov_acc": round(float(np.mean(ov_accs)), 4),
            "house_perp_val2_on_heldout_ov_cos": round(float(np.mean(ov_cos_list)), 4),
            "house_perp_val5_on_heldout_sg_acc": round(float(np.mean(sg_accs_5d)), 4),
            "house_perp_val5_on_heldout_ov_acc": round(float(np.mean(ov_accs_5d)), 4),
        }

    # 3. Gemma 4 -> 6 rival companies: bow_swap_60 + HOUSE_perp_val2 paired-wins vs BoW!
    src_k = "gemma4_e4b"
    targets = [k for k in PD.ALL_KEYS if k != src_k]
    u_house_src = u_house_m[src_k]
    u_sent_src = u_sent_m[src_k]
    u_outval_src = u_outval_m[src_k]
    u_house_p2_src = V234._project_out(u_house_src, Q_val2_m[src_k])
    u_house_p5_src = V234._project_out(u_house_src, Q_val5_m[src_k])

    transfer_bs = {}
    house_perp_vs_bow = {}
    for tk in targets:
        Qt2 = Q_val2_m[tk]
        Qt5 = Q_val5_m[tk]
        D_bs_t = Z_bs_D[tk]
        D_bs_t_p2 = V234._project_out_rows(D_bs_t, Qt2)
        D_bs_t_p5 = V234._project_out_rows(D_bs_t, Qt5)

        u_hp2_dbl = V234._project_out(u_house_p2_src, Qt2)
        u_hp5_dbl = V234._project_out(u_house_p5_src, Qt5)

        c_s, p_s = V234._centered_signflip_p(u_sent_src, D_bs_t, seed=2361)
        c_o, p_o = V234._centered_signflip_p(u_outval_src, D_bs_t, seed=2362)
        c_h, p_h = V234._centered_signflip_p(u_house_src, D_bs_t, seed=2363)
        c_hp2, p_hp2 = V234._centered_signflip_p(u_hp2_dbl, D_bs_t_p2, seed=2364)
        c_hp5, p_hp5 = V234._centered_signflip_p(u_hp5_dbl, D_bs_t_p5, seed=2365)

        # 20-split strictly disjoint 2-fold CV on bow_swap_60 (Gemma 4 -> target tk!)
        bs2_raw, bs2_p2, bs2_p5 = [], [], []
        for fl in bs_splits:
            ok_r, ok_2, ok_5 = [], [], []
            for f in (0, 1):
                tr, te = (fl != f), (fl == f)
                u_tr = AR.unit(Z_bs_D[src_k][tr].mean(axis=0))
                u_tr_2 = V234._project_out(V234._project_out(u_tr, Q_val2_m[src_k]), Qt2)
                u_tr_5 = V234._project_out(V234._project_out(u_tr, Q_val5_m[src_k]), Qt5)
                ok_r.extend((D_bs_t[te] @ u_tr > 0).tolist())
                ok_2.extend((D_bs_t_p2[te] @ u_tr_2 > 0).tolist())
                ok_5.extend((D_bs_t_p5[te] @ u_tr_5 > 0).tolist())
            bs2_raw.append(float(np.mean(ok_r)))
            bs2_p2.append(float(np.mean(ok_2)))
            bs2_p5.append(float(np.mean(ok_5)))

        fl0 = bs_splits[0]
        bc0, bp0 = V234._centered_signflip_p(AR.unit(Z_bs_D[src_k][fl0 == 0].mean(0)), D_bs_t[fl0 == 1], seed=2366)
        bc1, bp1 = V234._centered_signflip_p(AR.unit(Z_bs_D[src_k][fl0 == 1].mean(0)), D_bs_t[fl0 == 0], seed=2367)
        u0_2 = V234._project_out(V234._project_out(AR.unit(Z_bs_D[src_k][fl0 == 0].mean(0)), Q_val2_m[src_k]), Qt2)
        u1_2 = V234._project_out(V234._project_out(AR.unit(Z_bs_D[src_k][fl0 == 1].mean(0)), Q_val2_m[src_k]), Qt2)
        bc0_2, bp0_2 = V234._centered_signflip_p(u0_2, D_bs_t_p2[fl0 == 1], seed=2368)
        bc1_2, bp1_2 = V234._centered_signflip_p(u1_2, D_bs_t_p2[fl0 == 0], seed=2369)

        transfer_bs[tk] = {
            "zs_sent_to_bs_acc": round(float(np.mean((D_bs_t @ u_sent_src) > 0)), 4),
            "zs_sent_to_bs_cos": round(c_s, 4),
            "zs_sent_to_bs_p": round(p_s, 4),
            "zs_outval_to_bs_acc": round(float(np.mean((D_bs_t @ u_outval_src) > 0)), 4),
            "zs_outval_to_bs_cos": round(c_o, 4),
            "zs_outval_to_bs_p": round(p_o, 4),
            "zs_house_to_bs_acc": round(float(np.mean((D_bs_t @ u_house_src) > 0)), 4),
            "zs_house_to_bs_cos": round(c_h, 4),
            "zs_house_to_bs_p": round(p_h, 4),
            "zs_house_perp_val2_dbl_to_bs_acc": round(float(np.mean((D_bs_t_p2 @ u_hp2_dbl) > 0)), 4),
            "zs_house_perp_val2_dbl_to_bs_cos": round(c_hp2, 4),
            "zs_house_perp_val2_dbl_to_bs_p": round(p_hp2, 4),
            "zs_house_perp_val5_dbl_to_bs_acc": round(float(np.mean((D_bs_t_p5 @ u_hp5_dbl) > 0)), 4),
            "zs_house_perp_val5_dbl_to_bs_cos": round(c_hp5, 4),
            "zs_house_perp_val5_dbl_to_bs_p": round(p_hp5, 4),
            "bs_2fold_raw_20split_mean": round(float(np.mean(bs2_raw)), 4),
            "bs_2fold_raw_20split_std": round(float(np.std(bs2_raw)), 4),
            "bs_2fold_raw_cos": round(0.5 * (bc0 + bc1), 4),
            "bs_2fold_raw_p_max": round(max(bp0, bp1), 4),
            "bs_2fold_perp_val2_20split_mean": round(float(np.mean(bs2_p2)), 4),
            "bs_2fold_perp_val2_cos": round(0.5 * (bc0_2 + bc1_2), 4),
            "bs_2fold_perp_val2_p_max": round(max(bp0_2, bp1_2), 4),
            "bs_2fold_perp_val5_20split_mean": round(float(np.mean(bs2_p5)), 4),
            "bs_2fold_wins_over_uni_bow_20": int(sum(x > y for x, y in zip(bs2_raw, bow_bs_uni_20))),
            "bs_2fold_wins_over_bi_bow_20": int(sum(x > y for x, y in zip(bs2_raw, bow_bs_bi_20))),
            "bs_2fold_perp_val2_wins_over_uni_bow_20": int(sum(x > y for x, y in zip(bs2_p2, bow_bs_uni_20))),
        }

        # Paired wins of HOUSE_perp_val2_dbl and HOUSE_perp_val5_dbl vs unigram BoW on HOUSE 20 splits!
        h_p2_20, h_p5_20 = [], []
        for fl in h_splits:
            ok2, ok5 = [], []
            for f in (0, 1):
                tr, te = (fl != f), (fl == f)
                u_tr = AR.house_dir(Z_house_D[src_k][tr], house_cats[tr])
                u_2 = V234._project_out(V234._project_out(u_tr, Q_val2_m[src_k]), Qt2)
                u_5 = V234._project_out(V234._project_out(u_tr, Q_val5_m[src_k]), Qt5)
                ok2.extend((V234._project_out_rows(Z_house_D[tk][te], Qt2) @ u_2 > 0).tolist())
                ok5.extend((V234._project_out_rows(Z_house_D[tk][te], Qt5) @ u_5 > 0).tolist())
            h_p2_20.append(float(np.mean(ok2)))
            h_p5_20.append(float(np.mean(ok5)))

        house_perp_vs_bow[tk] = {
            "house_perp_val2_dbl_mean": round(float(np.mean(h_p2_20)), 4),
            "house_perp_val2_wins_over_bow_20": int(sum(x > y for x, y in zip(h_p2_20, bow_house_20))),
            "house_perp_val5_dbl_mean": round(float(np.mean(h_p5_20)), 4),
            "house_perp_val5_wins_over_bow_20": int(sum(x > y for x, y in zip(h_p5_20, bow_house_20))),
        }

    report = {
        "dataset_integrity": integrity,
        "lexical_baselines": {
            "bow_house_20split_mean": round(float(np.mean(bow_house_20)), 4),
            "bow_house_to_vm60_unigram": round(bow_house_to_vm_uni, 4),
            "bow_bs60_unigram_20split_mean": round(float(np.mean(bow_bs_uni_20)), 4),
            "bow_bs60_bigram_20split_mean": round(float(np.mean(bow_bs_bi_20)), 4),
            "bow_house_to_bs60_unigram": round(bow_house_to_bs_uni, 4),
            "bow_house_to_bs60_bigram": round(bow_house_to_bs_bi, 4),
        },
        "c5_geometry": c5_geom,
        "heldout_valence_erasure_on_residual": heldout_val_erasure,
        "house_perp_paired_wins_vs_bow": house_perp_vs_bow,
        "transfer_bow_swap_60": transfer_bs,
    }
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, OUT_JSON)
    print(f"SAVED {OUT_JSON}", flush=True)
    return report


def self_test():
    # Exact identical-BoW invariant test
    p = "Owen filed the signed ledger and shredded the draft memo."
    n = "Owen filed the draft memo and shredded the signed ledger."
    assert Counter(_words(p)) == Counter(_words(n))
    acc_uni = _pair_bow_fit_score([{"pos": p, "neg": n}], [{"pos": p, "neg": n}], "pos", "neg", "pos", "neg", _words)
    assert abs(acc_uni - 0.5) < 1e-12
    if DATA_JSON.exists():
        integ = verify_dataset_integrity()
        assert integ["n_pairs"] == 60 and integ["identical_unigram_multiset_count"] == 60
    print("BOW_SWAP236_TEST_OK")


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
        print("ALL_EXP236_DONE_OK")
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
