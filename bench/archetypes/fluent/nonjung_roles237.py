"""Experiment 237: Referee Controls — 8 Non-Jungian Everyday Roles Head-to-Head + Word-Order & Proper-Name Controls.

Directly settles the remaining referee controls in one reproducible, zero-GPU-testable script:
  1. **8 Non-Jungian Everyday Workplace Roles (`nonjung_roles237_data.json`, 320 stories across the EXACT SAME
     40 settings as `named_arch235`):** `Auditor`, `Repair_Tech`, `Negotiator`, `Scheduler`, `Trainer`,
     `Supplier`, `Journalist`, `Archivist` (zero role-name ban-list hits, matched ~51-word lengths) evaluated
     through the EXACT SAME 7-company neutral-text Procrustes pipeline, 20 setting-disjoint splits, `VAL3`
     double-projection, effective rank (PR), and BoW / TF-IDF Centroid / TF-IDF L2-Ridge baselines as `named_arch235`!
  2. **Strict Word-Order / Recency Control on `bow_swap_60` (`mean` AND `last` token):**
     Explicitly splits `bow_swap_60` into `good_object_first` (`38/60`) vs. `bad_object_first` (`22/60`),
     reports `HOUSE` / `H⊥V2` / `SENT` separately on both subsets, tests cross-order transfer (`gf -> bf` and
     `bf -> gf`), and evaluates **strictly 50/50 order-balanced 2-fold CV** (`11 bf + 11 gf` train -> `11 bf + 11 gf`
     test across 20 splits) in BOTH `mean` (`story`) AND `last`-token (`pts`) neutral-text Procrustes frames!
  3. **Zero-Proper-Name Control on `items3` `anchor` & `lure` (Myth-Pool-Excluded GPA):**
     Measures exact proper-name rates in `items3` myths vs. mechanisms and computes 2,000-draw bootstrap 95% CIs
     for Myth minus Mechanism strictly on zero-proper-name passages (`NO_PN`) on held-out `OLMo 3 7B` & `Phi-4-mini`!

Usage:
  python fluent/nonjung_roles237.py test
  python fluent/nonjung_roles237.py extract [KEY...]
  python fluent/nonjung_roles237.py analyze
  python fluent/nonjung_roles237.py all
"""
import gc
import json
import os
import random
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
DATA_JSON = ARCH / "selfshape" / "nonjung_roles237_data.json"
OUT_JSON = PLATO_LOCAL / "referee_ctrl237.json"

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import bow_swap236 as BS236  # noqa: E402

ROLES = (
    "Auditor",
    "Repair_Tech",
    "Negotiator",
    "Scheduler",
    "Trainer",
    "Supplier",
    "Journalist",
    "Archivist",
)

BAN_WORDS = (
    "auditor", "auditors", "audit", "audits", "audited", "auditing",
    "repair", "repairs", "repaired", "repairing", "technician", "technicians", "mechanic", "mechanics",
    "negotiator", "negotiators", "negotiate", "negotiates", "negotiated", "negotiating", "negotiation", "negotiations",
    "scheduler", "schedulers", "schedule", "schedules", "scheduled", "scheduling",
    "trainer", "trainers", "train", "trains", "trained", "training", "trainee", "trainees",
    "supplier", "suppliers", "supply", "supplies", "supplied", "supplying",
    "journalist", "journalists", "journalism", "reporter", "reporters",
    "archivist", "archivists", "archive", "archives", "archived", "archiving",
)
_BAN_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in BAN_WORDS) + r")\b", re.I)

BAD_FIRST_ADJS = {
    "obsolete", "split", "counterfeit", "shorted", "outdated", "frayed", "moldy",
    "contaminated", "pitted", "scorched", "unsealed", "drifting", "corroded",
    "tangled", "cracked", "fragile", "punctured", "flickering", "leaking",
    "misaligned", "clogged", "torn",
}


def load_nonjung_data():
    rows = json.loads(DATA_JSON.read_text())
    assert isinstance(rows, list) and len(rows) == 320, f"Expected 320 rows, got {len(rows)}"
    return rows


def verify_dataset_integrity():
    rows = load_nonjung_data()
    sids = sorted({int(r["setting_id"]) for r in rows})
    assert sids == list(range(1, 41)), f"Expected setting_ids 1..40, got {len(sids)}"
    seen = set()
    by_role_wc = {rl: [] for rl in ROLES}
    ban_hits = []
    for i, r in enumerate(rows):
        rl = r["role"]
        assert rl in ROLES, f"Unknown role {rl} at {i}"
        key = (int(r["setting_id"]), rl)
        assert key not in seen, f"Duplicate {key}"
        seen.add(key)
        txt = r["text"].strip()
        wc = len(txt.split())
        assert 40 <= wc <= 62, f"Word count {wc} out of [40, 62] at {key}"
        by_role_wc[rl].append(wc)
        m = _BAN_RE.findall(txt)
        if m:
            ban_hits.append((key, m))
    assert len(seen) == 320 and not ban_hits, f"Integrity error: seen={len(seen)}, ban={ban_hits[:5]}"
    wc_stats = {
        rl: {"mean": round(float(np.mean(ws)), 2), "std": round(float(np.std(ws)), 2),
             "min": int(min(ws)), "max": int(max(ws))}
        for rl, ws in by_role_wc.items()
    }
    means = [v["mean"] for v in wc_stats.values()]
    assert max(means) - min(means) <= 5.0, wc_stats
    return {
        "n_stories": len(rows),
        "n_settings": len(sids),
        "n_roles": len(ROLES),
        "ban_list_hits": 0,
        "word_count_by_role": wc_stats,
    }


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _save_nj_npz(out_npz, rows, h_last, h_mean, tok_lens, ok_flags, layer_idx, num_layers):
    hl, hm = np.stack(h_last, axis=0), np.stack(h_mean, axis=0)
    assert np.isfinite(hl).all() and np.isfinite(hm).all()
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        out_npz,
        last=pack(hl), mean=pack(hm),
        tok_len=np.array(tok_lens, dtype=np.int32),
        ok=np.array(ok_flags, dtype=bool),
        setting_ids=np.array([int(r["setting_id"]) for r in rows], dtype=np.int32),
        roles=np.array([r["role"] for r in rows]),
        layer_idx=np.int32(layer_idx), num_layers=np.int32(num_layers),
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
    print(f"NJ237 EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
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
            assert 2 <= len(toks) <= 512
            emb = post("/embedding", {"content": txt})
            arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[0] == len(toks)
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
    _save_nj_npz(out_npz, rows, h_last, h_mean, tok_lens, ok_flags, -1, 24)
    print(f"NJ237_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


def extract_model(key):
    assert key in PD.MODEL_SPECS
    out_npz = PLATO_CACHE / f"nj237_{key}.npz"
    if out_npz.exists():
        print(f"NJ237 EXTRACT {key}: already exists, skipping", flush=True)
        return
    rows = load_nonjung_data()
    spec = PD.MODEL_SPECS[key]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp(key, spec, rows, out_npz)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"NJ237 EXTRACT {key}: loading {spec['path']} ...", flush=True)
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

    h_last, h_mean, tok_lens, ok_flags = [], [], [], []
    try:
        for i, r in enumerate(rows):
            ids = PD._text_ids_mlx(tok, r["text"].strip())
            assert 2 <= len(ids) <= 512
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

    _save_nj_npz(out_npz, rows, h_last, h_mean, tok_lens, ok_flags, tgt_idx, nl_cnt)
    print(f"NJ237_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


def _has_proper_name(text):
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        ws = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
        for w in ws[1:]:
            if w[0].isupper() and w not in ("I", "I'm", "I've", "I'd", "I'll"):
                return True
    return False


def _bow_swap_bad_first_mask(bs_rows):
    mask = []
    for r in bs_rows:
        pw, nw = r["pos"].split(), r["neg"].split()
        fi = next(j for j in range(min(len(pw), len(nw))) if pw[j] != nw[j])
        toks = set(re.findall(r"[a-z]+", " ".join(pw[fi:fi + 4]).lower()))
        mask.append(bool(toks & BAD_FIRST_ADJS))
    return np.array(mask, dtype=bool)


def analyze_all():
    """Run all three referee controls: (1) 8 Non-Jungian everyday roles, (2) word-order-balanced &
    last-token bow_swap_60 audit, and (3) zero-proper-name items3 anchor/lure bootstrap CIs."""
    integrity = verify_dataset_integrity()
    rows = load_nonjung_data()
    texts = [r["text"].strip() for r in rows]
    role2idx = {rl: i for i, rl in enumerate(ROLES)}
    y_idx = np.array([role2idx[r["role"]] for r in rows], dtype=int)
    sids = np.array([int(r["setting_id"]) for r in rows], dtype=int)

    # Identical 20 setting-disjoint 2-fold splits as named_arch235!
    uniq_sids = sorted(set(int(s) for s in sids))
    splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23500 + s_i)
        perm = rng.permutation(uniq_sids)
        f0 = set(int(x) for x in perm[:20])
        splits.append(np.array([0 if int(s) in f0 else 1 for s in sids], dtype=int))

    # Temporarily swap NA235.ARCHETYPES to ROLES to reuse exact same vetted functions!
    orig_archs = NA235.ARCHETYPES
    NA235.ARCHETYPES = ROLES
    try:
        lex = NA235._lexical_baselines_20splits(texts, y_idx, splits)
        lex_summary = {
            name: {
                "macro_auroc_mean": round(float(np.mean(lex[name]["macro_auroc_splits"])), 4),
                "macro_auroc_std": round(float(np.std(lex[name]["macro_auroc_splits"])), 4),
                "top1_8way_mean": round(float(np.mean(lex[name]["top1_splits"])), 4),
                "top1_8way_std": round(float(np.std(lex[name]["top1_splits"])), 4),
                "per_role_auroc": {rl: round(float(np.mean(lex[name]["per_arch_auroc_splits"][rl])), 4) for rl in ROLES},
            }
            for name in ("bow_unigram", "tfidf_centroid", "tfidf_ridge")
        }

        orig_plato = AR.PLATO
        AR.PLATO = PLATO_CACHE
        try:
            rot_st = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
            rot_pts = AR.fit_neutral_or_full_rotations(pool="pts", neutral_only=True, d_plato=256)
        finally:
            AR.PLATO = orig_plato

        pairs_arch = AR.load_arch_pairs()
        house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
        house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])

        Z_nj, tok_lens = {}, {}
        u_house_m, u_sent_m, u_outval_m = {}, {}, {}
        Q_val2_m, Q_val3_m, U_nj_m = {}, {}, {}
        Z_bs_m, Z_bs_l, Q2_l, uh_l, us_l = {}, {}, {}, {}, {}

        for k in PD.ALL_KEYS:
            d237 = AR._load_npz_dict(PLATO_CACHE / f"nj237_{k}.npz")
            Z_nj[k] = NA235._map_stories(d237["mean"], rot_st, k, frame="to_c5")
            tok_lens[k] = d237["tok_len"].astype(np.int32)

            a = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
            v = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")
            b = AR._load_npz_dict(PLATO_CACHE / f"bs236_{k}.npz")
            _, _, D_h = AR.map_diffs(a["done_mean"][house_idx], a["neg_mean"][house_idx], rot_st, k, frame="to_c5")
            _, _, D_v = AR.map_diffs(v["pos_mean"], v["neg_mean"], rot_st, k, frame="to_c5")
            _, _, Z_bs_m[k] = AR.map_diffs(b["pos_mean"], b["neg_mean"], rot_st, k, frame="to_c5")
            groups = v["groups"]
            u_house_m[k] = AR.house_dir(D_h, house_cats)
            u_sent_m[k] = AR.unit(D_v[groups == "sentiment_gen"].mean(axis=0))
            u_outval_m[k] = AR.unit(D_v[groups == "outcome_valence"].mean(axis=0))
            Q_val2_m[k] = V234._orthonormal_basis([u_sent_m[k], u_outval_m[k]])
            Q_val3_m[k] = V234._orthonormal_basis([u_sent_m[k], u_outval_m[k], u_house_m[k]])

            Zc = Z_nj[k] - Z_nj[k].mean(axis=0, keepdims=True)
            U_nj_m[k] = NA235._ovr_dirs(Zc, y_idx, n_cls=8)

            # Last-token neutral-only pts frame for bow_swap_60
            _, _, Dh_l = AR.map_diffs(a["done_last"][house_idx], a["neg_last"][house_idx], rot_pts, k, frame="to_c5")
            _, _, Dv_l = AR.map_diffs(v["pos_last"], v["neg_last"], rot_pts, k, frame="to_c5")
            _, _, Z_bs_l[k] = AR.map_diffs(b["pos_last"], b["neg_last"], rot_pts, k, frame="to_c5")
            uh_l[k] = AR.house_dir(Dh_l, house_cats)
            us_l[k] = AR.unit(Dv_l[groups == "sentiment_gen"].mean(axis=0))
            Q2_l[k] = V234._orthonormal_basis([us_l[k], AR.unit(Dv_l[groups == "outcome_valence"].mean(axis=0))])

        # 1. C5 geometry for 8 Non-Jungian roles + word-space geometry correlation (r) & effective rank (PR)!
        def _word_space_geom(texts_in, y_in, n_cls=8):
            vocab = sorted({w for t in texts_in for w in re.findall(r"[a-z]+", t.lower())})
            w2i = {w: j for j, w in enumerate(vocab)}
            X = np.zeros((len(texts_in), len(vocab)), dtype=np.float64)
            for i, t in enumerate(texts_in):
                for w in set(re.findall(r"[a-z]+", t.lower())):
                    X[i, w2i[w]] = 1.0
            Xc = X - X.mean(axis=0, keepdims=True)
            Uw = NA235._ovr_dirs(Xc, y_in, n_cls=n_cls)
            pr_w, _ = NA235._effective_rank_pr(Uw)
            Cw = Uw @ Uw.T
            return round(pr_w, 3), Cw

        u_house_c5 = AR.unit(np.mean([u_house_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
        u_sent_c5 = AR.unit(np.mean([u_sent_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
        u_outval_c5 = AR.unit(np.mean([u_outval_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
        Q3_c5 = V234._orthonormal_basis([u_sent_c5, u_outval_c5, u_house_c5])
        U_c5_raw = np.stack([AR.unit(np.mean([U_nj_m[k][c] for k in PD.CONSENSUS_KEYS], axis=0)) for c in range(8)], axis=0)
        U_c5_p3 = np.stack([V234._project_out(U_c5_raw[c], Q3_c5) for c in range(8)], axis=0)
        pr_raw, sv_raw = NA235._effective_rank_pr(U_c5_raw)
        pr_p3, sv_p3 = NA235._effective_rank_pr(U_c5_p3)
        C_raw = U_c5_raw @ U_c5_raw.T
        off = ~np.eye(8, dtype=bool)
        tri = np.triu_indices(8, k=1)
        pr_word_nj, Cw_nj = _word_space_geom(texts, y_idx, n_cls=8)
        r_c5_vs_word_nj = float(np.corrcoef(C_raw[tri], Cw_nj[tri])[0, 1])

        # Also compute exact same word-space PR and r(C5, word) for named_arch235 (Jung's 8 roles) for apples-to-apples comparison!
        na_rows = NA235.load_named_arch_data()
        na_y = np.array([orig_archs.index(r["archetype"]) for r in na_rows], dtype=int)
        pr_word_jung, Cw_jung = _word_space_geom([r["text"] for r in na_rows], na_y, n_cls=8)
        na_rep = json.loads(NA235.OUT_JSON.read_text())
        C_jung = np.array([[na_rep["c5_geometry"]["pairwise_cos_raw"][a][b] for b in orig_archs] for a in orig_archs], dtype=np.float64)
        r_c5_vs_word_jung = float(np.corrcoef(C_jung[tri], Cw_jung[tri])[0, 1])

        nj_c5_geom = {
            "effective_rank_pr_raw": round(pr_raw, 3),
            "effective_rank_pr_perp_val3": round(pr_p3, 3),
            "word_space_effective_rank_pr": pr_word_nj,
            "corr_c5_vs_word_cos_r": round(r_c5_vs_word_nj, 4),
            "jung_word_space_effective_rank_pr": pr_word_jung,
            "jung_corr_c5_vs_word_cos_r": round(r_c5_vs_word_jung, 4),
            "mean_abs_offdiag_cos_raw": round(float(np.mean(np.abs(C_raw[off]))), 4),
            "per_role_sqnorm_in_val3": {rl: round(float(np.sum((Q3_c5.T @ U_c5_raw[c]) ** 2)), 4) for c, rl in enumerate(ROLES)},
            "pairwise_cos_raw": {a: {b: round(float(C_raw[i, j]), 3) for j, b in enumerate(ROLES)} for i, a in enumerate(ROLES)},
        }

        src_k = "gemma4_e4b"
        targets = [k for k in PD.ALL_KEYS if k != src_k]
        nj_gemma_to_6 = {}
        for tk in targets:
            r_raw = NA235._eval_transfer_splits(Z_nj[src_k], Z_nj[tk], y_idx, sids, splits)
            r_pv3 = NA235._eval_transfer_splits(Z_nj[src_k], Z_nj[tk], y_idx, sids, splits, Q_src=Q_val3_m[src_k], Q_tgt=Q_val3_m[tk])
            r_lr = NA235._eval_transfer_splits(Z_nj[src_k], Z_nj[tk], y_idx, sids, splits, tok_len_tgt=tok_lens[tk])
            nj_gemma_to_6[tk] = {
                "raw": NA235._summarize_block(r_raw, lex["bow_unigram"], lex["tfidf_ridge"]),
                "perp_val3_dbl": NA235._summarize_block(r_pv3, lex["bow_unigram"], lex["tfidf_ridge"]),
                "lenres_top1_mean": round(float(np.mean(r_lr["top1_splits"])), 4),
            }

        Z_c5_nj = np.mean([Z_nj[k] for k in PD.CONSENSUS_KEYS], axis=0)
        nj_c5_heldout = {}
        for tk in ("phi4_mini", "olmo3_7b"):
            rc_r = NA235._eval_transfer_splits(Z_c5_nj, Z_nj[tk], y_idx, sids, splits)
            rc_p = NA235._eval_transfer_splits(Z_c5_nj, Z_nj[tk], y_idx, sids, splits, Q_src=Q3_c5, Q_tgt=Q_val3_m[tk])
            nj_c5_heldout[tk] = {
                "raw_top1_mean": round(float(np.mean(rc_r["top1_splits"])), 4),
                "raw_wins_vs_tfr_20": int(sum(x > y for x, y in zip(rc_r["top1_splits"], lex["tfidf_ridge"]["top1_splits"]))),
                "perp_val3_top1_mean": round(float(np.mean(rc_p["top1_splits"])), 4),
                "perp_val3_wins_vs_tfr_20": int(sum(x > y for x, y in zip(rc_p["top1_splits"], lex["tfidf_ridge"]["top1_splits"]))),
            }
    finally:
        NA235.ARCHETYPES = orig_archs

    # 2. Strict Word-Order / Recency Control on bow_swap_60 (MEAN & LAST)
    bs_rows = BS236.load_bow_swap_data()
    bf_mask = _bow_swap_bad_first_mask(bs_rows)
    idx_bf, idx_gf = np.where(bf_mask)[0], np.where(~bf_mask)[0]
    bs_order_audit = {"n_bad_first": int(len(idx_bf)), "n_good_first": int(len(idx_gf)), "mean_pool": {}, "last_token": {}}

    for mode_key, Z_d, Q_d, u_h_d, u_s_d in [
        ("mean_pool", Z_bs_m, Q_val2_m, u_house_m, u_sent_m),
        ("last_token", Z_bs_l, Q2_l, uh_l, us_l),
    ]:
        uh_p2_s = V234._project_out(u_h_d[src_k], Q_d[src_k])
        for tk in targets:
            Dt = Z_d[tk]
            Dt2 = V234._project_out_rows(Dt, Q_d[tk])
            u2 = V234._project_out(uh_p2_s, Q_d[tk])
            u_gf = AR.unit(Z_d[src_k][idx_gf].mean(0))
            u_bf = AR.unit(Z_d[src_k][idx_bf].mean(0))

            # Strictly 50/50 order-balanced 2-fold CV (11 bf + 11 gf train -> 11 bf + 11 gf test, 20 splits)
            bal_r, bal_2, bal_bf, bal_gf = [], [], [], []
            for s_i in range(20):
                rng = np.random.default_rng(23650 + s_i)
                p_bf = rng.permutation(idx_bf)
                p_gf = rng.permutation(idx_gf)[:len(idx_bf)]
                f0 = np.concatenate([p_bf[:11], p_gf[:11]])
                f1 = np.concatenate([p_bf[11:], p_gf[11:]])
                okr, ok2, okbf, okgf = [], [], [], []
                for tr, te in [(f0, f1), (f1, f0)]:
                    ut = AR.unit(Z_d[src_k][tr].mean(0))
                    ut2 = V234._project_out(V234._project_out(ut, Q_d[src_k]), Q_d[tk])
                    okr.extend((Dt[te] @ ut > 0).tolist())
                    s2 = (Dt2[te] @ ut2 > 0)
                    ok2.extend(s2.tolist())
                    okbf.extend(s2[:11].tolist())
                    okgf.extend(s2[11:].tolist())
                bal_r.append(float(np.mean(okr)))
                bal_2.append(float(np.mean(ok2)))
                bal_bf.append(float(np.mean(okbf)))
                bal_gf.append(float(np.mean(okgf)))

            rng0 = np.random.default_rng(23650)
            p_bf = rng0.permutation(idx_bf)
            p_gf = rng0.permutation(idx_gf)[:len(idx_bf)]
            f0 = np.concatenate([p_bf[:11], p_gf[:11]])
            f1 = np.concatenate([p_bf[11:], p_gf[11:]])
            u0 = V234._project_out(V234._project_out(AR.unit(Z_d[src_k][f0].mean(0)), Q_d[src_k]), Q_d[tk])
            u1 = V234._project_out(V234._project_out(AR.unit(Z_d[src_k][f1].mean(0)), Q_d[src_k]), Q_d[tk])
            c0, p0 = V234._centered_signflip_p(u0, Dt2[f1], seed=2371)
            c1, p1 = V234._centered_signflip_p(u1, Dt2[f0], seed=2372)
            cs_all, ps_all = V234._centered_signflip_p(u_s_d[src_k], Dt, seed=2373)
            ch_all, ph_all = V234._centered_signflip_p(u2, Dt2, seed=2374)

            bs_order_audit[mode_key][tk] = {
                "zs_sent_all": round(float(np.mean((Dt @ u_s_d[src_k]) > 0)), 4),
                "zs_sent_p": round(ps_all, 4),
                "zs_house_gf": round(float(np.mean((Dt[idx_gf] @ u_h_d[src_k]) > 0)), 4),
                "zs_house_bf": round(float(np.mean((Dt[idx_bf] @ u_h_d[src_k]) > 0)), 4),
                "zs_house_perp_v2_all": round(float(np.mean((Dt2 @ u2) > 0)), 4),
                "zs_house_perp_v2_p": round(ph_all, 4),
                "zs_house_perp_v2_gf": round(float(np.mean((Dt2[idx_gf] @ u2) > 0)), 4),
                "zs_house_perp_v2_bf": round(float(np.mean((Dt2[idx_bf] @ u2) > 0)), 4),
                "cross_order_gf_to_bf": round(float(np.mean((Dt[idx_bf] @ u_gf) > 0)), 4),
                "cross_order_bf_to_gf": round(float(np.mean((Dt[idx_gf] @ u_bf) > 0)), 4),
                "balanced_2fold_raw": round(float(np.mean(bal_r)), 4),
                "balanced_2fold_perp_v2": round(float(np.mean(bal_2)), 4),
                "balanced_2fold_perp_v2_bf": round(float(np.mean(bal_bf)), 4),
                "balanced_2fold_perp_v2_gf": round(float(np.mean(bal_gf)), 4),
                "balanced_2fold_perp_v2_cos": round(0.5 * (c0 + c1), 4),
                "balanced_2fold_perp_v2_p_max": round(max(p0, p1), 4),
            }

    # 3. Zero-Proper-Name Control on items3 anchor & lure (Myth-Pool-Excluded GPA)
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    raw = {k: AR._load_npz_dict(PLATO_CACHE / f"raw_{k}.npz") for k in PD.ALL_KEYS}
    C5_ev = np.mean([
        PD.apply_procrustes_map(AR.apply_pca_params(raw[k]["ev_pts"], *rot_pts["pca"][k]), rot_pts["to_c5"][k])
        for k in PD.CONSENSUS_KEYS
    ], axis=0)
    ev_cnts = raw["gemma4_e4b"]["ev_cnt"]
    pt_pid = np.concatenate([[i] * int(c) for i, c in enumerate(ev_cnts)])
    rng_sp = random.Random(2288)
    groups_ev = [r["id"].rsplit("_", 1)[0] if r["domain"].startswith("items3") else r["id"] for r in ev_rows]
    uniq_g = sorted(set(groups_ev))
    rng_sp.shuffle(uniq_g)
    f0_g = set(uniq_g[:len(uniq_g) // 2])
    pt_fold = np.array([0 if groups_ev[pid] in f0_g else 1 for pid in pt_pid])
    rng_b = np.random.default_rng(237)

    pn_rates = {}
    for dom in ("items3_myth", "items3_mech"):
        for role in ("anchor", "lure", "true", "foil"):
            sub = [r for r in ev_rows if r["domain"] == dom and r["id"].endswith("_" + role)]
            pn_rates[f"{dom}_{role}"] = round(sum(1 for r in sub if _has_proper_name(r["text"])) / len(sub), 4)

    items3_no_pn = {"proper_name_rates": pn_rates, "heldout_myth_pool_excluded": {}}
    for bk in ("olmo3_7b", "phi4_mini"):
        Z_tgt = AR.apply_pca_params(raw[bk]["ev_pts"], *rot_pts["pca"][bk])
        pred = np.zeros_like(Z_tgt)
        for f in (0, 1):
            tr, te = (pt_fold != f), (pt_fold == f)
            pred[te] = PD.apply_procrustes_map(C5_ev[te], PD.fit_procrustes_map(C5_ev[tr], Z_tgt[tr]))
        Zc = Z_tgt - Z_tgt.mean(0, keepdims=True)
        Pc = pred - pred.mean(0, keepdims=True)
        pt_cos = np.sum(Zc * Pc, axis=1) / np.maximum(1e-12, np.linalg.norm(Zc, axis=1) * np.linalg.norm(Pc, axis=1))
        p_cos = np.array([pt_cos[pt_pid == i].mean() for i in range(len(ev_rows))])
        items3_no_pn["heldout_myth_pool_excluded"][bk] = {}
        for role in ("anchor", "lure"):
            m_no = np.array([p_cos[i] for i, r in enumerate(ev_rows) if r["domain"] == "items3_myth" and r["id"].endswith("_" + role) and not _has_proper_name(r["text"])])
            m_pn = np.array([p_cos[i] for i, r in enumerate(ev_rows) if r["domain"] == "items3_myth" and r["id"].endswith("_" + role) and _has_proper_name(r["text"])])
            c_no = np.array([p_cos[i] for i, r in enumerate(ev_rows) if r["domain"] == "items3_mech" and r["id"].endswith("_" + role) and not _has_proper_name(r["text"])])
            diffs = [rng_b.choice(m_no, len(m_no)).mean() - rng_b.choice(c_no, len(c_no)).mean() for _ in range(2000)]
            items3_no_pn["heldout_myth_pool_excluded"][bk][role] = {
                "n_myth_no_pn": int(len(m_no)),
                "n_myth_with_pn": int(len(m_pn)),
                "n_mech_no_pn": int(len(c_no)),
                "myth_with_pn_mean": round(float(m_pn.mean()), 4),
                "myth_no_pn_mean": round(float(m_no.mean()), 4),
                "mech_no_pn_mean": round(float(c_no.mean()), 4),
                "diff_no_pn": round(float(m_no.mean() - c_no.mean()), 4),
                "ci95_no_pn": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            }

    report = {
        "nonjung_roles_integrity": integrity,
        "nonjung_roles_lexical_20splits": lex_summary,
        "nonjung_roles_c5_geometry": nj_c5_geom,
        "nonjung_roles_gemma_to_6": nj_gemma_to_6,
        "nonjung_roles_c5_to_heldout": nj_c5_heldout,
        "bow_swap_word_order_and_last_token": bs_order_audit,
        "items3_zero_proper_name_control": items3_no_pn,
    }
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, OUT_JSON)
    print(f"SAVED {OUT_JSON}", flush=True)
    return report


def self_test():
    assert _has_proper_name("The boy saw Odysseus on the shore.") is True
    assert _has_proper_name("The young shepherd saw a wolf on the hill.") is False
    if DATA_JSON.exists():
        integ = verify_dataset_integrity()
        assert integ["n_stories"] == 320 and integ["ban_list_hits"] == 0
    if BS236.DATA_JSON.exists():
        m = _bow_swap_bad_first_mask(BS236.load_bow_swap_data())
        assert int(m.sum()) == 22 and int((~m).sum()) == 38
    print("NONJUNG_ROLES237_TEST_OK")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        self_test()
    elif cmd == "extract":
        keys = sys.argv[2:] if len(sys.argv) > 2 else list(PD.ALL_KEYS)
        for k in keys:
            extract_model(k)
    elif cmd == "analyze":
        analyze_all()
    elif cmd == "all":
        self_test()
        for k in PD.ALL_KEYS:
            extract_model(k)
        analyze_all()
        print("ALL_EXP237_DONE_OK")
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
