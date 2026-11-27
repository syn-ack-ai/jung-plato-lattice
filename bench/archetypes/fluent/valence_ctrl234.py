"""Experiment 234: Valence & Evaluative-Polarity Control Audit across All 7 LLM Families + Native Epistemic Probes.

Directly tests whether the cross-family `HOUSE` completion/coherence axis (`arch_rot229`) and native-layer
`HOUSE_PERP` epistemic probe (`audit233_epistemic`) survive strict control for domain-general good/bad
evaluative valence (Tigges et al., 2023):

  1. Extracts ~30%-depth activations across ALL 7 open model families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B,
     Mistral 7B, looped Inkling 2 Mini, held-out Phi-4-mini, held-out OLMo 3 7B) on 3 new zero-overlap benchmarks
     in `bench/archetypes/selfshape/valence_ctrl234_data.json`:
       - `sentiment_gen`   (60 pairs: domain-general positive vs. negative evaluative sentiment),
       - `outcome_valence` (40 pairs: fortunate vs. unfortunate story outcome with identical neutral action),
       - `valence_matched` (60 brand-new pairs across 60 settings disjoint from all 119 HOUSE/SELF2 settings:
                            30 both_negative + 30 both_positive contrasting structural integration/completion
                            against neglect/fragmentation under identical emotional & outcome valence).
  2. Projects into the EXACT neutral-text-only (`gut + proc + chat`) 256-dim Procrustes frame of `arch_rot229.py`
     (and `C5`) and measures:
       - Direct overlap `cos(u_house, u_sent)`, `cos(u_house, u_outval)`, and variance of `u_house` explained by
         `span(u_sent, u_outval)` inside each model and inside `C5`.
       - Cross-task transfer: pure `SENTIMENT -> HOUSE` vs. raw `HOUSE -> HOUSE` vs.
         **source-orthogonalized `HOUSE ⊥ SENTIMENT`** vs. **double-orthogonalized `HOUSE ⊥ VALENCE_2D`**
         (projecting out `span(u_sent, u_outval)` in BOTH source Gemma 4 AND target model!) across the exact same
         20 deterministic setting-disjoint splits from `audit233_mac.py` against unigram BoW!
       - **Valence-matched benchmark audit (`valence_matched`, 60 pairs: 30 both_neg + 30 both_pos):**
         - Empirically verifies `u_sent` & `u_outval` drop sharply on `valence_matched` (confirming valence control!),
         - Tests zero-shot transfer of Gemma 4's `HOUSE`, `HOUSE ⊥ VALENCE_2D`, and `u_sent` into all 6 rival
           families via neutral-text Procrustes on `valence_matched` (plus 2-fold CV fit directly on `valence_matched`
           and fold-local length residualization vs. unigram BoW!).
  3. Native-layer epistemic valence control (`epistemic_mlx`) on Google `Gemma 4 E4B` (bf16 MLX, L17 & L25) AND
     Alibaba `Qwen3.5-9B` (8-bit MLX, L12 & L19) on `clean_tense_clean_24`:
       - Compares `-CORRECTNESS`, `-SENTIMENT_PERP` (pure valence orthogonalized to `CORRECTNESS`),
         `-HOUSE_PERP` (orthogonal to `CORRECTNESS`), `-HOUSE_PERP_BOTH` (orthogonal to BOTH `CORRECTNESS` and
         `SENTIMENT`), `Combined(CORR, SENT_PERP)`, `Combined(CORR, HOUSE_PERP)`, and
         `Combined(CORR, HOUSE_PERP_BOTH)` against 500 mirrored orthogonal random pairs, PLUS paired bootstrap
         incremental lift (`Combined_Perp - neg_corr`).

Usage:
  python fluent/valence_ctrl234.py test
  python fluent/valence_ctrl234.py extract [KEY...]
  python fluent/valence_ctrl234.py analyze
  python fluent/valence_ctrl234.py epistemic
"""
import gc
import hashlib
import json
import os
import re
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
DATA_JSON = ARCH / "selfshape" / "valence_ctrl234_data.json"
OUT_JSON = PLATO_LOCAL / "valence_ctrl234.json"

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402


def load_valence_data():
    d = json.loads(DATA_JSON.read_text())
    sg = d["sentiment_gen"]
    ov = d["outcome_valence"]
    vm = d["valence_matched"]
    assert len(sg) == 60 and len(ov) == 40 and len(vm) == 60, (len(sg), len(ov), len(vm))
    return sg, ov, vm


def verify_dataset_integrity():
    """Verify zero setting overlap with all 119 HOUSE/SELF2 settings, unique settings inside valence_matched,
    balanced polarities (30 both_negative + 30 both_positive), and balanced word lengths."""
    sg, ov, vm = load_valence_data()
    existing_settings = {p["setting"].strip().lower() for p in AR.load_arch_pairs() if p.get("setting")}
    vm_settings = [r["setting"].strip().lower() for r in vm]
    assert len(vm_settings) == len(set(vm_settings)) == 60, "Duplicate settings in valence_matched!"
    overlap = existing_settings & set(vm_settings)
    assert not overlap, f"Setting overlap with HOUSE/SELF2: {overlap}"
    pols = [r["polarity"] for r in vm]
    assert pols.count("both_negative") == 30 and pols.count("both_positive") == 30

    # Verify reasonable length matching
    sg_diff = np.mean([len(r["pos"].split()) - len(r["neg"].split()) for r in sg])
    ov_diff = np.mean([len(r["pos"].split()) - len(r["neg"].split()) for r in ov])
    vm_diff = np.mean([len(r["integrated"].split()) - len(r["neglected"].split()) for r in vm])
    assert abs(sg_diff) < 1.5 and abs(ov_diff) < 2.5 and abs(vm_diff) < 2.5, (sg_diff, ov_diff, vm_diff)
    return {
        "n_sg": len(sg), "n_ov": len(ov), "n_vm": len(vm),
        "sg_mean_dw": round(float(sg_diff), 2),
        "ov_mean_dw": round(float(ov_diff), 2),
        "vm_mean_dw": round(float(vm_diff), 2),
        "n_existing_settings_checked": len(existing_settings),
    }


def _all_items():
    """Return ordered list of (group, id, pos_text, neg_text, meta_str) across all 160 pairs."""
    sg, ov, vm = load_valence_data()
    items = []
    for r in sg:
        items.append(("sentiment_gen", r["id"], r["pos"].strip(), r["neg"].strip(), r["domain"]))
    for r in ov:
        items.append(("outcome_valence", r["id"], r["pos"].strip(), r["neg"].strip(), "outcome"))
    for r in vm:
        items.append(("valence_matched", r["id"], r["integrated"].strip(), r["neglected"].strip(), r["polarity"]))
    return items


def _save_val_npz(out_npz, items, pl, pm, nl, nm, p_lens, n_lens, ok_flags, layer_idx, num_layers):
    pl = np.stack(pl, axis=0)
    pm = np.stack(pm, axis=0)
    nl = np.stack(nl, axis=0)
    nm = np.stack(nm, axis=0)
    assert np.isfinite(pl).all() and np.isfinite(pm).all() and np.isfinite(nl).all() and np.isfinite(nm).all()
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        out_npz,
        pos_last=pack(pl), pos_mean=pack(pm),
        neg_last=pack(nl), neg_mean=pack(nm),
        pos_len=np.array(p_lens, dtype=np.int32),
        neg_len=np.array(n_lens, dtype=np.int32),
        ok=np.array(ok_flags, dtype=bool),
        groups=np.array([x[0] for x in items]),
        ids=np.array([x[1] for x in items]),
        subtags=np.array([x[4] for x in items]),
        layer_idx=np.int32(layer_idx), num_layers=np.int32(num_layers),
    )


def _port_open(port):
    import socket
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _extract_llamacpp(key, spec, items, out_npz):
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
    print(f"VAL234 EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
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
            if not (2 <= len(toks) <= 512):
                return None, None, 0
            emb = post("/embedding", {"content": text})
            arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != len(toks):
                return None, None, 0
            return arr[-1].copy(), (arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0].copy()), len(toks)

        for i, (_, _, pt, nt, _) in enumerate(items):
            p_l, p_m, p_ln = embed_one(pt)
            n_l, n_m, n_ln = embed_one(nt)
            assert p_l is not None and n_l is not None, f"{key} failed item {i}"
            pl.append(p_l); pm.append(p_m); nl.append(n_l); nm.append(n_m)
            p_lens.append(p_ln); n_lens.append(n_ln); ok_flags.append(True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _save_val_npz(out_npz, items, pl, pm, nl, nm, p_lens, n_lens, ok_flags, -1, 24)
    print(f"VAL234_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(items)} in {time.time() - t0:.1f}s", flush=True)


def extract_model(key):
    assert key in PD.MODEL_SPECS, f"unknown key {key}"
    out_npz = PLATO_CACHE / f"val234_{key}.npz"
    if out_npz.exists():
        print(f"VAL234 EXTRACT {key}: already exists ({out_npz}), skipping", flush=True)
        return
    items = _all_items()
    spec = PD.MODEL_SPECS[key]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp(key, spec, items, out_npz)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"VAL234 EXTRACT {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl_cnt = len(layers)
    tgt_idx = max(1, min(nl_cnt - 2, int(round(spec["depth_frac"] * (nl_cnt - 1)))))
    tgt_blk = layers[tgt_idx]
    print(f"VAL234 EXTRACT {key}: L={nl_cnt}, hooking layer {tgt_idx} ({tgt_idx / (nl_cnt - 1):.1%})", flush=True)

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
            assert 2 <= len(ids) <= 512, f"{key} bad token len {len(ids)}"
            rec_buf.clear()
            _ = model(mx.array([ids]))
            assert len(rec_buf) == 1
            h = rec_buf[0].astype(mx.float32)
            hl = h[-1]
            hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
            mx.eval(hl, hm)
            rec_buf.clear()
            return np.array(hl, dtype=np.float32), np.array(hm, dtype=np.float32), len(ids)

        for i, (_, _, pt, nt, _) in enumerate(items):
            p_l, p_m, p_ln = embed_one(pt)
            n_l, n_m, n_ln = embed_one(nt)
            pl.append(p_l); pm.append(p_m); nl.append(n_l); nm.append(n_m)
            p_lens.append(p_ln); n_lens.append(n_ln); ok_flags.append(True)
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()

    _save_val_npz(out_npz, items, pl, pm, nl, nm, p_lens, n_lens, ok_flags, tgt_idx, nl_cnt)
    print(f"VAL234_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(items)} in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Analysis Helpers

def _orthonormal_basis(vecs):
    """Return orthonormal columns spanning `vecs` via QR."""
    M = np.stack([AR.unit(v) for v in vecs], axis=1)
    Q, _ = np.linalg.qr(M)
    return Q  # (D, k)


def _project_out(v, Q):
    """Project `v` strictly orthogonal to columns of Q and unit-normalize."""
    v = np.asarray(v, dtype=np.float64)
    vp = v - Q @ (Q.T @ v)
    return AR.unit(vp)


def _project_out_rows(D, Q):
    """Project every row of D strictly orthogonal to columns of Q."""
    return D - (D @ Q) @ Q.T


def _centered_signflip_p(u_pred, D_eval, n_perm=1000, seed=234):
    """Centered sign-flip permutation p-value for positive cosine with u_pred."""
    rng = np.random.default_rng(seed)
    u = AR.unit(u_pred)
    obs = float(np.dot(u, AR.unit(D_eval.mean(axis=0))))
    Dc = D_eval - D_eval.mean(axis=0, keepdims=True)
    # Vectorized sign flips
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(Dc)))
    means = (signs @ Dc) / float(len(Dc))
    norms = np.maximum(1e-12, np.linalg.norm(means, axis=1))
    null_cos = (means @ u) / norms
    p = float((1 + np.sum(null_cos >= obs)) / (n_perm + 1))
    return obs, p


def _bow_2fold_setting_disjoint(pairs, pos_key, neg_key, n_splits=20):
    """Binary unigram difference-of-means BoW classifier over 20 deterministic setting-disjoint 2-fold splits."""
    tok_fn = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    uniq_sets = sorted({p["setting"].strip().lower() for p in pairs})
    accs = []
    for s_i in range(n_splits):
        rng = np.random.default_rng(23400 + s_i)
        perm = rng.permutation(len(uniq_sets))
        f0 = {uniq_sets[j] for j in perm[:len(uniq_sets) // 2]}
        fold_of = np.array([0 if p["setting"].strip().lower() in f0 else 1 for p in pairs], dtype=int)
        ok = []
        for f in (0, 1):
            tr_idx = np.where(fold_of != f)[0]
            te_idx = np.where(fold_of == f)[0]
            # Strict train-only vocabulary and weights
            w_cnt = {}
            for i in tr_idx:
                pt, nt = tok_fn(pairs[i][pos_key]), tok_fn(pairs[i][neg_key])
                for w in pt - nt:
                    w_cnt[w] = w_cnt.get(w, 0.0) + 1.0
                for w in nt - pt:
                    w_cnt[w] = w_cnt.get(w, 0.0) - 1.0
            for i in te_idx:
                pt, nt = tok_fn(pairs[i][pos_key]), tok_fn(pairs[i][neg_key])
                sc = sum(w_cnt.get(w, 0.0) for w in pt) - sum(w_cnt.get(w, 0.0) for w in nt)
                ok.append(1.0 if sc > 0 else (0.5 if sc == 0 else 0.0))
        accs.append(float(np.mean(ok)))
    return accs


def _fold_local_lenres_acc(D_tr, dl_tr, dlog_tr, D_te, dl_te, dlog_te, u_src):
    """Fit length regression strictly on fold-local target train rows (ZERO test leak!) and score target test."""
    X_tr = np.stack([np.ones(len(dl_tr)), dl_tr, dlog_tr], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X_tr, D_tr, rcond=None)
    # Remove only length slope (keep intercept!)
    D_te_lr = D_te - (dl_te[:, None] * coef[1:2, :] + dlog_te[:, None] * coef[2:3, :])
    sc = D_te_lr @ AR.unit(u_src)
    return float(np.mean(sc > 0))


def analyze_7model():
    """Run complete 7-company valence-control analysis in the EXACT neutral-text-only (`gut+proc+chat`)
    256-dim Procrustes frame of `arch_rot229.py` (`map_diffs(..., frame='to_c5')`)."""
    integrity = verify_dataset_integrity()
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato

    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_pairs = [pairs_arch[i] for i in house_idx]
    house_cats = np.array([p["cat"] for p in house_pairs])

    sg_rows, ov_rows, vm_rows = load_valence_data()

    # Map all 7 models' HOUSE and VAL234 paired diffs directly into the shared neutral-text-only C5 frame!
    Z_house_D, Z_sg_D, Z_ov_D, Z_vm_D = {}, {}, {}, {}
    vm_dl, vm_dlog = {}, {}
    u_house_m, u_sent_m, u_outval_m, u_vm_m = {}, {}, {}, {}
    Q_sent_m, Q_val2_m = {}, {}

    for k in PD.ALL_KEYS:
        a = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
        v = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")

        _, _, Z_house_D[k] = AR.map_diffs(a["done_mean"][house_idx], a["neg_mean"][house_idx], rot, k, frame="to_c5")
        _, _, D_all = AR.map_diffs(v["pos_mean"], v["neg_mean"], rot, k, frame="to_c5")

        groups = v["groups"]
        m_sg = groups == "sentiment_gen"
        m_ov = groups == "outcome_valence"
        m_vm = groups == "valence_matched"
        Z_sg_D[k] = D_all[m_sg]
        Z_ov_D[k] = D_all[m_ov]
        Z_vm_D[k] = D_all[m_vm]

        pl = v["pos_len"][m_vm].astype(np.float64)
        nl = v["neg_len"][m_vm].astype(np.float64)
        vm_dl[k] = pl - nl
        vm_dlog[k] = np.log(np.maximum(1.0, pl)) - np.log(np.maximum(1.0, nl))

        u_house_m[k] = AR.house_dir(Z_house_D[k], house_cats)
        u_sent_m[k] = AR.unit(Z_sg_D[k].mean(axis=0))
        u_outval_m[k] = AR.unit(Z_ov_D[k].mean(axis=0))
        u_vm_m[k] = AR.unit(Z_vm_D[k].mean(axis=0))
        Q_sent_m[k] = _orthonormal_basis([u_sent_m[k]])
        Q_val2_m[k] = _orthonormal_basis([u_sent_m[k], u_outval_m[k]])

    # 1. Direct within-model & within-C5 geometry: how much of HOUSE lies in span(u_sent, u_outval)?
    within_geom = {}
    for k in PD.ALL_KEYS:
        uh, us, uo, uvm = u_house_m[k], u_sent_m[k], u_outval_m[k], u_vm_m[k]
        Q2 = Q_val2_m[k]
        r2_val2 = float(np.sum((Q2.T @ uh) ** 2))
        uh_p2 = _project_out(uh, Q2)
        within_geom[k] = {
            "cos_house_sent": round(float(np.dot(uh, us)), 4),
            "cos_house_outval": round(float(np.dot(uh, uo)), 4),
            "cos_sent_outval": round(float(np.dot(us, uo)), 4),
            "house_sqnorm_in_val2_span": round(r2_val2, 4),
            "cos_vm_house": round(float(np.dot(uvm, uh)), 4),
            "cos_vm_house_perp_val2": round(float(np.dot(uvm, uh_p2)), 4),
            "cos_vm_sent": round(float(np.dot(uvm, us)), 4),
            "cos_vm_outval": round(float(np.dot(uvm, uo)), 4),
            "sent_acc_on_sg_2fold": round(float(np.mean([
                np.mean((Z_sg_D[k][1::2] @ AR.unit(Z_sg_D[k][0::2].mean(0))) > 0),
                np.mean((Z_sg_D[k][0::2] @ AR.unit(Z_sg_D[k][1::2].mean(0))) > 0),
            ])), 4),
            "outval_acc_on_ov_2fold": round(float(np.mean([
                np.mean((Z_ov_D[k][1::2] @ AR.unit(Z_ov_D[k][0::2].mean(0))) > 0),
                np.mean((Z_ov_D[k][0::2] @ AR.unit(Z_ov_D[k][1::2].mean(0))) > 0),
            ])), 4),
            "sent_acc_on_house": round(float(np.mean((Z_house_D[k] @ us) > 0)), 4),
            "outval_acc_on_house": round(float(np.mean((Z_house_D[k] @ uo) > 0)), 4),
            "sent_acc_on_vm": round(float(np.mean((Z_vm_D[k] @ us) > 0)), 4),
            "outval_acc_on_vm": round(float(np.mean((Z_vm_D[k] @ uo) > 0)), 4),
            "house_acc_on_vm": round(float(np.mean((Z_vm_D[k] @ uh) > 0)), 4),
            "house_perp_val2_acc_on_vm": round(float(np.mean((Z_vm_D[k] @ uh_p2) > 0)), 4),
        }

    # Consensus C5 geometry (already in C5 frame!)
    u_house_c5 = AR.unit(np.mean([u_house_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_sent_c5 = AR.unit(np.mean([u_sent_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_outval_c5 = AR.unit(np.mean([u_outval_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_vm_c5 = AR.unit(np.mean([u_vm_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    Q2_c5 = _orthonormal_basis([u_sent_c5, u_outval_c5])
    u_house_perp2_c5 = _project_out(u_house_c5, Q2_c5)
    c5_geom = {
        "cos_house_sent": round(float(np.dot(u_house_c5, u_sent_c5)), 4),
        "cos_house_outval": round(float(np.dot(u_house_c5, u_outval_c5)), 4),
        "cos_sent_outval": round(float(np.dot(u_sent_c5, u_outval_c5)), 4),
        "house_sqnorm_in_val2_span": round(float(np.sum((Q2_c5.T @ u_house_c5) ** 2)), 4),
        "cos_vm_house": round(float(np.dot(u_vm_c5, u_house_c5)), 4),
        "cos_vm_house_perp_val2": round(float(np.dot(u_vm_c5, u_house_perp2_c5)), 4),
        "cos_vm_sent": round(float(np.dot(u_vm_c5, u_sent_c5)), 4),
        "cos_vm_outval": round(float(np.dot(u_vm_c5, u_outval_c5)), 4),
    }

    # 2. Precompute 20 deterministic setting-disjoint splits for HOUSE
    h_uniq_sets = sorted({p["setting"].strip().lower() for p in house_pairs})
    h_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23300 + s_i)
        perm = rng.permutation(len(h_uniq_sets))
        f0 = {h_uniq_sets[j] for j in perm[:len(h_uniq_sets) // 2]}
        h_splits.append(np.array([0 if p["setting"].strip().lower() in f0 else 1 for p in house_pairs], dtype=int))

    # BoW baselines on VALENCE_MATCHED_60
    bow_vm_accs = _bow_2fold_setting_disjoint(vm_rows, "integrated", "neglected", n_splits=20)
    bow_vm_neg = _bow_2fold_setting_disjoint([r for r in vm_rows if r["polarity"] == "both_negative"], "integrated", "neglected", n_splits=20)
    bow_vm_pos = _bow_2fold_setting_disjoint([r for r in vm_rows if r["polarity"] == "both_positive"], "integrated", "neglected", n_splits=20)

    vm_pols = np.array([r["polarity"] for r in vm_rows])
    m_vneg = vm_pols == "both_negative"
    m_vpos = vm_pols == "both_positive"

    # 20 stratified 2-fold splits for valence_matched (15 both_neg + 15 both_pos per fold; all 60 settings unique!)
    vm_splits = []
    idx_neg = np.where(m_vneg)[0]
    idx_pos = np.where(m_vpos)[0]
    for s_i in range(20):
        rng = np.random.default_rng(23400 + s_i)
        pn = rng.permutation(idx_neg)
        pp = rng.permutation(idx_pos)
        fl = np.zeros(len(vm_rows), dtype=int)
        fl[pn[15:]] = 1
        fl[pp[15:]] = 1
        vm_splits.append(fl)

    targets = [k for k in PD.ALL_KEYS if k != "gemma4_e4b"]
    transfer_house = {}
    transfer_vm = {}

    src_k = "gemma4_e4b"
    Q_sent_src = Q_sent_m[src_k]
    Q_val2_src = Q_val2_m[src_k]
    u_house_src_full = u_house_m[src_k]
    u_house_perp_val2_src = _project_out(u_house_src_full, Q_val2_src)
    u_sent_src = u_sent_m[src_k]
    u_outval_src = u_outval_m[src_k]

    for tk in targets:
        Qt_s = Q_sent_m[tk]
        Qt_v2 = Q_val2_m[tk]

        # ---- Test A: HOUSE (20 setting-disjoint splits in neutral-text C5 frame) ----
        raw_accs, src_ps_accs, dbl_ps_accs, src_pv2_accs, dbl_pv2_accs = [], [], [], [], []
        for fl in h_splits:
            ok_r, ok_sps, ok_dps, ok_sp2, ok_dp2 = [], [], [], [], []
            for f in (0, 1):
                tr, te = (fl != f), (fl == f)
                u_tr = AR.house_dir(Z_house_D[src_k][tr], house_cats[tr])
                u_tr_ps = _project_out(u_tr, Q_sent_src)
                u_tr_p2 = _project_out(u_tr, Q_val2_src)

                D_te = Z_house_D[tk][te]
                D_te_ps = _project_out_rows(D_te, Qt_s)
                D_te_p2 = _project_out_rows(D_te, Qt_v2)

                ok_r.extend((D_te @ u_tr > 0).tolist())
                ok_sps.extend((D_te @ u_tr_ps > 0).tolist())
                ok_dps.extend((D_te_ps @ _project_out(u_tr_ps, Qt_s) > 0).tolist())
                ok_sp2.extend((D_te @ u_tr_p2 > 0).tolist())
                ok_dp2.extend((D_te_p2 @ _project_out(u_tr_p2, Qt_v2) > 0).tolist())

            raw_accs.append(float(np.mean(ok_r)))
            src_ps_accs.append(float(np.mean(ok_sps)))
            dbl_ps_accs.append(float(np.mean(ok_dps)))
            src_pv2_accs.append(float(np.mean(ok_sp2)))
            dbl_pv2_accs.append(float(np.mean(ok_dp2)))

        # Zero-shot cross-task: Gemma's pure SENTIMENT / OUTCOME_VALENCE -> target HOUSE
        s2h_acc = float(np.mean((Z_house_D[tk] @ u_sent_src) > 0))
        o2h_acc = float(np.mean((Z_house_D[tk] @ u_outval_src) > 0))
        s2s_cos, s2s_p = _centered_signflip_p(u_sent_src, Z_sg_D[tk])
        s2s_acc = float(np.mean((Z_sg_D[tk] @ u_sent_src) > 0))

        # Centered sign-flip p on alternating 2-fold double-orthogonalized HOUSE_PERP_VAL2
        alt_fl = np.arange(len(house_pairs)) % 2
        u_f0_p2 = _project_out(AR.house_dir(Z_house_D[src_k][alt_fl == 0], house_cats[alt_fl == 0]), Q_val2_src)
        u_f1_p2 = _project_out(AR.house_dir(Z_house_D[src_k][alt_fl == 1], house_cats[alt_fl == 1]), Q_val2_src)
        c0, p0 = _centered_signflip_p(_project_out(u_f0_p2, Qt_v2), _project_out_rows(Z_house_D[tk][alt_fl == 1], Qt_v2))
        c1, p1 = _centered_signflip_p(_project_out(u_f1_p2, Qt_v2), _project_out_rows(Z_house_D[tk][alt_fl == 0], Qt_v2))

        transfer_house[tk] = {
            "sent_to_sent_zs_cos": round(s2s_cos, 4),
            "sent_to_sent_zs_p": round(s2s_p, 4),
            "sent_to_sent_zs_acc": round(s2s_acc, 4),
            "sent_to_house_zs_acc": round(s2h_acc, 4),
            "outval_to_house_zs_acc": round(o2h_acc, 4),
            "house_raw_20split_mean": round(float(np.mean(raw_accs)), 4),
            "house_perp_sent_src_20split_mean": round(float(np.mean(src_ps_accs)), 4),
            "house_perp_sent_dbl_20split_mean": round(float(np.mean(dbl_ps_accs)), 4),
            "house_perp_val2_src_20split_mean": round(float(np.mean(src_pv2_accs)), 4),
            "house_perp_val2_dbl_20split_mean": round(float(np.mean(dbl_pv2_accs)), 4),
            "house_perp_val2_dbl_20split_std": round(float(np.std(dbl_pv2_accs)), 4),
            "house_perp_val2_dbl_2fold_cos": round(0.5 * (c0 + c1), 4),
            "house_perp_val2_dbl_2fold_p_max": round(max(p0, p1), 4),
        }

        # ---- Test B: Brand-New VALENCE_MATCHED_60 (30 both_neg + 30 both_pos) ----
        D_vm_t = Z_vm_D[tk]
        u_hp2_dbl = _project_out(u_house_perp_val2_src, Qt_v2)

        zs_h_cos, zs_h_p = _centered_signflip_p(u_house_src_full, D_vm_t)
        zs_hp2_cos, zs_hp2_p = _centered_signflip_p(u_hp2_dbl, _project_out_rows(D_vm_t, Qt_v2))
        zs_s_cos, zs_s_p = _centered_signflip_p(u_sent_src, D_vm_t)

        # Strictly disjoint 2-fold CV fit ON valence_matched in Gemma 4 -> target tk (20 splits!)
        vm2_raw, vm2_dbl, vm2_neg, vm2_pos, vm2_lr = [], [], [], [], []
        for fl in vm_splits:
            ok_r, ok_d, ok_n, ok_p, ok_lr = [], [], [], [], []
            for f in (0, 1):
                tr, te = (fl != f), (fl == f)
                u_vm_tr = AR.unit(Z_vm_D[src_k][tr].mean(axis=0))
                u_vm_tr_p2 = _project_out(u_vm_tr, Q_val2_src)
                ud = _project_out(u_vm_tr_p2, Qt_v2)
                D_te = D_vm_t[te]
                D_te_p2 = _project_out_rows(D_te, Qt_v2)
                sc_r = D_te @ u_vm_tr > 0
                sc_d = D_te_p2 @ ud > 0
                ok_r.extend(sc_r.tolist())
                ok_d.extend(sc_d.tolist())
                ok_n.extend(sc_r[vm_pols[te] == "both_negative"].tolist())
                ok_p.extend(sc_r[vm_pols[te] == "both_positive"].tolist())
                ok_lr.append(_fold_local_lenres_acc(
                    D_vm_t[tr], vm_dl[tk][tr], vm_dlog[tk][tr],
                    D_te, vm_dl[tk][te], vm_dlog[tk][te],
                    u_vm_tr
                ))
            vm2_raw.append(float(np.mean(ok_r)))
            vm2_dbl.append(float(np.mean(ok_d)))
            vm2_neg.append(float(np.mean(ok_n)))
            vm2_pos.append(float(np.mean(ok_p)))
            vm2_lr.append(float(np.mean(ok_lr)))

        alt_vm = vm_splits[0]
        vc0, vp0 = _centered_signflip_p(AR.unit(Z_vm_D[src_k][alt_vm == 0].mean(0)), D_vm_t[alt_vm == 1])
        vc1, vp1 = _centered_signflip_p(AR.unit(Z_vm_D[src_k][alt_vm == 1].mean(0)), D_vm_t[alt_vm == 0])

        transfer_vm[tk] = {
            "zs_sent_to_vm_acc": round(float(np.mean((D_vm_t @ u_sent_src) > 0)), 4),
            "zs_sent_to_vm_cos": round(zs_s_cos, 4),
            "zs_sent_to_vm_p": round(zs_s_p, 4),
            "zs_outval_to_vm_acc": round(float(np.mean((D_vm_t @ u_outval_src) > 0)), 4),
            "zs_house_to_vm_acc": round(float(np.mean((D_vm_t @ u_house_src_full) > 0)), 4),
            "zs_house_to_vm_both_neg_acc": round(float(np.mean((D_vm_t[m_vneg] @ u_house_src_full) > 0)), 4),
            "zs_house_to_vm_both_pos_acc": round(float(np.mean((D_vm_t[m_vpos] @ u_house_src_full) > 0)), 4),
            "zs_house_to_vm_cos": round(zs_h_cos, 4),
            "zs_house_to_vm_p": round(zs_h_p, 4),
            "zs_house_perp_val2_dbl_to_vm_acc": round(float(np.mean((_project_out_rows(D_vm_t, Qt_v2) @ u_hp2_dbl) > 0)), 4),
            "zs_house_perp_val2_dbl_to_vm_both_neg_acc": round(float(np.mean((_project_out_rows(D_vm_t[m_vneg], Qt_v2) @ u_hp2_dbl) > 0)), 4),
            "zs_house_perp_val2_dbl_to_vm_both_pos_acc": round(float(np.mean((_project_out_rows(D_vm_t[m_vpos], Qt_v2) @ u_hp2_dbl) > 0)), 4),
            "zs_house_perp_val2_dbl_to_vm_cos": round(zs_hp2_cos, 4),
            "zs_house_perp_val2_dbl_to_vm_p": round(zs_hp2_p, 4),
            "vm_2fold_20split_mean": round(float(np.mean(vm2_raw)), 4),
            "vm_2fold_20split_std": round(float(np.std(vm2_raw)), 4),
            "vm_2fold_both_neg_mean": round(float(np.mean(vm2_neg)), 4),
            "vm_2fold_both_pos_mean": round(float(np.mean(vm2_pos)), 4),
            "vm_2fold_lenres_mean": round(float(np.mean(vm2_lr)), 4),
            "vm_2fold_perp_val2_dbl_mean": round(float(np.mean(vm2_dbl)), 4),
            "vm_2fold_cos": round(0.5 * (vc0 + vc1), 4),
            "vm_2fold_p_max": round(max(vp0, vp1), 4),
            "vm_wins_over_bow_20splits": int(sum(a > b for a, b in zip(vm2_raw, bow_vm_accs))),
        }

    # Merge into OUT_JSON preserving epistemic section if already present
    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    existing.update({
        "dataset_integrity": integrity,
        "within_model_geometry": within_geom,
        "c5_geometry": c5_geom,
        "bow_valence_matched": {
            "all_60_20split_mean": round(float(np.mean(bow_vm_accs)), 4),
            "all_60_20split_std": round(float(np.std(bow_vm_accs)), 4),
            "both_neg_30_20split_mean": round(float(np.mean(bow_vm_neg)), 4),
            "both_pos_30_20split_mean": round(float(np.mean(bow_vm_pos)), 4),
        },
        "transfer_house_valence_controlled": transfer_house,
        "transfer_valence_matched_60": transfer_vm,
    })
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(existing, indent=2))
    os.replace(tmp, OUT_JSON)
    print(f"SAVED {OUT_JSON}", flush=True)
    return existing


def run_epistemic_valence_mlx():
    """Native-layer epistemic valence control on Google Gemma 4 E4B (bf16 MLX, L17/L25) and Alibaba Qwen3.5-9B
    (8-bit MLX, L12/L19) comparing CORRECTNESS, SENTIMENT_PERP, HOUSE_PERP, and HOUSE_PERP_BOTH
    (orthogonal to BOTH CORRECTNESS and SENTIMENT), plus paired bootstrap incremental lift over -CORRECTNESS!"""
    import mlx.core as mx
    from mlx_lm import load
    from aion_8b import audit233_epistemic as ae, audit233_mac as am, aion_data230 as bd, sceptic_ctrl230 as sc

    sg_rows, ov_rows, _ = load_valence_data()
    house_pairs = [p for p in AR.load_arch_pairs() if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([p["cat"] for p in house_pairs])
    all_pairs = am.CLEAN_FALSE_PREMISE_12 + am.CLEAN_TENSE_MATCHED_UNK_12
    idx_24 = [(2 * i, 2 * i + 1) for i in range(24)]

    def score_with_sentiment(V_cal_tf, V_cal_house, V_cal_sg, V_eval, u_corr, u_house, u_sent, n_rand=500, seed=234):
        rng = np.random.default_rng(seed)
        uc = AR.unit(u_corr)
        uh = AR.unit(u_house)
        us = AR.unit(u_sent)

        # Orthogonal components
        uh_p_c = _project_out(uh, _orthonormal_basis([uc]))
        us_p_c = _project_out(us, _orthonormal_basis([uc]))
        Q_cs = _orthonormal_basis([uc, us])
        uh_p_cs = _project_out(uh, Q_cs)
        assert abs(np.dot(uc, uh_p_cs)) < 1e-5 and abs(np.dot(us, uh_p_cs)) < 1e-5

        V_cal = np.concatenate([V_cal_tf, V_cal_house, V_cal_sg], axis=0)
        def cal_z(V_ev, u):
            c_proj = -(V_cal @ u)
            mu, sd = float(c_proj.mean()), max(1e-6, float(c_proj.std()))
            return (-(V_ev @ u) - mu) / sd

        zc = cal_z(V_eval, uc)
        zhp = cal_z(V_eval, uh_p_c)
        zsp = cal_z(V_eval, us_p_c)
        zhpcs = cal_z(V_eval, uh_p_cs)

        comb_hp = 0.5 * zc + 0.5 * zhp
        comb_sp = 0.5 * zc + 0.5 * zsp
        comb_hpcs = 0.5 * zc + 0.5 * zhpcs

        D = V_eval.shape[1]
        Q3 = _orthonormal_basis([uc, uh, us])
        def sample_ortho():
            R = rng.normal(size=(n_rand, D))
            R = R - (R @ Q3) @ Q3.T
            return R / np.maximum(1e-12, np.linalg.norm(R, axis=1, keepdims=True))
        R1 = sample_ortho()
        R2 = sample_ortho()
        R2 = R2 - np.sum(R1 * R2, axis=1, keepdims=True) * R1
        R2 = R2 / np.maximum(1e-12, np.linalg.norm(R2, axis=1, keepdims=True))

        C1 = -(V_cal @ R1.T); mu1, sd1 = C1.mean(0, keepdims=True), np.maximum(1e-6, C1.std(0, keepdims=True))
        C2 = -(V_cal @ R2.T); mu2, sd2 = C2.mean(0, keepdims=True), np.maximum(1e-6, C2.std(0, keepdims=True))
        E1 = (-(V_eval @ R1.T) - mu1) / sd1
        E2 = (-(V_eval @ R2.T) - mu2) / sd2
        Ec = 0.5 * E1 + 0.5 * E2

        y = np.array([0, 1] * 24, dtype=int)
        r1_aucs = np.array([ae._auroc(y, E1[:, j]) for j in range(n_rand)])
        rc_aucs = np.array([ae._auroc(y, Ec[:, j]) for j in range(n_rand)])
        p1 = lambda val: round(float((1 + np.sum(r1_aucs >= val)) / (n_rand + 1)), 4)
        pc = lambda val: round(float((1 + np.sum(rc_aucs >= val)) / (n_rand + 1)), 4)

        # Pair-bootstrap 95% CI for incremental lift: Combined(CORR, HOUSE_PERP_BOTH) - neg_corr
        def boot_diff(sA, sB, n_boot=2000):
            diffs = []
            for _ in range(n_boot):
                pick = rng.integers(0, 24, 24)
                idx = np.concatenate([[2 * p, 2 * p + 1] for p in pick])
                diffs.append(ae._auroc(y, sA[idx]) - ae._auroc(y, sB[idx]))
            return round(float(ae._auroc(y, sA) - ae._auroc(y, sB)), 4), [
                round(float(np.percentile(diffs, 2.5)), 4),
                round(float(np.percentile(diffs, 97.5)), 4),
            ]

        auc_c = ae._auroc(y, zc)
        auc_hp = ae._auroc(y, zhp)
        auc_sp = ae._auroc(y, zsp)
        auc_hpcs = ae._auroc(y, zhpcs)
        auc_chp = ae._auroc(y, comb_hp)
        auc_csp = ae._auroc(y, comb_sp)
        auc_chpcs = ae._auroc(y, comb_hpcs)
        d_hp, ci_hp = boot_diff(comb_hp, zc)
        d_hpcs, ci_hpcs = boot_diff(comb_hpcs, zc)
        d_vs_sp, ci_vs_sp = boot_diff(comb_hp, comb_sp)

        return {
            "cos_corr_house": round(float(np.dot(uc, uh)), 4),
            "cos_corr_sent": round(float(np.dot(uc, us)), 4),
            "cos_house_sent": round(float(np.dot(uh, us)), 4),
            "neg_corr": {"auroc": round(auc_c, 4), "p": p1(auc_c)},
            "neg_sent_perp_corr": {"auroc": round(auc_sp, 4), "p": p1(auc_sp)},
            "neg_house_perp_corr": {"auroc": round(auc_hp, 4), "p": p1(auc_hp)},
            "neg_house_perp_corr_and_sent": {"auroc": round(auc_hpcs, 4), "p": p1(auc_hpcs)},
            "comb_corr_sent_perp": {"auroc": round(auc_csp, 4), "p": pc(auc_csp)},
            "comb_corr_house_perp": {"auroc": round(auc_chp, 4), "p": pc(auc_chp)},
            "comb_corr_house_perp_both": {"auroc": round(auc_chpcs, 4), "p": pc(auc_chpcs)},
            "lift_comb_hp_minus_corr": {"diff": d_hp, "ci95": ci_hp},
            "lift_comb_hp_both_minus_corr": {"diff": d_hpcs, "ci95": ci_hpcs},
            "lift_comb_hp_minus_comb_sent": {"diff": d_vs_sp, "ci95": ci_vs_sp},
        }

    epi_res = {}
    for mkey, layer_pair in [("gemma4_e4b", (17, 25)), ("qwen35_9b", (12, 19))]:
        spec = PD.MODEL_SPECS[mkey]
        mx.set_cache_limit(2 << 30)
        print(f"VAL234 EPISTEMIC {mkey}: loading {spec['path']} ...", flush=True)
        model, tok = load(spec["path"])
        layers = PD._find_layers(model)
        l_a, l_b = layer_pair
        blk_a, blk_b = layers[l_a], layers[l_b]
        rec = {}
        patched = {}
        for blk in layers:
            cls = type(blk)
            if cls not in patched:
                orig = cls.__call__
                patched[cls] = orig
                def make_hook(orig_fn):
                    def hooked(self, *a, **k):
                        out = orig_fn(self, *a, **k)
                        if self is blk_a:
                            h = out[0] if isinstance(out, tuple) else out
                            rec[l_a] = h[0, -1].astype(mx.float32)
                        elif self is blk_b:
                            h = out[0] if isinstance(out, tuple) else out
                            rec[l_b] = h[0, -1].astype(mx.float32)
                        return out
                    return hooked
                cls.__call__ = make_hook(orig)

        try:
            def get_last(txt, templated=False):
                if templated:
                    ids = tok.encode(txt, add_special_tokens=False)[:512]
                else:
                    ids = PD._text_ids_mlx(tok, txt)[:512]
                rec.clear()
                _ = model(mx.array([ids]))
                va, vb = rec[l_a], rec[l_b]
                mx.eval(va, vb)
                return {l_a: np.array(va, dtype=np.float32), l_b: np.array(vb, dtype=np.float32)}

            diffs_c, cal_tf = {l_a: [], l_b: []}, {l_a: [], l_b: []}
            for ts, fs in sc.TRUE_FALSE_20:
                vt, vf = get_last(ts), get_last(fs)
                for li in layer_pair:
                    diffs_c[li].append(vt[li] - vf[li])
                    cal_tf[li].extend([vt[li], vf[li]])
            u_corr = {li: AR.unit(np.stack(diffs_c[li]).mean(0)) for li in layer_pair}

            diffs_h, cal_h = {l_a: [], l_b: []}, {l_a: [], l_b: []}
            for p in house_pairs:
                vd, vn = get_last(p["done"]), get_last(p["neglected"])
                for li in layer_pair:
                    diffs_h[li].append(vd[li] - vn[li])
                    cal_h[li].extend([vd[li], vn[li]])
            u_house = {li: AR.house_dir(np.stack(diffs_h[li]), house_cats) for li in layer_pair}

            diffs_s, cal_s = {l_a: [], l_b: []}, {l_a: [], l_b: []}
            for r in sg_rows:
                vp, vn = get_last(r["pos"]), get_last(r["neg"])
                for li in layer_pair:
                    diffs_s[li].append(vp[li] - vn[li])
                    cal_s[li].extend([vp[li], vn[li]])
            u_sent = {li: AR.unit(np.stack(diffs_s[li]).mean(0)) for li in layer_pair}

            m_out = {}
            for pname, sys_p in [("neutral_prompt", sc.NEUTRAL_SYS), ("aion_prompt", bd.BUDDY_SYSTEM_PROMPT)]:
                ev_vecs = {l_a: [], l_b: []}
                for kq, uq in all_pairs:
                    for q in (kq, uq):
                        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": q}]
                        try:
                            pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        except TypeError:
                            pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        vs = get_last(pr, templated=True)
                        for li in layer_pair:
                            ev_vecs[li].append(vs[li])
                m_out[pname] = {
                    f"L{li}": score_with_sentiment(
                        np.stack(cal_tf[li]), np.stack(cal_h[li]), np.stack(cal_s[li]),
                        np.stack(ev_vecs[li]), u_corr[li], u_house[li], u_sent[li]
                    )
                    for li in layer_pair
                }
            epi_res[mkey] = m_out
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            rec.clear()
            del layers, blk_a, blk_b, model, tok
            gc.collect()
            mx.clear_cache()

    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    existing["epistemic_sentiment_control"] = epi_res
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(existing, indent=2))
    os.replace(tmp, OUT_JSON)
    print(f"SAVED EPISTEMIC VALENCE CONTROL -> {OUT_JSON}", flush=True)
    return epi_res


def run_unit_test():
    info = verify_dataset_integrity()
    # Verify Gram-Schmidt double projection mathematical exactness
    rng = np.random.default_rng(7)
    u1, u2, u3 = AR.unit(rng.normal(size=64)), AR.unit(rng.normal(size=64)), AR.unit(rng.normal(size=64))
    Q = _orthonormal_basis([u1, u2])
    u3p = _project_out(u3, Q)
    assert abs(np.dot(u3p, u1)) < 1e-10 and abs(np.dot(u3p, u2)) < 1e-10
    D = rng.normal(size=(30, 64))
    Dp = _project_out_rows(D, Q)
    assert np.max(np.abs(Dp @ Q)) < 1e-10
    print(f"VALENCE_CTRL234_TEST_OK {info}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        run_unit_test()
    elif mode == "extract":
        keys = sys.argv[2:] if len(sys.argv) > 2 else list(PD.ALL_KEYS)
        for k in keys:
            extract_model(k)
    elif mode == "analyze":
        res = analyze_7model()
        print("C5_GEOM:", json.dumps(res["c5_geometry"]))
        print("BOW_VM:", json.dumps(res["bow_valence_matched"]))
    elif mode == "epistemic":
        res = run_epistemic_valence_mlx()
        print("EPISTEMIC_DONE keys=", list(res.keys()))
    else:
        raise SystemExit("Usage: valence_ctrl234.py [test|extract|analyze|epistemic]")
