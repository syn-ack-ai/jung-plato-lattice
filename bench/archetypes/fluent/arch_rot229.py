"""Addendum 229 Stage 1 (Amendment 229a): Zero-Shot Plato's Key Archetype Rotation across 7 LLM Families.

Tests whether higher-order Jungian/narrative coherence directions (HOUSE & SELF2: done minus neglected)
measured in ONE model (Gemma 4 E4B) or averaged across 5 Consensus families teleport ZERO-SHOT via
neutral-language closed-form Procrustes rotations (fit strictly on gut + proc + chat, ZERO myth/archetype
text!) into six unrelated model families: Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, looped Inkling 2 Mini,
blind Phi-4-mini, and blind OLMo 3 7B.

Amendment 229a (after independent code review 584b817d, before any extraction/run):
  - Fixed neutral domain filter to exact ("gut", "proc", "chat") with assertions.
  - Primary evaluation uses STRICTLY DISJOINT cross-story splits (stratified 2-fold CV across stories per
    category: fit vec on fold A in source model(s), evaluate cos/centered_signflip_p/pair_acc on unseen
    fold B stories in target model), eliminating the N <= D shared-stimulus null inflation!
  - Saves exact per-model token lengths (done_len, neg_len), reports correlation with delta_tok_len,
    length-residualized transfer (regressing out delta_tok_len & delta_log_tok_len), and explicitly
    highlights length-reversed controls (perfection & HOUSE -> unseen items3).
  - Anchors pts_full_C5 directly to saved targets_train.npz["consensus_256"] (asserting cos > 0.99999)
    so exported U_jung_c5 is guaranteed bit-aligned with Stage 2 targets.
  - Hardened index guards, MLX memory release (del layers, tgt_blk before clear_cache), --no-webui +
    proc.poll() in llama-server health loop, and separate C4 frame for strict 4-fam vs 5-fam comparison.

Usage (Mac Studio):
  python fluent/arch_rot229.py test
  python   fluent/arch_rot229.py extract [KEY...]
  python fluent/arch_rot229.py analyze
"""
import gc, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO = HERE / "plato"
sys.path.insert(0, str(HERE))
import plato_data as PD  # noqa: E402

HOUSE_CATS = ("duty", "craft", "diagnosis", "order", "healing", "self", "perfection")
ALL_CATS = HOUSE_CATS + ("self2",)
NEUTRAL_DOMAINS = ("gut", "proc", "chat")


def load_arch_pairs():
    """Load the 312 public paired stories (done vs neglected) across the 8 categories."""
    pairs = []
    sources = [
        (ARCH / "archstory" / "stories.json", ("duty", "craft", "diagnosis", "order", "healing")),
        (ARCH / "selfshape" / "pairs" / "stories.json", ("self", "perfection")),
        (ARCH / "selfshape" / "pairs2" / "stories.json", ("self2",)),
    ]
    for path, cats in sources:
        data = json.loads(path.read_text())
        for cat in cats:
            rows = data[cat]
            for idx, r in enumerate(rows):
                done = r["done"].strip()
                neg = r["neglected"].strip()
                assert done and neg, f"empty text in {cat}[{idx}]"
                pairs.append({
                    "id": f"{cat}_{idx:03d}",
                    "cat": cat,
                    "setting": r.get("setting", ""),
                    "done": done,
                    "neglected": neg,
                })
    return pairs


def _save_arch_npz(out_npz, pairs, dl, dm, nl, nm, d_lens, n_lens, ok_flags, layer_idx, num_layers):
    dl = np.stack(dl, axis=0)
    dm = np.stack(dm, axis=0)
    nl = np.stack(nl, axis=0)
    nm = np.stack(nm, axis=0)
    assert np.isfinite(dl).all() and np.isfinite(dm).all() and np.isfinite(nl).all() and np.isfinite(nm).all(), "non-finite arch states!"
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        out_npz,
        done_last=pack(dl), done_mean=pack(dm),
        neg_last=pack(nl), neg_mean=pack(nm),
        done_len=np.array(d_lens, dtype=np.int32),
        neg_len=np.array(n_lens, dtype=np.int32),
        ok=np.array(ok_flags, dtype=bool),
        cats=np.array([p["cat"] for p in pairs]),
        ids=np.array([p["id"] for p in pairs]),
        layer_idx=np.int32(layer_idx), num_layers=np.int32(num_layers),
    )


def _extract_llamacpp_pairs(key, spec, pairs, out_npz):
    port = spec["port"]
    srv = str(Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server")))
    # Ensure port is not already held by a stale server
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
            raise RuntimeError(f"port {port} already serving before starting {key} llama-server!")
    except urllib.error.URLError:
        pass

    cmd = [
        srv, "-m", spec["path"], "--port", str(port), "--host", "127.0.0.1",
        "--embeddings", "--pooling", "none", "--embd-normalize", "-1",
        "--no-webui", "-c", "2048", "-b", "2048", "-ub", "2048", "-ngl", "99",
    ]
    t0 = time.time()
    print(f"ARCH229 EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post(ep, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{ep}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    done_last, done_mean, neg_last, neg_mean, d_lens, n_lens, ok_flags = [], [], [], [], [], [], []
    try:
        ready = False
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                raise RuntimeError(f"{key} llama-server exited early with rc={proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)
        assert ready, f"{key} llama-server failed to become healthy"

        def embed_one(text):
            toks = post("/tokenize", {"content": text, "add_special": True})["tokens"]
            if not (2 <= len(toks) <= 512):
                return None, None, 0
            emb = post("/embedding", {"content": text})
            arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != len(toks):
                return None, None, 0
            return arr[-1].copy(), (arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0].copy()), len(toks)

        D = None
        for i, p in enumerate(pairs):
            dl, dm, dln = embed_one(p["done"])
            nl_v, nm_v, nln = embed_one(p["neglected"])
            if dl is None or nl_v is None:
                if D is None:
                    e0, _, _ = embed_one("Hello world.")
                    assert e0 is not None, f"{key} dummy probe failed"
                    D = len(e0)
                z = np.zeros(D, dtype=np.float32)
                done_last.append(z); done_mean.append(z); neg_last.append(z); neg_mean.append(z)
                d_lens.append(0); n_lens.append(0); ok_flags.append(False)
            else:
                D = len(dl)
                done_last.append(dl); done_mean.append(dm); neg_last.append(nl_v); neg_mean.append(nm_v)
                d_lens.append(dln); n_lens.append(nln); ok_flags.append(True)
            if (i + 1) % 100 == 0 or i == len(pairs) - 1:
                print(f"  ARCH229 {key}: {i + 1}/{len(pairs)} ok={sum(ok_flags)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    _save_arch_npz(out_npz, pairs, done_last, done_mean, neg_last, neg_mean, d_lens, n_lens, ok_flags, -1, 24)
    print(f"ARCH229_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(pairs)} in {time.time() - t0:.0f} s", flush=True)


def extract_model(key):
    assert key in PD.MODEL_SPECS, f"unknown model key {key}"
    out_npz = PLATO / f"arch229_{key}.npz"
    if out_npz.exists():
        print(f"ARCH229 EXTRACT {key}: already exists ({out_npz}), skipping", flush=True)
        return
    pairs = load_arch_pairs()
    spec = PD.MODEL_SPECS[key]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp_pairs(key, spec, pairs, out_npz)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"ARCH229 EXTRACT {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = PD._find_layers(model)
    nl = len(layers)
    tgt_idx = max(1, min(nl - 2, int(round(spec["depth_frac"] * (nl - 1)))))
    tgt_blk = layers[tgt_idx]
    print(f"ARCH229 EXTRACT {key}: L={nl}, hooking layer {tgt_idx} ({tgt_idx / (nl - 1):.1%})", flush=True)

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

    done_last, done_mean, neg_last, neg_mean, d_lens, n_lens, ok_flags = [], [], [], [], [], [], []
    D = None
    try:
        def embed_one(text):
            ids = PD._text_ids_mlx(tok, text)
            if not (2 <= len(ids) <= 512):
                return None, None, 0
            rec_buf.clear()
            _ = model(mx.array([ids]))
            assert len(rec_buf) == 1, f"{key} hook fired {len(rec_buf)} times!"
            h = rec_buf[0].astype(mx.float32)
            assert h.ndim == 2 and h.shape[0] == len(ids), f"{key} shape mismatch {h.shape} vs {len(ids)}"
            hl = h[-1]
            hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
            mx.eval(hl, hm)
            rec_buf.clear()
            return np.array(hl, dtype=np.float32), np.array(hm, dtype=np.float32), len(ids)

        for i, p in enumerate(pairs):
            dl, dm, dln = embed_one(p["done"])
            nl_v, nm_v, nln = embed_one(p["neglected"])
            if dl is None or nl_v is None:
                if D is None:
                    e0, _, _ = embed_one("Hello world.")
                    assert e0 is not None, f"{key} dummy probe failed"
                    D = len(e0)
                z = np.zeros(D, dtype=np.float32)
                done_last.append(z); done_mean.append(z); neg_last.append(z); neg_mean.append(z)
                d_lens.append(0); n_lens.append(0); ok_flags.append(False)
            else:
                D = len(dl)
                done_last.append(dl); done_mean.append(dm); neg_last.append(nl_v); neg_mean.append(nm_v)
                d_lens.append(dln); n_lens.append(nln); ok_flags.append(True)
            if (i + 1) % 100 == 0 or i == len(pairs) - 1:
                print(f"  ARCH229 {key}: {i + 1}/{len(pairs)} ok={sum(ok_flags)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()

    _save_arch_npz(out_npz, pairs, done_last, done_mean, neg_last, neg_mean, d_lens, n_lens, ok_flags, tgt_idx, nl)
    print(f"ARCH229_EXTRACT_DONE {key}: ok={sum(ok_flags)}/{len(pairs)} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- Zero-Shot Rotation Math

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(1e-12, float(np.linalg.norm(v)))


def fit_pca_params(X_tr, d_out=256):
    X_tr = np.asarray(X_tr, dtype=np.float64)
    assert len(X_tr) >= d_out, f"insufficient rows {len(X_tr)} for d_out={d_out}"
    mu = X_tr.mean(axis=0, keepdims=True)
    Xc = X_tr - mu
    cov = (Xc.T @ Xc) / float(len(Xc))
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:d_out]
    V = vecs[:, idx]
    Z_tr = Xc @ V
    rms = max(1e-12, float(np.sqrt(np.mean(Z_tr ** 2))))
    return mu, V, rms, Z_tr / rms


def apply_pca_params(X, mu, V, rms):
    return ((np.asarray(X, dtype=np.float64) - mu) @ V) / rms


def _load_npz_dict(path):
    with np.load(path, allow_pickle=True) as d:
        return {fld: d[fld] for fld in d.files}


def fit_neutral_or_full_rotations(pool="pts", neutral_only=True, d_plato=256):
    """Fit 256-dim PCA + GPA Consensus + blind Procrustes maps STRICTLY on training passages.
    neutral_only=True uses ONLY ("gut", "proc", "chat") — ZERO myth/folklore/archetype text!
    When pool=="pts" and neutral_only=False, anchors directly to saved targets_train.npz["consensus_256"]
    so exported U_jung_c5 is guaranteed bit-aligned with Stage 2 targets!"""
    tk = PD.load_bpe8k()
    bpe_enc = lambda s: tk.encode(s).ids
    train_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    raw = {k: _load_npz_dict(PLATO / f"raw_{k}.npz") for k in PD.ALL_KEYS}

    story_keep = []
    used_domains = set()
    pt_masks = {k: [] for k in PD.ALL_KEYS}
    for r_i, r in enumerate(train_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = PD.boundary_token_indices(bpe_enc, r["segs"])
        dom_ok = (r["domain"] in NEUTRAL_DOMAINS) if neutral_only else True
        ok = (
            dom_ok and 2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raw[k]["tr_ok"][r_i]) and int(raw[k]["tr_cnt"][r_i]) == n_seg for k in PD.ALL_KEYS)
        )
        story_keep.append(ok)
        if ok:
            used_domains.add(r["domain"])
        for k in PD.ALL_KEYS:
            pt_masks[k].extend([ok] * int(raw[k]["tr_cnt"][r_i]))

    story_mask = np.array(story_keep, dtype=bool)
    assert story_mask.sum() > 500, f"too few training passages matched ({story_mask.sum()}); domains={used_domains}"
    if neutral_only:
        assert "myth" not in used_domains and used_domains.issubset(set(NEUTRAL_DOMAINS)), f"leak domain: {used_domains}"

    pca_params = {}
    Z_tr = {}
    for k in PD.ALL_KEYS:
        if pool == "pts":
            pm = np.array(pt_masks[k], dtype=bool)
            assert len(pm) == len(raw[k]["tr_pts"]), f"{k} pt_mask len {len(pm)} != {len(raw[k]['tr_pts'])}"
            X_k = raw[k]["tr_pts"][pm].astype(np.float64)
        else:
            X_k = raw[k]["tr_story"][story_mask].astype(np.float64)
        mu, V, rms, Z = fit_pca_params(X_k, d_out=d_plato)
        pca_params[k] = (mu, V, rms)
        Z_tr[k] = Z

    cons5_Z = [Z_tr[k] for k in PD.CONSENSUS_KEYS]
    cons4_Z = [Z_tr[k] for k in PD.MLX_CONSENSUS_KEYS]
    C5_tr, _, _, _, _ = PD.generalized_procrustes_consensus(cons5_Z, cons5_Z, n_iters=10)
    C4_tr, _, _, _, _ = PD.generalized_procrustes_consensus(cons4_Z, cons4_Z, n_iters=10)

    # Anchor pts_full_C5 explicitly to saved targets_train.npz["consensus_256"] (prevent BLAS sign flips!)
    if pool == "pts" and not neutral_only:
        saved = _load_npz_dict(PLATO / "targets_train.npz")
        saved_c5 = saved["consensus_256"].astype(np.float64)
        assert saved_c5.shape == C5_tr.shape, f"C5 shape mismatch {C5_tr.shape} vs {saved_c5.shape}"
        c_align = float(np.sum(C5_tr * saved_c5) / (np.linalg.norm(C5_tr) * np.linalg.norm(saved_c5)))
        # Verify subspace equivalence (Procrustes fit onto saved_c5 should match to >0.9999 even if BLAS flipped signs)
        aligned_c5 = PD.apply_procrustes_map(C5_tr, PD.fit_procrustes_map(C5_tr, saved_c5))
        c_proc = float(np.sum(aligned_c5 * saved_c5) / (np.linalg.norm(aligned_c5) * np.linalg.norm(saved_c5)))
        assert c_proc > 0.9999, f"recomputed C5 diverged from targets_train.npz (proc_cos={c_proc:.6f})"
        C5_tr = saved_c5  # exact saved canonical frame!
        print(f"  Anchored pts_full_C5 to targets_train.npz (raw_cos={c_align:.6f}, proc_cos={c_proc:.6f})", flush=True)

    to_c5 = {k: PD.fit_procrustes_map(Z_tr[k], C5_tr) for k in PD.ALL_KEYS}
    to_c4 = {k: PD.fit_procrustes_map(Z_tr[k], C4_tr) for k in PD.ALL_KEYS}
    return {
        "pool": pool,
        "neutral_only": neutral_only,
        "used_domains": sorted(used_domains),
        "n_passages": int(story_mask.sum()),
        "n_vecs": int(len(C5_tr)),
        "pca": pca_params,
        "to_c5": to_c5,
        "to_c4": to_c4,
    }


def map_diffs(done_raw, neg_raw, rot_bundle, key, frame="to_c5"):
    mu, V, rms = rot_bundle["pca"][key]
    params = rot_bundle[frame][key]
    z_d = apply_pca_params(done_raw, mu, V, rms)
    z_n = apply_pca_params(neg_raw, mu, V, rms)
    c_d = PD.apply_procrustes_map(z_d, params)
    c_n = PD.apply_procrustes_map(z_n, params)
    return c_d, c_n, c_d - c_n


def house_dir(diffs, cats):
    ds = []
    for c in HOUSE_CATS:
        m = (cats == c)
        assert int(m.sum()) >= 1, f"empty HOUSE category {c}"
        ds.append(unit(diffs[m].mean(axis=0)))
    return unit(np.mean(ds, axis=0))


def fit_length_ols(diffs, d_len, n_len):
    """Fit OLS Y ~ 1 + (dl - nl) + (log(dl) - log(nl)) strictly on a training subset; return length slopes beta_len."""
    dl = np.asarray(d_len, dtype=np.float64)
    nl = np.asarray(n_len, dtype=np.float64)
    X = np.column_stack([np.ones(len(dl)), dl - nl, np.log(np.maximum(1.0, dl)) - np.log(np.maximum(1.0, nl))])
    coef, _, _, _ = np.linalg.lstsq(X, diffs, rcond=None)
    return coef[1:]  # (2, D) length slopes only (preserves intercept = equal-length semantic mean!)


def apply_length_ols(diffs, d_len, n_len, beta_len):
    """Subtract pre-fit length-dependent component beta_len from paired diffs without touching held-out labels."""
    dl = np.asarray(d_len, dtype=np.float64)
    nl = np.asarray(n_len, dtype=np.float64)
    X_len = np.column_stack([dl - nl, np.log(np.maximum(1.0, dl)) - np.log(np.maximum(1.0, nl))])
    return diffs - (X_len @ beta_len)


def residualize_length(diffs, d_len, n_len):
    """Regress out linear & log token-length delta from paired diffs (fit strictly on passed rows)."""
    beta = fit_length_ols(diffs, d_len, n_len)
    return apply_length_ols(diffs, d_len, n_len, beta)


def eval_disjoint_2fold_lenres(src_diffs_list, src_lens_list, tgt_diffs, tgt_d_len, tgt_n_len, cats, folds, is_house=True, seed=229):
    """Strictly fold-local length-residualized 2-fold CV: fits length OLS slopes ONLY on train fold (both source
    and target), then residualizes test fold using train-fold slopes only (zero test-fold OLS leakage!)."""
    all_scores = np.zeros(len(tgt_diffs), dtype=np.float64)
    cos_folds, p_pos_folds = [], []
    for te_f in (0, 1):
        m_tr = (folds != te_f)
        m_te = (folds == te_f)
        vecs = []
        for sd, (sdl, snl) in zip(src_diffs_list, src_lens_list):
            sd_tr = residualize_length(sd[m_tr], sdl[m_tr], snl[m_tr])
            vecs.append(house_dir(sd_tr, cats[m_tr]) if is_house else unit(sd_tr.mean(axis=0)))
        v_src = unit(np.mean(vecs, axis=0))
        beta_tgt = fit_length_ols(tgt_diffs[m_tr], tgt_d_len[m_tr], tgt_n_len[m_tr])
        tgt_te_lr = apply_length_ols(tgt_diffs[m_te], tgt_d_len[m_te], tgt_n_len[m_te], beta_tgt)
        c_val, _, p_pos, _ = centered_signflip_p(v_src, tgt_te_lr, cats[m_te] if is_house else None, is_house=is_house, n_draws=1000, seed=seed + te_f)
        cos_folds.append(c_val)
        p_pos_folds.append(p_pos)
        all_scores[m_te] = tgt_te_lr @ v_src
    hits = (all_scores > 0.0).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(2000, len(hits)))
    bs = hits[idx].mean(axis=1)
    return {
        "cos": round(float(np.mean(cos_folds)), 4),
        "p_pos_max": round(float(max(p_pos_folds)), 4),
        "acc": round(float(hits.mean()), 4),
        "acc_ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
        "scores": all_scores,
    }


def make_stratified_2fold(cats):
    """Deterministic stratified 2-fold split across stories within each category (strictly disjoint!)."""
    folds = np.zeros(len(cats), dtype=np.int32)
    for c in sorted(set(cats)):
        idx = np.flatnonzero(cats == c)
        # Alternating indices avoid ordering drift
        folds[idx[1::2]] = 1
    return folds


def centered_signflip_p(vec_src, diffs_tgt, cats_tgt=None, is_house=False, n_draws=1000, seed=229):
    """Properly CENTERED sign-flip null (must only be called when vec_src is INDEPENDENT of diffs_tgt!).
    Returns (obs_cos, p_two_sided, p_pos_one_sided, med_abs_null)."""
    rng = np.random.default_rng(seed)
    vec_src = unit(vec_src)
    if is_house:
        obs_vec = house_dir(diffs_tgt, cats_tgt)
        centered = {c: diffs_tgt[cats_tgt == c] - diffs_tgt[cats_tgt == c].mean(axis=0, keepdims=True) for c in HOUSE_CATS}
        obs = float(vec_src @ obs_vec)
        null = np.empty(n_draws, dtype=np.float64)
        for i in range(n_draws):
            ds = []
            for c in HOUSE_CATS:
                x = centered[c]
                s = rng.choice([-1.0, 1.0], size=len(x))[:, None]
                ds.append(unit((s * x).mean(axis=0)))
            null[i] = float(vec_src @ unit(np.mean(ds, axis=0)))
    else:
        obs_vec = unit(diffs_tgt.mean(axis=0))
        obs = float(vec_src @ obs_vec)
        xc = diffs_tgt - diffs_tgt.mean(axis=0, keepdims=True)
        null = np.empty(n_draws, dtype=np.float64)
        for i in range(n_draws):
            s = rng.choice([-1.0, 1.0], size=len(xc))[:, None]
            null[i] = float(vec_src @ unit((s * xc).mean(axis=0)))
    p_two = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_draws + 1))
    p_pos = float((np.sum(null >= obs) + 1) / (n_draws + 1))
    return obs, p_two, p_pos, float(np.median(np.abs(null)))


def pair_acc_boot(vec, diffs, n_draws=2000, seed=229):
    scores = diffs @ unit(vec)
    hits = (scores > 0.0).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(n_draws, len(hits)))
    bs = hits[idx].mean(axis=1)
    return float(hits.mean()), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], scores


def eval_disjoint_2fold(src_diffs_list, tgt_diffs, cats, folds, is_house=True, seed=229):
    """STRICTLY DISJOINT 2-fold cross-story evaluation (fixes High #2 shared-stimulus N <= D inflation!):
    Fit direction on fold A in source model(s) -> evaluate cos, centered_signflip_p, and pair_acc on UNSEEN fold B in target model."""
    all_scores = np.zeros(len(tgt_diffs), dtype=np.float64)
    cos_folds, p_pos_folds = [], []
    for te_f in (0, 1):
        m_tr = (folds != te_f)
        m_te = (folds == te_f)
        if is_house:
            vecs = [house_dir(sd[m_tr], cats[m_tr]) for sd in src_diffs_list]
            v_src = unit(np.mean(vecs, axis=0))
            c_val, _, p_pos, _ = centered_signflip_p(v_src, tgt_diffs[m_te], cats[m_te], is_house=True, n_draws=1000, seed=seed + te_f)
        else:
            vecs = [unit(sd[m_tr].mean(axis=0)) for sd in src_diffs_list]
            v_src = unit(np.mean(vecs, axis=0))
            c_val, _, p_pos, _ = centered_signflip_p(v_src, tgt_diffs[m_te], is_house=False, n_draws=1000, seed=seed + te_f)
        cos_folds.append(c_val)
        p_pos_folds.append(p_pos)
        all_scores[m_te] = tgt_diffs[m_te] @ v_src
    hits = (all_scores > 0.0).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(2000, len(hits)))
    bs = hits[idx].mean(axis=1)
    return {
        "cos": round(float(np.mean(cos_folds)), 4),
        "p_pos_max": round(float(max(p_pos_folds)), 4),
        "acc": round(float(hits.mean()), 4),
        "acc_ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
        "scores": all_scores,
    }


def analyze():
    t0 = time.time()
    arch = {}
    for k in PD.ALL_KEYS:
        p = PLATO / f"arch229_{k}.npz"
        assert p.exists(), f"missing {p}; run extract first!"
        arch[k] = _load_npz_dict(p)

    # Verify identical ids/cats across all 7 models and build strict mask
    ref_ids = [str(x) for x in arch[PD.ALL_KEYS[0]]["ids"]]
    ref_cats = [str(x) for x in arch[PD.ALL_KEYS[0]]["cats"]]
    ok_all = np.ones(len(ref_ids), dtype=bool)
    for k in PD.ALL_KEYS:
        assert [str(x) for x in arch[k]["ids"]] == ref_ids, f"id mismatch in {k}"
        assert [str(x) for x in arch[k]["cats"]] == ref_cats, f"cat mismatch in {k}"
        ok_all &= arch[k]["ok"].astype(bool)
    cats = np.array(ref_cats)[ok_all]
    for c in ALL_CATS:
        assert int((cats == c).sum()) >= 4, f"too few valid pairs in category {c}"
    m_house = np.isin(cats, HOUSE_CATS)
    m_self2 = (cats == "self2")
    m_perf = (cats == "perfection")
    folds_all = make_stratified_2fold(cats)
    folds_house = folds_all[m_house]
    folds_self2 = folds_all[m_self2]
    print(f"ARCH229 strict-aligned pairs across all 7 models: {int(ok_all.sum())}/{len(ok_all)} (HOUSE={int(m_house.sum())}, SELF2={int(m_self2.sum())}, perf={int(m_perf.sum())})", flush=True)

    # Load items3 true vs foil with strict ev_ok & ev_cnt > 0 guards!
    ev_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl")]
    raw_ev = {k: _load_npz_dict(PLATO / f"raw_{k}.npz") for k in PD.ALL_KEYS}
    by_item = {}
    for idx, r in enumerate(ev_rows):
        if r["domain"] in ("items3_myth", "items3_mech") and r.get("role") in ("true", "foil"):
            by_item.setdefault((r["domain"], r["item_id"]), {})[r["role"]] = idx
    i3_pairs = []
    for (d, iid), mp in sorted(by_item.items()):
        if "true" not in mp or "foil" not in mp:
            continue
        ti, fi = mp["true"], mp["foil"]
        if all(bool(raw_ev[k]["ev_ok"][ti]) and bool(raw_ev[k]["ev_ok"][fi])
               and int(raw_ev[k]["ev_cnt"][ti]) > 0 and int(raw_ev[k]["ev_cnt"][fi]) > 0 for k in PD.ALL_KEYS):
            i3_pairs.append((d, iid, ti, fi))
    assert len(i3_pairs) == 200, f"expected 200 items3 pairs, got {len(i3_pairs)}"
    i3_tr_idx = np.array([x[2] for x in i3_pairs], dtype=np.int32)
    i3_fl_idx = np.array([x[3] for x in i3_pairs], dtype=np.int32)
    i3_is_myth = np.array([x[0] == "items3_myth" for x in i3_pairs], dtype=bool)
    i3_last_idx = {}
    for k in PD.ALL_KEYS:
        cnt = raw_ev[k]["ev_cnt"]
        assert int(cnt.sum()) == len(raw_ev[k]["ev_pts"]), f"{k} ev_cnt sum mismatch"
        off = np.zeros(len(cnt) + 1, dtype=np.int64)
        off[1:] = np.cumsum(cnt)
        i3_last_idx[k] = (off[i3_tr_idx + 1] - 1, off[i3_fl_idx + 1] - 1)

    report = {
        "n_pairs_ok": int(ok_all.sum()), "n_house": int(m_house.sum()),
        "n_self2": int(m_self2.sum()), "n_perfection": int(m_perf.sum()),
        "n_items3": len(i3_pairs), "modes": {},
    }
    saved_vectors = {}

    for mode_name, pool, neutral_only in [
        ("pts_neutral_only", "pts", True),
        ("story_neutral_only", "story", True),
        ("pts_full_C5", "pts", False),
    ]:
        print(f"\n================ {mode_name} (pool={pool}, neutral_only={neutral_only}) ================", flush=True)
        rb = fit_neutral_or_full_rotations(pool=pool, neutral_only=neutral_only, d_plato=256)
        print(f"  Rotations fit on {rb['n_passages']} train passages ({rb['n_vecs']} vecs), domains={rb['used_domains']}", flush=True)

        diffs_c5, diffs_c5_lenres, diffs_c4, i3_diffs_c5, i3_diffs_c4 = {}, {}, {}, {}, {}
        delta_lens, pair_lens = {}, {}
        for k in PD.ALL_KEYS:
            d_arr = arch[k]["done_last" if pool == "pts" else "done_mean"][ok_all]
            n_arr = arch[k]["neg_last" if pool == "pts" else "neg_mean"][ok_all]
            dl_k = arch[k]["done_len"][ok_all]
            nl_k = arch[k]["neg_len"][ok_all]
            delta_lens[k] = (dl_k - nl_k).astype(np.float64)
            pair_lens[k] = (dl_k, nl_k)
            _, _, dc5 = map_diffs(d_arr, n_arr, rb, k, frame="to_c5")
            _, _, dc4 = map_diffs(d_arr, n_arr, rb, k, frame="to_c4")
            diffs_c5[k] = dc5
            # Fit length OLS strictly on HOUSE rows for cross-dataset HOUSE->SELF2 & Stage-2 export!
            beta_h = fit_length_ols(dc5[m_house], dl_k[m_house], nl_k[m_house])
            diffs_c5_lenres[k] = apply_length_ols(dc5, dl_k, nl_k, beta_h)
            diffs_c4[k] = dc4

            if pool == "pts":
                t_i, f_i = i3_last_idx[k]
                i3_d, i3_n = raw_ev[k]["ev_pts"][t_i], raw_ev[k]["ev_pts"][f_i]
            else:
                i3_d, i3_n = raw_ev[k]["ev_story"][i3_tr_idx], raw_ev[k]["ev_story"][i3_fl_idx]
            _, _, i3_diffs_c5[k] = map_diffs(i3_d, i3_n, rb, k, frame="to_c5")
            _, _, i3_diffs_c4[k] = map_diffs(i3_d, i3_n, rb, k, frame="to_c4")

        # Full-sample directions (used ONLY on strictly unseen cross-dataset sets: HOUSE->SELF2, HOUSE->items3, and Stage-2 export!)
        h_dirs_c5 = {k: house_dir(diffs_c5[k][m_house], cats[m_house]) for k in PD.ALL_KEYS}
        s2_dirs_c5 = {k: unit(diffs_c5[k][m_self2].mean(axis=0)) for k in PD.ALL_KEYS}
        h_dirs_lr = {k: house_dir(diffs_c5_lenres[k][m_house], cats[m_house]) for k in PD.ALL_KEYS}
        h_dirs_c4 = {k: house_dir(diffs_c4[k][m_house], cats[m_house]) for k in PD.ALL_KEYS}
        u_house_c5 = unit(np.mean([h_dirs_c5[k] for k in PD.CONSENSUS_KEYS], axis=0))
        u_self2_c5 = unit(np.mean([s2_dirs_c5[k] for k in PD.CONSENSUS_KEYS], axis=0))
        u_house_lr = unit(np.mean([h_dirs_lr[k] for k in PD.CONSENSUS_KEYS], axis=0))
        u_house_c4 = unit(np.mean([h_dirs_c4[k] for k in PD.MLX_CONSENSUS_KEYS], axis=0))

        # 1. STRICTLY DISJOINT Zero-Shot: Gemma 4 -> 6 other companies
        gemma_to_others = {}
        print(f"  [1] DISJOINT Zero-Shot Gemma 4 -> 6 other companies (rot_domains={rb['used_domains']}):", flush=True)
        for tgt in PD.ALL_KEYS:
            if tgt == "gemma4_e4b":
                continue
            # Disjoint 2-fold on HOUSE (raw + strictly fold-local length-residualized) and SELF2
            rh = eval_disjoint_2fold([diffs_c5["gemma4_e4b"][m_house]], diffs_c5[tgt][m_house], cats[m_house], folds_house, is_house=True)
            rh_lr = eval_disjoint_2fold_lenres(
                [diffs_c5["gemma4_e4b"][m_house]],
                [(pair_lens["gemma4_e4b"][0][m_house], pair_lens["gemma4_e4b"][1][m_house])],
                diffs_c5[tgt][m_house], pair_lens[tgt][0][m_house], pair_lens[tgt][1][m_house],
                cats[m_house], folds_house, is_house=True,
            )
            rs2 = eval_disjoint_2fold([diffs_c5["gemma4_e4b"][m_self2]], diffs_c5[tgt][m_self2], cats[m_self2], folds_self2, is_house=False)
            rs2_lr = eval_disjoint_2fold_lenres(
                [diffs_c5["gemma4_e4b"][m_self2]],
                [(pair_lens["gemma4_e4b"][0][m_self2], pair_lens["gemma4_e4b"][1][m_self2])],
                diffs_c5[tgt][m_self2], pair_lens[tgt][0][m_self2], pair_lens[tgt][1][m_self2],
                cats[m_self2], folds_self2, is_house=False,
            )
            # Length-reversed perfection slice (subset of HOUSE disjoint scores)
            perf_acc = float((rh["scores"][cats[m_house] == "perfection"] > 0.0).mean())
            len_corr = float(np.corrcoef(rh["scores"], delta_lens[tgt][m_house])[0, 1])
            # Cross-dataset strictly unseen: Gemma HOUSE -> target's UNSEEN SELF2 & UNSEEN items3 (length-reversed!)
            ch_s2, _, ph_s2, _ = centered_signflip_p(h_dirs_c5["gemma4_e4b"], diffs_c5[tgt][m_self2], is_house=False)
            ah_s2, cih_s2, _ = pair_acc_boot(h_dirs_c5["gemma4_e4b"], diffs_c5[tgt][m_self2])
            ah_s2_lr, cih_s2_lr, _ = pair_acc_boot(h_dirs_lr["gemma4_e4b"], diffs_c5_lenres[tgt][m_self2])
            ch_i3, _, ph_i3, _ = centered_signflip_p(h_dirs_c5["gemma4_e4b"], i3_diffs_c5[tgt], is_house=False)
            ah_i3, cih_i3, _ = pair_acc_boot(h_dirs_c5["gemma4_e4b"], i3_diffs_c5[tgt])
            gemma_to_others[tgt] = {
                "house_disjoint": {k: v for k, v in rh.items() if k != "scores"},
                "house_lenres_disjoint": {k: v for k, v in rh_lr.items() if k != "scores"},
                "self2_disjoint": {k: v for k, v in rs2.items() if k != "scores"},
                "self2_lenres_disjoint": {k: v for k, v in rs2_lr.items() if k != "scores"},
                "perfection_acc_len_reversed": round(perf_acc, 4),
                "score_vs_delta_len_r": round(len_corr, 4),
                "house_to_unseen_self2": {"cos": round(ch_s2, 4), "p_pos": round(ph_s2, 4), "acc": round(ah_s2, 4), "ci": cih_s2, "acc_lenres": round(ah_s2_lr, 4), "ci_lenres": cih_s2_lr},
                "house_to_unseen_items3": {"cos": round(ch_i3, 4), "p_pos": round(ph_i3, 4), "acc": round(ah_i3, 4), "ci": cih_i3},
            }
            print(f"    Gemma4 -> {tgt:13s}: H_2f cos={rh['cos']:+.3f}(p={rh['p_pos_max']:.3f}) acc={rh['acc']*100:5.1f}% (lenres {rh_lr['acc']*100:5.1f}%, perf {perf_acc*100:4.1f}%, r_len={len_corr:+.2f}) | S2_2f={rs2['acc']*100:5.1f}%(lr {rs2_lr['acc']*100:5.1f}%) | H->S2={ah_s2*100:5.1f}% | H->i3={ah_i3*100:5.1f}%(cos={ch_i3:+.3f},p={ph_i3:.3f})", flush=True)

        # 2. Single Teachers vs 4-Fam (in C4!) & 5-Fam (in C5!) Denoised Consensus -> BLIND Phi-4 & OLMo 3!
        blind_res = {}
        print("  [2] DISJOINT Zero-Shot Single Teachers vs C4 / C5 Consensus -> BLIND Phi-4 & OLMo 3:", flush=True)
        for bk in PD.BLIND_KEYS:
            sub = {}
            for cname in list(PD.CONSENSUS_KEYS) + ["CONSENSUS_4FAM", "CONSENSUS_5FAM"]:
                if cname == "CONSENSUS_4FAM":
                    src_h = [diffs_c4[k][m_house] for k in PD.MLX_CONSENSUS_KEYS]
                    src_s2 = [diffs_c4[k][m_self2] for k in PD.MLX_CONSENSUS_KEYS]
                    tgt_h, tgt_s2, tgt_i3 = diffs_c4[bk][m_house], diffs_c4[bk][m_self2], i3_diffs_c4[bk]
                    v_h_full = u_house_c4
                    src_h_lens = None
                elif cname == "CONSENSUS_5FAM":
                    src_h = [diffs_c5[k][m_house] for k in PD.CONSENSUS_KEYS]
                    src_h_lens = [(pair_lens[k][0][m_house], pair_lens[k][1][m_house]) for k in PD.CONSENSUS_KEYS]
                    src_s2 = [diffs_c5[k][m_self2] for k in PD.CONSENSUS_KEYS]
                    tgt_h, tgt_s2, tgt_i3 = diffs_c5[bk][m_house], diffs_c5[bk][m_self2], i3_diffs_c5[bk]
                    v_h_full = u_house_c5
                else:
                    src_h = [diffs_c5[cname][m_house]]
                    src_h_lens = [(pair_lens[cname][0][m_house], pair_lens[cname][1][m_house])]
                    src_s2 = [diffs_c5[cname][m_self2]]
                    tgt_h, tgt_s2, tgt_i3 = diffs_c5[bk][m_house], diffs_c5[bk][m_self2], i3_diffs_c5[bk]
                    v_h_full = h_dirs_c5[cname]

                rh = eval_disjoint_2fold(src_h, tgt_h, cats[m_house], folds_house, is_house=True)
                rs2 = eval_disjoint_2fold(src_s2, tgt_s2, cats[m_self2], folds_self2, is_house=False)
                rh_lr = eval_disjoint_2fold_lenres(
                    src_h, src_h_lens, tgt_h, pair_lens[bk][0][m_house], pair_lens[bk][1][m_house],
                    cats[m_house], folds_house, is_house=True,
                ) if src_h_lens is not None else None
                perf_acc = float((rh["scores"][cats[m_house] == "perfection"] > 0.0).mean())
                ax_s2, cix_s2, _ = pair_acc_boot(v_h_full, tgt_s2)
                ch_i3, _, ph_i3, _ = centered_signflip_p(v_h_full, tgt_i3, is_house=False)
                ax_i3, cix_i3, _ = pair_acc_boot(v_h_full, tgt_i3)
                ax_i3m, _, _ = pair_acc_boot(v_h_full, tgt_i3[i3_is_myth])
                ax_i3b, _, _ = pair_acc_boot(v_h_full, tgt_i3[~i3_is_myth])
                sub[cname] = {
                    "house_2f": {k: v for k, v in rh.items() if k != "scores"},
                    "house_2f_lenres": None if rh_lr is None else {k: v for k, v in rh_lr.items() if k != "scores"},
                    "self2_2f": {k: v for k, v in rs2.items() if k != "scores"},
                    "perfection_acc": round(perf_acc, 4),
                    "house_to_self2_acc": round(ax_s2, 4), "house_to_self2_ci": cix_s2,
                    "house_to_items3": {"cos": round(ch_i3, 4), "p_pos": round(ph_i3, 4), "acc": round(ax_i3, 4), "ci": cix_i3, "myth_acc": round(ax_i3m, 4), "mech_acc": round(ax_i3b, 4)},
                }
                lr_s = f" (lr {rh_lr['acc']*100:4.1f}%)" if rh_lr is not None else ""
                print(f"    {cname:14s} -> {bk:9s}: H_2f cos={rh['cos']:+.3f} acc={rh['acc']*100:5.1f}%{lr_s} perf={perf_acc*100:4.1f}% | S2_2f={rs2['acc']*100:5.1f}% | H->S2={ax_s2*100:5.1f}% | H->i3={ax_i3*100:5.1f}%(cos={ch_i3:+.3f},p={ph_i3:.3f}, m={ax_i3m*100:.0f}%/b={ax_i3b*100:.0f}%)", flush=True)
            blind_res[bk] = sub

        report["modes"][mode_name] = {
            "used_domains": rb["used_domains"],
            "n_train_passages": rb["n_passages"],
            "n_train_vecs": rb["n_vecs"],
            "gemma_to_others": gemma_to_others,
            "blind_transfer": blind_res,
            "house_self2_cos_in_c5": round(float(u_house_c5 @ u_self2_c5), 4),
            "house_raw_vs_lenres_cos": round(float(u_house_c5 @ u_house_lr), 4),
        }

        if mode_name == "pts_full_C5":
            q0 = unit(u_house_c5)
            q1 = unit(u_self2_c5 - float(u_self2_c5 @ q0) * q0)
            qlr0 = unit(u_house_lr)
            saved_vectors["u_house_c5"] = u_house_c5.astype(np.float32)
            saved_vectors["u_self2_c5"] = u_self2_c5.astype(np.float32)
            saved_vectors["u_house_c5_lenres"] = qlr0.astype(np.float32)
            saved_vectors["U_jung_c5"] = np.stack([q0, q1], axis=0).astype(np.float32)  # (2, 256) orthonormal
        elif mode_name == "pts_neutral_only":
            saved_vectors["u_house_neutral"] = u_house_c5.astype(np.float32)
            saved_vectors["u_self2_neutral"] = u_self2_c5.astype(np.float32)

    report["elapsed_s"] = round(time.time() - t0, 1)
    (PLATO / "arch229_report.json").write_text(json.dumps(report, indent=2) + "\n")
    PD._atomic_savez(PLATO / "arch229_consensus.npz", **saved_vectors)
    print(f"\nARCH229_ANALYZE_DONE ({report['elapsed_s']:.1f} s) -> {PLATO / 'arch229_report.json'}", flush=True)


def self_test():
    """Unit checks: pair loader, domain filter on passages.jsonl, centered null calibration (high-SNR & pure-noise
    DISJOINT 2-fold!), length residualization, and Procrustes exact rotation recovery."""
    pairs = load_arch_pairs()
    assert len(pairs) == 312, f"expected 312 pairs, got {len(pairs)}"
    assert set(p["cat"] for p in pairs) == set(ALL_CATS)

    # 1. Verify passages.jsonl domain strings match NEUTRAL_DOMAINS!
    if (PLATO / "passages.jsonl").exists():
        doms = {json.loads(l)["domain"] for l in open(PLATO / "passages.jsonl")}
        assert set(NEUTRAL_DOMAINS).issubset(doms), f"NEUTRAL_DOMAINS {NEUTRAL_DOMAINS} not in {doms}"
        n_neu = sum(1 for l in open(PLATO / "passages.jsonl") if json.loads(l)["domain"] in NEUTRAL_DOMAINS)
        assert n_neu >= 2000, f"expected >=2000 neutral passages, got {n_neu}"

    # 2. Verify centered sign-flip null on high-SNR signal AND disjoint pure-noise Type-I calibration!
    rng = np.random.default_rng(7)
    D, N = 256, 80
    true_u = unit(rng.standard_normal(D))
    X_sig = 1.5 * true_u[None, :] + 0.5 * rng.standard_normal((N, D)) / np.sqrt(D)
    obs, p_two, p_pos, med_null = centered_signflip_p(true_u, X_sig, is_house=False, n_draws=500, seed=1)
    assert obs > 0.95 and p_pos < 0.01 and med_null < 0.15, f"high-SNR failed: {obs}, {p_pos}, {med_null}"

    # Pure-noise shared-stimulus simulation (Amendment 229a check: disjoint 2-fold MUST stay ~50% acc & ~0 cos!)
    z_stim = rng.standard_normal((N, D))
    src_noise = z_stim + 0.5 * rng.standard_normal((N, D))
    tgt_noise = z_stim + 0.5 * rng.standard_normal((N, D))
    folds = np.array([i % 2 for i in range(N)], dtype=np.int32)
    cats_dummy = np.array(["self2"] * N)
    res_null = eval_disjoint_2fold([src_noise], tgt_noise, cats_dummy, folds, is_house=False, seed=3)
    assert abs(res_null["cos"]) < 0.20 and 0.35 < res_null["acc"] < 0.65, f"disjoint 2-fold inflated on pure noise: {res_null}"

    # 3. Verify length residualization preserves zero-length semantic mean while removing pure length confound
    dl = rng.integers(40, 80, size=200)
    nl = dl - rng.integers(1, 10, size=200)
    u_sem = unit(rng.standard_normal(32))
    u_len = unit(rng.standard_normal(32))
    Y = 1.0 * u_sem[None, :] + 0.3 * (dl - nl)[:, None] * u_len[None, :] + 0.1 * rng.standard_normal((200, 32))
    Y_res = residualize_length(Y, dl, nl)
    c_sem = float(unit(Y_res.mean(0)) @ u_sem)
    assert c_sem > 0.95, f"length residualization lost semantic intercept: {c_sem}"
    # Also verify strictly fold-local length residualization does not inflate pure length-confounded noise!
    Y_pure_len = 0.3 * (dl - nl)[:, None] * u_len[None, :] + 0.5 * rng.standard_normal((200, 32))
    Y_pure_len2 = 0.3 * (dl - nl)[:, None] * u_len[None, :] + 0.5 * rng.standard_normal((200, 32))
    f200 = np.array([i % 2 for i in range(200)], dtype=np.int32)
    lr_null = eval_disjoint_2fold_lenres([Y_pure_len], [(dl, nl)], Y_pure_len2, dl, nl, np.array(["self2"] * 200), f200, is_house=False, seed=9)
    assert 0.35 < lr_null["acc"] < 0.65, f"fold-local lenres failed to neutralize pure length confound: {lr_null}"

    # 4. Verify Procrustes rotation recovers an exact rotated direction zero-shot
    A = rng.standard_normal((400, 64))
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    B = 2.5 * (A @ Q.T) + rng.standard_normal((1, 64))
    params = PD.fit_procrustes_map(A, B)
    d_a = rng.standard_normal((50, 64))
    d_b = 2.5 * (d_a @ Q.T)
    pred = PD.apply_procrustes_map(d_a, params) - PD.apply_procrustes_map(np.zeros_like(d_a), params)
    err = float(np.max(np.abs(pred - d_b)))
    assert err < 1e-10, f"Procrustes diff rotation error: {err}"
    print(f"ARCH229_TEST_OK pairs={len(pairs)} highSNR(obs={obs:.3f},p={p_pos:.4f}) disjoint_null(cos={res_null['cos']:+.3f},acc={res_null['acc']:.3f}) lenres_sem={c_sem:.3f} rot_err={err:.1e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        self_test()
    elif cmd == "extract":
        keys = sys.argv[2:] or list(PD.ALL_KEYS)
        for k in keys:
            extract_model(k)
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown cmd: {cmd}")
