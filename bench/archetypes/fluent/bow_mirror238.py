"""Experiment 238: Mirrored 120-Pair Order-Balanced Identical-BoW Swaps + Per-Model Log-Probability (Plausibility) Control.

Directly resolves the final two referee controls on `bow_swap_60`:
  1. **Mirrored 120-pair strictly order-balanced swap benchmark (`bow_mirror238_data.json`):**
     Takes all 60 settings from `bow_swap_60` and pairs each original pair (`orig`) with an exact
     clause-order-mirrored counterpart (`mirror`), yielding 120 pairs (`60` favorable-first + `60`
     unfavorable-first) with `Counter(tokens(pos)) == Counter(tokens(neg))` on all `120/120` pairs.
     Setting-disjoint 2-fold CV (both mirrors of setting `i` always stay in the same fold) cancels
     BOTH mean-pooling primacy bias AND last-token recency bias to exact mathematical zero, while
     providing full 60-setting power and a strict "both-mirrors-right" metric (where any pure
     order-only classifier scores 0/60 = 0.0%!).
  2. **Per-model token log-probability / plausibility baseline (`ΔNLL = NLL(neg) - NLL(pos)`):**
     Computes exact per-sentence mean and total NLL directly from each of the 6 MLX models' logits
     during the same forward pass that extracts hidden states, measuring standalone plausibility
     accuracy, correlation with cross-family neural transfer (`Gemma 4 -> target`, `⊥ VAL2`),
     accuracy on NLL-wrong / low-ΔNLL pairs, and the setting-clustered bootstrap 95% CI on the
     OLS intercept at `ΔNLL = 0`.

Usage:
  python fluent/bow_mirror238.py test
  python fluent/bow_mirror238.py extract [KEY...]
  python fluent/bow_mirror238.py analyze
  python fluent/bow_mirror238.py all
"""
import gc
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO_LOCAL = HERE / "plato"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(PLATO_LOCAL)))
DATA_JSON = ARCH / "selfshape" / "bow_mirror238_data.json"
OUT_JSON = PLATO_LOCAL / "bow_mirror238.json"

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import bow_swap236 as BS236  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402


def _word_bag(text):
    return Counter(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def load_mirror_data():
    obj = json.loads(DATA_JSON.read_text())
    rows = obj["bow_mirror_120"]
    assert len(rows) == 120, f"Expected 120 rows, got {len(rows)}"
    return rows


def verify_dataset_integrity():
    rows = load_mirror_data()
    by_sid = {}
    for r in rows:
        sid = r["setting_id"]
        assert r["order"] in ("orig", "mirror")
        assert r["fav_pos"] in ("first", "second")
        p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
        assert p_txt != n_txt, f"Identical pos/neg in {r['id']}"
        assert _word_bag(p_txt) == _word_bag(n_txt), f"Word bag mismatch in {r['id']}"
        by_sid.setdefault(sid, []).append(r)
    assert len(by_sid) == 60, f"Expected 60 settings, got {len(by_sid)}"
    n_first, n_second = 0, 0
    for sid, pair_rows in by_sid.items():
        assert len(pair_rows) == 2, f"Setting {sid} does not have 2 orders"
        orders = {r["order"] for r in pair_rows}
        fpos = {r["fav_pos"] for r in pair_rows}
        assert orders == {"orig", "mirror"}, f"Setting {sid} missing orig/mirror"
        assert fpos == {"first", "second"}, f"Setting {sid} not order-mirrored: {fpos}"
        n_first += sum(1 for r in pair_rows if r["fav_pos"] == "first")
        n_second += sum(1 for r in pair_rows if r["fav_pos"] == "second")
    assert n_first == 60 and n_second == 60
    return {
        "n_pairs": len(rows),
        "n_settings": len(by_sid),
        "n_fav_first": n_first,
        "n_fav_second": n_second,
        "exact_word_bag_matches": len(rows),
    }


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _save_bm_npz(out_npz, rows, pl, pm, nl, nm, p_len, n_len, p_nll_mean, n_nll_mean, p_nll_sum, n_nll_sum, layer_idx, num_layers):
    pl_a, pm_a = np.stack(pl, 0), np.stack(pm, 0)
    nl_a, nm_a = np.stack(nl, 0), np.stack(nm, 0)
    assert np.isfinite(pl_a).all() and np.isfinite(pm_a).all()
    assert np.isfinite(nl_a).all() and np.isfinite(nm_a).all()
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        out_npz,
        pos_last=pack(pl_a), pos_mean=pack(pm_a),
        neg_last=pack(nl_a), neg_mean=pack(nm_a),
        pos_len=np.array(p_len, dtype=np.int32),
        neg_len=np.array(n_len, dtype=np.int32),
        pos_nll_mean=np.array(p_nll_mean, dtype=np.float32),
        neg_nll_mean=np.array(n_nll_mean, dtype=np.float32),
        pos_nll_sum=np.array(p_nll_sum, dtype=np.float32),
        neg_nll_sum=np.array(n_nll_sum, dtype=np.float32),
        ids=np.array([r["id"] for r in rows]),
        setting_ids=np.array([r["setting_id"] for r in rows]),
        orders=np.array([r["order"] for r in rows]),
        fav_pos=np.array([r["fav_pos"] for r in rows]),
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
    print(f"BM238 EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post(ep, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{ep}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    pl, pm, nl, nm, p_len, n_len = [], [], [], [], [], []
    nan_arr = [float("nan")] * len(rows)
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
        for r in rows:
            for side in ("pos", "neg"):
                txt = r[side].strip()
                toks = post("/tokenize", {"content": txt, "add_special": True})["tokens"]
                assert 2 <= len(toks) <= 512
                emb = post("/embedding", {"content": txt})
                arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
                assert arr.ndim == 2 and arr.shape[0] == len(toks)
                hl = arr[-1].copy()
                hm = (arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0]).copy()
                if side == "pos":
                    pl.append(hl); pm.append(hm); p_len.append(len(toks))
                else:
                    nl.append(hl); nm.append(hm); n_len.append(len(toks))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _save_bm_npz(out_npz, rows, pl, pm, nl, nm, p_len, n_len, nan_arr, nan_arr, nan_arr, nan_arr, -1, 24)
    print(f"BM238_EXTRACT_DONE {key}: ok={len(rows)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


def extract_model(key):
    assert key in PD.MODEL_SPECS
    out_npz = PLATO_CACHE / f"bm238_{key}.npz"
    if out_npz.exists():
        print(f"BM238 EXTRACT {key}: already exists, skipping", flush=True)
        return
    rows = load_mirror_data()
    spec = PD.MODEL_SPECS[key]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp(key, spec, rows, out_npz)
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"BM238 EXTRACT {key}: loading {spec['path']} ...", flush=True)
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

    pl, pm, nl, nm, p_len, n_len = [], [], [], [], [], []
    p_nll_m, n_nll_m, p_nll_s, n_nll_s = [], [], [], []
    try:
        for r in rows:
            for side in ("pos", "neg"):
                ids = PD._text_ids_mlx(tok, r[side].strip())
                assert 2 <= len(ids) <= 512
                rec_buf.clear()
                x = mx.array([ids])
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                assert len(rec_buf) == 1
                h = rec_buf[0].astype(mx.float32)
                hl = h[-1]
                hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
                # Exact next-token cross-entropy NLL over all predicted tokens ids[1:]
                ce = nn.losses.cross_entropy(logits[0, :-1, :].astype(mx.float32), x[0, 1:], reduction="none")
                nll_mean = mx.mean(ce)
                nll_sum = mx.sum(ce)
                mx.eval(hl, hm, nll_mean, nll_sum)
                rec_buf.clear()
                if side == "pos":
                    pl.append(np.array(hl, dtype=np.float32))
                    pm.append(np.array(hm, dtype=np.float32))
                    p_len.append(len(ids))
                    p_nll_m.append(float(nll_mean.item()))
                    p_nll_s.append(float(nll_sum.item()))
                else:
                    nl.append(np.array(hl, dtype=np.float32))
                    nm.append(np.array(hm, dtype=np.float32))
                    n_len.append(len(ids))
                    n_nll_m.append(float(nll_mean.item()))
                    n_nll_s.append(float(nll_sum.item()))
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()

    _save_bm_npz(out_npz, rows, pl, pm, nl, nm, p_len, n_len, p_nll_m, n_nll_m, p_nll_s, n_nll_s, tgt_idx, nl_cnt)
    print(f"BM238_EXTRACT_DONE {key}: ok={len(rows)}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)


def _setting_clustered_signflip_p(u, D_pairs, sid_idx, n_draws=1000, seed=2380):
    """Exact setting-clustered centered sign-flip test: flips BOTH mirrors of each setting by the SAME ±1 sign
    (60 independent setting units -> zero pseudoreplication!)."""
    u = AR.unit(u)
    uniq_s = sorted(set(int(s) for s in sid_idx))
    # Setting-averaged difference vector (averages the two mirrored clause orders per setting -> order bias = 0!)
    D_set = np.stack([D_pairs[sid_idx == s].mean(axis=0) for s in uniq_s], axis=0)
    obs_cos = float(u @ AR.unit(D_set.mean(axis=0)))
    mu = D_set.mean(axis=0, keepdims=True)
    Dc = D_set - mu
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_draws):
        sg = rng.choice([-1.0, 1.0], size=(len(uniq_s), 1))
        m_sh = (Dc * sg).mean(axis=0)
        if float(u @ AR.unit(m_sh)) >= obs_cos:
            ge += 1
    return obs_cos, float((ge + 1) / (n_draws + 1))


def _setting_clustered_ols_intercept_ci(scores, dnll, sid_idx, n_boot=2000, seed=2389):
    """Setting-clustered bootstrap CI on OLS intercept `a` in `score = a + b * ΔNLL` (value of neural margin at ΔNLL = 0)."""
    uniq_s = np.array(sorted(set(int(s) for s in sid_idx)))
    by_s = [np.where(sid_idx == s)[0] for s in uniq_s]

    def fit_ab(idx):
        x = dnll[idx]
        y = scores[idx]
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[0]), float(beta[1])

    all_idx = np.arange(len(scores))
    a_obs, b_obs = fit_ab(all_idx)
    r_obs = float(np.corrcoef(dnll, scores)[0, 1]) if np.std(dnll) > 1e-12 and np.std(scores) > 1e-12 else 0.0
    rng = np.random.default_rng(seed)
    a_draws = []
    for _ in range(n_boot):
        chosen = rng.choice(len(uniq_s), size=len(uniq_s), replace=True)
        idx = np.concatenate([by_s[c] for c in chosen])
        a_b, _ = fit_ab(idx)
        a_draws.append(a_b)
    return {
        "r_score_vs_dnll": round(r_obs, 4),
        "ols_intercept_at_dnll0": round(a_obs, 4),
        "ols_intercept_ci95": [round(float(np.percentile(a_draws, 2.5)), 4), round(float(np.percentile(a_draws, 97.5)), 4)],
        "ols_slope": round(b_obs, 4),
        "mean_score_overall": round(float(np.mean(scores)), 4),
        "intercept_share_of_mean": round(float(a_obs / max(1e-9, abs(np.mean(scores)))), 4),
    }


def _both_mirrors_right(correct_mask, sid_idx):
    uniq_s = sorted(set(int(s) for s in sid_idx))
    both = sum(1 for s in uniq_s if bool(np.all(correct_mask[sid_idx == s])))
    either = sum(1 for s in uniq_s if bool(np.any(correct_mask[sid_idx == s])))
    return both / len(uniq_s), both, either


def analyze_all():
    integrity = verify_dataset_integrity()
    rows = load_mirror_data()
    sids_str = [r["setting_id"] for r in rows]
    uniq_sids = sorted(set(sids_str))
    sid2i = {s: i for i, s in enumerate(uniq_sids)}
    sid_idx = np.array([sid2i[s] for s in sids_str], dtype=int)
    is_orig = np.array([r["order"] == "orig" for r in rows], dtype=bool)
    is_first = np.array([r["fav_pos"] == "first" for r in rows], dtype=bool)
    is_second = ~is_first

    # 20 deterministic setting-disjoint 2-fold splits (30 settings = 60 mirrored pairs per fold!)
    splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23800 + s_i)
        perm = rng.permutation(len(uniq_sids))
        f0 = set(int(x) for x in perm[:30])
        splits.append(np.array([0 if int(s) in f0 else 1 for s in sid_idx], dtype=int))

    # Lexical baselines on all 120 mirrored pairs
    bow_uni = BS236._bow_2fold_pairs(rows, "pos", "neg", BS236._words, splits)
    bow_bi = BS236._bow_2fold_pairs(rows, "pos", "neg", BS236._bigrams, splits)
    lex_res = {
        "bow_bm120_unigram_20split_mean": round(float(np.mean(bow_uni)), 4),
        "bow_bm120_bigram_20split_mean": round(float(np.mean(bow_bi)), 4),
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
    sg_rows, _, _ = V234.load_valence_data()
    sg_dom_arr = np.array([r["domain"] for r in sg_rows])
    sg_domains = sorted(set(sg_dom_arr.tolist()))

    Z_m, Z_l = {}, {}
    Q2_m, Q5_m, uh_m, us_m = {}, {}, {}, {}
    Q2_l, Q5_l, uh_l, us_l = {}, {}, {}, {}
    dnll_mean, dnll_sum = {}, {}
    plausibility = {}

    for k in PD.ALL_KEYS:
        d = AR._load_npz_dict(PLATO_CACHE / f"bm238_{k}.npz")
        a = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
        v = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")
        groups = v["groups"]

        # Mean pool frame
        _, _, Dh_m = AR.map_diffs(a["done_mean"][house_idx], a["neg_mean"][house_idx], rot_st, k, frame="to_c5")
        _, _, Dv_m = AR.map_diffs(v["pos_mean"], v["neg_mean"], rot_st, k, frame="to_c5")
        _, _, Z_m[k] = AR.map_diffs(d["pos_mean"], d["neg_mean"], rot_st, k, frame="to_c5")
        uh_m[k] = AR.house_dir(Dh_m, house_cats)
        sg_m = Dv_m[groups == "sentiment_gen"]
        us_m[k] = AR.unit(sg_m.mean(0))
        uo_m = AR.unit(Dv_m[groups == "outcome_valence"].mean(0))
        Q2_m[k] = V234._orthonormal_basis([us_m[k], uo_m])
        Q5_m[k] = V234._orthonormal_basis([AR.unit(sg_m[sg_dom_arr == dm].mean(0)) for dm in sg_domains] + [uo_m])

        # Last token frame
        _, _, Dh_l = AR.map_diffs(a["done_last"][house_idx], a["neg_last"][house_idx], rot_pts, k, frame="to_c5")
        _, _, Dv_l = AR.map_diffs(v["pos_last"], v["neg_last"], rot_pts, k, frame="to_c5")
        _, _, Z_l[k] = AR.map_diffs(d["pos_last"], d["neg_last"], rot_pts, k, frame="to_c5")
        uh_l[k] = AR.house_dir(Dh_l, house_cats)
        sg_l = Dv_l[groups == "sentiment_gen"]
        us_l[k] = AR.unit(sg_l.mean(0))
        uo_l = AR.unit(Dv_l[groups == "outcome_valence"].mean(0))
        Q2_l[k] = V234._orthonormal_basis([us_l[k], uo_l])
        Q5_l[k] = V234._orthonormal_basis([AR.unit(sg_l[sg_dom_arr == dm].mean(0)) for dm in sg_domains] + [uo_l])

        # Log-probability / plausibility baseline (MLX models)
        pm_n = d["pos_nll_mean"].astype(np.float64)
        nm_n = d["neg_nll_mean"].astype(np.float64)
        ps_n = d["pos_nll_sum"].astype(np.float64)
        ns_n = d["neg_nll_sum"].astype(np.float64)
        if np.isfinite(pm_n).all():
            dm = nm_n - pm_n  # > 0 means pos is MORE plausible (lower NLL) than neg!
            ds = ns_n - ps_n
            dnll_mean[k] = dm
            dnll_sum[k] = ds
            bm_rate, bm_n, _ = _both_mirrors_right(dm > 0, sid_idx)
            bs_rate, bs_n, _ = _both_mirrors_right(ds > 0, sid_idx)
            plausibility[k] = {
                "orig60_mean_nll_acc": round(float(np.mean(dm[is_orig] > 0)), 4),
                "orig60_sum_nll_acc": round(float(np.mean(ds[is_orig] > 0)), 4),
                "mirror120_mean_nll_acc": round(float(np.mean(dm > 0)), 4),
                "mirror120_sum_nll_acc": round(float(np.mean(ds > 0)), 4),
                "fav_first_mean_nll_acc": round(float(np.mean(dm[is_first] > 0)), 4),
                "fav_second_mean_nll_acc": round(float(np.mean(dm[is_second] > 0)), 4),
                "both_mirrors_right_mean_nll": round(bm_rate, 4),
                "both_mirrors_right_mean_nll_n": int(bm_n),
                "both_mirrors_right_sum_nll": round(bs_rate, 4),
                "both_mirrors_right_sum_nll_n": int(bs_n),
                "mean_dnll_nats": round(float(np.mean(dm)), 4),
            }

    # C5 geometry on mirrored 120 pairs
    u_h_c5 = AR.unit(np.mean([uh_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_s_c5 = AR.unit(np.mean([us_m[k] for k in PD.CONSENSUS_KEYS], axis=0))
    u_bm_c5 = AR.unit(np.mean([AR.unit(Z_m[k].mean(0)) for k in PD.CONSENSUS_KEYS], axis=0))
    Q2_c5 = V234._orthonormal_basis([u_s_c5, AR.unit(np.mean([Q2_m[k][:, 1] for k in PD.CONSENSUS_KEYS], axis=0))])
    c5_geom = {
        "cos_bm120_house_mean": round(float(u_bm_c5 @ u_h_c5), 4),
        "cos_bm120_sent_mean": round(float(u_bm_c5 @ u_s_c5), 4),
        "bm120_sqnorm_in_val2": round(float(np.sum((Q2_c5.T @ u_bm_c5) ** 2)), 4),
    }

    src_k = "gemma4_e4b"
    targets = [k for k in PD.ALL_KEYS if k != src_k]
    modes_out = {}

    for mode_key, Z_d, Q2_d, Q5_d, uh_d, us_d in [
        ("mean_pool", Z_m, Q2_m, Q5_m, uh_m, us_m),
        ("last_token", Z_l, Q2_l, Q5_l, uh_l, us_l),
    ]:
        uh_p2_s = V234._project_out(uh_d[src_k], Q2_d[src_k])
        mode_res = {}
        for tk in targets:
            Ds, Dt = Z_d[src_k], Z_d[tk]
            Dt2 = V234._project_out_rows(Dt, Q2_d[tk])
            Dt5 = V234._project_out_rows(Dt, Q5_d[tk])
            u_hp2 = V234._project_out(uh_p2_s, Q2_d[tk])

            # Zero-shot SENT, HOUSE, HOUSE ⊥ VAL2 on all 120 mirrored pairs (with setting-clustered p & both-mirrors-right!)
            sc_s = Dt @ us_d[src_k]
            sc_h = Dt @ uh_d[src_k]
            sc_hp2 = Dt2 @ u_hp2
            c_s, p_s = _setting_clustered_signflip_p(us_d[src_k], Dt, sid_idx, seed=2381)
            c_h, p_h = _setting_clustered_signflip_p(uh_d[src_k], Dt, sid_idx, seed=2382)
            c_hp2, p_hp2 = _setting_clustered_signflip_p(u_hp2, Dt2, sid_idx, seed=2383)
            bm_s, bm_sn, _ = _both_mirrors_right(sc_s > 0, sid_idx)
            bm_h, bm_hn, _ = _both_mirrors_right(sc_h > 0, sid_idx)
            bm_hp2, bm_hp2n, _ = _both_mirrors_right(sc_hp2 > 0, sid_idx)

            # 20 setting-disjoint 2-fold splits (exact 30 settings = 60 mirrored pairs train -> 60 mirrored pairs test!)
            r_accs, p2_accs, p5_accs = [], [], []
            p2_first_accs, p2_second_accs, p2_both_rates = [], [], []
            xord_f2s_accs, xord_s2f_accs = [], []
            oof_p2_split0 = np.zeros(len(rows), dtype=np.float64)

            for s_i, sp in enumerate(splits):
                ok_r, ok_2, ok_5 = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
                ok_f2s, ok_s2f = [], []
                for f in (0, 1):
                    tr, te = (sp != f), (sp == f)
                    ut = AR.unit(Ds[tr].mean(0))
                    ut2 = V234._project_out(V234._project_out(ut, Q2_d[src_k]), Q2_d[tk])
                    ut5 = V234._project_out(V234._project_out(ut, Q5_d[src_k]), Q5_d[tk])
                    ok_r[te] = (Dt[te] @ ut) > 0
                    s2 = Dt2[te] @ ut2
                    ok_2[te] = s2 > 0
                    ok_5[te] = (Dt5[te] @ ut5) > 0
                    if s_i == 0:
                        oof_p2_split0[te] = s2
                    # Setting-disjoint strict cross-order transfer (train fav-first in tr -> test fav-second in te, & vice versa!)
                    uf = V234._project_out(V234._project_out(AR.unit(Ds[tr & is_first].mean(0)), Q2_d[src_k]), Q2_d[tk])
                    us2 = V234._project_out(V234._project_out(AR.unit(Ds[tr & is_second].mean(0)), Q2_d[src_k]), Q2_d[tk])
                    ok_f2s.extend(((Dt2[te & is_second] @ uf) > 0).tolist())
                    ok_s2f.extend(((Dt2[te & is_first] @ us2) > 0).tolist())
                r_accs.append(float(np.mean(ok_r)))
                p2_accs.append(float(np.mean(ok_2)))
                p5_accs.append(float(np.mean(ok_5)))
                p2_first_accs.append(float(np.mean(ok_2[is_first])))
                p2_second_accs.append(float(np.mean(ok_2[is_second])))
                b2r, _, _ = _both_mirrors_right(ok_2, sid_idx)
                p2_both_rates.append(b2r)
                xord_f2s_accs.append(float(np.mean(ok_f2s)))
                xord_s2f_accs.append(float(np.mean(ok_s2f)))

            # Setting-clustered sign-flip p on Split 0 folds (30 independent settings per fold -> conservative p_max!)
            sp0 = splits[0]
            f_c_r, f_p_r, f_c_2, f_p_2 = [], [], [], []
            for f in (0, 1):
                tr, te = (sp0 != f), (sp0 == f)
                ut = AR.unit(Ds[tr].mean(0))
                ut2 = V234._project_out(V234._project_out(ut, Q2_d[src_k]), Q2_d[tk])
                cr, pr = _setting_clustered_signflip_p(ut, Dt[te], sid_idx[te], seed=2384 + f)
                c2, p2 = _setting_clustered_signflip_p(ut2, Dt2[te], sid_idx[te], seed=2386 + f)
                f_c_r.append(cr); f_p_r.append(pr); f_c_2.append(c2); f_p_2.append(p2)

            # Plausibility (ΔNLL) interaction diagnostics on split-0 OOF `⊥ VAL2` projections
            nll_ctrl = None
            if tk in dnll_mean:
                dm = dnll_mean[tk]
                nll_wrong = dm <= 0
                nll_low = np.abs(dm) <= np.median(np.abs(dm))
                nll_ctrl = {
                    "target_plausibility_acc_120": round(float(np.mean(dm > 0)), 4),
                    "n_nll_wrong": int(np.sum(nll_wrong)),
                    "neural_p2_acc_on_nll_wrong": round(float(np.mean(oof_p2_split0[nll_wrong] > 0)), 4) if np.sum(nll_wrong) > 0 else None,
                    "neural_p2_acc_on_low_half_abs_dnll": round(float(np.mean(oof_p2_split0[nll_low] > 0)), 4),
                    "ols_p2_vs_target_dnll": _setting_clustered_ols_intercept_ci(oof_p2_split0, dm, sid_idx, seed=2389),
                    "ols_hp2_vs_target_dnll": _setting_clustered_ols_intercept_ci(sc_hp2, dm, sid_idx, seed=2390),
                }

            mode_res[tk] = {
                "zs_sent_120": round(float(np.mean(sc_s > 0)), 4),
                "zs_sent_both_mirrors": round(bm_s, 4),
                "zs_sent_p": round(p_s, 4),
                "zs_house_120": round(float(np.mean(sc_h > 0)), 4),
                "zs_house_both_mirrors": round(bm_h, 4),
                "zs_house_p": round(p_h, 4),
                "zs_house_perp_v2_120": round(float(np.mean(sc_hp2 > 0)), 4),
                "zs_house_perp_v2_first": round(float(np.mean(sc_hp2[is_first] > 0)), 4),
                "zs_house_perp_v2_second": round(float(np.mean(sc_hp2[is_second] > 0)), 4),
                "zs_house_perp_v2_both_mirrors": round(bm_hp2, 4),
                "zs_house_perp_v2_both_n": int(bm_hp2n),
                "zs_house_perp_v2_cos": round(c_hp2, 4),
                "zs_house_perp_v2_p": round(p_hp2, 4),
                "cv2_raw_120": round(float(np.mean(r_accs)), 4),
                "cv2_raw_cos": round(float(np.mean(f_c_r)), 4),
                "cv2_raw_p_max": round(float(max(f_p_r)), 4),
                "cv2_perp_v2_120": round(float(np.mean(p2_accs)), 4),
                "cv2_perp_v2_first": round(float(np.mean(p2_first_accs)), 4),
                "cv2_perp_v2_second": round(float(np.mean(p2_second_accs)), 4),
                "cv2_perp_v2_both_mirrors": round(float(np.mean(p2_both_rates)), 4),
                "cv2_perp_v2_cos": round(float(np.mean(f_c_2)), 4),
                "cv2_perp_v2_p_max": round(float(max(f_p_2)), 4),
                "cv2_perp_v5_120": round(float(np.mean(p5_accs)), 4),
                "xorder_perp_v2_first_to_second": round(float(np.mean(xord_f2s_accs)), 4),
                "xorder_perp_v2_second_to_first": round(float(np.mean(xord_s2f_accs)), 4),
                "plausibility_control": nll_ctrl,
            }
        modes_out[mode_key] = mode_res

    # Holm-Bonferroni adjusted p-values across the 6 rival target families
    def _holm_adjust(p_map):
        items = sorted(p_map.items(), key=lambda kv: kv[1])
        m = len(items)
        adj = {}
        running = 0.0
        for rank, (tk, pv) in enumerate(items):
            val = min(1.0, (m - rank) * pv)
            running = max(running, val)
            adj[tk] = round(running, 4)
        return adj

    holm_summary = {}
    for mode_key in ("mean_pool", "last_token"):
        mr = modes_out[mode_key]
        holm_summary[mode_key] = {
            "zs_house_perp_v2_p_holm": _holm_adjust({tk: mr[tk]["zs_house_perp_v2_p"] for tk in targets}),
            "cv2_perp_v2_p_holm": _holm_adjust({tk: mr[tk]["cv2_perp_v2_p_max"] for tk in targets}),
        }

    # Shuffled-label effective-rank (PR) null in C5 for both 8-role benchmarks (500 within-setting permutations)
    rng_pr = np.random.default_rng(238)
    j_rows = NA235.load_named_arch_data()
    j_y = np.array([NA235.ARCHETYPES.index(r["archetype"]) for r in j_rows])
    n_rows = NJ237.load_nonjung_data()
    n_y = np.array([NJ237.ROLES.index(r["role"]) for r in n_rows])
    Zj_c5 = np.mean([NA235._map_stories(AR._load_npz_dict(PLATO_CACHE / f"arch235_{k}.npz")["mean"], rot_st, k, frame="to_c5") for k in PD.CONSENSUS_KEYS], axis=0)
    Zn_c5 = np.mean([NA235._map_stories(AR._load_npz_dict(PLATO_CACHE / f"nj237_{k}.npz")["mean"], rot_st, k, frame="to_c5") for k in PD.CONSENSUS_KEYS], axis=0)
    pr_shuffled_null = {}
    for tag, Z_mat, y_arr in [("jung_8_c5", Zj_c5, j_y), ("nonjung_8_c5", Zn_c5, n_y)]:
        Zc = Z_mat - Z_mat.mean(0, keepdims=True)
        obs_pr, _ = NA235._effective_rank_pr(NA235._ovr_dirs(Zc, y_arr, n_cls=8))
        null_prs = []
        for _ in range(500):
            ys = y_arr.copy()
            for s in range(40):
                ys[s * 8:(s + 1) * 8] = rng_pr.permutation(ys[s * 8:(s + 1) * 8])
            pr_s, _ = NA235._effective_rank_pr(NA235._ovr_dirs(Zc, ys, n_cls=8))
            null_prs.append(pr_s)
        pr_shuffled_null[tag] = {
            "obs_pr": round(float(obs_pr), 3),
            "null_mean_pr": round(float(np.mean(null_prs)), 3),
            "null_p05_p95": [round(float(np.percentile(null_prs, 5)), 3), round(float(np.percentile(null_prs, 95)), 3)],
            "p_null_le_obs": round(float((sum(x <= obs_pr for x in null_prs) + 1) / (len(null_prs) + 1)), 4),
        }

    report = {
        "integrity": integrity,
        "lexical_baselines": lex_res,
        "c5_geometry": c5_geom,
        "output_prob_nll_baseline": plausibility,
        "transfer_bm120": modes_out,
        "holm_adjusted_p": holm_summary,
        "role_pr_shuffled_null_c5": pr_shuffled_null,
    }
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, OUT_JSON)
    print(f"SAVED {OUT_JSON}", flush=True)
    return report


def self_test():
    if DATA_JSON.exists():
        integ = verify_dataset_integrity()
        assert integ["n_pairs"] == 120 and integ["n_fav_first"] == 60 and integ["n_fav_second"] == 60
    # Verify both-mirrors-right returns 0.0 for a pure order-only classifier!
    sid = np.repeat(np.arange(10), 2)
    order_only = np.tile([True, False], 10)
    rate, cnt, _ = _both_mirrors_right(order_only, sid)
    assert rate == 0.0 and cnt == 0
    print("BOW_MIRROR238_TEST_OK")


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
        print("ALL_EXP238_DONE_OK")
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
