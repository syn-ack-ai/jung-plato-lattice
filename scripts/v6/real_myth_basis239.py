"""Experiment 239: Real Historical World Folktales in C5 + Held-Out Third-Role-Set Basis Variance Test.

Implements BOTH remaining direct, pre-specified falsifiable tests for the Jungian hypothesis:
  Test 1 — Independently Labeled Real Historical World Folktales in C5 (ZERO LLM Authoring):
    500 genuine human-written 19th/early-20th-century world folktales (`aft_real_myths_500.json`,
    10 official Aarne-Thompson-Uther [ATU] divisions x 50 tales, 132 non-Western tales, 123 unique
    ATU tale types, zero 12-gram overlap with C5 passages) projected through frozen neutral-text
    Procrustes rotations across all 7 LLM families:
      - 10-way ATU zero-shot transfer (Gemma 4 -> 6 rival labs, C5 -> held-out Phi-4 / OLMo 3),
        raw & perp VAL3, on (a) 20 stratified 2-fold splits, (b) 20 strict ATU-tale-type-disjoint
        2-fold splits, and (c) strict Western (n=368) -> Non-Western (n=132) cross-cultural transfer,
        head-to-head against Unigram BoW, TF-IDF Centroid, and TF-IDF L2-Ridge.
      - Effective dimensionality (PR) of 10 real ATU divisions vs. 500-draw shuffled-label PR null
        in C5, plus r(C5_cos, word_cos) and Western vs. Non-Western ATU geometry correlation.
      - Zero-shot cross-century alignment of modern Jungian directions (named_arch235) vs. modern
        Job directions (nonjung_roles237) onto the 10 real historical ATU divisions.

  Test 2 — Reviewer's Held-Out Third-Role-Set Basis Variance Test (V(Q_jung7) vs. V(Q_jobs7)):
    Compares how much variance of completely unseen third role taxonomies is captured by:
      - Q_jung7 (exact rank-7 orthonormal centroid basis of Jung's 8 archetypes in 40 workplace settings)
      - Q_jobs7 (exact rank-7 orthonormal centroid basis of 8 workplace jobs in the same 40 settings)
      - 500 within-setting shuffled-label 7D null bases from each training set (ambient C5 null!)
    Evaluated in C5 (raw & perp VAL3), held-out Phi-4-mini, held-out OLMo 3 7B, and word space on:
      - Held-Out Set 2A: 500 Real Historical World Folktales (10 ATU divisions + Non-Western subset)
      - Held-Out Set 2B: 320 Everyday Family / Kinship Stories (8 roles x 40 brand-new settings,
        zero kinship/role-name ban-list hits, matched ~51-word modern prose)
    With 2,000-draw paired bootstrap 95% CIs on Delta V = V(Q_jung7) - V(Q_jobs7) (resampling
    held-out items/settings AND separately resampling the 40 basis-training settings!).

Usage:
  python real_myth_basis239.py extract_aft [KEY...]
  python real_myth_basis239.py merge_fam
  python real_myth_basis239.py extract_fam [KEY...]
  python real_myth_basis239.py analyze
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

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
FOLK_DIR = SCRATCH_DIR / "folklore_test"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))

AFT_JSON = FOLK_DIR / "aft_real_myths_500.json"
FAM_JSON = FOLK_DIR / "family_roles239_data.json"
OUT_JSON = FOLK_DIR / "real_myth_basis239.json"

for p in (str(REPO_ARCH), str(REPO_FLUENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402

ATU_DIVS = (
    "01_Animal_Trickster",
    "02_Supernatural_Adversary",
    "03_Supernatural_Spouse",
    "05_Supernatural_Helpers",
    "06_Magic_Objects",
    "07_Supernatural_Power",
    "08_Religious_Tales",
    "09_Realistic_Wisdom",
    "10_Stupid_Ogre_Devil",
    "11_Jokes_Tricksters",
)

FAM_ROLES = (
    "Aging_Grandparent",
    "Protective_Parent",
    "Rebellious_Teen",
    "Peacekeeping_Sibling",
    "Estranged_InLaw",
    "Curious_Child",
    "Caregiving_Spouse",
    "Returning_Relative",
)

FAM_BAN_WORDS = (
    "grandparent", "grandparents", "grandfather", "grandmother", "grandpa", "grandma", "nana", "gramps",
    "elderly", "aging", "elder", "elders",
    "parent", "parents", "mother", "mothers", "father", "fathers", "mom", "moms", "dad", "dads",
    "maternal", "paternal", "protective", "protect", "protects", "protected", "protecting",
    "son", "sons", "daughter", "daughters",
    "teen", "teens", "teenager", "teenagers", "adolescent", "adolescents",
    "rebel", "rebels", "rebellious", "rebellion",
    "sibling", "siblings", "brother", "brothers", "sister", "sisters",
    "peacemaker", "peacekeeping", "peacekeeper",
    "in-law", "in-laws", "inlaw", "estranged", "estrangement", "marriage", "married",
    "child", "children", "kid", "kids", "toddler", "toddlers", "curious", "curiosity",
    "spouse", "spouses", "husband", "husbands", "wife", "wives", "partner", "partners", "mate",
    "caregiver", "caregivers", "caregiving",
    "relative", "relatives", "cousin", "cousins", "uncle", "uncles", "aunt", "aunts", "niece", "nieces", "nephew", "nephews",
    "returning", "returnee", "homecoming", "reunion", "family", "families",
    "archetype", "archetypes", "archetypal", "jung", "jungian",
    "hero", "heroic", "shadow", "trickster", "sage", "mentor", "persona", "mask", "anima", "animus", "guardian", "threshold",
    "auditor", "technician", "negotiator", "scheduler", "trainer", "supplier", "journalist", "archivist",
)
_FAM_BAN_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in FAM_BAN_WORDS) + r")\b", re.I)


def merge_and_verify_family():
    rows = []
    for part in range(1, 5):
        p = SCRATCH_DIR / f"family239_part{part}.json"
        assert p.exists(), f"Missing {p}"
        rows.extend(json.loads(p.read_text()))
    rows.sort(key=lambda r: (int(r["setting_id"]), FAM_ROLES.index(r["role"])))
    assert len(rows) == 320, f"Expected 320 rows, got {len(rows)}"

    sids = sorted({int(r["setting_id"]) for r in rows})
    assert sids == list(range(1, 41)), f"Expected setting_ids 1..40, got {len(sids)}"

    ex_settings = {p["setting"].strip().lower() for p in AR.load_arch_pairs() if p.get("setting")}
    _, _, vm = V234.load_valence_data()
    ex_settings |= {x["setting"].strip().lower() for x in vm if x.get("setting")}
    ex_settings |= {x["setting"].strip().lower() for x in NA235.load_named_arch_data()}

    seen = set()
    new_settings = set()
    by_role_wc = {rl: [] for rl in FAM_ROLES}
    ban_hits = []
    wc_bad = []
    for r in rows:
        sid = int(r["setting_id"])
        rl = r["role"]
        assert rl in FAM_ROLES, f"Unknown role {rl}"
        key = (sid, rl)
        assert key not in seen, f"Duplicate {key}"
        seen.add(key)
        new_settings.add(r["setting"].strip().lower())
        txt = r["text"].strip()
        wc = len(txt.split())
        if not (38 <= wc <= 64):
            wc_bad.append((key, wc))
        by_role_wc[rl].append(wc)
        m = _FAM_BAN_RE.findall(txt)
        if m:
            ban_hits.append((key, m))

    overlap = ex_settings & new_settings
    if overlap or ban_hits or wc_bad:
        print("FAMILY INTEGRITY ISSUES:", flush=True)
        if overlap:
            print("  setting overlap:", overlap)
        if ban_hits:
            print(f"  ban hits ({len(ban_hits)}):", ban_hits[:20])
        if wc_bad:
            print(f"  wc out of range ({len(wc_bad)}):", wc_bad[:20])
        raise SystemExit(1)

    wc_stats = {
        rl: {"mean": round(float(np.mean(ws)), 2), "std": round(float(np.std(ws)), 2),
             "min": int(min(ws)), "max": int(max(ws))}
        for rl, ws in by_role_wc.items()
    }
    means = [v["mean"] for v in wc_stats.values()]
    assert max(means) - min(means) <= 6.0, f"Role mean wc spread too wide: {wc_stats}"
    FAM_JSON.write_text(json.dumps(rows, indent=2))
    print(f"MERGED & VERIFIED {FAM_JSON}: 320 stories, 0 setting overlap, 0 ban hits, wc means {min(means):.1f}-{max(means):.1f}", flush=True)
    return {
        "n_stories": len(rows),
        "n_settings": len(sids),
        "n_roles": len(FAM_ROLES),
        "setting_overlap_with_existing": 0,
        "ban_list_hits": 0,
        "word_count_by_role": wc_stats,
    }


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _extract_texts_generic(prefix, key, texts, meta_dict, max_toks=768):
    assert key in PD.MODEL_SPECS
    out_npz = PLATO_CACHE / f"{prefix}_{key}.npz"
    if out_npz.exists():
        print(f"{prefix.upper()} EXTRACT {key}: already exists ({out_npz}), skipping", flush=True)
        return
    spec = PD.MODEL_SPECS[key]

    def save_it(hl_list, hm_list, tlens, ok_f, layer_idx, num_layers):
        hl, hm = np.stack(hl_list, 0), np.stack(hm_list, 0)
        assert np.isfinite(hl).all() and np.isfinite(hm).all()
        pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
        payload = dict(
            last=pack(hl), mean=pack(hm),
            tok_len=np.array(tlens, dtype=np.int32),
            ok=np.array(ok_f, dtype=bool),
            layer_idx=np.int32(layer_idx), num_layers=np.int32(num_layers),
        )
        payload.update(meta_dict)
        PD._atomic_savez(out_npz, **payload)

    if spec["backend"] == "llamacpp":
        port = spec["port"]
        srv = str(Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server")))
        if _port_open(port):
            raise RuntimeError(f"port {port} already open before starting {key}")
        cmd = [
            srv, "-m", spec["path"], "--port", str(port), "--host", "127.0.0.1",
            "--embeddings", "--pooling", "none", "--embd-normalize", "-1",
            "--no-webui", "-c", "2048", "-b", "2048", "-ub", "2048", "-ngl", "99",
        ]
        t0 = time.time()
        print(f"{prefix.upper()} EXTRACT {key}: starting llama-server on :{port} ...", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def post(ep, payload):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{ep}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())

        hl_l, hm_l, tlens, ok_f = [], [], [], []
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
            assert ready
            for i, txt in enumerate(texts):
                toks = post("/tokenize", {"content": txt, "add_special": True})["tokens"]
                assert 2 <= len(toks) <= max_toks, f"{key} bad tok len {len(toks)} at {i}"
                emb = post("/embedding", {"content": txt})
                arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
                assert arr.ndim == 2 and arr.shape[0] == len(toks)
                hl_l.append(arr[-1].copy())
                hm_l.append((arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0]).copy())
                tlens.append(len(toks))
                ok_f.append(True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        save_it(hl_l, hm_l, tlens, ok_f, -1, 24)
        print(f"{prefix.upper()}_EXTRACT_DONE {key}: ok={sum(ok_f)}/{len(texts)} max_tok={max(tlens)} in {time.time() - t0:.1f}s", flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"{prefix.upper()} EXTRACT {key}: loading {spec['path']} ...", flush=True)
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

    hl_l, hm_l, tlens, ok_f = [], [], [], []
    try:
        for i, txt in enumerate(texts):
            ids = PD._text_ids_mlx(tok, txt)
            assert 2 <= len(ids) <= max_toks, f"{key} bad tok len {len(ids)} at {i}"
            rec_buf.clear()
            _ = model(mx.array([ids]))
            assert len(rec_buf) == 1
            h = rec_buf[0].astype(mx.float32)
            hl = h[-1]
            hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
            mx.eval(hl, hm)
            rec_buf.clear()
            hl_l.append(np.array(hl, dtype=np.float32))
            hm_l.append(np.array(hm, dtype=np.float32))
            tlens.append(len(ids))
            ok_f.append(True)
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        rec_buf.clear()
        del layers, tgt_blk, model, tok
        gc.collect()
        mx.clear_cache()

    save_it(hl_l, hm_l, tlens, ok_f, tgt_idx, nl_cnt)
    print(f"{prefix.upper()}_EXTRACT_DONE {key}: ok={sum(ok_f)}/{len(texts)} max_tok={max(tlens)} in {time.time() - t0:.1f}s", flush=True)


def extract_aft(keys):
    rows = json.loads(AFT_JSON.read_text())
    texts = [r["text"].strip() for r in rows]
    meta = {
        "divisions": np.array([r["division"] for r in rows]),
        "atu_ids": np.array([str(r["atu_id"]) for r in rows]),
        "provenance": np.array([r["provenance"] for r in rows]),
        "non_western": np.array([bool(r["non_western"]) for r in rows], dtype=bool),
        "aft_idx": np.array([int(r["aft_idx"]) for r in rows], dtype=np.int32),
    }
    for k in keys:
        _extract_texts_generic("aft239", k, texts, meta, max_toks=768)


def extract_fam(keys):
    rows = json.loads(FAM_JSON.read_text())
    texts = [r["text"].strip() for r in rows]
    meta = {
        "setting_ids": np.array([int(r["setting_id"]) for r in rows], dtype=np.int32),
        "roles": np.array([r["role"] for r in rows]),
    }
    for k in keys:
        _extract_texts_generic("fam239", k, texts, meta, max_toks=512)


# ---------------------------------------------------------------- Analysis Primitives

def _centroid_basis_k(Z, y_idx, n_cls, k_dim=None):
    """Exact rank-(n_cls-1) orthonormal basis spanning the centered class centroids in Z."""
    if k_dim is None:
        k_dim = n_cls - 1
    Zc = Z - Z.mean(axis=0, keepdims=True)
    M = np.stack([Zc[y_idx == c].mean(axis=0) for c in range(n_cls)], axis=0)  # (n_cls, D)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k_dim].T  # (D, k_dim), orthonormal columns


def _eval_basis_capture(Q, Z_tgt, y_tgt, n_cls):
    """Compute exact variance capture metrics of orthonormal basis Q (D, k) on target dataset (Z_tgt, y_tgt):
      - v_dir: mean_c ||Q^T u_c||^2 over unit OvR directions u_c (range [0, 1])
      - v_between: tr(Q^T S_B Q) / tr(S_B), fraction of total between-class centroid variance captured
      - v_item: mean_i ||Q^T z_hat_i||^2 over centered unit-norm individual items
      - per_class_v: [||Q^T u_c||^2 for c in range(n_cls)]
    """
    Zc = Z_tgt - Z_tgt.mean(axis=0, keepdims=True)
    U = NA235._ovr_dirs(Zc, y_tgt, n_cls=n_cls)
    per_c = [float(np.sum((Q.T @ U[c]) ** 2)) for c in range(n_cls)]
    v_dir = float(np.mean(per_c))

    # Between-class scatter S_B = sum_c n_c * mu_c mu_c^T
    M_w = []
    for c in range(n_cls):
        m = (y_tgt == c)
        if m.sum() > 0:
            M_w.append(np.sqrt(float(m.sum())) * Zc[m].mean(axis=0))
    M_w = np.stack(M_w, axis=0)  # (n_cls, D)
    tot_sb = float(np.sum(M_w ** 2))
    cap_sb = float(np.sum((M_w @ Q) ** 2))
    v_between = cap_sb / max(1e-12, tot_sb)

    norms = np.maximum(1e-12, np.linalg.norm(Zc, axis=1, keepdims=True))
    Z_u = Zc / norms
    v_item = float(np.mean(np.sum((Z_u @ Q) ** 2, axis=1)))
    return {"v_dir": v_dir, "v_between": v_between, "v_item": v_item, "per_class_v": per_c}


def _bootstrap_basis_diff_items(Q_a, Q_b, Z_tgt, y_tgt, n_cls, cluster_ids=None, n_boot=2000, seed=239):
    """Paired 2,000-draw bootstrap 95% CI on Delta = V(Q_a) - V(Q_b), either stratified by class
    (when cluster_ids is None) or clustered by setting_id (when cluster_ids is provided)."""
    obs_a = _eval_basis_capture(Q_a, Z_tgt, y_tgt, n_cls)
    obs_b = _eval_basis_capture(Q_b, Z_tgt, y_tgt, n_cls)
    rng = np.random.default_rng(seed)

    if cluster_ids is not None:
        uniq = np.array(sorted(set(int(s) for s in cluster_ids)))
        by_cl = [np.where(cluster_ids == s)[0] for s in uniq]
        def sample_idx():
            ch = rng.choice(len(uniq), size=len(uniq), replace=True)
            return np.concatenate([by_cl[c] for c in ch])
    else:
        by_cls = [np.where(y_tgt == c)[0] for c in range(n_cls)]
        def sample_idx():
            return np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_cls])

    d_dir, d_bet, d_itm = [], [], []
    for _ in range(n_boot):
        idx = sample_idx()
        ea = _eval_basis_capture(Q_a, Z_tgt[idx], y_tgt[idx], n_cls)
        eb = _eval_basis_capture(Q_b, Z_tgt[idx], y_tgt[idx], n_cls)
        d_dir.append(ea["v_dir"] - eb["v_dir"])
        d_bet.append(ea["v_between"] - eb["v_between"])
        d_itm.append(ea["v_item"] - eb["v_item"])

    def pack(va, vb, diffs):
        arr = np.asarray(diffs)
        p_two = float(2.0 * min((np.sum(arr <= 0) + 1) / (len(arr) + 1), (np.sum(arr >= 0) + 1) / (len(arr) + 1)))
        return {
            "v_jung7": round(va, 4),
            "v_jobs7": round(vb, 4),
            "diff_jung_minus_jobs": round(va - vb, 4),
            "ci95": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
            "p_boot_two_sided": round(min(1.0, p_two), 4),
        }

    return {
        "v_dir": pack(obs_a["v_dir"], obs_b["v_dir"], d_dir),
        "v_between": pack(obs_a["v_between"], obs_b["v_between"], d_bet),
        "v_item": pack(obs_a["v_item"], obs_b["v_item"], d_itm),
        "per_class_jung7": [round(x, 4) for x in obs_a["per_class_v"]],
        "per_class_jobs7": [round(x, 4) for x in obs_b["per_class_v"]],
    }


def _bootstrap_basis_diff_train_settings(Z_j, y_j, sids_j, Z_n, y_n, sids_n, Z_tgt, y_tgt, n_cls, n_boot=1000, seed=2395):
    """Bootstrap over the 40 paired training settings defining Q_jung7 and Q_jobs7 simultaneously!"""
    uniq = np.array(sorted(set(int(s) for s in sids_j)))
    by_j = [np.where(sids_j == s)[0] for s in uniq]
    by_n = [np.where(sids_n == s)[0] for s in uniq]
    rng = np.random.default_rng(seed)
    d_dir, d_bet = [], []
    for _ in range(n_boot):
        ch = rng.choice(len(uniq), size=len(uniq), replace=True)
        ij = np.concatenate([by_j[c] for c in ch])
        in_ = np.concatenate([by_n[c] for c in ch])
        Qj = _centroid_basis_k(Z_j[ij], y_j[ij], 8, k_dim=7)
        Qn = _centroid_basis_k(Z_n[in_], y_n[in_], 8, k_dim=7)
        ej = _eval_basis_capture(Qj, Z_tgt, y_tgt, n_cls)
        en = _eval_basis_capture(Qn, Z_tgt, y_tgt, n_cls)
        d_dir.append(ej["v_dir"] - en["v_dir"])
        d_bet.append(ej["v_between"] - en["v_between"])
    return {
        "v_dir_diff_train_boot_ci95": [round(float(np.percentile(d_dir, 2.5)), 4), round(float(np.percentile(d_dir, 97.5)), 4)],
        "v_between_diff_train_boot_ci95": [round(float(np.percentile(d_bet, 2.5)), 4), round(float(np.percentile(d_bet, 97.5)), 4)],
        "jung_gt_jobs_frac_v_dir": round(float(np.mean(np.asarray(d_dir) > 0)), 4),
    }


def _shuffled_null_basis_capture(Z_train, y_train, sids_train, Z_tgt, y_tgt, n_cls, n_draws=500, seed=2391):
    """500-draw within-setting label-shuffled 7D null bases from Z_train evaluated on (Z_tgt, y_tgt)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(int(s) for s in sids_train))
    groups = [np.where(sids_train == s)[0] for s in uniq]
    v_dirs, v_bets, v_itms = [], [], []
    for _ in range(n_draws):
        ys = y_train.copy()
        for g in groups:
            ys[g] = rng.permutation(ys[g])
        Qs = _centroid_basis_k(Z_train, ys, 8, k_dim=7)
        es = _eval_basis_capture(Qs, Z_tgt, y_tgt, n_cls)
        v_dirs.append(es["v_dir"])
        v_bets.append(es["v_between"])
        v_itms.append(es["v_item"])
    return {
        "v_dir_null_mean": round(float(np.mean(v_dirs)), 4),
        "v_dir_null_p95": round(float(np.percentile(v_dirs, 95)), 4),
        "v_between_null_mean": round(float(np.mean(v_bets)), 4),
        "v_between_null_p95": round(float(np.percentile(v_bets, 95)), 4),
        "v_item_null_mean": round(float(np.mean(v_itms)), 4),
        "v_item_null_p95": round(float(np.percentile(v_itms, 95)), 4),
        "_raw_v_dir": v_dirs,
        "_raw_v_bet": v_bets,
    }


def _transfer_multiclass_splits(Z_src, Z_tgt, y_idx, splits, n_cls, Q_src=None, Q_tgt=None):
    """Evaluate zero-shot cross-family OvR transfer across given 2-fold splits (0/1 arrays)."""
    top1s, top2s, maucs = [], [], []
    per_c_aucs = [[] for _ in range(n_cls)]
    for fl in splits:
        all_sc = np.zeros((len(y_idx), n_cls), dtype=np.float64)
        for f in (0, 1):
            tr, te = (fl != f), (fl == f)
            Zs_tr = Z_src[tr] - Z_src[tr].mean(0, keepdims=True)
            Zt_te = Z_tgt[te] - Z_tgt[tr].mean(0, keepdims=True)
            if Q_src is not None:
                Zs_tr = V234._project_out_rows(Zs_tr, Q_src)
            if Q_tgt is not None:
                Zt_te = V234._project_out_rows(Zt_te, Q_tgt)
            U = NA235._ovr_dirs(Zs_tr, y_idx[tr], n_cls=n_cls)
            if Q_src is not None:
                U = np.stack([V234._project_out(u, Q_src) for u in U], axis=0)
            if Q_tgt is not None:
                U = np.stack([V234._project_out(u, Q_tgt) for u in U], axis=0)
            all_sc[te] = Zt_te @ U.T
        aucs = []
        for c in range(n_cls):
            a = NA235._auroc((y_idx == c).astype(int), all_sc[:, c])
            per_c_aucs[c].append(a)
            aucs.append(a)
        maucs.append(float(np.mean(aucs)))
        pred1 = np.argmax(all_sc, axis=1)
        top2 = np.argsort(all_sc, axis=1)[:, -2:]
        top1s.append(float(np.mean(pred1 == y_idx)))
        top2s.append(float(np.mean([y_idx[i] in top2[i] for i in range(len(y_idx))])))
    return {
        "top1_mean": round(float(np.mean(top1s)), 4),
        "top1_std": round(float(np.std(top1s)), 4),
        "top2_mean": round(float(np.mean(top2s)), 4),
        "macro_auroc_mean": round(float(np.mean(maucs)), 4),
        "macro_auroc_std": round(float(np.std(maucs)), 4),
        "per_class_auroc": [round(float(np.mean(x)), 4) for x in per_c_aucs],
        "_top1_splits": top1s,
        "_mauc_splits": maucs,
    }


def _build_word_mats(texts_fit, texts_list, kind="binary_unigram", min_df=1):
    """Pure-NumPy binary unigram or sublinear unigram+bigram TF-IDF vectorizer matching named_arch235."""
    def tok_fn(t):
        ts = re.findall(r"[a-z]{2,}", t.lower())
        if kind == "binary_unigram":
            return ts
        return ts + [f"{ts[j]}_{ts[j + 1]}" for j in range(len(ts) - 1)]

    df = {}
    for t in texts_fit:
        for w in set(tok_fn(t)):
            df[w] = df.get(w, 0) + 1
    vocab = sorted(w for w, c in df.items() if c >= min_df)
    w2i = {w: j for j, w in enumerate(vocab)}
    n_fit = float(len(texts_fit))
    idf = np.array([math.log((1.0 + n_fit) / (1.0 + df[w])) + 1.0 for w in vocab], dtype=np.float64)

    mats = []
    for ts_grp in texts_list:
        X = np.zeros((len(ts_grp), len(vocab)), dtype=np.float64)
        for i, t in enumerate(ts_grp):
            toks = tok_fn(t)
            if kind == "binary_unigram":
                for w in set(toks):
                    j = w2i.get(w)
                    if j is not None:
                        X[i, j] = 1.0
            else:
                cnt = {}
                for w in toks:
                    j = w2i.get(w)
                    if j is not None:
                        cnt[j] = cnt.get(j, 0) + 1
                for j, c in cnt.items():
                    X[i, j] = (1.0 + math.log(c)) * idf[j]
        if kind != "binary_unigram":
            X /= np.maximum(1e-12, np.linalg.norm(X, axis=1, keepdims=True))
        mats.append(X)
    return mats


def _fit_eval_lexical_fold(t_tr, y_tr, t_te, n_cls):
    """Return dict of score matrices (n_te, n_cls) for bow_unigram, tfidf_centroid, tfidf_ridge."""
    Xtr_u, Xte_u = _build_word_mats(t_tr, [t_tr, t_te], kind="binary_unigram", min_df=1)
    mu_u = Xtr_u.mean(0, keepdims=True)
    sc_bow = (Xte_u - mu_u) @ NA235._ovr_dirs(Xtr_u - mu_u, y_tr, n_cls=n_cls).T

    Xtr_tf, Xte_tf = _build_word_mats(t_tr, [t_tr, t_te], kind="tfidf_12gram", min_df=2)
    mu_tf = Xtr_tf.mean(0, keepdims=True)
    sc_tfc = (Xte_tf - mu_tf) @ NA235._ovr_dirs(Xtr_tf - mu_tf, y_tr, n_cls=n_cls).T

    Y_oh = np.full((len(y_tr), n_cls), -1.0, dtype=np.float64)
    for row_i, c in enumerate(y_tr):
        Y_oh[row_i, c] = 1.0
    Xc_tr = Xtr_tf - mu_tf
    Xc_te = Xte_tf - mu_tf
    mu_y = Y_oh.mean(0, keepdims=True)
    dual = np.linalg.solve(Xc_tr @ Xc_tr.T + 1.0 * np.eye(len(y_tr)), Y_oh - mu_y)
    sc_tfr = (Xc_te @ Xc_tr.T) @ dual + mu_y
    return {"bow_unigram": sc_bow, "tfidf_centroid": sc_tfc, "tfidf_ridge": sc_tfr}


def _lexical_multiclass_splits(texts, y_idx, splits, n_cls):
    """Train-only Unigram BoW, TF-IDF Centroid, and TF-IDF L2-Ridge on arbitrary splits & n_cls."""
    out = {name: {"_top1_splits": [], "_mauc_splits": []} for name in ("bow_unigram", "tfidf_centroid", "tfidf_ridge")}
    for fl in splits:
        all_sc = {name: np.zeros((len(y_idx), n_cls), dtype=np.float64) for name in out}
        for f in (0, 1):
            tr, te = np.where(fl != f)[0], np.where(fl == f)[0]
            t_tr = [texts[i] for i in tr]
            t_te = [texts[i] for i in te]
            sc_map = _fit_eval_lexical_fold(t_tr, y_idx[tr], t_te, n_cls)
            for name, sc in sc_map.items():
                all_sc[name][te] = sc
        for name, sc in all_sc.items():
            aucs = [NA235._auroc((y_idx == c).astype(int), sc[:, c]) for c in range(n_cls)]
            out[name]["_mauc_splits"].append(float(np.mean(aucs)))
            out[name]["_top1_splits"].append(float(np.mean(np.argmax(sc, axis=1) == y_idx)))
    for name in out:
        t1, ma = out[name]["_top1_splits"], out[name]["_mauc_splits"]
        out[name]["top1_mean"] = round(float(np.mean(t1)), 4)
        out[name]["top1_std"] = round(float(np.std(t1)), 4)
        out[name]["macro_auroc_mean"] = round(float(np.mean(ma)), 4)
        out[name]["macro_auroc_std"] = round(float(np.std(ma)), 4)
    return out


def _word_space_basis_test(texts_j, y_j, texts_n, y_n, texts_tgt, y_tgt, n_cls_tgt):
    """Exact same 7D basis capture test in shared binary unigram word space & sublinear TF-IDF space!"""
    res = {}
    for kind, mdf in [("binary_unigram", 1), ("tfidf_12gram", 2)]:
        Xj, Xn, Xt = _build_word_mats(texts_j + texts_n + texts_tgt, [texts_j, texts_n, texts_tgt], kind=kind, min_df=mdf)
        Qj = _centroid_basis_k(Xj, y_j, 8, k_dim=7)
        Qn = _centroid_basis_k(Xn, y_n, 8, k_dim=7)
        ej = _eval_basis_capture(Qj, Xt, y_tgt, n_cls_tgt)
        en = _eval_basis_capture(Qn, Xt, y_tgt, n_cls_tgt)
        res[kind] = {
            "vocab_dim": int(Xj.shape[1]),
            "v_dir_jung7": round(ej["v_dir"], 4),
            "v_dir_jobs7": round(en["v_dir"], 4),
            "v_dir_diff": round(ej["v_dir"] - en["v_dir"], 4),
            "v_between_jung7": round(ej["v_between"], 4),
            "v_between_jobs7": round(en["v_between"], 4),
            "v_between_diff": round(ej["v_between"] - en["v_between"], 4),
            "v_item_jung7": round(ej["v_item"], 4),
            "v_item_jobs7": round(en["v_item"], 4),
            "v_item_diff": round(ej["v_item"] - en["v_item"], 4),
        }
    return res


def analyze_all():
    t0 = time.time()
    aft_rows = json.loads(AFT_JSON.read_text())
    fam_rows = json.loads(FAM_JSON.read_text())
    j_rows = NA235.load_named_arch_data()
    n_rows = NJ237.load_nonjung_data()

    aft_texts = [r["text"].strip() for r in aft_rows]
    fam_texts = [r["text"].strip() for r in fam_rows]
    j_texts = [r["text"].strip() for r in j_rows]
    n_texts = [r["text"].strip() for r in n_rows]

    y_aft = np.array([ATU_DIVS.index(r["division"]) for r in aft_rows], dtype=int)
    nw_mask = np.array([bool(r["non_western"]) for r in aft_rows], dtype=bool)
    atu_ids = [str(r["atu_id"]) for r in aft_rows]

    y_fam = np.array([FAM_ROLES.index(r["role"]) for r in fam_rows], dtype=int)
    sids_fam = np.array([int(r["setting_id"]) for r in fam_rows], dtype=int)

    y_j = np.array([NA235.ARCHETYPES.index(r["archetype"]) for r in j_rows], dtype=int)
    sids_j = np.array([int(r["setting_id"]) for r in j_rows], dtype=int)
    y_n = np.array([NJ237.ROLES.index(r["role"]) for r in n_rows], dtype=int)
    sids_n = np.array([int(r["setting_id"]) for r in n_rows], dtype=int)

    # Fit frozen neutral-text Procrustes rotations
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot_st = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato

    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])

    Z_j, Z_n, Z_aft, Z_fam = {}, {}, {}, {}
    Q_val3, u_house, u_sent, u_outval = {}, {}, {}, {}
    for k in PD.ALL_KEYS:
        dj = AR._load_npz_dict(PLATO_CACHE / f"arch235_{k}.npz")
        dn = AR._load_npz_dict(PLATO_CACHE / f"nj237_{k}.npz")
        da = AR._load_npz_dict(PLATO_CACHE / f"aft239_{k}.npz")
        df = AR._load_npz_dict(PLATO_CACHE / f"fam239_{k}.npz")
        Z_j[k] = NA235._map_stories(dj["mean"], rot_st, k, frame="to_c5")
        Z_n[k] = NA235._map_stories(dn["mean"], rot_st, k, frame="to_c5")
        Z_aft[k] = NA235._map_stories(da["mean"], rot_st, k, frame="to_c5")
        Z_fam[k] = NA235._map_stories(df["mean"], rot_st, k, frame="to_c5")

        a = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
        v = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")
        _, _, D_h = AR.map_diffs(a["done_mean"][house_idx], a["neg_mean"][house_idx], rot_st, k, frame="to_c5")
        _, _, D_v = AR.map_diffs(v["pos_mean"], v["neg_mean"], rot_st, k, frame="to_c5")
        groups = v["groups"]
        u_house[k] = AR.house_dir(D_h, house_cats)
        u_sent[k] = AR.unit(D_v[groups == "sentiment_gen"].mean(0))
        u_outval[k] = AR.unit(D_v[groups == "outcome_valence"].mean(0))
        Q_val3[k] = V234._orthonormal_basis([u_sent[k], u_outval[k], u_house[k]])

    # C5 consensus representations
    Zj_c5 = np.mean([Z_j[k] for k in PD.CONSENSUS_KEYS], axis=0)
    Zn_c5 = np.mean([Z_n[k] for k in PD.CONSENSUS_KEYS], axis=0)
    Zaft_c5 = np.mean([Z_aft[k] for k in PD.CONSENSUS_KEYS], axis=0)
    Zfam_c5 = np.mean([Z_fam[k] for k in PD.CONSENSUS_KEYS], axis=0)
    Q3_c5 = V234._orthonormal_basis([
        AR.unit(np.mean([u_sent[k] for k in PD.CONSENSUS_KEYS], axis=0)),
        AR.unit(np.mean([u_outval[k] for k in PD.CONSENSUS_KEYS], axis=0)),
        AR.unit(np.mean([u_house[k] for k in PD.CONSENSUS_KEYS], axis=0)),
    ])

    # =========================================================================
    # TEST 2: REVIEWER'S HELD-OUT THIRD-ROLE-SET BASIS VARIANCE TEST
    # =========================================================================
    print("Running Test 2: Held-Out Basis Variance Test (Q_jung7 vs. Q_jobs7) ...", flush=True)
    basis_results = {}

    def run_basis_suite(space_name, Zj_s, Zn_s, Zaft_s, Zfam_s, Q3_s=None, include_nulls=False):
        out_s = {}
        modes = [("raw", None)]
        if Q3_s is not None:
            modes.append(("perp_val3", Q3_s))
        for mname, Qp in modes:
            Aj = V234._project_out_rows(Zj_s, Qp) if Qp is not None else Zj_s
            An = V234._project_out_rows(Zn_s, Qp) if Qp is not None else Zn_s
            Aa = V234._project_out_rows(Zaft_s, Qp) if Qp is not None else Zaft_s
            Af = V234._project_out_rows(Zfam_s, Qp) if Qp is not None else Zfam_s

            Qj = _centroid_basis_k(Aj, y_j, 8, k_dim=7)
            Qn = _centroid_basis_k(An, y_n, 8, k_dim=7)

            # Set 2B: Everyday Family / Kinship Roles (320 stories, 40 brand-new settings)
            fam_boot = _bootstrap_basis_diff_items(Qj, Qn, Af, y_fam, 8, cluster_ids=sids_fam, n_boot=2000, seed=2390)
            fam_tr_boot = _bootstrap_basis_diff_train_settings(Aj, y_j, sids_j, An, y_n, sids_n, Af, y_fam, 8, n_boot=1000, seed=2391)
            fam_block = {**fam_boot, "train_setting_boot": fam_tr_boot, "roles": list(FAM_ROLES)}

            # Set 2A: Real Historical World Folktales (500 tales, 10 ATU divisions)
            aft_boot = _bootstrap_basis_diff_items(Qj, Qn, Aa, y_aft, 10, cluster_ids=None, n_boot=2000, seed=2392)
            aft_tr_boot = _bootstrap_basis_diff_train_settings(Aj, y_j, sids_j, An, y_n, sids_n, Aa, y_aft, 10, n_boot=1000, seed=2393)
            aft_block = {**aft_boot, "train_setting_boot": aft_tr_boot, "divisions": list(ATU_DIVS)}

            # Set 2A-NW: Non-Western Real Historical Folktales (n=132, item-level + 7 divisions with n>=6)
            nw_divs = [c for c in range(10) if int(np.sum((y_aft == c) & nw_mask)) >= 6]
            nw_keep = nw_mask & np.isin(y_aft, nw_divs)
            y_nw = np.array([nw_divs.index(int(c)) for c in y_aft[nw_keep]], dtype=int)
            aft_nw_boot = _bootstrap_basis_diff_items(Qj, Qn, Aa[nw_keep], y_nw, len(nw_divs), cluster_ids=None, n_boot=2000, seed=2394)
            aft_nw_block = {**aft_nw_boot, "n_tales": int(nw_keep.sum()), "n_divs": len(nw_divs), "divisions": [ATU_DIVS[c] for c in nw_divs]}

            if include_nulls:
                nj_fam = _shuffled_null_basis_capture(Aj, y_j, sids_j, Af, y_fam, 8, n_draws=500, seed=2395)
                nn_fam = _shuffled_null_basis_capture(An, y_n, sids_n, Af, y_fam, 8, n_draws=500, seed=2396)
                fam_block["null_jung_shuf"] = {k: v for k, v in nj_fam.items() if not k.startswith("_")}
                fam_block["null_jobs_shuf"] = {k: v for k, v in nn_fam.items() if not k.startswith("_")}
                fam_block["p_jung_gt_shuf_v_dir"] = round(float((sum(x >= fam_boot["v_dir"]["v_jung7"] for x in nj_fam["_raw_v_dir"]) + 1) / 501), 4)
                fam_block["p_jobs_gt_shuf_v_dir"] = round(float((sum(x >= fam_boot["v_dir"]["v_jobs7"] for x in nn_fam["_raw_v_dir"]) + 1) / 501), 4)

                nj_aft = _shuffled_null_basis_capture(Aj, y_j, sids_j, Aa, y_aft, 10, n_draws=500, seed=2397)
                nn_aft = _shuffled_null_basis_capture(An, y_n, sids_n, Aa, y_aft, 10, n_draws=500, seed=2398)
                aft_block["null_jung_shuf"] = {k: v for k, v in nj_aft.items() if not k.startswith("_")}
                aft_block["null_jobs_shuf"] = {k: v for k, v in nn_aft.items() if not k.startswith("_")}
                aft_block["p_jung_gt_shuf_v_dir"] = round(float((sum(x >= aft_boot["v_dir"]["v_jung7"] for x in nj_aft["_raw_v_dir"]) + 1) / 501), 4)
                aft_block["p_jobs_gt_shuf_v_dir"] = round(float((sum(x >= aft_boot["v_dir"]["v_jobs7"] for x in nn_aft["_raw_v_dir"]) + 1) / 501), 4)

            out_s[mname] = {
                "heldout_family_roles_320": fam_block,
                "heldout_real_myths_500": aft_block,
                "heldout_real_myths_nonwestern": aft_nw_block,
            }
        return out_s

    basis_results["c5_consensus"] = run_basis_suite("c5", Zj_c5, Zn_c5, Zaft_c5, Zfam_c5, Q3_s=Q3_c5, include_nulls=True)
    for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b"):
        basis_results[hk] = run_basis_suite(hk, Z_j[hk], Z_n[hk], Z_aft[hk], Z_fam[hk], Q3_s=Q_val3[hk], include_nulls=False)

    # Word-space basis capture control
    basis_results["word_space_control"] = {
        "heldout_family_roles_320": _word_space_basis_test(j_texts, y_j, n_texts, y_n, fam_texts, y_fam, 8),
        "heldout_real_myths_500": _word_space_basis_test(j_texts, y_j, n_texts, y_n, aft_texts, y_aft, 10),
    }

    # Cross-set direction cosines: 8 Family Roles vs 8 Jungian & vs 8 Jobs in C5 (perp VAL3 + raw) + shuffled null max |cos|
    def cross_cos_audit(Z_a, y_a, names_a, sids_a, Z_b, y_b, names_b, Q3):
        Za_p = V234._project_out_rows(Z_a - Z_a.mean(0, keepdims=True), Q3)
        Zb_p = V234._project_out_rows(Z_b - Z_b.mean(0, keepdims=True), Q3)
        Ua = NA235._ovr_dirs(Za_p, y_a, n_cls=len(names_a))
        Ub = NA235._ovr_dirs(Zb_p, y_b, n_cls=len(names_b))
        M = Ub @ Ua.T  # (len_b, len_a)
        # Shuffled null for max |cos| (500 within-setting shuffles of a)
        rng = np.random.default_rng(2399)
        uniq = sorted(set(int(s) for s in sids_a))
        grps = [np.where(sids_a == s)[0] for s in uniq]
        null_max = []
        for _ in range(500):
            ys = y_a.copy()
            for g in grps:
                ys[g] = rng.permutation(ys[g])
            Us = NA235._ovr_dirs(Za_p, ys, n_cls=len(names_a))
            null_max.append(float(np.max(np.abs(Ub @ Us.T))))
        pairs = []
        for i, nb in enumerate(names_b):
            for j, na in enumerate(names_a):
                pairs.append((float(M[i, j]), nb, na))
        pairs.sort(key=lambda x: abs(x[0]), reverse=True)
        return {
            "max_abs_cos": round(float(np.max(np.abs(M))), 4),
            "null_p95_max_abs_cos": round(float(np.percentile(null_max, 95)), 4),
            "top_pairs": [{"target": nb, "source": na, "cos": round(v, 3)} for v, nb, na in pairs[:10]],
            "matrix": {nb: {na: round(float(M[i, j]), 3) for j, na in enumerate(names_a)} for i, nb in enumerate(names_b)},
        }

    basis_results["c5_perp_val3_cross_cos"] = {
        "family_vs_jung": cross_cos_audit(Zj_c5, y_j, NA235.ARCHETYPES, sids_j, Zfam_c5, y_fam, FAM_ROLES, Q3_c5),
        "family_vs_jobs": cross_cos_audit(Zn_c5, y_n, NJ237.ROLES, sids_n, Zfam_c5, y_fam, FAM_ROLES, Q3_c5),
        "aft_vs_jung": cross_cos_audit(Zj_c5, y_j, NA235.ARCHETYPES, sids_j, Zaft_c5, y_aft, ATU_DIVS, Q3_c5),
        "aft_vs_jobs": cross_cos_audit(Zn_c5, y_n, NJ237.ROLES, sids_n, Zaft_c5, y_aft, ATU_DIVS, Q3_c5),
    }

    # Also evaluate Family Roles 8-way transfer (sanity check that Family Roles are clean & strong in C5!)
    uniq_fs = sorted(set(int(s) for s in sids_fam))
    fam_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23950 + s_i)
        perm = rng.permutation(uniq_fs)
        f0 = set(int(x) for x in perm[:20])
        fam_splits.append(np.array([0 if int(s) in f0 else 1 for s in sids_fam], dtype=int))
    fam_lex = _lexical_multiclass_splits(fam_texts, y_fam, fam_splits, 8)
    fam_transfer = {"lexical": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fam_lex.items()}, "gemma_to_6": {}}
    src_k = "gemma4_e4b"
    targets = [k for k in PD.ALL_KEYS if k != src_k]
    for tk in targets:
        r_r = _transfer_multiclass_splits(Z_fam[src_k], Z_fam[tk], y_fam, fam_splits, 8)
        r_p = _transfer_multiclass_splits(Z_fam[src_k], Z_fam[tk], y_fam, fam_splits, 8, Q_src=Q_val3[src_k], Q_tgt=Q_val3[tk])
        fam_transfer["gemma_to_6"][tk] = {
            "raw_top1": r_r["top1_mean"], "raw_mauc": r_r["macro_auroc_mean"],
            "perp_val3_top1": r_p["top1_mean"], "perp_val3_mauc": r_p["macro_auroc_mean"],
        }

    # Family PR vs 500 shuffled null in C5
    Zc_fam = Zfam_c5 - Zfam_c5.mean(0, keepdims=True)
    fam_pr_obs, _ = NA235._effective_rank_pr(NA235._ovr_dirs(Zc_fam, y_fam, 8))
    rng_fpr = np.random.default_rng(23977)
    fam_null_prs = []
    for _ in range(500):
        ys = y_fam.copy()
        for s in uniq_fs:
            ix = np.where(sids_fam == s)[0]
            ys[ix] = rng_fpr.permutation(ys[ix])
        fam_null_prs.append(NA235._effective_rank_pr(NA235._ovr_dirs(Zc_fam, ys, 8))[0])
    fam_transfer["c5_pr_vs_shuffled_null"] = {
        "obs_pr": round(float(fam_pr_obs), 3),
        "null_mean_pr": round(float(np.mean(fam_null_prs)), 3),
        "null_p05_p95": [round(float(np.percentile(fam_null_prs, 5)), 3), round(float(np.percentile(fam_null_prs, 95)), 3)],
        "p_null_le_obs": round(float((sum(x <= fam_pr_obs for x in fam_null_prs) + 1) / 501), 4),
    }

    # =========================================================================
    # TEST 1: REAL HISTORICAL WORLD FOLKTALES (500 TALES, 10 ATU DIVISIONS) IN C5
    # =========================================================================
    print("Running Test 1: 500 Real Historical World Folktales in C5 ...", flush=True)

    # (1a) 20 Stratified 2-fold splits (exact 25/division per fold)
    aft_strat_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(239100 + s_i)
        fl = np.zeros(len(y_aft), dtype=int)
        for c in range(10):
            ix = rng.permutation(np.where(y_aft == c)[0])
            fl[ix[25:]] = 1
        aft_strat_splits.append(fl)

    # (1b) 20 Strict ATU-tale-type-disjoint 2-fold splits (ALL variants of same ATU ID stay in same fold!)
    aft_atu_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(239200 + s_i)
        fl = np.zeros(len(y_aft), dtype=int)
        for c in range(10):
            ix_c = np.where(y_aft == c)[0]
            # Group by atu_id inside division c and greedily balance fold counts
            by_atu = {}
            for i in ix_c:
                by_atu.setdefault(atu_ids[i], []).append(i)
            keys_a = list(by_atu.keys())
            rng.shuffle(keys_a)
            keys_a.sort(key=lambda k: len(by_atu[k]), reverse=True)
            c0, c1 = 0, 0
            for ka in keys_a:
                if c0 <= c1:
                    for i in by_atu[ka]:
                        fl[i] = 0
                    c0 += len(by_atu[ka])
                else:
                    for i in by_atu[ka]:
                        fl[i] = 1
                    c1 += len(by_atu[ka])
        aft_atu_splits.append(fl)

    lex_strat = _lexical_multiclass_splits(aft_texts, y_aft, aft_strat_splits, 10)
    lex_atu = _lexical_multiclass_splits(aft_texts, y_aft, aft_atu_splits, 10)

    def eval_aft_split_suite(splits, lex_d):
        res = {
            "lexical": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in lex_d.items()},
            "gemma_to_6": {},
            "c5_to_heldout": {},
        }
        for tk in targets:
            r_r = _transfer_multiclass_splits(Z_aft[src_k], Z_aft[tk], y_aft, splits, 10)
            r_p = _transfer_multiclass_splits(Z_aft[src_k], Z_aft[tk], y_aft, splits, 10, Q_src=Q_val3[src_k], Q_tgt=Q_val3[tk])
            res["gemma_to_6"][tk] = {
                "raw_top1": r_r["top1_mean"], "raw_top2": r_r["top2_mean"], "raw_mauc": r_r["macro_auroc_mean"],
                "raw_wins_vs_tfr_top1_20": int(sum(x > y for x, y in zip(r_r["_top1_splits"], lex_d["tfidf_ridge"]["_top1_splits"]))),
                "raw_wins_vs_tfr_mauc_20": int(sum(x > y for x, y in zip(r_r["_mauc_splits"], lex_d["tfidf_ridge"]["_mauc_splits"]))),
                "perp_val3_top1": r_p["top1_mean"], "perp_val3_top2": r_p["top2_mean"], "perp_val3_mauc": r_p["macro_auroc_mean"],
                "perp_val3_wins_vs_tfr_top1_20": int(sum(x > y for x, y in zip(r_p["_top1_splits"], lex_d["tfidf_ridge"]["_top1_splits"]))),
                "perp_val3_wins_vs_tfr_mauc_20": int(sum(x > y for x, y in zip(r_p["_mauc_splits"], lex_d["tfidf_ridge"]["_mauc_splits"]))),
                "perp_val3_per_div_auroc": {ATU_DIVS[c]: r_p["per_class_auroc"][c] for c in range(10)},
            }
        for hk in ("phi4_mini", "olmo3_7b"):
            r_r = _transfer_multiclass_splits(Zaft_c5, Z_aft[hk], y_aft, splits, 10)
            r_p = _transfer_multiclass_splits(Zaft_c5, Z_aft[hk], y_aft, splits, 10, Q_src=Q3_c5, Q_tgt=Q_val3[hk])
            res["c5_to_heldout"][hk] = {
                "raw_top1": r_r["top1_mean"], "raw_mauc": r_r["macro_auroc_mean"],
                "raw_wins_vs_tfr_top1_20": int(sum(x > y for x, y in zip(r_r["_top1_splits"], lex_d["tfidf_ridge"]["_top1_splits"]))),
                "raw_wins_vs_tfr_mauc_20": int(sum(x > y for x, y in zip(r_r["_mauc_splits"], lex_d["tfidf_ridge"]["_mauc_splits"]))),
                "perp_val3_top1": r_p["top1_mean"], "perp_val3_mauc": r_p["macro_auroc_mean"],
                "perp_val3_wins_vs_tfr_top1_20": int(sum(x > y for x, y in zip(r_p["_top1_splits"], lex_d["tfidf_ridge"]["_top1_splits"]))),
                "perp_val3_wins_vs_tfr_mauc_20": int(sum(x > y for x, y in zip(r_p["_mauc_splits"], lex_d["tfidf_ridge"]["_mauc_splits"]))),
            }
        return res

    aft_transfer_strat = eval_aft_split_suite(aft_strat_splits, lex_strat)
    aft_transfer_atu = eval_aft_split_suite(aft_atu_splits, lex_atu)

    # (1c) Strict Cross-Cultural Transfer: Train strictly on Western (n=368) -> Test on Non-Western (n=132)!
    tr_w = ~nw_mask
    te_nw = nw_mask
    t_tr = [aft_texts[i] for i in np.where(tr_w)[0]]
    t_te = [aft_texts[i] for i in np.where(te_nw)[0]]
    sc_xc = _fit_eval_lexical_fold(t_tr, y_aft[tr_w], t_te, 10)
    xc_lex = {}
    for name, sc in sc_xc.items():
        aucs = [NA235._auroc((y_aft[te_nw] == c).astype(int), sc[:, c]) for c in range(10)]
        xc_lex[name] = {
            "top1_10way": round(float(np.mean(np.argmax(sc, axis=1) == y_aft[te_nw])), 4),
            "macro_auroc": round(float(np.mean(aucs)), 4),
        }

    xc_neural = {}
    for label, Zs, Zt, Qs, Qt in (
        [(f"gemma_to_{tk}", Z_aft[src_k], Z_aft[tk], Q_val3[src_k], Q_val3[tk]) for tk in targets]
        + [(f"c5_to_{hk}", Zaft_c5, Z_aft[hk], Q3_c5, Q_val3[hk]) for hk in ("phi4_mini", "olmo3_7b")]
    ):
        Zs_tr = Zs[tr_w] - Zs[tr_w].mean(0, keepdims=True)
        Zt_te = Zt[te_nw] - Zt[tr_w].mean(0, keepdims=True)
        U_r = NA235._ovr_dirs(Zs_tr, y_aft[tr_w], 10)
        sc_r = Zt_te @ U_r.T
        aucs_r = [NA235._auroc((y_aft[te_nw] == c).astype(int), sc_r[:, c]) for c in range(10)]

        Zs_p = V234._project_out_rows(Zs_tr, Qs)
        Zt_p = V234._project_out_rows(Zt_te, Qt)
        U_p = np.stack([V234._project_out(V234._project_out(u, Qs), Qt) for u in NA235._ovr_dirs(Zs_p, y_aft[tr_w], 10)], 0)
        sc_p = Zt_p @ U_p.T
        aucs_p = [NA235._auroc((y_aft[te_nw] == c).astype(int), sc_p[:, c]) for c in range(10)]

        xc_neural[label] = {
            "raw_top1": round(float(np.mean(np.argmax(sc_r, axis=1) == y_aft[te_nw])), 4),
            "raw_mauc": round(float(np.mean(aucs_r)), 4),
            "perp_val3_top1": round(float(np.mean(np.argmax(sc_p, axis=1) == y_aft[te_nw])), 4),
            "perp_val3_mauc": round(float(np.mean(aucs_p)), 4),
        }

    # (1d) Real Historical ATU Geometry in C5: PR vs 500-draw shuffled null + word-space correlation + Western vs Non-Western invariance
    Zc_aft = Zaft_c5 - Zaft_c5.mean(0, keepdims=True)
    Zc_aft_p = V234._project_out_rows(Zc_aft, Q3_c5)
    U_aft_raw = NA235._ovr_dirs(Zc_aft, y_aft, 10)
    U_aft_p3 = NA235._ovr_dirs(Zc_aft_p, y_aft, 10)
    pr_aft_raw, _ = NA235._effective_rank_pr(U_aft_raw)
    pr_aft_p3, _ = NA235._effective_rank_pr(U_aft_p3)
    rng_apr = np.random.default_rng(23988)
    null_pr_raw, null_pr_p3 = [], []
    for _ in range(500):
        ys = rng_apr.permutation(y_aft)
        null_pr_raw.append(NA235._effective_rank_pr(NA235._ovr_dirs(Zc_aft, ys, 10))[0])
        null_pr_p3.append(NA235._effective_rank_pr(NA235._ovr_dirs(Zc_aft_p, ys, 10))[0])

    # Word-space geometry correlation on the 10 ATU divisions
    (Xw_aft,) = _build_word_mats(aft_texts, [aft_texts], kind="binary_unigram", min_df=1)
    Xw_c = Xw_aft - Xw_aft.mean(0, keepdims=True)
    Uw_aft = NA235._ovr_dirs(Xw_c, y_aft, 10)
    pr_aft_word, _ = NA235._effective_rank_pr(Uw_aft)
    C_aft = U_aft_raw @ U_aft_raw.T
    Cw_aft = Uw_aft @ Uw_aft.T
    tri10 = np.triu_indices(10, k=1)
    r_aft_c5_word = float(np.corrcoef(C_aft[tri10], Cw_aft[tri10])[0, 1])

    # Western vs Non-Western ATU pairwise cosine correlation (over 7 divisions with >=6 non-Western tales!)
    nw_divs7 = [c for c in range(10) if int(np.sum((y_aft == c) & nw_mask)) >= 6]
    tri7 = np.triu_indices(len(nw_divs7), k=1)
    y_w7 = np.array([nw_divs7.index(int(c)) for c in y_aft[tr_w & np.isin(y_aft, nw_divs7)]])
    y_nw7 = np.array([nw_divs7.index(int(c)) for c in y_aft[te_nw & np.isin(y_aft, nw_divs7)]])
    Zw7 = Zaft_c5[tr_w & np.isin(y_aft, nw_divs7)]
    Znw7 = Zaft_c5[te_nw & np.isin(y_aft, nw_divs7)]
    Uw_c5_7 = NA235._ovr_dirs(Zw7 - Zw7.mean(0, keepdims=True), y_w7, len(nw_divs7))
    Unw_c5_7 = NA235._ovr_dirs(Znw7 - Znw7.mean(0, keepdims=True), y_nw7, len(nw_divs7))
    r_w_vs_nw_c5 = float(np.corrcoef((Uw_c5_7 @ Uw_c5_7.T)[tri7], (Unw_c5_7 @ Unw_c5_7.T)[tri7])[0, 1])
    diag_w_vs_nw_c5 = [round(float(np.dot(Uw_c5_7[i], Unw_c5_7[i])), 3) for i in range(len(nw_divs7))]

    Xw7 = Xw_aft[tr_w & np.isin(y_aft, nw_divs7)]
    Xnw7 = Xw_aft[te_nw & np.isin(y_aft, nw_divs7)]
    Uw_wd_7 = NA235._ovr_dirs(Xw7 - Xw7.mean(0, keepdims=True), y_w7, len(nw_divs7))
    Unw_wd_7 = NA235._ovr_dirs(Xnw7 - Xnw7.mean(0, keepdims=True), y_nw7, len(nw_divs7))
    r_w_vs_nw_word = float(np.corrcoef((Uw_wd_7 @ Uw_wd_7.T)[tri7], (Unw_wd_7 @ Unw_wd_7.T)[tri7])[0, 1])

    # (1e) Cross-Century Zero-Shot Scoring: Project 500 real tales onto 8 modern Jungian vs 8 modern Job directions (eta^2 & z-score table!)
    U_jung_p3 = NA235._ovr_dirs(V234._project_out_rows(Zj_c5 - Zj_c5.mean(0, keepdims=True), Q3_c5), y_j, 8)
    U_jobs_p3 = NA235._ovr_dirs(V234._project_out_rows(Zn_c5 - Zn_c5.mean(0, keepdims=True), Q3_c5), y_n, 8)

    def projection_eta2_table(U_src, src_names, Z_t, y_t, tgt_names):
        sc = Z_t @ U_src.T  # (N, 8)
        sc = (sc - sc.mean(0, keepdims=True)) / np.maximum(1e-12, sc.std(0, keepdims=True))
        eta2 = {}
        table = {}
        for j, sn in enumerate(src_names):
            tot_ss = float(np.sum((sc[:, j] - sc[:, j].mean()) ** 2))
            bet_ss = sum(float(np.sum(y_t == c)) * float((sc[y_t == c, j].mean() - sc[:, j].mean()) ** 2) for c in range(len(tgt_names)))
            eta2[sn] = round(bet_ss / max(1e-12, tot_ss), 4)
        for c, tn in enumerate(tgt_names):
            table[tn] = {sn: round(float(sc[y_t == c, j].mean()), 3) for j, sn in enumerate(src_names)}
        return {"mean_eta2": round(float(np.mean(list(eta2.values()))), 4), "per_dir_eta2": eta2, "div_mean_z": table}

    cross_century = {
        "jung_perp_val3_on_aft500": projection_eta2_table(U_jung_p3, NA235.ARCHETYPES, Zc_aft_p, y_aft, ATU_DIVS),
        "jobs_perp_val3_on_aft500": projection_eta2_table(U_jobs_p3, NJ237.ROLES, Zc_aft_p, y_aft, ATU_DIVS),
        "jung_perp_val3_on_fam320": projection_eta2_table(U_jung_p3, NA235.ARCHETYPES, V234._project_out_rows(Zc_fam, Q3_c5), y_fam, FAM_ROLES),
        "jobs_perp_val3_on_fam320": projection_eta2_table(U_jobs_p3, NJ237.ROLES, V234._project_out_rows(Zc_fam, Q3_c5), y_fam, FAM_ROLES),
    }

    aft_geometry = {
        "pr_raw": round(float(pr_aft_raw), 3),
        "pr_raw_null_mean": round(float(np.mean(null_pr_raw)), 3),
        "pr_raw_null_p05_p95": [round(float(np.percentile(null_pr_raw, 5)), 3), round(float(np.percentile(null_pr_raw, 95)), 3)],
        "pr_raw_p_null_le_obs": round(float((sum(x <= pr_aft_raw for x in null_pr_raw) + 1) / 501), 4),
        "pr_perp_val3": round(float(pr_aft_p3), 3),
        "pr_perp_val3_null_mean": round(float(np.mean(null_pr_p3)), 3),
        "pr_perp_val3_null_p05_p95": [round(float(np.percentile(null_pr_p3, 5)), 3), round(float(np.percentile(null_pr_p3, 95)), 3)],
        "pr_perp_val3_p_null_le_obs": round(float((sum(x <= pr_aft_p3 for x in null_pr_p3) + 1) / 501), 4),
        "word_space_pr": round(float(pr_aft_word), 3),
        "corr_c5_vs_word_cos_r": round(r_aft_c5_word, 4),
        "western_vs_nonwestern_7div_geom_r_c5": round(r_w_vs_nw_c5, 4),
        "western_vs_nonwestern_7div_geom_r_word": round(r_w_vs_nw_word, 4),
        "western_vs_nonwestern_7div_same_div_cos_c5": {ATU_DIVS[nw_divs7[i]]: diag_w_vs_nw_c5[i] for i in range(len(nw_divs7))},
    }

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "test2_heldout_basis_variance": basis_results,
        "test2b_family_roles_transfer": fam_transfer,
        "test1_real_myths_500_stratified_20splits": aft_transfer_strat,
        "test1_real_myths_500_atu_disjoint_20splits": aft_transfer_atu,
        "test1_real_myths_western_to_nonwestern": {"lexical": xc_lex, "neural": xc_neural},
        "test1_real_myths_c5_geometry": aft_geometry,
        "test1_cross_century_projections": cross_century,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} in {report['elapsed_sec']}s", flush=True)
    return report


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "extract_aft":
        keys = sys.argv[2:] if len(sys.argv) > 2 else list(PD.ALL_KEYS)
        extract_aft(keys)
    elif cmd == "merge_fam":
        merge_and_verify_family()
    elif cmd == "extract_fam":
        keys = sys.argv[2:] if len(sys.argv) > 2 else list(PD.ALL_KEYS)
        extract_fam(keys)
    elif cmd == "analyze":
        analyze_all()
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
