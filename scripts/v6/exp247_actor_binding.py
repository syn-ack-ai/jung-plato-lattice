"""Experiment 247 (FROZEN PRE-SPECIFIED PROTOCOL): 2x2 Crossed Minimal-Pair Actor-Conditioned
Compositional Binding Benchmark (`bm_actor2x2_247`) + 500-Random-Direction Specificity Nulls +
Independent Refit Erasure Decoders.

THIS SCRIPT IS WRITTEN AND SHA-256 FROZEN PRIOR TO AUTHORING ANY ITEM OF `bm_actor2x2_247_data.json`.

Pre-specified dataset contract (`bm_actor2x2_247_data.json`):
  - 240 exact-BoW pairs = 60 distinct institutional/social settings x 2 actor types (`benign` vs `adverse`)
    x 2 clause orders (`orig` vs `mirror`).
  - MINIMAL-PAIR ACTOR CROSSING (HARD-ASSERTED BY CODE!):
    For every setting `s` and order `o`, the `benign` and `adverse` pairs use the EXACT SAME predicate
    clauses (exact same verbs, recipient adjectives, and recipient nouns!) — ONLY the subject/actor NP
    changes from benign (e.g., "The ethical inspector") to adverse (e.g., "The corrupt inspector") —
    and the gold label REVERSES (`adverse.pos` predicate == `benign.neg` predicate, and `adverse.neg`
    predicate == `benign.pos` predicate!).
  - MATHEMATICAL GUARANTEE (verified by hard assertions in `merge_and_verify()`):
    1. Exact identical word-bag on ALL 240/240 pairs (`Counter(words(pos)) == Counter(words(neg))`).
    2. Local `lex_verb_adj_polarity` (AND its reversed sign, AND Astra's first-diff-word local rule!)
       MUST score EXACT `0.5000` (`120/240`)! Any purely local verb-x-recipient rule is mathematically
       capped at 50% regardless of sign!
    3. Zero setting overlap with `bm120`, `bm_rel242`, `bm_neut245`, or `bm_polconf245`.

Single Pre-Specified Primary Cell & Strict Controls:
  - Primary cell: `d50` (50% depth), `mean` pool & `last` token (co-primary `gemma4_e4b -> 5` and
    `qwen35_9b -> 5`; `d50.last` is primary for `cv2_val5` and `d50.mean` is primary for zero-shot
    `xdom_bmneut_val5` and `xdom_bm120_val5`, matching Round-2 empirical structure without post-hoc fishing!).
  - Controls addressing EVERY Round-2 finding from Fable 5.2 & Astra 6.2:
    1. **500-random-direction two-sided specificity null** (strictly orthogonal to erased subspace!) for
       EVERY `xdom_*` readout AND every `cv2` readout across ALL 5 benchmarks (`bm120`, `bm_rel242`,
       `bm_neut245`, `bm_polconf245`, `bm_actor2x2_247`), with exact Holm-adjusted p-values saved explicitly!
    2. **Independent refit 2-fold CV evaluative decoders** (`refit_sent_cv_acc`, `refit_house_cv_acc`)
       trained freshly on the erased `val234` & `HOUSE` states (ZERO floating-point residue renormalization!).
    3. **Full 20-split refit-CV permutation test** (500 setting-flips refitting ALL 20 splits — exact match
       to the 20-split headline accuracy!).
    4. **Quad-accuracy ("ALL-4-RIGHT per setting": benign_orig & benign_mirror & adverse_orig & adverse_mirror)**
       plus separate `benign_acc` and `adverse_acc` so a model cannot score >50% by solving only one actor polarity!
    5. **Exact whole-sentence AND disjoint-changed-token-only `DeltaNLL <= 0` strata** with item AND setting
       counts for EVERY readout (including `bm_neut245 -> bm_polconf245` and `bm_actor2x2_247`)!
    6. **Refit-in-bootstrap paired depth contrasts** (`d50 - d30` and `d50 - d70`, 1,000 setting resamples
       refitting source directions in every draw!) across all 5 depths (`15%..85%`)!
"""
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH_DIR / "exp247_round3"
EXP245_DIR = SCRATCH_DIR / "exp245_controls"
EXP242_DIR = SCRATCH_DIR / "exp240_242"
REPO_ARCH = SCRIPT_DIR.parents[1] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(REPO_FLUENT / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_swap236 as BS236  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import exp245_binding as E245  # noqa: E402

MLX_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)

DATA_JSON = REPO_ARCH / "selfshape" / "bm_actor2x2_247_data.json"
OUT_JSON = REPO_FLUENT / "plato" / "v6_exp247_binding_results.json"


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _words(t):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower())


def _first_diff_local_rule(pos_txt, neg_txt):
    """Exact Astra-6.2 first-differing-word local verb-x-adj rule (checks both +1 and -1 orientations!)."""
    wp, wn = _words(pos_txt), _words(neg_txt)
    def score_one(ws, idx0):
        # Find first frozen polarity adjective within [idx0, idx0+5]
        adj_i, adj_p = None, 0
        for i in range(idx0, min(len(ws), idx0 + 6)):
            if ws[i] in E245.POS_LEX:
                adj_i, adj_p = i, 1; break
            if ws[i] in E245.NEG_LEX:
                adj_i, adj_p = i, -1; break
        if adj_i is None:
            return 0
        # Nearest frozen polarity verb/word within +-5 of adj_i
        best_d, v_p = 99, 0
        for j in range(max(0, adj_i - 5), min(len(ws), adj_i + 6)):
            if j == adj_i:
                continue
            p = 1 if ws[j] in E245.POS_LEX else (-1 if ws[j] in E245.NEG_LEX else 0)
            if p != 0 and abs(j - adj_i) < best_d:
                best_d, v_p = abs(j - adj_i), p
        return adj_p * v_p

    i0 = 0
    while i0 < len(wp) and i0 < len(wn) and wp[i0] == wn[i0]:
        i0 += 1
    return score_one(wp, i0) - score_one(wn, i0)


def _strip_subject_prefix(sent, actor_np):
    """Verify sentence starts with actor_np and return exact predicate string after actor_np."""
    s = sent.strip()
    a = actor_np.strip()
    assert s.startswith(a), f"Sentence does not start with actor '{a}': {s}"
    return s[len(a):].strip()


def merge_and_verify():
    rows = []
    for part in (1, 2):
        p = EXP_DIR / f"bm_actor2x2_247_part{part}.json"
        assert p.exists(), f"Missing {p}"
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("bm_actor2x2_240") or data.get("pairs") or data.get("items")
        rows.extend(data)

    rows.sort(key=lambda r: (str(r["setting_id"]), 0 if r["actor_type"] == "benign" else 1, 0 if r["order"] == "orig" else 1))
    assert len(rows) == 240, f"Expected 240 rows, got {len(rows)}"

    existing_settings = {r["setting"].strip().lower() for r in BM238.load_mirror_data()}
    for pth, k_j in [
        (EXP242_DIR / "bm_rel242_data.json", "bm_rel_120"),
        (EXP245_DIR / "bm_neut245_data.json", "bm_neut_120"),
        (EXP245_DIR / "bm_polconf245_data.json", "bm_polconf_120"),
    ]:
        if pth.exists():
            existing_settings |= {r["setting"].strip().lower() for r in json.loads(pth.read_text())[k_j]}

    by_sid = {}
    bad = []
    for r in rows:
        sid = str(r["setting_id"])
        assert r["actor_type"] in ("benign", "adverse"), f"bad actor_type {r}"
        assert r["order"] in ("orig", "mirror"), f"bad order {r}"
        assert r["fav_pos"] in ("first", "second"), f"bad fav_pos {r}"
        p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
        if p_txt == n_txt or Counter(_words(p_txt)) != Counter(_words(n_txt)):
            bad.append((r["id"], "bow_mismatch"))
        if r["setting"].strip().lower() in existing_settings:
            bad.append((r["id"], "setting_overlap"))
        by_sid.setdefault(sid, []).append(r)

    assert not bad, f"Verification failures: {bad[:10]}"
    assert len(by_sid) == 60, f"Expected 60 settings, got {len(by_sid)}"

    # HARD MINIMAL-PAIR CROSSING ASSERTION: exact predicate reversal between benign & adverse!
    for sid, grp in by_sid.items():
        assert len(grp) == 4, f"{sid} has {len(grp)} rows"
        combos = {(x["actor_type"], x["order"]): x for x in grp}
        assert set(combos.keys()) == {("benign", "orig"), ("benign", "mirror"), ("adverse", "orig"), ("adverse", "mirror")}
        for ord_k in ("orig", "mirror"):
            rb = combos[("benign", ord_k)]
            ra = combos[("adverse", ord_k)]
            pb_pos = _strip_subject_prefix(rb["pos"], rb["actor_np"])
            pb_neg = _strip_subject_prefix(rb["neg"], rb["actor_np"])
            pa_pos = _strip_subject_prefix(ra["pos"], ra["actor_np"])
            pa_neg = _strip_subject_prefix(ra["neg"], ra["actor_np"])
            assert pb_pos == pa_neg and pb_neg == pa_pos, (
                f"Minimal-pair predicate mismatch on {sid} {ord_k}!\n  benign pos={pb_pos}\n  adverse neg={pa_neg}\n  benign neg={pb_neg}\n  adverse pos={pa_pos}"
            )
            assert rb["actor_np"].strip() != ra["actor_np"].strip()

    # Check BOTH local polarity rules score EXACT 0.5000 (on predicate alone & full sentence)!
    pred_diffs = []
    first_diff_scores = []
    for r in rows:
        pp = _strip_subject_prefix(r["pos"], r["actor_np"])
        pn = _strip_subject_prefix(r["neg"], r["actor_np"])
        pred_diffs.append(E245._lex_polarity_pair_score(pp) - E245._lex_polarity_pair_score(pn))
        first_diff_scores.append(_first_diff_local_rule(r["pos"], r["neg"]))
    pred_diffs = np.asarray(pred_diffs)
    first_diff_scores = np.asarray(first_diff_scores)
    pred_acc = float(np.mean(np.where(pred_diffs > 0, 1.0, np.where(pred_diffs < 0, 0.0, 0.5))))
    fd_acc = float(np.mean(np.where(first_diff_scores > 0, 1.0, np.where(first_diff_scores < 0, 0.0, 0.5))))
    assert abs(pred_acc - 0.5) < 1e-9, f"Predicate polarity not 0.5000: {pred_acc}"
    assert abs(fd_acc - 0.5) < 1e-9, f"First-diff local rule not 0.5000: {fd_acc}"
    # Verify predicate polarity actually discriminates within benign (100%) and adverse (0%)!
    ben_mask = np.array([r["actor_type"] == "benign" for r in rows])
    ben_pred_acc = float(np.mean(pred_diffs[ben_mask] > 0))
    adv_pred_acc = float(np.mean(pred_diffs[~ben_mask] > 0))
    assert ben_pred_acc >= 0.95 and adv_pred_acc <= 0.05, f"Predicate polarity weak: ben={ben_pred_acc}, adv={adv_pred_acc}"

    DATA_JSON.write_text(json.dumps({"bm_actor2x2_240": rows}, indent=2))
    print(
        f"MERGED & VERIFIED {DATA_JSON} (sha256={_sha256_file(DATA_JSON)[:16]}...): "
        f"240/240 exact BoW, 60 minimal-pair 2x2 settings, pred_lex_acc={pred_acc:.4f} "
        f"(benign={ben_pred_acc:.3f}, adverse={adv_pred_acc:.3f}), astra_first_diff_acc={fd_acc:.4f}",
        flush=True,
    )
    return rows


def _disjoint_changed_token_masks(tok, p_txt, n_txt, ids_p, ids_n):
    """Mask ONLY tokens whose decoded word differs at aligned positions (excluding unchanged intervening clauses!)."""
    # Use difflib SequenceMatcher on token IDs directly!
    import difflib
    sm = difflib.SequenceMatcher(None, ids_p[1:], ids_n[1:])
    mp = np.zeros(max(1, len(ids_p) - 1), dtype=bool)
    mn = np.zeros(max(1, len(ids_n) - 1), dtype=bool)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            mp[i1:i2] = True
            mn[j1:j2] = True
    if not mp.any():
        mp[:] = True
    if not mn.any():
        mn[:] = True
    return mp, mn


def extract_actor2x2(keys=MLX_KEYS):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    rows = json.loads(DATA_JSON.read_text())["bm_actor2x2_240"]
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    for key in keys:
        out_npz = DEPTH_DIR / f"bmactor247_{key}.npz"
        if out_npz.exists():
            print(f"BMACTOR247 EXTRACT {key}: exists, skipping", flush=True)
            continue
        spec = PD.MODEL_SPECS[key]
        mx.set_cache_limit(2 << 30)
        t0 = time.time()
        print(f"BMACTOR247 EXTRACT {key}: loading {spec['path']} ...", flush=True)
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

        n_d = len(DEPTH_FRACS)
        pl, pm, nl, nm = ([[] for _ in range(n_d)] for _ in range(4))
        p_nll, n_nll, p_chg, n_chg = [], [], [], []
        try:
            for r in rows:
                p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
                ids_p = PD._text_ids_mlx(tok, p_txt)
                ids_n = PD._text_ids_mlx(tok, n_txt)
                mp, mn = _disjoint_changed_token_masks(tok, p_txt, n_txt, ids_p, ids_n)
                for side, ids, tmask in (("pos", ids_p, mp), ("neg", ids_n, mn)):
                    rec_buf.clear()
                    x = mx.array([ids])
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    ce = nn.losses.cross_entropy(logits[0, :-1, :].astype(mx.float32), x[0, 1:], reduction="none")
                    nll_m = mx.mean(ce)
                    eval_list = [ce, nll_m]
                    svecs = []
                    for s in range(n_d):
                        h = rec_buf[s].astype(mx.float32)
                        hl = h[-1]
                        hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
                        svecs.append((hl, hm))
                        eval_list.extend([hl, hm])
                    mx.eval(*eval_list)
                    rec_buf.clear()
                    ce_np = np.array(ce, dtype=np.float32)
                    chg_m = float(ce_np[tmask].mean())
                    if side == "pos":
                        p_nll.append(float(nll_m.item())); p_chg.append(chg_m)
                        for s, (hl, hm) in enumerate(svecs):
                            pl[s].append(np.array(hl, dtype=np.float32))
                            pm[s].append(np.array(hm, dtype=np.float32))
                    else:
                        n_nll.append(float(nll_m.item())); n_chg.append(chg_m)
                        for s, (hl, hm) in enumerate(svecs):
                            nl[s].append(np.array(hl, dtype=np.float32))
                            nm[s].append(np.array(hm, dtype=np.float32))
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            rec_buf.clear()
            del layers, model, tok
            gc.collect()
            mx.clear_cache()

        pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
        PD._atomic_savez(
            out_npz,
            pos_last=pack(np.stack([np.stack(x, 0) for x in pl], 0)),
            pos_mean=pack(np.stack([np.stack(x, 0) for x in pm], 0)),
            neg_last=pack(np.stack([np.stack(x, 0) for x in nl], 0)),
            neg_mean=pack(np.stack([np.stack(x, 0) for x in nm], 0)),
            pos_nll_mean=np.array(p_nll, dtype=np.float32),
            neg_nll_mean=np.array(n_nll, dtype=np.float32),
            pos_changed_nll_mean=np.array(p_chg, dtype=np.float32),
            neg_changed_nll_mean=np.array(n_chg, dtype=np.float32),
            layer_idxs=np.array(layer_idxs, dtype=np.int32),
            num_layers=np.int32(nl_cnt),
            depth_fracs=np.array(DEPTH_FRACS, dtype=np.float32),
        )
        print(f"BMACTOR247_EXTRACT_DONE {key}: layers={layer_idxs}/{nl_cnt} in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Analysis with ALL Round-2 Controls!

def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for rank, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - rank) * pv))
        out[k] = round(run, 5)
    return out


def _refit_cv_eval_probe(D_erased, group_ids, n_splits=10, seed=2471):
    """REAL independent 2-fold group-disjoint linear probe refitted on erased evaluative states!
    (Fixes the Round-2 floating-point residue renormalization bug!)"""
    uniq = sorted(set(group_ids))
    g2i = {g: i for i, g in enumerate(uniq)}
    gi = np.array([g2i[g] for g in group_ids], dtype=int)
    accs = []
    for si in range(n_splits):
        rng = np.random.default_rng(seed + si)
        f0 = set(int(x) for x in rng.permutation(len(uniq))[: len(uniq) // 2])
        fl = np.array([0 if int(x) in f0 else 1 for x in gi])
        oof = np.zeros(len(gi))
        for f in (0, 1):
            tr, te = (fl != f), (fl == f)
            mu = D_erased[tr].mean(0)
            nrm = float(np.linalg.norm(mu))
            if nrm < 1e-10:
                oof[te] = 0.0
            else:
                oof[te] = D_erased[te] @ (mu / nrm)
        accs.append(float(np.mean(np.where(oof > 0, 1.0, np.where(oof < 0, 0.0, 0.5)))))
    return round(float(np.mean(accs)), 4)


def _rand500_specificity_null(Dt_erased, Qt_erased, obs_acc, n_rand=500, seed=2472):
    """500-random-direction two-sided specificity null strictly orthogonal to Qt_erased!"""
    D = Dt_erased.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_rand, D))
    if Qt_erased is not None and Qt_erased.shape[1] > 0:
        R = V234._project_out_rows(R, Qt_erased)
    R /= np.maximum(1e-12, np.linalg.norm(R, axis=1, keepdims=True))
    sc = Dt_erased @ R.T  # (N, 500)
    accs = (sc > 0).mean(axis=0)
    abs_dev = np.abs(accs - 0.5)
    obs_dev = abs(obs_acc - 0.5)
    p_two = float((np.sum(abs_dev >= obs_dev - 1e-12) + 1) / (n_rand + 1))
    p_one = float((np.sum(accs >= obs_acc - 1e-12) + 1) / (n_rand + 1))
    return {
        "rand500_max_acc": round(float(accs.max()), 4),
        "rand500_p95_two_sided": round(float(0.5 + np.percentile(abs_dev, 95)), 4),
        "p_vs_rand500_two_sided": round(p_two, 5),
        "p_vs_rand500_one_sided": round(p_one, 5),
    }


def _all_k_right_per_setting(correct_bool, sid_idx):
    uniq = sorted(set(int(s) for s in sid_idx))
    n_all, n_none = 0, 0
    for s in uniq:
        ix = np.where(sid_idx == s)[0]
        c = correct_bool[ix]
        if np.all(c):
            n_all += 1
        elif not np.any(c):
            n_none += 1
    G = len(uniq)
    return round(n_all / G, 4), [n_all, G - n_all - n_none, n_none]


def _setting_uncentered_perm_generic(scores, sid_idx, n_perm=2000, seed=2473):
    """Exact uncentered setting-flip Monte Carlo null for 2-item OR 4-item settings!"""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    grp_sc = [scores[sid_idx == s] for s in uniq]
    obs_acc = float(np.mean(scores > 0))
    obs_all, counts = _all_k_right_per_setting(scores > 0, sid_idx)
    M = np.stack(grp_sc, axis=0)  # (G, K)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, G, 1))
    F = flips * M[None, :, :]
    null_acc = (F > 0).mean(axis=(1, 2))
    null_all = np.all(F > 0, axis=2).mean(axis=1)
    p_acc = float((np.sum(null_acc >= obs_acc - 1e-12) + 1) / (n_perm + 1))
    p_all = float((np.sum(null_all >= obs_all - 1e-12) + 1) / (n_perm + 1))
    return {
        "acc": round(obs_acc, 4),
        "all_in_setting_right": round(obs_all, 4),
        "counts_all_split_none": counts,
        "p_mc_uncentered_acc": round(p_acc, 5),
        "p_mc_uncentered_all": round(p_all, 5),
    }


def _cv2_20split_refit_perm(Ds_p, Dt_p, Qt, sid_idx, splits, n_perm=500, seed=2474):
    """Refit ALL 20 splits in every setting-flip permutation so permutation statistic == headline 20-split mean!"""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    sid2g = {s: i for i, s in enumerate(uniq)}
    g_of = np.array([sid2g[int(s)] for s in sid_idx], dtype=int)

    def eval_20(signs_g):
        sgn = signs_g[g_of]
        Ds_f = Ds_p * sgn[:, None]
        Dt_f = Dt_p * sgn[:, None]
        accs, alls = [], []
        oof_sum = np.zeros(len(sid_idx), dtype=np.float64)
        for fl in splits:
            oof = np.zeros(len(sid_idx), dtype=np.float64)
            for f in (0, 1):
                tr, te = (fl != f), (fl == f)
                ut = AR.unit(Ds_f[tr].mean(0))
                if Qt is not None and Qt.shape[1] > 0:
                    ut = V234._project_out(ut, Qt)
                oof[te] = Dt_f[te] @ ut
            accs.append(float(np.mean(oof > 0)))
            ar, _ = _all_k_right_per_setting(oof > 0, sid_idx)
            alls.append(ar)
            oof_sum += oof / len(splits)
        return float(np.mean(accs)), float(np.mean(alls)), oof_sum

    obs_acc, obs_all, obs_oof_mean = eval_20(np.ones(G))
    rng = np.random.default_rng(seed)
    ge_a, ge_k = 0, 0
    for _ in range(n_perm):
        sg = rng.choice([-1.0, 1.0], size=G)
        a, k, _ = eval_20(sg)
        if a >= obs_acc - 1e-12:
            ge_a += 1
        if k >= obs_all - 1e-12:
            ge_k += 1
    return {
        "cv2_20split_acc": round(obs_acc, 4),
        "cv2_20split_all_right": round(obs_all, 4),
        "p_mc_20split_refit_acc": round((ge_a + 1) / (n_perm + 1), 5),
        "p_mc_20split_refit_all": round((ge_k + 1) / (n_perm + 1), 5),
        "_oof_mean": obs_oof_mean,
    }


def _refit_paired_depth_boot(Ds_by_d, Dt_by_d, Qs_by_d, Qt_by_d, sid_idx, splits, tgts, n_boot=1000, seed=2475):
    """Astra-6.2-grade paired depth bootstrap: resamples whole settings and REFITS source directions
    under the resampled setting weights in ALL 20 splits at d30, d50, and d70 simultaneously!"""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    g_rows = [np.where(sid_idx == s)[0] for s in uniq]
    # Pre-project Ds and Dt at each depth (d30, d50, d70)
    Ds_p, Dt_p = {}, {}
    for dk in ("d30", "d50", "d70"):
        Ds_p[dk] = V234._project_out_rows(Ds_by_d[dk], Qs_by_d[dk])
        Dt_p[dk] = {tk: V234._project_out_rows(Dt_by_d[dk][tk], Qt_by_d[dk][tk]) for tk in tgts}

    def eval_with_weights(w_g):
        # w_g: integer count per setting (G,)
        w_row = np.zeros(len(sid_idx), dtype=np.float64)
        for gi, rix in enumerate(g_rows):
            w_row[rix] = w_g[gi]
        tot_w = float(w_row.sum())
        res_d = {}
        for dk in ("d30", "d50", "d70"):
            t_accs = []
            for tk in tgts:
                Qt = Qt_by_d[dk][tk]
                sp_accs = []
                for fl in splits:
                    oof = np.zeros(len(sid_idx))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        wtr = w_row[tr]
                        if wtr.sum() <= 0:
                            continue
                        ut = AR.unit((Ds_p[dk][tr] * wtr[:, None]).sum(0))
                        ut = V234._project_out(ut, Qt)
                        oof[te] = Dt_p[dk][tk][te] @ ut
                    sp_accs.append(float(np.sum(w_row * (oof > 0)) / max(1e-12, tot_w)))
                t_accs.append(float(np.mean(sp_accs)))
            res_d[dk] = float(np.mean(t_accs))
        return res_d

    obs = eval_with_weights(np.ones(G))
    rng = np.random.default_rng(seed)
    d50_30, d50_70 = [], []
    for _ in range(n_boot):
        ch = rng.integers(0, G, size=G)
        wg = np.bincount(ch, minlength=G).astype(np.float64)
        r = eval_with_weights(wg)
        d50_30.append(r["d50"] - r["d30"])
        d50_70.append(r["d50"] - r["d70"])
    d50_30, d50_70 = np.asarray(d50_30), np.asarray(d50_70)
    return {
        "d30": round(obs["d30"], 4), "d50": round(obs["d50"], 4), "d70": round(obs["d70"], 4),
        "d50_minus_d30": {
            "obs": round(obs["d50"] - obs["d30"], 4),
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_30, 1.25)), 4), round(float(np.percentile(d50_30, 98.75)), 4)],
        },
        "d50_minus_d70": {
            "obs": round(obs["d50"] - obs["d70"], 4),
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_70, 1.25)), 4), round(float(np.percentile(d50_70, 98.75)), 4)],
        },
    }


def analyze_all():
    t0 = time.time()
    actor_rows = json.loads(DATA_JSON.read_text())["bm_actor2x2_240"]
    benches = {
        "bm120": BM238.load_mirror_data(),
        "bm_rel242": json.loads((EXP242_DIR / "bm_rel242_data.json").read_text())["bm_rel_120"],
        "bm_neut245": json.loads((EXP245_DIR / "bm_neut245_data.json").read_text())["bm_neut_120"],
        "bm_polconf245": json.loads((EXP245_DIR / "bm_polconf245_data.json").read_text())["bm_polconf_120"],
        "bm_actor2x2_247": actor_rows,
    }
    ben_mask_247 = np.array([r["actor_type"] == "benign" for r in actor_rows], dtype=bool)
    adv_mask_247 = ~ben_mask_247

    manifest = {
        "script_sha256": _sha256_file(__file__),
        "data_sha256": _sha256_file(DATA_JSON),
    }

    sid_map, split_map = {}, {}
    for bname, rows in benches.items():
        sbase = {"bm120": 23800, "bm_rel242": 24200, "bm_neut245": 24500, "bm_polconf245": 24600, "bm_actor2x2_247": 24700}[bname]
        sid_idx, splits = E245._sid_and_splits(rows, sbase)
        sid_map[bname] = sid_idx
        split_map[bname] = splits

    # Lexical & local polarity baselines on bm_actor2x2_247
    lex_247 = {
        "bow_unigram": round(float(np.mean(BS236._bow_2fold_pairs(actor_rows, "pos", "neg", BS236._words, split_map["bm_actor2x2_247"]))), 4),
        "bow_bigram": round(float(np.mean(BS236._bow_2fold_pairs(actor_rows, "pos", "neg", BS236._bigrams, split_map["bm_actor2x2_247"]))), 4),
        "bow_skipbigram_g4": round(float(np.mean(BS236._bow_2fold_pairs(actor_rows, "pos", "neg", E245._skip_bigrams, split_map["bm_actor2x2_247"]))), 4),
        "bow_trigram": round(float(np.mean(BS236._bow_2fold_pairs(actor_rows, "pos", "neg", E245._trigrams, split_map["bm_actor2x2_247"]))), 4),
        "full_sent_lex_polarity": E245._eval_lex_polarity(actor_rows, sid_map["bm_actor2x2_247"]),
        "predicate_only_lex_polarity_acc": 0.5000,
        "astra_first_diff_local_rule_acc": 0.5000,
    }

    # NLL baselines (whole-sentence AND disjoint-changed-tokens-only on bm_actor2x2_247!)
    dnll_whole, dnll_chg = {}, {}
    nll_247 = {}
    for bname, pref in [("bm_rel242", "bmrel242"), ("bm_neut245", "bmneut245"), ("bm_polconf245", "bmpolconf245"), ("bm_actor2x2_247", "bmactor247")]:
        dnll_whole[bname] = {}
        dnll_chg[bname] = {}
        for k in MLX_KEYS:
            dr = np.load(DEPTH_DIR / f"{pref}_{k}.npz")
            dm = dr["neg_nll_mean"].astype(np.float64) - dr["pos_nll_mean"].astype(np.float64)
            dnll_whole[bname][k] = dm
            if "neg_changed_nll_mean" in dr:
                dc = dr["neg_changed_nll_mean"].astype(np.float64) - dr["pos_changed_nll_mean"].astype(np.float64)
                dnll_chg[bname][k] = dc
                nll_247[k] = {
                    "whole_sent": _setting_uncentered_perm_generic(dm, sid_map[bname], n_perm=1000, seed=2471),
                    "whole_benign_acc": round(float(np.mean(dm[ben_mask_247] > 0)), 4),
                    "whole_adverse_acc": round(float(np.mean(dm[adv_mask_247] > 0)), 4),
                    "changed_tokens_only": _setting_uncentered_perm_generic(dc, sid_map[bname], n_perm=1000, seed=2472),
                }

    # Evaluative calibration metadata (with group IDs for REAL refit CV decoders!)
    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    house_groups = [str(pairs_arch[i].get("setting", pairs_arch[i].get("id", i))) for i in house_idx]
    sg_rows, ov_rows, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    sg_groups = [f"{r['domain']}_{i}" for i, r in enumerate(sg_rows)]
    n_sg = len(sg_rows)

    # Cache mapped matrices across depths (for 5-depth curves & refit-in-bootstrap depth CIs on bm_actor2x2_247!)
    Z_all_d = {pool: {} for pool in ("last", "mean")}
    Q5_all_d = {pool: {} for pool in ("last", "mean")}
    by_depth_actor = {}
    d50_full = {}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        print(f"Exp247 analyzing depth {dkey} ...", flush=True)
        by_depth_actor[dkey] = {}
        for pool_key in ("last", "mean"):
            rot = E245._fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
            Z = {b: {} for b in benches}
            bases_k, us_k, uh_k, Dv_k, Dh_k = {}, {}, {}, {}, {}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                Z["bm120"][k] = E245._map_mat(d240[f"bm120_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"bm120_neg_{pool_key}"][slot], rot, k)
                for bname, pref in [("bm_rel242", "bmrel242"), ("bm_neut245", "bmneut245"), ("bm_polconf245", "bmpolconf245"), ("bm_actor2x2_247", "bmactor247")]:
                    dr = np.load(DEPTH_DIR / f"{pref}_{k}.npz")
                    Z[bname][k] = E245._map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - E245._map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
                Dh = E245._map_mat(d240[f"house_pos_{pool_key}"][slot][house_idx], rot, k) - E245._map_mat(d240[f"house_neg_{pool_key}"][slot][house_idx], rot, k)
                Dv = E245._map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
                Dv_k[k], Dh_k[k] = Dv, Dh
                bases_k[k], us_k[k], uh_k[k] = E245._build_valence_erasure_bases(Dv, Dh, sg_doms, house_cats)

            if dkey in ("d30", "d50", "d70"):
                Z_all_d[pool_key][dkey] = Z["bm_actor2x2_247"]
                Q5_all_d[pool_key][dkey] = {k: bases_k[k]["val5"] for k in MLX_KEYS}

            # Track 5-depth curve for bm_actor2x2_247
            by_depth_actor[dkey][pool_key] = {}
            for src_k in ("gemma4_e4b", "qwen35_9b"):
                tgts = [k for k in MLX_KEYS if k != src_k]
                Ds = Z["bm_actor2x2_247"][src_k]
                Qs5 = bases_k[src_k]["val5"]
                u_bm120 = AR.unit(Z["bm120"][src_k].mean(0))
                u_neut = AR.unit(Z["bm_neut245"][src_k].mean(0))
                accs_cv, accs_xd_bm, accs_xd_nt, accs_sent, bens_cv, advs_cv, alls_cv = [], [], [], [], [], [], []
                for tk in tgts:
                    Dt = Z["bm_actor2x2_247"][tk]
                    Qt5 = bases_k[tk]["val5"]
                    Dt5 = V234._project_out_rows(Dt, Qt5)
                    accs_sent.append(float(np.mean((Dt @ us_k[src_k]) > 0)))
                    accs_xd_bm.append(float(np.mean((Dt5 @ V234._project_out(V234._project_out(u_bm120, Qs5), Qt5)) > 0)))
                    accs_xd_nt.append(float(np.mean((Dt5 @ V234._project_out(V234._project_out(u_neut, Qs5), Qt5)) > 0)))
                    sp_a, sp_b, sp_ad, sp_al = [], [], [], []
                    for fl in split_map["bm_actor2x2_247"]:
                        oof = np.zeros(len(actor_rows))
                        for f in (0, 1):
                            tr, te = (fl != f), (fl == f)
                            ut = V234._project_out(V234._project_out(AR.unit(Ds[tr].mean(0)), Qs5), Qt5)
                            oof[te] = Dt5[te] @ ut
                        sp_a.append(float(np.mean(oof > 0)))
                        sp_b.append(float(np.mean(oof[ben_mask_247] > 0)))
                        sp_ad.append(float(np.mean(oof[adv_mask_247] > 0)))
                        al_, _ = _all_k_right_per_setting(oof > 0, sid_map["bm_actor2x2_247"])
                        sp_al.append(al_)
                    accs_cv.append(float(np.mean(sp_a)))
                    bens_cv.append(float(np.mean(sp_b)))
                    advs_cv.append(float(np.mean(sp_ad)))
                    alls_cv.append(float(np.mean(sp_al)))
                by_depth_actor[dkey][pool_key][src_k] = {
                    "zs_sent": round(float(np.mean(accs_sent)), 4),
                    "xdom_bm120_val5": round(float(np.mean(accs_xd_bm)), 4),
                    "xdom_bmneut_val5": round(float(np.mean(accs_xd_nt)), 4),
                    "cv2_val5": round(float(np.mean(accs_cv)), 4),
                    "cv2_val5_benign": round(float(np.mean(bens_cv)), 4),
                    "cv2_val5_adverse": round(float(np.mean(advs_cv)), 4),
                    "cv2_val5_all4_right": round(float(np.mean(alls_cv)), 4),
                }

            # At primary d50: FULL Round-2 Audit Suite across ALL 5 benchmarks!
            if dkey == "d50":
                # First: REAL independent refit CV sentiment & HOUSE decodability under each rank-k erasure!
                refit_erasure_check = {}
                for ek in ("raw", "val2", "val5", "val5_house", "eval_svd_12", "eval_svd_24", "eval_svd_40"):
                    sent_cv_models, house_cv_models = {}, {}
                    for k in MLX_KEYS:
                        Qe = bases_k[k][ek]
                        Dv_e = V234._project_out_rows(Dv_k[k][:n_sg], Qe) if Qe.shape[1] > 0 else Dv_k[k][:n_sg]
                        Dh_e = V234._project_out_rows(Dh_k[k], Qe) if Qe.shape[1] > 0 else Dh_k[k]
                        sent_cv_models[k] = _refit_cv_eval_probe(Dv_e, sg_groups, n_splits=10, seed=24710)
                        house_cv_models[k] = _refit_cv_eval_probe(Dh_e, house_groups, n_splits=10, seed=24720)
                    refit_erasure_check[ek] = {
                        "rank": int(bases_k["gemma4_e4b"][ek].shape[1]),
                        "refit_sent_cv_mean": round(float(np.mean(list(sent_cv_models.values()))), 4),
                        "refit_house_cv_mean": round(float(np.mean(list(house_cv_models.values()))), 4),
                        "per_model_sent_cv": sent_cv_models,
                        "per_model_house_cv": house_cv_models,
                    }

                pool_full = {"refit_evaluative_erasure_check": refit_erasure_check, "sources": {}}
                for src_k in ("gemma4_e4b", "qwen35_9b"):
                    tgts = [k for k in MLX_KEYS if k != src_k]
                    Qs5 = bases_k[src_k]["val5"]
                    u_bm120 = AR.unit(Z["bm120"][src_k].mean(0))
                    u_neut = AR.unit(Z["bm_neut245"][src_k].mean(0))
                    src_out = {}

                    for bname in ("bm_actor2x2_247", "bm_polconf245", "bm_neut245", "bm_rel242", "bm120"):
                        sid_idx = sid_map[bname]
                        splits = split_map[bname]
                        Ds = Z[bname][src_k]
                        Ds5 = V234._project_out_rows(Ds, Qs5)
                        tgt_out = {}

                        for tk in tgts:
                            Dt = Z[bname][tk]
                            Qt5 = bases_k[tk]["val5"]
                            Dt5 = V234._project_out_rows(Dt, Qt5)

                            # Zero-shot sentiment (both orientations!)
                            sc_s = Dt @ us_k[src_k]
                            sent_acc = round(float(np.mean(sc_s > 0)), 4)

                            # Zero-shot xdom bm120 & bm_neut245 (+ 500-random-direction specificity null + uncentered setting-flip perm!)
                            ub5 = V234._project_out(V234._project_out(u_bm120, Qs5), Qt5)
                            sc_xb = Dt5 @ ub5
                            res_xb = _setting_uncentered_perm_generic(sc_xb, sid_idx, n_perm=2000, seed=2473)
                            res_xb["specificity_rand500"] = _rand500_specificity_null(Dt5, Qt5, res_xb["acc"], n_rand=500, seed=2474)

                            un5 = V234._project_out(V234._project_out(u_neut, Qs5), Qt5)
                            sc_xn = Dt5 @ un5
                            res_xn = _setting_uncentered_perm_generic(sc_xn, sid_idx, n_perm=2000, seed=2475)
                            res_xn["specificity_rand500"] = _rand500_specificity_null(Dt5, Qt5, res_xn["acc"], n_rand=500, seed=2476)

                            # Full 20-split refit-CV permutation test + 500-rand specificity null on OOF mean!
                            cv_res = _cv2_20split_refit_perm(Ds5, Dt5, Qt5, sid_idx, splits, n_perm=500, seed=2477)
                            oof_m = cv_res.pop("_oof_mean")
                            cv_res["specificity_rand500"] = _rand500_specificity_null(Dt5, Qt5, cv_res["cv2_20split_acc"], n_rand=500, seed=2478)

                            entry = {
                                "zs_sent_pos_acc": sent_acc,
                                "zs_sent_neg_acc": round(1.0 - sent_acc, 4),
                                "xdom_bm120_val5": res_xb,
                                "xdom_bmneut_val5": res_xn,
                                "cv2_val5": cv_res,
                            }

                            if bname == "bm_actor2x2_247":
                                entry["actor_split_breakdown"] = {
                                    "zs_sent_benign": round(float(np.mean(sc_s[ben_mask_247] > 0)), 4),
                                    "zs_sent_adverse": round(float(np.mean(sc_s[adv_mask_247] > 0)), 4),
                                    "xdom_bm120_benign": round(float(np.mean(sc_xb[ben_mask_247] > 0)), 4),
                                    "xdom_bm120_adverse": round(float(np.mean(sc_xb[adv_mask_247] > 0)), 4),
                                    "xdom_bmneut_benign": round(float(np.mean(sc_xn[ben_mask_247] > 0)), 4),
                                    "xdom_bmneut_adverse": round(float(np.mean(sc_xn[adv_mask_247] > 0)), 4),
                                    "cv2_benign": round(float(np.mean(oof_m[ben_mask_247] > 0)), 4),
                                    "cv2_adverse": round(float(np.mean(oof_m[adv_mask_247] > 0)), 4),
                                }
                                # Rank-k erasure sweep on bm_actor2x2_247 (20 splits!)
                                esw = {}
                                for ek in ("raw", "val2", "val5", "val5_house", "eval_svd_12", "eval_svd_24", "eval_svd_40"):
                                    Qse, Qte = bases_k[src_k][ek], bases_k[tk][ek]
                                    Dse = V234._project_out_rows(Ds, Qse) if Qse.shape[1] > 0 else Ds
                                    Dte = V234._project_out_rows(Dt, Qte) if Qte.shape[1] > 0 else Dt
                                    ube = V234._project_out(V234._project_out(u_bm120, Qse), Qte) if Qte.shape[1] > 0 else u_bm120
                                    une = V234._project_out(V234._project_out(u_neut, Qse), Qte) if Qte.shape[1] > 0 else u_neut
                                    ca, cb, cad = [], [], []
                                    for fl in splits:
                                        oo = np.zeros(len(actor_rows))
                                        for f in (0, 1):
                                            tr, te = (fl != f), (fl == f)
                                            ut = AR.unit(Dse[tr].mean(0))
                                            if Qte.shape[1] > 0:
                                                ut = V234._project_out(ut, Qte)
                                            oo[te] = Dte[te] @ ut
                                        ca.append(float(np.mean(oo > 0)))
                                        cb.append(float(np.mean(oo[ben_mask_247] > 0)))
                                        cad.append(float(np.mean(oo[adv_mask_247] > 0)))
                                    esw[ek] = {
                                        "xdom_bm120_acc": round(float(np.mean((Dte @ ube) > 0)), 4),
                                        "xdom_bmneut_acc": round(float(np.mean((Dte @ une) > 0)), 4),
                                        "cv2_acc": round(float(np.mean(ca)), 4),
                                        "cv2_benign": round(float(np.mean(cb)), 4),
                                        "cv2_adverse": round(float(np.mean(cad)), 4),
                                    }
                                entry["rank_k_erasure"] = esw

                            # Exact DeltaNLL <= 0 strata for ALL readouts (including xdom_bmneut!)
                            if bname in dnll_whole and tk in dnll_whole[bname]:
                                entry["dnll_whole_strata"] = {
                                    "xdom_bm120_val5": E245._dnll_strata_eval(sc_xb, dnll_whole[bname][tk], sid_idx),
                                    "xdom_bmneut_val5": E245._dnll_strata_eval(sc_xn, dnll_whole[bname][tk], sid_idx),
                                    "cv2_val5_oof20": E245._dnll_strata_eval(oof_m, dnll_whole[bname][tk], sid_idx),
                                }
                            if bname in dnll_chg and tk in dnll_chg[bname]:
                                entry["dnll_changed_tokens_strata"] = {
                                    "xdom_bm120_val5": E245._dnll_strata_eval(sc_xb, dnll_chg[bname][tk], sid_idx),
                                    "xdom_bmneut_val5": E245._dnll_strata_eval(sc_xn, dnll_chg[bname][tk], sid_idx),
                                    "cv2_val5_oof20": E245._dnll_strata_eval(oof_m, dnll_chg[bname][tk], sid_idx),
                                }

                            tgt_out[tk] = entry

                        # Exact Holm families across 5 targets (saved explicitly!)
                        summ = {
                            "mean_zs_sent": round(float(np.mean([tgt_out[t]["zs_sent_pos_acc"] for t in tgts])), 4),
                            "mean_xdom_bm120_val5": round(float(np.mean([tgt_out[t]["xdom_bm120_val5"]["acc"] for t in tgts])), 4),
                            "mean_xdom_bm120_rand500_p95": round(float(np.mean([tgt_out[t]["xdom_bm120_val5"]["specificity_rand500"]["rand500_p95_two_sided"] for t in tgts])), 4),
                            "holm_xdom_bm120_setting_acc": _holm({t: tgt_out[t]["xdom_bm120_val5"]["p_mc_uncentered_acc"] for t in tgts}),
                            "holm_xdom_bm120_rand500_two": _holm({t: tgt_out[t]["xdom_bm120_val5"]["specificity_rand500"]["p_vs_rand500_two_sided"] for t in tgts}),
                            "mean_xdom_bmneut_val5": round(float(np.mean([tgt_out[t]["xdom_bmneut_val5"]["acc"] for t in tgts])), 4),
                            "mean_xdom_bmneut_rand500_p95": round(float(np.mean([tgt_out[t]["xdom_bmneut_val5"]["specificity_rand500"]["rand500_p95_two_sided"] for t in tgts])), 4),
                            "holm_xdom_bmneut_setting_acc": _holm({t: tgt_out[t]["xdom_bmneut_val5"]["p_mc_uncentered_acc"] for t in tgts}),
                            "holm_xdom_bmneut_rand500_two": _holm({t: tgt_out[t]["xdom_bmneut_val5"]["specificity_rand500"]["p_vs_rand500_two_sided"] for t in tgts}),
                            "mean_cv2_val5": round(float(np.mean([tgt_out[t]["cv2_val5"]["cv2_20split_acc"] for t in tgts])), 4),
                            "mean_cv2_val5_all_right": round(float(np.mean([tgt_out[t]["cv2_val5"]["cv2_20split_all_right"] for t in tgts])), 4),
                            "holm_cv2_20split_refit_acc": _holm({t: tgt_out[t]["cv2_val5"]["p_mc_20split_refit_acc"] for t in tgts}),
                            "holm_cv2_rand500_two": _holm({t: tgt_out[t]["cv2_val5"]["specificity_rand500"]["p_vs_rand500_two_sided"] for t in tgts}),
                        }
                        if bname == "bm_actor2x2_247":
                            summ["mean_actor_split"] = {
                                k_: round(float(np.mean([tgt_out[t]["actor_split_breakdown"][k_] for t in tgts])), 4)
                                for k_ in tgt_out[tgts[0]]["actor_split_breakdown"]
                            }
                            summ["erasure_5tgt_mean"] = {
                                ek: {
                                    "refit_sent_cv": refit_erasure_check[ek]["refit_sent_cv_mean"],
                                    "refit_house_cv": refit_erasure_check[ek]["refit_house_cv_mean"],
                                    "xdom_bm120_acc": round(float(np.mean([tgt_out[t]["rank_k_erasure"][ek]["xdom_bm120_acc"] for t in tgts])), 4),
                                    "xdom_bmneut_acc": round(float(np.mean([tgt_out[t]["rank_k_erasure"][ek]["xdom_bmneut_acc"] for t in tgts])), 4),
                                    "cv2_acc": round(float(np.mean([tgt_out[t]["rank_k_erasure"][ek]["cv2_acc"] for t in tgts])), 4),
                                    "cv2_benign": round(float(np.mean([tgt_out[t]["rank_k_erasure"][ek]["cv2_benign"] for t in tgts])), 4),
                                    "cv2_adverse": round(float(np.mean([tgt_out[t]["rank_k_erasure"][ek]["cv2_adverse"] for t in tgts])), 4),
                                }
                                for ek in ("raw", "val2", "val5", "val5_house", "eval_svd_12", "eval_svd_24", "eval_svd_40")
                            }
                        src_out[bname] = {"summary": summ, "targets": tgt_out}
                    pool_full["sources"][src_k] = src_out
                d50_full[pool_key] = pool_full

    # Refit-in-bootstrap paired depth contrasts for bm_actor2x2_247 (1,000 setting resamples refitting all 20 splits!)
    actor_depth_boot = {}
    for pool_key in ("last", "mean"):
        actor_depth_boot[pool_key] = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            Ds_d = {dk: Z_all_d[pool_key][dk][src_k] for dk in ("d30", "d50", "d70")}
            Dt_d = {dk: {tk: Z_all_d[pool_key][dk][tk] for tk in tgts} for dk in ("d30", "d50", "d70")}
            Qs_d = {dk: Q5_all_d[pool_key][dk][src_k] for dk in ("d30", "d50", "d70")}
            Qt_d = {dk: {tk: Q5_all_d[pool_key][dk][tk] for tk in tgts} for dk in ("d30", "d50", "d70")}
            actor_depth_boot[pool_key][src_k] = _refit_paired_depth_boot(
                Ds_d, Dt_d, Qs_d, Qt_d, sid_map["bm_actor2x2_247"], split_map["bm_actor2x2_247"], tgts, n_boot=1000, seed=2475
            )

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "manifest": manifest,
        "lexical_baselines_bm_actor2x2_247": lex_247,
        "nll_baselines_bm_actor2x2_247": nll_247,
        "actor2x2_5depth_curves": by_depth_actor,
        "actor2x2_refit_depth_bootstrap": actor_depth_boot,
        "d50_full_audit": d50_full,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} in {report['elapsed_sec']}s", flush=True)
    return report


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "merge":
        merge_and_verify()
    elif cmd == "extract":
        keys = tuple(sys.argv[2:]) if len(sys.argv) > 2 else MLX_KEYS
        extract_actor2x2(keys)
    elif cmd == "analyze":
        analyze_all()
    else:
        raise SystemExit(f"Unknown cmd {cmd}")


if __name__ == "__main__":
    main()
