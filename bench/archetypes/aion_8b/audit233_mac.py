"""Addendum 233 (Mac Studio): Fast, ruthless falsification checks answering every Astra + Fable scientific critique!

Computes & saves `bench/archetypes/fluent/plato/audit233_mac.json`:
  1. Sec 4.1 & 4.2.1:
     - Unified exact `plato_data.py` `bpe8k` eligibility screen (`20,339` train pts) + `Random(2288)` domain-stratified
       2-fold CV with tie-safe fractional retrieval (`n_ties`), strictly train-fold-only `r2_opt`, and 2,000-draw
       group-bootstrap 95% CIs (`C5 - best_single`, `C5 - C4`)!
     - Full vs. dedicated-myth-pool-excluded (`gut+proc+chat`) 7-company GPA & `items3` role breakdown
       (`anchor`, `lure`, `true`, `foil` + 1-to-1 word-matched pairs + 2,000-draw item bootstrap 95% CIs)!
  2. Sec 4.4.2 (`arch_rot229`):
     - Strictly fold-local length residualization (ZERO held-out leak!),
     - Strictly cross-construct `HOUSE_NO_SELF -> SELF2` (excluding `self` category from HOUSE!),
     - Deterministic `hashlib.md5` setting-disjoint folds PLUS 20 deterministic seeded random setting-disjoint splits
       with paired neural-vs-BoW spread, AND
     - Exact binary-unigram BoW lexical baseline on identical splits!
  3. Sec 4.6.1 (`myth_proto229`):
     - Fold-local beat centering + equal-family weighting + exact `within_kind_top1` (`Myth` / `Mech`), AND
     - Strictly leave-query-out vocabulary & IDF unigram TF-IDF lexical baseline on identical folds!

Note: The 2x2 crossed epistemic audit (Gemma 4 E4B + Qwen3.5-9B MLX) runs via `aion_8b/audit233_epistemic.py`.
"""
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO = ARCH / "fluent" / "plato"
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from fluent import arch_rot229, myth_proto229, plato_data

# Strictly ZERO entity overlap with TRUE_FALSE_20 (replaces France, Mars/Earth/planets, photosynthesis, Egypt/Nile!)
CLEAN_FALSE_PREMISE_12 = [
    ("What is the capital city of Italy?", "What is the capital city of Tuesday?"),
    ("What is the boiling point of ethanol?", "What is the boiling point of laughter?"),
    ("Which ocean borders the coast of Oregon?", "Which ocean borders the coast of Nebraska?"),
    ("Who wrote the famous play Hamlet?", "Who wrote the famous play about cloud mining?"),
    ("What metal is refined from bauxite ore?", "What metal is refined from quiet whispers?"),
    ("How many sides does a regular octagon have?", "How many sides does a regular circle have?"),
    ("What organ filters waste from human blood?", "What organ filters waste from wooden chairs?"),
    ("Which continent is the country Kenya located on?", "Which continent is the emotion anger located on?"),
    ("Who painted the famous mural The Last Supper?", "Who painted the famous mural of silence?"),
    ("What currency is officially used in Sweden?", "What currency is officially used on comets?"),
    ("Which mountain range runs along South America?", "Which mountain range runs along open vacuum?"),
    ("What mineral strengthens human tooth enamel?", "What mineral strengthens empty shadows?"),
]

CLEAN_TENSE_MATCHED_UNK_12 = [
    ("What is the color of the daytime sky?", "What is the color of the neighbor's shirt?"),
    ("What is the shape of the full moon?", "What is the shape of the hidden stone?"),
    ("What is the season after the autumn equinox?", "What is the message inside the sealed envelope?"),
    ("What is the direction of the morning sunrise?", "What is the direction of the parked bicycle?"),
    ("What is the capital of the United States?", "What is the passcode of the front entrance?"),
    ("What is the language of the United Kingdom?", "What is the password of the office network?"),
    ("What is the largest desert in northern Africa?", "What is the oldest folder inside the cabinet?"),
    ("What is the longest river in South America?", "What is the newest letter sitting on the desk?"),
    ("What is the tallest peak in the Himalayas?", "What is the exact weight of the brown parcel?"),
    ("What is the metal liquid at room heat?", "What is the object hidden under the table?"),
    ("What is the animal famous for its trunk?", "What is the book borrowed from the neighbor?"),
    ("What is the fruit grown on apple orchards?", "What is the dish cooked for tonight's dinner?"),
]


def _frac_retrieval(Z_pred, Z_tgt, eligible_mask):
    """Tie-safe fractional Top-1 and Top-5 retrieval + exact tie counter."""
    P = Z_pred[eligible_mask]
    T = Z_tgt[eligible_mask]
    Pn = P / np.maximum(1e-8, np.linalg.norm(P, axis=1, keepdims=True))
    Tn = T / np.maximum(1e-8, np.linalg.norm(T, axis=1, keepdims=True))
    S = Pn @ Tn.T  # (M, M)
    diag = np.diag(S)[:, None]
    above = (S > diag + 1e-12).sum(axis=1)
    tied = (np.abs(S - diag) <= 1e-12).sum(axis=1)  # includes self (>= 1)
    # Fractional credit for Top-1 and Top-5
    top1 = np.where(above == 0, 1.0 / tied, 0.0)
    # Top-k: slots remaining after strictly-above items
    rem5 = np.maximum(0, 5 - above)
    top5 = np.minimum(1.0, rem5 / np.maximum(1, tied))
    n_ties = int((tied > 1).sum())
    return float(top1.mean()), float(top5.mean()), n_ties, int(len(diag))


def _fit_gpa_and_eval_blind(neutral_only=False):
    """Run exact 10-iter GPA (C4 & C5) using either FULL train pool (`20,339` pts matching `plato_data.build_consensus`!)
    or dedicated-myth-pool-excluded (`gut+proc+chat`) neutral pool, then evaluate blind transfer to Phi-4 & OLMo 3
    on the EXACT same `Random(2288)` domain-stratified 2-fold CV split as `plato_data.eval_procrustes_transfer`,
    with train-fold `r2_opt`, tie-safe retrieval, `by_domain_cos`, and `items3` role breakdown + 2,000-draw CIs!"""
    tk = plato_data.load_bpe8k()
    bpe_enc = lambda s: tk.encode(s).ids
    tr_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    ev_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl")]

    raws = {k: np.load(PLATO / f"raw_{k}.npz") for k in plato_data.ALL_KEYS}
    # Exact `plato_data.build_consensus` alignment: intersect ALL 7 models + bpe8k boundary & <=512 screen!
    tr_ok_all = np.zeros(len(tr_rows), dtype=bool)
    for r_i, r in enumerate(tr_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = plato_data.boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raws[k]["tr_ok"][r_i]) and int(raws[k]["tr_cnt"][r_i]) == n_seg for k in plato_data.ALL_KEYS)
        )
        if neutral_only:
            ok = ok and (r["domain"] in ("gut", "proc", "chat"))
        tr_ok_all[r_i] = ok

    ev_ok_all = np.zeros(len(ev_rows), dtype=bool)
    for r_i, r in enumerate(ev_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = plato_data.boundary_token_indices(bpe_enc, r["segs"])
        ev_ok_all[r_i] = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raws[k]["ev_ok"][r_i]) and int(raws[k]["ev_cnt"][r_i]) == n_seg for k in plato_data.ALL_KEYS)
        )

    def slice_pts(raw, ok_mask, split):
        cnts = raw[f"{split}_cnt"].astype(int)
        pts = raw[f"{split}_pts"].astype(np.float32)
        out_p = []
        off = 0
        for i, c in enumerate(cnts):
            if ok_mask[i] and c > 0:
                out_p.append(pts[off:off + c])
            off += c
        return np.concatenate(out_p, axis=0)

    ev_doms, ev_groups, ev_roles, ev_kinds, ev_words, ev_elig = [], [], [], [], [], []
    for i, (r, c) in enumerate(zip(ev_rows, raws["gemma4_e4b"]["ev_cnt"])):
        if not ev_ok_all[i]:
            continue
        c = int(c)
        wc = len(r["text"].split())
        ev_doms.extend([r["domain"]] * c)
        ev_groups.extend([r["group_id"]] * c)
        ev_roles.extend([r.get("role", "none")] * c)
        ev_kinds.extend([r.get("kind", "none")] * c)
        ev_words.extend([wc] * c)
        ev_elig.extend([r.get("role", "none") != "foil"] * c)
    ev_doms = np.array(ev_doms)
    ev_groups = np.array(ev_groups)
    ev_roles = np.array(ev_roles)
    ev_kinds = np.array(ev_kinds)
    ev_words = np.array(ev_words)
    ev_elig = np.array(ev_elig, dtype=bool)

    # Fit 256-dim PCA strictly on tr_ok_all for every model
    Z_tr, Z_ev, var_exp = {}, {}, {}
    n_tr_pts = 0
    for k in plato_data.ALL_KEYS:
        Xtr = slice_pts(raws[k], tr_ok_all, "tr")
        Xev = slice_pts(raws[k], ev_ok_all, "ev")
        n_tr_pts = len(Xtr)
        mu = Xtr.mean(axis=0, keepdims=True)
        Xc = Xtr - mu
        cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        W = evecs[:, idx[:256]].astype(np.float32)
        var_exp[k] = float(evals[idx[:256]].sum() / max(1e-12, evals.sum()))
        P_tr = (Xtr - mu) @ W
        rms = float(np.sqrt(np.mean(P_tr ** 2))) + 1e-8
        Z_tr[k] = P_tr / rms
        Z_ev[k] = ((Xev - mu) @ W) / rms

    # 10-iteration GPA on consensus teachers using plato_data.generalized_procrustes_consensus
    _, C4_ev, _, _, _ = plato_data.generalized_procrustes_consensus([Z_tr[k] for k in plato_data.MLX_CONSENSUS_KEYS], [Z_ev[k] for k in plato_data.MLX_CONSENSUS_KEYS], n_iters=10)
    _, C5_ev, _, _, _ = plato_data.generalized_procrustes_consensus([Z_tr[k] for k in plato_data.CONSENSUS_KEYS], [Z_ev[k] for k in plato_data.CONSENSUS_KEYS], n_iters=10)
    Z_ev["C4"] = C4_ev.astype(np.float32)
    Z_ev["C5"] = C5_ev.astype(np.float32)

    # Exact domain-stratified, group-aware 2-fold CV matching `plato_data.eval_procrustes_transfer` (`Random(2288)`)!
    folds = np.zeros(len(ev_groups), dtype=int)
    by_dom_g = defaultdict(list)
    seen_g = set()
    for d, g in zip(ev_doms, ev_groups):
        if g not in seen_g:
            seen_g.add(g)
            by_dom_g[d].append(g)
    g2fold = {}
    fold_rng = random.Random(2288)
    for d in sorted(by_dom_g):
        gs = list(by_dom_g[d])
        fold_rng.shuffle(gs)
        half = len(gs) // 2
        for i, g in enumerate(gs):
            g2fold[g] = 0 if i < half else 1
    for i, g in enumerate(ev_groups):
        folds[i] = g2fold[g]

    def eval_pair(src_ev, tgt_ev):
        preds = np.zeros_like(tgt_ev)
        preds_opt = np.zeros_like(tgt_ev)
        t1s, t5s, ties = [], [], 0
        for f in (0, 1):
            tr_m = folds != f
            te_m = folds == f
            params = plato_data.fit_procrustes_map(src_ev[tr_m], tgt_ev[tr_m])
            p_tr = plato_data.apply_procrustes_map(src_ev[tr_m], params)
            p_te = plato_data.apply_procrustes_map(src_ev[te_m], params)
            preds[te_m] = p_te
            # Strictly train-fold optimal scalar (NO test leak!)
            ym = tgt_ev[tr_m].mean(axis=0, keepdims=True)
            pm = p_tr.mean(axis=0, keepdims=True)
            num = float(np.sum((p_tr - pm) * (tgt_ev[tr_m] - ym)))
            den = float(np.sum((p_tr - pm) ** 2)) + 1e-12
            alpha = num / den
            preds_opt[te_m] = ym + alpha * (p_te - pm)
            t1, t5, nt, _ = _frac_retrieval(p_te, tgt_ev[te_m], ev_elig[te_m])
            t1s.append(t1); t5s.append(t5); ties += nt

        cos_vec = np.sum(preds * tgt_ev, axis=1) / np.maximum(1e-8, np.linalg.norm(preds, axis=1) * np.linalg.norm(tgt_ev, axis=1))
        tc = tgt_ev - tgt_ev.mean(axis=0, keepdims=True)
        pc = preds - preds.mean(axis=0, keepdims=True)
        cos_cen = np.sum(pc * tc, axis=1) / np.maximum(1e-8, np.linalg.norm(pc, axis=1) * np.linalg.norm(tc, axis=1))
        sst = float(np.sum((tgt_ev - tgt_ev.mean(axis=0, keepdims=True)) ** 2)) + 1e-12
        r2_frob = 1.0 - float(np.sum((tgt_ev - preds) ** 2)) / sst
        r2_opt_oof = 1.0 - float(np.sum((tgt_ev - preds_opt) ** 2)) / sst
        by_dom = {str(d): round(float(cos_vec[ev_doms == d].mean()), 4) for d in sorted(set(ev_doms))}
        return {
            "cos": round(float(cos_vec.mean()), 4),
            "r2_frob": round(r2_frob, 4),
            "r2_opt_train_scalar": round(r2_opt_oof, 4),
            "top1_tiesafe": round(float(np.mean(t1s)), 4),
            "top5_tiesafe": round(float(np.mean(t5s)), 4),
            "n_ties": ties,
            "by_domain_cos": by_dom,
            "cos_vec": cos_vec,
            "cos_cen": cos_cen,
        }

    out = {"neutral_only": neutral_only, "n_train_pts": int(n_tr_pts), "var_exp_256": {k: round(v, 4) for k, v in var_exp.items()}, "blind": {}}
    rng = np.random.default_rng(233)
    for bkey in plato_data.BLIND_KEYS:
        res_b = {}
        vecs = {}
        for skey in ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "inkling2_mini", "C4", "C5"):
            r = eval_pair(Z_ev[skey], Z_ev[bkey])
            vecs[skey] = (r.pop("cos_vec"), r.pop("cos_cen"))
            res_b[skey] = r
        # Group-bootstrap 95% CIs (2,000 draws over unique group_ids) for C5 - best_single and C5 - C4
        best_s = max(("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b"), key=lambda k: res_b[k]["cos"])
        uniq_g = np.unique(ev_groups)
        g_idx = {g: np.where(ev_groups == g)[0] for g in uniq_g}
        d_bs = vecs["C5"][0] - vecs[best_s][0]
        d_c4 = vecs["C5"][0] - vecs["C4"][0]
        g_d_bs = np.array([d_bs[g_idx[g]].mean() for g in uniq_g])
        g_d_c4 = np.array([d_c4[g_idx[g]].mean() for g in uniq_g])
        g_w = np.array([len(g_idx[g]) for g in uniq_g], dtype=float)
        bs_draws, c4_draws = [], []
        for _ in range(2000):
            pick = rng.integers(0, len(uniq_g), len(uniq_g))
            w = g_w[pick]
            bs_draws.append(float(np.sum(g_d_bs[pick] * w) / np.sum(w)))
            c4_draws.append(float(np.sum(g_d_c4[pick] * w) / np.sum(w)))
        res_b["bootstrap_C5_minus_best_single"] = {
            "best_single": best_s,
            "diff_clause_weighted": round(float(d_bs.mean()), 4),
            "ci95": [round(float(np.percentile(bs_draws, 2.5)), 4), round(float(np.percentile(bs_draws, 97.5)), 4)],
        }
        res_b["bootstrap_C5_minus_C4"] = {
            "diff_clause_weighted": round(float(d_c4.mean()), 4),
            "ci95": [round(float(np.percentile(c4_draws, 2.5)), 4), round(float(np.percentile(c4_draws, 97.5)), 4)],
        }

        # items3 role breakdown on C5 centered cosine (with 2,000-draw item bootstrap 95% CIs + word-matched anchor!)
        cc = vecs["C5"][1]
        role_res = {}
        for role in ("anchor", "lure", "true", "foil"):
            m_my = (ev_doms == "items3_myth") & (ev_roles == role)
            m_me = (ev_doms == "items3_mech") & (ev_roles == role)
            # Item-level means (100 myths, 100 mechs)
            my_gs = np.unique(ev_groups[m_my])
            me_gs = np.unique(ev_groups[m_me])
            my_vals = np.array([cc[m_my & (ev_groups == g)].mean() for g in my_gs])
            me_vals = np.array([cc[m_me & (ev_groups == g)].mean() for g in me_gs])
            draws = [float(rng.choice(my_vals, len(my_vals), True).mean() - rng.choice(me_vals, len(me_vals), True).mean()) for _ in range(2000)]
            entry = {
                "myth": round(float(my_vals.mean()), 4),
                "mech": round(float(me_vals.mean()), 4),
                "diff": round(float(my_vals.mean() - me_vals.mean()), 4),
                "ci95": [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)],
            }
            if role == "anchor":
                # Strict 1-to-1 word-length matching (|dw| <= 2)
                my_w = [int(ev_words[m_my & (ev_groups == g)][0]) for g in my_gs]
                me_w = [int(ev_words[m_me & (ev_groups == g)][0]) for g in me_gs]
                used_me = set()
                p_diffs = []
                for i, w1 in enumerate(my_w):
                    cands = [j for j, w2 in enumerate(me_w) if j not in used_me and abs(w1 - w2) <= 2]
                    if cands:
                        best_j = min(cands, key=lambda j: abs(w1 - me_w[j]))
                        used_me.add(best_j)
                        p_diffs.append(float(my_vals[i] - me_vals[best_j]))
                p_arr = np.array(p_diffs)
                pd_draws = [float(rng.choice(p_arr, len(p_arr), True).mean()) for _ in range(2000)]
                entry["word_matched_n"] = len(p_arr)
                entry["word_matched_diff"] = round(float(p_arr.mean()), 4)
                entry["word_matched_ci95"] = [round(float(np.percentile(pd_draws, 2.5)), 4), round(float(np.percentile(pd_draws, 97.5)), 4)]
            role_res[role] = entry
        res_b["items3_roles_C5_centered"] = role_res
        out["blind"][bkey] = res_b
    return out


def _audit_arch_rot():
    """Audit HOUSE/SELF2 with: (1) fold-local length residualization, (2) HOUSE_NO_SELF -> SELF2,
    (3) setting-disjoint folds, and (4) binary-unigram BoW lexical baseline!"""
    pairs = arch_rot229.load_arch_pairs()
    house_pairs = [p for p in pairs if p["cat"] in arch_rot229.HOUSE_CATS]
    house_noself = [p for p in pairs if p["cat"] in arch_rot229.HOUSE_CATS and p["cat"] != "self"]
    self2_pairs = [p for p in pairs if p["cat"] == "self2"]

    # 1. Binary unigram BoW baseline (Astra's exact test + setting-disjoint folds!)
    tok = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    vocab = sorted({w for p in pairs for w in tok(p["done"]) | tok(p["neglected"])})
    v2i = {w: i for i, w in enumerate(vocab)}
    def bow_diff(plist):
        X = np.zeros((len(plist), len(vocab)), dtype=np.float32)
        for idx, p in enumerate(plist):
            for w in tok(p["done"]): X[idx, v2i[w]] += 1.0
            for w in tok(p["neglected"]): X[idx, v2i[w]] -= 1.0
        return X

    X_h = bow_diff(house_pairs)
    cats_h = np.array([p["cat"] for p in house_pairs])
    X_hns = bow_diff(house_noself)
    cats_hns = np.array([p["cat"] for p in house_noself])
    X_s2 = bow_diff(self2_pairs)

    def eval_bow_2f(X, cats, folds):
        hits = []
        for f in (0, 1):
            tr = folds != f; te = folds == f
            d = arch_rot229.house_dir(X[tr], cats[tr])
            hits.extend((X[te] @ d > 0).astype(float).tolist())
        return round(float(np.mean(hits)), 4)

    def cat_mean_dir(X, cats, cat_set):
        vecs = []
        for c in sorted(cat_set):
            m = cats == c
            if int(m.sum()) >= 1:
                vecs.append(arch_rot229.unit(X[m].mean(axis=0)))
        return arch_rot229.unit(np.stack(vecs).mean(axis=0))

    NOSELF_CATS = tuple(c for c in arch_rot229.HOUSE_CATS if c != "self")
    alt_folds = np.array([i % 2 for i in range(len(house_pairs))])
    # Strictly deterministic cross-process setting-disjoint folds via MD5 digest (NEVER Python salted hash()!)
    settings_h = [p.get("setting", str(i)) for i, p in enumerate(house_pairs)]
    set_folds = np.array([hashlib.md5(s.encode("utf-8")).digest()[0] % 2 for s in settings_h])

    # Also build 20 deterministic seeded setting-disjoint 2-fold splits ensuring every HOUSE category exists in both folds!
    uniq_settings = sorted(set(settings_h))
    split_rng = np.random.default_rng(2330)
    setting_splits_20 = []
    attempts = 0
    while len(setting_splits_20) < 20 and attempts < 2000:
        attempts += 1
        perm = split_rng.permutation(len(uniq_settings))
        s2f = {uniq_settings[idx]: int(k % 2) for k, idx in enumerate(perm)}
        cand = np.array([s2f[s] for s in settings_h], dtype=int)
        # Require every HOUSE category to have >= 1 story in BOTH folds
        if all((cand[cats_h == c] == 0).any() and (cand[cats_h == c] == 1).any() for c in arch_rot229.HOUSE_CATS):
            setting_splits_20.append(cand)
    assert len(setting_splits_20) == 20, f"Only generated {len(setting_splits_20)} valid setting-disjoint splits"

    bow_20 = np.array([eval_bow_2f(X_h, cats_h, sf) for sf in setting_splits_20], dtype=float)
    d_bow_all = cat_mean_dir(X_h, cats_h, arch_rot229.HOUSE_CATS)
    d_bow_noself = cat_mean_dir(X_hns, cats_hns, NOSELF_CATS)
    bow_res = {
        "house_2f_alt_folds": eval_bow_2f(X_h, cats_h, alt_folds),
        "house_2f_setting_disjoint_md5": eval_bow_2f(X_h, cats_h, set_folds),
        "house_2f_setting_disjoint_20splits_mean": round(float(bow_20.mean()), 4),
        "house_2f_setting_disjoint_20splits_std": round(float(bow_20.std(ddof=1)), 4),
        "house_2f_setting_disjoint_20splits_min": round(float(bow_20.min()), 4),
        "house_2f_setting_disjoint_20splits_max": round(float(bow_20.max()), 4),
        "house_to_self2": round(float((X_s2 @ d_bow_all > 0).mean()), 4),
        "house_noself_to_self2": round(float((X_s2 @ d_bow_noself > 0).mean()), 4),
        "perfection_acc": round(float(np.mean([
            (X_h[i] @ arch_rot229.house_dir(X_h[alt_folds != alt_folds[i]], cats_h[alt_folds != alt_folds[i]]) > 0)
            for i, p in enumerate(house_pairs) if p["cat"] == "perfection"
        ])), 4),
    }

    # 2. Neural Gemma 4 -> 6 companies with STRICTLY FOLD-LOCAL length residualization & HOUSE_NO_SELF -> SELF2!
    # Load cached arch229_{k}.npz and fit neutral-only story rotations (intersecting tr_ok across all 7!)
    tr_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    raws = {k: np.load(PLATO / f"raw_{k}.npz") for k in plato_data.ALL_KEYS}
    archs = {k: np.load(PLATO / f"arch229_{k}.npz") for k in plato_data.ALL_KEYS}
    neut_mask = np.array([r["domain"] in ("gut", "proc", "chat") for r in tr_rows], dtype=bool)
    for k in plato_data.ALL_KEYS:
        neut_mask &= raws[k]["tr_ok"].astype(bool)

    # Project each model's story states into 256-dim neutral-only PCA
    Z_neut, D_all, L_done, L_neg = {}, {}, {}, {}
    for k in plato_data.ALL_KEYS:
        Xtr = raws[k]["tr_story"].astype(np.float32)[neut_mask]
        mu = Xtr.mean(axis=0, keepdims=True)
        Xc = Xtr - mu
        cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
        evals, evecs = np.linalg.eigh(cov)
        W = evecs[:, np.argsort(evals)[::-1][:256]].astype(np.float32)
        P_tr = (Xtr - mu) @ W
        rms = float(np.sqrt(np.mean(P_tr ** 2))) + 1e-8
        Z_neut[k] = P_tr / rms
        D_done = ((archs[k]["done_mean"].astype(np.float32) - mu) @ W) / rms
        D_neg = ((archs[k]["neg_mean"].astype(np.float32) - mu) @ W) / rms
        D_all[k] = D_done - D_neg
        L_done[k] = archs[k]["done_len"].astype(np.float32)
        L_neg[k] = archs[k]["neg_len"].astype(np.float32)

    idx_h = np.array([i for i, p in enumerate(pairs) if p["cat"] in arch_rot229.HOUSE_CATS])
    idx_hns = np.array([i for i, p in enumerate(pairs) if p["cat"] in arch_rot229.HOUSE_CATS and p["cat"] != "self"])
    idx_s2 = np.array([i for i, p in enumerate(pairs) if p["cat"] == "self2"])

    def fold_local_lenres(D_src, ld_s, ln_s, D_tgt, ld_t, ln_t, cats, folds):
        """Strictly fit length OLS ONLY on train fold, subtract length component (preserving equal-length intercept) on test fold!"""
        hits = []
        for f in (0, 1):
            tr = folds != f; te = folds == f
            def fit_apply_lr(D, ld, ln):
                dlen = (ld - ln)[:, None]
                dlgl = (np.log1p(ld) - np.log1p(ln))[:, None]
                X = np.concatenate([np.ones((len(D), 1)), dlen, dlgl], axis=1)
                B, *_ = np.linalg.lstsq(X[tr], D[tr], rcond=None)
                # Remove ONLY slope terms (dlen, dlgl) using train-fit B[1:]!
                D_tr_lr = D[tr] - X[tr, 1:] @ B[1:]
                D_te_lr = D[te] - X[te, 1:] @ B[1:]
                return D_tr_lr, D_te_lr
            Ds_tr, _ = fit_apply_lr(D_src, ld_s, ln_s)
            _, Dt_te = fit_apply_lr(D_tgt, ld_t, ln_t)
            u = arch_rot229.house_dir(Ds_tr, cats[tr])
            hits.extend((Dt_te @ u > 0).astype(float).tolist())
        return round(float(np.mean(hits)), 4)

    neural_res = {}
    for tgt_k in ("qwen35_9b", "llama31_8b", "mistral7b", "inkling2_mini", "phi4_mini", "olmo3_7b"):
        _, _, R, s = plato_data.fit_procrustes_map(Z_neut["gemma4_e4b"], Z_neut[tgt_k])
        # Linear rotation of difference vectors (translation cancels in done - neg!)
        D_rot_h = (D_all["gemma4_e4b"][idx_h] @ R.T) * s
        D_tgt_h = D_all[tgt_k][idx_h]
        D_rot_hns = (D_all["gemma4_e4b"][idx_hns] @ R.T) * s
        D_tgt_s2 = D_all[tgt_k][idx_s2]

        def _eval_neural_sd(folds_vec):
            hits = []
            for f in (0, 1):
                tr = folds_vec != f; te = folds_vec == f
                u = arch_rot229.house_dir(D_rot_h[tr], cats_h[tr])
                hits.extend((D_tgt_h[te] @ u > 0).astype(float).tolist())
            return float(np.mean(hits))

        sd_md5 = _eval_neural_sd(set_folds)
        sd_20 = np.array([_eval_neural_sd(sf) for sf in setting_splits_20], dtype=float)
        diff_vs_bow = sd_20 - bow_20
        u_hns = cat_mean_dir(D_rot_hns, cats_hns, NOSELF_CATS)
        neural_res[tgt_k] = {
            "house_2f_lenres_fold_local": fold_local_lenres(
                D_rot_h, L_done["gemma4_e4b"][idx_h], L_neg["gemma4_e4b"][idx_h],
                D_tgt_h, L_done[tgt_k][idx_h], L_neg[tgt_k][idx_h], cats_h, alt_folds,
            ),
            "house_2f_setting_disjoint_md5": round(sd_md5, 4),
            "house_2f_setting_disjoint_20splits_mean": round(float(sd_20.mean()), 4),
            "house_2f_setting_disjoint_20splits_std": round(float(sd_20.std(ddof=1)), 4),
            "house_2f_setting_disjoint_20splits_min": round(float(sd_20.min()), 4),
            "house_2f_setting_disjoint_20splits_max": round(float(sd_20.max()), 4),
            "wins_over_bow_20splits": int(np.sum(sd_20 > bow_20)),
            "mean_diff_vs_bow_20splits": round(float(diff_vs_bow.mean()), 4),
            "house_noself_to_self2": round(float((D_tgt_s2 @ u_hns > 0).mean()), 4),
        }

    return {"bow_lexical_baseline": bow_res, "neural_gemma_to_6_strict": neural_res}


def _audit_myth_loro():
    """Exact fold-local centering + equal-family weighting + within-kind Top-1 + unigram TF-IDF baseline on 975 stories!"""
    t_npz = np.load(PLATO / "targets_myth229.npz", allow_pickle=True)
    story_ids = [str(x) for x in t_npz["story_id"]]
    rows_by_id = {}
    with open(PLATO / "myth229_raw.jsonl") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                rows_by_id[r["id"]] = r
    keep = [rows_by_id[sid] for sid in story_ids]
    assert len(keep) == 975
    fams = np.array([r["fam_id"] for r in keep])
    kinds = np.array([r["kind"] for r in keep])
    fam_counts = Counter(fams)

    c5_pts = t_npz["c5_instance_256"].astype(np.float32)  # (975*4, 256)
    assert len(c5_pts) == 975 * 4
    c5_by_beat = c5_pts.reshape(975, 4, 256)

    # Load blind Phi-4 & OLMo 3 myth229 raw states and verify identical story ordering
    phi_d = np.load(PLATO / "myth229_raw_phi4_mini.npz", allow_pickle=True)
    olm_d = np.load(PLATO / "myth229_raw_olmo3_7b.npz", allow_pickle=True)
    assert [str(x) for x in phi_d["ids"]] == story_ids and [str(x) for x in olm_d["ids"]] == story_ids
    phi_raw = phi_d["pts"].astype(np.float32).reshape(975, 4, -1)
    olm_raw = olm_d["pts"].astype(np.float32).reshape(975, 4, -1)

    def eval_loro_strict(V_beat, equal_fam=True):
        """Strictly leave-query-out centering (mean over gallery excluding query) + fractional tie credit + within-kind Top-1."""
        N = len(V_beat)
        fam_to_idx = defaultdict(list)
        for i, f in enumerate(fams):
            fam_to_idx[f].append(i)
        fam_list = sorted(fam_to_idx)
        f2pos = {f: i for i, f in enumerate(fam_list)}
        t1_all, t5_all = [], []
        t1_by_fam = defaultdict(list)
        t1_myth, t1_mech = [], []
        for q in range(N):
            qf = fams[q]
            qk = kinds[q]
            # Strictly exclude query q from beat mean!
            mask_gal = np.ones(N, dtype=bool); mask_gal[q] = False
            mu = V_beat[mask_gal].mean(axis=0, keepdims=True)
            Vc = V_beat - mu
            qv = Vc[q]
            qn = float(np.linalg.norm(qv)) + 1e-8
            qv = qv / qn
            # Build 199 family prototypes strictly excluding q
            P = np.zeros((len(fam_list), Vc.shape[1]), dtype=np.float32)
            for fi, f in enumerate(fam_list):
                idxs = [i for i in fam_to_idx[f] if i != q]
                p = Vc[idxs].mean(axis=0)
                P[fi] = p / (float(np.linalg.norm(p)) + 1e-8)
            sims = P @ qv
            own = f2pos[qf]
            above = int((sims > sims[own] + 1e-12).sum())
            tied = int((np.abs(sims - sims[own]) <= 1e-12).sum())
            c1 = (1.0 / tied) if above == 0 else 0.0
            c5 = min(1.0, max(0, 5 - above) / max(1, tied))
            t1_all.append(c1); t5_all.append(c5)
            t1_by_fam[qf].append(c1)
            # Within-kind (100 myth or 99 mech)
            kind_mask = np.array([kinds[fam_to_idx[f][0]] == qk for f in fam_list])
            ksims = sims[kind_mask]
            own_s = sims[own]
            ka = int((ksims > own_s + 1e-12).sum())
            kt = int((np.abs(ksims - own_s) <= 1e-12).sum())
            kc1 = (1.0 / kt) if ka == 0 else 0.0
            if qk == "myth": t1_myth.append(kc1)
            else: t1_mech.append(kc1)
        return {
            "top1_story_avg": round(float(np.mean(t1_all)), 4),
            "top1_equal_fam": round(float(np.mean([np.mean(v) for v in t1_by_fam.values()])), 4),
            "top5_story_avg": round(float(np.mean(t5_all)), 4),
            "within_kind_myth_top1": round(float(np.mean(t1_myth)), 4),
            "within_kind_mech_top1": round(float(np.mean(t1_mech)), 4),
        }

    # Unigram TF-IDF lexical baseline (cumulative prefix up to beat b, STRICTLY leave-query-out vocab & IDF!)
    def eval_tfidf_strict_lqo(b_idx):
        texts = ["".join(r["segs"][:b_idx + 1]) for r in keep]
        toks = [re.findall(r"[a-z]+", t.lower()) for t in texts]
        vocab = sorted({w for d in toks for w in d})
        v2i = {w: i for i, w in enumerate(vocab)}
        N = len(toks)
        V = len(vocab)
        # Precompute sublinear TF matrix and document-presence matrix (exact O(1) leave-query-out DF subtraction!)
        TF = np.zeros((N, V), dtype=np.float32)
        PRES = np.zeros((N, V), dtype=np.float32)
        for i, d in enumerate(toks):
            c = Counter(d)
            for w, cnt in c.items():
                j = v2i[w]
                TF[i, j] = 1.0 + math.log(cnt)
                PRES[i, j] = 1.0
        df_all = PRES.sum(axis=0)  # (V,)

        fam_to_idx = defaultdict(list)
        for i, f in enumerate(fams):
            fam_to_idx[f].append(i)
        fam_list = sorted(fam_to_idx)
        f2pos = {f: i for i, f in enumerate(fam_list)}
        tf_t1, tf_t5 = [], []
        # Precompute family raw TF sums so per-query prototype mean is O(1) subtract + multiply by idf_gal!
        fam_tf_sum = np.stack([TF[fam_to_idx[f]].sum(axis=0) for f in fam_list], axis=0)  # (F, V)
        fam_n = np.array([len(fam_to_idx[f]) for f in fam_list], dtype=np.float32)[:, None]

        for q in range(N):
            # Fit vocabulary & IDF strictly on the N-1 gallery stories (zero query leakage!)
            df_gal = df_all - PRES[q]
            in_gal_vocab = df_gal > 0.0
            idf_gal = np.where(in_gal_vocab, np.log((N - 1.0 + 1.0) / (df_gal + 1.0)) + 1.0, 0.0).astype(np.float32)

            qv_raw = TF[q] * idf_gal
            qv = qv_raw / (float(np.linalg.norm(qv_raw)) + 1e-8)

            own = f2pos[fams[q]]
            # Mean TF per family excluding query q, then scale by gallery IDF!
            P_tf = fam_tf_sum / fam_n
            P_tf[own] = (fam_tf_sum[own] - TF[q]) / max(1.0, float(fam_n[own, 0] - 1.0))
            P = P_tf * idf_gal[None, :]
            Pn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
            P = P / Pn

            sims = P @ qv
            ab = int((sims > sims[own] + 1e-12).sum())
            td = int((np.abs(sims - sims[own]) <= 1e-12).sum())
            tf_t1.append((1.0 / td) if ab == 0 else 0.0)
            tf_t5.append(min(1.0, max(0, 5 - ab) / max(1, td)))
        return {
            "top1_story_avg": round(float(np.mean(tf_t1)), 4),
            "top5_story_avg": round(float(np.mean(tf_t5)), 4),
            "idf_mode": "strict_leave_query_out_vocab_and_idf",
        }

    out = {"family_size_dist": dict(Counter(fam_counts.values())), "beats": {}}
    for b in range(4):
        out["beats"][f"beat_{b}"] = {
            "c5_strict_loro": eval_loro_strict(c5_by_beat[:, b, :]),
            "phi4_strict_loro": eval_loro_strict(phi_raw[:, b, :]),
            "olmo3_strict_loro": eval_loro_strict(olm_raw[:, b, :]),
            "tfidf_unigram_baseline": eval_tfidf_strict_lqo(b),
        }
    return out


def main():
    print("Running Addendum 233 Mac Audit (Neutral-Only GPA + ArchRot Strict + Myth TF-IDF)...", flush=True)
    res = {
        "plato_full_tiesafe": _fit_gpa_and_eval_blind(neutral_only=False),
        "plato_neutral_only_no_myth": _fit_gpa_and_eval_blind(neutral_only=True),
        "arch_rot_strict_and_bow": _audit_arch_rot(),
        "myth_loro_strict_and_tfidf": _audit_myth_loro(),
    }
    out_p = PLATO / "audit233_mac.json"
    out_p.write_text(json.dumps(res, indent=1))
    print("AUDIT233_MAC_DONE ->", out_p, flush=True)
    # Print executive highlights
    for mode in ("plato_full_tiesafe", "plato_neutral_only_no_myth"):
        print(f"=== {mode} (n_tr={res[mode]['n_train_pts']}) ===")
        for bk in ("phi4_mini", "olmo3_7b"):
            b = res[mode]["blind"][bk]
            print(f"  {bk}: C5 cos={b['C5']['cos']} r2f={b['C5']['r2_frob']} r2o_tr={b['C5']['r2_opt_train_scalar']} top1={b['C5']['top1_tiesafe']}(ties={b['C5']['n_ties']})")
            print(f"    items3 roles: anchor={b['items3_roles_C5_centered']['anchor']} lure={b['items3_roles_C5_centered']['lure']['diff']} true={b['items3_roles_C5_centered']['true']['diff']} foil={b['items3_roles_C5_centered']['foil']['diff']}")
    print("=== arch_rot_strict_and_bow ===", json.dumps(res["arch_rot_strict_and_bow"], indent=1))
    print("=== myth_loro_strict_and_tfidf ===", json.dumps(res["myth_loro_strict_and_tfidf"], indent=1))


if __name__ == "__main__":
    main()
