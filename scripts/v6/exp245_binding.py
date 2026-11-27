"""Experiment 245 (FROZEN PRE-SPECIFIED PROTOCOL): Valence-Neutral & Polarity-Conflict
Mirrored Identical-BoW Compositional Binding Controls + Rank-k Valence Erasure.

THIS SCRIPT IS WRITTEN AND SHA-256 FROZEN PRIOR TO AUTHORING ANY ITEM OF
`bm_neut245_data.json` OR `bm_polconf245_data.json`.

Pre-specified dataset contracts:
  1. `bm_neut245_data.json` (Valence-Neutral Physical/Functional/Spatial/Type Binding):
     - 120 mirrored pairs = 60 settings x 2 clause orders (`orig` and `mirror`), exact 60/60 `fav_pos`.
     - Exact identical bag-of-words on all 120/120 pairs (`Counter(words(pos)) == Counter(words(neg))` & `pos != neg`).
     - ZERO morally good/bad verbs or adjectives (pure physical, categorical, spatial, recipient, tool-material fit).
     - Varied syntactic frames (active 2-clause, subordinate, passive, two-sentence).
     - Zero setting overlap with `bm120` or `bm_rel242`.
  2. `bm_polconf245_data.json` (Moral Polarity-Conflict / Adverse-Actor Binding):
     - 120 mirrored pairs = 60 settings x 2 clause orders (`orig` and `mirror`), exact 60/60 `fav_pos`.
     - Exact identical bag-of-words on all 120/120 pairs.
     - Every setting features an explicitly adverse/corrupt/hostile actor whose globally role-coherent
       action (`pos`) pairs a favorable verb with a bad/corrupt recipient AND an adverse verb with a
       good/honest recipient (`sign(val(verb)*val(adj)) == -1`), whereas the swapped sentence (`neg`)
       has locally desirable moral congruence (`sign(val(verb)*val(adj)) == +1`).
     - Zero setting overlap with `bm120`, `bm_rel242`, or `bm_neut245`.

Single Pre-Specified Primary Cell & Hypotheses:
  - Primary cell: `d50` (50% depth), `last` token, `gemma4_e4b` -> 5 rival MLX labs
    (co-primary replication: `qwen35_9b` -> 5 rival MLX labs; `mean` pool + all 5 depths reported as secondary).
  - Test A (`bm_neut245`): Does compositional binding survive when moral/sentiment cues vanish?
    * Manipulation check: zero-shot `zs_sent` <= 0.60 AND lexicon polarity <= 0.60.
    * Transfer: frozen `bm120` direction (unchanged) AND 2-fold setting-disjoint CV (after rank-k valence erasure
      until `SENT <= 0.55` & `HOUSE <= 0.55`) beat the exact uncentered setting-flip null (both accuracy &
      exact both-mirrors-right rate) at Holm-adjusted p < 0.05.
  - Test B (`bm_polconf245`): Does frozen `bm120` encode global actor-conditioned role binding (`>> 0.50`)
    or local moral action-recipient congruence (`<< 0.50`)?
    * Manipulation check: `zs_sent` < 0.25 AND lexicon `sign(val(verb)*val(adj))` < 0.25.
    * Evaluates BOTH frozen `bm120 -> bm_polconf245`, frozen `bm_neut245 -> bm_polconf245`, and in-domain
      2-fold setting-disjoint CV (raw and after rank-k valence erasure).
  - Test C (Rank-k Iterative Valence Erasure on `bm120`, `bm_rel242`, `bm_neut245`, `bm_polconf245`):
    * Removes VAL2, VAL5 (4 sentiment domains + outcome valence), VAL5+HOUSE, AND rank-k top PCs / iterative
      INLP directions trained strictly on non-binding evaluative data (`val234` + `HOUSE` single-clause) until
      `SENT` and `HOUSE` read out <= 0.55.
  - Test D (Exact Plausibility & Lexical Controls):
    * Stratified accuracy on `DeltaNLL <= 0` and `|DeltaNLL| <= 0.01`, PLUS exact swapped-span local `DeltaNLL`,
      skip-bigram, trigram, and lexicon verb-x-adj polarity baselines.
  - Test E (Exact Uncentered Setting-Flip Null & Paired Depth-Difference CIs):
    * Uncentered setting-flip permutations refitting CV inside every permutation (testing BOTH accuracy and
      exact both-mirrors-right rate against the true setting-flip null, NEVER 25%).
    * Simultaneous paired setting-clustered bootstrap 95% CIs on `acc(d50) - acc(d30)` and `acc(d50) - acc(d70)`.
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

SCRATCH_DIR = Path("./local_scratch")
EXP_DIR = SCRATCH_DIR / "exp245_controls"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OLD_EXP = SCRATCH_DIR / "exp240_242"
REPO_ARCH = Path("./bench/archetypes")
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", "./bench/archetypes/fluent/plato"))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_FLUENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_swap236 as BS236  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402

MLX_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)

DATASETS = {
    "bm_neut245": {
        "json": EXP_DIR / "bm_neut245_data.json",
        "key": "bm_neut_120",
        "prefix": "bmneut245",
    },
    "bm_polconf245": {
        "json": EXP_DIR / "bm_polconf245_data.json",
        "key": "bm_polconf_120",
        "prefix": "bmpolconf245",
    },
}
OUT_JSON = EXP_DIR / "exp245_binding_results.json"

# Fixed, pre-registered sentiment/moral polarity lexicon (frozen before authoring data!)
POS_LEX = {
    "rewarded", "promoted", "honored", "praised", "certified", "shielded", "protected", "funded",
    "endorsed", "pardoned", "cleared", "acquitted", "backed", "elevated", "validated", "trusted",
    "spared", "assisted", "defended", "favored", "approved", "commended", "compensated", "welcomed",
    "granted", "enriched", "supported", "hired", "retained", "preserved", "saved", "rescued",
    "honest", "transparent", "diligent", "ethical", "lawful", "faithful", "loyal", "innocent",
    "truthful", "accountable", "clean", "reliable", "sincere", "devoted", "upright", "fair",
    "principled", "trustworthy", "qualified", "competent", "dedicated", "scrupulous", "candid",
    "legitimate", "genuine", "peaceful", "compliant", "careful", "meritorious", "honorable",
    "uncorrupted", "impartial", "independent", "verified", "reputable", "conscientious",
}
NEG_LEX = {
    "punished", "silenced", "sanctioned", "fired", "disqualified", "curtailed", "penalized", "jailed",
    "blocked", "blacklisted", "expelled", "indicted", "framed", "betrayed", "demoted", "fined",
    "banned", "revoked", "rejected", "buried", "suppressed", "censored", "threatened", "sabotaged",
    "targeted", "harassed", "removed", "ousted", "dismissed", "prosecuted", "imprisoned", "condemned",
    "excluded", "denied", "crushed", "raided", "seized", "audited", "interrogated", "exiled",
    "corrupt", "collusive", "wasteful", "plagiarizing", "fraudulent", "dishonest", "bribed", "complicit",
    "crooked", "reckless", "negligent", "deceptive", "shady", "illicit", "abusive", "guilty",
    "forged", "rigged", "biased", "venal", "unethical", "illegal", "predatory", "hostile",
    "faked", "tainted", "compromised", "fraud", "dirty", "corrupted", "scheming", "lawless",
    "unqualified", "incompetent", "delinquent", "substandard", "counterfeit", "fabricated", "false",
}


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _word_bag(text):
    return Counter(_words(text))


def _skip_bigrams(text, max_gap=4):
    ws = _words(text)
    feats = []
    for i in range(len(ws)):
        for g in range(1, min(max_gap + 2, len(ws) - i)):
            feats.append(f"{ws[i]}__{ws[i + g]}")
    return feats


def _trigrams(text):
    ws = _words(text)
    return [f"{ws[i]}_{ws[i+1]}_{ws[i+2]}" for i in range(max(0, len(ws) - 2))]


def _lex_polarity_pair_score(sent):
    """Score local verb-to-nearby-adjective moral congruence: sum over words of pol(w_i)*pol(w_j) for 1<=j-i<=5."""
    ws = _words(sent)
    pol = [1 if w in POS_LEX else (-1 if w in NEG_LEX else 0) for w in ws]
    score = 0.0
    for i in range(len(ws)):
        if pol[i] == 0:
            continue
        for j in range(i + 1, min(len(ws), i + 6)):
            if pol[j] != 0:
                score += pol[i] * pol[j]
    return score


def _eval_lex_polarity(rows, sid_idx):
    diffs = np.array([_lex_polarity_pair_score(r["pos"]) - _lex_polarity_pair_score(r["neg"]) for r in rows], dtype=np.float64)
    # Break exact ties as 0.5 credit (unbiased chance!)
    wins = np.where(diffs > 0, 1.0, np.where(diffs < 0, 0.0, 0.5))
    strict = diffs > 0
    both_r, both_n, _ = BM238._both_mirrors_right(strict, sid_idx)
    return {
        "acc_tie_half": round(float(np.mean(wins)), 4),
        "acc_strict_gt0": round(float(np.mean(strict)), 4),
        "n_nonzero": int(np.sum(diffs != 0)),
        "both_mirrors_strict": round(float(both_r), 4),
        "both_mirrors_n": int(both_n),
    }


def _char_diff_span(a, b):
    """Find character start/end [i0, i1] bounding all differing substrings between strings a and b."""
    i0 = 0
    na, nb = len(a), len(b)
    while i0 < na and i0 < nb and a[i0] == b[i0]:
        i0 += 1
    ra, rb = na, nb
    while ra > i0 and rb > i0 and a[ra - 1] == b[rb - 1]:
        ra -= 1
        rb -= 1
    return (i0, ra), (i0, rb)


def _token_mask_for_char_span(tok, text, ids, span):
    """Map character span (c0, c1) onto token indices [1..len(ids)-1] (for CE loss alignment)."""
    c0, c1 = span
    n = len(ids)
    # Build cumulative decoded lengths cleanly (works for all HF/MLX tokenizers)
    mask = np.zeros(max(1, n - 1), dtype=bool)
    pref_len = []
    for k in range(1, n + 1):
        dec = tok.decode(ids[:k], skip_special_tokens=True)
        pref_len.append(len(dec))
    # Token t (predicting ids[t] at CE index t-1) covers chars [prev_len, cur_len)
    for t in range(1, n):
        s_c = pref_len[t - 1] if t > 0 else 0
        e_c = pref_len[t]
        if e_c > c0 and s_c < c1:
            mask[t - 1] = True
    if not mask.any():
        mask[:] = True
    return mask


def merge_and_verify(ds_name):
    cfg = DATASETS[ds_name]
    rows = []
    for part in (1, 2):
        p = EXP_DIR / f"{ds_name}_part{part}.json"
        assert p.exists(), f"Missing {p}"
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get(cfg["key"]) or data.get("pairs") or data.get("items")
        rows.extend(data)

    rows.sort(key=lambda r: (str(r["setting_id"]), 0 if r["order"] == "orig" else 1))
    assert len(rows) == 120, f"{ds_name}: expected 120 rows, got {len(rows)}"

    existing_settings = {r["setting"].strip().lower() for r in BM238.load_mirror_data()}
    bmrel_path = OLD_EXP / "bm_rel242_data.json"
    if bmrel_path.exists():
        existing_settings |= {r["setting"].strip().lower() for r in json.loads(bmrel_path.read_text())["bm_rel_120"]}
    for other_name, ocfg in DATASETS.items():
        if other_name != ds_name and ocfg["json"].exists():
            existing_settings |= {r["setting"].strip().lower() for r in json.loads(ocfg["json"].read_text())[ocfg["key"]]}

    by_sid = {}
    bad = []
    lex_hits = 0
    for r in rows:
        sid = str(r["setting_id"])
        assert r["order"] in ("orig", "mirror"), f"bad order {r}"
        assert r["fav_pos"] in ("first", "second"), f"bad fav_pos {r}"
        p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
        if p_txt == n_txt or _word_bag(p_txt) != _word_bag(n_txt):
            bad.append((r["id"], "bow_mismatch"))
        if r["setting"].strip().lower() in existing_settings:
            bad.append((r["id"], "setting_overlap"))
        if ds_name == "bm_neut245":
            ws = set(_words(p_txt))
            overlap_lex = ws & (POS_LEX | NEG_LEX)
            if overlap_lex:
                lex_hits += 1
                bad.append((r["id"], f"moral_lex_hit:{sorted(overlap_lex)}"))
        by_sid.setdefault(sid, []).append(r)

    assert not bad, f"{ds_name} verification failures ({len(bad)}): {bad[:12]}"
    assert len(by_sid) == 60, f"Expected 60 unique settings, got {len(by_sid)}"
    n_first, n_second = 0, 0
    for sid, grp in by_sid.items():
        assert len(grp) == 2, f"Setting {sid} count={len(grp)}"
        assert {x["order"] for x in grp} == {"orig", "mirror"}, f"Setting {sid} orders"
        assert {x["fav_pos"] for x in grp} == {"first", "second"}, f"Setting {sid} fav_pos"
        n_first += sum(1 for x in grp if x["fav_pos"] == "first")
        n_second += sum(1 for x in grp if x["fav_pos"] == "second")
    assert n_first == 60 and n_second == 60

    cfg["json"].write_text(json.dumps({cfg["key"]: rows}, indent=2))
    print(f"MERGED & VERIFIED {cfg['json']} (sha256={_sha256_file(cfg['json'])[:16]}...): 120/120 exact BoW, 60 settings, 60/60 order balance, 0 overlap", flush=True)
    return rows


def extract_dataset(ds_name, keys=MLX_KEYS):
    """Extract hidden states at all 5 depths + whole-sentence NLL + swapped-span local NLL."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    cfg = DATASETS[ds_name]
    rows = json.loads(cfg["json"].read_text())[cfg["key"]]
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    for key in keys:
        out_npz = DEPTH_DIR / f"{cfg['prefix']}_{key}.npz"
        if out_npz.exists():
            print(f"{ds_name.upper()} EXTRACT {key}: exists, skipping", flush=True)
            continue
        spec = PD.MODEL_SPECS[key]
        mx.set_cache_limit(2 << 30)
        t0 = time.time()
        print(f"{ds_name.upper()} EXTRACT {key}: loading {spec['path']} ...", flush=True)
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
        p_nll_m, n_nll_m, p_nll_s, n_nll_s = [], [], [], []
        p_span_m, n_span_m = [], []
        try:
            for r in rows:
                p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
                sp_p, sp_n = _char_diff_span(p_txt, n_txt)
                for side, txt, sp in (("pos", p_txt, sp_p), ("neg", n_txt, sp_n)):
                    ids = PD._text_ids_mlx(tok, txt)
                    assert 2 <= len(ids) <= 512
                    rec_buf.clear()
                    x = mx.array([ids])
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    assert len(rec_buf) == n_d
                    ce = nn.losses.cross_entropy(logits[0, :-1, :].astype(mx.float32), x[0, 1:], reduction="none")
                    nll_m = mx.mean(ce)
                    nll_s = mx.sum(ce)
                    eval_list = [ce, nll_m, nll_s]
                    slot_vecs = []
                    for s in range(n_d):
                        h = rec_buf[s].astype(mx.float32)
                        hl = h[-1]
                        hm = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
                        slot_vecs.append((hl, hm))
                        eval_list.extend([hl, hm])
                    mx.eval(*eval_list)
                    rec_buf.clear()
                    ce_np = np.array(ce, dtype=np.float32)
                    tmask = _token_mask_for_char_span(tok, txt, ids, sp)
                    span_mean = float(ce_np[tmask].mean()) if len(tmask) == len(ce_np) else float(ce_np.mean())
                    if side == "pos":
                        p_nll_m.append(float(nll_m.item()))
                        p_nll_s.append(float(nll_s.item()))
                        p_span_m.append(span_mean)
                        for s, (hl, hm) in enumerate(slot_vecs):
                            pl[s].append(np.array(hl, dtype=np.float32))
                            pm[s].append(np.array(hm, dtype=np.float32))
                    else:
                        n_nll_m.append(float(nll_m.item()))
                        n_nll_s.append(float(nll_s.item()))
                        n_span_m.append(span_mean)
                        for s, (hl, hm) in enumerate(slot_vecs):
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
            pos_nll_mean=np.array(p_nll_m, dtype=np.float32),
            neg_nll_mean=np.array(n_nll_m, dtype=np.float32),
            pos_nll_sum=np.array(p_nll_s, dtype=np.float32),
            neg_nll_sum=np.array(n_nll_s, dtype=np.float32),
            pos_span_nll_mean=np.array(p_span_m, dtype=np.float32),
            neg_span_nll_mean=np.array(n_span_m, dtype=np.float32),
            layer_idxs=np.array(layer_idxs, dtype=np.int32),
            num_layers=np.int32(nl_cnt),
            depth_fracs=np.array(DEPTH_FRACS, dtype=np.float32),
        )
        print(f"{ds_name.upper()}_EXTRACT_DONE {key}: layers={layer_idxs}/{nl_cnt} in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Analysis Primitives

def _holm_adjust(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (tk, pv) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * pv))
        adj[tk] = round(running, 4)
    return adj


def _fit_neutral_rot_at_slot(slot, pool_key, d_plato=256):
    pca_params, Z_tr = {}, {}
    for k in MLX_KEYS:
        d = np.load(DEPTH_DIR / f"depth240_{k}.npz", allow_pickle=True)
        X = d[f"neut_{pool_key}"][slot].astype(np.float64)
        mu, V, rms, z_norm = AR.fit_pca_params(X, d_plato)
        pca_params[k] = (mu, V, rms)
        Z_tr[k] = z_norm
    C = Z_tr[C4_KEYS[0]].copy()
    for _ in range(10):
        aligned = [PD.apply_procrustes_map(Z_tr[k], PD.fit_procrustes_map(Z_tr[k], C)) for k in C4_KEYS]
        C = np.mean(aligned, axis=0)
        C = C / max(1e-12, float(np.sqrt(np.mean(C ** 2))))
    to_c4 = {k: PD.fit_procrustes_map(Z_tr[k], C) for k in MLX_KEYS}
    return {"pca": pca_params, "to_c4": to_c4}


def _map_mat(X, rot, key):
    mu, V, rms = rot["pca"][key]
    return PD.apply_procrustes_map(AR.apply_pca_params(X.astype(np.float64), mu, V, rms), rot["to_c4"][key])


def _setting_counts_and_uncentered_perm(scores, sid_idx, n_perm=2000, seed=2450):
    """Exact uncentered setting-flip null for BOTH accuracy AND exact both-mirrors-right rate!
    Under a setting-level sign flip s_g in {-1, +1}, both mirrors of setting g flip sign together.
    So a 'both_right' pair flips to 'both_wrong', a 'both_wrong' flips to 'both_right', and a 'split'
    pair stays 'split'! Notice the exact null expectation of both_right is (n_both_right + n_both_wrong)/2 / G,
    NOT 25%!"""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    pair_sc = np.zeros((G, 2), dtype=np.float64)
    for gi, s in enumerate(uniq):
        ix = np.where(sid_idx == s)[0]
        assert len(ix) == 2
        pair_sc[gi, 0] = scores[ix[0]]
        pair_sc[gi, 1] = scores[ix[1]]
    obs_acc = float(np.mean(pair_sc > 0))
    n_br = int(np.sum((pair_sc[:, 0] > 0) & (pair_sc[:, 1] > 0)))
    n_bw = int(np.sum((pair_sc[:, 0] <= 0) & (pair_sc[:, 1] <= 0)))
    n_sp = G - n_br - n_bw
    obs_br = n_br / G
    exp_br_null = 0.5 * (n_br + n_bw) / G

    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, G))
    p0 = flips * pair_sc[None, :, 0]
    p1 = flips * pair_sc[None, :, 1]
    null_acc = 0.5 * ((p0 > 0).mean(axis=1) + (p1 > 0).mean(axis=1))
    null_br = ((p0 > 0) & (p1 > 0)).mean(axis=1)
    p_acc = float((np.sum(null_acc >= obs_acc - 1e-12) + 1) / (n_perm + 1))
    p_br = float((np.sum(null_br >= obs_br - 1e-12) + 1) / (n_perm + 1))
    return {
        "acc": round(obs_acc, 4),
        "both_right": round(obs_br, 4),
        "counts_br_sp_bw": [n_br, n_sp, n_bw],
        "null_expected_both_right": round(exp_br_null, 4),
        "p_uncentered_acc": round(p_acc, 4),
        "p_uncentered_both_right": round(p_br, 4),
    }


def _cv2_refit_perm_p(Ds, Dt, Qs, Qt, sid_idx, splits, n_perm=500, seed=2455):
    """Refit the ENTIRE 2-fold CV direction inside every setting-flip permutation (uncentered!)."""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    sid_to_g = {s: i for i, s in enumerate(uniq)}
    g_of_row = np.array([sid_to_g[int(s)] for s in sid_idx], dtype=int)

    Ds_p = V234._project_out_rows(Ds, Qs) if Qs is not None and Qs.shape[1] > 0 else Ds
    Dt_p = V234._project_out_rows(Dt, Qt) if Qt is not None and Qt.shape[1] > 0 else Dt
    sp0 = splits[0]

    def run_once(signs_g):
        sgn = signs_g[g_of_row]
        Ds_f = Ds_p * sgn[:, None]
        Dt_f = Dt_p * sgn[:, None]
        oof = np.zeros(len(sid_idx), dtype=np.float64)
        for f in (0, 1):
            tr, te = (sp0 != f), (sp0 == f)
            ut = AR.unit(Ds_f[tr].mean(0))
            if Qt is not None and Qt.shape[1] > 0:
                ut = V234._project_out(ut, Qt)
            oof[te] = Dt_f[te] @ ut
        acc = float(np.mean(oof > 0))
        br, _, _ = BM238._both_mirrors_right(oof > 0, sid_idx)
        return acc, br, oof

    obs_acc, obs_br, obs_oof = run_once(np.ones(G))
    rng = np.random.default_rng(seed)
    ge_acc, ge_br = 0, 0
    for _ in range(n_perm):
        sg = rng.choice([-1.0, 1.0], size=G)
        a, b, _ = run_once(sg)
        if a >= obs_acc - 1e-12:
            ge_acc += 1
        if b >= obs_br - 1e-12:
            ge_br += 1
    return {
        "split0_acc": round(obs_acc, 4),
        "split0_both": round(obs_br, 4),
        "p_refit_acc": round((ge_acc + 1) / (n_perm + 1), 4),
        "p_refit_both": round((ge_br + 1) / (n_perm + 1), 4),
        "_oof": obs_oof,
    }


def _build_valence_erasure_bases(Dv, Dh, sg_doms, house_cats, k_list=(2, 5, 6, 12, 24, 40)):
    """Build strictly non-binding evaluative erasure bases of increasing rank:
      - rank 2 (VAL2): sentiment mean + outcome valence mean
      - rank 5 (VAL5): 4 sentiment domain means + outcome valence mean
      - rank 6 (VAL5+HOUSE): VAL5 + HOUSE mean
      - rank k (VAL_INLP_k): top-k SVD directions spanning ALL evaluative differences [Dv; Dh]
        (purely non-binding evaluative data — ZERO swap sentences ever used!)."""
    n_sg = len(sg_doms)
    u_sent = AR.unit(Dv[:n_sg].mean(0))
    u_ov = AR.unit(Dv[n_sg:].mean(0))
    Q2 = V234._orthonormal_basis([u_sent, u_ov])

    dom_vecs = [AR.unit(Dv[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
    Q5 = V234._orthonormal_basis(dom_vecs + [u_ov])

    uh = AR.house_dir(Dh, house_cats)
    Q6 = V234._orthonormal_basis(dom_vecs + [u_ov, uh])

    # Stack centered/unit-weighted evaluative differences [Dv; Dh] for rank-k PCA/SVD erasure
    M_eval = np.vstack([Dv, Dh])
    # Include the mean explicitly as rank-1, plus top residual PCs -> exact SVD of uncentered M_eval!
    _, _, Vt = np.linalg.svd(M_eval, full_matrices=False)
    bases = {"raw": np.zeros((Dv.shape[1], 0)), "val2": Q2, "val5": Q5, "val5_house": Q6}
    for k in k_list:
        if k > 6:
            # Always include Q6 plus top remaining singular vectors of M_eval projected out of Q6
            M_res = V234._project_out_rows(M_eval, Q6)
            _, _, Vt_r = np.linalg.svd(M_res, full_matrices=False)
            extra = Vt_r[: max(0, k - Q6.shape[1])].T
            Q_full, _ = np.linalg.qr(np.hstack([Q6, extra]))
            bases[f"eval_svd_{k}"] = Q_full[:, :k]
    return bases, u_sent, uh


def _dnll_strata_eval(scores, dnll_vec, sid_idx):
    """Exact accuracy inside DeltaNLL <= 0 and |DeltaNLL| <= 0.01 (never just noisy OLS intercept!)."""
    out = {}
    for name, mask in [
        ("all", np.ones(len(scores), dtype=bool)),
        ("dnll_le_0", dnll_vec <= 0.0),
        ("dnll_gt_0", dnll_vec > 0.0),
        ("abs_dnll_le_0p01", np.abs(dnll_vec) <= 0.01),
        ("abs_dnll_le_0p02", np.abs(dnll_vec) <= 0.02),
    ]:
        n = int(np.sum(mask))
        n_sets = int(len(set(sid_idx[mask].tolist()))) if n > 0 else 0
        out[name] = {
            "n_items": n,
            "n_settings": n_sets,
            "acc": round(float(np.mean(scores[mask] > 0)), 4) if n > 0 else None,
        }
    return out


def _paired_depth_bootstrap(acc_by_depth_by_setting, n_boot=2000, seed=2459):
    """Simultaneous setting-clustered bootstrap 95% CIs (Bonferroni alpha/2 -> 97.5% per contrast)
    for acc(d50) - acc(d30) and acc(d50) - acc(d70)."""
    a30 = np.asarray(acc_by_depth_by_setting["d30"], dtype=np.float64)  # (G,)
    a50 = np.asarray(acc_by_depth_by_setting["d50"], dtype=np.float64)
    a70 = np.asarray(acc_by_depth_by_setting["d70"], dtype=np.float64)
    G = len(a50)
    rng = np.random.default_rng(seed)
    d50_30, d50_70 = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, G, size=G)
        d50_30.append(float(np.mean(a50[ix] - a30[ix])))
        d50_70.append(float(np.mean(a50[ix] - a70[ix])))
    d50_30 = np.asarray(d50_30)
    d50_70 = np.asarray(d50_70)
    return {
        "d50_minus_d30": {
            "obs": round(float(np.mean(a50 - a30)), 4),
            "ci95": [round(float(np.percentile(d50_30, 2.5)), 4), round(float(np.percentile(d50_30, 97.5)), 4)],
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_30, 1.25)), 4), round(float(np.percentile(d50_30, 98.75)), 4)],
        },
        "d50_minus_d70": {
            "obs": round(float(np.mean(a50 - a70)), 4),
            "ci95": [round(float(np.percentile(d50_70, 2.5)), 4), round(float(np.percentile(d50_70, 97.5)), 4)],
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_70, 1.25)), 4), round(float(np.percentile(d50_70, 98.75)), 4)],
        },
    }


def _load_all_four_benchmarks():
    bm120_rows = BM238.load_mirror_data()
    bmrel_rows = json.loads((OLD_EXP / "bm_rel242_data.json").read_text())["bm_rel_120"]
    out = {"bm120": bm120_rows, "bm_rel242": bmrel_rows}
    for name, cfg in DATASETS.items():
        if cfg["json"].exists():
            out[name] = json.loads(cfg["json"].read_text())[cfg["key"]]
    return out


def _sid_and_splits(rows, seed_base):
    sids_str = [str(r["setting_id"]) for r in rows]
    uniq = sorted(set(sids_str))
    sid2i = {s: i for i, s in enumerate(uniq)}
    sid_idx = np.array([sid2i[s] for s in sids_str], dtype=int)
    splits = []
    for s_i in range(20):
        rng = np.random.default_rng(seed_base + s_i)
        perm = rng.permutation(len(uniq))
        f0 = set(int(x) for x in perm[: len(uniq) // 2])
        splits.append(np.array([0 if int(s) in f0 else 1 for s in sid_idx], dtype=int))
    return sid_idx, splits


def analyze_all():
    t0 = time.time()
    benches = _load_all_four_benchmarks()
    assert "bm_neut245" in benches and "bm_polconf245" in benches, "Missing new 245 datasets!"

    # Hash manifest for full auditability
    manifest = {
        "script_sha256": _sha256_file(__file__),
        "deps_sha256": {
            fn: _sha256_file(REPO_FLUENT / fn)
            for fn in ("plato_data.py", "arch_rot229.py", "valence_ctrl234.py", "bow_swap236.py", "bow_mirror238.py")
        },
        "data_sha256": {
            "bm_rel242": _sha256_file(OLD_EXP / "bm_rel242_data.json"),
            "bm_neut245": _sha256_file(DATASETS["bm_neut245"]["json"]),
            "bm_polconf245": _sha256_file(DATASETS["bm_polconf245"]["json"]),
        },
    }

    sid_map, split_map, lexical_map = {}, {}, {}
    for bname, rows in benches.items():
        sbase = {"bm120": 23800, "bm_rel242": 24200, "bm_neut245": 24500, "bm_polconf245": 24600}[bname]
        sid_idx, splits = _sid_and_splits(rows, sbase)
        sid_map[bname] = sid_idx
        split_map[bname] = splits
        lexical_map[bname] = {
            "bow_unigram": round(float(np.mean(BS236._bow_2fold_pairs(rows, "pos", "neg", BS236._words, splits))), 4),
            "bow_bigram": round(float(np.mean(BS236._bow_2fold_pairs(rows, "pos", "neg", BS236._bigrams, splits))), 4),
            "bow_skipbigram_g4": round(float(np.mean(BS236._bow_2fold_pairs(rows, "pos", "neg", _skip_bigrams, splits))), 4),
            "bow_trigram": round(float(np.mean(BS236._bow_2fold_pairs(rows, "pos", "neg", _trigrams, splits))), 4),
            "lex_verb_adj_polarity": _eval_lex_polarity(rows, sid_idx),
        }

    # NLL & Swapped-Span NLL baselines
    nll_map = {}
    dnll_store = {}
    for bname in ("bm_rel242", "bm_neut245", "bm_polconf245"):
        pref = "bmrel242" if bname == "bm_rel242" else DATASETS[bname]["prefix"]
        nll_map[bname] = {}
        dnll_store[bname] = {}
        for k in MLX_KEYS:
            dr = np.load(DEPTH_DIR / f"{pref}_{k}.npz")
            dm = dr["neg_nll_mean"].astype(np.float64) - dr["pos_nll_mean"].astype(np.float64)
            dnll_store[bname][k] = dm
            entry = {
                "whole_sent": _setting_counts_and_uncentered_perm(dm, sid_map[bname], n_perm=1000, seed=2451),
            }
            if "neg_span_nll_mean" in dr:
                ds_ = dr["neg_span_nll_mean"].astype(np.float64) - dr["pos_span_nll_mean"].astype(np.float64)
                entry["swapped_span"] = _setting_counts_and_uncentered_perm(ds_, sid_map[bname], n_perm=1000, seed=2452)
            nll_map[bname][k] = entry

    # Valence & HOUSE metadata
    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    sg_rows, ov_rows, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)

    # Full 5-depth evaluation (both pools) + deep Rank-k Erasure & Stratified NLL at primary d50!
    by_depth = {}
    # Store per-setting mean-across-5-targets accuracy for paired depth bootstrap
    depth_setting_acc = {
        (pool, src_k, bname, readout): {}
        for pool in ("last", "mean")
        for src_k in ("gemma4_e4b", "qwen35_9b")
        for bname in benches
        for readout in ("cv2_val5", "xdom_bm120_val5", "zs_sent")
    }

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        print(f"Analyzing depth {dkey} ...", flush=True)
        by_depth[dkey] = {}
        for pool_key in ("last", "mean"):
            rot = _fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
            Z = {bname: {} for bname in benches}
            bases_by_k, u_sent_k, u_house_k = {}, {}, {}
            Dv_by_k, Dh_by_k = {}, {}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                Z["bm120"][k] = _map_mat(d240[f"bm120_pos_{pool_key}"][slot], rot, k) - _map_mat(d240[f"bm120_neg_{pool_key}"][slot], rot, k)
                for bname, pref in [("bm_rel242", "bmrel242"), ("bm_neut245", "bmneut245"), ("bm_polconf245", "bmpolconf245")]:
                    dr = np.load(DEPTH_DIR / f"{pref}_{k}.npz")
                    Z[bname][k] = _map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - _map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
                Dh = _map_mat(d240[f"house_pos_{pool_key}"][slot][house_idx], rot, k) - _map_mat(d240[f"house_neg_{pool_key}"][slot][house_idx], rot, k)
                Dv = _map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - _map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
                Dv_by_k[k], Dh_by_k[k] = Dv, Dh
                bases_by_k[k], u_sent_k[k], u_house_k[k] = _build_valence_erasure_bases(Dv, Dh, sg_doms, house_cats)

            pool_res = {}
            for src_k in ("gemma4_e4b", "qwen35_9b"):
                tgts = [k for k in MLX_KEYS if k != src_k]
                src_out = {}
                for bname in benches:
                    sid_idx = sid_map[bname]
                    splits = split_map[bname]
                    Ds = Z[bname][src_k]
                    u_bm120_raw = AR.unit(Z["bm120"][src_k].mean(0))
                    u_neut_raw = AR.unit(Z["bm_neut245"][src_k].mean(0))

                    tgt_out = {}
                    # Accumulators for setting-level paired depth bootstrap
                    uniq_g = sorted(set(int(s) for s in sid_idx))
                    per_g_cv2 = np.zeros(len(uniq_g))
                    per_g_xdom = np.zeros(len(uniq_g))
                    per_g_sent = np.zeros(len(uniq_g))

                    for tk in tgts:
                        Dt = Z[bname][tk]
                        # Zero-shot sentiment & HOUSE raw
                        sc_sent = Dt @ u_sent_k[src_k]
                        res_sent = _setting_counts_and_uncentered_perm(sc_sent, sid_idx, n_perm=1000, seed=2451)
                        sc_hraw = Dt @ u_house_k[src_k]
                        res_hraw = _setting_counts_and_uncentered_perm(sc_hraw, sid_idx, n_perm=1000, seed=2452)

                        # Standard VAL5 erasure (primary clean non-binding valence control!)
                        Qs5, Qt5 = bases_by_k[src_k]["val5"], bases_by_k[tk]["val5"]
                        Dt5 = V234._project_out_rows(Dt, Qt5)
                        u_bm_5 = V234._project_out(V234._project_out(u_bm120_raw, Qs5), Qt5)
                        sc_xdom5 = Dt5 @ u_bm_5
                        res_xdom5 = _setting_counts_and_uncentered_perm(sc_xdom5, sid_idx, n_perm=2000 if dkey == "d50" else 500, seed=2453)

                        u_neut_5 = V234._project_out(V234._project_out(u_neut_raw, Qs5), Qt5)
                        sc_xneut5 = Dt5 @ u_neut_5
                        res_xneut5 = _setting_counts_and_uncentered_perm(sc_xneut5, sid_idx, n_perm=1000 if dkey == "d50" else 500, seed=2454)

                        # 20-split setting-disjoint 2F CV (raw & perp VAL5)
                        accs_raw, accs_v5, boths_v5 = [], [], []
                        oof_v5_mean = np.zeros(len(sid_idx), dtype=np.float64)
                        for fl in splits:
                            oof_r = np.zeros(len(sid_idx))
                            oof_5 = np.zeros(len(sid_idx))
                            for f in (0, 1):
                                tr, te = (fl != f), (fl == f)
                                ut = AR.unit(Ds[tr].mean(0))
                                ut5 = V234._project_out(V234._project_out(ut, Qs5), Qt5)
                                oof_r[te] = Dt[te] @ ut
                                oof_5[te] = Dt5[te] @ ut5
                            accs_raw.append(float(np.mean(oof_r > 0)))
                            accs_v5.append(float(np.mean(oof_5 > 0)))
                            br, _, _ = BM238._both_mirrors_right(oof_5 > 0, sid_idx)
                            boths_v5.append(br)
                            oof_v5_mean += (oof_5 > 0).astype(np.float64) / len(splits)

                        # Setting-level mean accuracy for paired depth bootstrap
                        for gi, s in enumerate(uniq_g):
                            ix = np.where(sid_idx == s)[0]
                            per_g_cv2[gi] += float(np.mean(oof_v5_mean[ix])) / len(tgts)
                            per_g_xdom[gi] += float(np.mean(sc_xdom5[ix] > 0)) / len(tgts)
                            per_g_sent[gi] += float(np.mean(sc_sent[ix] > 0)) / len(tgts)

                        entry = {
                            "zs_sent": res_sent,
                            "zs_house_raw": res_hraw,
                            "xdom_bm120_raw_acc": round(float(np.mean((Dt @ u_bm120_raw) > 0)), 4),
                            "xdom_bm120_val5": res_xdom5,
                            "xdom_bmneut_val5": res_xneut5,
                            "cv2_raw_20split": round(float(np.mean(accs_raw)), 4),
                            "cv2_val5_20split": round(float(np.mean(accs_v5)), 4),
                            "cv2_val5_both_20split": round(float(np.mean(boths_v5)), 4),
                        }

                        # At primary d50: FULL Rank-k Erasure Sweep + CV Refit Permutation + Exact DeltaNLL Strata!
                        if dkey == "d50":
                            cv_perm = _cv2_refit_perm_p(Ds, Dt, Qs5, Qt5, sid_idx, splits, n_perm=500, seed=2455)
                            entry["cv2_val5_refit_perm"] = {k_: v_ for k_, v_ in cv_perm.items() if k_ != "_oof"}

                            if bname in dnll_store:
                                entry["dnll_strata_xdom_val5"] = _dnll_strata_eval(sc_xdom5, dnll_store[bname][tk], sid_idx)
                                entry["dnll_strata_cv2_val5"] = _dnll_strata_eval(cv_perm["_oof"], dnll_store[bname][tk], sid_idx)

                            erasure_sweep = {}
                            for ek in ("raw", "val2", "val5", "val5_house", "eval_svd_12", "eval_svd_24", "eval_svd_40"):
                                Qs_e, Qt_e = bases_by_k[src_k][ek], bases_by_k[tk][ek]
                                Dt_e = V234._project_out_rows(Dt, Qt_e) if Qt_e.shape[1] > 0 else Dt
                                # Check residual SENT & HOUSE readability on evaluative sets after erasure!
                                Dv_t_e = V234._project_out_rows(Dv_by_k[tk][:n_sg], Qt_e) if Qt_e.shape[1] > 0 else Dv_by_k[tk][:n_sg]
                                Dh_t_e = V234._project_out_rows(Dh_by_k[tk], Qt_e) if Qt_e.shape[1] > 0 else Dh_by_k[tk]
                                us_e = V234._project_out(V234._project_out(u_sent_k[src_k], Qs_e), Qt_e) if Qt_e.shape[1] > 0 else u_sent_k[src_k]
                                uh_e = V234._project_out(V234._project_out(u_house_k[src_k], Qs_e), Qt_e) if Qt_e.shape[1] > 0 else u_house_k[src_k]
                                sent_eval_acc = float(np.mean((Dv_t_e @ us_e) > 0)) if np.linalg.norm(us_e) > 1e-8 else 0.5
                                house_eval_acc = float(np.mean((Dh_t_e @ uh_e) > 0)) if np.linalg.norm(uh_e) > 1e-8 else 0.5
                                sent_swap_acc = float(np.mean((Dt_e @ us_e) > 0)) if np.linalg.norm(us_e) > 1e-8 else 0.5

                                ub_e = V234._project_out(V234._project_out(u_bm120_raw, Qs_e), Qt_e) if Qt_e.shape[1] > 0 else u_bm120_raw
                                sc_e = Dt_e @ ub_e
                                xdom_acc_e = float(np.mean(sc_e > 0))
                                xdom_br_e, _, _ = BM238._both_mirrors_right(sc_e > 0, sid_idx)

                                # 2F CV split-0..4 mean under this rank-k erasure
                                c_accs, c_brs = [], []
                                for fl in splits[:5]:
                                    oof = np.zeros(len(sid_idx))
                                    for f in (0, 1):
                                        tr, te = (fl != f), (fl == f)
                                        ut = AR.unit(Ds[tr].mean(0))
                                        if Qt_e.shape[1] > 0:
                                            ut = V234._project_out(V234._project_out(ut, Qs_e), Qt_e)
                                        oof[te] = Dt_e[te] @ ut
                                    c_accs.append(float(np.mean(oof > 0)))
                                    br_, _, _ = BM238._both_mirrors_right(oof > 0, sid_idx)
                                    c_brs.append(br_)
                                erasure_sweep[ek] = {
                                    "rank": int(Qt_e.shape[1]),
                                    "resid_sent_on_val234": round(sent_eval_acc, 4),
                                    "resid_house_on_house233": round(house_eval_acc, 4),
                                    "resid_sent_on_swap": round(sent_swap_acc, 4),
                                    "xdom_bm120_acc": round(xdom_acc_e, 4),
                                    "xdom_bm120_both": round(xdom_br_e, 4),
                                    "cv2_acc": round(float(np.mean(c_accs)), 4),
                                    "cv2_both": round(float(np.mean(c_brs)), 4),
                                }
                            entry["rank_k_erasure_sweep"] = erasure_sweep

                        tgt_out[tk] = entry

                    depth_setting_acc[(pool_key, src_k, bname, "cv2_val5")][dkey] = per_g_cv2
                    depth_setting_acc[(pool_key, src_k, bname, "xdom_bm120_val5")][dkey] = per_g_xdom
                    depth_setting_acc[(pool_key, src_k, bname, "zs_sent")][dkey] = per_g_sent

                    # Summaries + Holm across the 5 targets
                    summ = {
                        "mean_zs_sent": round(float(np.mean([tgt_out[t]["zs_sent"]["acc"] for t in tgts])), 4),
                        "mean_xdom_bm120_raw": round(float(np.mean([tgt_out[t]["xdom_bm120_raw_acc"] for t in tgts])), 4),
                        "mean_xdom_bm120_val5": round(float(np.mean([tgt_out[t]["xdom_bm120_val5"]["acc"] for t in tgts])), 4),
                        "mean_xdom_bm120_val5_both": round(float(np.mean([tgt_out[t]["xdom_bm120_val5"]["both_right"] for t in tgts])), 4),
                        "mean_xdom_bmneut_val5": round(float(np.mean([tgt_out[t]["xdom_bmneut_val5"]["acc"] for t in tgts])), 4),
                        "mean_cv2_raw": round(float(np.mean([tgt_out[t]["cv2_raw_20split"] for t in tgts])), 4),
                        "mean_cv2_val5": round(float(np.mean([tgt_out[t]["cv2_val5_20split"] for t in tgts])), 4),
                        "mean_cv2_val5_both": round(float(np.mean([tgt_out[t]["cv2_val5_both_20split"] for t in tgts])), 4),
                        "holm_xdom_bm120_val5_acc": _holm_adjust({t: tgt_out[t]["xdom_bm120_val5"]["p_uncentered_acc"] for t in tgts}),
                        "holm_xdom_bm120_val5_both": _holm_adjust({t: tgt_out[t]["xdom_bm120_val5"]["p_uncentered_both_right"] for t in tgts}),
                    }
                    if dkey == "d50":
                        summ["holm_cv2_val5_refit_acc"] = _holm_adjust({t: tgt_out[t]["cv2_val5_refit_perm"]["p_refit_acc"] for t in tgts})
                        summ["holm_cv2_val5_refit_both"] = _holm_adjust({t: tgt_out[t]["cv2_val5_refit_perm"]["p_refit_both"] for t in tgts})
                        summ["erasure_5tgt_mean"] = {
                            ek: {
                                "resid_sent_val234": round(float(np.mean([tgt_out[t]["rank_k_erasure_sweep"][ek]["resid_sent_on_val234"] for t in tgts])), 4),
                                "resid_sent_swap": round(float(np.mean([tgt_out[t]["rank_k_erasure_sweep"][ek]["resid_sent_on_swap"] for t in tgts])), 4),
                                "xdom_bm120_acc": round(float(np.mean([tgt_out[t]["rank_k_erasure_sweep"][ek]["xdom_bm120_acc"] for t in tgts])), 4),
                                "cv2_acc": round(float(np.mean([tgt_out[t]["rank_k_erasure_sweep"][ek]["cv2_acc"] for t in tgts])), 4),
                                "cv2_both": round(float(np.mean([tgt_out[t]["rank_k_erasure_sweep"][ek]["cv2_both"] for t in tgts])), 4),
                            }
                            for ek in ("raw", "val2", "val5", "val5_house", "eval_svd_12", "eval_svd_24", "eval_svd_40")
                        }
                    src_out[bname] = {"summary": summ, "targets": tgt_out}
                pool_res[src_k] = src_out
            by_depth[dkey][pool_key] = pool_res

    # Paired setting-clustered depth-difference bootstrap CIs (d50 - d30 and d50 - d70)
    depth_contrasts = {}
    for (pool, src_k, bname, readout), dmap in depth_setting_acc.items():
        depth_contrasts.setdefault(pool, {}).setdefault(src_k, {}).setdefault(bname, {})[readout] = _paired_depth_bootstrap(dmap)

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "primary_cell": "d50.last.gemma4_e4b (co-primary: d50.last.qwen35_9b)",
        "manifest": manifest,
        "lexical_baselines": lexical_map,
        "output_prob_nll_baselines": nll_map,
        "paired_depth_contrasts_d50_vs_d30_d70": depth_contrasts,
        "by_depth": by_depth,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} in {report['elapsed_sec']}s", flush=True)
    return report


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "merge":
        merge_and_verify(sys.argv[2])
    elif cmd == "extract":
        ds_name = sys.argv[2]
        keys = tuple(sys.argv[3:]) if len(sys.argv) > 3 else MLX_KEYS
        extract_dataset(ds_name, keys)
    elif cmd == "analyze":
        analyze_all()
    else:
        raise SystemExit(f"Unknown cmd: {cmd}")


if __name__ == "__main__":
    main()
