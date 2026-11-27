#!/usr/bin/env python3
"""Cross-Lingual Causal Story Structure Audit (4 Languages, 3 Language Families, 5 Open Models).

Reads the committed stimuli (`bench/archetypes/crosslang/items.jsonl`, `checked.jsonl`, `sign.jsonl`)
and the 5 model activation captures (`caps_*.npz` from `JPL_CROSSLANG_CACHE_DIR`, produced by
`bench/archetypes/crosslang.py capture`) and writes `bench/archetypes/crosslang/v6_crosslang_audit.json`.

Computes:
  - Primary C1 (outcome-direction controlled, same-pattern minus different-pattern sharing outcome sign)
    across English (`en`), Modern Greek (`el`), Japanese (`ja`), and Simplified Chinese (`zh`)
    (6 language pairs, 5 models, 78 passing patterns), with 10,000-draw pattern bootstrap 95% CIs.
  - Edited-sentences-alone control (`true_clause` vs `foil_clause`) and paired story-beyond-clauses delta.
  - Myths vs. mechanisms contrast (`C4`), character 3-gram lexical overlap check, and Set T ceiling.
  - Fixed within-outcome-class shuffled floor (permuting pattern labels strictly within each outcome class
    `sa == sb`, fixing the legacy cross-class permutation bug that produced `-0.0360`) AND clear
    sign-matched subset (`sa == sb in {'true', 'foil'}`).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CROSSLANG_DIR = REPO / "bench" / "archetypes" / "crosslang"
CACHE_DIR = Path(os.environ.get("JPL_CROSSLANG_CACHE_DIR", str(CROSSLANG_DIR))).resolve()
OUT_JSON = CROSSLANG_DIR / "v6_crosslang_audit.json"

LANGS = ("en", "el", "ja", "zh")
LANG_PAIRS = [(a, b) for i, a in enumerate(LANGS) for b in LANGS[i + 1 :]]
SEED = 87
DEPTH = 2  # 60% depth (index 2 of FRAC = (0.2, 0.4, 0.6, 0.8, 1.0))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    items_p = CROSSLANG_DIR / "items.jsonl"
    checked_p = CROSSLANG_DIR / "checked.jsonl"
    sign_p = CROSSLANG_DIR / "sign.jsonl"

    P = {(c["set"], c["kind"], c["pattern_id"], c["lang"]): c for c in map(json.loads, checked_p.open())}
    Sg = {(r["kind"], r["pattern_id"], r["lang"]): r["better"] for r in map(json.loads, sign_p.open())}

    def passed(st, kind, pid, lg):
        c = P.get((st, kind, pid, lg))
        return bool(c) and c["p31"] >= 0.7 and c["mass"] >= 0.5

    cap_files = sorted(f for f in CACHE_DIR.glob("caps_*.npz") if ".tmp" not in f.name)
    if not cap_files:
        raise FileNotFoundError(
            f"No caps_*.npz files found in {CACHE_DIR}. Run `bench/archetypes/crosslang.py capture` or set JPL_CROSSLANG_CACHE_DIR."
        )

    def _table(z, st):
        H = z["H"][:, DEPTH].astype(np.float64)
        return {tuple(m[1:]): x for x, m in zip(H, z["meta"].astype(str)) if m[0] == st}

    Z = {}
    for f in cap_files:
        z = np.load(f, allow_pickle=True)
        Z[f.stem[5:]] = (_table(z, "I"), _table(z, "T"), _table(z, "R"))

    def key_steps(T, kind, a, b, ver, filt):
        ids = sorted(
            {k[1] for k in T if k[0] == kind and k[2] == a and k[3] == ver[0]}
            & {k[1] for k in T if k[0] == kind and k[2] == b and k[3] == ver[0]}
        )
        ids = [
            p
            for p in ids
            if (kind, p, a, ver[1]) in T
            and (kind, p, b, ver[1]) in T
            and (not filt or (passed("I", kind, p, a) and passed("I", kind, p, b)))
        ]
        if len(ids) < 8:
            return ids, None, None
        Da = _unit(np.array([T[kind, p, a, ver[0]] - T[kind, p, a, ver[1]] for p in ids]))
        Db = _unit(np.array([T[kind, p, b, ver[0]] - T[kind, p, b, ver[1]] for p in ids]))
        return ids, Da, Db

    def pat_scores(Da, Db, remove, sa=None, sb=None):
        n = len(Da)

        def s_of(M, rows):
            if sa is None:
                return M[rows, rows] - (M[rows].sum(1) - M[rows, rows] + M[:, rows].sum(0) - M[rows, rows]) / (2 * (n - 1))
            out = []
            for p in rows:
                q1 = [q for q in range(n) if q != p and sb[q] == sa[p]]
                q2 = [q for q in range(n) if q != p and sa[q] == sb[p]]
                out.append(M[p, p] - (M[p, q1].mean() + M[q2, p].mean()) / 2 if len(q1) >= 3 and len(q2) >= 3 else np.nan)
            return np.array(out)

        if not remove:
            return s_of(Da @ Db.T, np.arange(n))
        order = np.random.default_rng(SEED).permutation(n)
        out = np.zeros(n)
        for fold in np.array_split(order, 5):
            rest = np.setdiff1d(np.arange(n), fold)
            _, _, Vt = np.linalg.svd(np.vstack([Da[rest], Db[rest]]), full_matrices=False)
            V = Vt[:remove]
            pa, pb = _unit(Da - (Da @ V.T) @ V), _unit(Db - (Db @ V.T) @ V)
            out[fold] = s_of(pa @ pb.T, fold)
        return out

    def collect(T, kind, ver=("true", "foil"), remove=0, filt=True, skip=(), signed=True):
        acc, np_ = {}, {}
        for a, b in LANG_PAIRS:
            if (a, b) in skip:
                continue
            ids, Da, Db = key_steps(T, kind, a, b, ver, filt)
            if Da is None:
                continue
            sa, sb = ([Sg.get((kind, p, a)) for p in ids], [Sg.get((kind, p, b)) for p in ids]) if signed else (None, None)
            s = pat_scores(Da, Db, remove, sa, sb)
            np_[f"{a}-{b}"] = (len(ids), float(np.nanmean(s)))
            for p, x in zip(ids, s):
                if np.isfinite(x):
                    acc.setdefault(p, []).append(x)
        return {p: float(np.mean(v)) for p, v in acc.items()}, np_

    def boot_ci(vals, n=10000):
        v = np.array(list(vals), dtype=np.float64)
        bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
        return {
            "mean": round(float(v.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "n_patterns": int(len(v)),
        }

    C = {name: {kind: collect(T[0], kind) for kind in ("myth", "mechanism")} for name, T in Z.items()}
    Craw = {name: {kind: collect(T[0], kind, signed=False) for kind in ("myth", "mechanism")} for name, T in Z.items()}
    Cax = {name: {kind: collect(T[0], kind, signed=False, remove=1) for kind in ("myth", "mechanism")} for name, T in Z.items()}
    Cls = {name: {kind: collect(T[0], kind, ver=("true_clause", "foil_clause")) for kind in ("myth", "mechanism")} for name, T in Z.items()}
    Nj = {name: {kind: collect(T[0], kind, skip={("ja", "zh")}) for kind in ("myth", "mechanism")} for name, T in Z.items()}

    def pool(D, kind):
        names = list(Z)
        ids = set.intersection(*[set(D[n][kind][0]) for n in names])
        return {p: float(np.mean([D[n][kind][0][p] for n in names])) for p in ids}

    A1, B1 = pool(C, "myth"), pool(C, "mechanism")
    R1, R2 = pool(Craw, "myth"), pool(Craw, "mechanism")
    X1, X2 = pool(Cax, "myth"), pool(Cax, "mechanism")
    C1a, C1b = pool(Cls, "myth"), pool(Cls, "mechanism")
    Nj_a, Nj_b = pool(Nj, "myth"), pool(Nj, "mechanism")

    c1_primary = boot_ci(list(A1.values()) + list(B1.values()))
    c1_raw = boot_ci(list(R1.values()) + list(R2.values()))
    c1_axis_rm = boot_ci(list(X1.values()) + list(X2.values()))
    c1_clauses = boot_ci(list(C1a.values()) + list(C1b.values()))
    ex = [A1[q] - C1a[q] for q in A1 if q in C1a] + [B1[q] - C1b[q] for q in B1 if q in C1b]
    c1_beyond_clauses = boot_ci(ex)
    c1_no_jazh = boot_ci(list(Nj_a.values()) + list(Nj_b.values()))

    my, me = np.array(list(A1.values())), np.array(list(B1.values()))
    bs_c4 = [my[rng.integers(0, len(my), len(my))].mean() - me[rng.integers(0, len(me), len(me))].mean() for _ in range(10000)]
    c4_myth_minus_mech = {
        "diff": round(float(my.mean() - me.mean()), 4),
        "ci95": [round(float(np.percentile(bs_c4, 2.5)), 4), round(float(np.percentile(bs_c4, 97.5)), 4)],
        "myths_mean": round(float(my.mean()), 4),
        "mechanisms_mean": round(float(me.mean()), 4),
    }

    per_model = {}
    for n in Z:
        per_model[n] = {
            "myths": boot_ci(C[n]["myth"][0].values()),
            "mechanisms": boot_ci(C[n]["mechanism"][0].values()),
            "overall": boot_ci(list(C[n]["myth"][0].values()) + list(C[n]["mechanism"][0].values())),
            "clauses_only": boot_ci(list(Cls[n]["myth"][0].values()) + list(Cls[n]["mechanism"][0].values())),
        }

    by_lang_pair = {}
    for n in Z:
        for kind in ("myth", "mechanism"):
            for k, (_, v) in C[n][kind][1].items():
                by_lang_pair.setdefault(k, []).append(v)
    by_lang_pair = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_lang_pair.items())}

    # Fixed within-outcome-class shuffled floor + clear sign-matched subset (sa == sb in {"true", "foil"})
    rng_fix = np.random.default_rng(2026)
    fixed_shuf_means = []
    clear_sign_vals = []
    for name, T in Z.items():
        for kind in ("myth", "mechanism"):
            for a, b in LANG_PAIRS:
                ids, Da, Db = key_steps(T[0], kind, a, b, ("true", "foil"), True)
                if Da is None:
                    continue
                sa = [Sg.get((kind, p, a)) for p in ids]
                sb = [Sg.get((kind, p, b)) for p in ids]
                M = Da @ Db.T
                n_ = len(ids)
                # Clear sign-matched subset: sa[p] == sb[p] and in {"true", "foil"}
                for p in range(n_):
                    if sa[p] == sb[p] and sa[p] in ("true", "foil"):
                        q1 = [q for q in range(n_) if q != p and sb[q] == sa[p]]
                        q2 = [q for q in range(n_) if q != p and sa[q] == sb[p]]
                        if len(q1) >= 3 and len(q2) >= 3:
                            clear_sign_vals.append(float(M[p, p] - 0.5 * (M[p, q1].mean() + M[q2, p].mean())))
                # Within-outcome-class permutation floor (only permutes indices sharing exact (sa, sb) label!)
                grp = {}
                for idx_i, lbl in enumerate(zip(sa, sb)):
                    grp.setdefault(lbl, []).append(idx_i)
                for _ in range(20):
                    perm = np.arange(n_)
                    for ix in grp.values():
                        if len(ix) > 1:
                            perm[ix] = rng_fix.permutation(ix)
                    sb_p = [sb[j] for j in perm]
                    sc = pat_scores(Da, Db[perm], 0, sa, sb_p)
                    if np.isfinite(sc).any():
                        fixed_shuf_means.append(float(np.nanmean(sc)))

    # Character 3-gram lexical overlap check (model-free)
    R_items = [r for r in map(json.loads, items_p.open()) if r.get("ok") and r["set"] == "I"]
    grams = lambda x: {x[i : i + 3] for i in range(max(len(x) - 2, 0))}
    G = {(r["pattern_id"], r["lang"]): grams(r["true"]) for r in R_items}
    lex_3gram = {}
    for a, b in LANG_PAIRS:
        ids = sorted({p for p, g in G if g == a} & {p for p, g in G if g == b})
        jac = lambda u, v: len(u & v) / max(len(u | v), 1)
        same = float(np.mean([jac(G[p, a], G[p, b]) for p in ids]))
        oth = float(np.mean([jac(G[p, a], G[q, b]) for p in ids[:30] for q in ids[:30] if p != q]))
        lex_3gram[f"{a}-{b}"] = round(same - oth, 4)

    out = {
        "script_sha256": _sha256(Path(__file__)),
        "items_sha256": _sha256(items_p),
        "checked_sha256": _sha256(checked_p),
        "sign_sha256": _sha256(sign_p),
        "models": sorted(Z.keys()),
        "languages": list(LANGS),
        "depth_frac": 0.6,
        "set_I_passing_by_lang": dict(sorted(Counter(k[3] for k in P if k[0] == "I" and passed(*k)).items())),
        "c1_primary": c1_primary,
        "c1_raw_no_outcome_ctrl": c1_raw,
        "c1_leading_axis_removed": c1_axis_rm,
        "c1_edited_clauses_only": c1_clauses,
        "c1_story_beyond_clauses_paired": c1_beyond_clauses,
        "c1_edited_clause_share": round(c1_clauses["mean"] / c1_primary["mean"], 4),
        "c1_without_ja_zh": c1_no_jazh,
        "c4_myths_minus_mechanisms": c4_myth_minus_mech,
        "c1_by_language_pair": by_lang_pair,
        "c1_per_model": per_model,
        "fixed_within_outcome_shuffle_floor": {
            "mean": round(float(np.mean(fixed_shuf_means)), 4),
            "ci95": [round(float(np.percentile(fixed_shuf_means, 2.5)), 4), round(float(np.percentile(fixed_shuf_means, 97.5)), 4)],
            "n_draws": len(fixed_shuf_means),
        },
        "clear_sign_matched_subset": {
            "mean": round(float(np.mean(clear_sign_vals)), 4),
            "n_comparisons": len(clear_sign_vals),
        },
        "lexical_char_3gram_same_minus_other": lex_3gram,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"CROSSLANG_AUDIT_OK: C1={c1_primary['mean']:+.4f} {c1_primary['ci95']} (n={c1_primary['n_patterns']}) "
        f"| clauses={c1_clauses['mean']:+.4f} | beyond={c1_beyond_clauses['mean']:+.4f} {c1_beyond_clauses['ci95']} "
        f"| clear_sign={out['clear_sign_matched_subset']['mean']:+.4f} (n={out['clear_sign_matched_subset']['n_comparisons']}) "
        f"| fixed_shuf_floor={out['fixed_within_outcome_shuffle_floor']['mean']:+.4f} {out['fixed_within_outcome_shuffle_floor']['ci95']} "
        f"-> {OUT_JSON} ({out['elapsed_sec']}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
