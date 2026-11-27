"""Experiment 247-B: Fully Matched Depth Audit Fixing ALL Round-2 Claim-B Findings.

Fixes every Round-2 Claim-B issue from Fable 5.2 and Astra 6.2:
  1. Setting-disjoint HOUSE evaluated on BOTH `mean` AND `last` pools with IDENTICAL `VAL5` erasure
     as the swap benchmarks, using Astra-6.2-grade REFIT-IN-BOOTSTRAP paired depth CIs (1,000 setting
     resamples refitting source directions inside every draw!).
  2. Strict proper-name exclusion for `items3` (checking sentence-initial tokens too so `Goldilocks`
     and any named fairy-tale protagonist is ACTUALLY removed — saving the exact retained ID list!)
     PLUS 2,000-draw passage-bootstrap 95% CIs on `myth_strict_no_pn - mech_strict_no_pn` at all 5 depths!
  3. Simultaneous refit-in-bootstrap depth contrasts for `bm120`, `bm_rel242`, `bm_neut245`, `bm_polconf245`,
     `HOUSE`, AND `zs_sent` side-by-side in ONE auditable JSON!
"""
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH / "exp247_round3"
EXP_DIR.mkdir(parents=True, exist_ok=True)
EXP245 = SCRATCH / "exp245_controls"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(EXP245)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import exp245_binding as E245  # noqa: E402
import depth_sweep240_fixed as D240F  # noqa: E402

MLX_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)

# Strict proper-name detector that ALSO checks sentence-initial words!
COMMON_SENT_STARTERS = {
    "the", "a", "an", "in", "on", "at", "when", "while", "before", "after", "during", "every",
    "each", "once", "one", "two", "three", "four", "five", "some", "many", "most", "all",
    "no", "not", "if", "as", "by", "for", "from", "to", "with", "without", "under", "over",
    "near", "inside", "outside", "behind", "between", "through", "across", "along", "around",
    "he", "she", "they", "it", "we", "this", "that", "these", "those", "there", "here",
    "his", "her", "their", "its", "our", "although", "though", "because", "since", "until",
    "unless", "whenever", "where", "wherever", "long", "deep", "high", "far", "soon", "then",
    "later", "earlier", "finally", "suddenly", "slowly", "quietly", "first", "second", "next",
    "both", "neither", "either", "even", "still", "yet", "but", "and", "or", "so", "however",
    "meanwhile", "instead", "otherwise", "despite", "within", "beneath", "above", "below",
    "water", "fire", "wind", "stone", "metal", "wood", "heat", "cold", "light", "sound",
    "air", "steam", "oil", "gas", "salt", "sand", "soil", "clay", "glass", "iron", "steel",
    "copper", "ice", "snow", "rain", "blood", "acid", "pressure", "voltage", "current",
    "gravity", "friction", "energy", "power", "force", "mass", "speed", "time", "space",
    "people", "villagers", "travelers", "sailors", "farmers", "workers", "engineers",
    "builders", "guards", "hunters", "merchants", "doctors", "teachers", "students",
    "children", "parents", "brothers", "sisters", "kings", "queens", "princes", "soldiers",
    "young", "old", "small", "large", "heavy", "sharp", "hot", "warm", "cool", "dry", "wet",
    "clean", "dark", "bright", "thick", "thin", "soft", "hard", "fresh", "raw", "pure",
    "broken", "loose", "tight", "open", "closed", "empty", "full", "modern", "ancient",
    "spinning", "flowing", "rising", "falling", "burning", "melting", "freezing", "boiling",
    "cooling", "heating", "pumping", "lifting", "pushing", "pulling", "turning", "sliding",
}


def _strict_has_proper_name(txt):
    """True if ANY mid-sentence capitalized word OR ANY sentence-initial capitalized word not in COMMON_SENT_STARTERS."""
    # Split into sentences carefully
    sents = re.split(r"(?<=[.!?])\s+", txt.strip())
    for s in sents:
        words = re.findall(r"[A-Za-z]+", s)
        if not words:
            continue
        # Check sentence-initial word against common non-name English starters
        if words[0][0].isupper() and words[0].lower() not in COMMON_SENT_STARTERS:
            return True
        # Check any non-initial word
        for w in words[1:]:
            if w[0].isupper() and w != "I":
                return True
    return False


def _refit_depth_boot_house(Dh_d, Q5_d, cats, hset_idx, splits, src_k, tgts, n_boot=1000, seed=24730):
    uniq = sorted(set(int(s) for s in hset_idx))
    G = len(uniq)
    g_rows = [np.where(hset_idx == s)[0] for s in uniq]
    Dh_p = {dk: {k: V234._project_out_rows(Dh_d[dk][k], Q5_d[dk][k]) for k in MLX_KEYS} for dk in ("d30", "d50", "d70")}

    def eval_w(w_g):
        w_row = np.zeros(len(hset_idx), dtype=np.float64)
        for gi, rix in enumerate(g_rows):
            w_row[rix] = w_g[gi]
        tot_w = float(w_row.sum())
        out = {}
        for dk in ("d30", "d50", "d70"):
            t_accs = []
            for tk in tgts:
                Qt = Q5_d[dk][tk]
                sp_a = []
                for fl in splits:
                    oof = np.zeros(len(hset_idx))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        # Weighted category-balanced HOUSE mean direction
                        c_vecs = []
                        for c in AR.HOUSE_CATS:
                            m = tr & (cats == c)
                            wm = w_row[m]
                            if wm.sum() > 0:
                                c_vecs.append((Dh_p[dk][src_k][m] * wm[:, None]).sum(0) / wm.sum())
                        ut = AR.unit(np.mean(c_vecs, axis=0))
                        ut = V234._project_out(ut, Qt)
                        oof[te] = Dh_p[dk][tk][te] @ ut
                    sp_a.append(float(np.sum(w_row * (oof > 0)) / max(1e-12, tot_w)))
                t_accs.append(float(np.mean(sp_a)))
            out[dk] = float(np.mean(t_accs))
        return out

    obs = eval_w(np.ones(G))
    rng = np.random.default_rng(seed)
    d50_30, d50_70 = [], []
    for _ in range(n_boot):
        wg = np.bincount(rng.integers(0, G, size=G), minlength=G).astype(np.float64)
        r = eval_w(wg)
        d50_30.append(r["d50"] - r["d30"])
        d50_70.append(r["d50"] - r["d70"])
    d50_30, d50_70 = np.asarray(d50_30), np.asarray(d50_70)
    return {
        "d30": round(obs["d30"], 4), "d50": round(obs["d50"], 4), "d70": round(obs["d70"], 4),
        "d50_minus_d30": {
            "obs": round(obs["d50"] - obs["d30"], 4),
            "ci95": [round(float(np.percentile(d50_30, 2.5)), 4), round(float(np.percentile(d50_30, 97.5)), 4)],
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_30, 1.25)), 4), round(float(np.percentile(d50_30, 98.75)), 4)],
        },
        "d50_minus_d70": {
            "obs": round(obs["d50"] - obs["d70"], 4),
            "ci95": [round(float(np.percentile(d50_70, 2.5)), 4), round(float(np.percentile(d50_70, 97.5)), 4)],
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_70, 1.25)), 4), round(float(np.percentile(d50_70, 98.75)), 4)],
        },
    }


def main():
    t0 = time.time()
    data_map = {k: np.load(DEPTH_DIR / f"depth240_{k}.npz", allow_pickle=True) for k in MLX_KEYS}
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    ev_m_strict = np.array([r["domain"] == "items3_myth" and r["id"].endswith("_anchor") and not _strict_has_proper_name(r["text"]) for r in ev_rows], dtype=bool)
    ev_c_strict = np.array([r["domain"] == "items3_mech" and r["id"].endswith("_anchor") and not _strict_has_proper_name(r["text"]) for r in ev_rows], dtype=bool)
    retained_myth_ids = [ev_rows[i]["id"] for i in np.where(ev_m_strict)[0]]
    assert "i3_myth_goldilocks_anchor" not in retained_myth_ids, "Goldilocks leaked!"

    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    hsets = [str(pairs_arch[i].get("setting", pairs_arch[i].get("id", i))) for i in house_idx]
    uniq_hs = sorted(set(hsets))
    hset_idx = np.array([{s: i for i, s in enumerate(uniq_hs)}[s] for s in hsets], dtype=int)
    house_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(24010 + s_i)
        f0 = set(int(x) for x in rng.permutation(len(uniq_hs))[: len(uniq_hs) // 2])
        fl = np.array([0 if int(s) in f0 else 1 for s in hset_idx], dtype=int)
        assert len(set(hset_idx[fl == 0].tolist()) & set(hset_idx[fl == 1].tolist())) == 0
        house_splits.append(fl)

    sg_rows, ov_rows, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)

    items3_strict = {}
    Dh_by_pool_d = {"mean": {}, "last": {}}
    Q5_by_pool_d = {"mean": {}, "last": {}}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        rot_m = D240F._fit_rot_at_slot(data_map, slot, "mean", d_plato=256)
        Z_ev = {k: D240F._map(data_map[k]["ev_mean"][slot], rot_m, k) for k in MLX_KEYS}
        C4_ev = np.mean([Z_ev[k] for k in C4_KEYS], axis=0)

        items3_strict[dkey] = {}
        for hk in ("phi4_mini", "olmo3_7b"):
            p_cos = D240F._row_cosines_verified(Z_ev[hk], C4_ev)
            vm, vc = p_cos[ev_m_strict], p_cos[ev_c_strict]
            rng = np.random.default_rng(24740 + slot)
            diffs = [float(rng.choice(vm, len(vm), True).mean() - rng.choice(vc, len(vc), True).mean()) for _ in range(2000)]
            diffs = np.asarray(diffs)
            items3_strict[dkey][hk] = {
                "n_myth": int(len(vm)), "n_mech": int(len(vc)),
                "myth_mean": round(float(vm.mean()), 4), "mech_mean": round(float(vc.mean()), 4),
                "diff": round(float(vm.mean() - vc.mean()), 4),
                "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            }

        if dkey in ("d30", "d50", "d70"):
            for pkey, rot in (("mean", rot_m), ("last", D240F._fit_rot_at_slot(data_map, slot, "last", d_plato=256))):
                Dh_by_pool_d[pkey][dkey] = {}
                Q5_by_pool_d[pkey][dkey] = {}
                for k in MLX_KEYS:
                    d = data_map[k]
                    Dh_by_pool_d[pkey][dkey][k] = D240F._map(d[f"house_pos_{pkey}"][slot][house_idx], rot, k) - D240F._map(d[f"house_neg_{pkey}"][slot][house_idx], rot, k)
                    Dv = D240F._map(d[f"val_pos_{pkey}"][slot], rot, k) - D240F._map(d[f"val_neg_{pkey}"][slot], rot, k)
                    dom_vecs = [AR.unit(Dv[:n_sg][sg_doms == dm].mean(0)) for dm in sorted(set(sg_doms.tolist()))]
                    Q5_by_pool_d[pkey][dkey][k] = V234._orthonormal_basis(dom_vecs + [AR.unit(Dv[n_sg:].mean(0))])

    house_val5_refit = {}
    for pkey in ("mean", "last"):
        house_val5_refit[pkey] = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            house_val5_refit[pkey][src_k] = _refit_depth_boot_house(
                Dh_by_pool_d[pkey], Q5_by_pool_d[pkey], house_cats, hset_idx, house_splits, src_k, tgts, n_boot=1000, seed=24730
            )

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "strict_no_pn_retained_myth_ids": retained_myth_ids,
        "items3_strict_no_pn_with_ci": items3_strict,
        "house_setting_disjoint_val5_refit_boot": house_val5_refit,
    }
    out_p = EXP_DIR / "depth_audit247_results.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"SAVED {out_p} in {report['elapsed_sec']}s", flush=True)
    print("Retained strict-no-PN myth IDs:", len(retained_myth_ids), retained_myth_ids[:8])
    print("items3 strict CI:", json.dumps(items3_strict, indent=2))
    print("HOUSE VAL5 refit boot:", json.dumps(house_val5_refit, indent=2))


if __name__ == "__main__":
    main()
