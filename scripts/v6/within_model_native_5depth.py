"""Native Full-Dimensional Within-Model 5-Depth Audit (ZERO Neutral PCA, ZERO Procrustes).

Verifies that the 50%-depth maximum holds directly in each model's own raw full-dimensional
hidden space (cv2 perp native VAL5, 20 setting-disjoint splits) — completely independent
of both Procrustes alignment AND 256-D neutral-text PCA truncation!
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP4 = SCRATCH / "exp248_round4"
EXP3 = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
EXP242 = SCRATCH / "exp240_242"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(EXP245), str(EXP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import exp245_binding as E245  # noqa: E402
import exp247_actor_binding as E247  # noqa: E402

MLX_KEYS = E247.MLX_KEYS
DEPTH_FRACS = E247.DEPTH_FRACS


def main():
    t0 = time.time()
    benches = {
        "bm_actor2x2_247": json.loads((EXP3 / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"],
        "bm120": BM238.load_mirror_data(),
        "bm_rel242": json.loads((EXP242 / "bm_rel242_data.json").read_text())["bm_rel_120"],
        "bm_neut245": json.loads((EXP245 / "bm_neut245_data.json").read_text())["bm_neut_120"],
        "bm_polconf245": json.loads((EXP245 / "bm_polconf245_data.json").read_text())["bm_polconf_120"],
    }
    sid_map, split_map = {}, {}
    for bn, rows in benches.items():
        sb = {"bm120": 23800, "bm_rel242": 24200, "bm_neut245": 24500, "bm_polconf245": 24600, "bm_actor2x2_247": 24700}[bn]
        sid_map[bn], split_map[bn] = E245._sid_and_splits(rows, sb)

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
        house_splits.append(np.array([0 if int(s) in f0 else 1 for s in hset_idx], dtype=int))

    sg_rows, _, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)

    # Pre-split setting indices for bm_actor2x2_247 refit-in-bootstrap
    sid_a = sid_map["bm_actor2x2_247"]
    spl_a = split_map["bm_actor2x2_247"]
    G_a = len(set(int(s) for s in sid_a))
    by_s = [np.where(sid_a == s)[0] for s in range(G_a)]
    split_s_fold = np.array([[int(fl[by_s[g][0]]) for g in range(G_a)] for fl in spl_a], dtype=int)

    def _boot_native(Dp, boot_w_mat):
        B = boot_w_mat.shape[0]
        S = np.stack([Dp[by_s[g]].sum(0) for g in range(G_a)], axis=0)
        s_of_i = sid_a.astype(int)
        tot_hits = np.zeros(B, dtype=np.float64)
        for si, fl in enumerate(spl_a):
            sf = split_s_fold[si]
            for f in (0, 1):
                tr_g = (sf != f).astype(np.float64)
                te_i = np.where(fl == f)[0]
                te_s = s_of_i[te_i]
                U = (boot_w_mat * tr_g[None, :]) @ S
                U /= np.maximum(1e-12, np.linalg.norm(U, axis=1, keepdims=True))
                scores = U @ Dp[te_i].T
                tot_hits += ((scores > 0).astype(np.float64) * boot_w_mat[:, te_s]).sum(axis=1)
        return tot_hits / (boot_w_mat.sum(axis=1) * 4.0 * len(spl_a))

    rng_b = np.random.default_rng(24899)
    boot_w = np.array([np.bincount(rng_b.integers(0, G_a, size=G_a), minlength=G_a) for _ in range(1000)], dtype=np.float64)
    ones_w = np.ones((1, G_a), dtype=np.float64)

    native_curves = {pool: {bn: {} for bn in list(benches.keys()) + ["house_disjoint"]} for pool in ("mean", "last")}
    actor_native_boot = {pool: {} for pool in ("mean", "last")}
    native_dims = {}

    for pool_key in ("mean", "last"):
        wm_obs = {dk: [] for dk in ("d30", "d50", "d70")}
        wm_draws = {dk: [] for dk in ("d30", "d50", "d70")}
        for slot, df in enumerate(DEPTH_FRACS):
            dkey = f"d{int(round(df * 100))}"
            for bn in list(benches.keys()) + ["house_disjoint"]:
                native_curves[pool_key][bn][dkey] = {"per_model": {}}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                Dv = d240[f"val_pos_{pool_key}"][slot].astype(np.float64) - d240[f"val_neg_{pool_key}"][slot].astype(np.float64)
                native_dims[k] = int(Dv.shape[1])
                dom_vecs = [AR.unit(Dv[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
                Q5 = V234._orthonormal_basis(dom_vecs + [AR.unit(Dv[n_sg:].mean(0))])

                # Swap benchmarks
                for bn, pref in [("bm120", None), ("bm_rel242", "bmrel242"), ("bm_neut245", "bmneut245"), ("bm_polconf245", "bmpolconf245"), ("bm_actor2x2_247", "bmactor247")]:
                    if pref is None:
                        D = d240[f"bm120_pos_{pool_key}"][slot].astype(np.float64) - d240[f"bm120_neg_{pool_key}"][slot].astype(np.float64)
                    else:
                        dr = np.load(DEPTH_DIR / f"{pref}_{k}.npz")
                        D = dr[f"pos_{pool_key}"][slot].astype(np.float64) - dr[f"neg_{pool_key}"][slot].astype(np.float64)
                    Dp = V234._project_out_rows(D, Q5)
                    sp_a = []
                    for fl in split_map[bn]:
                        oof = np.zeros(len(Dp))
                        for f in (0, 1):
                            tr, te = (fl != f), (fl == f)
                            ut = AR.unit(Dp[tr].mean(0))
                            oof[te] = Dp[te] @ ut
                        sp_a.append(float(np.mean(oof > 0)))
                    native_curves[pool_key][bn][dkey]["per_model"][k] = round(float(np.mean(sp_a)), 4)
                    if bn == "bm_actor2x2_247" and dkey in ("d30", "d50", "d70"):
                        wm_obs[dkey].append(float(_boot_native(Dp, ones_w)[0]))
                        wm_draws[dkey].append(_boot_native(Dp, boot_w))

                # HOUSE setting-disjoint
                Dh = d240[f"house_pos_{pool_key}"][slot][house_idx].astype(np.float64) - d240[f"house_neg_{pool_key}"][slot][house_idx].astype(np.float64)
                Dhp = V234._project_out_rows(Dh, Q5)
                sp_h = []
                for fl in house_splits:
                    oof = np.zeros(len(Dhp))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        ut = AR.house_dir(Dhp[tr], house_cats[tr])
                        oof[te] = Dhp[te] @ ut
                    sp_h.append(float(np.mean(oof > 0)))
                native_curves[pool_key]["house_disjoint"][dkey]["per_model"][k] = round(float(np.mean(sp_h)), 4)

            for bn in list(benches.keys()) + ["house_disjoint"]:
                vals = list(native_curves[pool_key][bn][dkey]["per_model"].values())
                native_curves[pool_key][bn][dkey]["mean_6models"] = round(float(np.mean(vals)), 4)

        obs_m = {dk: float(np.mean(wm_obs[dk])) for dk in ("d30", "d50", "d70")}
        dr_m = {dk: np.mean(wm_draws[dk], axis=0) for dk in ("d30", "d50", "d70")}
        d50_30 = dr_m["d50"] - dr_m["d30"]
        d50_70 = dr_m["d50"] - dr_m["d70"]
        actor_native_boot[pool_key] = {
            "d30": round(obs_m["d30"], 4), "d50": round(obs_m["d50"], 4), "d70": round(obs_m["d70"], 4),
            "d50_minus_d30": {
                "obs": round(obs_m["d50"] - obs_m["d30"], 4),
                "ci95": [round(float(np.percentile(d50_30, 2.5)), 4), round(float(np.percentile(d50_30, 97.5)), 4)],
                "ci95_bonf2": [round(float(np.percentile(d50_30, 1.25)), 4), round(float(np.percentile(d50_30, 98.75)), 4)],
            },
            "d50_minus_d70": {
                "obs": round(obs_m["d50"] - obs_m["d70"], 4),
                "ci95": [round(float(np.percentile(d50_70, 2.5)), 4), round(float(np.percentile(d50_70, 97.5)), 4)],
                "ci95_bonf2": [round(float(np.percentile(d50_70, 1.25)), 4), round(float(np.percentile(d50_70, 98.75)), 4)],
            },
        }

    out = {
        "elapsed_sec": round(time.time() - t0, 1),
        "native_hidden_dims": native_dims,
        "within_model_native_5depth": native_curves,
        "actor2x2_native_within_model_bonf2": actor_native_boot,
    }
    p = EXP4 / "within_model_native_5depth.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"SAVED {p} in {out['elapsed_sec']}s", flush=True)
    for pool in ("mean", "last"):
        print(f"NATIVE FULL-DIM 6-model mean [{pool}]:")
        for bn in ("bm_actor2x2_247", "bm120", "bm_rel242", "bm_neut245", "bm_polconf245", "house_disjoint"):
            cs = [native_curves[pool][bn][dk]["mean_6models"] for dk in ("d15", "d30", "d50", "d70", "d85")]
            print(f"  {bn:16s}: {cs}")
        print(f"  actor2x2 bonf2 [{pool}]:", actor_native_boot[pool])


if __name__ == "__main__":
    main()
