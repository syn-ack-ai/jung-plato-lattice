"""Run the Full 5-Seed Paired Causal Ladder on Untouched 3rd-Party Benchmarks (Exp 257).

Executes on `Qwen/Qwen3-0.6B` (exact fp32) across 5 fresh seeds `{101, 102, 103, 104, 105}`
(ZERO overlap with Exp 256 tuning seed 256):
  - `base` (unmodified Qwen3-0.6B fp32)
  - `kl_only` (seed 101 sanity check: zero geometry weight -> verifies ~0 drift)
  - `true_stratified_r3` (5 seeds; stratified 400/400/400 PCA frame, ce_w=0)
  - `random_lora_matched` (5 seeds; matched to exact per-layer ||ΔW_l||_F of true_stratified_r3)
  - `shuffle_within_domain` (5 seeds; within-(domain, segment_index) permutation + refit R*)
  - `true_gut_only` (5 seeds; audited Gutenberg-only narrative consensus)
  - `true_dual_domain` (5 seeds; 75% stratified R3 prose + 25% 8B Gemma 4 stdlib code)

Persists every single run's full per-item JSON + gzipped raw generations incrementally
(resumable!) and computes exact `ddof=1` SDs, paired t-intervals across seeds, and
10,000-resample paired bootstrap 95% CIs over items/articles/repos.
"""
import copy
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_rigorous as ev  # noqa: E402
import recipe_rigorous as rec  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]
# T-critical values (two-sided 95% CI) for df = 1..10
T_CRIT_95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469}


def sample_stats(vals):
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t_c = T_CRIT_95.get(n - 1, 1.96) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 6),
        "sd_ddof1": round(sd, 6),
        "ci95_lo": round(mean - t_c * se, 6),
        "ci95_hi": round(mean + t_c * se, 6),
        "values": [round(float(x), 6) for x in arr],
    }


def paired_seed_contrast(vals_a, vals_b):
    diffs = np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64)
    st = sample_stats(diffs)
    n = len(diffs)
    if n > 1 and st["sd_ddof1"] > 0:
        t_stat = float(diffs.mean() / (diffs.std(ddof=1) / math.sqrt(n)))
    else:
        t_stat = 0.0
    st["t_stat"] = round(t_stat, 4)
    st["df"] = max(0, n - 1)
    st["excludes_zero"] = bool(st["ci95_lo"] > 0.0 or st["ci95_hi"] < 0.0)
    return st


def cluster_bootstrap_diff(diffs_by_cluster, n_boot=10000, seed=257):
    """Paired cluster bootstrap (over articles, repos, passages, or tasks) averaged across seeds."""
    rng = np.random.default_rng(seed)
    # diffs_by_cluster: list of arrays/floats, one per cluster (or (n_clusters,) mean diff)
    arr = np.asarray(diffs_by_cluster, dtype=np.float64)
    m = len(arr)
    idx = rng.integers(0, m, size=(n_boot, m))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
    return {
        "n_clusters": m,
        "mean_diff": round(float(arr.mean()), 6),
        "boot_ci95_lo": round(float(lo), 6),
        "boot_ci95_hi": round(float(hi), 6),
        "boot_p_two_sided": round(min(1.0, p_two), 5),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def load_base_sd(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_base_sd(model, base_sd_cpu):
    sd = model.state_dict()
    for k, v in base_sd_cpu.items():
        sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
    model.eval()


def summarize_results(runs_dir: Path, out_path: Path):
    base = json.loads((runs_dir / "base.json").read_text())
    bsum = base["summary"]
    arms = ["true_stratified_r3", "true_gut_only", "true_dual_domain", "shuffle_within_domain", "random_lora_matched"]

    loaded = {}
    for arm in arms:
        loaded[arm] = []
        for s in SEEDS:
            p = runs_dir / f"{arm}_seed{s}.json"
            if p.exists():
                loaded[arm].append(json.loads(p.read_text()))

    kl_p = runs_dir / "kl_only_seed101.json"
    kl_only = json.loads(kl_p.read_text()) if kl_p.exists() else None

    metrics = [
        "wikitext2_nll", "held_gut_nll",
        "humaneval_pass1", "mbpp_test_pass1",
        "swe_full_acc", "swe_full_margin", "swe_edit_acc", "swe_edit_margin",
        "actor_acc_240", "actor_margin_240", "actor_mirror_consistent",
        "proc_acc_200",
        "blind_30_r2_frob", "blind_30_r2_opt", "blind_30_cos",
        "blind_50_r2_frob", "blind_50_r2_opt", "blind_50_cos",
    ]

    arm_tables = {}
    for arm in arms:
        recs = loaded[arm]
        if not recs:
            continue
        tbl = {"n_seeds": len(recs), "abs": {}, "delta_vs_base": {}}
        for m in metrics:
            vals = [r["summary"][m] for r in recs]
            tbl["abs"][m] = sample_stats(vals)
            tbl["delta_vs_base"][m] = sample_stats([v - bsum[m] for v in vals])
        tbl["total_fro_norm"] = sample_stats([r["train_meta"]["total_fro_norm"] for r in recs])
        if "merge_kl_fp32" in recs[0]["train_meta"]:
            tbl["merge_kl_fp32_max"] = max(r["train_meta"]["merge_kl_fp32"] for r in recs)
        arm_tables[arm] = tbl

    # Paired seed contrasts (True arms vs Controls!)
    contrasts = {}
    pairs = [
        ("true_stratified_r3", "shuffle_within_domain"),
        ("true_stratified_r3", "random_lora_matched"),
        ("true_gut_only", "shuffle_within_domain"),
        ("true_gut_only", "random_lora_matched"),
        ("true_dual_domain", "shuffle_within_domain"),
        ("true_dual_domain", "random_lora_matched"),
        ("true_dual_domain", "true_stratified_r3"),
    ]
    for a, b in pairs:
        if len(loaded.get(a, [])) == len(SEEDS) and len(loaded.get(b, [])) == len(SEEDS):
            c_key = f"{a}_minus_{b}"
            contrasts[c_key] = {}
            for m in metrics:
                va = [r["summary"][m] for r in loaded[a]]
                vb = [r["summary"][m] for r in loaded[b]]
                contrasts[c_key][m] = paired_seed_contrast(va, vb)

            # Item/cluster-level paired bootstrap (averaging seed-level scores per item/cluster!)
            boot = {}
            # 1. WikiText-2 articles (62 clusters)
            wt_a = np.mean([[x["nll"] for x in r["wikitext2_per_article"]] for r in loaded[a]], axis=0)
            wt_b = np.mean([[x["nll"] for x in r["wikitext2_per_article"]] for r in loaded[b]], axis=0)
            boot["wikitext2_article_nll"] = cluster_bootstrap_diff(wt_a - wt_b)
            # 2. Held-out Gutenberg passages (100 clusters)
            gt_a = np.mean([[x["nll"] for x in r["held_gut_per_passage"]] for r in loaded[a]], axis=0)
            gt_b = np.mean([[x["nll"] for x in r["held_gut_per_passage"]] for r in loaded[b]], axis=0)
            boot["held_gut_passage_nll"] = cluster_bootstrap_diff(gt_a - gt_b)
            # 3. HumanEval tasks (164 tasks)
            he_keys = sorted(loaded[a][0]["humaneval_per_task_bool"].keys())
            he_a = np.mean([[float(r["humaneval_per_task_bool"][k]) for k in he_keys] for r in loaded[a]], axis=0)
            he_b = np.mean([[float(r["humaneval_per_task_bool"][k]) for k in he_keys] for r in loaded[b]], axis=0)
            boot["humaneval_task_pass1"] = cluster_bootstrap_diff(he_a - he_b)
            # 4. MBPP test tasks (257 tasks)
            mb_keys = sorted(loaded[a][0]["mbpp_per_task_bool"].keys(), key=int)
            mb_a = np.mean([[float(r["mbpp_per_task_bool"][k]) for k in mb_keys] for r in loaded[a]], axis=0)
            mb_b = np.mean([[float(r["mbpp_per_task_bool"][k]) for k in mb_keys] for r in loaded[b]], axis=0)
            boot["mbpp_task_pass1"] = cluster_bootstrap_diff(mb_a - mb_b)
            # 5. SWE-Smith repo-clustered full_margin (89 repos)
            repos = sorted(set(x["repo"] for x in loaded[a][0]["swe_per_pair"]))
            repo_diffs = []
            for rp in repos:
                idxs = [i for i, x in enumerate(loaded[a][0]["swe_per_pair"]) if x["repo"] == rp]
                ra = np.mean([loaded[a][s]["swe_per_pair"][i]["full_margin"] for s in range(len(SEEDS)) for i in idxs])
                rb = np.mean([loaded[b][s]["swe_per_pair"][i]["full_margin"] for s in range(len(SEEDS)) for i in idxs])
                repo_diffs.append(ra - rb)
            boot["swe_repo_full_margin"] = cluster_bootstrap_diff(repo_diffs)
            contrasts[c_key]["cluster_bootstrap"] = boot

    out = {
        "evaluator_sha256": bsum["evaluator_sha256"],
        "model_id": ev.MODEL_ID,
        "seeds": SEEDS,
        "base": bsum,
        "kl_only_seed101": (
            {
                "summary": kl_only["summary"],
                "delta_vs_base": {m: round(kl_only["summary"][m] - bsum[m], 6) for m in metrics},
                "total_fro_norm": kl_only["train_meta"]["total_fro_norm"],
                "merge_kl_fp32": kl_only["train_meta"]["merge_kl_fp32"],
            }
            if kl_only else None
        ),
        "arms": arm_tables,
        "paired_contrasts": contrasts,
    }
    out_path.write_text(json.dumps(out, indent=1))
    return out


def main():
    paths = ev.default_paths()
    res_dir = paths["exp_dir"] / "results"
    runs_dir = res_dir / "runs"
    gens_dir = res_dir / "generations"
    runs_dir.mkdir(parents=True, exist_ok=True)
    gens_dir.mkdir(parents=True, exist_ok=True)

    d_t = np.load(paths["targets_npz"], allow_pickle=True)
    tok = AutoTokenizer.from_pretrained(ev.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(ev.MODEL_ID, dtype=torch.float32, device_map="cuda:0")
    model.eval()
    base_sd_cpu = load_base_sd(model)

    # 0. Base reference
    base_path = runs_dir / "base.json"
    if not base_path.exists():
        print("[EXP257] Running Base Qwen3-0.6B (exact fp32)...", flush=True)
        recs, gens = ev.evaluate_all(model, tok, d_t, paths)
        base_path.write_text(json.dumps(recs, indent=1))
        with gzip.open(gens_dir / "base.json.gz", "wt") as f:
            json.dump(gens, f)
        print("[EXP257] BASE:", json.dumps(recs["summary"]), flush=True)
    else:
        print("[EXP257] Cached Base loaded.", flush=True)

    # 1. KL-Only sanity check (seed 101)
    kl_path = runs_dir / "kl_only_seed101.json"
    if not kl_path.exists():
        print("[EXP257] Running kl_only seed=101...", flush=True)
        restore_base_sd(model, base_sd_cpu)
        tm = rec.run_arm(model, tok, d_t, arm="kl_only", seed=101)
        recs, gens = ev.evaluate_all(model, tok, d_t, paths)
        recs["train_meta"] = tm
        kl_path.write_text(json.dumps(recs, indent=1))
        with gzip.open(gens_dir / "kl_only_seed101.json.gz", "wt") as f:
            json.dump(gens, f)
        print("[EXP257] kl_only:", json.dumps({"fro": tm["total_fro_norm"], "wt2": recs["summary"]["wikitext2_nll"], "he": recs["summary"]["humaneval_hits"], "mbpp": recs["summary"]["mbpp_test_hits"]}), flush=True)

    # 2. Full 5-seed paired causal ladder!
    arm_order = [
        "true_stratified_r3",
        "random_lora_matched",
        "shuffle_within_domain",
        "true_gut_only",
        "true_dual_domain",
    ]
    for seed in SEEDS:
        for arm in arm_order:
            run_path = runs_dir / f"{arm}_seed{seed}.json"
            if run_path.exists():
                print(f"[EXP257] Cached {arm} seed={seed}.", flush=True)
                continue
            print(f"[EXP257] Running {arm} seed={seed}...", flush=True)
            restore_base_sd(model, base_sd_cpu)
            matched_norms = None
            if arm == "random_lora_matched":
                ref_r3 = json.loads((runs_dir / f"true_stratified_r3_seed{seed}.json").read_text())
                matched_norms = ref_r3["train_meta"]["delta_fro_norms"]
            tm = rec.run_arm(
                model, tok, d_t, arm=arm, seed=seed,
                code_targets_npz=paths["code_targets_npz"],
                matched_fro_norms=matched_norms,
            )
            recs, gens = ev.evaluate_all(model, tok, d_t, paths)
            recs["train_meta"] = tm
            run_path.write_text(json.dumps(recs, indent=1))
            with gzip.open(gens_dir / f"{arm}_seed{seed}.json.gz", "wt") as f:
                json.dump(gens, f)
            s_out = recs["summary"]
            print(
                f"[EXP257] DONE {arm} s={seed} | wt2={s_out['wikitext2_nll']:.5f} gut={s_out['held_gut_nll']:.5f} "
                f"HE={s_out['humaneval_hits']} MBPP={s_out['mbpp_test_hits']} "
                f"SWE={s_out['swe_full_acc']:.4f}({s_out['swe_full_margin']:+.4f}) "
                f"Act={s_out['actor_acc_240']:.4f} Proc={s_out['proc_acc_200']:.4f} "
                f"L8={s_out['blind_30_r2_frob']:.4f} L14={s_out['blind_50_r2_frob']:.4f} "
                f"fro={tm['total_fro_norm']:.3f} t={tm['train_sec']}+{s_out['eval_sec']}s",
                flush=True,
            )
        # Update rolling summary after each completed seed!
        summarize_results(runs_dir, res_dir / "causal_ladder_summary.json")

    final_out = summarize_results(runs_dir, res_dir / "causal_ladder_summary.json")
    print("[EXP257] ALL_DONE! Summary written to results/causal_ladder_summary.json", flush=True)
    print(json.dumps({
        "base": final_out["base"],
        "arms_delta_vs_base_means": {
            arm: {k: v["mean"] for k, v in tbl["delta_vs_base"].items()}
            for arm, tbl in final_out["arms"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
