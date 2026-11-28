"""Locked-Vault Verification Stage A (Pillar 1 on HumanEval 164 + MBPP Test 257) &
Stage B (Pillar 3 5-Seed Verification across Seeds 101..105).

Runs SOLO on <GPU_HOST> RTX 4090 (~22 minutes total!).
UNLOCKING THE VAULT FOR THE FIRST & ONLY TIME AFTER ALL 24 AUTORESEARCH ITERATIONS ARE FROZEN!
  - Stage A (Pillar 1): Uses ALREADY-FROZEN `cache/best_pillar1_compass.npz` (fit strictly on
    `code_train.jsonl`, ZERO vault touch!) + frozen `Own-0.6B Native Probe` (fit strictly on
    `code_train.jsonl`) to score `Greedy + K=8` exact FP32 `Qwen3-0.6B` candidates on:
      1. LOCKED `HumanEval (164)` (`bwrap` sandbox)
      2. LOCKED `MBPP Sanitized Test (257)` (`task_id 11..510`, `bwrap` sandbox)
    Reports exact pass@1 and 10,000-resample paired bootstraps vs Greedy, Self-LogProb,
    Self-Consistency, 0.6B Self-Audit, Own-0.6B Native Probe, and Shuffled-Map Control!
  - Stage B (Pillar 3): Runs 5 paired seeds `{101, 102, 103, 104, 105}` of 4 matched
    from-scratch `30.53M` causal LMs (`scratch_ce_only`, `scratch_31b_gpa_delta_decay`,
    `scratch_31b_gcca_both`, `scratch_shuf_ctrl`), computing exact `ddof=1` SDs, paired
    `df=4` 95% t-CIs, and t-stats on `code_eval_nll`, OOD `wikitext2_nll`, and blind `Phi-4 + OLMo 3`!
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import json
import math
from pathlib import Path
import secrets
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
VAULT_OUT = ROOT / "results_vault"
VAULT_OUT.mkdir(parents=True, exist_ok=True)

EXP258_DIR = ROOT.parent / "exp258_code_lattice"
EXP257_DIR = ROOT.parent / "exp257_rigorous"
for p in (str(ROOT), str(EXP258_DIR), str(EXP257_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_rigorous as ev257  # noqa: E402
import pillar1_compass as p1  # noqa: E402
import run_dev_code_cuda as base258  # noqa: E402
import sample_k8_and_extract_06b_cuda as p1s  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]
T_CRIT_95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764}
MODEL_ID = "Qwen/Qwen3-0.6B"


def sample_stats(vals):
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    tc = T_CRIT_95.get(n - 1, 1.96)
    return {"mean": round(m, 6), "sd": round(sd, 6), "ci95_lo": round(m - tc * se, 6), "ci95_hi": round(m + tc * se, 6),
            "values": [round(float(x), 6) for x in arr]}


def paired_t(va, vb):
    diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
    st = sample_stats(diff)
    sd = diff.std(ddof=1)
    st["t_stat"] = round(float(diff.mean() / (sd / math.sqrt(len(diff)))), 4) if sd > 0 else 0.0
    st["excludes_zero"] = bool(st["ci95_lo"] > 0.0 or st["ci95_hi"] < 0.0)
    return st


def grade_humaneval_one(task: dict, raw_gen: str):
    code = base258.extract_python_code(raw_gen)
    # Handle either full def or bare body completion
    if f"def {task['entry_point']}" in code:
        full_src = code
    else:
        full_src = task["prompt"] + "\n" + code
    sent = f"__VAULT_HE_{secrets.token_hex(8)}__"
    script = f"{full_src}\n\n{task['test']}\ncheck({task['entry_point']})\nprint({sent!r})\n"
    rc, out, _ = base258.run_sandbox(script, timeout_s=4.0)
    return {"task_id": task["task_id"], "ok": bool(rc == 0 and sent in out.splitlines())}


def score_frozen_compass_np(H_by_depth: dict, compass_d: dict, shuffled: bool = False):
    cfg = json.loads(str(compass_d["cfg_json"].item()))
    depths = tuple(cfg["depths"])
    feat_mode = cfg["feat_mode"]
    Z_by_d = {}
    for dep in depths:
        H = H_by_depth[dep].astype(np.float64)  # (N, 6, 1024)
        Z = compass_d[f"mb_{dep}"].astype(np.float64) + (H - compass_d[f"ma_{dep}"].astype(np.float64)) @ compass_d[f"M_{dep}"].astype(np.float64)
        Z_by_d[dep] = Z
    F_mat = p1.extract_trajectory_features(Z_by_d, depths, feat_mode)
    sc = F_mat @ compass_d["w"].astype(np.float64)
    if shuffled:
        rng = np.random.default_rng(26399)
        sc = rng.permutation(sc)
    return sc


def fit_frozen_own06b_probe():
    """Fit Own-0.6B Native Probe STRICTLY on `verifier_pairs_code.jsonl` train split (same as Pillar 1!)."""
    sp = dict(np.load(CACHE / "p1_student_qwen06b_pairs.npz", allow_pickle=True))
    tr_mask = ~sp["is_cal"].astype(bool)
    F_tr = p1.extract_trajectory_features({d: sp[f"h_{d}"].astype(np.float64) for d in p1s.DEPTHS}, p1s.DEPTHS, "audit_plus_code")
    v = p1.fit_pairwise_verifier([F_tr[tr_mask]], sp["pair_ids"][tr_mask], sp["labels"][tr_mask], clf_type="pairwise_ridge", lam=50.0)
    return v


def run_stage_a_pillar1_vault():
    out_json = VAULT_OUT / "vault_stageA_pillar1_summary.json"
    if out_json.exists():
        print(f"[VAULT_A] Already exists: {out_json.name}", flush=True)
        return json.loads(out_json.read_text())

    t0 = time.time()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    compass_d = dict(np.load(CACHE / "best_pillar1_compass.npz", allow_pickle=True))
    own_v = fit_frozen_own06b_probe()

    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )

    # Load LOCKED HumanEval (164) & LOCKED MBPP Test (257)
    with gzip.open(base258.EXP257_DATA / "HumanEval.jsonl.gz", "rt") as f:
        he_tasks = [json.loads(l) for l in f if l.strip()]
    mbpp_all = json.loads((base258.EXP257_DATA / "sanitized-mbpp.json").read_text())
    mbpp_test = [t for t in mbpp_all if 11 <= int(t["task_id"]) <= 510]
    assert len(he_tasks) == 164 and len(mbpp_test) == 257

    summary = {}
    K = 8

    for bname, tasks, is_he in (("humaneval_164", he_tasks, True), ("mbpp_test_257", mbpp_test, False)):
        tb = time.time()
        prompts, raw_instrs = [], []
        for t in tasks:
            if is_he:
                u_msg = f"Complete the following Python function:\n```python\n{t['prompt'].strip()}\n```"
                raw_instrs.append(t["prompt"].strip())
            else:
                sigs = base258.extract_mbpp_sigs(t["code"])
                imp = ("\nRequired imports:\n" + "\n".join(t["test_imports"])) if t.get("test_imports") else ""
                u_msg = f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{sigs}\n```{imp}"
                raw_instrs.append(t["prompt"].strip())
            prompts.append(
                tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": u_msg}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            )

        orig_side = tok.padding_side
        tok.padding_side = "left"
        greedy_gens = []
        with torch.no_grad():
            for s in range(0, len(prompts), 24):
                bp = prompts[s : s + 24]
                enc = tok(bp, return_tensors="pt", padding=True).to(dev)
                in_len = enc["input_ids"].shape[1]
                out = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
                for bi in range(len(bp)):
                    greedy_gens.append(tok.decode(out[bi, in_len:], skip_special_tokens=True))
                del out, enc; torch.cuda.empty_cache()

            torch.manual_seed(2639 if is_he else 2640)
            np.random.seed(2639 if is_he else 2640)
            stoch_gens = [[] for _ in range(len(prompts))]
            for s in range(0, len(prompts), 8):
                bp = prompts[s : s + 8]
                enc = tok(bp, return_tensors="pt", padding=True).to(dev)
                in_len = enc["input_ids"].shape[1]
                out = model.generate(
                    **enc, max_new_tokens=320, do_sample=True, temperature=0.7, top_p=0.95,
                    num_return_sequences=K, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                )
                for bi in range(len(bp)):
                    for ki in range(K):
                        stoch_gens[s + bi].append(tok.decode(out[bi * K + ki, in_len:], skip_special_tokens=True))
                del out, enc; torch.cuda.empty_cache()
        tok.padding_side = orig_side

        flat_tasks, flat_raws, flat_segs, flat_ast, flat_hash = [], [], [], [], []
        for ti, t in enumerate(tasks):
            for raw in ([greedy_gens[ti]] + stoch_gens[ti]):
                code = base258.extract_python_code(raw)
                aok, ah = p1s.ast_canonical_hash(code)
                _, segs = p1s.segment_trajectory_6_safe(raw_instrs[ti], code)
                flat_tasks.append(t); flat_raws.append(raw); flat_segs.append(segs)
                flat_ast.append(aok); flat_hash.append(ah)

        grader = grade_humaneval_one if is_he else base258.grade_mbpp_one
        with ThreadPoolExecutor(max_workers=12) as ex:
            graded = list(ex.map(lambda p: grader(p[0], p[1]), zip(flat_tasks, flat_raws)))
        ok = np.array([bool(g["ok"]) for g in graded], dtype=bool)
        ast_ok = np.array(flat_ast, dtype=bool)

        H_c, lp_c = p1s.extract_hidden_and_logprobs(model, tok, flat_segs, dev, batch_size=16)

        # Score ALL methods (FROZEN Compass vs ALL fair non-Platonic baselines!)
        n_t = len(tasks)
        K_tot = K + 1
        sc_compass = score_frozen_compass_np(H_c, compass_d, shuffled=False)
        sc_shuf = score_frozen_compass_np(H_c, compass_d, shuffled=True)
        F_own = p1.extract_trajectory_features({d: H_c[d].astype(np.float64) for d in p1s.DEPTHS}, p1s.DEPTHS, "audit_plus_code")
        sc_own = p1.score_verifier(F_own, own_v)

        sc_cons = np.zeros(len(ok), dtype=np.float64)
        for ti in range(n_t):
            cnt = Counter(h for h in flat_hash[ti * K_tot + 1 : (ti + 1) * K_tot] if h is not None)
            for j in range(ti * K_tot, (ti + 1) * K_tot):
                sc_cons[j] = float(cnt.get(flat_hash[j], 0)) + 0.01 * float(lp_c["code_mean_lp"][j])

        greedy_b = ok.reshape(n_t, K_tot)[:, 0]
        oracle8 = int(ok.reshape(n_t, K_tot)[:, 1:].any(axis=1).sum())
        oracle9 = int(ok.reshape(n_t, K_tot).any(axis=1).sum())

        methods = {
            "self_logprob_mean_ast": lp_c["code_mean_lp"].astype(np.float64),
            "self_logprob_sum_ast": lp_c["code_sum_lp"].astype(np.float64),
            "self_consistency_ast": sc_cons,
            "self_audit_yes_no_ast": lp_c["audit_yes_no_diff"].astype(np.float64),
            "own_06b_native_probe": sc_own,
            "shuffled_compass_ctrl": sc_shuf,
            "platonic_31b_gcca_compass": sc_compass,
        }
        res_m = {
            "greedy": {"hits": int(greedy_b.sum()), "pass1": round(float(greedy_b.mean()), 5)},
            "random_k8_mean": round(float(ok.reshape(n_t, K_tot)[:, 1:].mean()), 5),
            "oracle_k8_hits": oracle8, "oracle_k8_pass1": round(oracle8 / n_t, 5),
            "oracle_k9_hits": oracle9, "oracle_k9_pass1": round(oracle9 / n_t, 5),
        }
        bools_k8, bools_k9 = {}, {}
        for mn, sc in methods.items():
            h8, p8, b8 = p1.eval_reranker_on_mbpp(sc, ok, ast_ok, n_tasks=n_t, K_total=K_tot, include_greedy=False, ast_gate=True)
            h9, p9, b9 = p1.eval_reranker_on_mbpp(sc, ok, ast_ok, n_tasks=n_t, K_total=K_tot, include_greedy=True, ast_gate=True)
            bools_k8[mn], bools_k9[mn] = b8, b9
            res_m[mn] = {"k8_hits": h8, "k8_pass1": round(p8, 5), "k9_hits": h9, "k9_pass1": round(p9, 5)}

        c_k8, c_k9 = bools_k8["platonic_31b_gcca_compass"], bools_k9["platonic_31b_gcca_compass"]
        res_m["bootstraps"] = {
            "k9_compass_vs_greedy": p1.paired_bootstrap_bool(c_k9, greedy_b),
            "k8_compass_vs_lp_mean": p1.paired_bootstrap_bool(c_k8, bools_k8["self_logprob_mean_ast"]),
            "k9_compass_vs_lp_mean": p1.paired_bootstrap_bool(c_k9, bools_k9["self_logprob_mean_ast"]),
            "k8_compass_vs_own06b": p1.paired_bootstrap_bool(c_k8, bools_k8["own_06b_native_probe"]),
            "k9_compass_vs_own06b": p1.paired_bootstrap_bool(c_k9, bools_k9["own_06b_native_probe"]),
            "k8_compass_vs_shuf": p1.paired_bootstrap_bool(c_k8, bools_k8["shuffled_compass_ctrl"]),
            "k9_compass_vs_shuf": p1.paired_bootstrap_bool(c_k9, bools_k9["shuffled_compass_ctrl"]),
        }
        summary[bname] = res_m
        print(f"[VAULT_A:{bname}] ({time.time()-tb:.1f}s) Greedy={res_m['greedy']['hits']}/{n_t} | Oracle9={oracle9}/{n_t} | "
              f"LP_mean={res_m['self_logprob_mean_ast']['k8_hits']}/{res_m['self_logprob_mean_ast']['k9_hits']} | "
              f"Own06B={res_m['own_06b_native_probe']['k8_hits']}/{res_m['own_06b_native_probe']['k9_hits']} | "
              f"Shuf={res_m['shuffled_compass_ctrl']['k8_hits']}/{res_m['shuffled_compass_ctrl']['k9_hits']} | "
              f"PLATONIC_31B={res_m['platonic_31b_gcca_compass']['k8_hits']}/{res_m['platonic_31b_gcca_compass']['k9_hits']}", flush=True)
        print(f"  Bootstraps: {res_m['bootstraps']}", flush=True)

    del model
    gc.collect(); torch.cuda.empty_cache()
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[VAULT_A] SAVED {out_json.name} in {time.time()-t0:.1f}s!", flush=True)
    return summary


def run_stage_b_pillar3_5seed():
    out_json = VAULT_OUT / "vault_stageB_pillar3_5seed_summary.json"
    if out_json.exists():
        print(f"[VAULT_B] Already exists: {out_json.name}", flush=True)
        return json.loads(out_json.read_text())

    t0 = time.time()
    arms = {
        "p3_scratch_ce_only": ["--scaffold_w", "0.0"],
        "p3_scratch_31b_gpa_delta_decay": ["--scaffold_w", "1.20", "--geo_type", "gpa", "--align_mode", "delta", "--decay_scaffold"],
        "p3_scratch_31b_gcca_both": ["--scaffold_w", "0.60", "--geo_type", "gcca", "--align_mode", "both"],
        "p3_scratch_shuf_ctrl": ["--scaffold_w", "0.60", "--geo_type", "gcca", "--align_mode", "both", "--shuffle_geo"],
    }
    py = sys.executable
    per_arm = {a: [] for a in arms}
    for sd in SEEDS:
        for arm, flags in arms.items():
            res_p = ROOT / "results" / f"p3_{arm}_s{sd}.json"
            if not res_p.exists():
                cmd = [py, "-u", str(ROOT / "pillar3_pretrain_scratch.py"), "--arm", arm, "--seed", str(sd),
                       "--steps", "1000", "--batch_size", "12", "--lr", "1.5e-3", *flags]
                subprocess.run(cmd, check=True)
            per_arm[arm].append(json.loads(res_p.read_text()))

    metrics = ("code_eval_nll", "wikitext2_nll", "blind_30_r2", "blind_50_r2", "blind_70_r2")
    arm_stats = {arm: {m: sample_stats([r[m] for r in rows]) for m in metrics} for arm, rows in per_arm.items()}
    contrasts = {}
    for a, b in (
        ("p3_scratch_31b_gpa_delta_decay", "p3_scratch_ce_only"),
        ("p3_scratch_31b_gpa_delta_decay", "p3_scratch_shuf_ctrl"),
        ("p3_scratch_31b_gcca_both", "p3_scratch_ce_only"),
        ("p3_scratch_31b_gcca_both", "p3_scratch_shuf_ctrl"),
        ("p3_scratch_shuf_ctrl", "p3_scratch_ce_only"),
    ):
        contrasts[f"{a}_minus_{b}"] = {m: paired_t([r[m] for r in per_arm[a]], [r[m] for r in per_arm[b]]) for m in metrics}

    summary = {"seeds": SEEDS, "arms": arm_stats, "paired_contrasts": contrasts, "elapsed_sec": round(time.time() - t0, 1)}
    out_json.write_text(json.dumps(summary, indent=2))
    print("\n=== [VAULT_B] PILLAR 3 (FROM-SCRATCH 30.53M LM) 5-SEED SUMMARY ===", flush=True)
    for arm, st in arm_stats.items():
        print(f"  {arm:34s} | code_nll={st['code_eval_nll']['mean']:.5f}±{st['code_eval_nll']['sd']:.5f} | "
              f"wt2={st['wikitext2_nll']['mean']:.5f}±{st['wikitext2_nll']['sd']:.5f} | "
              f"blind={st['blind_30_r2']['mean']:.4f}/{st['blind_50_r2']['mean']:.4f}/{st['blind_70_r2']['mean']:.4f}", flush=True)
    for ck, cv in contrasts.items():
        print(f"  CONTRAST {ck}: code_nll={cv['code_eval_nll']['mean']:+.5f} [{cv['code_eval_nll']['ci95_lo']:+.5f},{cv['code_eval_nll']['ci95_hi']:+.5f}] t={cv['code_eval_nll']['t_stat']} | "
              f"wt2={cv['wikitext2_nll']['mean']:+.5f} [{cv['wikitext2_nll']['ci95_lo']:+.5f},{cv['wikitext2_nll']['ci95_hi']:+.5f}] t={cv['wikitext2_nll']['t_stat']}", flush=True)
    return summary


if __name__ == "__main__":
    run_stage_a_pillar1_vault()
    run_stage_b_pillar3_5seed()
