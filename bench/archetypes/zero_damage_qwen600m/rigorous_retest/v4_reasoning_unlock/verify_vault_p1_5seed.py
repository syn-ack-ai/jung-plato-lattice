"""Pillar 1 Full 5-Seed Locked-Vault & Dev Verification (`Seeds 201..205`, Exact FP32 `Qwen/Qwen3-0.6B`).

Fixes the HumanEval prompt discrepancy from Stage A by using `eval_rigorous.py`'s EXACT canonical
HumanEval prompt + `grade_humaneval_one` (reproducing Greedy `61/164 = 37.20%` bit-for-bit!) and
exact MBPP prompt (`71/170 = 41.76%` dev, `110/257 = 42.80%` test).

Runs 5 independent stochastic `K=8` sampling seeds `{201, 202, 203, 204, 205}` (`temp=0.7, top_p=0.95`)
across ALL 3 coding benchmarks (`591` total tasks per seed: `dev_mbpp_170`, locked `HumanEval 164`,
locked `MBPP Sanitized Test 257`), grading all completions in `bwrap --unshare-all` + `prlimit`, and
scoring with ALREADY-FROZEN `cache/best_pillar1_compass.npz` (ZERO vault fit, ZERO 0.6B labels!) vs.:
  - Greedy pass@1
  - Random-K8 mean
  - Self-LogProb (mean + AST) & (sum + AST)
  - Self-Consistency (AST majority + mean LP)
  - 0.6B Prompted Self-Audit ("Yes" - "No" logit)
  - Own-0.6B Native Supervised Probe (fit strictly on `code_train` pairs!)
  - Matched 5-seed Shuffled-Map Control

Computes BOTH seed-level `df=4` paired 95% t-CIs AND 10,000-resample item-level paired bootstraps
(per-benchmark AND pooled across all 591 coding tasks!). Checkpointed cleanly per `(benchmark, seed)`.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
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

SEEDS = [201, 202, 203, 204, 205]
K = 8
T_CRIT_95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764}
MODEL_ID = "Qwen/Qwen3-0.6B"


def sample_stats(vals):
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    tc = T_CRIT_95.get(n - 1, 1.96)
    return {
        "n": n, "mean": round(m, 6), "sd": round(sd, 6),
        "ci95_lo": round(m - tc * se, 6), "ci95_hi": round(m + tc * se, 6),
        "values": [round(float(x), 6) for x in arr],
    }


def paired_seed_t(va, vb):
    diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
    st = sample_stats(diff)
    sd = diff.std(ddof=1)
    st["t_stat"] = round(float(diff.mean() / (sd / math.sqrt(len(diff)))), 4) if sd > 0 else 0.0
    st["excludes_zero"] = bool(st["ci95_lo"] > 0.0 or st["ci95_hi"] < 0.0)
    return st


def bootstrap_item_diff(mat_a, mat_b, n_boot=10000, seed=26311):
    rng = np.random.default_rng(seed)
    item_diff = np.asarray(mat_a, dtype=np.float64).mean(axis=0) - np.asarray(mat_b, dtype=np.float64).mean(axis=0)
    idx = rng.integers(0, len(item_diff), size=(n_boot, len(item_diff)))
    b = item_diff[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p_two = float(2.0 * min(np.mean(b <= 0.0), np.mean(b >= 0.0)))
    return {
        "n_items": len(item_diff),
        "mean_diff_pp": round(100.0 * float(item_diff.mean()), 3),
        "boot_ci95_lo_pp": round(100.0 * float(lo), 3),
        "boot_ci95_hi_pp": round(100.0 * float(hi), 3),
        "boot_p_two_sided": round(min(1.0, p_two), 5),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def score_frozen_compass_np(H_by_depth: dict, compass_d: dict, shuffle_seed=None):
    cfg = json.loads(str(compass_d["cfg_json"].item()))
    depths = tuple(cfg["depths"])
    feat_mode = cfg["feat_mode"]
    Z_by_d = {}
    for dep in depths:
        H = H_by_depth[dep].astype(np.float64)
        Z = compass_d[f"mb_{dep}"].astype(np.float64) + (H - compass_d[f"ma_{dep}"].astype(np.float64)) @ compass_d[f"M_{dep}"].astype(np.float64)
        Z_by_d[dep] = Z
    F_mat = p1.extract_trajectory_features(Z_by_d, depths, feat_mode)
    sc = F_mat @ compass_d["w"].astype(np.float64)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        sc = rng.permutation(sc)
    return sc


def fit_frozen_own06b_probe():
    sp = dict(np.load(CACHE / "p1_student_qwen06b_pairs.npz", allow_pickle=True))
    tr_mask = ~sp["is_cal"].astype(bool)
    F_tr = p1.extract_trajectory_features({d: sp[f"h_{d}"].astype(np.float64) for d in p1s.DEPTHS}, p1s.DEPTHS, "audit_plus_code")
    return p1.fit_pairwise_verifier([F_tr[tr_mask]], sp["pair_ids"][tr_mask], sp["labels"][tr_mask], clf_type="pairwise_ridge", lam=50.0)


def build_canonical_prompts(tok, tasks, is_he: bool):
    """Exact canonical prompts matching `eval_rigorous.py` bit-for-bit!"""
    prompts, raw_instrs = [], []
    if is_he:
        sys_msg = (
            "You are an expert Python programmer. Complete the given Python function accurately. "
            "Output only the complete Python implementation inside a single ```python``` block."
        )
        for t in tasks:
            prompts.append(
                tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": f"Complete the following Python function:\n\n```python\n{t['prompt']}\n```"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            )
            raw_instrs.append(t["prompt"].strip())
    else:
        sys_msg = (
            "You are an expert Python programmer. Solve the task using the exact requested function signature. "
            "Output only the complete Python implementation inside a single ```python``` block."
        )
        for t in tasks:
            sigs = ev257.extract_mbpp_signatures(t["code"])
            imp = ("\nRequired imports:\n" + "\n".join(t["test_imports"])) if t.get("test_imports") else ""
            u_msg = f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{sigs}\n```{imp}"
            prompts.append(
                tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": u_msg}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            )
            raw_instrs.append(t["prompt"].strip())
    return prompts, raw_instrs


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    compass_d = dict(np.load(CACHE / "best_pillar1_compass.npz", allow_pickle=True))
    own_v = fit_frozen_own06b_probe()

    with gzip.open(base258.EXP257_DATA / "HumanEval.jsonl.gz", "rt") as f:
        he_tasks = [json.loads(l) for l in f if l.strip()]
    mbpp_all = json.loads((base258.EXP257_DATA / "sanitized-mbpp.json").read_text())
    mbpp_dev = [t for t in mbpp_all if int(t["task_id"]) < 11 or int(t["task_id"]) > 510]
    mbpp_test = [t for t in mbpp_all if 11 <= int(t["task_id"]) <= 510]
    assert len(he_tasks) == 164 and len(mbpp_dev) == 170 and len(mbpp_test) == 257

    benchmarks = [
        ("humaneval_164", he_tasks, True),
        ("mbpp_test_257", mbpp_test, False),
        ("dev_mbpp_170", mbpp_dev, False),
    ]
    method_names = [
        "self_logprob_mean_ast", "self_logprob_sum_ast", "self_consistency_ast",
        "self_audit_yes_no_ast", "own_06b_native_probe", "shuffled_compass_ctrl",
        "platonic_31b_gcca_compass",
    ]

    all_records = {b[0]: [] for b in benchmarks}
    for bname, tasks, is_he in benchmarks:
        prompts, raw_instrs = build_canonical_prompts(tok, tasks, is_he)
        n_t = len(tasks)
        grader = ev257.grade_humaneval_one if is_he else base258.grade_mbpp_one

        # Exact Greedy once per benchmark (deterministic FP32!)
        g_cache = VAULT_OUT / f"p1_5s_greedy_{bname}.json"
        if g_cache.exists():
            gd = json.loads(g_cache.read_text())
            greedy_gens = gd["raw_gens"]
            greedy_ok = gd["ok"]
        else:
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
            tok.padding_side = orig_side
            with ThreadPoolExecutor(max_workers=12) as ex:
                gr = list(ex.map(lambda p: grader(p[0], p[1]), zip(tasks, greedy_gens)))
            greedy_ok = [bool(x["ok"]) for x in gr]
            g_cache.write_text(json.dumps({"hits": sum(greedy_ok), "total": n_t, "ok": greedy_ok, "raw_gens": greedy_gens}))
        print(f"[P1_5SEED:{bname}] Canonical Greedy = {sum(greedy_ok)}/{n_t} ({100*sum(greedy_ok)/n_t:.2f}%)", flush=True)

        # Extract Greedy hidden states + logprobs once per benchmark
        g_segs, g_ast, g_hash = [], [], []
        for ti, raw in enumerate(greedy_gens):
            code = base258.extract_python_code(raw)
            aok, ah = p1s.ast_canonical_hash(code)
            _, segs = p1s.segment_trajectory_6_safe(raw_instrs[ti], code)
            g_segs.append(segs); g_ast.append(aok); g_hash.append(ah)
        H_g, lp_g = p1s.extract_hidden_and_logprobs(model, tok, g_segs, dev, batch_size=16)

        for sd in SEEDS:
            cp = VAULT_OUT / f"p1_5s_{bname}_s{sd}.json"
            if cp.exists():
                rec = json.loads(cp.read_text())
                all_records[bname].append(rec)
                print(f"[P1_5SEED:{bname}] CACHED s={sd} | K8 Plato={rec['k8_hits']['platonic_31b_gcca_compass']}/{n_t} | G+8 Plato={rec['k9_hits']['platonic_31b_gcca_compass']}/{n_t}", flush=True)
                continue

            ts = time.time()
            orig_side = tok.padding_side
            tok.padding_side = "left"
            torch.manual_seed(sd * 100 + (1 if is_he else (2 if bname == "mbpp_test_257" else 3)))
            np.random.seed(sd * 100)
            stoch_gens = [[] for _ in range(n_t)]
            with torch.no_grad():
                for s in range(0, n_t, 8):
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

            # Grade K=8 stochastic candidates ONCE in bwrap
            s_tasks, s_raws, s_segs, s_ast, s_hash = [], [], [], [], []
            for ti, t in enumerate(tasks):
                for raw in stoch_gens[ti]:
                    code = base258.extract_python_code(raw)
                    aok, ah = p1s.ast_canonical_hash(code)
                    _, segs = p1s.segment_trajectory_6_safe(raw_instrs[ti], code)
                    s_tasks.append(t); s_raws.append(raw); s_segs.append(segs)
                    s_ast.append(aok); s_hash.append(ah)
            with ThreadPoolExecutor(max_workers=12) as ex:
                s_gr = list(ex.map(lambda p: grader(p[0], p[1]), zip(s_tasks, s_raws)))
            s_ok = [bool(x["ok"]) for x in s_gr]
            H_s, lp_s = p1s.extract_hidden_and_logprobs(model, tok, s_segs, dev, batch_size=16)

            # Interleave [greedy_i, stoch_{i,0..7}] -> (n_t * 9)
            K_tot = K + 1
            H_all = {}
            for dep in p1s.DEPTHS:
                # Reshape H_g (n_t, 1, 6, D), H_s (n_t, K, 6, D) -> (n_t * 9, 6, D)
                hg = H_g[dep][:, None, :, :]
                hs = H_s[dep].reshape(n_t, K, 6, -1)
                H_all[dep] = np.concatenate([hg, hs], axis=1).reshape(n_t * K_tot, 6, -1)
            lp_all = {}
            for lk in ("code_sum_lp", "code_mean_lp", "audit_yes_no_diff"):
                lp_all[lk] = np.concatenate([lp_g[lk][:, None], lp_s[lk].reshape(n_t, K)], axis=1).reshape(-1)
            ok_all = np.concatenate([np.array(greedy_ok, dtype=bool)[:, None], np.array(s_ok, dtype=bool).reshape(n_t, K)], axis=1).reshape(-1)
            ast_all = np.concatenate([np.array(g_ast, dtype=bool)[:, None], np.array(s_ast, dtype=bool).reshape(n_t, K)], axis=1).reshape(-1)
            hash_all = []
            for ti in range(n_t):
                hash_all.append(g_hash[ti])
                hash_all.extend(s_hash[ti * K : (ti + 1) * K])

            sc_compass = score_frozen_compass_np(H_all, compass_d, shuffle_seed=None)
            sc_shuf = score_frozen_compass_np(H_all, compass_d, shuffle_seed=sd * 997 + n_t)
            F_own = p1.extract_trajectory_features({d: H_all[d].astype(np.float64) for d in p1s.DEPTHS}, p1s.DEPTHS, "audit_plus_code")
            sc_own = p1.score_verifier(F_own, own_v)
            sc_cons = np.zeros(len(ok_all), dtype=np.float64)
            for ti in range(n_t):
                cnt = Counter(h for h in hash_all[ti * K_tot + 1 : (ti + 1) * K_tot] if h is not None)
                for j in range(ti * K_tot, (ti + 1) * K_tot):
                    sc_cons[j] = float(cnt.get(hash_all[j], 0)) + 0.01 * float(lp_all["code_mean_lp"][j])

            scores_map = {
                "self_logprob_mean_ast": lp_all["code_mean_lp"].astype(np.float64),
                "self_logprob_sum_ast": lp_all["code_sum_lp"].astype(np.float64),
                "self_consistency_ast": sc_cons,
                "self_audit_yes_no_ast": lp_all["audit_yes_no_diff"].astype(np.float64),
                "own_06b_native_probe": sc_own,
                "shuffled_compass_ctrl": sc_shuf,
                "platonic_31b_gcca_compass": sc_compass,
            }
            rec = {
                "benchmark": bname, "seed": sd, "n_tasks": n_t,
                "greedy_hits": int(sum(greedy_ok)), "greedy_bool": greedy_ok,
                "random_k8_mean": round(float(np.mean(s_ok)), 5),
                "oracle_k8_hits": int(ok_all.reshape(n_t, K_tot)[:, 1:].any(axis=1).sum()),
                "oracle_k9_hits": int(ok_all.reshape(n_t, K_tot).any(axis=1).sum()),
                "k8_hits": {}, "k9_hits": {}, "k8_bool": {}, "k9_bool": {},
            }
            for mn, sc in scores_map.items():
                h8, _, b8 = p1.eval_reranker_on_mbpp(sc, ok_all, ast_all, n_tasks=n_t, K_total=K_tot, include_greedy=False, ast_gate=True)
                h9, _, b9 = p1.eval_reranker_on_mbpp(sc, ok_all, ast_all, n_tasks=n_t, K_total=K_tot, include_greedy=True, ast_gate=True)
                rec["k8_hits"][mn] = int(h8); rec["k9_hits"][mn] = int(h9)
                rec["k8_bool"][mn] = [bool(x) for x in b8]; rec["k9_bool"][mn] = [bool(x) for x in b9]
            cp.write_text(json.dumps(rec))
            all_records[bname].append(rec)
            print(
                f"[P1_5SEED:{bname}] s={sd} ({time.time()-ts:.1f}s) Greedy={rec['greedy_hits']} | Oracle9={rec['oracle_k9_hits']} | "
                f"LP={rec['k8_hits']['self_logprob_mean_ast']}/{rec['k9_hits']['self_logprob_mean_ast']} | "
                f"Own06B={rec['k8_hits']['own_06b_native_probe']}/{rec['k9_hits']['own_06b_native_probe']} | "
                f"Shuf={rec['k8_hits']['shuffled_compass_ctrl']}/{rec['k9_hits']['shuffled_compass_ctrl']} | "
                f"PLATO31B={rec['k8_hits']['platonic_31b_gcca_compass']}/{rec['k9_hits']['platonic_31b_gcca_compass']}",
                flush=True,
            )
            del H_s, H_all; gc.collect(); torch.cuda.empty_cache()

    # Aggregate 5-seed stats per benchmark AND pooled across all 591 coding tasks!
    summary = {"seeds": SEEDS, "K": K, "benchmarks": {}, "pooled_591": {}}
    pooled_k8 = {mn: [] for mn in method_names}
    pooled_k9 = {mn: [] for mn in method_names}
    pooled_greedy = []
    for si in range(len(SEEDS)):
        g_row = []
        for bname, _, _ in benchmarks:
            g_row.extend(all_records[bname][si]["greedy_bool"])
        pooled_greedy.append(g_row)
        for mn in method_names:
            r8, r9 = [], []
            for bname, _, _ in benchmarks:
                r8.extend(all_records[bname][si]["k8_bool"][mn])
                r9.extend(all_records[bname][si]["k9_bool"][mn])
            pooled_k8[mn].append(r8); pooled_k9[mn].append(r9)

    def summarize_block(greedy_mat, k8_dict, k9_dict):
        n_t = len(greedy_mat[0])
        g_acc = float(np.mean(greedy_mat[0]))
        blk = {"n_tasks": n_t, "greedy_pass1": round(g_acc, 5), "greedy_hits": int(sum(greedy_mat[0])), "methods": {}, "contrasts": {}}
        for mn in method_names:
            k8_rates = [float(np.mean(r)) for r in k8_dict[mn]]
            k9_rates = [float(np.mean(r)) for r in k9_dict[mn]]
            blk["methods"][mn] = {"k8": sample_stats(k8_rates), "k9": sample_stats(k9_rates)}
        p_k8, p_k9 = k8_dict["platonic_31b_gcca_compass"], k9_dict["platonic_31b_gcca_compass"]
        p_k8_r = [float(np.mean(r)) for r in p_k8]
        p_k9_r = [float(np.mean(r)) for r in p_k9]
        blk["contrasts"]["k9_plato_vs_greedy"] = {
            "seed_t": paired_seed_t(p_k9_r, [g_acc] * len(SEEDS)),
            "item_boot": bootstrap_item_diff(p_k9, greedy_mat),
        }
        blk["contrasts"]["k8_plato_vs_greedy"] = {
            "seed_t": paired_seed_t(p_k8_r, [g_acc] * len(SEEDS)),
            "item_boot": bootstrap_item_diff(p_k8, greedy_mat),
        }
        for ctrl in ("self_logprob_mean_ast", "self_logprob_sum_ast", "self_consistency_ast",
                     "self_audit_yes_no_ast", "own_06b_native_probe", "shuffled_compass_ctrl"):
            c8_r = [float(np.mean(r)) for r in k8_dict[ctrl]]
            c9_r = [float(np.mean(r)) for r in k9_dict[ctrl]]
            blk["contrasts"][f"k8_plato_vs_{ctrl}"] = {
                "seed_t": paired_seed_t(p_k8_r, c8_r),
                "item_boot": bootstrap_item_diff(p_k8, k8_dict[ctrl]),
            }
            blk["contrasts"][f"k9_plato_vs_{ctrl}"] = {
                "seed_t": paired_seed_t(p_k9_r, c9_r),
                "item_boot": bootstrap_item_diff(p_k9, k9_dict[ctrl]),
            }
        return blk

    for bname, _, _ in benchmarks:
        recs = all_records[bname]
        summary["benchmarks"][bname] = summarize_block(
            [r["greedy_bool"] for r in recs],
            {mn: [r["k8_bool"][mn] for r in recs] for mn in method_names},
            {mn: [r["k9_bool"][mn] for r in recs] for mn in method_names},
        )
    summary["pooled_591"] = summarize_block(pooled_greedy, pooled_k8, pooled_k9)
    summary["elapsed_sec"] = round(time.time() - t0, 1)

    out_p = VAULT_OUT / "vault_stageA_pillar1_5seed_summary.json"
    out_p.write_text(json.dumps(summary, indent=2))
    print("\n=== [P1_5SEED] FINAL 5-SEED SUMMARY (ALL 591 TASKS) ===", flush=True)
    for tag, blk in list(summary["benchmarks"].items()) + [("pooled_591", summary["pooled_591"])]:
        m = blk["methods"]
        print(f"  {tag:14s} | Greedy={100*blk['greedy_pass1']:.2f}% | "
              f"LP={100*m['self_logprob_mean_ast']['k9']['mean']:.2f}% | "
              f"Own06B={100*m['own_06b_native_probe']['k9']['mean']:.2f}% | "
              f"Shuf={100*m['shuffled_compass_ctrl']['k9']['mean']:.2f}% | "
              f"PLATO_K8={100*m['platonic_31b_gcca_compass']['k8']['mean']:.2f}±{100*m['platonic_31b_gcca_compass']['k8']['sd']:.2f}% | "
              f"PLATO_G8={100*m['platonic_31b_gcca_compass']['k9']['mean']:.2f}±{100*m['platonic_31b_gcca_compass']['k9']['sd']:.2f}%", flush=True)
    print("\n  POOLED 591 CONTRASTS:", flush=True)
    for ck, cv in summary["pooled_591"]["contrasts"].items():
        st, ib = cv["seed_t"], cv["item_boot"]
        print(f"    {ck:34s}: {100*st['mean']:+.2f}pp seed_CI=[{100*st['ci95_lo']:+.2f},{100*st['ci95_hi']:+.2f}] t={st['t_stat']} | item_boot_p={ib['boot_p_two_sided']}", flush=True)


if __name__ == "__main__":
    main()
