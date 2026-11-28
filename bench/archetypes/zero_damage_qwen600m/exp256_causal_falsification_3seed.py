"""Paper-Grade Causal Falsification & Multi-Seed Verification for Exp 256 (`Qwen/Qwen3-0.6B`).
Tests directly whether the 7B-9B Rival Platonic SHAPE ITSELF causes the coding/language/reasoning gains!

Arms evaluated via the EXACT locked immutable `evaluate_remote.eval_model_full` harness:
  1. Champion `Iter 15 (Dual-Domain True Shape)` x 3 seeds (256, 257, 258)
  2. `Iter 15 SHUFFLED-SHAPE Control` (EXACT same code, ce_w=0, KL, PCGrad, rank=24 LoRA on L2..L20;
     ONLY teacher target rows randomly permuted across passages!) x 2 seeds (256, 257)
  3. `Iter 14 Pure-Prose True Shape` (ZERO code data anywhere!) x 2 seeds (256, 257)

Saves full JSON to `causal_falsification_3seed.json`.
"""
import copy
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ARCH = Path(os.environ.get("JPL_ARCH_DIR", Path(__file__).resolve().parent.parent))
QDIR = Path(os.environ.get("JPL_Q600_DIR", Path(__file__).resolve().parent))
TARGETS_NPZ = QDIR / "targets_qwen600m_rival30_50.npz"
CODE_NPZ = QDIR / "stdlib_code_gemma4_8b_targets.npz"
BASE_REF_PATH = QDIR / "base_ref.json"
OUT_JSON = QDIR / "causal_falsification_3seed.json"

if str(QDIR) not in sys.path:
    sys.path.insert(0, str(QDIR))

import evaluate_remote as ev_mod  # noqa: E402


def load_module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_against_base(ev, base_ref):
    d_nll = round(ev["gut_nll"] - base_ref["gut_nll"], 5)
    d_code_pass1 = round(ev["code_pass1"] - base_ref["code_pass1"], 5)
    d_code_units = round(ev["code_units"] - base_ref["code_units"], 5)
    d_swe = round(ev["swe_acc"] - base_ref["swe_acc"], 5)
    d_proc = round(ev["proc_acc"] - base_ref["proc_acc"], 5)
    d_actor = round(ev["actor_acc"] - base_ref["actor_acc"], 5)
    d_al30 = round(ev["mean_blind_30"] - base_ref["mean_blind_30"], 5)
    d_al50 = round(ev["mean_blind_50"] - base_ref["mean_blind_50"], 5)
    d_peak = round(max(d_al30, d_al50), 5)
    score = round((-100.0 * d_nll) + (80.0 * d_code_pass1) + (80.0 * d_code_units) + (80.0 * d_swe) + (80.0 * d_proc) + (50.0 * d_actor), 4)
    guard_ok = bool(d_peak >= 0.050)
    passed = bool(guard_ok and d_nll <= 0.0 and d_code_units >= 0.0 and d_swe >= 0.0 and d_proc >= 0.0 and score >= 3.00)
    return {
        "pass": passed,
        "score": score,
        "guard_ok": guard_ok,
        "deltas": {
            "d_nll": d_nll, "d_code_pass1": d_code_pass1, "d_code_units": d_code_units,
            "d_swe": d_swe, "d_proc": d_proc, "d_actor": d_actor,
            "d_align_30": d_al30, "d_align_50": d_al50, "d_align_peak": d_peak,
        },
        "eval": {k: v for k, v in ev.items() if k != "code_per_task"},
    }


def make_shuffled_targets(d_t_dict, d_c_dict, shuf_seed):
    """Permute teacher target rows across ALL points while keeping exact passage structure & norms identical!"""
    rng = np.random.default_rng(shuf_seed)
    dt_s = dict(d_t_dict)
    p1 = rng.permutation(len(dt_s["r3_30_tr"]))
    dt_s["r3_30_tr"] = dt_s["r3_30_tr"][p1].copy()
    dt_s["r3_50_tr"] = dt_s["r3_50_tr"][p1].copy()

    dc_s = dict(d_c_dict)
    p2 = rng.permutation(len(dc_s["g4_code_30"]))
    dc_s["g4_code_30"] = dc_s["g4_code_30"][p2].copy()
    dc_s["g4_code_50"] = dc_s["g4_code_50"][p2].copy()
    return dt_s, dc_s


def main():
    t0 = time.time()
    d_t_raw = np.load(TARGETS_NPZ, allow_pickle=True)
    d_t_dict = {k: d_t_raw[k] for k in d_t_raw.files}
    d_c_raw = np.load(CODE_NPZ, allow_pickle=True)
    d_c_dict = {k: d_c_raw[k] for k in d_c_raw.files}
    base_ref = json.loads(BASE_REF_PATH.read_text())
    tok = AutoTokenizer.from_pretrained(ev_mod.MODEL_ID)

    mod15 = load_module_from(QDIR / "transplant_recipe.iter15_pass.py", "rec15")
    mod14 = load_module_from(QDIR / "transplant_recipe.iter14.py", "rec14")

    runs = []

    def _run_one(arm_label, mod, seed, shuffled=False):
        tr0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(ev_mod.MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        orig_code_path = getattr(mod, "CODE_TARGETS_NPZ", None)
        tmp_code_path = QDIR / f"_tmp_shuf_code_{seed}.npz"
        try:
            if shuffled:
                dt_use, dc_use = make_shuffled_targets(d_t_dict, d_c_dict, shuf_seed=seed + 9000)
                if orig_code_path is not None:
                    np.savez(tmp_code_path, **dc_use)
                    mod.CODE_TARGETS_NPZ = tmp_code_path
            else:
                dt_use = d_t_dict
            tmeta = mod.run_transplant(model, tok, dt_use, seed=seed)
            model.eval()
            ev = ev_mod.eval_model_full(model, tok, d_t_dict)
            res = score_against_base(ev, base_ref)
            res["arm"] = arm_label
            res["seed"] = seed
            res["shuffled"] = shuffled
            res["elapsed_sec"] = round(time.time() - tr0, 1)
            res["train_sec"] = tmeta.get("train_sec")
            runs.append(res)
            print(
                f"[{arm_label} s={seed}] pass={res['pass']} score={res['score']:+.2f} "
                f"code={ev['code_hits']} swe={res['deltas']['d_swe']:+.4f} "
                f"nll={res['deltas']['d_nll']:+.5f} proc={res['deltas']['d_proc']:+.4f} "
                f"actor={res['deltas']['d_actor']:+.4f} al30={res['deltas']['d_align_30']:+.4f} al50={res['deltas']['d_align_50']:+.4f}",
                flush=True,
            )
        finally:
            if orig_code_path is not None:
                mod.CODE_TARGETS_NPZ = orig_code_path
            if tmp_code_path.exists():
                tmp_code_path.unlink()
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # 1. Champion Iter 15 (Dual-Domain True Shape) x 3 seeds!
    for sd in (256, 257, 258):
        _run_one("Iter15_DualDomain_TRUE", mod15, seed=sd, shuffled=False)

    # 2. Shuffled-Shape Control (exact same Iter 15 code, ce_w=0, KL, PCGrad, ONLY target rows permuted!) x 2 seeds!
    for sd in (256, 257):
        _run_one("Iter15_DualDomain_SHUFFLED", mod15, seed=sd, shuffled=True)

    # 3. Pure-Prose Iter 14 (ZERO code data anywhere!) x 2 seeds!
    for sd in (256, 257):
        _run_one("Iter14_PureProse_TRUE", mod14, seed=sd, shuffled=False)

    # Aggregate mean +/- std per arm
    summary = {}
    for arm_name in ("Iter15_DualDomain_TRUE", "Iter15_DualDomain_SHUFFLED", "Iter14_PureProse_TRUE"):
        sub = [r for r in runs if r["arm"] == arm_name]
        agg = {"n_seeds": len(sub), "n_pass": sum(int(r["pass"]) for r in sub)}
        for k in ("score",):
            vals = [r[k] for r in sub]
            agg[k] = f"{np.mean(vals):+.2f} ± {np.std(vals):.2f}"
        for dk in ("d_nll", "d_code_pass1", "d_code_units", "d_swe", "d_proc", "d_actor", "d_align_30", "d_align_50"):
            vals = [r["deltas"][dk] for r in sub]
            agg[dk] = f"{np.mean(vals):+.4f} ± {np.std(vals):.4f}"
        summary[arm_name] = agg

    out = {"total_sec": round(time.time() - t0, 1), "summary": summary, "runs": runs}
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print("\n=== CAUSAL FALSIFICATION & MULTI-SEED SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
