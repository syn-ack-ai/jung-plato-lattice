"""Mechanical JSON Evaluator (`evaluate.py`) for Exp 263 Autoresearch Loop.
Contract: outputs JSON `{"pass": bool, "score": float, ...}` on stdout!
  - `score`: disjoint `dev_mbpp_170` pass@1 (higher is better)
  - Hard Guard: `wt2_drift <= +0.010` (for Pillar 1 test-time compass, weights are untouched -> `wt2_drift = 0.0`).
  - Strict Vault Lock: NEVER touches `HumanEval (164)` or `MBPP Test (257)`!
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WT2_BASE = 3.291073
WT2_GUARD_MAX_DRIFT = 0.010
BASE_SCORE = 71 / 170.0


def evaluate_result_file(path: Path):
    d = json.loads(path.read_text())
    score = float(d["dev_mbpp_pass1"])
    wt2 = float(d.get("wikitext2_nll", WT2_BASE))
    wt2_drift = wt2 - WT2_BASE
    guard_ok = (wt2_drift <= WT2_GUARD_MAX_DRIFT)
    out = {
        "pass": bool(guard_ok and score >= BASE_SCORE),
        "guard_ok": bool(guard_ok),
        "score": round(score, 6),
        "hits": int(d.get("dev_mbpp_hits_int", round(score * 170))),
        "total": int(d.get("dev_mbpp_total", 170)),
        "delta_vs_base": round(score - BASE_SCORE, 6),
        "wikitext2_nll": round(wt2, 6),
        "wt2_drift": round(wt2_drift, 6),
        "code_eval_nll": d.get("dev_code_nll"),
        "proc_unseen_acc": d.get("proc_unseen_acc"),
        "arm": d.get("arm", path.stem),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", type=Path)
    args = ap.parse_args()
    evaluate_result_file(args.result_json)
