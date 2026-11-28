#!/usr/bin/env python3
"""Immutable local evaluator wrapper adhering strictly to `autoresearch-skill/evaluator-contract.md`.
Syncs `hardened_memory.py` to `dreamer`, executes the immutable remote RTX 4090 harness,
pulls diagnostics locally, and prints the single contract JSON line to stdout.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    # Syntax-check candidate first (fast fail!)
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "hardened_memory.py")], check=True)
    # Sync ONLY mutable `hardened_memory.py` to dreamer
    subprocess.run(["scp", "-q", str(ROOT / "hardened_memory.py"), "dreamer:~/systemone/bench/archetypes/exp255_ar/hardened_memory.py"], check=True)
    # Run immutable remote evaluator on RTX 4090
    cmd = (
        "cd ~/systemone/bench/archetypes && "
        "PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "
        "/home/skypanther/anaconda3/envs/ai-lab/bin/python exp255_ar/evaluate_remote.py"
    )
    proc = subprocess.run(["ssh", "dreamer", cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    # Pull full diagnostics JSON locally for hypothesis analysis
    subprocess.run(["scp", "-q", "dreamer:~/systemone/bench/archetypes/exp252/results/exp255_ar_last_eval.json", str(ROOT / "last_eval.json")], check=True)
    # Emit the single contract JSON line from remote stdout
    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            obj = json.loads(s)
            print(json.dumps(obj), flush=True)
            return
    sys.stderr.write("No JSON output from evaluate_remote.py\n" + proc.stdout + "\n" + proc.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
