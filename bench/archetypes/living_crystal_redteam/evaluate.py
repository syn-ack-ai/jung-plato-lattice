#!/usr/bin/env python3
"""Immutable local evaluator wrapper adhering strictly to `autoresearch-skill/evaluator-contract.md`.
Syncs `hardened_memory.py` to the configured GPU host (`JPL_REMOTE_HOST`), executes the
immutable remote harness, pulls diagnostics locally, and prints the single contract JSON line.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE_HOST = os.environ.get("JPL_REMOTE_HOST", "gpu-host")
REMOTE_DIR = os.environ.get("JPL_REMOTE_ARCH_DIR", "bench/archetypes")
REMOTE_PY = os.environ.get("JPL_REMOTE_PYTHON", "python3")


def main():
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "hardened_memory.py")], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "hardened_memory.py"), f"{REMOTE_HOST}:{REMOTE_DIR}/exp255_ar/hardened_memory.py"], check=True)
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "
        f"{REMOTE_PY} exp255_ar/evaluate_remote.py"
    )
    proc = subprocess.run(["ssh", REMOTE_HOST, cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    subprocess.run(["scp", "-q", f"{REMOTE_HOST}:{REMOTE_DIR}/exp252/results/exp255_ar_last_eval.json", str(ROOT / "last_eval.json")], check=True)
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
