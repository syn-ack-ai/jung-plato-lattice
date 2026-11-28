"""Local wrapper for immutable mechanical evaluation on dreamer (RTX 4090).
Usage: python evaluate.py [--init-base-only] [--promote-best]
Adheres strictly to `autoresearch-skill/evaluator-contract.md`.
"""
import json
import subprocess
import sys
from pathlib import Path

LOCAL_DIR = Path("/Users/benjaminbabeshkin/systemone/exp256_qwen600m_autoresearch")
REMOTE_DIR = "~/systemone/bench/archetypes/exp256_q600"
REMOTE_PY = "/home/skypanther/anaconda3/envs/ai-lab/bin/python"


def main():
    init_only = "--init-base-only" in sys.argv
    promote = "--promote-best" in sys.argv
    if promote:
        subprocess.run(
            ["ssh", "dreamer", f"cp {REMOTE_DIR}/candidate_delta.pt {REMOTE_DIR}/best_delta.pt"],
            check=True,
        )
        print(json.dumps({"promoted": True}))
        return

    # Upload ONLY mutable transplant_recipe.py (or immutable harness on init)
    files = [str(LOCAL_DIR / "transplant_recipe.py")]
    if init_only:
        files.extend([str(LOCAL_DIR / "evaluate_remote.py"), str(LOCAL_DIR / "code_agent_bench.py")])
    subprocess.run(["scp", "-q", *files, f"dreamer:{REMOTE_DIR}/"], check=True)

    flag = " --init-base-only" if init_only else ""
    res = subprocess.run(
        ["ssh", "dreamer", f"{REMOTE_PY} {REMOTE_DIR}/evaluate_remote.py{flag}"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr)
        sys.exit(res.returncode)

    if not init_only:
        subprocess.run(["scp", "-q", f"dreamer:{REMOTE_DIR}/last_eval.json", str(LOCAL_DIR / "last_eval.json")], check=False)
    else:
        subprocess.run(["scp", "-q", f"dreamer:{REMOTE_DIR}/base_ref.json", str(LOCAL_DIR / "base_ref.json")], check=False)

    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            json.loads(line)  # validate JSON
            print(line)
            return
    sys.stderr.write(res.stdout + "\n" + res.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
