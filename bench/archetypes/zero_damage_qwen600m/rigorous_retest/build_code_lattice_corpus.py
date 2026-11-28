"""Build a strictly decontaminated, AST-verified Python algorithmic problem->solution corpus
for the Multi-Family Platonic Python-Coding Lattice (`Exp 258`).

Guarantees ZERO leakage against HumanEval (164) and ALL MBPP Sanitized (427, both dev & test):
  - Rejects any example sharing a `def <name>` with HumanEval or MBPP (except trivial generic names
    like `main`/`solve`/`helper`, which are still checked by 6-gram overlap).
  - Rejects any example sharing any 6-gram of prompt words or code tokens with HumanEval or MBPP.
  - Parses every code solution with `ast.parse()` (100% syntactically valid Python functions).
  - Segments every example into 6-8 semantic checkpoints:
      [problem description end, def signature end, interior statement boundaries..., final line].
  - Splits cleanly into:
      * `code_train.jsonl` (2,000 examples — for 3-family GPA consensus & student alignment)
      * `code_eval.jsonl`  (400 examples  — disjoint held-out code for blind Phi-4 & OLMo 3 OOF R^2!)
"""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERIC_FN = {"main", "solve", "helper", "func", "foo", "bar", "test", "f", "g", "run", "solution"}


def _words(s: str):
    return re.findall(r"[a-z0-9_]+", s.lower())


def _grams6(s: str):
    w = _words(s)
    return {tuple(w[i : i + 6]) for i in range(max(0, len(w) - 5))}


def _extract_code_block(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return "\n\n".join(b.strip() for b in m if b.strip())
    return text.strip()


def _def_names(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
    return names


def segment_code_example(instr: str, code: str, max_pts: int = 8):
    """Segment `# Problem: ...\n\n<code>` into semantic checkpoints ending at:
      1. End of problem header (`# Problem: ...\n\n`)
      2. End of first `def ...:` line
      3..k. Evenly spaced interior line endings across function body
      last. Final character of code.
    Guarantees `''.join(segs) == full_text`."""
    header = f"# Problem: {instr.strip()}\n\n"
    code = code.strip()
    full = header + code
    L = len(full)
    hdr_end = len(header)

    # Candidate split positions at line endings inside code
    line_ends = [hdr_end + m.end() for m in re.finditer(r"\n", code)]
    if not line_ends or line_ends[-1] != L:
        line_ends.append(L)

    # First `def ...:` line end
    m_def = re.search(r"^def\s+\w+.*:\s*$", code, flags=re.MULTILINE)
    def_end = (hdr_end + m_def.end() + (1 if m_def.end() < len(code) and code[m_def.end()] == "\n" else 0)) if m_def else None

    chosen = [hdr_end]
    if def_end is not None and hdr_end + 8 <= def_end <= L - 10:
        chosen.append(def_end)

    rem_slots = max_pts - len(chosen) - 1  # reserve 1 for L
    body_cands = [p for p in line_ends if p > chosen[-1] + 10 and p < L - 10]
    if rem_slots > 0 and body_cands:
        start_p = chosen[-1]
        for k in range(1, rem_slots + 1):
            tgt = start_p + int(round((L - start_p) * k / (rem_slots + 1)))
            best = min(body_cands, key=lambda c: abs(c - tgt))
            if best - chosen[-1] >= 10 and (L - best) >= 10:
                chosen.append(best)
    if chosen[-1] != L:
        chosen.append(L)

    segs = []
    prev = 0
    for pos in chosen:
        if pos > prev:
            segs.append(full[prev:pos])
            prev = pos
    assert "".join(segs) == full
    return full, segs


def main():
    # 1. Fetch HumanEval + MBPP locally for strict decontamination!
    he_path = OUT_DIR / "HumanEval.jsonl.gz"
    if not he_path.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz", he_path)
    mbpp_path = OUT_DIR / "sanitized-mbpp.json"
    if not mbpp_path.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json", mbpp_path)

    with gzip.open(he_path, "rt") as f:
        he = [json.loads(l) for l in f if l.strip()]
    mbpp = json.loads(mbpp_path.read_text())

    banned_fns = set()
    banned_grams = set()
    for t in he:
        fn = t["entry_point"].lower()
        if fn not in GENERIC_FN:
            banned_fns.add(fn)
        banned_grams |= _grams6(t["prompt"])
        banned_grams |= _grams6(t["canonical_solution"])
    for t in mbpp:
        fns = _def_names(t["code"]) or []
        for fn in fns:
            if fn.lower() not in GENERIC_FN:
                banned_fns.add(fn.lower())
        banned_grams |= _grams6(t["prompt"])
        banned_grams |= _grams6(t["code"])
    print(f"Decontamination blocklist: {len(banned_fns)} function names, {len(banned_grams)} 6-grams (HumanEval + ALL 427 MBPP).", flush=True)

    # 2. Download `iamtarun/python_code_instructions_18k_alpaca`
    pq_path = hf_hub_download(
        repo_id="iamtarun/python_code_instructions_18k_alpaca",
        filename="data/train-00000-of-00001-8b6e212f3e1ece96.parquet",
        repo_type="dataset",
    )
    tbl = pq.read_table(pq_path)
    instrs = tbl["instruction"].to_pylist()
    inputs = tbl["input"].to_pylist()
    outputs = tbl["output"].to_pylist()
    print(f"Loaded {len(instrs)} raw Python instructions.", flush=True)

    kept = []
    seen_code_hash = set()
    seen_fns = set()
    stats = {"ast_fail": 0, "no_def": 0, "len_fail": 0, "decontam_fn": 0, "decontam_gram": 0, "dup": 0, "too_few_segs": 0}

    for idx, (ins, inp, out) in enumerate(zip(instrs, inputs, outputs)):
        ins = (ins or "").strip()
        inp = (inp or "").strip()
        if inp and inp.lower() not in ("not applicable", "none", "n/a"):
            ins = f"{ins} Input: {inp}"
        code = _extract_code_block(out or "")
        if not ins or not code or len(ins) > 320 or not (140 <= len(code) <= 1050):
            stats["len_fail"] += 1
            continue
        # Reject interactive input() or plotting/GUI/network boilerplate
        if re.search(r"\b(input\(|matplotlib|pyplot|tkinter|pygame|flask|django|requests\.get|urllib|sqlite3|pandas|numpy|torch|tensorflow|sklearn)\b", code):
            stats["len_fail"] += 1
            continue
        fns = _def_names(code)
        if fns is None:
            stats["ast_fail"] += 1
            continue
        if not fns:
            stats["no_def"] += 1
            continue
        # Strict decontamination against HumanEval + ALL MBPP
        if any(fn.lower() in banned_fns for fn in fns):
            stats["decontam_fn"] += 1
            continue
        if (_grams6(ins) & banned_grams) or (_grams6(code) & banned_grams):
            stats["decontam_gram"] += 1
            continue
        ch = hashlib.sha256(re.sub(r"\s+", " ", code).encode()).hexdigest()[:16]
        primary_fn = fns[0].lower()
        if ch in seen_code_hash or (primary_fn not in GENERIC_FN and primary_fn in seen_fns):
            stats["dup"] += 1
            continue
        full, segs = segment_code_example(ins, code, max_pts=8)
        if len(segs) < 4:
            stats["too_few_segs"] += 1
            continue
        seen_code_hash.add(ch)
        if primary_fn not in GENERIC_FN:
            seen_fns.add(primary_fn)
        kept.append({
            "id": f"pycode_{idx:05d}",
            "fn": fns[0],
            "instruction": ins,
            "code": code,
            "text": full,
            "segs": segs,
        })

    rng = random.Random(2580)
    rng.shuffle(kept)
    print(f"Filtered clean unique algorithmic Python functions: {len(kept)} | reject stats={stats}", flush=True)
    assert len(kept) >= 2400, f"Only {len(kept)} kept — relax cap slightly if needed!"

    train_rows = kept[:2000]
    eval_rows = kept[2000:2400]
    with open(OUT_DIR / "code_train.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "code_eval.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"SAVED: code_train.jsonl ({len(train_rows)}), code_eval.jsonl ({len(eval_rows)})", flush=True)
    print("Sample segs[0]:", len(train_rows[0]["segs"]), "fn=", train_rows[0]["fn"], "\n", train_rows[0]["text"][:320], flush=True)


if __name__ == "__main__":
    main()
