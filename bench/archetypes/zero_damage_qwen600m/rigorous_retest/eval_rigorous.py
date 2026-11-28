"""Rigorous, Immutable, Bug-Fixed Mechanical Evaluator (Experiment 257).

Fixes every confirmed statistical, mathematical, and systems defect from PR #2 audit:
  - Exact FP32 evaluation (zero bf16 merge/logit rounding error).
  - Real OS sandbox (`bwrap --unshare-all` + `prlimit`) for all untrusted generated Python
    code, with strict exit-code + cryptographic pass-sentinel verification.
  - Untouched third-party benchmarks NEVER used during Exp 256 hyperparameter search:
      1. HumanEval (164 official tasks, pass@1)
      2. MBPP Sanitized official test split (257 tasks, task_id 11..510, pass@1;
         prompt supplies task description + bare function signature only, zero test cases)
      3. WikiText-2 raw test split (62 Wikipedia articles, 100% non-Gutenberg NLL)
  - Corrected internal benchmarks:
      4. Held-out Gutenberg NLL (100 passages, per-passage NLLs persisted)
      5. Exact Jung-Plato paper OOF Procrustes CV (`plato_data.py` formula verbatim:
         un-renormalized `r2_frob = 1 - SSE/SST`, train-fold scalar `r2_opt`, `cos`,
         tie-safe `top1`/`top5`) against blind judges Phi-4-mini & OLMo 3 7B at L8 & L14
      6. Balanced ALL-240 `bm_actor2x2` (120 orig + 120 mirror, reporting overall &
         mirror-consistent pair accuracy)
      7. Symmetric ALL-240 `SWE-Smith` (untruncated edit-centered <=1024-tok window with
         `autojunk=False`; reports BOTH symmetric full-window contrast AND symmetric
         +/-8-token padded edit-site contrast — zero asymmetric empty-mask fallback)
      8. Procedural Reasoning MCQ (`proc_held`, 200 stratified items)
  - Logs SHA-256 hash of this evaluator file on every run and persists full per-item records.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import difflib
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("EXP257_MODEL_ID", "Qwen/Qwen3-0.6B")
LAYER_30 = 8   # 8/28 = 0.286
LAYER_50 = 14  # 14/28 = 0.500


def default_paths():
    root = Path(os.environ.get("EXP257_ARCH_DIR", Path(__file__).resolve().parent.parent))
    exp_dir = Path(os.environ.get("EXP257_DIR", Path(__file__).resolve().parent))
    qdir = Path(os.environ.get("EXP257_Q600_DIR", root / "exp256_q600"))
    return {
        "arch": root,
        "exp_dir": exp_dir,
        "data_dir": exp_dir / "data",
        "plato_dir": Path(os.environ.get("EXP257_PLATO_DIR", root / "fluent" / "plato")),
        "probes_dir": Path(os.environ.get("EXP257_PROBES_DIR", root / "exp252" / "probes")),
        "targets_npz": Path(os.environ.get("EXP257_TARGETS_NPZ", qdir / "targets_qwen600m_rival30_50.npz")),
        "code_targets_npz": Path(os.environ.get("EXP257_CODE_TARGETS_NPZ", qdir / "stdlib_code_gemma4_8b_targets.npz")),
        "swe_path": Path(os.environ.get("EXP257_SWE_PATH", qdir / "swe_code_240.jsonl")),
    }


def evaluator_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Real OS Sandbox (`bwrap --unshare-all` + `prlimit`)
# ---------------------------------------------------------------------------

def _sandbox_cmd():
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    if not bwrap:
        raise RuntimeError("bwrap is required for OS-isolated code execution but was not found on PATH.")
    py_prefix = str(Path(sys.prefix).resolve())
    py_rel = str(Path(sys.executable).resolve().relative_to(Path(py_prefix)))
    cmd = [
        bwrap,
        "--unshare-all",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/bin", "/bin",
    ]
    if Path("/lib64").exists():
        cmd.extend(["--ro-bind", "/lib64", "/lib64"])
    cmd.extend([
        "--ro-bind", py_prefix, "/opt/py",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--chdir", "/tmp",
        "--die-with-parent",
        "--",
    ])
    if prlimit:
        cmd.extend([prlimit, "--as=536870912", "--cpu=3", "--fsize=1048576", "--"])
    cmd.extend([f"/opt/py/{py_rel}", "-I", "-S", "-c"])
    return cmd


_SANDBOX_BASE_CMD = None


def run_in_sandbox(script: str, timeout_s: float = 4.0):
    global _SANDBOX_BASE_CMD
    if _SANDBOX_BASE_CMD is None:
        _SANDBOX_BASE_CMD = _sandbox_cmd()
    try:
        res = subprocess.run(
            [*_SANDBOX_BASE_CMD, script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"


def extract_python_code(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    m = re.search(r"^(?:import |from |def |class )", cleaned, flags=re.MULTILINE)
    if m:
        return cleaned[m.start():].strip()
    return cleaned


def assemble_humaneval_code(prompt: str, entry_point: str, extracted: str) -> str:
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", extracted, flags=re.MULTILINE):
        header_imports = [
            ln for ln in prompt.splitlines()
            if ln.startswith("import ") or ln.startswith("from ")
        ]
        prefix = "\n".join(header_imports) + "\n\n" if header_imports else ""
        return prefix + extracted
    # Model output body only
    lines = extracted.splitlines()
    if lines and not lines[0].startswith((" ", "\t")):
        body = "\n".join("    " + ln for ln in lines)
    else:
        body = extracted
    return prompt + "\n" + body


def grade_humaneval_one(task: dict, raw_gen: str, timeout_s: float = 4.0) -> dict:
    code = extract_python_code(raw_gen)
    full_code = assemble_humaneval_code(task["prompt"], task["entry_point"], code)
    sentinel = f"__HE_PASS_{secrets.token_hex(12)}__"
    script = (
        f"{full_code}\n\n"
        f"{task['test']}\n\n"
        f"check({task['entry_point']})\n"
        f"print({sentinel!r})\n"
    )
    rc, stdout, stderr = run_in_sandbox(script, timeout_s=timeout_s)
    ok = (rc == 0) and (sentinel in stdout.splitlines())
    err = ""
    if not ok:
        if stderr.strip():
            err = stderr.strip().splitlines()[-1][:140]
        else:
            err = f"rc={rc}"
    return {
        "task_id": task["task_id"],
        "ok": bool(ok),
        "err": err,
        "code": code,
        "raw": raw_gen,
    }


def extract_mbpp_signatures(ref_code: str) -> str:
    sigs = re.findall(r"^(?:def|class)\s+\w+.*:", ref_code, flags=re.MULTILINE)
    return "\n".join(sigs)


def grade_mbpp_one(task: dict, raw_gen: str, timeout_s: float = 4.0) -> dict:
    code = extract_python_code(raw_gen)
    sentinel = f"__MBPP_PASS_{secrets.token_hex(12)}__"
    imports_blk = "\n".join(task.get("test_imports", []))
    asserts_blk = "\n".join(task["test_list"])
    script = (
        f"{imports_blk}\n"
        f"{code}\n\n"
        f"{asserts_blk}\n"
        f"print({sentinel!r})\n"
    )
    rc, stdout, stderr = run_in_sandbox(script, timeout_s=timeout_s)
    ok = (rc == 0) and (sentinel in stdout.splitlines())
    err = ""
    if not ok:
        if stderr.strip():
            err = stderr.strip().splitlines()[-1][:140]
        else:
            err = f"rc={rc}"
    return {
        "task_id": int(task["task_id"]),
        "ok": bool(ok),
        "err": err,
        "code": code,
        "raw": raw_gen,
    }


@torch.no_grad()
def _generate_batched_fp32(model, tok, prompts, max_new_tokens=320, batch_size=24):
    dev = next(model.parameters()).device
    orig_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gens = []
    try:
        for s in range(0, len(prompts), batch_size):
            batch_p = prompts[s : s + batch_size]
            enc = tok(batch_p, return_tensors="pt", padding=True).to(dev)
            in_len = enc["input_ids"].shape[1]
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            for bi in range(len(batch_p)):
                gens.append(tok.decode(out[bi, in_len:], skip_special_tokens=True))
            del out, enc
            torch.cuda.empty_cache()
    finally:
        tok.padding_side = orig_side
    return gens


def eval_humaneval(model, tok, data_dir: Path, max_workers: int = 12):
    with gzip.open(data_dir / "HumanEval.jsonl.gz", "rt") as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    sys_msg = (
        "You are an expert Python programmer. Complete the given Python function accurately. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    prompts = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Complete the following Python function:\n\n```python\n{t['prompt']}\n```"},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for t in tasks
    ]
    gens = _generate_batched_fp32(model, tok, prompts, max_new_tokens=320, batch_size=24)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        per_task = list(ex.map(lambda pair: grade_humaneval_one(pair[0], pair[1]), zip(tasks, gens)))
    hits = sum(1 for r in per_task if r["ok"])
    return {
        "pass_at_1": round(hits / max(1, len(per_task)), 5),
        "hits": hits,
        "total": len(per_task),
        "per_task_bool": {r["task_id"]: r["ok"] for r in per_task},
        "per_task_full": per_task,
    }


def eval_mbpp_test(model, tok, data_dir: Path, max_workers: int = 12):
    mbpp = json.loads((data_dir / "sanitized-mbpp.json").read_text())
    tasks = [x for x in mbpp if 11 <= int(x["task_id"]) <= 510]
    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    prompts = []
    for t in tasks:
        sigs = extract_mbpp_signatures(t["code"])
        imp = ("\nRequired imports:\n" + "\n".join(t["test_imports"])) if t.get("test_imports") else ""
        u_msg = (
            f"{t['prompt'].strip()}\n\n"
            f"Use this exact signature:\n```python\n{sigs}\n```{imp}"
        )
        prompts.append(
            tok.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": u_msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    gens = _generate_batched_fp32(model, tok, prompts, max_new_tokens=320, batch_size=24)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        per_task = list(ex.map(lambda pair: grade_mbpp_one(pair[0], pair[1]), zip(tasks, gens)))
    hits = sum(1 for r in per_task if r["ok"])
    return {
        "pass_at_1": round(hits / max(1, len(per_task)), 5),
        "hits": hits,
        "total": len(per_task),
        "per_task_bool": {str(r["task_id"]): r["ok"] for r in per_task},
        "per_task_full": per_task,
    }


# ---------------------------------------------------------------------------
# 2. Exact Paper OOF Procrustes CV (`plato_data.py` verbatim!)
# ---------------------------------------------------------------------------

def fit_procrustes_map(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mu_a = A.mean(axis=0, keepdims=True)
    mu_b = B.mean(axis=0, keepdims=True)
    Ac = A - mu_a
    Bc = B - mu_b
    M = Bc.T @ Ac
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    na = np.linalg.norm(Ac @ R.T, "fro")
    nb = np.linalg.norm(Bc, "fro")
    s = float(nb / max(1e-12, na))
    return mu_a, mu_b, R, s


def apply_procrustes_map(A, params):
    mu_a, mu_b, R, s = params
    return mu_b + s * ((np.asarray(A, dtype=np.float64) - mu_a) @ R.T)


def eval_procrustes_paper_exact(src_ev, tgt_ev, group_ids, domains, ret_mask=None):
    """Exact `plato_data.py:686-775` domain-stratified group-aware 2-fold OOF Procrustes CV.
    Fits top-d_tgt PCA strictly on each train fold when src_ev.shape[1] != tgt_ev.shape[1]."""
    src_ev = np.asarray(src_ev, dtype=np.float64)
    tgt_ev = np.asarray(tgt_ev, dtype=np.float64)
    d_tgt = tgt_ev.shape[1]
    group_ids = [str(g) for g in group_ids]
    domains = [str(d) for d in domains]

    dom_groups = defaultdict(list)
    seen_g = set()
    for g, d in zip(group_ids, domains):
        if g not in seen_g:
            seen_g.add(g)
            dom_groups[d].append(g)
    g_fold = {}
    rng = random.Random(2288)
    for d in sorted(dom_groups):
        gs = list(dom_groups[d])
        rng.shuffle(gs)
        half = len(gs) // 2
        for i, g in enumerate(gs):
            g_fold[g] = 0 if i < half else 1
    folds = np.array([g_fold[g] for g in group_ids], dtype=np.int32)

    cos_all, pred_all, pred_opt_all, true_all = [], [], [], []
    top1_list, top5_list = [], []
    dom_cos = defaultdict(list)

    for test_f in (0, 1):
        m_tr = folds != test_f
        m_te = folds == test_f
        X_tr, X_te = src_ev[m_tr], src_ev[m_te]
        if X_tr.shape[1] != d_tgt:
            mu_x = X_tr.mean(axis=0, keepdims=True)
            Xc_tr = X_tr - mu_x
            cov = (Xc_tr.T @ Xc_tr) / float(len(Xc_tr))
            vals, vecs = np.linalg.eigh(cov)
            idx = np.argsort(vals)[::-1][:d_tgt]
            V = vecs[:, idx]
            Z_tr = Xc_tr @ V
            Z_te = (X_te - mu_x) @ V
            rms = float(np.sqrt(np.mean(Z_tr ** 2)))
            Z_tr = Z_tr / max(1e-12, rms)
            Z_te = Z_te / max(1e-12, rms)
        else:
            Z_tr, Z_te = X_tr, X_te

        params = fit_procrustes_map(Z_tr, tgt_ev[m_tr])
        pred_tr = apply_procrustes_map(Z_tr, params)
        pred = apply_procrustes_map(Z_te, params)
        true_tr = tgt_ev[m_tr]
        true = tgt_ev[m_te]

        # Strictly train-fold optimal scalar for r2_opt (zero test-fold leakage!)
        ym_tr = true_tr.mean(axis=0, keepdims=True)
        pm_tr = pred_tr.mean(axis=0, keepdims=True)
        alpha_f = float(np.sum((true_tr - ym_tr) * (pred_tr - pm_tr)) / max(1e-12, np.sum((pred_tr - pm_tr) ** 2)))
        pred_opt_all.append(ym_tr + alpha_f * (pred - pm_tr))

        pn = pred / np.maximum(1e-12, np.linalg.norm(pred, axis=1, keepdims=True))
        tn = true / np.maximum(1e-12, np.linalg.norm(true, axis=1, keepdims=True))
        cs = np.sum(pn * tn, axis=1)
        cos_all.append(cs)
        pred_all.append(pred)
        true_all.append(true)
        te_doms = [domains[i] for i in np.flatnonzero(m_te)]
        for c_val, d_val in zip(cs, te_doms):
            dom_cos[d_val].append(float(c_val))

        if ret_mask is not None:
            rm = np.asarray(ret_mask, dtype=bool)[m_te]
            pn_r, tn_r = pn[rm], tn[rm]
        else:
            pn_r, tn_r = pn, tn
        if len(pn_r) >= 6:
            sim = pn_r @ tn_r.T
            diag = np.diag(sim)[:, None]
            strictly_better = (sim > diag + 1e-12).sum(axis=1)
            tied = (np.abs(sim - diag) <= 1e-12).sum(axis=1)
            top1_list.append(float(np.mean(np.where(strictly_better == 0, 1.0 / np.maximum(1, tied), 0.0))))
            rem5 = np.maximum(0, 5 - strictly_better)
            top5_list.append(float(np.mean(np.minimum(1.0, rem5 / np.maximum(1, tied)))))

    pred_cat = np.concatenate(pred_all, axis=0)
    pred_opt_cat = np.concatenate(pred_opt_all, axis=0)
    true_cat = np.concatenate(true_all, axis=0)
    ss_res = float(np.sum((true_cat - pred_cat) ** 2))
    ss_tot = float(np.sum((true_cat - true_cat.mean(axis=0, keepdims=True)) ** 2))
    r2_frob = 1.0 - ss_res / max(1e-12, ss_tot)
    r2_opt = 1.0 - float(np.sum((true_cat - pred_opt_cat) ** 2)) / max(1e-12, ss_tot)
    return {
        "r2_frob": round(r2_frob, 5),
        "r2_opt": round(r2_opt, 5),
        "cos": round(float(np.concatenate(cos_all).mean()), 5),
        "top1": round(float(np.mean(top1_list)), 5) if top1_list else None,
        "top5": round(float(np.mean(top5_list)), 5) if top5_list else None,
        "by_domain_cos": {d: round(float(np.mean(v)), 5) for d, v in sorted(dom_cos.items())},
    }


@torch.no_grad()
def eval_blind_alignment(model, tok, d_t):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ev_ids_f, ev_ids_o = d_t["ev_ids_flat"], d_t["ev_ids_off"]
    ev_pts_f, ev_pts_o = d_t["ev_pts_flat"], d_t["ev_pts_off"]
    n_ev = len(ev_ids_o) - 1

    cap = {}
    def make_hook(li):
        def _hk(_m, _i, out):
            cap[li] = out[0] if isinstance(out, tuple) else out
        return _hk

    hooks = [
        model.model.layers[LAYER_30].register_forward_hook(make_hook(LAYER_30)),
        model.model.layers[LAYER_50].register_forward_hook(make_hook(LAYER_50)),
    ]
    h30_list, h50_list = [], []
    try:
        for s in range(0, n_ev, 32):
            e = min(n_ev, s + 32)
            batch_ids = [ev_ids_f[int(ev_ids_o[i]):int(ev_ids_o[i + 1])].astype(np.int64) for i in range(s, e)]
            batch_pts = [ev_pts_f[int(ev_pts_o[i]):int(ev_pts_o[i + 1])].astype(np.int64) for i in range(s, e)]
            ml = max(len(x) for x in batch_ids)
            inp = torch.full((len(batch_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(batch_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(batch_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            c30 = cap[LAYER_30].float().cpu().numpy()
            c50 = cap[LAYER_50].float().cpu().numpy()
            for bi, pts in enumerate(batch_pts):
                h30_list.append(c30[bi, pts, :])
                h50_list.append(c50[bi, pts, :])
    finally:
        for hk in hooks:
            hk.remove()

    H30 = np.concatenate(h30_list, axis=0)
    H50 = np.concatenate(h50_list, axis=0)
    doms, grps, ret_m = d_t["ev_pt_domains"], d_t["ev_pt_groups"], d_t["ev_pt_ret"]

    p4_30 = eval_procrustes_paper_exact(H30, d_t["phi4_30_ev"], grps, doms, ret_m)
    ol_30 = eval_procrustes_paper_exact(H30, d_t["olmo3_30_ev"], grps, doms, ret_m)
    p4_50 = eval_procrustes_paper_exact(H50, d_t["phi4_50_ev"], grps, doms, ret_m)
    ol_50 = eval_procrustes_paper_exact(H50, d_t["olmo3_50_ev"], grps, doms, ret_m)
    return {
        "blind_30_r2_frob": round(0.5 * (p4_30["r2_frob"] + ol_30["r2_frob"]), 5),
        "blind_30_r2_opt": round(0.5 * (p4_30["r2_opt"] + ol_30["r2_opt"]), 5),
        "blind_30_cos": round(0.5 * (p4_30["cos"] + ol_30["cos"]), 5),
        "blind_50_r2_frob": round(0.5 * (p4_50["r2_frob"] + ol_50["r2_frob"]), 5),
        "blind_50_r2_opt": round(0.5 * (p4_50["r2_opt"] + ol_50["r2_opt"]), 5),
        "blind_50_cos": round(0.5 * (p4_50["cos"] + ol_50["cos"]), 5),
        "detail": {"phi4_30": p4_30, "olmo3_30": ol_30, "phi4_50": p4_50, "olmo3_50": ol_50},
    }


# ---------------------------------------------------------------------------
# 3. Exact FP32 Language NLL: WikiText-2 (OOD Wikipedia) & Held-Out Gutenberg
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_token_chunks_fp32(model, pad_id, chunks, batch_size=8):
    """Returns per-chunk (sum_nll, n_tok) in exact fp32 using low-VRAM cross_entropy."""
    dev = next(model.parameters()).device
    out_nll, out_tok = [], []
    for s in range(0, len(chunks), batch_size):
        batch = chunks[s : s + batch_size]
        ml = max(len(x) for x in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            t = torch.tensor(ids, dtype=torch.long, device=dev)
            inp[bi, : len(ids)] = t
            attn[bi, : len(ids)] = 1
        logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        for bi, ids in enumerate(batch):
            L = len(ids) - 1
            tok_nll = F.cross_entropy(logits[bi, :L].float(), inp[bi, 1 : L + 1], reduction="none")
            out_nll.append(float(tok_nll.sum().item()))
            out_tok.append(int(L))
        del logits, inp, attn
    torch.cuda.empty_cache()
    return out_nll, out_tok


def eval_wikitext2_nll(model, tok, data_dir: Path, chunk_len: int = 512):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lines = json.loads((data_dir / "wikitext2_raw_test.json").read_text())
    articles, cur = [], []
    for line in lines:
        if re.match(r"^\s*=\s[^=]+=\s*$", line) and cur:
            txt = "".join(cur).strip()
            if len(txt) >= 100:
                articles.append(txt)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        txt = "".join(cur).strip()
        if len(txt) >= 100:
            articles.append(txt)

    chunks, chunk_art = [], []
    for ai, art in enumerate(articles):
        ids = tok.encode(art, add_special_tokens=False)
        for st in range(0, len(ids), chunk_len):
            ch = ids[st : st + chunk_len]
            if len(ch) >= 2:
                chunks.append(ch)
                chunk_art.append(ai)

    ch_nll, ch_tok = _score_token_chunks_fp32(model, pad_id, chunks, batch_size=16)
    art_nll = np.zeros(len(articles), dtype=np.float64)
    art_tok = np.zeros(len(articles), dtype=np.int64)
    for ai, sn, nt in zip(chunk_art, ch_nll, ch_tok):
        art_nll[ai] += sn
        art_tok[ai] += nt

    tot_nll = float(art_nll.sum())
    tot_tok = int(art_tok.sum())
    per_article = [
        {"article_idx": int(i), "nll": round(float(art_nll[i] / max(1, art_tok[i])), 6), "n_tok": int(art_tok[i])}
        for i in range(len(articles)) if art_tok[i] > 0
    ]
    return {
        "nll": round(tot_nll / max(1, tot_tok), 6),
        "ppl": round(float(np.exp(tot_nll / max(1, tot_tok))), 4),
        "n_articles": len(per_article),
        "n_tokens": tot_tok,
        "per_article": per_article,
    }


def eval_held_gut_nll(model, tok, plato_dir: Path):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ev_rows = [json.loads(l) for l in open(plato_dir / "eval_passages.jsonl") if l.strip()]
    gut_rows = [r for r in ev_rows if r["domain"] == "held_gut"][:100]
    chunks = [tok.encode(r["text"], add_special_tokens=False) for r in gut_rows]
    valid_idx = [i for i, ids in enumerate(chunks) if len(ids) >= 2]
    chunks = [chunks[i] for i in valid_idx]
    ch_nll, ch_tok = _score_token_chunks_fp32(model, pad_id, chunks, batch_size=16)
    tot_nll = float(sum(ch_nll))
    tot_tok = int(sum(ch_tok))
    per_passage = [
        {"idx": int(valid_idx[i]), "nll": round(float(ch_nll[i] / max(1, ch_tok[i])), 6), "n_tok": int(ch_tok[i])}
        for i in range(len(chunks))
    ]
    return {
        "nll": round(tot_nll / max(1, tot_tok), 6),
        "n_passages": len(per_passage),
        "n_tokens": tot_tok,
        "per_passage": per_passage,
    }


# ---------------------------------------------------------------------------
# 4. Balanced ALL-240 Actor2x2 & Symmetric ALL-240 SWE-Smith & Proc-200 MCQ
# ---------------------------------------------------------------------------

def _symmetric_edit_window_and_masks(ids_p, ids_n, max_len=1024, pad_ctx=8):
    """Find exact diff on UNTRUNCATED token sequences (`autojunk=False`), center a
    symmetric <=max_len window around the diff span, and return symmetric +/-pad_ctx
    edit-site masks (zero asymmetric empty-mask fallback on pure deletions!)."""
    sm = difflib.SequenceMatcher(None, ids_p, ids_n, autojunk=False)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if opcodes:
        p_lo = min(op[1] for op in opcodes)
        p_hi = max(op[2] for op in opcodes)
        n_lo = min(op[3] for op in opcodes)
        n_hi = max(op[4] for op in opcodes)
    else:
        p_lo, p_hi, n_lo, n_hi = 0, len(ids_p), 0, len(ids_n)

    def _crop(ids, lo, hi):
        if len(ids) <= max_len:
            return ids, 0
        mid = (lo + hi) // 2
        st = max(0, min(len(ids) - max_len, mid - max_len // 2))
        return ids[st : st + max_len], st

    cp, off_p = _crop(ids_p, p_lo, p_hi)
    cn, off_n = _crop(ids_n, n_lo, n_hi)
    if len(cp) < 2: cp = cp + [cp[0] if cp else 0]
    if len(cn) < 2: cn = cn + [cn[0] if cn else 0]

    # Target token positions correspond to indices 1..len-1 (offset by -1 in lp[:, :-1])
    mp = np.zeros(len(cp) - 1, dtype=bool)
    mn = np.zeros(len(cn) - 1, dtype=bool)
    for _, i1, i2, j1, j2 in opcodes:
        pi1 = max(0, (i1 - off_p - 1) - pad_ctx)
        pi2 = min(len(mp), max(i2 - off_p - 1, i1 - off_p - 1) + pad_ctx)
        nj1 = max(0, (j1 - off_n - 1) - pad_ctx)
        nj2 = min(len(mn), max(j2 - off_n - 1, j1 - off_n - 1) + pad_ctx)
        if pi2 > pi1:
            mp[pi1:pi2] = True
        if nj2 > nj1:
            mn[nj1:nj2] = True
    if not mp.any(): mp[:] = True
    if not mn.any(): mn[:] = True
    return cp, cn, mp, mn


@torch.no_grad()
def eval_swe_240_symmetric(model, tok, swe_path: Path):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [json.loads(l) for l in open(swe_path) if l.strip()]

    seqs, edit_masks = [], []
    for r in rows:
        ip = tok.encode(r["before"].strip(), add_special_tokens=False)
        in_ = tok.encode(r["after"].strip(), add_special_tokens=False)
        cp, cn, mp, mn = _symmetric_edit_window_and_masks(ip, in_, max_len=1024, pad_ctx=8)
        seqs.extend([cp, cn])
        edit_masks.extend([mp, mn])

    full_nlls = np.zeros(len(seqs), dtype=np.float64)
    edit_nlls = np.zeros(len(seqs), dtype=np.float64)
    for s in range(0, len(seqs), 4):
        batch = seqs[s : s + 4]
        ml = max(len(x) for x in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[bi, : len(ids)] = 1
        logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        for bi, ids in enumerate(batch):
            L = len(ids) - 1
            tok_nll = F.cross_entropy(logits[bi, :L].float(), inp[bi, 1 : L + 1], reduction="none").cpu().numpy()
            full_nlls[s + bi] = float(tok_nll.mean())
            edit_nlls[s + bi] = float(tok_nll[edit_masks[s + bi][:L]].mean())
        del logits, inp, attn
    torch.cuda.empty_cache()

    per_pair = []
    for idx, r in enumerate(rows):
        fm = float(full_nlls[2 * idx + 1] - full_nlls[2 * idx])
        em = float(edit_nlls[2 * idx + 1] - edit_nlls[2 * idx])
        per_pair.append({
            "id": r.get("id", idx),
            "repo": r.get("repo", ""),
            "cls": r.get("cls", ""),
            "full_margin": round(fm, 6),
            "full_ok": bool(fm > 0.0),
            "edit_margin": round(em, 6),
            "edit_ok": bool(em > 0.0),
        })
    return {
        "swe_full_acc": round(float(np.mean([p["full_ok"] for p in per_pair])), 5),
        "swe_full_margin": round(float(np.mean([p["full_margin"] for p in per_pair])), 6),
        "swe_edit_acc": round(float(np.mean([p["edit_ok"] for p in per_pair])), 5),
        "swe_edit_margin": round(float(np.mean([p["edit_margin"] for p in per_pair])), 6),
        "n_pairs": len(per_pair),
        "per_pair": per_pair,
    }


@torch.no_grad()
def eval_actor_240_balanced(model, tok, probes_dir: Path):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = json.loads((probes_dir / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]

    seqs, masks = [], []
    for r in rows:
        ip = tok.encode(r["pos"].strip(), add_special_tokens=False)[:256]
        in_ = tok.encode(r["neg"].strip(), add_special_tokens=False)[:256]
        sm = difflib.SequenceMatcher(None, ip[1:], in_[1:], autojunk=False)
        mp = np.zeros(max(1, len(ip) - 1), dtype=bool)
        mn = np.zeros(max(1, len(in_) - 1), dtype=bool)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                mp[i1:i2] = True
                mn[j1:j2] = True
        if not mp.any(): mp[:] = True
        if not mn.any(): mn[:] = True
        seqs.extend([ip, in_])
        masks.extend([mp, mn])

    nlls = np.zeros(len(seqs), dtype=np.float64)
    for s in range(0, len(seqs), 16):
        batch = seqs[s : s + 16]
        ml = max(len(x) for x in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[bi, : len(ids)] = 1
        logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        for bi, ids in enumerate(batch):
            L = len(ids) - 1
            tok_nll = F.cross_entropy(logits[bi, :L].float(), inp[bi, 1 : L + 1], reduction="none").cpu().numpy()
            nlls[s + bi] = float(tok_nll[masks[s + bi][:L]].mean())
        del logits, inp, attn
    torch.cuda.empty_cache()

    per_item = []
    for i, r in enumerate(rows):
        margin = float(nlls[2 * i + 1] - nlls[2 * i])
        per_item.append({
            "id": r.get("id", i),
            "order": r.get("order", ""),
            "fav_pos": r.get("fav_pos", ""),
            "margin": round(margin, 6),
            "ok": bool(margin > 0.0),
        })
    # Mirror-consistent pairs: consecutive (2k, 2k+1) are orig/mirror twins
    mirror_ok = [bool(per_item[2 * k]["ok"] and per_item[2 * k + 1]["ok"]) for k in range(len(per_item) // 2)]
    return {
        "actor_acc_240": round(float(np.mean([x["ok"] for x in per_item])), 5),
        "actor_margin_240": round(float(np.mean([x["margin"] for x in per_item])), 6),
        "actor_mirror_consistent": round(float(np.mean(mirror_ok)), 5),
        "n_items": len(per_item),
        "per_item": per_item,
    }


@torch.no_grad()
def eval_proc_mcq_200(model, tok, probes_dir: Path):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    proc_all = [json.loads(l) for l in open(probes_dir / "proc_held.jsonl") if l.strip()]
    by_k = {}
    for r in proc_all:
        by_k.setdefault(r["kind"], []).append(r)
    proc_rows = [r for k in sorted(by_k.keys()) for r in by_k[k][:40]]

    seqs, meta = [], []
    for ri, r in enumerate(proc_rows):
        prefix = f"{r['passage'].strip()}\nQuestion: {r['question'].strip()}\nAnswer:"
        p_ids = tok.encode(prefix, add_special_tokens=False)
        for opt_txt, is_ans in [(r["answer"], True)] + [(w, False) for w in r["wrong"]]:
            o_ids = tok.encode(" " + opt_txt.strip(), add_special_tokens=False) or tok.encode(opt_txt.strip(), add_special_tokens=False)
            full = p_ids + o_ids
            seqs.append(full)
            meta.append((ri, is_ans, len(p_ids), len(full)))

    scores = np.zeros(len(seqs), dtype=np.float64)
    for s in range(0, len(seqs), 16):
        batch = seqs[s : s + 16]
        ml = max(len(x) for x in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[bi, : len(ids)] = 1
        logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        for bi in range(len(batch)):
            _, _, st, en = meta[s + bi]
            tok_nll = F.cross_entropy(logits[bi, st - 1 : en - 1].float(), inp[bi, st:en], reduction="none")
            scores[s + bi] = float(-tok_nll.mean().item())
        del logits, inp, attn
    torch.cuda.empty_cache()

    by_row = {}
    for idx, (ri, is_ans, _, _) in enumerate(meta):
        by_row.setdefault(ri, []).append((scores[idx], is_ans))
    per_item, by_kind = [], {}
    for ri, r in enumerate(proc_rows):
        opts = by_row[ri]
        ans_sc = [sc for sc, isa in opts if isa][0]
        best_wrong = max(sc for sc, isa in opts if not isa)
        ok = bool(ans_sc > best_wrong)
        per_item.append({"idx": ri, "kind": r["kind"], "ok": ok, "margin": round(float(ans_sc - best_wrong), 6)})
        by_kind.setdefault(r["kind"], [0, 0])
        by_kind[r["kind"]][0] += int(ok)
        by_kind[r["kind"]][1] += 1
    return {
        "proc_acc_200": round(float(np.mean([x["ok"] for x in per_item])), 5),
        "by_kind": {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(by_kind.items())},
        "n_items": len(per_item),
        "per_item": per_item,
    }


def evaluate_all(model, tok, d_t, paths=None):
    paths = paths or default_paths()
    t0 = time.time()
    model.eval()

    align = eval_blind_alignment(model, tok, d_t)
    wt2 = eval_wikitext2_nll(model, tok, paths["data_dir"])
    gut = eval_held_gut_nll(model, tok, paths["plato_dir"])
    swe = eval_swe_240_symmetric(model, tok, paths["swe_path"])
    actor = eval_actor_240_balanced(model, tok, paths["probes_dir"])
    proc = eval_proc_mcq_200(model, tok, paths["probes_dir"])
    he = eval_humaneval(model, tok, paths["data_dir"])
    mbpp = eval_mbpp_test(model, tok, paths["data_dir"])

    summary = {
        "evaluator_sha256": evaluator_sha256(),
        "eval_dtype": str(next(model.parameters()).dtype),
        "wikitext2_nll": wt2["nll"],
        "wikitext2_ppl": wt2["ppl"],
        "held_gut_nll": gut["nll"],
        "humaneval_pass1": he["pass_at_1"],
        "humaneval_hits": f"{he['hits']}/{he['total']}",
        "mbpp_test_pass1": mbpp["pass_at_1"],
        "mbpp_test_hits": f"{mbpp['hits']}/{mbpp['total']}",
        "swe_full_acc": swe["swe_full_acc"],
        "swe_full_margin": swe["swe_full_margin"],
        "swe_edit_acc": swe["swe_edit_acc"],
        "swe_edit_margin": swe["swe_edit_margin"],
        "actor_acc_240": actor["actor_acc_240"],
        "actor_margin_240": actor["actor_margin_240"],
        "actor_mirror_consistent": actor["actor_mirror_consistent"],
        "proc_acc_200": proc["proc_acc_200"],
        "blind_30_r2_frob": align["blind_30_r2_frob"],
        "blind_30_r2_opt": align["blind_30_r2_opt"],
        "blind_30_cos": align["blind_30_cos"],
        "blind_50_r2_frob": align["blind_50_r2_frob"],
        "blind_50_r2_opt": align["blind_50_r2_opt"],
        "blind_50_cos": align["blind_50_cos"],
        "eval_sec": round(time.time() - t0, 1),
    }
    records = {
        "summary": summary,
        "align_detail": align["detail"],
        "wikitext2_per_article": wt2["per_article"],
        "held_gut_per_passage": gut["per_passage"],
        "swe_per_pair": swe["per_pair"],
        "actor_per_item": actor["per_item"],
        "proc_per_item": proc["per_item"],
        "proc_by_kind": proc["by_kind"],
        "humaneval_per_task_bool": he["per_task_bool"],
        "mbpp_per_task_bool": mbpp["per_task_bool"],
    }
    generations = {
        "humaneval": he["per_task_full"],
        "mbpp_test": mbpp["per_task_full"],
    }
    return records, generations
