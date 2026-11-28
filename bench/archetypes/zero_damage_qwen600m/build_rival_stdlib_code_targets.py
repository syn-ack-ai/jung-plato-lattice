"""Extract 8B Rival (`google/gemma-4-E4B-it`, ZERO Qwen!) 30% (L12) & 50% (L20) hidden-state geometry
on ~400 clean Python standard-library code snippets (100% disjoint from our 40 eval tasks & 240 SWE-Smith repos!).

Run on `dreamer` (RTX 4090, ~15s). Saves `stdlib_code_gemma4_8b_targets.npz` for Dual-Domain
(Prose + Code) Pure-Geometry Shape Transfer (ce_w = 0.0!).
"""
import gc
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ARCH = Path("/home/skypanther/systemone/bench/archetypes")
QDIR = ARCH / "exp256_q600"
OUT_NPZ = QDIR / "stdlib_code_gemma4_8b_targets.npz"

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from buddy import e4b_living as el  # noqa: E402

STDLIB_MODS = [
    "argparse", "ast", "bisect", "calendar", "collections", "configparser", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "difflib", "enum", "fnmatch",
    "fractions", "functools", "getopt", "glob", "gzip", "hashlib", "heapq", "hmac",
    "html", "inspect", "ipaddress", "itertools", "json", "logging", "mimetypes", "numbers",
    "operator", "pathlib", "pprint", "queue", "random", "re", "sched", "secrets",
    "shlex", "shutil", "statistics", "string", "tarfile", "tempfile", "textwrap",
    "threading", "timeit", "tokenize", "traceback", "typing", "unittest", "urllib",
    "uuid", "warnings", "weakref", "zipfile",
]


def collect_stdlib_snippets(max_snippets=420):
    """Collect clean 350-900 char Python function/class blocks from Python's own stdlib."""
    stdlib_dir = Path(inspect.getfile(json := __import__("json"))).parent.parent
    snippets = []
    seen = set()
    for mod in STDLIB_MODS:
        cand = stdlib_dir / f"{mod}.py"
        if not cand.exists():
            cand = stdlib_dir / mod / "__init__.py"
        if not cand.exists():
            continue
        try:
            src = cand.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # Split into 4-segment chunks on blank-line / def boundaries
        lines = src.splitlines(keepends=True)
        cur = []
        cur_len = 0
        for ln in lines:
            cur.append(ln)
            cur_len += len(ln)
            if cur_len >= 650 and ln.strip() == "":
                txt = "".join(cur).strip()
                cur, cur_len = [], 0
                if 320 <= len(txt) <= 1100 and ("def " in txt or "class " in txt):
                    key = txt[:80]
                    if key not in seen:
                        seen.add(key)
                        # Split into 4 line-aligned segments for boundary probes
                        s_lines = txt.splitlines(keepends=True)
                        if len(s_lines) >= 8:
                            step = len(s_lines) // 4
                            segs = [
                                "".join(s_lines[0:step]),
                                "".join(s_lines[step : 2 * step]),
                                "".join(s_lines[2 * step : 3 * step]),
                                "".join(s_lines[3 * step :]),
                            ]
                            if all(len(s.strip()) > 15 for s in segs):
                                snippets.append(segs)
                                if len(snippets) >= max_snippets:
                                    return snippets
    return snippets


def boundary_pts(tok, segs):
    full_ids = tok.encode("".join(segs), add_special_tokens=False)
    if len(full_ids) < 8:
        return full_ids, []
    pts, acc = [], ""
    for j, s in enumerate(segs):
        acc += s
        if j == len(segs) - 1:
            idx = len(full_ids) - 1
        else:
            p_ids = tok.encode(acc.rstrip(), add_special_tokens=False)
            idx = max(1, min(len(full_ids) - 1, len(p_ids) - 1))
        pts.append(idx if (not pts or idx > pts[-1]) else min(len(full_ids) - 1, pts[-1]))
    return full_ids, pts


@torch.no_grad()
def main():
    t0 = time.time()
    snippets = collect_stdlib_snippets(max_snippets=420)
    print(f"Collected {len(snippets)} clean Python stdlib snippets")

    # Tokenize with Qwen3-0.6B tokenizer first so we drop any snippet > 360 Qwen tokens
    q_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    kept = []
    q_ids_list, q_pts_list = [], []
    q_ids_off, q_pts_off = [0], [0]
    for segs in snippets:
        q_ids, q_pts = boundary_pts(q_tok, segs)
        if 16 <= len(q_ids) <= 360 and len(q_pts) == 4:
            kept.append(segs)
            q_ids_list.extend(q_ids)
            q_pts_list.extend(q_pts)
            q_ids_off.append(len(q_ids_list))
            q_pts_off.append(len(q_pts_list))
    print(f"Kept {len(kept)} stdlib snippets ({len(q_pts_list)} boundary points)")

    # Extract 8B Rival Gemma 4 E4B L12 (30%) & L20 (50%) hidden states!
    e4b = el.load_living_e4b(slice_path=None, with_plato_head=False, offload_mm=False)
    if 20 not in e4b._hooked_layers:
        e4b._hooks.append(e4b.lm.layers[20].register_forward_hook(e4b._make_hook(20)))
        e4b._hooked_layers.add(20)
    e4b.eval()
    dev = next(e4b.lm.parameters()).device
    g_tok = e4b.tok
    bos = g_tok.bos_token_id
    pad_id = g_tok.pad_token_id if g_tok.pad_token_id is not None else (g_tok.eos_token_id or 0)

    h12_list, h20_list = [], []
    bs = 12
    for s in range(0, len(kept), bs):
        batch = kept[s : s + bs]
        g_ids_b, g_pts_b = [], []
        for segs in batch:
            ids, pts = boundary_pts(g_tok, segs)
            if bos is not None:
                ids = [bos] + ids
                pts = [p + 1 for p in pts]
            g_ids_b.append(ids)
            g_pts_b.append(pts)
        ml = max(len(x) for x in g_ids_b)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(g_ids_b):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[bi, : len(ids)] = 1
        with e4b.run_ctx(enabled=False, state=None, update=False, capture_layers=(12, 20)) as ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = e4b.base.model(input_ids=inp, attention_mask=attn, use_cache=False)
            c12 = ctx["capture"][12].float().cpu().numpy()
            c20 = ctx["capture"][20].float().cpu().numpy()
        for bi, pts in enumerate(g_pts_b):
            h12_list.append(c12[bi, pts, :])
            h20_list.append(c20[bi, pts, :])

    H12 = np.concatenate(h12_list, axis=0).astype(np.float64)
    H20 = np.concatenate(h20_list, axis=0).astype(np.float64)

    def pca256(X):
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ vt[:256].T
        sc = float(np.sqrt(np.mean(np.sum(Z ** 2, axis=1)))) + 1e-12
        return (Z / sc).astype(np.float32)

    g4_code_30 = pca256(H12)
    g4_code_50 = pca256(H20)

    np.savez_compressed(
        OUT_NPZ,
        code_ids_flat=np.array(q_ids_list, dtype=np.int32),
        code_ids_off=np.array(q_ids_off, dtype=np.int64),
        code_pts_flat=np.array(q_pts_list, dtype=np.int32),
        code_pts_off=np.array(q_pts_off, dtype=np.int64),
        g4_code_30=g4_code_30,
        g4_code_50=g4_code_50,
    )
    print(f"Saved {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.1f} MB) | {len(kept)} stdlib snippets in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
