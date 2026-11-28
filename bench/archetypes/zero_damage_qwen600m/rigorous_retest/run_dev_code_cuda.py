"""Fast CUDA Strictly Disjoint-Dev Autoresearch Sweep for Exp 258 (Python-Coding Platonic Lattice).
Runs SOLO on GPU (~1.7 min per hypothesis!) with ZERO peeking at locked holdouts
(`HumanEval 164`, `MBPP test 257`, `WikiText-2`).

Evaluates strictly on disjoint dev benchmarks (`bwrap --unshare-all` + `prlimit` OS sandbox):
  - `dev_mbpp_170`: ALL 170 non-test `sanitized-mbpp.json` tasks (`task_id <= 10` or `task_id >= 511`)
  - `dev_code_nll`: Held-out `code_eval.jsonl` (399 algorithmic Python solutions, exact fp32 NLL)
  - `dev_code_blind_align`: Exact `plato_data.py` OOF `r2_frob` against strictly blind `Phi-4-mini`
    and `OLMo 3 7B` at `30% (L8)`, `50% (L14)`, and `70% (L19)`!

ALL hypotheses strictly enforce `ce_w = 0.0` (ZERO next-token CE, ZERO teacher logits, ZERO Qwen!).
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
EXP257_DATA = ROOT.parent / "exp257_rigorous" / "data"
TARGETS_NPZ = ROOT / "targets_qwen600m_code_lattice_30_50_70.npz"
OUT_JSON = ROOT / "dev_code_autoresearch_results.json"

LAYERS_MAP = {30: 8, 50: 14, 70: 19}


def _sandbox_cmd():
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    if not bwrap:
        raise RuntimeError("bwrap required!")
    py_prefix = str(Path(sys.prefix).resolve())
    py_rel = str(Path(sys.executable).resolve().relative_to(Path(py_prefix)))
    cmd = [
        bwrap, "--unshare-all",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/bin", "/bin",
    ]
    if Path("/lib64").exists():
        cmd.extend(["--ro-bind", "/lib64", "/lib64"])
    cmd.extend([
        "--ro-bind", py_prefix, "/opt/py",
        "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
        "--chdir", "/tmp", "--die-with-parent", "--",
    ])
    if prlimit:
        cmd.extend([prlimit, "--as=536870912", "--cpu=3", "--fsize=1048576", "--"])
    cmd.extend([f"/opt/py/{py_rel}", "-I", "-S", "-c"])
    return cmd


_SB = None


def run_sandbox(script: str, timeout_s: float = 4.0):
    global _SB
    if _SB is None:
        _SB = _sandbox_cmd()
    try:
        r = subprocess.run([*_SB, script], capture_output=True, text=True, timeout=timeout_s)
        return r.returncode, r.stdout, r.stderr
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
    return cleaned[m.start():].strip() if m else cleaned


def extract_mbpp_sigs(code: str) -> str:
    return "\n".join(re.findall(r"^(?:def|class)\s+\w+.*:", code, flags=re.MULTILINE))


def grade_mbpp_one(task: dict, raw_gen: str):
    code = extract_python_code(raw_gen)
    sent = f"__DEV258_{secrets.token_hex(10)}__"
    imp = "\n".join(task.get("test_imports", []))
    tst = "\n".join(task["test_list"])
    rc, out, err = run_sandbox(f"{imp}\n{code}\n\n{tst}\nprint({sent!r})\n")
    ok = (rc == 0) and (sent in out.splitlines())
    return {"task_id": int(task["task_id"]), "ok": bool(ok)}


def _fit_proc(A, B):
    ma, mb = A.mean(0, keepdims=True), B.mean(0, keepdims=True)
    Ac, Bc = A - ma, B - mb
    U, _, Vt = np.linalg.svd(Bc.T @ Ac, full_matrices=False)
    R = U @ Vt
    s = float(np.linalg.norm(Bc, "fro") / max(1e-12, np.linalg.norm(Ac @ R.T, "fro")))
    return ma, mb, R, s


def _apply_proc(A, p):
    ma, mb, R, s = p
    return mb + s * ((A - ma) @ R.T)


def _pca_proj(X_tr, X_ev, d_out=256):
    mu = X_tr.mean(0, keepdims=True)
    Xc = X_tr - mu
    vals, vecs = np.linalg.eigh((Xc.T @ Xc) / float(len(Xc)))
    V = vecs[:, np.argsort(vals)[::-1][:d_out]]
    Z_tr, Z_ev = Xc @ V, (X_ev - mu) @ V
    rms = max(1e-12, float(np.sqrt(np.mean(Z_tr ** 2))))
    return Z_tr / rms, Z_ev / rms


@torch.no_grad()
def evaluate_dev(model, tok, d_c, mbpp_dev_tasks):
    t0 = time.time()
    model.eval()
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # 1. Held-out code_eval NLL + blind Phi-4 & OLMo 3 alignment at 30/50/70%
    ev_f, ev_o = d_c["ev_ids_flat"], d_c["ev_ids_off"]
    pt_f, pt_o = d_c["ev_pts_flat"], d_c["ev_pts_off"]
    n_ev = len(ev_o) - 1
    cap = {}
    hooks = [
        model.model.layers[li].register_forward_hook(
            lambda _m, _i, out, idx=li: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for li in LAYERS_MAP.values()
    ]
    H = {dep: [] for dep in LAYERS_MAP}
    tot_nll, tot_tok = 0.0, 0
    try:
        for s in range(0, n_ev, 8):
            e = min(n_ev, s + 8)
            b_ids = [ev_f[int(ev_o[i]):int(ev_o[i + 1])].astype(np.int64) for i in range(s, e)]
            b_pts = [pt_f[int(pt_o[i]):int(pt_o[i + 1])].astype(np.int64) for i in range(s, e)]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
            for bi, ids in enumerate(b_ids):
                L = len(ids) - 1
                tot_nll += float(F.cross_entropy(logits[bi, :L].float(), inp[bi, 1 : L + 1], reduction="sum").item())
                tot_tok += int(L)
            for dep, li in LAYERS_MAP.items():
                arr = cap[li].float().cpu().numpy()
                for bi, pts in enumerate(b_pts):
                    H[dep].append(arr[bi, pts, :])
            del logits, inp, attn
    finally:
        for hk in hooks:
            hk.remove()
    torch.cuda.empty_cache()

    grps = d_c["ev_pt_groups"].tolist()
    uniq = sorted(set(grps))
    rng = random.Random(2588); rng.shuffle(uniq)
    g_fold = {g: (0 if i < len(uniq) // 2 else 1) for i, g in enumerate(uniq)}
    folds = np.array([g_fold[g] for g in grps], dtype=np.int32)
    align = {}
    for dep in (30, 50, 70):
        X = np.concatenate(H[dep], axis=0).astype(np.float64)
        vals = []
        for bk in ("phi4_mini", "olmo3_7b"):
            Y = d_c[f"{bk}_{dep}_ev"].astype(np.float64)
            pa, ta = [], []
            for tf in (0, 1):
                tr, te = folds != tf, folds == tf
                Z_tr, Z_te = _pca_proj(X[tr], X[te], d_out=256)
                pa.append(_apply_proc(Z_te, _fit_proc(Z_tr, Y[tr])))
                ta.append(Y[te])
            pc, tc = np.concatenate(pa), np.concatenate(ta)
            r2 = 1.0 - float(np.sum((tc - pc) ** 2)) / max(1e-12, float(np.sum((tc - tc.mean(0, keepdims=True)) ** 2)))
            vals.append(r2)
        align[f"code_blind_{dep}_r2"] = round(float(np.mean(vals)), 5)

    # 2. Disjoint dev_mbpp_170 generation (exact fp32, bs=24) + bwrap grading
    orig_side = tok.padding_side
    tok.padding_side = "left"
    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    prompts = []
    for t in mbpp_dev_tasks:
        sigs = extract_mbpp_sigs(t["code"])
        imp = ("\nRequired imports:\n" + "\n".join(t["test_imports"])) if t.get("test_imports") else ""
        prompts.append(
            tok.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{sigs}\n```{imp}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )
    gens = []
    try:
        for s in range(0, len(prompts), 24):
            bp = prompts[s : s + 24]
            enc = tok(bp, return_tensors="pt", padding=True).to(dev)
            in_len = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
            for bi in range(len(bp)):
                gens.append(tok.decode(out[bi, in_len:], skip_special_tokens=True))
            del out, enc
            torch.cuda.empty_cache()
    finally:
        tok.padding_side = orig_side

    with ThreadPoolExecutor(max_workers=12) as ex:
        per_t = list(ex.map(lambda p: grade_mbpp_one(p[0], p[1]), zip(mbpp_dev_tasks, gens)))
    hits = sum(1 for r in per_t if r["ok"])
    return {
        "dev_code_nll": round(tot_nll / max(1, tot_tok), 6),
        "dev_mbpp_pass1": round(hits / len(per_t), 5),
        "dev_mbpp_hits": f"{hits}/{len(per_t)}",
        "dev_mbpp_hits_int": hits,
        **align,
        "eval_sec": round(time.time() - t0, 1),
    }


class LoRALinearWrap(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 24, alpha: float = 24.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        dev = base_linear.weight.device
        self.scale = alpha / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.active = True

    def forward(self, x):
        out = self.base(x)
        if self.active:
            delta = (x.float() @ self.lora_A.T) @ self.lora_B.T
            out = out + (delta * self.scale).to(out.dtype)
        return out

    @torch.no_grad()
    def delta_fro(self):
        return float(torch.norm((self.lora_B @ self.lora_A) * self.scale, p="fro").item())

    @torch.no_grad()
    def merge_into_base(self):
        dw = (self.lora_B @ self.lora_A) * self.scale
        self.base.weight.data.add_(dw.to(self.base.weight.dtype))
        return self.base


def global_frob_cos_gram_loss(pred, target):
    p_u = F.normalize(pred, dim=-1, eps=1e-6)
    t_u = F.normalize(target, dim=-1, eps=1e-6)
    cos_l = (1.0 - (p_u * t_u).sum(dim=-1)).mean()
    if pred.shape[0] >= 4:
        pc = pred - pred.mean(dim=0, keepdim=True)
        tc = target - target.mean(dim=0, keepdim=True)
        pn = pc / (torch.norm(pc) + 1e-6)
        tn = tc / (torch.norm(tc) + 1e-6)
        frob_l = torch.sum((pn - tn) ** 2)
        gram_l = F.mse_loss(p_u @ p_u.T, t_u @ t_u.T)
    else:
        frob_l = pred.new_tensor(0.0)
        gram_l = pred.new_tensor(0.0)
    return 0.45 * cos_l + 0.90 * frob_l + 0.45 * gram_l


def shuffle_within_seg_index(tgts_by_dep, pts_off, seed=258):
    rng = np.random.default_rng(seed + 5555)
    buckets = defaultdict(list)
    for pi in range(len(pts_off) - 1):
        for si, row_i in enumerate(range(int(pts_off[pi]), int(pts_off[pi + 1]))):
            buckets[si].append(row_i)
    out = {dep: arr.copy() for dep, arr in tgts_by_dep.items()}
    for si, idxs in sorted(buckets.items()):
        arr_i = np.array(idxs, dtype=np.int64)
        perm = rng.permutation(arr_i)
        for dep in out:
            out[dep][arr_i] = tgts_by_dep[dep][perm]
    return out


@torch.no_grad()
def fit_native_frames(model, d_c, tgts_by_layer, pad_id, dev, n_fit=1200):
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_tot = len(ids_o) - 1
    rng = np.random.default_rng(2580)
    fit_idx = sorted(rng.choice(n_tot, size=min(n_fit, n_tot), replace=False).tolist())
    layers = sorted(tgts_by_layer.keys())
    cap = {}
    hooks = [
        model.model.layers[li].register_forward_hook(
            lambda _m, _i, out, idx=li: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for li in layers
    ]
    Xs, Ys = {li: [] for li in layers}, {li: [] for li in layers}
    try:
        for s in range(0, len(fit_idx), 32):
            sub = fit_idx[s : s + 32]
            b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64) for i in sub]
            b_pts = [pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64) for i in sub]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            for li in layers:
                h = cap[li].float().cpu().numpy()
                tgt = tgts_by_layer[li]
                for bi, (i_abs, pts) in enumerate(zip(sub, b_pts)):
                    Xs[li].append(h[bi, pts, :])
                    Ys[li].append(tgt[int(pts_o[i_abs]):int(pts_o[i_abs + 1])])
    finally:
        for hk in hooks:
            hk.remove()
    frames = {}
    for li in layers:
        X = np.concatenate(Xs[li], axis=0).astype(np.float64)
        Y = np.concatenate(Ys[li], axis=0).astype(np.float64)
        mu_x = X.mean(0, keepdims=True)
        Xc = X - mu_x
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        V_q = vt[: Y.shape[1]].T
        Zx = Xc @ V_q
        s_x = float(np.sqrt(np.mean(np.sum(Zx ** 2, axis=1)))) + 1e-12
        rms_x = float(np.sqrt(np.mean(Xc ** 2))) + 1e-12
        mu_y = Y.mean(0, keepdims=True)
        Yc = Y - mu_y
        s_y = float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1)))) + 1e-12
        U, _, Vt = np.linalg.svd((Yc / s_y).T @ (Zx / s_x), full_matrices=False)
        R_star = U @ Vt
        frames[li] = {
            "mu_x": torch.from_numpy(mu_x.squeeze(0).astype(np.float32)).to(dev),
            "V_q": torch.from_numpy(V_q.astype(np.float32)).to(dev),
            "mu_y": torch.from_numpy(mu_y.squeeze(0).astype(np.float32)).to(dev),
            "R_star": torch.from_numpy((R_star * (s_x / s_y)).astype(np.float32)).to(dev),
            "rms_x": rms_x,
        }
    return frames


def train_hypothesis(model, tok, d_c, cfg: dict, seed: int = 258):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    depths = cfg["depths"]
    weights = cfg["weights"]
    active_layers = cfg["active_layers"]
    modules = cfg["modules"]
    rank = cfg["rank"]
    steps = cfg["steps"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    kl_w = cfg["kl_w"]
    kl_hinge = cfg.get("kl_hinge", 0.0)
    kl_floor = cfg.get("kl_floor", 0.0)
    perp_w = cfg["perp_w"]
    use_pcgrad = cfg.get("pcgrad", True)
    shuffled = cfg.get("shuffled", False)

    tgts_by_dep = {dep: d_c[f"code_r3_{dep}_tr"] for dep in depths}
    if shuffled:
        tgts_by_dep = shuffle_within_seg_index(tgts_by_dep, d_c["tr_pts_off"], seed=seed)
    tgts_by_layer = {LAYERS_MAP[dep]: tgts_by_dep[dep] for dep in depths}
    frames = fit_native_frames(model, d_c, tgts_by_layer, pad_id, dev, n_fit=1200)

    for p in model.parameters():
        p.requires_grad_(False)
    wrappers, all_lora = [], []
    for li in active_layers:
        layer = model.model.layers[li]
        mmap = {"v_proj": (layer.self_attn, "v_proj"), "o_proj": (layer.self_attn, "o_proj"), "down_proj": (layer.mlp, "down_proj")}
        for mname in modules:
            parent, attr = mmap[mname]
            w = LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((parent, attr, w))
            all_lora.extend([w.lora_A, w.lora_B])

    cap = {}
    hooks = [
        model.model.layers[LAYERS_MAP[dep]].register_forward_hook(
            lambda _m, _i, out, idx=LAYERS_MAP[dep]: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for dep in depths
    ]
    opt = torch.optim.AdamW(all_lora, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_ex = len(ids_o) - 1
    perm, cur = rng.permutation(n_ex), 0

    history = []
    completed = False
    tot_fro = 0.0
    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 20.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul
            pw_scale = 1.0 if frac < 0.78 else (max(0.0, 1.0 - (frac - 0.78) / 0.17) if frac < 0.95 else 0.0)

            batch = []
            while len(batch) < batch_size:
                if cur >= n_ex:
                    perm, cur = rng.permutation(n_ex), 0
                pi = int(perm[cur]); cur += 1
                ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                if len(ids) >= 4 and len(pts) >= 1:
                    ts = {dep: tgts_by_dep[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in depths}
                    batch.append((ids, pts, ts))

            ml = max(len(b[0]) for b in batch)
            B = len(batch)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((B, ml), dtype=torch.long, device=dev)
            valid_tok = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
            b_idx_l, t_idx_l = [], []
            yt_lists = {dep: [] for dep in depths}
            for bi, (ids, pts, ts) in enumerate(batch):
                L = len(ids)
                inp[bi, :L] = torch.from_numpy(ids).to(dev)
                attn[bi, :L] = 1
                valid_tok[bi, : L - 1] = 1.0
                b_idx_l.append(torch.full((len(pts),), bi, dtype=torch.long, device=dev))
                t_idx_l.append(torch.from_numpy(pts).to(dev))
                for dep in depths:
                    yt_lists[dep].append(ts[dep])

            b_idx, t_idx = torch.cat(b_idx_l), torch.cat(t_idx_l)
            y_t = {dep: torch.from_numpy(np.concatenate(yt_lists[dep], axis=0)).to(dev, dtype=torch.float32) for dep in depths}
            v_sum = valid_tok.sum().clamp_min(1.0)

            for _, _, w in wrappers:
                w.active = False
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_r = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
                lp_ref = F.log_softmax(lg_r.float(), dim=-1)
                p_ref = lp_ref.exp()
                perp_b = {}
                for dep in depths:
                    li = LAYERS_MAP[dep]
                    fr = frames[li]
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z = hc @ fr["V_q"]
                    perp_b[li] = (hc - z @ fr["V_q"].T).detach()
                del lg_r
            for _, _, w in wrappers:
                w.active = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            lp_cur = F.log_softmax(lg.float(), dim=-1)
            tok_kl = (p_ref * (lp_ref - lp_cur)).sum(dim=-1)
            kl = (tok_kl * valid_tok).sum() / v_sum
            kl_loss = (kl_w * F.relu(kl - kl_hinge) + kl_floor * kl) if kl_hinge > 0.0 else (kl_w * kl)

            g_pl = None
            geo_vals = {}
            if pw_scale > 0.0:
                loss_geo = lg.new_tensor(0.0, dtype=torch.float32)
                for dep in depths:
                    li = LAYERS_MAP[dep]
                    fr = frames[li]
                    hc = cap[li][b_idx, t_idx, :].float() - fr["mu_x"]
                    z_cur = hc @ fr["V_q"]
                    h_perp = hc - z_cur @ fr["V_q"].T
                    z_star = (y_t[dep] - fr["mu_y"]) @ fr["R_star"]
                    l_sub = global_frob_cos_gram_loss(z_cur, z_star)
                    l_perp = F.mse_loss(h_perp, perp_b[li]) / (fr["rms_x"] ** 2)
                    loss_geo = loss_geo + weights[dep] * l_sub + perp_w * l_perp
                    geo_vals[f"p{dep}"] = round(float(l_sub.item()), 4)
                g_pl = torch.autograd.grad(pw_scale * loss_geo, all_lora, retain_graph=True, allow_unused=True)

            g_kl = torch.autograd.grad(kl_loss, all_lora, retain_graph=False, allow_unused=True)
            opt.zero_grad(set_to_none=True)
            for i, p in enumerate(all_lora):
                gk = g_kl[i] if g_kl[i] is not None else torch.zeros_like(p)
                gp = (g_pl[i] if (g_pl is not None and g_pl[i] is not None) else torch.zeros_like(p))
                if use_pcgrad and g_pl is not None and g_pl[i] is not None:
                    dot = torch.sum(gp * gk)
                    if dot < 0.0:
                        gp = gp - (dot / torch.sum(gk * gk).clamp_min(1e-12)) * gk
                p.grad = gk + gp
            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()

            if s % 100 == 0 or s == steps - 1:
                history.append({"step": s, "kl": round(float(kl.item()), 5), **geo_vals})
        completed = True
    finally:
        for hk in hooks:
            hk.remove()
        cap.clear()
        opt.zero_grad(set_to_none=True)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if completed:
            tot_fro = float(np.sqrt(sum(w.delta_fro() ** 2 for _, _, w in wrappers)))
            for parent, attr, w in wrappers:
                setattr(parent, attr, w.merge_into_base())
        else:
            for parent, attr, w in wrappers:
                setattr(parent, attr, w.base)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        del frames, wrappers
        gc.collect()
        torch.cuda.empty_cache()
    return {"name": cfg["name"], "ce_w": 0.0, "total_fro_norm": round(tot_fro, 4), "train_sec": round(time.time() - t0, 1), "history": history}


HYPOTHESES = [
    dict(
        name="H0_r3_30_50_stdkl",
        depths=(30, 50), weights={30: 1.65, 50: 2.75},
        active_layers=list(range(2, 21)), modules=("o_proj", "down_proj"),
        rank=24, steps=300, batch_size=8, lr=2.4e-4, kl_w=3.20, perp_w=0.85, pcgrad=True,
    ),
    dict(
        name="H1_r3_tri_30_50_70_stdkl",
        depths=(30, 50, 70), weights={30: 1.40, 50: 1.80, 70: 2.20},
        active_layers=list(range(2, 25)), modules=("o_proj", "down_proj"),
        rank=24, steps=300, batch_size=8, lr=2.4e-4, kl_w=3.20, perp_w=0.85, pcgrad=True,
    ),
    dict(
        name="H2_r3_tri_trustkl",
        depths=(30, 50, 70), weights={30: 1.40, 50: 1.80, 70: 2.20},
        active_layers=list(range(2, 25)), modules=("o_proj", "down_proj"),
        rank=24, steps=300, batch_size=8, lr=2.4e-4, kl_w=2.50, kl_hinge=0.015, kl_floor=0.30, perp_w=0.55, pcgrad=True,
    ),
    dict(
        name="H3_r3_tri_vodown_r32_trustkl",
        depths=(30, 50, 70), weights={30: 1.40, 50: 1.80, 70: 2.20},
        active_layers=list(range(2, 25)), modules=("v_proj", "o_proj", "down_proj"),
        rank=32, steps=300, batch_size=8, lr=2.0e-4, kl_w=2.50, kl_hinge=0.015, kl_floor=0.30, perp_w=0.55, pcgrad=True,
    ),
    dict(
        name="H4_r3_30only_strong",
        depths=(30,), weights={30: 3.00},
        active_layers=list(range(2, 15)), modules=("o_proj", "down_proj"),
        rank=24, steps=300, batch_size=8, lr=2.4e-4, kl_w=2.00, kl_hinge=0.010, kl_floor=0.30, perp_w=0.60, pcgrad=True,
    ),
    dict(
        name="H5_tri_SHUFFLED_ctrl",
        depths=(30, 50, 70), weights={30: 1.40, 50: 1.80, 70: 2.20},
        active_layers=list(range(2, 25)), modules=("o_proj", "down_proj"),
        rank=24, steps=300, batch_size=8, lr=2.4e-4, kl_w=2.50, kl_hinge=0.015, kl_floor=0.30, perp_w=0.55, pcgrad=True,
        shuffled=True,
    ),
]


def main():
    d_c = np.load(TARGETS_NPZ, allow_pickle=True)
    mbpp = json.loads((EXP257_DATA / "sanitized-mbpp.json").read_text())
    mbpp_dev = [x for x in mbpp if int(x["task_id"]) < 11 or int(x["task_id"]) > 510]
    assert len(mbpp_dev) == 170

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, device_map="cuda:0")
    model.eval()
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    base_ev = evaluate_dev(model, tok, d_c, mbpp_dev)
    print("[EXP258_CUDA] BASE:", json.dumps(base_ev), flush=True)
    results = {"base": base_ev, "runs": []}

    for cfg in HYPOTHESES:
        sd = model.state_dict()
        for k, v in base_sd.items():
            sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))
        model.eval()
        print(f"[EXP258_CUDA] Running {cfg['name']}...", flush=True)
        tm = train_hypothesis(model, tok, d_c, cfg, seed=258)
        ev = evaluate_dev(model, tok, d_c, mbpp_dev)
        deltas = {
            "d_code_nll": round(ev["dev_code_nll"] - base_ev["dev_code_nll"], 6),
            "d_mbpp_hits": ev["dev_mbpp_hits_int"] - base_ev["dev_mbpp_hits_int"],
            "d_mbpp_pp": round(100.0 * (ev["dev_mbpp_pass1"] - base_ev["dev_mbpp_pass1"]), 2),
            "d_al30": round(ev["code_blind_30_r2"] - base_ev["code_blind_30_r2"], 5),
            "d_al50": round(ev["code_blind_50_r2"] - base_ev["code_blind_50_r2"], 5),
            "d_al70": round(ev["code_blind_70_r2"] - base_ev["code_blind_70_r2"], 5),
        }
        results["runs"].append({"name": cfg["name"], "deltas": deltas, "eval": ev, "train_meta": tm})
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(
            f"[EXP258_CUDA] DONE {cfg['name']} | "
            f"MBPP_dev={ev['dev_mbpp_hits']} ({deltas['d_mbpp_hits']:+d} / {deltas['d_mbpp_pp']:+.2f}pp) | "
            f"code_nll={ev['dev_code_nll']:.5f} ({deltas['d_code_nll']:+.5f}) | "
            f"blind_code L8={deltas['d_al30']:+.4f} L14={deltas['d_al50']:+.4f} L19={deltas['d_al70']:+.4f} | "
            f"fro={tm['total_fro_norm']:.2f} t={tm['train_sec']}+{ev['eval_sec']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
