"""Pillar 2 Engine: On-Policy Verifiable Coding RL (GRPO in `bwrap`) + Discardable 2-Layer MLP
Transition Alignment (`Δh = h_{step k+1} - h_{step k}`) + 31B Platonic Trajectory Compass Shaping (`Exp 263`).

Runs SOLO on <GPU_HOST> RTX 4090 (strictly 1 GPU job at a time; exact FP32 evaluation; ZERO vault peeking!).
Evaluates strictly on disjoint dev + generalization benchmarks:
  - `dev_mbpp_170` pass@1 (`bwrap` sandbox)  [Primary Metric]
  - `wikitext2_nll` (62 OOD Wikipedia articles) [Hard Guard <= +0.010 drift + Language Proficiency]
  - `dev_code_nll` (399 held-out Python functions) [Coding Generalization]
  - `proc_unseen_41_80` (200 unseen procedural/narrative reasoning items) [Reasoning Generalization]
  - Blind `Phi-4 + OLMo 3` OOF alignment (`30/50/70%`)

NEVER touches `HumanEval (164)` or `MBPP Test (257)` during autoresearch!
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import random
import secrets
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EXP258_DIR = ROOT.parent / "exp258_code_lattice"
EXP257_DIR = ROOT.parent / "exp257_rigorous"
for p in (str(ROOT), str(EXP258_DIR), str(EXP257_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_rigorous as ev257  # noqa: E402
import run_dev_code_cuda as base258  # noqa: E402
import sample_k8_and_extract_06b_cuda as p1s  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
ACTIVE_LAYERS = list(range(2, 28))  # L2..L27 unfrozen
DEPTHS_GEO = (30, 50, 70)
LAYERS_GEO = {30: 8, 50: 14, 70: 19}
DEPTHS_COMP = (30, 50, 70, 85)
LAYERS_COMP = {30: 8, 50: 14, 70: 19, 85: 23}


class DiscardableTransitionMLP(nn.Module):
    """2-layer GELU MLP mapping student transitions Δh (1024) -> universal 31B GCCA Δz (256).
    Discarded completely after training (zero inference overhead, zero coordinate-system friction!)."""
    def __init__(self, d_in=1024, d_hid=512, d_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        return self.net(x)


def grade_kodcode_one(task: dict, raw_gen: str):
    code = base258.extract_python_code(raw_gen)
    try:
        ast.parse(code)
        ast_ok = True
    except SyntaxError:
        ast_ok = False
    if not ast_ok or not code.strip():
        return {"ok": False, "ast_ok": False, "code": code}
    sent = f"__P2RL_{secrets.token_hex(8)}__"
    script = f"{code}\n\n{task['test_script']}\nprint({sent!r})\n"
    rc, out, _ = base258.run_sandbox(script, timeout_s=3.5)
    ok = (rc == 0) and (sent in out.splitlines())
    return {"ok": bool(ok), "ast_ok": True, "code": code}


def set_lora_enabled(wrappers, enabled: bool):
    for _, _, _, w in wrappers:
        w.enabled = enabled


class ToggleLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: float = 16.0):
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
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale).to(y.dtype)


def load_frozen_compass_params(compass_npz: Path, dev):
    """Load frozen Pillar 1 31B Whitened GCCA Compass parameters (zero 31B inference at train/test time!)."""
    if not compass_npz.exists():
        return None
    d = dict(np.load(compass_npz, allow_pickle=True))
    cfg = json.loads(str(d["cfg_json"].item()))
    depths = tuple(cfg["depths"])
    params = {"cfg": cfg, "depths": depths, "maps": {}, "w": torch.from_numpy(d["w"].astype(np.float32)).to(dev)}
    for dep in depths:
        params["maps"][dep] = {
            "ma": torch.from_numpy(d[f"ma_{dep}"].astype(np.float32)).to(dev),
            "mb": torch.from_numpy(d[f"mb_{dep}"].astype(np.float32)).to(dev),
            "M": torch.from_numpy(d[f"M_{dep}"].astype(np.float32)).to(dev),  # linear (1024 -> d_out) combined PCA+Procrustes or Ridge!
        }
    return params


@torch.no_grad()
def score_rollouts_with_compass(model, tok, prompts_raw, codes, compass_params, dev, shuffled=False):
    """Compute frozen 31B Platonic Trajectory Compass score for generated rollouts (0.6B forward pass only!)."""
    segs_list = [p1s.segment_trajectory_6_safe(pr, cd)[1] for pr, cd in zip(prompts_raw, codes)]
    depths = compass_params["depths"]
    feat_mode = compass_params["cfg"]["feat_mode"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    b_ids, b_pts = [], []
    for segs in segs_list:
        ids, pts = p1s.boundary_indices_qwen(tok, segs)
        if len(ids) > 576:
            ids = ids[:544] + ids[-32:]
            pts = [min(p, len(ids) - 1) for p in pts]
            pts[-1] = len(ids) - 1
        b_ids.append(ids)
        b_pts.append(pts)

    ml = max(len(x) for x in b_ids)
    inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
    attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
    for bi, ids in enumerate(b_ids):
        inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
        attn[bi, : len(ids)] = 1

    cap = {}
    hooks = [
        model.model.layers[LAYERS_COMP[d]].register_forward_hook(
            lambda _m, _i, out, idx=LAYERS_COMP[d]: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for d in depths
    ]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
        parts = []
        for d in depths:
            li = LAYERS_COMP[d]
            mp = compass_params["maps"][d]
            H = torch.stack([cap[li][bi, b_pts[bi], :].float() for bi in range(len(b_ids))], dim=0)  # (N, 6, 1024)
            Z = mp["mb"] + (H - mp["ma"]) @ mp["M"]  # (N, 6, d_out)
            if feat_mode == "code_end":
                f = Z[:, 4, :]
            elif feat_mode == "audit_end":
                f = Z[:, 5, :]
            elif feat_mode == "audit_plus_code":
                f = torch.cat([Z[:, 4, :], Z[:, 5, :]], dim=1)
            elif feat_mode == "deltas_plan_body_audit":
                f = torch.cat([Z[:, 1] - Z[:, 0], Z[:, 4] - Z[:, 1], Z[:, 5] - Z[:, 4]], dim=1)
            elif feat_mode == "full_trajectory_deltas":
                ds = [Z[:, k + 1] - Z[:, k] for k in range(5)]
                f = torch.cat(ds + [Z[:, 4], Z[:, 5]], dim=1)
            elif feat_mode == "code_deltas_only":
                ds = [Z[:, k + 1] - Z[:, k] for k in range(4)]
                f = torch.cat(ds + [Z[:, 4]], dim=1)
            elif feat_mode == "audit_minus_code":
                f = torch.cat([Z[:, 5] - Z[:, 4], Z[:, 5]], dim=1)
            else:
                f = torch.cat([Z[:, 4], Z[:, 5]], dim=1)
            f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
            parts.append(f)
        F_all = torch.cat(parts, dim=1)
        scores = (F_all @ compass_params["w"]).cpu().numpy().astype(np.float64)
    finally:
        for hk in hooks:
            hk.remove()
    if shuffled:
        rng = np.random.default_rng(int(time.time_ns()) % (2**31))
        scores = rng.permutation(scores)
    return scores


@torch.no_grad()
def eval_proc_unseen_41_80(model, tok, dev):
    """Evaluate strictly unseen Proc 41-80 (200 procedural/narrative reasoning items) for generalization."""
    import run_proc_unseen_and_ablations as r257b
    probes_dir = ev257.default_paths()["probes_dir"]
    if not (probes_dir / "proc_held.jsonl").exists():
        return None, None
    res = r257b.eval_proc_mcq_slice(model, tok, probes_dir, 40, 80)
    return res["proc_acc"], res["proc_margin"]


def run_pillar2_arm(args):
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(dev)

    # Load RL training tasks (500 strictly decontaminated KodCode tasks)
    rl_tasks = [json.loads(l) for l in open(DATA / "rl_train_tasks_500.jsonl") if l.strip()]
    # Load dev_mbpp_170 + code_eval + 31B GCCA targets
    mbpp_all = json.loads((base258.EXP257_DATA / "sanitized-mbpp.json").read_text())
    mbpp_dev_tasks = [t for t in mbpp_all if int(t["task_id"]) < 11 or int(t["task_id"]) > 510]
    d_c = dict(np.load(EXP258_DIR / "targets_qwen600m_code_lattice_30_50_70.npz", allow_pickle=True))
    d_31b = dict(np.load(EXP258_DIR / "targets_qwen600m_31b_gpa_gcca.npz", allow_pickle=True))

    # Optional frozen Pillar 1 31B Compass for dense GRPO reward shaping
    compass_params = None
    if args.compass_w > 0.0:
        compass_params = load_frozen_compass_params(CACHE / "best_pillar1_compass.npz", dev)
        if compass_params is None:
            raise RuntimeError("Missing cache/best_pillar1_compass.npz required for compass reward shaping!")

    # Attach unfrozen L2..L27 toggleable LoRA (so base ref logprobs need ZERO second model in VRAM!)
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers, train_params = [], []
    for li in ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mname, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            w = ToggleLoRALinear(getattr(parent, attr), r=args.rank, alpha=float(args.rank))
            setattr(parent, attr, w)
            wrappers.append((f"L{li}.{mname}", parent, attr, w))
            train_params.extend([w.lora_A, w.lora_B])

    # Optional discardable 2-layer MLP projector for reasoning transitions Δh -> 31B Whitened GCCA Δz!
    use_trans = (args.trans_w > 0.0)
    proj_mlps = {}
    geo_tgts = {}
    if use_trans:
        key_prefix = "gcca31" if args.geo_type == "gcca" else "gpa31"
        for dep in DEPTHS_GEO:
            arr = d_31b[f"{key_prefix}_{dep}_tr"].astype(np.float32)
            # Standardize per-dimension scale
            arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-6)
            if args.shuffle_geo:
                rng_s = np.random.default_rng(args.seed + dep)
                arr = arr[rng_s.permutation(len(arr))]
            geo_tgts[dep] = arr
            mlp = DiscardableTransitionMLP(d_in=1024, d_hid=512, d_out=arr.shape[1]).to(dev)
            proj_mlps[dep] = mlp
            train_params.extend(list(mlp.parameters()))

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Pre-warm DiscardableTransitionMLP on frozen Base 0.6B (200 steps, ~5s!) so transition gradients
    # into LoRA are clean from step 1 instead of random-init noise!
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_geo_tot = len(ids_o) - 1
    if use_trans:
        t_pw = time.time()
        set_lora_enabled(wrappers, False)
        mlp_params = [p for m in proj_mlps.values() for p in m.parameters()]
        pw_opt = torch.optim.AdamW(mlp_params, lr=2.5e-3, weight_decay=0.01)
        rng_pw = np.random.default_rng(args.seed + 777)
        # Pre-extract 180 frozen base sequences once (~1.5s) then fit MLPs cleanly!
        sub_idx = rng_pw.choice(n_geo_tot, size=min(240, n_geo_tot), replace=False)
        cap_pw = {}
        hks_pw = [
            model.model.layers[LAYERS_GEO[dep]].register_forward_hook(
                lambda _m, _i, out, idx=LAYERS_GEO[dep]: cap_pw.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
            )
            for dep in DEPTHS_GEO
        ]
        pw_dh, pw_dz = {dep: [] for dep in DEPTHS_GEO}, {dep: [] for dep in DEPTHS_GEO}
        try:
            with torch.no_grad():
                for s_i in range(0, len(sub_idx), 16):
                    chunk = sub_idx[s_i : s_i + 16]
                    b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64) for i in chunk]
                    b_pts = [pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64) for i in chunk]
                    ml = max(len(x) for x in b_ids)
                    inp_p = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
                    att_p = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
                    for bi, ids in enumerate(b_ids):
                        inp_p[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                        att_p[bi, : len(ids)] = 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model.model(input_ids=inp_p, attention_mask=att_p, use_cache=False)
                    for dep in DEPTHS_GEO:
                        li = LAYERS_GEO[dep]
                        for bi, (pi, pts) in enumerate(zip(chunk, b_pts)):
                            if len(pts) < 3:
                                continue
                            h_s = cap_pw[li][bi, pts, :].float()
                            z_s = torch.from_numpy(geo_tgts[dep][int(pts_o[pi]):int(pts_o[pi + 1])]).to(dev)
                            if args.align_mode == "delta":
                                pw_dh[dep].append(h_s[1:] - h_s[:-1]); pw_dz[dep].append(z_s[1:] - z_s[:-1])
                            elif args.align_mode == "both":
                                pw_dh[dep].append(torch.cat([h_s, h_s[1:] - h_s[:-1]], dim=0))
                                pw_dz[dep].append(torch.cat([z_s, z_s[1:] - z_s[:-1]], dim=0))
                            else:
                                pw_dh[dep].append(h_s); pw_dz[dep].append(z_s)
        finally:
            for hk in hks_pw:
                hk.remove()
        for dep in DEPTHS_GEO:
            X_pw = torch.cat(pw_dh[dep], dim=0)
            Y_pw = torch.cat(pw_dz[dep], dim=0)
            for _ in range(250):
                ii = torch.randint(0, len(X_pw), (min(256, len(X_pw)),), device=dev)
                pr = proj_mlps[dep](X_pw[ii])
                l_pw = (1.0 - F.cosine_similarity(pr, Y_pw[ii], dim=-1).mean()) + F.mse_loss(F.normalize(pr, dim=-1), F.normalize(Y_pw[ii], dim=-1))
                pw_opt.zero_grad(); l_pw.backward(); pw_opt.step()
            with torch.no_grad():
                pr_f = proj_mlps[dep](X_pw)
                c_end = float(F.cosine_similarity(pr_f, Y_pw, dim=-1).mean().item())
            print(f"[PILLAR2] Pre-warmed TransitionMLP dep={dep}: cos={c_end:.4f} ({time.time()-t_pw:.1f}s)", flush=True)
        del pw_dh, pw_dz, pw_opt
        set_lora_enabled(wrappers, True)
        torch.cuda.empty_cache()

    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    rl_prompts = []
    for t in rl_tasks:
        rl_prompts.append(
            tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{t['sigs']}\n```"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )

    # Language-proficiency replay pool (optional tiny Gutenberg/code CE anchor if lang_ce_w > 0)
    rng = np.random.default_rng(args.seed)
    task_perm = rng.permutation(len(rl_tasks))
    geo_perm = rng.permutation(n_geo_tot)
    t_cur, g_cur = 0, 0

    step_logs = []
    print(f"[PILLAR2] Starting arm={args.arm} seed={args.seed} steps={args.steps} B={args.batch_tasks} G={args.group_size} "
          f"lr={args.lr} kl={args.kl_w} trans_w={args.trans_w} ({args.geo_type}, shuf={args.shuffle_geo}) "
          f"compass_w={args.compass_w} (shuf={args.shuffle_compass})", flush=True)

    for step in range(args.steps):
        frac = step / max(1, args.steps - 1)
        lr_mul = min(1.0, (step + 1) / 8.0) * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        opt.param_groups[0]["lr"] = args.lr * lr_mul

        # 1. Sample batch of RL tasks
        b_idx = []
        for _ in range(args.batch_tasks):
            if t_cur >= len(task_perm):
                task_perm, t_cur = rng.permutation(len(rl_tasks)), 0
            b_idx.append(int(task_perm[t_cur]))
            t_cur += 1
        b_tasks = [rl_tasks[i] for i in b_idx]
        b_prompts = [rl_prompts[i] for i in b_idx]

        # 2. On-policy rollout generation (current policy, bf16 autocast for fast generation!)
        model.eval()
        orig_side = tok.padding_side
        tok.padding_side = "left"
        enc = tok(b_prompts, return_tensors="pt", padding=True).to(dev)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen_ids = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=0.75, top_p=0.95, num_return_sequences=args.group_size,
                pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
            )
        tok.padding_side = orig_side

        G = args.group_size
        raw_gens, task_rep, prompt_rep = [], [], []
        for bi in range(len(b_tasks)):
            for gi in range(G):
                row_i = bi * G + gi
                raw_gens.append(tok.decode(gen_ids[row_i, in_len:], skip_special_tokens=True))
                task_rep.append(b_tasks[bi])
                prompt_rep.append(b_tasks[bi]["prompt"])

        # 3. Grade rollouts in real OS sandbox (`bwrap`)
        with ThreadPoolExecutor(max_workers=12) as ex:
            graded = list(ex.map(lambda p: grade_kodcode_one(p[0], p[1]), zip(task_rep, raw_gens)))
        pass_r = np.array([1.0 if g["ok"] else 0.0 for g in graded], dtype=np.float64)
        ast_r = np.array([0.05 if g["ast_ok"] else 0.0 for g in graded], dtype=np.float64)
        total_r = pass_r + ast_r

        # Optional 31B Platonic Trajectory Compass dense reward shaping!
        if compass_params is not None and args.compass_w > 0.0:
            set_lora_enabled(wrappers, False)  # project through FROZEN base 0.6B -> 31B GCCA map (immune to reward hacking!)
            c_raw = score_rollouts_with_compass(
                model, tok, prompt_rep, [g["code"] for g in graded],
                compass_params, dev, shuffled=args.shuffle_compass,
            )
            set_lora_enabled(wrappers, True)
            # Group-standardize compass score inside each task group (zero drift across problems!)
            c_mat = c_raw.reshape(len(b_tasks), G)
            c_z = (c_mat - c_mat.mean(axis=1, keepdims=True)) / (c_mat.std(axis=1, keepdims=True) + 1e-6)
            # Only reward-shape AST-valid candidates; cap magnitude cleanly with tanh
            c_shaped = np.tanh(0.5 * c_z).reshape(-1) * np.array([1.0 if g["ast_ok"] else 0.0 for g in graded])
            if args.compass_dead_only:
                # Apply dense compass signal strictly when binary bwrap group has zero variance!
                dead_mask = np.repeat((pass_r.reshape(len(b_tasks), G).std(axis=1) < 1e-6), G)
                total_r = total_r + args.compass_w * c_shaped * dead_mask.astype(np.float64)
            else:
                total_r = total_r + args.compass_w * c_shaped

        # Group-relative GRPO advantages
        r_mat = total_r.reshape(len(b_tasks), G)
        adv_mat = r_mat - r_mat.mean(axis=1, keepdims=True)
        sd_mat = r_mat.std(axis=1, keepdims=True)
        adv_mat = np.where(sd_mat > 1e-6, adv_mat / (sd_mat + 1e-6), 0.0)
        adv = torch.from_numpy(adv_mat.reshape(-1).astype(np.float32)).to(dev)

        # 4. Compute exact GRPO policy-gradient + reverse-KL loss on generated tokens!
        model.train()
        opt.zero_grad(set_to_none=True)
        N_seq = gen_ids.shape[0]
        full_attn = (gen_ids != pad_id).long()
        # Mask strictly over generated response tokens (after in_len, up to first eos/pad)
        resp_len = gen_ids.shape[1] - in_len
        resp_mask = torch.zeros((N_seq, resp_len), dtype=torch.float32, device=dev)
        for ri in range(N_seq):
            toks = gen_ids[ri, in_len:].tolist()
            end_k = len(toks)
            for k, tid in enumerate(toks):
                if tid == tok.eos_token_id or tid == pad_id:
                    end_k = k + 1
                    break
            resp_mask[ri, :end_k] = 1.0

        # Frozen reference log-probs (adapter disabled -> ZERO extra model in VRAM!)
        with torch.no_grad():
            set_lora_enabled(wrappers, False)
            ref_lp_all = []
            for s_i in range(0, N_seq, 4):
                sl = slice(s_i, min(N_seq, s_i + 4))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_r = model(input_ids=gen_ids[sl], attention_mask=full_attn[sl], use_cache=False).logits[:, in_len - 1 : -1, :]
                lp_r = F.log_softmax(lg_r.float(), dim=-1)
                tgt_r = gen_ids[sl, in_len:].unsqueeze(-1)
                ref_lp_all.append(lp_r.gather(-1, tgt_r).squeeze(-1))
                del lg_r, lp_r
            ref_lp = torch.cat(ref_lp_all, dim=0)
            set_lora_enabled(wrappers, True)

        # Policy log-probs in micro-batches of 4 (accumulate gradient cleanly, zero OOM!)
        rl_loss_val, kl_val = 0.0, 0.0
        n_mb = math.ceil(N_seq / 4)
        for s_i in range(0, N_seq, 4):
            sl = slice(s_i, min(N_seq, s_i + 4))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg_p = model(input_ids=gen_ids[sl], attention_mask=full_attn[sl], use_cache=False).logits[:, in_len - 1 : -1, :]
            lp_p = F.log_softmax(lg_p.float(), dim=-1)
            tgt_p = gen_ids[sl, in_len:].unsqueeze(-1)
            cur_lp = lp_p.gather(-1, tgt_p).squeeze(-1)
            m = resp_mask[sl]
            m_sum = m.sum(dim=1).clamp_min(1.0)
            # Per-token Schulman KL approx: exp(ref - cur) - (ref - cur) - 1 >= 0
            log_ratio = (ref_lp[sl] - cur_lp).clamp(-10.0, 10.0)
            kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
            # GRPO surrogate (on-policy, ratio=1 at first step -> -adv * cur_lp)
            pg_seq = -(adv[sl] * ((cur_lp * m).sum(dim=1) / m_sum))
            kl_seq = (kl_tok * m).sum(dim=1) / m_sum
            mb_loss = (pg_seq.mean() + args.kl_w * kl_seq.mean()) * (len(adv[sl]) / float(N_seq))
            mb_loss.backward()
            rl_loss_val += float(pg_seq.mean().item()) * (len(adv[sl]) / float(N_seq))
            kl_val += float(kl_seq.mean().item()) * (len(adv[sl]) / float(N_seq))
            del lg_p, lp_p, cur_lp, mb_loss

        # 5. Optional Discardable 2-Layer MLP Transition Alignment (`Δh = h_{k+1} - h_k` -> 31B GCCA `Δz`)!
        trans_val = 0.0
        if use_trans:
            g_batch = []
            while len(g_batch) < 6:
                if g_cur >= len(geo_perm):
                    geo_perm, g_cur = rng.permutation(n_geo_tot), 0
                pi = int(geo_perm[g_cur]); g_cur += 1
                ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                if len(ids) >= 8 and len(pts) >= 3:
                    ts = {dep: geo_tgts[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in DEPTHS_GEO}
                    g_batch.append((ids, pts, ts))

            ml_g = max(len(x[0]) for x in g_batch)
            inp_g = torch.full((len(g_batch), ml_g), pad_id, dtype=torch.long, device=dev)
            attn_g = torch.zeros((len(g_batch), ml_g), dtype=torch.long, device=dev)
            for bi, (ids, _, _) in enumerate(g_batch):
                inp_g[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn_g[bi, : len(ids)] = 1

            cap_g = {}
            hooks_g = [
                model.model.layers[LAYERS_GEO[dep]].register_forward_hook(
                    lambda _m, _i, out, idx=LAYERS_GEO[dep]: cap_g.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
                )
                for dep in DEPTHS_GEO
            ]
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.model(input_ids=inp_g, attention_mask=attn_g, use_cache=False)
                t_loss = inp_g.new_tensor(0.0, dtype=torch.float32)
                for dep in DEPTHS_GEO:
                    li = LAYERS_GEO[dep]
                    dh_list, dz_list = [], []
                    for bi, (_, pts, ts) in enumerate(g_batch):
                        h_seq = cap_g[li][bi, pts, :].float()  # (M, 1024)
                        z_seq = torch.from_numpy(ts[dep]).to(dev)  # (M, 256)
                        if args.align_mode == "delta":
                            dh_list.append(h_seq[1:] - h_seq[:-1])
                            dz_list.append(z_seq[1:] - z_seq[:-1])
                        elif args.align_mode == "both":
                            dh_list.append(torch.cat([h_seq, h_seq[1:] - h_seq[:-1]], dim=0))
                            dz_list.append(torch.cat([z_seq, z_seq[1:] - z_seq[:-1]], dim=0))
                        else:
                            dh_list.append(h_seq)
                            dz_list.append(z_seq)
                    dh = torch.cat(dh_list, dim=0)
                    dz = torch.cat(dz_list, dim=0)
                    pred = proj_mlps[dep](dh)
                    # Cosine + normalized MSE transition loss (scale-invariant & smooth!)
                    cos_l = 1.0 - F.cosine_similarity(pred, dz, dim=-1).mean()
                    mse_l = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(dz, dim=-1))
                    t_loss = t_loss + (cos_l + mse_l) / float(len(DEPTHS_GEO))
                (args.trans_w * t_loss).backward()
                trans_val = float(t_loss.item())
            finally:
                for hk in hooks_g:
                    hk.remove()
            del inp_g, attn_g, cap_g

        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        del gen_ids, full_attn, resp_mask, ref_lp, enc
        torch.cuda.empty_cache()

        step_logs.append({"step": step + 1, "pass_rate": round(float(pass_r.mean()), 4),
                          "kl": round(kl_val, 5), "trans_loss": round(trans_val, 5)})
        if (step + 1) % 10 == 0 or step == args.steps - 1:
            avg_pr = float(np.mean([x["pass_rate"] for x in step_logs[-10:]]))
            print(f"[PILLAR2] {args.arm} step {step+1}/{args.steps} | rollout_pass={100*avg_pr:.1f}% kl={kl_val:.4f} trans={trans_val:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Discard the MLP projector completely! Merge LoRA weights in exact FP32 for exact evaluation!
    for _, parent, attr, w in wrappers:
        with torch.no_grad():
            delta = (w.lora_B @ w.lora_A) * w.scale
            w.base.weight.data.add_(delta.to(w.base.weight.dtype))
        setattr(parent, attr, w.base)
    del wrappers, proj_mlps, opt
    gc.collect()
    torch.cuda.empty_cache()

    # Exact FP32 Evaluation strictly on disjoint dev + generalization benchmarks (ZERO vault peeking!)
    model.eval()
    dev_res = base258.evaluate_dev(model, tok, d_c, mbpp_dev_tasks)
    wt2_res = ev257.eval_wikitext2_nll(model, tok, EXP257_DIR / "data")
    proc_acc, proc_margin = eval_proc_unseen_41_80(model, tok, dev)

    out = {
        "arm": args.arm,
        "seed": args.seed,
        "args": vars(args),
        "dev_mbpp_pass1": dev_res["dev_mbpp_pass1"],
        "dev_mbpp_hits": dev_res["dev_mbpp_hits"],
        "dev_mbpp_hits_int": dev_res["dev_mbpp_hits_int"],
        "dev_mbpp_total": 170,
        "dev_code_nll": dev_res["dev_code_nll"],
        "code_blind_30_r2": dev_res["code_blind_30_r2"],
        "code_blind_50_r2": dev_res["code_blind_50_r2"],
        "code_blind_70_r2": dev_res["code_blind_70_r2"],
        "wikitext2_nll": wt2_res["nll"],
        "wt2_drift": round(wt2_res["nll"] - 3.291073, 6),
        "proc_unseen_acc": proc_acc,
        "proc_unseen_margin": proc_margin,
        "elapsed_sec": round(time.time() - t0, 1),
        "step_logs": step_logs,
    }
    out_path = RESULTS / f"p2_{args.arm}_s{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"[PILLAR2] DONE {args.arm} s={args.seed} | dev_mbpp={dev_res['dev_mbpp_hits']} ({100*dev_res['dev_mbpp_pass1']:.2f}%) | "
        f"wt2={wt2_res['nll']:.5f} (drift={wt2_res['nll']-3.291073:+.5f}) | code_nll={dev_res['dev_code_nll']:.5f} | "
        f"proc41_80={proc_acc} | blind={dev_res['code_blind_30_r2']:.4f}/{dev_res['code_blind_50_r2']:.4f}/{dev_res['code_blind_70_r2']:.4f} | "
        f"t={time.time()-t0:.1f}s",
        flush=True,
    )
    if args.save_ckpt:
        ckpt_dir = CACHE / f"ckpt_{args.arm}_s{args.seed}"
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch_tasks", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl_w", type=float, default=0.05)
    ap.add_argument("--trans_w", type=float, default=0.0)
    ap.add_argument("--geo_type", choices=["gcca", "gpa"], default="gcca")
    ap.add_argument("--align_mode", choices=["delta", "static", "both"], default="delta")
    ap.add_argument("--shuffle_geo", action="store_true")
    ap.add_argument("--compass_w", type=float, default=0.0)
    ap.add_argument("--compass_dead_only", action="store_true")
    ap.add_argument("--shuffle_compass", action="store_true")
    ap.add_argument("--save_ckpt", action="store_true")
    run_pillar2_arm(ap.parse_args())
