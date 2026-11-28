"""5-Family 4-Depth (`30, 50, 70, 85%` -> `L8, L14, L19, L23`) Jung-Plato Lattice (`JPL`) Showdown on `Qwen3-0.6B`.

Unlocks the newly healed `85%` pre-decision layer (`L23`, where `Inkling-2-Mini` exits its 3x looped core and
`5-Family JPL` jumps `+8.1 pts` to `0.598 r2 / 95.8% Top-1`) alongside `30/50/70%` (`L8, L14, L19`).

Supports clean head-to-head testing across ANY 5-family `JPL` crystal (separate or combined, with or without
`JPL-Code` 50/50 blend, with or without Python GRPO, and exact matched `--shuffle_geo` placebo controls):
  - `jpl_myth5_4d_plus_code`     : 5-Fam 25-Epic World-Myth (4-depth `30/50/70/85%`, w=0.20) + Code (`30/50/70%`, w=0.20)
  - `jpl_torah_he_4d_plus_code`  : 5-Fam Biblical Hebrew Torah (4-depth, w=0.20) + Code (w=0.20)
  - `jpl_bible_en_4d_plus_code`  : 5-Fam English KJV Bible (4-depth, w=0.20) + Code (w=0.20)
  - `jpl_logic_4d_plus_code`     : 5-Fam Formal/Symbolic Logic (4-depth, w=0.20) + Code (w=0.20)
  - `jpl_all4_mega_4d_plus_code` : 5-Fam All-4 Mega-Crystal (World-Myth + Bible + Hebrew Torah + Logic, 4-depth, w=0.20) + Code (w=0.20)
  - `jpl_logic_4d_only`          : Pure Formal Logic (4-depth, w=0.40) + Python GRPO (ZERO code in lattice!)
  - `jpl_torah_he_4d_only`       : Pure Hebrew Torah (4-depth, w=0.40) + Python GRPO (ZERO code, ZERO English in lattice!)
  - `jpl_bible_en_4d_only`       : Pure English Bible (4-depth, w=0.40) + Python GRPO (ZERO code in lattice!)
  - `jpl_all4_mega_4d_only`      : Pure All-4 Mega-Crystal (4-depth, w=0.40) + Python GRPO (ZERO code in lattice!)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
VAULT_OUT = ROOT / "results_vault"
VAULT_OUT.mkdir(parents=True, exist_ok=True)

EXP258_DIR = ROOT.parent / "exp258_code_lattice"
EXP257_DIR = ROOT.parent / "exp257_rigorous"
for p in (str(ROOT), str(EXP258_DIR), str(EXP257_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pillar2_grpo_transition_mlp as p2  # noqa: E402
import run_dev_code_cuda as base258  # noqa: E402
import run_p2_myth_code_dual_5seed as dual  # noqa: E402
import verify_vault_p2_5seed as vp2  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
# Exact Qwen3-0.6B (28 layers: 0..27) layer map across 4 depths including pre-decision L23 (85%)!
LAYERS_4D = {30: 8, 50: 14, 70: 19, 85: 23}
DEPTHS_4D = (30, 50, 70, 85)
DEPTHS_3D = (30, 50, 70)


def prewarm_mlps_depths(model, wrappers, d_dom, tgts_dom, mlps_dom, depths, pad_id, dev, seed):
    p2.set_lora_enabled(wrappers, False)
    pw_opt = torch.optim.AdamW([prm for m in mlps_dom.values() for prm in m.parameters()], lr=2.5e-3, weight_decay=0.01)
    ids_f, ids_o = d_dom["tr_ids_flat"], d_dom["tr_ids_off"]
    pts_f, pts_o = d_dom["tr_pts_flat"], d_dom["tr_pts_off"]
    n_tot = len(ids_o) - 1
    sub_idx = np.random.default_rng(seed).choice(n_tot, size=min(240, n_tot), replace=False)
    cap = {}
    hks = [
        model.model.layers[LAYERS_4D[dep]].register_forward_hook(
            lambda _m, _i, out, idx=LAYERS_4D[dep]: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for dep in depths
    ]
    pw_dh, pw_dz = {d: [] for d in depths}, {d: [] for d in depths}
    try:
        with torch.no_grad():
            for s_i in range(0, len(sub_idx), 16):
                chunk = sub_idx[s_i : s_i + 16]
                b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64)[:512] for i in chunk]
                b_pts = [np.clip(pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64), 0, len(b_ids[bi]) - 1) for bi, i in enumerate(chunk)]
                ml = max(len(x) for x in b_ids)
                inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
                att = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
                for bi, ids in enumerate(b_ids):
                    inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev); att[bi, : len(ids)] = 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.model(input_ids=inp, attention_mask=att, use_cache=False)
                for dep in depths:
                    li = LAYERS_4D[dep]
                    for bi, (pi, pts) in enumerate(zip(chunk, b_pts)):
                        if len(pts) < 3:
                            continue
                        h_s = cap[li][bi, pts, :].float()
                        z_s = torch.from_numpy(tgts_dom[dep][int(pts_o[pi]):int(pts_o[pi + 1])]).to(dev)
                        pw_dh[dep].append(h_s[1:] - h_s[:-1]); pw_dz[dep].append(z_s[1:] - z_s[:-1])
    finally:
        for hk in hks:
            hk.remove()
    for dep in depths:
        X_pw, Y_pw = torch.cat(pw_dh[dep], dim=0), torch.cat(pw_dz[dep], dim=0)
        for _ in range(250):
            ii = torch.randint(0, len(X_pw), (min(256, len(X_pw)),), device=dev)
            pr = mlps_dom[dep](X_pw[ii])
            l_pw = (1.0 - F.cosine_similarity(pr, Y_pw[ii], dim=-1).mean()) + F.mse_loss(F.normalize(pr, dim=-1), F.normalize(Y_pw[ii], dim=-1))
            pw_opt.zero_grad(); l_pw.backward(); pw_opt.step()
    del pw_dh, pw_dz, pw_opt
    p2.set_lora_enabled(wrappers, True)
    torch.cuda.empty_cache()


def domain_delta_step_depths(model, d_dom, tgts_dom, mlps_dom, depths, perm, cur, rng, pad_id, dev, weight):
    ids_f, ids_o = d_dom["tr_ids_flat"], d_dom["tr_ids_off"]
    pts_f, pts_o = d_dom["tr_pts_flat"], d_dom["tr_pts_off"]
    n_tot = len(ids_o) - 1
    batch = []
    while len(batch) < 4:
        if cur >= len(perm):
            perm, cur = rng.permutation(n_tot), 0
        pi = int(perm[cur]); cur += 1
        ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)[:512]
        pts = np.clip(pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64), 0, len(ids) - 1)
        if len(ids) >= 8 and len(pts) >= 3:
            batch.append((ids, pts, {dep: tgts_dom[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in depths}))
    ml = max(len(x[0]) for x in batch)
    inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
    att = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
    for bi, (ids, _, _) in enumerate(batch):
        inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev); att[bi, : len(ids)] = 1
    cap = {}
    hks = [
        model.model.layers[LAYERS_4D[dep]].register_forward_hook(
            lambda _m, _i, out, idx=LAYERS_4D[dep]: cap.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
        )
        for dep in depths
    ]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.model(input_ids=inp, attention_mask=att, use_cache=False)
        loss = inp.new_tensor(0.0, dtype=torch.float32)
        for dep in depths:
            li = LAYERS_4D[dep]
            dh = torch.cat([cap[li][bi, pts, :].float()[1:] - cap[li][bi, pts, :].float()[:-1] for bi, (_, pts, _) in enumerate(batch)], dim=0)
            dz = torch.cat([torch.from_numpy(ts[dep]).to(dev)[1:] - torch.from_numpy(ts[dep]).to(dev)[:-1] for _, _, ts in batch], dim=0)
            pred = mlps_dom[dep](dh)
            loss = loss + ((1.0 - F.cosine_similarity(pred, dz, dim=-1).mean()) + F.mse_loss(F.normalize(pred, dim=-1), F.normalize(dz, dim=-1))) / float(len(depths))
        (weight * loss).backward()
    finally:
        for hk in hks:
            hk.remove()
    del inp, att, cap
    return perm, cur


def train_jpl_4d(model, tok, rl_tasks, rl_prompts, d_jpl, d_c, d_31b, seed: int,
                 jpl_depths=DEPTHS_4D, jpl_key_pfx="myth5_gcca", jpl_w=0.20, code_w=0.20,
                 use_grpo=True, shuffle_geo=False):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    steps, batch_tasks, G, lr, kl_w, rank = 60, 4, 6, 1.5e-4, 0.02, 16

    for prm in model.parameters():
        prm.requires_grad_(False)
    wrappers, train_params = [], []
    for li in p2.ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mname, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            w = p2.ToggleLoRALinear(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((f"L{li}.{mname}", parent, attr, w))
            train_params.extend([w.lora_A, w.lora_B])

    jpl_tgts, jpl_mlps = {}, {}
    for dep in jpl_depths:
        am = d_jpl[f"{jpl_key_pfx}_{dep}_tr"].astype(np.float32)
        am = (am - am.mean(0, keepdims=True)) / (am.std(0, keepdims=True) + 1e-6)
        if shuffle_geo:
            am = am[np.random.default_rng(seed + dep + 500).permutation(len(am))]
        jpl_tgts[dep] = am
        mm = p2.DiscardableTransitionMLP(d_in=1024, d_hid=512, d_out=am.shape[1]).to(dev)
        jpl_mlps[dep] = mm
        train_params.extend(list(mm.parameters()))
    prewarm_mlps_depths(model, wrappers, d_jpl, jpl_tgts, jpl_mlps, jpl_depths, pad_id, dev, seed + 888)

    code_tgts, code_mlps = {}, {}
    if code_w > 0.0:
        for dep in DEPTHS_3D:
            ac = d_31b[f"gcca31_{dep}_tr"].astype(np.float32)
            ac = (ac - ac.mean(0, keepdims=True)) / (ac.std(0, keepdims=True) + 1e-6)
            if shuffle_geo:
                ac = ac[np.random.default_rng(seed + dep).permutation(len(ac))]
            code_tgts[dep] = ac
            mc = p2.DiscardableTransitionMLP(d_in=1024, d_hid=512, d_out=ac.shape[1]).to(dev)
            code_mlps[dep] = mc
            train_params.extend(list(mc.parameters()))
        prewarm_mlps_depths(model, wrappers, d_c, code_tgts, code_mlps, DEPTHS_3D, pad_id, dev, seed + 777)

    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    task_perm = rng.permutation(len(rl_tasks)); t_cur = 0
    c_perm = rng.permutation(len(d_c["tr_ids_off"]) - 1); c_cur = 0
    j_perm = rng.permutation(len(d_jpl["tr_ids_off"]) - 1); j_cur = 0

    try:
        for step in range(steps):
            frac = step / max(1, steps - 1)
            lr_mul = min(1.0, (step + 1) / 8.0) * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul
            opt.zero_grad(set_to_none=True)

            if use_grpo:
                b_idx = []
                for _ in range(batch_tasks):
                    if t_cur >= len(task_perm):
                        task_perm, t_cur = rng.permutation(len(rl_tasks)), 0
                    b_idx.append(int(task_perm[t_cur])); t_cur += 1
                b_tasks = [rl_tasks[i] for i in b_idx]
                b_prompts = [rl_prompts[i] for i in b_idx]

                model.eval()
                orig_side = tok.padding_side; tok.padding_side = "left"
                enc = tok(b_prompts, return_tensors="pt", padding=True).to(dev)
                in_len = enc["input_ids"].shape[1]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    gen_ids = model.generate(
                        **enc, max_new_tokens=256, do_sample=True, temperature=0.75, top_p=0.95,
                        num_return_sequences=G, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                    )
                tok.padding_side = orig_side

                raw_gens, task_rep = [], []
                for bi in range(len(b_tasks)):
                    for gi in range(G):
                        raw_gens.append(tok.decode(gen_ids[bi * G + gi, in_len:], skip_special_tokens=True))
                        task_rep.append(b_tasks[bi])
                with ThreadPoolExecutor(max_workers=12) as ex:
                    graded = list(ex.map(lambda q: p2.grade_kodcode_one(q[0], q[1]), zip(task_rep, raw_gens)))
                total_r = np.array([(1.0 if g["ok"] else 0.0) + (0.05 if g["ast_ok"] else 0.0) for g in graded], dtype=np.float64)
                r_mat = total_r.reshape(len(b_tasks), G)
                adv_mat = r_mat - r_mat.mean(axis=1, keepdims=True)
                sd_mat = r_mat.std(axis=1, keepdims=True)
                adv = torch.from_numpy(np.where(sd_mat > 1e-6, adv_mat / (sd_mat + 1e-6), 0.0).reshape(-1).astype(np.float32)).to(dev)

                model.train()
                N_seq = gen_ids.shape[0]
                full_attn = (gen_ids != pad_id).long()
                resp_len = gen_ids.shape[1] - in_len
                resp_mask = torch.zeros((N_seq, resp_len), dtype=torch.float32, device=dev)
                for ri in range(N_seq):
                    toks = gen_ids[ri, in_len:].tolist()
                    end_k = len(toks)
                    for k, tid in enumerate(toks):
                        if tid == tok.eos_token_id or tid == pad_id:
                            end_k = k + 1; break
                    resp_mask[ri, :end_k] = 1.0

                with torch.no_grad():
                    p2.set_lora_enabled(wrappers, False)
                    ref_lp_all = []
                    for s_i in range(0, N_seq, 4):
                        sl = slice(s_i, min(N_seq, s_i + 4))
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg_r = model(input_ids=gen_ids[sl], attention_mask=full_attn[sl], use_cache=False).logits[:, in_len - 1 : -1, :]
                        lp_r = F.log_softmax(lg_r.float(), dim=-1)
                        ref_lp_all.append(lp_r.gather(-1, gen_ids[sl, in_len:].unsqueeze(-1)).squeeze(-1))
                        del lg_r, lp_r
                    ref_lp = torch.cat(ref_lp_all, dim=0)
                    p2.set_lora_enabled(wrappers, True)

                for s_i in range(0, N_seq, 4):
                    sl = slice(s_i, min(N_seq, s_i + 4))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg_p = model(input_ids=gen_ids[sl], attention_mask=full_attn[sl], use_cache=False).logits[:, in_len - 1 : -1, :]
                    lp_p = F.log_softmax(lg_p.float(), dim=-1)
                    cur_lp = lp_p.gather(-1, gen_ids[sl, in_len:].unsqueeze(-1)).squeeze(-1)
                    m = resp_mask[sl]; m_sum = m.sum(dim=1).clamp_min(1.0)
                    log_ratio = (ref_lp[sl] - cur_lp).clamp(-10.0, 10.0)
                    kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
                    pg_seq = -(adv[sl] * ((cur_lp * m).sum(dim=1) / m_sum))
                    kl_seq = (kl_tok * m).sum(dim=1) / m_sum
                    ((pg_seq.mean() + kl_w * kl_seq.mean()) * (len(adv[sl]) / float(N_seq))).backward()
                    del lg_p, lp_p, cur_lp
                del gen_ids, full_attn, resp_mask, ref_lp, enc

            if code_w > 0.0:
                c_perm, c_cur = domain_delta_step_depths(model, d_c, code_tgts, code_mlps, DEPTHS_3D, c_perm, c_cur, rng, pad_id, dev, weight=code_w)
            j_perm, j_cur = domain_delta_step_depths(model, d_jpl, jpl_tgts, jpl_mlps, jpl_depths, j_perm, j_cur, rng, pad_id, dev, weight=jpl_w)

            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            torch.cuda.empty_cache()

        deltas = []
        fro_tot = 0.0
        for _, parent, attr, w in wrappers:
            with torch.no_grad():
                dw = ((w.lora_B @ w.lora_A) * w.scale).to(w.base.weight.dtype)
                fro_tot += float(torch.linalg.matrix_norm(dw.float()).item() ** 2)
                w.base.weight.data.add_(dw)
                deltas.append((w.base, dw.detach().cpu()))
            setattr(parent, attr, w.base)
        return deltas, round(math.sqrt(fro_tot), 4), round(time.time() - t0, 1)
    except Exception:
        for _, parent, attr, w in wrappers:
            setattr(parent, attr, w.base)
        raise
    finally:
        del wrappers, jpl_mlps, code_mlps, opt
        gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", type=str, required=True)
    ap.add_argument("--jpl-npz", type=str, required=True)
    ap.add_argument("--jpl-key", type=str, default="myth5_gcca")
    ap.add_argument("--depths", type=str, default="30,50,70,85")
    ap.add_argument("--jpl-w", type=float, default=0.20)
    ap.add_argument("--code-w", type=float, default=0.20)
    ap.add_argument("--seeds", type=str, default="101,102,103,104,105")
    ap.add_argument("--shuffle-geo", action="store_true")
    ap.add_argument("--no-grpo", action="store_true", help="Pure ZERO-Code / ZERO-GRPO JPL alignment only!")
    args = ap.parse_args()

    jpl_depths = tuple(int(x) for x in args.depths.split(",") if x.strip())
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev)

    data_dir = base258.EXP257_DATA
    mbpp_all = json.loads((data_dir / "sanitized-mbpp.json").read_text())
    mbpp_dev = [t for t in mbpp_all if int(t["task_id"]) < 11 or int(t["task_id"]) > 510]
    mbpp_test = [t for t in mbpp_all if 11 <= int(t["task_id"]) <= 510]
    d_c = dict(np.load(EXP258_DIR / "targets_qwen600m_code_lattice_30_50_70.npz", allow_pickle=True))
    d_31b = dict(np.load(EXP258_DIR / "targets_qwen600m_31b_gpa_gcca.npz", allow_pickle=True))
    d_jpl = dict(np.load(args.jpl_npz, allow_pickle=True))
    d_m_eval = dict(np.load(CACHE / "targets_myth_epic_31b_gpa_gcca.npz", allow_pickle=True))
    rl_tasks = [json.loads(l) for l in open(DATA / "rl_train_tasks_500.jsonl") if l.strip()]

    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    rl_prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{t['sigs']}\n```"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for t in rl_tasks
    ]

    runs = []
    for sd in seeds:
        cp = VAULT_OUT / f"p2_vault_{args.arm_name}_s{sd}.json"
        if cp.exists():
            ev = json.loads(cp.read_text())
            runs.append(ev)
            s = ev["summary"]
            print(f"[JPL_4D:{args.arm_name}] CACHED s={sd} | HE={s['humaneval_hits']} MBPP_T={s['mbpp_test_hits']} DEV={s['dev_mbpp_hits']} WT2={s['wikitext2_nll']:.5f} CODE={s['dev_code_nll']:.5f} MYTH={s['held_myth_nll']:.5f} PROC={s['proc41_80_acc']:.4f}", flush=True)
            continue
        print(f"[JPL_4D:{args.arm_name}] Running s={sd} (depths={jpl_depths}, jpl_w={args.jpl_w}, code_w={args.code_w}, grpo={not args.no_grpo}, shuf={args.shuffle_geo})...", flush=True)
        deltas, fro, tr_sec = train_jpl_4d(
            model, tok, rl_tasks, rl_prompts, d_jpl, d_c, d_31b, sd,
            jpl_depths=jpl_depths, jpl_key_pfx=args.jpl_key, jpl_w=args.jpl_w, code_w=args.code_w,
            use_grpo=not args.no_grpo, shuffle_geo=args.shuffle_geo,
        )
        try:
            ev = vp2.evaluate_full_vault(model, tok, d_c, mbpp_dev, mbpp_test, data_dir)
            ev["summary"]["held_myth_nll"] = dual.eval_myth_nll(model, d_m_eval, pad_id, dev)
        finally:
            with torch.no_grad():
                for mod, dw_cpu in deltas:
                    mod.weight.data.sub_(dw_cpu.to(dev))
            del deltas; gc.collect(); torch.cuda.empty_cache()
        ev["arm"] = args.arm_name; ev["seed"] = sd; ev["fro_norm"] = fro; ev["train_sec"] = tr_sec
        cp.write_text(json.dumps(ev))
        runs.append(ev)
        s = ev["summary"]
        print(f"[JPL_4D:{args.arm_name}] DONE s={sd} ({tr_sec:.1f}s) | HE={s['humaneval_hits']} MBPP_T={s['mbpp_test_hits']} DEV={s['dev_mbpp_hits']} WT2={s['wikitext2_nll']:.5f} CODE={s['dev_code_nll']:.5f} MYTH={s['held_myth_nll']:.5f} PROC={s['proc41_80_acc']:.4f} fro={fro}", flush=True)

    keys = ("humaneval_pass1", "mbpp_test_pass1", "dev_mbpp_pass1", "wikitext2_nll",
            "dev_code_nll", "proc41_80_acc", "held_myth_nll",
            "code_blind_30_r2", "code_blind_50_r2", "code_blind_70_r2")
    st = {k: vp2.sample_stats([r["summary"][k] for r in runs]) for k in keys}
    out_p = VAULT_OUT / f"vault_jpl4d_{args.arm_name}_summary.json"
    out_p.write_text(json.dumps({"arm": args.arm_name, "seeds": seeds, "depths": list(jpl_depths), "jpl_npz": Path(args.jpl_npz).name, "stats": st}, indent=2))
    tot591 = [r["summary"]["humaneval_hits_int"] + r["summary"]["mbpp_test_hits_int"] + r["summary"]["dev_mbpp_hits_int"] for r in runs]
    print(f"\n=== [JPL_4D:{args.arm_name}] COMPLETE ({len(seeds)} seeds) | "
          f"HE={100*st['humaneval_pass1']['mean']:.2f}±{100*st['humaneval_pass1']['sd']:.2f}% | "
          f"MBPP_T={100*st['mbpp_test_pass1']['mean']:.2f}±{100*st['mbpp_test_pass1']['sd']:.2f}% | "
          f"DEV={100*st['dev_mbpp_pass1']['mean']:.2f}% | TOT591={np.mean(tot591):.1f}/591 | "
          f"WT2={st['wikitext2_nll']['mean']:.5f} | CODE={st['dev_code_nll']['mean']:.5f} | "
          f"MYTH={st['held_myth_nll']['mean']:.5f} | PROC={100*st['proc41_80_acc']['mean']:.2f}% ===", flush=True)


if __name__ == "__main__":
    main()
