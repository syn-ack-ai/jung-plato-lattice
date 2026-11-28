"""Locked-Vault Verification Stage C: Pillar 2 5-Seed Exact Paired Verification (`Seeds 101..105`).
Runs on a single RTX 4090 GPU after Stage A & B complete!

Evaluates 3 EXACT PAIRED arms across 5 fresh seeds `{101, 102, 103, 104, 105}` (`steps=60, B=4, G=6, lr=1.5e-4, kl_w=0.02`,
unfrozen `L2..L27` `o_proj + down_proj` LoRA `r=16`, exact FP32 merge + exact FP32 eval):
  1. `plain_grpo`: On-policy verifiable GRPO (`bwrap` unit tests on 500 decontaminated KodCode tasks, ZERO geometry)
  2. `grpo_31b_delta_mlp`: Exact same + Pre-Warmed Discardable 2-Layer GELU MLP aligning step transitions
     `Δh_k = h_{k+1} - h_k -> 31B + Llama 3.1 8B + Mistral 7B` Whitened GCCA `Δz_k` (`trans_w=0.40`, MLP discarded at eval!)
  3. `grpo_shuf_delta_mlp`: Exact matched `--shuffle_geo` negative control (identical MLP, identical pre-warm, shuffled targets!)

Evaluates EVERY checkpoint (plus unmodified exact FP32 Base) on ALL 6 benchmarks (persisting EVERY per-item bool/NLL):
  1. LOCKED `HumanEval (164)` (`bwrap` OS sandbox)
  2. LOCKED `MBPP Sanitized Test (257)` (`task_id 11..510`, `bwrap` OS sandbox)
  3. Disjoint `dev_mbpp_170` (`bwrap` OS sandbox)
  4. `WikiText-2` raw test NLL (62 Wikipedia articles — OOD language proficiency!)
  5. `code_eval` NLL (399 strictly held-out Python functions)
  6. Strictly unseen `Proc 41-80` (200 procedural/narrative reasoning items) + Blind `Phi-4 + OLMo 3` OOF `r2`!

Computes BOTH:
  - Seed-level sample SDs (`ddof=1`), paired `df=4` 95% t-CIs, and paired `t`-stats
  - 10,000-resample item-level paired bootstraps (pooling per-item pass rates / NLLs across all 5 seeds!)
Fully checkpointed per `(arm, seed)` for 100% crash-resumability!
"""
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
VAULT_OUT = ROOT / "results_vault"
VAULT_OUT.mkdir(parents=True, exist_ok=True)

EXP258_DIR = ROOT.parent / "exp258_code_lattice"
EXP257_DIR = ROOT.parent / "exp257_rigorous"
for p in (str(ROOT), str(EXP258_DIR), str(EXP257_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_rigorous as ev257  # noqa: E402
import pillar2_grpo_transition_mlp as p2  # noqa: E402
import run_dev_code_cuda as base258  # noqa: E402
import run_exp258_5seed as r258  # noqa: E402
import run_proc_unseen_and_ablations as r257b  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]
T_CRIT_95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764}
MODEL_ID = "Qwen/Qwen3-0.6B"


def sample_stats(vals):
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    tc = T_CRIT_95.get(n - 1, 1.96)
    return {"n": n, "mean": round(m, 6), "sd": round(sd, 6),
            "ci95_lo": round(m - tc * se, 6), "ci95_hi": round(m + tc * se, 6),
            "values": [round(float(x), 6) for x in arr]}


def paired_seed_t(va, vb):
    diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
    st = sample_stats(diff)
    sd = diff.std(ddof=1)
    st["t_stat"] = round(float(diff.mean() / (sd / math.sqrt(len(diff)))), 4) if sd > 0 else 0.0
    st["excludes_zero"] = bool(st["ci95_lo"] > 0.0 or st["ci95_hi"] < 0.0)
    return st


def bootstrap_item_diff(mat_a, mat_b, n_boot=10000, seed=2635):
    """Item-level paired bootstrap over per-item mean across 5 seeds (shape: (n_seeds, n_items))."""
    rng = np.random.default_rng(seed)
    item_diff = np.asarray(mat_a, dtype=np.float64).mean(axis=0) - np.asarray(mat_b, dtype=np.float64).mean(axis=0)
    idx = rng.integers(0, len(item_diff), size=(n_boot, len(item_diff)))
    b = item_diff[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p_two = float(2.0 * min(np.mean(b <= 0.0), np.mean(b >= 0.0)))
    return {
        "n_items": len(item_diff),
        "mean_diff": round(float(item_diff.mean()), 6),
        "boot_ci95_lo": round(float(lo), 6),
        "boot_ci95_hi": round(float(hi), 6),
        "boot_p_two_sided": round(min(1.0, p_two), 5),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


@torch.no_grad()
def evaluate_full_vault(model, tok, d_c, mbpp_dev_tasks, mbpp_test_tasks, data_dir: Path):
    """Exact FP32 evaluation on ALL 6 benchmarks with full per-item persistence!"""
    t0 = time.time()
    model.eval()
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # 1. code_eval (399) + dev_mbpp_170 + blind Phi-4 + OLMo 3 OOF r2
    dev_res = r258.evaluate_dev_per_item(model, tok, d_c, mbpp_dev_tasks)

    # 2. LOCKED HumanEval (164)
    he_res = ev257.eval_humaneval(model, tok, data_dir, max_workers=12)

    # 3. LOCKED MBPP Sanitized Test (257) — exact same prompt format as dev_mbpp_170!
    orig_side = tok.padding_side
    tok.padding_side = "left"
    sys_msg = (
        "You are an expert Python programmer. Solve the task using the exact requested function signature. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    prompts_te = []
    for t in mbpp_test_tasks:
        sigs = base258.extract_mbpp_sigs(t["code"])
        imp = ("\nRequired imports:\n" + "\n".join(t["test_imports"])) if t.get("test_imports") else ""
        prompts_te.append(
            tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"{t['prompt'].strip()}\n\nUse this exact signature:\n```python\n{sigs}\n```{imp}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )
    gens_te = []
    try:
        for s in range(0, len(prompts_te), 24):
            bp = prompts_te[s : s + 24]
            enc = tok(bp, return_tensors="pt", padding=True).to(dev)
            in_len = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
            for bi in range(len(bp)):
                gens_te.append(tok.decode(out[bi, in_len:], skip_special_tokens=True))
            del out, enc; torch.cuda.empty_cache()
    finally:
        tok.padding_side = orig_side

    with ThreadPoolExecutor(max_workers=12) as ex:
        mbpp_te_per = list(ex.map(lambda p: base258.grade_mbpp_one(p[0], p[1]), zip(mbpp_test_tasks, gens_te)))
    mbpp_te_bools = [bool(x["ok"]) for x in mbpp_te_per]
    mbpp_te_hits = sum(mbpp_te_bools)

    # 4. WikiText-2 raw test (62 articles)
    wt2_res = ev257.eval_wikitext2_nll(model, tok, data_dir)

    # 5. Strictly unseen Proc 41-80 (200 items)
    proc_res = r257b.eval_proc_mcq_slice(model, tok, ev257.default_paths()["probes_dir"], 40, 80)

    summary = dict(dev_res["summary"])
    summary["humaneval_pass1"] = he_res["pass_at_1"]
    summary["humaneval_hits"] = f"{he_res['hits']}/{he_res['total']}"
    summary["humaneval_hits_int"] = he_res["hits"]
    summary["mbpp_test_pass1"] = round(mbpp_te_hits / len(mbpp_te_bools), 5)
    summary["mbpp_test_hits"] = f"{mbpp_te_hits}/{len(mbpp_te_bools)}"
    summary["mbpp_test_hits_int"] = mbpp_te_hits
    summary["wikitext2_nll"] = wt2_res["nll"]
    summary["proc41_80_acc"] = proc_res["proc_acc"]
    summary["proc41_80_margin"] = proc_res["proc_margin"]
    summary["eval_sec"] = round(time.time() - t0, 1)

    return {
        "summary": summary,
        "per_fn_nll": [float(x["nll"]) for x in dev_res["per_fn_nll"]],
        "dev_mbpp_bool": [bool(x["ok"]) if isinstance(x, dict) else bool(x) for x in (dev_res["mbpp_per_task_bool"].values() if isinstance(dev_res["mbpp_per_task_bool"], dict) else dev_res["mbpp_per_task_bool"])],
        "mbpp_test_bool": mbpp_te_bools,
        "humaneval_bool": [bool(he_res["per_task_bool"][k]) for k in sorted(he_res["per_task_bool"].keys())] if isinstance(he_res["per_task_bool"], dict) else [bool(x["ok"]) if isinstance(x, dict) else bool(x) for x in he_res["per_task_bool"]],
        "wt2_per_article": [float(x["nll"]) for x in wt2_res["per_article"]],
        "proc41_80_bool": [bool(x["ok"]) for x in proc_res["per_item"]],
        "proc41_80_margin": [float(x["margin"]) for x in proc_res["per_item"]],
    }


def train_one_p2(model, tok, rl_tasks, rl_prompts, d_c, d_31b, arm: str, seed: int):
    """Exact Iter 13 / 14 / 15 paired training protocol (`steps=60, B=4, G=6, lr=1.5e-4, kl_w=0.02`)."""
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    use_trans = (arm in ("grpo_31b_delta_mlp", "grpo_shuf_delta_mlp"))
    shuffle_geo = (arm == "grpo_shuf_delta_mlp")
    trans_w = 0.40 if use_trans else 0.0
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

    proj_mlps, geo_tgts = {}, {}
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_geo_tot = len(ids_o) - 1

    if use_trans:
        for dep in p2.DEPTHS_GEO:
            arr = d_31b[f"gcca31_{dep}_tr"].astype(np.float32)
            arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-6)
            if shuffle_geo:
                arr = arr[np.random.default_rng(seed + dep).permutation(len(arr))]
            geo_tgts[dep] = arr
            mlp = p2.DiscardableTransitionMLP(d_in=1024, d_hid=512, d_out=arr.shape[1]).to(dev)
            proj_mlps[dep] = mlp
            train_params.extend(list(mlp.parameters()))

        # Pre-warm MLP on frozen Base (250 steps, ~2s)
        p2.set_lora_enabled(wrappers, False)
        pw_opt = torch.optim.AdamW([prm for m in proj_mlps.values() for prm in m.parameters()], lr=2.5e-3, weight_decay=0.01)
        sub_idx = np.random.default_rng(seed + 777).choice(n_geo_tot, size=min(240, n_geo_tot), replace=False)
        cap_pw = {}
        hks_pw = [
            model.model.layers[p2.LAYERS_GEO[dep]].register_forward_hook(
                lambda _m, _i, out, idx=p2.LAYERS_GEO[dep]: cap_pw.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
            )
            for dep in p2.DEPTHS_GEO
        ]
        pw_dh, pw_dz = {d: [] for d in p2.DEPTHS_GEO}, {d: [] for d in p2.DEPTHS_GEO}
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
                        inp_p[bi, : len(ids)] = torch.from_numpy(ids).to(dev); att_p[bi, : len(ids)] = 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model.model(input_ids=inp_p, attention_mask=att_p, use_cache=False)
                    for dep in p2.DEPTHS_GEO:
                        li = p2.LAYERS_GEO[dep]
                        for bi, (pi, pts) in enumerate(zip(chunk, b_pts)):
                            if len(pts) < 3:
                                continue
                            h_s = cap_pw[li][bi, pts, :].float()
                            z_s = torch.from_numpy(geo_tgts[dep][int(pts_o[pi]):int(pts_o[pi + 1])]).to(dev)
                            pw_dh[dep].append(h_s[1:] - h_s[:-1]); pw_dz[dep].append(z_s[1:] - z_s[:-1])
        finally:
            for hk in hks_pw:
                hk.remove()
        for dep in p2.DEPTHS_GEO:
            X_pw, Y_pw = torch.cat(pw_dh[dep], dim=0), torch.cat(pw_dz[dep], dim=0)
            for _ in range(250):
                ii = torch.randint(0, len(X_pw), (min(256, len(X_pw)),), device=dev)
                pr = proj_mlps[dep](X_pw[ii])
                l_pw = (1.0 - F.cosine_similarity(pr, Y_pw[ii], dim=-1).mean()) + F.mse_loss(F.normalize(pr, dim=-1), F.normalize(Y_pw[ii], dim=-1))
                pw_opt.zero_grad(); l_pw.backward(); pw_opt.step()
        del pw_dh, pw_dz, pw_opt
        p2.set_lora_enabled(wrappers, True)
        torch.cuda.empty_cache()

    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    task_perm, geo_perm = rng.permutation(len(rl_tasks)), rng.permutation(n_geo_tot)
    t_cur, g_cur = 0, 0

    try:
        for step in range(steps):
            frac = step / max(1, steps - 1)
            lr_mul = min(1.0, (step + 1) / 8.0) * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul

            b_idx = []
            for _ in range(batch_tasks):
                if t_cur >= len(task_perm):
                    task_perm, t_cur = rng.permutation(len(rl_tasks)), 0
                b_idx.append(int(task_perm[t_cur])); t_cur += 1
            b_tasks = [rl_tasks[i] for i in b_idx]
            b_prompts = [rl_prompts[i] for i in b_idx]

            model.eval()
            orig_side = tok.padding_side
            tok.padding_side = "left"
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
            adv_mat = np.where(sd_mat > 1e-6, adv_mat / (sd_mat + 1e-6), 0.0)
            adv = torch.from_numpy(adv_mat.reshape(-1).astype(np.float32)).to(dev)

            model.train()
            opt.zero_grad(set_to_none=True)
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
                m = resp_mask[sl]
                m_sum = m.sum(dim=1).clamp_min(1.0)
                log_ratio = (ref_lp[sl] - cur_lp).clamp(-10.0, 10.0)
                kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
                pg_seq = -(adv[sl] * ((cur_lp * m).sum(dim=1) / m_sum))
                kl_seq = (kl_tok * m).sum(dim=1) / m_sum
                mb_loss = (pg_seq.mean() + kl_w * kl_seq.mean()) * (len(adv[sl]) / float(N_seq))
                mb_loss.backward()
                del lg_p, lp_p, cur_lp, mb_loss

            if use_trans:
                g_batch = []
                while len(g_batch) < 6:
                    if g_cur >= len(geo_perm):
                        geo_perm, g_cur = rng.permutation(n_geo_tot), 0
                    pi = int(geo_perm[g_cur]); g_cur += 1
                    ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                    pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
                    if len(ids) >= 8 and len(pts) >= 3:
                        g_batch.append((ids, pts, {dep: geo_tgts[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in p2.DEPTHS_GEO}))
                ml_g = max(len(x[0]) for x in g_batch)
                inp_g = torch.full((len(g_batch), ml_g), pad_id, dtype=torch.long, device=dev)
                attn_g = torch.zeros((len(g_batch), ml_g), dtype=torch.long, device=dev)
                for bi, (ids, _, _) in enumerate(g_batch):
                    inp_g[bi, : len(ids)] = torch.from_numpy(ids).to(dev); attn_g[bi, : len(ids)] = 1
                cap_g = {}
                hooks_g = [
                    model.model.layers[p2.LAYERS_GEO[dep]].register_forward_hook(
                        lambda _m, _i, out, idx=p2.LAYERS_GEO[dep]: cap_g.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
                    )
                    for dep in p2.DEPTHS_GEO
                ]
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model.model(input_ids=inp_g, attention_mask=attn_g, use_cache=False)
                    t_loss = inp_g.new_tensor(0.0, dtype=torch.float32)
                    for dep in p2.DEPTHS_GEO:
                        li = p2.LAYERS_GEO[dep]
                        dh = torch.cat([cap_g[li][bi, pts, :].float()[1:] - cap_g[li][bi, pts, :].float()[:-1] for bi, (_, pts, _) in enumerate(g_batch)], dim=0)
                        dz = torch.cat([torch.from_numpy(ts[dep]).to(dev)[1:] - torch.from_numpy(ts[dep]).to(dev)[:-1] for _, _, ts in g_batch], dim=0)
                        pred = proj_mlps[dep](dh)
                        t_loss = t_loss + ((1.0 - F.cosine_similarity(pred, dz, dim=-1).mean()) + F.mse_loss(F.normalize(pred, dim=-1), F.normalize(dz, dim=-1))) / 3.0
                    (trans_w * t_loss).backward()
                finally:
                    for hk in hooks_g:
                        hk.remove()
                del inp_g, attn_g, cap_g

            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            del gen_ids, full_attn, resp_mask, ref_lp, enc
            torch.cuda.empty_cache()

        # Return merged FP32 deltas cleanly (caller restores base weights after eval!)
        deltas = []
        fro_tot = 0.0
        for name, parent, attr, w in wrappers:
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
        del wrappers, proj_mlps, opt
        gc.collect(); torch.cuda.empty_cache()


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev)

    data_dir = base258.EXP257_DATA
    mbpp_all = json.loads((data_dir / "sanitized-mbpp.json").read_text())
    mbpp_dev = [t for t in mbpp_all if int(t["task_id"]) < 11 or int(t["task_id"]) > 510]
    mbpp_test = [t for t in mbpp_all if 11 <= int(t["task_id"]) <= 510]
    d_c = dict(np.load(EXP258_DIR / "targets_qwen600m_code_lattice_30_50_70.npz", allow_pickle=True))
    d_31b = dict(np.load(EXP258_DIR / "targets_qwen600m_31b_gpa_gcca.npz", allow_pickle=True))
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

    # 1. Exact FP32 Base reference on ALL 6 vault benchmarks
    base_p = VAULT_OUT / "p2_vault_base.json"
    if base_p.exists():
        base_res = json.loads(base_p.read_text())
    else:
        base_res = evaluate_full_vault(model, tok, d_c, mbpp_dev, mbpp_test, data_dir)
        base_p.write_text(json.dumps(base_res))
    print(f"[VAULT_C] BASE: {base_res['summary']}", flush=True)

    arms = ("plain_grpo", "grpo_31b_delta_mlp", "grpo_shuf_delta_mlp")
    runs = {a: [] for a in arms}
    for sd in SEEDS:
        for arm in arms:
            cp = VAULT_OUT / f"p2_vault_{arm}_s{sd}.json"
            if cp.exists():
                rec = json.loads(cp.read_text())
                runs[arm].append(rec)
                s = rec["summary"]
                print(f"[VAULT_C] CACHED {arm} s={sd} | HE={s['humaneval_hits']} MBPP_T={s['mbpp_test_hits']} DEV={s['dev_mbpp_hits']} WT2={s['wikitext2_nll']:.5f} CODE={s['dev_code_nll']:.5f} PROC={s['proc41_80_acc']:.4f}", flush=True)
                continue
            print(f"[VAULT_C] Running {arm} s={sd}...", flush=True)
            deltas, fro, tr_sec = train_one_p2(model, tok, rl_tasks, rl_prompts, d_c, d_31b, arm, sd)
            try:
                ev = evaluate_full_vault(model, tok, d_c, mbpp_dev, mbpp_test, data_dir)
            finally:
                with torch.no_grad():
                    for mod, dw_cpu in deltas:
                        mod.weight.data.sub_(dw_cpu.to(dev))
                del deltas
                gc.collect(); torch.cuda.empty_cache()
            ev["arm"] = arm; ev["seed"] = sd; ev["fro_norm"] = fro; ev["train_sec"] = tr_sec
            cp.write_text(json.dumps(ev))
            runs[arm].append(ev)
            s = ev["summary"]
            print(f"[VAULT_C] DONE {arm} s={sd} | HE={s['humaneval_hits']} MBPP_T={s['mbpp_test_hits']} DEV={s['dev_mbpp_hits']} WT2={s['wikitext2_nll']:.5f} CODE={s['dev_code_nll']:.5f} PROC={s['proc41_80_acc']:.4f} fro={fro} t={tr_sec}+{s['eval_sec']}s", flush=True)

    # Aggregate 5-seed stats + paired df=4 t-CIs + 10,000-resample item-level paired bootstraps!
    scalar_keys = ("humaneval_pass1", "mbpp_test_pass1", "dev_mbpp_pass1", "wikitext2_nll",
                   "dev_code_nll", "proc41_80_acc", "proc41_80_margin",
                   "code_blind_30_r2", "code_blind_50_r2", "code_blind_70_r2")
    item_map = {
        "humaneval_pass1": "humaneval_bool",
        "mbpp_test_pass1": "mbpp_test_bool",
        "dev_mbpp_pass1": "dev_mbpp_bool",
        "wikitext2_nll": "wt2_per_article",
        "dev_code_nll": "per_fn_nll",
        "proc41_80_acc": "proc41_80_bool",
        "proc41_80_margin": "proc41_80_margin",
    }
    arm_stats = {arm: {k: sample_stats([r["summary"][k] for r in runs[arm]]) for k in scalar_keys} for arm in arms}
    contrasts = {}
    for a, b in (
        ("grpo_31b_delta_mlp", "plain_grpo"),
        ("grpo_31b_delta_mlp", "grpo_shuf_delta_mlp"),
        ("grpo_shuf_delta_mlp", "plain_grpo"),
    ):
        ck = f"{a}_minus_{b}"
        contrasts[ck] = {}
        for k in scalar_keys:
            c = paired_seed_t([r["summary"][k] for r in runs[a]], [r["summary"][k] for r in runs[b]])
            if k in item_map:
                ik = item_map[k]
                c["item_bootstrap"] = bootstrap_item_diff([r[ik] for r in runs[a]], [r[ik] for r in runs[b]])
            contrasts[ck][k] = c

    # Also contrast vs Base
    vs_base = {}
    for arm in arms:
        vs_base[arm] = {}
        for k in scalar_keys:
            c = paired_seed_t([r["summary"][k] for r in runs[arm]], [base_res["summary"][k]] * len(SEEDS))
            if k in item_map:
                ik = item_map[k]
                c["item_bootstrap"] = bootstrap_item_diff([r[ik] for r in runs[arm]], [base_res[ik]] * len(SEEDS))
            vs_base[arm][k] = c

    final = {"seeds": SEEDS, "base": base_res["summary"], "arms": arm_stats, "vs_base": vs_base, "paired_contrasts": contrasts}
    (VAULT_OUT / "vault_stageC_pillar2_5seed_summary.json").write_text(json.dumps(final, indent=2))
    print("\n=== [VAULT_C] PILLAR 2 LOCKED-VAULT 5-SEED FINAL SUMMARY ===", flush=True)
    for arm in arms:
        st = arm_stats[arm]
        print(f"  {arm:22s} | HE={100*st['humaneval_pass1']['mean']:.2f}±{100*st['humaneval_pass1']['sd']:.2f}% | "
              f"MBPP_T={100*st['mbpp_test_pass1']['mean']:.2f}±{100*st['mbpp_test_pass1']['sd']:.2f}% | "
              f"DEV={100*st['dev_mbpp_pass1']['mean']:.2f}±{100*st['dev_mbpp_pass1']['sd']:.2f}% | "
              f"WT2={st['wikitext2_nll']['mean']:.5f} | CODE={st['dev_code_nll']['mean']:.5f} | PROC={100*st['proc41_80_acc']['mean']:.2f}%", flush=True)
    for ck in ("grpo_31b_delta_mlp_minus_plain_grpo", "grpo_31b_delta_mlp_minus_grpo_shuf_delta_mlp"):
        cv = contrasts[ck]
        print(f"\n  CONTRAST {ck}:", flush=True)
        for k in ("humaneval_pass1", "mbpp_test_pass1", "dev_mbpp_pass1", "wikitext2_nll", "dev_code_nll", "proc41_80_acc"):
            s = cv[k]; b = s.get("item_bootstrap", {})
            print(f"    {k:18s}: seed_mean={s['mean']:+.5f} CI=[{s['ci95_lo']:+.5f},{s['ci95_hi']:+.5f}] t={s['t_stat']} | item_boot_p={b.get('boot_p_two_sided')}", flush=True)


if __name__ == "__main__":
    main()
