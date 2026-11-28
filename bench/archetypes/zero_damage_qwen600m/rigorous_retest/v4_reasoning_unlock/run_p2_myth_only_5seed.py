"""Myth-Only Factorial Ablation (`Seeds 101..105`, Exact FP32 `Qwen/Qwen3-0.6B`).

Completes the exact 2x2 factorial against `plain_grpo (0/0)`, `grpo_31b_delta_mlp (0.40 Code / 0 Myth)`,
and `grpo_31b_myth_code_delta_mlp (0.20 Code / 0.20 Myth)` by testing TWO strictly ZERO-CODE-LATTICE regimes:
  1. `grpo_31b_myth_only_delta_mlp`: Python GRPO + `0.40` Myth-Only `Δh`-MLP (ZERO code in geometric lattice!)
     + exact matched `grpo_shuf_myth_only_delta_mlp` control (`shuffle_geo=True`).
  2. `pure_myth_only_zero_code`: ZERO Python GRPO, ZERO Code anywhere in training — pure `0.40` Myth `Δh`-MLP
     with `0.02` base-KL anchor on myth passages (pure zero-shot cross-domain transfer from myth -> code!)
     + exact matched `pure_shuf_myth_zero_code` control (`shuffle_geo=True`).

All 4 arms share exact `L2..L27` LoRA (`r=16, steps=60, lr=1.5e-4, kl_w=0.02`), exact FP32 evaluation across all
6 Vault benchmarks (`HumanEval 164`, `MBPP Test 257`, `dev_mbpp_170`, `WikiText-2 62`, `code_eval 399`, `Proc 41-80 200`)
+ strictly zero-shot held-out *Odyssey* + *Phantastes* Myth NLL. Fully checkpointed per `(arm, seed)`.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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

SEEDS = [101, 102, 103, 104, 105]
MODEL_ID = "Qwen/Qwen3-0.6B"


def train_myth_only(model, tok, rl_tasks, rl_prompts, d_m, seed: int, use_grpo: bool, shuffle_geo: bool,
                    myth_key_prefix: str = "myth_gcca", trans_w: float = 0.40, depths=None):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    steps, batch_tasks, G, lr, kl_w, rank = 60, 4, 6, 1.5e-4, 0.02, 16
    use_depths = tuple(depths) if depths is not None else p2.DEPTHS_GEO

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

    myth_tgts, myth_mlps = {}, {}
    for dep in use_depths:
        am = d_m[f"{myth_key_prefix}_{dep}_tr"].astype(np.float32)
        am = (am - am.mean(0, keepdims=True)) / (am.std(0, keepdims=True) + 1e-6)
        if shuffle_geo:
            am = am[np.random.default_rng(seed + dep + 500).permutation(len(am))]
        myth_tgts[dep] = am
        mm = p2.DiscardableTransitionMLP(d_in=1024, d_hid=512, d_out=am.shape[1]).to(dev)
        myth_mlps[dep] = mm
        train_params.extend(list(mm.parameters()))

    # Pre-warm Myth Δh-MLP on frozen Base (250 steps, ~2s)
    dual.prewarm_domain_mlps(model, wrappers, d_m, myth_tgts, myth_mlps, pad_id, dev, seed + 888)

    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    task_perm = rng.permutation(len(rl_tasks)); t_cur = 0
    m_perm = rng.permutation(len(d_m["tr_ids_off"]) - 1); m_cur = 0
    ids_f, ids_o = d_m["tr_ids_flat"], d_m["tr_ids_off"]
    n_myth_tot = len(ids_o) - 1

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
            else:
                # ZERO CODE / ZERO GRPO: Base-KL trust anchor directly on myth passages!
                model.train()
                kl_idx = rng.integers(0, n_myth_tot, size=4)
                b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64)[:512] for i in kl_idx]
                ml = max(len(x) for x in b_ids)
                inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
                att = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
                for bi, ids in enumerate(b_ids):
                    inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev); att[bi, : len(ids)] = 1
                with torch.no_grad():
                    p2.set_lora_enabled(wrappers, False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg_ref = model(input_ids=inp, attention_mask=att, use_cache=False).logits[:, :-1, :].float()
                    p_ref = F.softmax(lg_ref, dim=-1)
                    log_p_ref = F.log_softmax(lg_ref, dim=-1)
                    p2.set_lora_enabled(wrappers, True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_cur = model(input_ids=inp, attention_mask=att, use_cache=False).logits[:, :-1, :].float()
                log_p_cur = F.log_softmax(lg_cur, dim=-1)
                mask = att[:, 1:].float()
                kl_tok = (p_ref * (log_p_ref - log_p_cur)).sum(dim=-1)
                kl_loss = (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)
                (kl_w * kl_loss).backward()
                del inp, att, lg_ref, p_ref, log_p_ref, lg_cur, log_p_cur

            # Pure Myth step-transition gradient (weight = 0.40, ZERO code transitions!)
            m_perm, m_cur = dual.domain_delta_step(model, d_m, myth_tgts, myth_mlps, m_perm, m_cur, rng, pad_id, dev, weight=trans_w)

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
        del wrappers, myth_mlps, opt
        gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--myth-npz", type=str, default=str(CACHE / "targets_myth_epic_31b_gpa_gcca.npz"))
    ap.add_argument("--myth-key-prefix", type=str, default="myth_gcca")
    ap.add_argument("--tag", type=str, default="myth_only")
    ap.add_argument("--arms", type=str, default="grpo_myth,pure_myth,grpo_shuf_myth,pure_shuf_myth")
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev)

    data_dir = base258.EXP257_DATA
    mbpp_all = json.loads((data_dir / "sanitized-mbpp.json").read_text())
    mbpp_dev = [t for t in mbpp_all if int(t["task_id"]) < 11 or int(t["task_id"]) > 510]
    mbpp_test = [t for t in mbpp_all if 11 <= int(t["task_id"]) <= 510]
    d_c = dict(np.load(EXP258_DIR / "targets_qwen600m_code_lattice_30_50_70.npz", allow_pickle=True))
    d_m = dict(np.load(args.myth_npz, allow_pickle=True))
    # Always evaluate held-out Myth NLL on the canonical locked *Odyssey* + *Phantastes* holdout!
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

    base_myth_nll = dual.eval_myth_nll(model, d_m_eval, pad_id, dev)
    print(f"[{args.tag.upper()}] Base locked *Odyssey*+*Phantastes* NLL: {base_myth_nll} | lattice={Path(args.myth_npz).name}", flush=True)

    arm_specs = {
        "grpo_myth": (f"grpo_31b_{args.tag}_delta_mlp", True, False),
        "pure_myth": (f"pure_31b_{args.tag}_zero_code", False, False),
        "grpo_shuf_myth": (f"grpo_shuf_{args.tag}_delta_mlp", True, True),
        "pure_shuf_myth": (f"pure_shuf_{args.tag}_zero_code", False, True),
    }
    wanted = [x.strip() for x in args.arms.split(",") if x.strip()]

    keys = ("humaneval_pass1", "mbpp_test_pass1", "dev_mbpp_pass1", "wikitext2_nll",
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

    existing = {a: [json.loads((VAULT_OUT / f"p2_vault_{a}_s{sd}.json").read_text()) for sd in SEEDS]
                for a in ("plain_grpo", "grpo_31b_delta_mlp", "grpo_shuf_delta_mlp")}
    if all((VAULT_OUT / f"p2_vault_grpo_31b_myth_code_delta_mlp_s{sd}.json").exists() for sd in SEEDS):
        existing["grpo_31b_myth_code_delta_mlp"] = [
            json.loads((VAULT_OUT / f"p2_vault_grpo_31b_myth_code_delta_mlp_s{sd}.json").read_text()) for sd in SEEDS
        ]
    base_d = json.loads((VAULT_OUT / "p2_vault_base.json").read_text())

    def build_contrast(runs_a, runs_b, include_myth=False):
        c = {}
        c_keys = list(keys) + (["held_myth_nll"] if include_myth else [])
        for k in c_keys:
            st = vp2.paired_seed_t([r["summary"][k] for r in runs_a], [r["summary"][k] for r in runs_b])
            if k in item_map:
                ik = item_map[k]
                if all(ik in r and len(r[ik]) > 0 for r in runs_a) and all(ik in r and len(r[ik]) > 0 for r in runs_b):
                    legacy_all_true = (k in ("humaneval_pass1", "dev_mbpp_pass1") and all(all(x is True for x in r[ik]) for r in runs_b))
                    if not legacy_all_true:
                        st["item_bootstrap"] = vp2.bootstrap_item_diff([r[ik] for r in runs_a], [r[ik] for r in runs_b])
            c[k] = st
        return c

    all_runs, all_stats, all_contrasts = {}, {}, {}
    summary_path = VAULT_OUT / f"vault_stageC_{args.tag}_5seed_summary.json"

    for short_k in wanted:
        arm_name, use_grpo, shuf = arm_specs[short_k]
        runs = []
        for sd in SEEDS:
            cp = VAULT_OUT / f"p2_vault_{arm_name}_s{sd}.json"
            if cp.exists():
                ev = json.loads(cp.read_text())
                runs.append(ev)
                s = ev["summary"]
                print(f"[{args.tag.upper()}:{arm_name}] CACHED s={sd} | HE={s['humaneval_hits']} MBPP_T={s['mbpp_test_hits']} DEV={s['dev_mbpp_hits']} WT2={s['wikitext2_nll']:.5f} CODE={s['dev_code_nll']:.5f} MYTH={s['held_myth_nll']:.5f} PROC={s['proc41_80_acc']:.4f}", flush=True)
                continue
            print(f"[{args.tag.upper()}:{arm_name}] Running s={sd} (grpo={use_grpo}, shuf={shuf})...", flush=True)
            deltas, fro, tr_sec = train_myth_only(
                model, tok, rl_tasks, rl_prompts, d_m, sd,
                use_grpo=use_grpo, shuffle_geo=shuf, myth_key_prefix=args.myth_key_prefix, trans_w=0.40,
            )
            try:
                ev = vp2.evaluate_full_vault(model, tok, d_c, mbpp_dev, mbpp_test, data_dir)
                ev["summary"]["held_myth_nll"] = dual.eval_myth_nll(model, d_m_eval, pad_id, dev)
            finally:
                with torch.no_grad():
                    for mod, dw_cpu in deltas:
                        mod.weight.data.sub_(dw_cpu.to(dev))
                del deltas; gc.collect(); torch.cuda.empty_cache()
            ev["arm"] = arm_name; ev["seed"] = sd; ev["fro_norm"] = fro; ev["train_sec"] = tr_sec
            cp.write_text(json.dumps(ev))
            runs.append(ev)
            s = ev["summary"]
            print(f"[{args.tag.upper()}:{arm_name}] DONE s={sd} ({tr_sec:.1f}s) | HE={s['humaneval_hits']} MBPP_T={s['mbpp_test_hits']} DEV={s['dev_mbpp_hits']} WT2={s['wikitext2_nll']:.5f} CODE={s['dev_code_nll']:.5f} MYTH={s['held_myth_nll']:.5f} PROC={s['proc41_80_acc']:.4f} fro={fro}", flush=True)

        all_runs[short_k] = runs
        st = {k: vp2.sample_stats([r["summary"][k] for r in runs]) for k in keys}
        st["held_myth_nll"] = vp2.sample_stats([r["summary"]["held_myth_nll"] for r in runs])
        all_stats[arm_name] = st
        all_contrasts[f"{arm_name}_minus_base"] = build_contrast(runs, [base_d] * 5, include_myth=False)
        all_contrasts[f"{arm_name}_minus_base"]["held_myth_nll"] = vp2.paired_seed_t([r["summary"]["held_myth_nll"] for r in runs], [base_myth_nll] * 5)
        for ref_name, ref_runs in existing.items():
            all_contrasts[f"{arm_name}_minus_{ref_name}"] = build_contrast(runs, ref_runs, include_myth=("myth" in ref_name))
        if short_k == "grpo_shuf_myth" and "grpo_myth" in all_runs:
            all_contrasts[f"{arm_specs['grpo_myth'][0]}_minus_{arm_name}"] = build_contrast(all_runs["grpo_myth"], runs, include_myth=True)
        if short_k == "pure_shuf_myth" and "pure_myth" in all_runs:
            all_contrasts[f"{arm_specs['pure_myth'][0]}_minus_{arm_name}"] = build_contrast(all_runs["pure_myth"], runs, include_myth=True)

        # Persist progressive summary after each completed arm!
        summary_path.write_text(json.dumps({
            "seeds": SEEDS, "tag": args.tag, "myth_npz": Path(args.myth_npz).name,
            "base_myth_nll": base_myth_nll, "arms": all_stats, "paired_contrasts": all_contrasts,
        }, indent=2))
        print(f"=== [{args.tag.upper()}] ARM COMPLETE: {arm_name} | "
              f"HE={100*st['humaneval_pass1']['mean']:.2f}±{100*st['humaneval_pass1']['sd']:.2f}% | "
              f"MBPP_T={100*st['mbpp_test_pass1']['mean']:.2f}±{100*st['mbpp_test_pass1']['sd']:.2f}% | "
              f"DEV={100*st['dev_mbpp_pass1']['mean']:.2f}% | WT2={st['wikitext2_nll']['mean']:.5f} | "
              f"CODE={st['dev_code_nll']['mean']:.5f} | MYTH={st['held_myth_nll']['mean']:.5f} | "
              f"PROC={100*st['proc41_80_acc']['mean']:.2f}% ===", flush=True)


if __name__ == "__main__":
    main()
