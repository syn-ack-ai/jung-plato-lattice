"""Experiment 257b: Strictly Unseen Proc-200 Replication (Items 41-80 per Kind) +
Same-Family Qwen3-1.7B Teacher & Plain Next-Token CE Ablation Arms (5 Paired Seeds).

Addresses external audit follow-ups #1 and #3:
  1. Scores strictly unseen items 41-80 of each of the 5 kinds in `proc_held.jsonl`
     (200 brand-new items, ZERO overlap with items 1-40 used in Exp 257), while also
     scoring items 1-40 to verify exact reproduction of Exp 257 checkpoints.
  2. Adds mechanistic ablation arms on the exact same LoRA architecture & passages:
       - `same_family_qwen17b_gut`: Same-family Qwen/Qwen3-1.7B 30%/50% hidden-state
         teacher on audited Gutenberg prose (ce_w=0.0, exact same Native-Frame recipe).
       - `plain_ce_gut`: Plain next-token cross-entropy (`ce_w=0.50`, `kl_w=3.20`,
         zero geometric loss) on the exact same audited Gutenberg passages.
       - `plain_ce_stratified`: Plain next-token cross-entropy (`ce_w=0.50`,
         `kl_w=3.20`, zero geometric loss) on all 3 domains.
  3. Computes BOTH seed-level paired t-intervals (`df=4`, `ddof=1`) AND 10,000-resample
     item-level paired bootstraps (accuracy + answer margin + by-kind breakdown) on
     BOTH `proc_0_39` and strictly unseen `proc_40_79`, plus `held_gut_nll`,
     `wikitext2_nll`, and blind `Phi-4` / `OLMo 3` OOF `r2_frob`.
"""
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_rigorous as ev  # noqa: E402
import recipe_rigorous as rec  # noqa: E402
import run_causal_ladder as rcl  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]
QWEN_TEACHER_ID = os.environ.get("EXP257B_QWEN_TEACHER_ID", "Qwen/Qwen3-1.7B")


@torch.no_grad()
def eval_proc_mcq_slice(model, tok, probes_dir: Path, start_idx: int, end_idx: int):
    """Score items [start_idx:end_idx] of each kind in proc_held.jsonl in exact fp32."""
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    proc_all = [json.loads(l) for l in open(probes_dir / "proc_held.jsonl") if l.strip()]
    by_k = {}
    for r in proc_all:
        by_k.setdefault(r["kind"], []).append(r)
    proc_rows = [r for k in sorted(by_k.keys()) for r in by_k[k][start_idx:end_idx]]

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
        ans_s = [sc for sc, ok in opts if ok][0]
        wrg_s = max(sc for sc, ok in opts if not ok)
        ok = ans_s > wrg_s
        per_item.append({"idx": ri, "kind": r["kind"], "ok": bool(ok), "margin": round(float(ans_s - wrg_s), 6)})
        by_kind.setdefault(r["kind"], []).append(ok)
    return {
        "proc_acc": round(float(np.mean([x["ok"] for x in per_item])), 5),
        "proc_margin": round(float(np.mean([x["margin"] for x in per_item])), 6),
        "by_kind": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_kind.items())},
        "by_kind_hits": {k: int(sum(v)) for k, v in sorted(by_kind.items())},
        "n_items": len(per_item),
        "per_item": per_item,
    }


@torch.no_grad()
def extract_same_family_qwen_targets(d_t, tok, cache_path: Path):
    """Extract same-family Qwen/Qwen3-1.7B 30% & 50% states on d_t train passages and PCA-256."""
    if cache_path.exists():
        print(f"[EXP257B] Loaded cached same-family teacher targets: {cache_path}", flush=True)
        return np.load(cache_path, allow_pickle=True)

    print(f"[EXP257B] Extracting same-family teacher {QWEN_TEACHER_ID} on d_t passages...", flush=True)
    t0 = time.time()
    t_model = AutoModelForCausalLM.from_pretrained(
        QWEN_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    t_model.eval()
    n_layers = t_model.config.num_hidden_layers
    l30 = round(0.30 * n_layers)
    l50 = round(0.50 * n_layers)
    # Qwen3-1.7B has 28 layers -> l30=8, l50=14 (exact match to Qwen3-0.6B!)
    print(f"[EXP257B] {QWEN_TEACHER_ID}: L={n_layers}, hooking l30={l30}, l50={l50}", flush=True)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(t_model.parameters()).device
    ids_f, ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    pts_f, pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
    n_pass = len(ids_o) - 1

    cap = {}
    hooks = [
        t_model.model.layers[l30].register_forward_hook(lambda _m, _i, out: cap.__setitem__(30, out[0] if isinstance(out, tuple) else out)),
        t_model.model.layers[l50].register_forward_hook(lambda _m, _i, out: cap.__setitem__(50, out[0] if isinstance(out, tuple) else out)),
    ]
    h30_l, h50_l = [], []
    try:
        for s in range(0, n_pass, 16):
            e = min(n_pass, s + 16)
            b_ids = [ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64) for i in range(s, e)]
            b_pts = [pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64) for i in range(s, e)]
            ml = max(len(x) for x in b_ids)
            inp = torch.full((len(b_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(b_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(b_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            _ = t_model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            c30 = cap[30].float().cpu().numpy()
            c50 = cap[50].float().cpu().numpy()
            for bi, pts in enumerate(b_pts):
                h30_l.append(c30[bi, pts, :])
                h50_l.append(c50[bi, pts, :])
            del inp, attn
    finally:
        for hk in hooks:
            hk.remove()
        del t_model
        gc.collect()
        torch.cuda.empty_cache()

    H30 = np.concatenate(h30_l, axis=0).astype(np.float64)
    H50 = np.concatenate(h50_l, axis=0).astype(np.float64)
    d_out = d_t["r3_30_tr"].shape[1]

    def _pca_unit(H):
        mu = H.mean(axis=0, keepdims=True)
        Hc = H - mu
        cov = (Hc.T @ Hc) / float(len(Hc))
        vals, vecs = np.linalg.eigh(cov)
        V = vecs[:, np.argsort(vals)[::-1][:d_out]]
        Z = Hc @ V
        rms = float(np.sqrt(np.mean(Z ** 2))) + 1e-12
        return (Z / rms).astype(np.float32)

    q30 = _pca_unit(H30)
    q50 = _pca_unit(H50)
    np.savez_compressed(cache_path, q17b_30_tr=q30, q17b_50_tr=q50, l30=l30, l50=l50)
    print(f"[EXP257B] Saved same-family Qwen3-1.7B targets {q30.shape} in {time.time()-t0:.1f}s", flush=True)
    return np.load(cache_path, allow_pickle=True)


def run_plain_ce_arm(model, tok, d_t, gut_only: bool, seed: int, ce_w: float = 0.50, kl_w: float = 3.20):
    """Plain next-token CE + KL-anchor baseline on exact same L2..L20 LoRA & passages (zero geometry)."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rank, steps, batch_size, lr = 24, 360, 8, 2.4e-4
    if gut_only:
        train_pool, n_rej = rec.get_audited_gut_indices(d_t, tok)
        extra_meta = {"gut_audited_kept": len(train_pool), "gut_regex_rejected": n_rej}
    else:
        train_pool = list(range(len(d_t["tr_domains"])))
        extra_meta = {"n_train_passages": len(train_pool)}

    for p in model.parameters():
        p.requires_grad_(False)
    wrappers, all_lora = [], []
    for li in rec.ACTIVE_LAYERS:
        layer = model.model.layers[li]
        for mod_name, parent, attr in (("o_proj", layer.self_attn, "o_proj"), ("down_proj", layer.mlp, "down_proj")):
            w = rec.LoRALinearWrap(getattr(parent, attr), r=rank, alpha=float(rank))
            setattr(parent, attr, w)
            wrappers.append((f"L{li}.{mod_name}", parent, attr, w))
            all_lora.extend([w.lora_A, w.lora_B])

    opt = torch.optim.AdamW(all_lora, lr=lr, weight_decay=0.012, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    pool_arr = np.array(train_pool, dtype=np.int64)
    perm_p, cur_p = rng.permutation(pool_arr), 0
    ids_f, ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]

    history = []
    completed = False
    merge_kl = max_logit_diff = 0.0
    fro_norms = {}
    try:
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr_mul = min(1.0, (s + 1) / 20.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            opt.param_groups[0]["lr"] = lr * lr_mul

            batch = []
            while len(batch) < batch_size:
                if cur_p >= len(perm_p):
                    perm_p, cur_p = rng.permutation(pool_arr), 0
                pi = int(perm_p[cur_p]); cur_p += 1
                ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                if len(ids) >= 4:
                    batch.append(ids)

            ml = max(len(x) for x in batch)
            B = len(batch)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((B, ml), dtype=torch.long, device=dev)
            valid_tok = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
            for bi, ids in enumerate(batch):
                L = len(ids)
                inp[bi, :L] = torch.from_numpy(ids).to(dev)
                attn[bi, :L] = 1
                valid_tok[bi, : L - 1] = 1.0
            v_sum = valid_tok.sum().clamp_min(1.0)

            for _, _, _, w in wrappers:
                w.active = False
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_r = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
                lp_ref = F.log_softmax(lg_r.float(), dim=-1)
                p_ref = lp_ref.exp()
                del lg_r
            for _, _, _, w in wrappers:
                w.active = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            lp_cur = F.log_softmax(lg.float(), dim=-1)
            del lg
            tok_kl = (p_ref * (lp_ref - lp_cur)).sum(dim=-1)
            kl = (tok_kl * valid_tok).sum() / v_sum

            tgt = inp[:, 1:].unsqueeze(-1)
            tok_ce = -lp_cur.gather(-1, tgt).squeeze(-1)
            ce = (tok_ce * valid_tok).sum() / v_sum

            loss = ce_w * ce + kl_w * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            del lp_cur, lp_ref, p_ref, tok_kl, tok_ce
            torch.nn.utils.clip_grad_norm_(all_lora, 1.0)
            opt.step()

            if s % 90 == 0 or s == steps - 1:
                history.append({"step": s, "kl": round(float(kl.item()), 6), "ce": round(float(ce.item()), 4)})
        completed = True
    finally:
        opt.zero_grad(set_to_none=True)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if completed:
            merge_kl, max_logit_diff, fro_norms = rec._measure_merge_kl_fp32(model, wrappers, d_t, pad_id, dev)
        else:
            for _, parent, attr, w in wrappers:
                setattr(parent, attr, w.base)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        del wrappers
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "arm": "plain_ce_gut" if gut_only else "plain_ce_stratified",
        "seed": seed,
        "ce_w": ce_w,
        "kl_w": kl_w,
        "rank": rank,
        "steps": steps,
        "merge_kl_fp32": round(merge_kl, 9),
        "total_fro_norm": round(float(np.sqrt(sum(v * v for v in fro_norms.values()))), 6),
        "delta_fro_norms": {k: round(v, 6) for k, v in fro_norms.items()},
        "train_sec": round(time.time() - t0, 1),
        "history": history,
        **extra_meta,
    }


def eval_fast_replication(model, tok, d_t, paths):
    t0 = time.time()
    model.eval()
    p_old = eval_proc_mcq_slice(model, tok, paths["probes_dir"], 0, 40)
    p_new = eval_proc_mcq_slice(model, tok, paths["probes_dir"], 40, 80)
    gut = ev.eval_held_gut_nll(model, tok, paths["plato_dir"])
    wt2 = ev.eval_wikitext2_nll(model, tok, paths["data_dir"])
    al = ev.eval_blind_alignment(model, tok, d_t)
    return {
        "summary": {
            "proc_old_acc": p_old["proc_acc"],
            "proc_old_margin": p_old["proc_margin"],
            "proc_unseen_acc": p_new["proc_acc"],
            "proc_unseen_margin": p_new["proc_margin"],
            "proc_all400_acc": round(0.5 * (p_old["proc_acc"] + p_new["proc_acc"]), 5),
            "proc_all400_margin": round(0.5 * (p_old["proc_margin"] + p_new["proc_margin"]), 6),
            "held_gut_nll": gut["nll"],
            "wikitext2_nll": wt2["nll"],
            "blind_30_r2_frob": al["blind_30_r2_frob"],
            "blind_50_r2_frob": al["blind_50_r2_frob"],
            "eval_sec": round(time.time() - t0, 1),
        },
        "proc_old_by_kind_hits": p_old["by_kind_hits"],
        "proc_unseen_by_kind_hits": p_new["by_kind_hits"],
        "proc_old_per_item": p_old["per_item"],
        "proc_unseen_per_item": p_new["per_item"],
        "held_gut_per_passage": gut["per_passage"],
        "wikitext2_per_article": wt2["per_article"],
    }


def summarize_257b(runs_dir: Path, out_path: Path):
    base = json.loads((runs_dir / "base.json").read_text())
    bsum = base["summary"]
    arms = [
        "true_gut_only",
        "true_stratified_r3",
        "true_dual_domain",
        "same_family_qwen17b_gut",
        "plain_ce_gut",
        "plain_ce_stratified",
        "shuffle_within_domain",
        "random_lora_matched",
    ]
    loaded = {}
    for arm in arms:
        loaded[arm] = [json.loads((runs_dir / f"{arm}_seed{s}.json").read_text()) for s in SEEDS if (runs_dir / f"{arm}_seed{s}.json").exists()]

    metrics = [
        "proc_old_acc", "proc_old_margin",
        "proc_unseen_acc", "proc_unseen_margin",
        "proc_all400_acc", "proc_all400_margin",
        "held_gut_nll", "wikitext2_nll",
        "blind_30_r2_frob", "blind_50_r2_frob",
    ]
    arm_tables = {}
    for arm in arms:
        recs = loaded[arm]
        if not recs:
            continue
        tbl = {"n_seeds": len(recs), "abs": {}, "delta_vs_base": {}, "proc_unseen_by_kind_mean_hits": {}}
        for m in metrics:
            vals = [r["summary"][m] for r in recs]
            tbl["abs"][m] = rcl.sample_stats(vals)
            tbl["delta_vs_base"][m] = rcl.sample_stats([v - bsum[m] for v in vals])
        tbl["total_fro_norm"] = rcl.sample_stats([r["train_meta"]["total_fro_norm"] for r in recs])
        kinds = sorted(base["proc_unseen_by_kind_hits"].keys())
        for k in kinds:
            tbl["proc_unseen_by_kind_mean_hits"][k] = round(float(np.mean([r["proc_unseen_by_kind_hits"][k] for r in recs])), 2)
        arm_tables[arm] = tbl

    contrasts = {}
    pairs = [
        ("true_gut_only", "random_lora_matched"),
        ("true_stratified_r3", "random_lora_matched"),
        ("true_dual_domain", "random_lora_matched"),
        ("same_family_qwen17b_gut", "random_lora_matched"),
        ("plain_ce_gut", "random_lora_matched"),
        ("plain_ce_stratified", "random_lora_matched"),
        ("true_gut_only", "same_family_qwen17b_gut"),
        ("true_gut_only", "plain_ce_gut"),
        ("true_stratified_r3", "plain_ce_stratified"),
        ("true_gut_only", "shuffle_within_domain"),
        ("true_stratified_r3", "shuffle_within_domain"),
    ]
    for a, b in pairs:
        if len(loaded.get(a, [])) == len(SEEDS) and len(loaded.get(b, [])) == len(SEEDS):
            ck = f"{a}_minus_{b}"
            contrasts[ck] = {}
            for m in metrics:
                contrasts[ck][m] = rcl.paired_seed_contrast([r["summary"][m] for r in loaded[a]], [r["summary"][m] for r in loaded[b]])
            boot = {}
            for tag, key in (("proc_old", "proc_old_per_item"), ("proc_unseen", "proc_unseen_per_item")):
                pa = np.mean([[float(x["ok"]) for x in r[key]] for r in loaded[a]], axis=0)
                pb = np.mean([[float(x["ok"]) for x in r[key]] for r in loaded[b]], axis=0)
                ma = np.mean([[float(x["margin"]) for x in r[key]] for r in loaded[a]], axis=0)
                mb = np.mean([[float(x["margin"]) for x in r[key]] for r in loaded[b]], axis=0)
                boot[f"{tag}_item_acc"] = rcl.cluster_bootstrap_diff(pa - pb)
                boot[f"{tag}_item_margin"] = rcl.cluster_bootstrap_diff(ma - mb)
            # Combined 400-item bootstrap
            p4a = np.mean([[float(x["ok"]) for x in (r["proc_old_per_item"] + r["proc_unseen_per_item"])] for r in loaded[a]], axis=0)
            p4b = np.mean([[float(x["ok"]) for x in (r["proc_old_per_item"] + r["proc_unseen_per_item"])] for r in loaded[b]], axis=0)
            m4a = np.mean([[float(x["margin"]) for x in (r["proc_old_per_item"] + r["proc_unseen_per_item"])] for r in loaded[a]], axis=0)
            m4b = np.mean([[float(x["margin"]) for x in (r["proc_old_per_item"] + r["proc_unseen_per_item"])] for r in loaded[b]], axis=0)
            boot["proc_all400_item_acc"] = rcl.cluster_bootstrap_diff(p4a - p4b)
            boot["proc_all400_item_margin"] = rcl.cluster_bootstrap_diff(m4a - m4b)
            gt_a = np.mean([[x["nll"] for x in r["held_gut_per_passage"]] for r in loaded[a]], axis=0)
            gt_b = np.mean([[x["nll"] for x in r["held_gut_per_passage"]] for r in loaded[b]], axis=0)
            boot["held_gut_passage_nll"] = rcl.cluster_bootstrap_diff(gt_a - gt_b)
            wt_a = np.mean([[x["nll"] for x in r["wikitext2_per_article"]] for r in loaded[a]], axis=0)
            wt_b = np.mean([[x["nll"] for x in r["wikitext2_per_article"]] for r in loaded[b]], axis=0)
            boot["wikitext2_article_nll"] = rcl.cluster_bootstrap_diff(wt_a - wt_b)
            contrasts[ck]["cluster_bootstrap"] = boot

    out = {
        "model_id": ev.MODEL_ID,
        "same_family_teacher_id": QWEN_TEACHER_ID,
        "seeds": SEEDS,
        "base": bsum,
        "base_proc_old_by_kind_hits": base["proc_old_by_kind_hits"],
        "base_proc_unseen_by_kind_hits": base["proc_unseen_by_kind_hits"],
        "arms": arm_tables,
        "paired_contrasts": contrasts,
    }
    out_path.write_text(json.dumps(out, indent=1))
    return out


def main():
    paths = ev.default_paths()
    res_dir = paths["exp_dir"] / "results_257b"
    runs_dir = res_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    d_t = dict(np.load(paths["targets_npz"], allow_pickle=True))
    tok = AutoTokenizer.from_pretrained(ev.MODEL_ID)

    # 0. Pre-extract same-family Qwen3-1.7B targets before loading fp32 student (clean VRAM!)
    q_cache = res_dir / "qwen17b_teacher_30_50.npz"
    q_t = extract_same_family_qwen_targets(d_t, tok, q_cache)
    d_q17 = dict(d_t)
    d_q17["r3_30_tr"] = q_t["q17b_30_tr"]
    d_q17["r3_50_tr"] = q_t["q17b_50_tr"]

    model = AutoModelForCausalLM.from_pretrained(ev.MODEL_ID, dtype=torch.float32, device_map="cuda:0")
    model.eval()
    base_sd_cpu = rcl.load_base_sd(model)

    base_path = runs_dir / "base.json"
    if not base_path.exists():
        print("[EXP257B] Running Base Qwen3-0.6B (exact fp32)...", flush=True)
        recs = eval_fast_replication(model, tok, d_t, paths)
        base_path.write_text(json.dumps(recs, indent=1))
        print("[EXP257B] BASE:", json.dumps(recs["summary"]), "unseen_by_kind:", recs["proc_unseen_by_kind_hits"], flush=True)
    else:
        print("[EXP257B] Cached Base loaded.", flush=True)

    arm_order = [
        "true_stratified_r3",
        "random_lora_matched",
        "true_gut_only",
        "same_family_qwen17b_gut",
        "plain_ce_gut",
        "plain_ce_stratified",
        "shuffle_within_domain",
        "true_dual_domain",
    ]
    for seed in SEEDS:
        for arm in arm_order:
            run_path = runs_dir / f"{arm}_seed{seed}.json"
            if run_path.exists():
                print(f"[EXP257B] Cached {arm} s={seed}.", flush=True)
                continue
            print(f"[EXP257B] Running {arm} s={seed}...", flush=True)
            rcl.restore_base_sd(model, base_sd_cpu)
            if arm == "random_lora_matched":
                ref_r3 = json.loads((runs_dir / f"true_stratified_r3_seed{seed}.json").read_text())
                tm = rec.run_arm(model, tok, d_t, arm="random_lora_matched", seed=seed, matched_fro_norms=ref_r3["train_meta"]["delta_fro_norms"])
            elif arm == "same_family_qwen17b_gut":
                tm = rec.run_arm(model, tok, d_q17, arm="true_gut_only", seed=seed)
                tm["arm"] = "same_family_qwen17b_gut"
                tm["teacher_id"] = QWEN_TEACHER_ID
            elif arm == "plain_ce_gut":
                tm = run_plain_ce_arm(model, tok, d_t, gut_only=True, seed=seed, ce_w=0.50, kl_w=3.20)
            elif arm == "plain_ce_stratified":
                tm = run_plain_ce_arm(model, tok, d_t, gut_only=False, seed=seed, ce_w=0.50, kl_w=3.20)
            else:
                tm = rec.run_arm(model, tok, d_t, arm=arm, seed=seed, code_targets_npz=paths["code_targets_npz"])

            recs = eval_fast_replication(model, tok, d_t, paths)
            recs["train_meta"] = tm
            run_path.write_text(json.dumps(recs, indent=1))
            s_out = recs["summary"]
            print(
                f"[EXP257B] DONE {arm} s={seed} | old={s_out['proc_old_acc']*100:.1f}% unseen={s_out['proc_unseen_acc']*100:.1f}% "
                f"(m_unseen={s_out['proc_unseen_margin']:+.4f}) gut={s_out['held_gut_nll']:.5f} wt2={s_out['wikitext2_nll']:.5f} "
                f"L8={s_out['blind_30_r2_frob']:.4f} L14={s_out['blind_50_r2_frob']:.4f} fro={tm['total_fro_norm']:.3f} t={tm['train_sec']}+{s_out['eval_sec']}s",
                flush=True,
            )
        summarize_257b(runs_dir, res_dir / "proc_unseen_and_ablations_summary.json")

    final_out = summarize_257b(runs_dir, res_dir / "proc_unseen_and_ablations_summary.json")
    print("[EXP257B] ALL_DONE!", flush=True)
    for arm, tbl in final_out["arms"].items():
        print(
            f"  {arm:24s} | old={tbl['delta_vs_base']['proc_old_acc']['mean']*100:+.2f}pp "
            f"unseen={tbl['delta_vs_base']['proc_unseen_acc']['mean']*100:+.2f}pp "
            f"gut={tbl['delta_vs_base']['held_gut_nll']['mean']:+.5f} "
            f"wt2={tbl['delta_vs_base']['wikitext2_nll']['mean']:+.5f} "
            f"fro={tbl['total_fro_norm']['mean']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
