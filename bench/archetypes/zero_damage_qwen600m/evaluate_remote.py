"""IMMUTABLE MECHANICAL EVALUATOR (Remote RTX 4090 Harness for Exp 256: Qwen3-0.6B).
DO NOT MODIFY AFTER BASELINE INITIALIZATION!

Evaluates `transplant_recipe.run_transplant(model, tok, d_t, seed=256)` on official unmodified
`Qwen/Qwen3-0.6B` (28 layers, D=1024, L8=30%, L14=50%) strictly against cached `base_ref.json`:
  1. Executable Coding-Agent Benchmark (`code_pass1` + granular `code_units`, 40 tasks / ~165 real sandboxed unit tests)
  2. Real Open-Source Python Bug Detection (`swe_code_240`, ALL 240 SWE-Smith repo pairs from pandas/dspy/hydra/etc.)
  3. Procedural Reasoning MCQ (`proc_held`, 200 stratified items = 40/kind)
  4. Compositional Exact-BoW Actor-Intent Binding NLL (`bm_actor2x2`, 120 pairs)
  5. Held-Out Project Gutenberg Fluency NLL (`held_gut`, ALL 100 passages)
  6. Blind Cross-Family Alignment (`Phi-4-mini` + `OLMo 3 7B`, 1,000 passages / 3,192 pts at L8=30% & L14=50%)

Saves modified parameter delta to `candidate_delta.pt` (for exact reproducibility in Living Crystal tests).
Emits ONE JSON line to stdout adhering to `autoresearch-skill/evaluator-contract.md`.
"""
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ARCH = Path("/home/skypanther/systemone/bench/archetypes")
PLATO = ARCH / "fluent" / "plato"
PROBES = ARCH / "exp252" / "probes"
QDIR = ARCH / "exp256_q600"
TARGETS_NPZ = QDIR / "targets_qwen600m_rival30_50.npz"
SWE_CODE_PATH = QDIR / "swe_code_240.jsonl"
BASE_REF_PATH = QDIR / "base_ref.json"
LAST_EVAL_PATH = QDIR / "last_eval.json"
CAND_DELTA_PATH = QDIR / "candidate_delta.pt"

if str(QDIR) not in sys.path:
    sys.path.insert(0, str(QDIR))

import code_agent_bench as cab  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
LAYER_30 = 8    # 8/28 = 0.286
LAYER_50 = 14   # 14/28 = 0.500


def orthogonal_procrustes_map(A, B):
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def eval_procrustes_2fold(H_src, H_tgt, domains, groups):
    """Exact 2-fold group-aware Procrustes CV (matching Exp 228-252)."""
    X = np.asarray(H_src, dtype=np.float64)
    Y = np.asarray(H_tgt, dtype=np.float64)
    uniq_g = sorted(set(str(g) for g in groups))
    g_fold = {g: (i % 2) for i, g in enumerate(uniq_g)}
    folds = np.array([g_fold[str(g)] for g in groups], dtype=np.int32)

    preds = np.zeros_like(Y)
    for f in (0, 1):
        tr = (folds != f)
        te = (folds == f)
        mu_x = X[tr].mean(axis=0, keepdims=True)
        Xc_tr = X[tr] - mu_x
        _, _, vt = np.linalg.svd(Xc_tr, full_matrices=False)
        k = min(Y.shape[1], vt.shape[0])
        V = vt[:k].T
        Zx_tr = Xc_tr @ V
        Zx_te = (X[te] - mu_x) @ V
        if k < Y.shape[1]:
            Zx_tr = np.pad(Zx_tr, ((0, 0), (0, Y.shape[1] - k)))
            Zx_te = np.pad(Zx_te, ((0, 0), (0, Y.shape[1] - k)))

        mu_y = Y[tr].mean(axis=0, keepdims=True)
        Yc_tr = Y[tr] - mu_y
        sx = np.linalg.norm(Zx_tr) + 1e-12
        sy = np.linalg.norm(Yc_tr) + 1e-12
        R = orthogonal_procrustes_map(Zx_tr / sx, Yc_tr / sy)
        preds[te] = ((Zx_te / sx) @ R) * sy + mu_y

    Yc = Y - Y.mean(axis=0, keepdims=True)
    Pc = preds - preds.mean(axis=0, keepdims=True)
    Yn = Yc / (np.linalg.norm(Yc) + 1e-12)
    Pn = Pc / (np.linalg.norm(Pc) + 1e-12)
    r2_frob = float(1.0 - np.sum((Pn - Yn) ** 2))
    alpha = float(np.sum(Pc * Yc)) / (float(np.sum(Pc * Pc)) + 1e-12)
    r2_opt = float(1.0 - np.sum((alpha * Pc - Yc) ** 2) / (np.sum(Yc * Yc) + 1e-12))
    return {"r2_frob": round(r2_frob, 5), "r2_opt": round(r2_opt, 5)}


@torch.no_grad()
def eval_blind_alignment(model, tok, d_t):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ev_ids_f, ev_ids_o = d_t["ev_ids_flat"], d_t["ev_ids_off"]
    ev_pts_f, ev_pts_o = d_t["ev_pts_flat"], d_t["ev_pts_off"]
    n_ev = len(ev_ids_o) - 1

    cap = {}
    def make_hook(li):
        def _hk(_mod, _inp, out):
            cap[li] = out[0] if isinstance(out, tuple) else out
        return _hk

    hooks = [
        model.model.layers[LAYER_30].register_forward_hook(make_hook(LAYER_30)),
        model.model.layers[LAYER_50].register_forward_hook(make_hook(LAYER_50)),
    ]
    h30_list, h50_list = [], []
    bs = 32
    try:
        for s in range(0, n_ev, bs):
            e = min(n_ev, s + bs)
            batch_ids = [ev_ids_f[int(ev_ids_o[i]):int(ev_ids_o[i + 1])].astype(np.int64) for i in range(s, e)]
            batch_pts = [ev_pts_f[int(ev_pts_o[i]):int(ev_pts_o[i + 1])].astype(np.int64) for i in range(s, e)]
            ml = max(len(x) for x in batch_ids)
            inp = torch.full((len(batch_ids), ml), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(batch_ids), ml), dtype=torch.long, device=dev)
            for bi, ids in enumerate(batch_ids):
                inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
                attn[bi, : len(ids)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
    doms, grps = d_t["ev_pt_domains"], d_t["ev_pt_groups"]

    phi4_30 = eval_procrustes_2fold(H30, d_t["phi4_30_ev"], doms, grps)
    olmo_30 = eval_procrustes_2fold(H30, d_t["olmo3_30_ev"], doms, grps)
    phi4_50 = eval_procrustes_2fold(H50, d_t["phi4_50_ev"], doms, grps)
    olmo_50 = eval_procrustes_2fold(H50, d_t["olmo3_50_ev"], doms, grps)
    return {
        "phi4_30": phi4_30["r2_frob"],
        "olmo3_30": olmo_30["r2_frob"],
        "mean_blind_30": round(0.5 * (phi4_30["r2_frob"] + olmo_30["r2_frob"]), 5),
        "phi4_50": phi4_50["r2_frob"],
        "olmo3_50": olmo_50["r2_frob"],
        "mean_blind_50": round(0.5 * (phi4_50["r2_frob"] + olmo_50["r2_frob"]), 5),
    }


@torch.no_grad()
def eval_held_gut_nll(model, tok, gut_rows):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tot_nll, tot_tok = 0.0, 0
    for s in range(0, len(gut_rows), 16):
        batch = [tok.encode(r["text"], add_special_tokens=False) for r in gut_rows[s : s + 16]]
        batch = [ids for ids in batch if len(ids) >= 2]
        ml = max(len(ids) for ids in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        labels = torch.full((len(batch), ml), -100, dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            t = torch.tensor(ids, dtype=torch.long, device=dev)
            inp[bi, : len(ids)] = t
            attn[bi, : len(ids)] = 1
            labels[bi, : len(ids)] = t
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        tot_nll += float(loss.item())
        tot_tok += int((labels[:, 1:] != -100).sum().item())
    return round(tot_nll / max(1, tot_tok), 5)


@torch.no_grad()
def eval_proc_mcq(model, tok, proc_rows):
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
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
    for s in range(0, len(seqs), 32):
        batch = seqs[s : s + 32]
        ml = max(len(x) for x in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[bi, : len(ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        for bi in range(len(batch)):
            _, _, st, en = meta[s + bi]
            tgt = inp[bi, st:en]
            scores[s + bi] = float(lp[bi, st - 1 : en - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item())

    by_row = {}
    for idx, (ri, is_ans, _, _) in enumerate(meta):
        by_row.setdefault(ri, []).append((scores[idx], is_ans))
    hits, by_kind = 0, {}
    for ri, r in enumerate(proc_rows):
        ok = max(by_row[ri], key=lambda x: x[0])[1]
        hits += int(ok)
        by_kind.setdefault(r["kind"], [0, 0])
        by_kind[r["kind"]][0] += int(ok)
        by_kind[r["kind"]][1] += 1
    return {
        "acc": round(hits / max(1, len(proc_rows)), 5),
        "by_kind": {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(by_kind.items())},
    }


def _disjoint_changed_masks(ids_p, ids_n):
    sm = difflib.SequenceMatcher(None, ids_p[1:], ids_n[1:])
    mp = np.zeros(max(1, len(ids_p) - 1), dtype=bool)
    mn = np.zeros(max(1, len(ids_n) - 1), dtype=bool)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            mp[i1:i2] = True
            mn[j1:j2] = True
    if not mp.any(): mp[:] = True
    if not mn.any(): mn[:] = True
    return mp, mn


@torch.no_grad()
def eval_pair_nll(model, tok, pairs, max_len=512):
    """Exact changed-token NLL comparison for paired (pos, neg) strings."""
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs, masks = [], []
    for p_txt, n_txt in pairs:
        ip = tok.encode(p_txt.strip(), add_special_tokens=False)[:max_len]
        in_ = tok.encode(n_txt.strip(), add_special_tokens=False)[:max_len]
        if len(ip) < 2: ip = ip + [pad_id]
        if len(in_) < 2: in_ = in_ + [pad_id]
        mp, mn = _disjoint_changed_masks(ip, in_)
        seqs.extend([ip, in_])
        masks.extend([mp, mn])

    nlls = np.zeros(len(seqs), dtype=np.float64)
    for s in range(0, len(seqs), 24):
        batch = seqs[s : s + 24]
        ml = max(len(x) for x in batch)
        inp = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((len(batch), ml), dtype=torch.long, device=dev)
        for bi, ids in enumerate(batch):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[bi, : len(ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
        lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        for bi, ids in enumerate(batch):
            L = len(ids) - 1
            m = masks[s + bi][:L]
            tgt = inp[bi, 1 : L + 1]
            tok_nll = -lp[bi, :L].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            nlls[s + bi] = float(tok_nll[m].mean())

    hits = sum(1 for i in range(0, len(nlls), 2) if nlls[i] < nlls[i + 1])
    margin = float(np.mean([nlls[i + 1] - nlls[i] for i in range(0, len(nlls), 2)]))
    return round(hits / max(1, len(pairs)), 5), round(margin, 5)


@torch.no_grad()
def eval_code_bench(model, tok, tasks, max_new_tokens=320, batch_size=20):
    dev = next(model.parameters()).device
    orig_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    sys_msg = (
        "You are an expert Python coding agent. Solve the problem cleanly and accurately. "
        "Output only the complete Python implementation inside a single ```python``` block."
    )
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": t["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for t in tasks
    ]

    gens = []
    try:
        for s in range(0, len(prompts), batch_size):
            batch_p = prompts[s : s + batch_size]
            enc = tok(batch_p, return_tensors="pt", padding=True).to(dev)
            in_len = enc["input_ids"].shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            for bi in range(len(batch_p)):
                gens.append(tok.decode(out[bi, in_len:], skip_special_tokens=True))
    finally:
        tok.padding_side = orig_side

    return cab.grade_generations(gens, tasks=tasks, max_workers=10, timeout_s=2.0)


def eval_model_full(model, tok, d_t):
    t0 = time.time()
    align = eval_blind_alignment(model, tok, d_t)

    ev_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl") if l.strip()]
    gut_nll = eval_held_gut_nll(model, tok, [r for r in ev_rows if r["domain"] == "held_gut"][:100])

    proc_all = [json.loads(l) for l in open(PROBES / "proc_held.jsonl") if l.strip()]
    by_k = {}
    for r in proc_all:
        by_k.setdefault(r["kind"], []).append(r)
    proc_200 = [r for k in sorted(by_k.keys()) for r in by_k[k][:40]]
    proc_res = eval_proc_mcq(model, tok, proc_200)

    actor_all = json.loads((PROBES / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    actor_acc, _ = eval_pair_nll(model, tok, [(r["pos"], r["neg"]) for r in actor_all[::2]], max_len=256)

    swe_rows = [json.loads(l) for l in open(SWE_CODE_PATH) if l.strip()]
    swe_acc, swe_margin = eval_pair_nll(model, tok, [(r["before"], r["after"]) for r in swe_rows], max_len=512)

    code_res = eval_code_bench(model, tok, cab.CODE_TASKS, max_new_tokens=320, batch_size=20)

    return {
        "gut_nll": gut_nll,
        "code_pass1": code_res["pass_at_1"],
        "code_units": code_res["unit_rate"],
        "code_hits": f"{code_res['hits']}/{code_res['total']} (u:{code_res['unit_hits']}/{code_res['unit_total']})",
        "code_by_cat": code_res["by_cat_counts"],
        "swe_acc": swe_acc,
        "swe_margin": swe_margin,
        "proc_acc": proc_res["acc"],
        "proc_by_kind": proc_res["by_kind"],
        "actor_acc": actor_acc,
        **align,
        "eval_sec": round(time.time() - t0, 1),
        "code_per_task": code_res["per_task"],
    }


def main():
    t0 = time.time()
    d_t = np.load(TARGETS_NPZ, allow_pickle=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    if not BASE_REF_PATH.exists():
        b = eval_model_full(model, tok, d_t)
        BASE_REF_PATH.write_text(json.dumps(b, indent=1))
        if "--init-base-only" in sys.argv:
            summary = {k: v for k, v in b.items() if k != "code_per_task"}
            print(json.dumps({"base_initialized": True, **summary}), flush=True)
            return

    base_ref = json.loads(BASE_REF_PATH.read_text())
    orig_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    import transplant_recipe  # noqa: E402
    train_meta = transplant_recipe.run_transplant(model, tok, d_t, seed=256)
    model.eval()

    delta = {}
    cur_sd = model.state_dict()
    for k, orig_v in orig_sd_cpu.items():
        cur_v = cur_sd[k].detach().cpu()
        if not torch.equal(cur_v, orig_v):
            delta[k] = cur_v
    torch.save({"recipe": train_meta.get("recipe", ""), "state_dict": delta}, CAND_DELTA_PATH)
    del orig_sd_cpu

    ev = eval_model_full(model, tok, d_t)

    d_nll = round(ev["gut_nll"] - base_ref["gut_nll"], 5)
    d_code_pass1 = round(ev["code_pass1"] - base_ref["code_pass1"], 5)
    d_code_units = round(ev["code_units"] - base_ref["code_units"], 5)
    d_swe = round(ev["swe_acc"] - base_ref["swe_acc"], 5)
    d_proc = round(ev["proc_acc"] - base_ref["proc_acc"], 5)
    d_actor = round(ev["actor_acc"] - base_ref["actor_acc"], 5)
    d_align_30 = round(ev["mean_blind_30"] - base_ref["mean_blind_30"], 5)
    d_align_50 = round(ev["mean_blind_50"] - base_ref["mean_blind_50"], 5)
    d_align_peak = round(max(d_align_30, d_align_50), 5)

    # Pure downstream capability & quality lift (0.0000 = unmodified Base Qwen3-0.6B!)
    score = round(
        (-100.0 * d_nll)
        + (80.0 * d_code_pass1)
        + (80.0 * d_code_units)
        + (80.0 * d_swe)
        + (80.0 * d_proc)
        + (50.0 * d_actor),
        4,
    )

    guard_ok = bool(d_align_peak >= 0.050)
    passed = bool(
        guard_ok
        and d_nll <= 0.0
        and d_code_units >= 0.0
        and d_swe >= 0.0
        and d_proc >= 0.0
        and score >= 3.00
    )

    out = {
        "pass": passed,
        "score": score,
        "guard_ok": guard_ok,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_changed_tensors": len(delta),
        "deltas": {
            "d_nll": d_nll,
            "d_code_pass1": d_code_pass1,
            "d_code_units": d_code_units,
            "d_swe": d_swe,
            "d_proc": d_proc,
            "d_actor": d_actor,
            "d_align_30": d_align_30,
            "d_align_50": d_align_50,
            "d_align_peak": d_align_peak,
        },
        "eval": {k: v for k, v in ev.items() if k != "code_per_task"},
        "base_ref": {k: v for k, v in base_ref.items() if k != "code_per_task"},
        "train_meta": train_meta,
        "code_per_task": ev["code_per_task"],
    }
    LAST_EVAL_PATH.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "pass": passed,
                "score": score,
                "guard_ok": guard_ok,
                "elapsed_sec": out["elapsed_sec"],
                "deltas": out["deltas"],
                "eval_summary": {
                    k: ev[k]
                    for k in (
                        "gut_nll", "code_pass1", "code_units", "code_hits",
                        "swe_acc", "proc_acc", "actor_acc", "mean_blind_30", "mean_blind_50",
                    )
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
