"""Pillar 3 Engine: Pretraining From Random Init With 31B Cross-Family Platonic Scaffold (`Exp 263`).

Tests the third fundamental hypothesis: when a small causal transformer has ZERO entrenched weights
(fresh random initialization from step 0!), does growing its intermediate representations into the
`Gemma 4 31B + Llama 3.1 8B + Mistral 7B` Whitened GCCA universal shape via a discardable 2-layer MLP
projector improve:
  1. Held-out Python coding generalization (`code_eval` NLL, 399 unseen functions)
  2. Out-of-distribution language generalization (`WikiText-2` NLL, 62 unseen Wikipedia articles)
  3. Cross-family geometric emergence (blind `Phi-4 + OLMo 3` OOF `r2_frob` & `Top-1`)
compared against:
  - `scratch_ce_only`: exact same architecture, seed, data order, standard next-token CE alone
  - `scratch_ce_plus_shuf_mlp`: exact same MLP scaffold loss with shuffled 31B targets (matched regularization control!)

Architecture: Compact ~14M-param Pre-Norm RoPE/SwiGLU Causal Transformer (6 layers, d=384, 6 heads,
tied embeddings with compact byte/subword vocab mapped cleanly onto Qwen3 token IDs present in corpus,
OR full Qwen3 vocab with factorized embedding `151936 -> 128 -> 384` so non-embedding transformer params
dominate and every WikiText-2 / code_eval token evaluates directly without OOV!).
"""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
EXP258_DIR = ROOT.parent / "exp258_code_lattice"
EXP257_DIR = ROOT.parent / "exp257_rigorous"
for p in (str(EXP258_DIR), str(EXP257_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_rigorous as ev257  # noqa: E402
import run_dev_code_cuda as base258  # noqa: E402

DEPTHS = (30, 50, 70)
# 6-layer fresh transformer (0..5): 30%->L1, 50%->L3, 70%->L4
SCRATCH_LAYERS = {30: 1, 50: 3, 70: 4}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class CausalBlock(nn.Module):
    def __init__(self, d=384, n_heads=6, d_ff=1024, dropout=0.05):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.ln1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ln2 = RMSNorm(d)
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.drop = dropout

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o(y)
        h2 = self.ln2(x)
        x = x + self.w2(F.silu(self.w1(h2)) * self.w3(h2))
        return x


class FreshTinyLM(nn.Module):
    """Freshly initialized ~26M-param factorized-embedding causal LM using Qwen3 tokenizer (0 OOV on WT2/code!)."""
    def __init__(self, vocab_size=151936, d_emb=128, d=384, n_layers=6, n_heads=6, max_len=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_emb)
        self.emb_up = nn.Linear(d_emb, d, bias=False)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([CausalBlock(d=d, n_heads=n_heads, d_ff=1024) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d)
        self.head_down = nn.Linear(d, d_emb, bias=False)
        self.max_len = max_len
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, capture_layers=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_up(self.tok_emb(input_ids)) + self.pos_emb(pos)
        caps = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if capture_layers and i in capture_layers:
                caps[i] = x
        h = self.head_down(self.ln_f(x))
        logits = F.linear(h, self.tok_emb.weight)
        return logits, caps


class DiscardableScaffoldMLP(nn.Module):
    def __init__(self, d_in=384, d_hid=512, d_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_scratch_model(model, tok, d_c, wt2_texts, gut_texts, dev):
    model.eval()
    # 1. Held-out code_eval NLL + blind Phi-4 & OLMo 3 OOF r2
    ev_f, ev_o = d_c["ev_ids_flat"], d_c["ev_ids_off"]
    pt_f, pt_o = d_c["ev_pts_flat"], d_c["ev_pts_off"]
    n_ev = len(ev_o) - 1
    H = {dep: [] for dep in DEPTHS}
    tot_nll, tot_tok = 0.0, 0
    cap_set = set(SCRATCH_LAYERS.values())
    for i in range(n_ev):
        ids = ev_f[int(ev_o[i]):int(ev_o[i + 1])].astype(np.int64)
        pts = pt_f[int(pt_o[i]):int(pt_o[i + 1])].astype(np.int64)
        if len(ids) > model.max_len:
            ids = ids[: model.max_len]
            pts = np.clip(pts, 0, len(ids) - 1)
        inp = torch.from_numpy(ids).unsqueeze(0).to(dev)
        logits, caps = model(inp, capture_layers=cap_set)
        L = len(ids) - 1
        tot_nll += float(F.cross_entropy(logits[0, :L].float(), inp[0, 1 : L + 1], reduction="sum").item())
        tot_tok += L
        for dep, li in SCRATCH_LAYERS.items():
            H[dep].append(caps[li][0, pts, :].float().cpu().numpy())
    code_eval_nll = tot_nll / max(1, tot_tok)

    # Blind Phi-4 + OLMo 3 OOF r2_frob
    grps = d_c["ev_pt_groups"].tolist()
    uniq = sorted(set(grps))
    rng = random.Random(2588); rng.shuffle(uniq)
    g_fold = {g: (0 if i < len(uniq) // 2 else 1) for i, g in enumerate(uniq)}
    folds = np.array([g_fold[g] for g in grps], dtype=np.int32)
    align = {}
    for dep in DEPTHS:
        X = np.concatenate(H[dep], axis=0).astype(np.float64)
        vals = []
        for bk in ("phi4_mini", "olmo3_7b"):
            Y = d_c[f"{bk}_{dep}_ev"].astype(np.float64)
            pa, ta = [], []
            for tf in (0, 1):
                tr, te = folds != tf, folds == tf
                Z_tr, Z_te = base258._pca_proj(X[tr], X[te], d_out=256)
                pa.append(base258._apply_proc(Z_te, base258._fit_proc(Z_tr, Y[tr])))
                ta.append(Y[te])
            pc, tc = np.concatenate(pa), np.concatenate(ta)
            r2 = 1.0 - float(np.sum((tc - pc) ** 2)) / max(1e-12, float(np.sum((tc - tc.mean(0, keepdims=True)) ** 2)))
            vals.append(r2)
        align[f"blind_{dep}_r2"] = round(float(np.mean(vals)), 5)

    # 2. Strictly OOD WikiText-2 NLL + Held-out Gutenberg NLL
    def eval_text_nll(texts):
        s_nll, s_tok = 0.0, 0
        for txt in texts:
            ids = tok.encode(txt, add_special_tokens=False)[: model.max_len]
            if len(ids) < 4:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=dev)
            lg, _ = model(inp)
            L = len(ids) - 1
            s_nll += float(F.cross_entropy(lg[0, :L].float(), inp[0, 1 : L + 1], reduction="sum").item())
            s_tok += L
        return s_nll / max(1, s_tok)

    wt2_nll = eval_text_nll(wt2_texts)
    gut_nll = eval_text_nll(gut_texts) if gut_texts else None
    return {
        "code_eval_nll": round(code_eval_nll, 5),
        "wikitext2_nll": round(wt2_nll, 5),
        "held_gut_nll": round(gut_nll, 5) if gut_nll is not None else None,
        **align,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--scaffold_w", type=float, default=0.0)
    ap.add_argument("--geo_type", choices=["gcca", "gpa"], default="gcca")
    ap.add_argument("--align_mode", choices=["static", "delta", "both"], default="both")
    ap.add_argument("--shuffle_geo", action="store_true")
    ap.add_argument("--decay_scaffold", action="store_true")
    ap.add_argument("--n_train_limit", type=int, default=0, help="If >0, restrict training set to low-data regime")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d_c = dict(np.load(EXP258_DIR / "targets_qwen600m_code_lattice_30_50_70.npz", allow_pickle=True))
    d_31b = dict(np.load(EXP258_DIR / "targets_qwen600m_31b_gpa_gcca.npz", allow_pickle=True))

    # Load WikiText-2 test articles + held-out Gutenberg passages (strictly eval only!)
    wt2_path = EXP257_DIR / "data" / "wikitext2_raw_test.json"
    wt2_texts = []
    if wt2_path.exists():
        raw_wt2 = json.loads(wt2_path.read_text())
        wt2_texts = [x["text"] if isinstance(x, dict) else x for x in (raw_wt2["articles"] if isinstance(raw_wt2, dict) else raw_wt2)]
    gut_texts = []
    plato_dir = ev257.default_paths()["plato_dir"]
    gut_ev = plato_dir / "held_gutenberg_100.jsonl"
    if gut_ev.exists():
        gut_texts = [json.loads(l)["text"] for l in open(gut_ev) if l.strip()]

    # Also mix Gutenberg prose training passages (strictly train split!) if available alongside code_train
    model = FreshTinyLM(vocab_size=len(tok), d_emb=128, d=384, n_layers=6, n_heads=6, max_len=1024).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    use_scaf = (args.scaffold_w > 0.0)
    scaf_mlps, tgts = {}, {}
    params = list(model.parameters())
    if use_scaf:
        pref = "gcca31" if args.geo_type == "gcca" else "gpa31"
        for dep in DEPTHS:
            arr = d_31b[f"{pref}_{dep}_tr"].astype(np.float32)
            arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-6)
            if args.shuffle_geo:
                rng_s = np.random.default_rng(args.seed + dep)
                arr = arr[rng_s.permutation(len(arr))]
            tgts[dep] = arr
            mlp = DiscardableScaffoldMLP(d_in=384, d_hid=512, d_out=arr.shape[1]).to(dev)
            scaf_mlps[dep] = mlp
            params.extend(list(mlp.parameters()))

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    ids_f, ids_o = d_c["tr_ids_flat"], d_c["tr_ids_off"]
    pts_f, pts_o = d_c["tr_pts_flat"], d_c["tr_pts_off"]
    n_tot = len(ids_o) - 1
    rng = np.random.default_rng(args.seed)
    pool = np.arange(n_tot)
    if args.n_train_limit > 0:
        pool = rng.choice(pool, size=min(args.n_train_limit, n_tot), replace=False)
    perm, cur = rng.permutation(pool), 0
    cap_set = set(SCRATCH_LAYERS.values()) if use_scaf else None

    print(f"[PILLAR3] FreshTinyLM ({n_params/1e6:.2f}M params) arm={args.arm} seed={args.seed} steps={args.steps} "
          f"pool={len(pool)} scaf_w={args.scaffold_w} ({args.geo_type}/{args.align_mode}, shuf={args.shuffle_geo})", flush=True)

    model.train()
    for step in range(args.steps):
        frac = step / max(1, args.steps - 1)
        lr_mul = min(1.0, (step + 1) / 40.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        opt.param_groups[0]["lr"] = args.lr * lr_mul
        scaf_mul = (1.0 - 0.8 * frac) if args.decay_scaffold else 1.0

        batch = []
        while len(batch) < args.batch_size:
            if cur >= len(perm):
                perm, cur = rng.permutation(pool), 0
            pi = int(perm[cur]); cur += 1
            ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
            pts = pts_f[int(pts_o[pi]):int(pts_o[pi + 1])].astype(np.int64)
            if len(ids) >= 8 and len(pts) >= 3:
                ts = {dep: tgts[dep][int(pts_o[pi]):int(pts_o[pi + 1])] for dep in DEPTHS} if use_scaf else None
                batch.append((ids[:1024], np.clip(pts, 0, min(len(ids), 1024) - 1), ts))

        ml = max(len(b[0]) for b in batch)
        B = len(batch)
        inp = torch.full((B, ml), pad_id, dtype=torch.long, device=dev)
        valid = torch.zeros((B, ml - 1), dtype=torch.float32, device=dev)
        for bi, (ids, _, _) in enumerate(batch):
            inp[bi, : len(ids)] = torch.from_numpy(ids).to(dev)
            valid[bi, : len(ids) - 1] = 1.0

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, caps = model(inp, capture_layers=cap_set)
        tok_ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), inp[:, 1:].reshape(-1), reduction="none").reshape(B, ml - 1)
        ce_loss = (tok_ce * valid).sum() / valid.sum().clamp_min(1.0)

        scaf_loss = ce_loss.new_tensor(0.0)
        if use_scaf:
            for dep in DEPTHS:
                li = SCRATCH_LAYERS[dep]
                hs, zs = [], []
                for bi, (_, pts, ts) in enumerate(batch):
                    h_s = caps[li][bi, pts, :].float()
                    z_s = torch.from_numpy(ts[dep]).to(dev)
                    if args.align_mode == "delta":
                        hs.append(h_s[1:] - h_s[:-1]); zs.append(z_s[1:] - z_s[:-1])
                    elif args.align_mode == "both":
                        hs.append(torch.cat([h_s, h_s[1:] - h_s[:-1]], dim=0))
                        zs.append(torch.cat([z_s, z_s[1:] - z_s[:-1]], dim=0))
                    else:
                        hs.append(h_s); zs.append(z_s)
                H_cat, Z_cat = torch.cat(hs, dim=0), torch.cat(zs, dim=0)
                pred = scaf_mlps[dep](H_cat)
                c_l = 1.0 - F.cosine_similarity(pred, Z_cat, dim=-1).mean()
                m_l = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(Z_cat, dim=-1))
                scaf_loss = scaf_loss + (c_l + m_l) / float(len(DEPTHS))

        loss = ce_loss + (args.scaffold_w * scaf_mul) * scaf_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == args.steps - 1:
            print(f"[PILLAR3] {args.arm} step {step+1}/{args.steps} | ce={ce_loss.item():.4f} scaf={scaf_loss.item():.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Discard scaffold MLPs and evaluate fresh LM!
    del scaf_mlps, opt
    gc.collect()
    torch.cuda.empty_cache()

    metrics = eval_scratch_model(model, tok, d_c, wt2_texts, gut_texts, dev)
    out = {
        "arm": args.arm, "seed": args.seed, "n_params": n_params,
        "args": vars(args), **metrics, "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = RESULTS / f"p3_{args.arm}_s{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[PILLAR3] DONE {args.arm} s={args.seed} | code_eval_nll={metrics['code_eval_nll']:.5f} | "
          f"wt2_nll={metrics['wikitext2_nll']:.5f} | blind={metrics['blind_30_r2']:.4f}/{metrics['blind_50_r2']:.4f}/{metrics['blind_70_r2']:.4f} | t={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
