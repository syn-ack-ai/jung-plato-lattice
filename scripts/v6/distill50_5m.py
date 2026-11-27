"""Experiment 246: 50%-Depth Relational Platonic Consensus (C4_50) Extraction & 5M Student Distillation.

Tests whether distilling the newly discovered 50%-depth multi-family Platonic consensus (C4_50) —
alone at layer 3 (50% depth) or paired with 30%-depth C4_30 at layer 2 (33% depth) in a dual-depth
sandwich — improves a 5M transformer (identical 4,722,304 inference non-embed params, training-only
projection heads dropped at inference) on:
  1. Held-out unseen-family (Phi-4-mini & OLMo 3 7B) 2-fold Procrustes alignment at BOTH 30% & 50% depth
  2. Core LM & reasoning benchmarks: held books bpb, chat bpb, proc_held MCQ gain, COPA, BLiMP
  3. Internal relational/binding representations: HOUSE 2-fold CV, bm120, bm_rel242, bm_neut245!

Four strictly matched arms (identical main stream, identical aux passages, identical seeds, annealed
plato_w 0.25 -> 0.0 over [65%, 90%] so final 10% is pure-NTP polish):
  D0_ctrl      : plato_w = 0.0 (aux NTP 0.10 only)
  D1_c4_30_L2  : C4_30 @ layer 2 (33% depth, plato_w = 0.25)
  D2_c4_50_L3  : C4_50 @ layer 3 (50% depth, plato_w = 0.25)
  D3_dual_30_50: C4_30 @ layer 2 (0.125) + C4_50 @ layer 3 (0.125) [matched total plato_w = 0.25!]

Usage:
  python distill50_5m.py extract50 [MODEL_KEY...]
  python distill50_5m.py consensus50
  python distill50_5m.py train NAME arm=D0_ctrl|D1_c4_30_L2|D2_c4_50_L3|D3_dual_30_50 seed=1 [tokens=25e6]
  python distill50_5m.py score NAME...
"""
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH_DIR / "exp246_distill50"
EXP_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = EXP_DIR / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
D50_CACHE = PLATO_CACHE / "depth50_distill"
D50_CACHE.mkdir(parents=True, exist_ok=True)

for p in (str(REPO_ARCH), str(REPO_FLUENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402

MLX_CONSENSUS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
BLIND_KEYS = ("phi4_mini", "olmo3_7b")
ALL_6 = MLX_CONSENSUS + BLIND_KEYS


# ---------------------------------------------------------------- Step 1: Extract 50%-depth checkpoint states (MLX)

def extract50(keys=ALL_6):
    import mlx.core as mx
    from mlx_lm import load

    train_rows = [json.loads(l) for l in open(PLATO_CACHE / "passages.jsonl")]
    eval_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    # Neutral training passages ONLY (gut + proc + chat -> 2,394 passages; skip archaic myth!)
    tr_neut_idx = [i for i, r in enumerate(train_rows) if r["domain"] in ("gut", "proc", "chat")]

    for key in keys:
        out_npz = D50_CACHE / f"raw50_{key}.npz"
        if out_npz.exists():
            print(f"D50 EXTRACT {key}: exists, skipping", flush=True)
            continue
        spec = PD.MODEL_SPECS[key]
        mx.set_cache_limit(2 << 30)
        t0 = time.time()
        print(f"D50 EXTRACT {key}: loading {spec['path']} ...", flush=True)
        model, tok = load(spec["path"])
        layers = PD._find_layers(model)
        nl_cnt = len(layers)
        tgt_idx = max(1, min(nl_cnt - 2, int(round(0.50 * (nl_cnt - 1)))))
        tgt_blk = layers[tgt_idx]
        print(f"  {key}: L={nl_cnt}, hooking 50%-depth layer {tgt_idx} ({tgt_idx/(nl_cnt-1):.2f})", flush=True)

        rec_buf = []
        patched = {}
        for blk in layers:
            cls = type(blk)
            if cls not in patched:
                orig = cls.__call__
                patched[cls] = orig
                def make_hook(orig_fn):
                    def hooked(self, *a, **k):
                        out = orig_fn(self, *a, **k)
                        if self is tgt_blk:
                            h = out[0] if isinstance(out, tuple) else out
                            rec_buf.append(h[0])
                        return out
                    return hooked
                cls.__call__ = make_hook(orig)

        enc_fn = lambda s: PD._text_ids_mlx(tok, s)

        def run_rows(row_list, label):
            pts_all, cnts, oks = [], [], []
            for ri, r in enumerate(row_list):
                ids, pts = PD.boundary_token_indices(enc_fn, r["segs"])
                if not (2 <= len(ids) <= 512 and len(pts) == len(r["segs"])):
                    oks.append(False); cnts.append(0)
                    continue
                rec_buf.clear()
                _ = model(mx.array([ids]))
                assert len(rec_buf) == 1
                h = rec_buf[0].astype(mx.float32)
                sel = h[mx.array(pts)]
                mx.eval(sel)
                rec_buf.clear()
                pts_all.append(np.array(sel, dtype=np.float32))
                oks.append(True); cnts.append(len(pts))
                if (ri + 1) % 600 == 0:
                    print(f"    {key} {label}: {ri+1}/{len(row_list)} ({time.time()-t0:.0f}s)", flush=True)
            mat = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 1), dtype=np.float32)
            return mat, np.array(cnts, dtype=np.int32), np.array(oks, dtype=bool)

        try:
            tr_pts, tr_cnt, tr_ok = run_rows([train_rows[i] for i in tr_neut_idx], "train_neut")
            ev_pts, ev_cnt, ev_ok = run_rows(eval_rows, "eval")
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            rec_buf.clear()
            del layers, tgt_blk, model, tok
            gc.collect()
            mx.clear_cache()

        pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
        PD._atomic_savez(
            out_npz,
            tr_neut_idx=np.array(tr_neut_idx, dtype=np.int32),
            tr_pts=pack(tr_pts), tr_cnt=tr_cnt, tr_ok=tr_ok,
            ev_pts=pack(ev_pts), ev_cnt=ev_cnt, ev_ok=ev_ok,
            layer_idx=np.int32(tgt_idx), num_layers=np.int32(nl_cnt),
        )
        print(f"D50_EXTRACT_DONE {key}: tr_pts={tr_pts.shape}, ev_pts={ev_pts.shape} in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Step 2: Build aligned 30% + 50% C4 consensus packages

def consensus50(d_plato=256):
    """Build exact checkpoint-aligned C4_30 AND C4_50 targets on the identical neutral passages
    (so D1_c4_30_L2, D2_c4_50_L3, and D3_dual_30_50 see 100% IDENTICAL passages & checkpoints!)."""
    t0 = time.time()
    tk = PD.load_bpe8k()
    bpe_enc = lambda s: tk.encode(s).ids
    train_rows = [json.loads(l) for l in open(PLATO_CACHE / "passages.jsonl")]
    eval_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    tr_neut_idx = [i for i, r in enumerate(train_rows) if r["domain"] in ("gut", "proc", "chat")]

    raw30 = {k: dict(np.load(PLATO_CACHE / f"raw_{k}.npz")) for k in ALL_6}
    raw50 = {k: dict(np.load(D50_CACHE / f"raw50_{k}.npz")) for k in ALL_6}

    # Build strict intersection mask across ALL 6 models at BOTH 30% and 50% depth + bpe8k!
    # Map full train_rows offset for raw30
    off30_tr = {k: np.concatenate([[0], np.cumsum(raw30[k]["tr_cnt"])]) for k in ALL_6}
    off50_tr = {k: np.concatenate([[0], np.cumsum(raw50[k]["tr_cnt"])]) for k in ALL_6}
    off30_ev = {k: np.concatenate([[0], np.cumsum(raw30[k]["ev_cnt"])]) for k in ALL_6}
    off50_ev = {k: np.concatenate([[0], np.cumsum(raw50[k]["ev_cnt"])]) for k in ALL_6}

    tr_bpe_ids, tr_bpe_pts, tr_doms = [], [], []
    tr_sl30, tr_sl50 = {k: [] for k in ALL_6}, {k: [] for k in ALL_6}
    for local_i, orig_i in enumerate(tr_neut_idx):
        r = train_rows[orig_i]
        n_seg = len(r["segs"])
        b_ids, b_pts = PD.boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raw30[k]["tr_ok"][orig_i]) and int(raw30[k]["tr_cnt"][orig_i]) == n_seg for k in ALL_6)
            and all(bool(raw50[k]["tr_ok"][local_i]) and int(raw50[k]["tr_cnt"][local_i]) == n_seg for k in ALL_6)
        )
        if ok:
            tr_bpe_ids.append(b_ids); tr_bpe_pts.append(b_pts); tr_doms.append(r["domain"])
            for k in ALL_6:
                tr_sl30[k].append(raw30[k]["tr_pts"][off30_tr[k][orig_i]:off30_tr[k][orig_i + 1]])
                tr_sl50[k].append(raw50[k]["tr_pts"][off50_tr[k][local_i]:off50_tr[k][local_i + 1]])

    ev_bpe_ids, ev_bpe_pts, ev_pt_groups, ev_pt_doms, ev_pt_ret = [], [], [], [], []
    ev_sl30, ev_sl50 = {k: [] for k in ALL_6}, {k: [] for k in ALL_6}
    for r_i, r in enumerate(eval_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = PD.boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raw30[k]["ev_ok"][r_i]) and int(raw30[k]["ev_cnt"][r_i]) == n_seg for k in ALL_6)
            and all(bool(raw50[k]["ev_ok"][r_i]) and int(raw50[k]["ev_cnt"][r_i]) == n_seg for k in ALL_6)
        )
        if ok:
            ev_bpe_ids.append(b_ids); ev_bpe_pts.append(b_pts)
            is_ret = (r.get("role", "") != "foil")
            for _ in range(n_seg):
                ev_pt_groups.append(r.get("group_id", str(r_i)))
                ev_pt_doms.append(r.get("domain", ""))
                ev_pt_ret.append(is_ret)
            for k in ALL_6:
                ev_sl30[k].append(raw30[k]["ev_pts"][off30_ev[k][r_i]:off30_ev[k][r_i + 1]])
                ev_sl50[k].append(raw50[k]["ev_pts"][off50_ev[k][r_i]:off50_ev[k][r_i + 1]])

    print(f"ALIGNED DUAL-DEPTH PASSAGES: train={len(tr_bpe_ids)}/{len(tr_neut_idx)} eval={len(ev_bpe_ids)}/{len(eval_rows)}", flush=True)

    # Fit 256D PCA strictly on train + 10-iter GPA consensus C4 for BOTH 30% and 50%!
    Y30_tr, Y30_ev, Y50_tr, Y50_ev = {}, {}, {}, {}
    for k in ALL_6:
        Y30_tr[k], Y30_ev[k], _ = PD.pca_project(np.concatenate(tr_sl30[k], 0).astype(np.float64), np.concatenate(ev_sl30[k], 0).astype(np.float64), d_plato)
        Y50_tr[k], Y50_ev[k], _ = PD.pca_project(np.concatenate(tr_sl50[k], 0).astype(np.float64), np.concatenate(ev_sl50[k], 0).astype(np.float64), d_plato)

    C30_tr, C30_ev, _, _, _ = PD.generalized_procrustes_consensus([Y30_tr[k] for k in MLX_CONSENSUS], [Y30_ev[k] for k in MLX_CONSENSUS], n_iters=10)
    C50_tr, C50_ev, _, _, _ = PD.generalized_procrustes_consensus([Y50_tr[k] for k in MLX_CONSENSUS], [Y50_ev[k] for k in MLX_CONSENSUS], n_iters=10)

    # Diagnostic: Blind teacher -> unseen family (Phi-4, OLMo 3) transfer at 30% vs 50%!
    diag = {}
    for depth_lbl, C_ev, Y_ev in (("d30", C30_ev, Y30_ev), ("d50", C50_ev, Y50_ev)):
        diag[depth_lbl] = {}
        for bk in BLIND_KEYS:
            r_c4 = PD.eval_procrustes_transfer(C_ev, Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_doms, ret_mask=ev_pt_ret)
            r_gem = PD.eval_procrustes_transfer(Y_ev["gemma4_e4b"], Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_doms, ret_mask=ev_pt_ret)
            r_qwn = PD.eval_procrustes_transfer(Y_ev["qwen35_9b"], Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_doms, ret_mask=ev_pt_ret)
            diag[depth_lbl][bk] = {"C4": r_c4, "gemma4": r_gem, "qwen35": r_qwn}
            print(f"  [{depth_lbl}] -> {bk}: C4 cos={r_c4['cos']} r2o={r_c4['r2_opt']} top1={r_c4['top1']} | gem cos={r_gem['cos']} | qwn cos={r_qwn['cos']}", flush=True)

    def pack_seq(seq_list):
        off = np.zeros(len(seq_list) + 1, dtype=np.int64)
        for i, s in enumerate(seq_list):
            off[i + 1] = off[i] + len(s)
        flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in seq_list], axis=0)
        return flat, off

    tr_ids_f, tr_ids_o = pack_seq(tr_bpe_ids)
    tr_pts_f, tr_pts_o = pack_seq(tr_bpe_pts)
    ev_ids_f, ev_ids_o = pack_seq(ev_bpe_ids)
    ev_pts_f, ev_pts_o = pack_seq(ev_bpe_pts)

    PD._atomic_savez(
        D50_CACHE / "targets_dual30_50_train.npz",
        ids_flat=tr_ids_f, ids_off=tr_ids_o, pts_flat=tr_pts_f, pts_off=tr_pts_o,
        domains=np.array(tr_doms),
        c4_30=C30_tr.astype(np.float32),
        c4_50=C50_tr.astype(np.float32),
    )
    PD._atomic_savez(
        D50_CACHE / "targets_dual30_50_eval.npz",
        ids_flat=ev_ids_f, ids_off=ev_ids_o, pts_flat=ev_pts_f, pts_off=ev_pts_o,
        pt_groups=np.array(ev_pt_groups), pt_domains=np.array(ev_pt_doms), pt_ret=np.array(ev_pt_ret, dtype=bool),
        c4_30=C30_ev.astype(np.float32), c4_50=C50_ev.astype(np.float32),
        phi4_30=Y30_ev["phi4_mini"].astype(np.float32), phi4_50=Y50_ev["phi4_mini"].astype(np.float32),
        olmo3_30=Y30_ev["olmo3_7b"].astype(np.float32), olmo3_50=Y50_ev["olmo3_7b"].astype(np.float32),
    )
    (EXP_DIR / "consensus50_diag.json").write_text(json.dumps(diag, indent=2))
    print(f"CONSENSUS50_DONE in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------- Step 3: Paired 5M Dual-Depth Student Training & Eval (PyTorch MPS)

ARM_SPEC = {
    "D0_ctrl":       dict(w30=0.0,   w50=0.0),
    "D1_c4_30_L2":   dict(w30=0.25,  w50=0.0),
    "D2_c4_50_L3":   dict(w30=0.0,   w50=0.25),
    "D3_dual_30_50": dict(w30=0.125, w50=0.125),
}


def train_5m(name, arm, seed=1, tokens=25_000_000):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from fluent import exp228, exp229

    # Point data/mid/comp paths to local populated bench tree (keeping repo clone clean!)
    SYS_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
    exp229.GUT = SYS_ARCH / "smallworld"
    exp229.MID = SYS_ARCH / "fluent" / "mid"
    exp228.MID = SYS_ARCH / "fluent" / "mid"
    exp228.COMP = SYS_ARCH / "fluent" / "comp"
    if not exp228.TOK_PATH.exists():
        exp228.TOK_PATH = SYS_ARCH / "transfer" / "tok" / "bpe8k.json"

    assert arm in ARM_SPEC
    dev = exp228.DEV
    c = dict(exp228.BASE_CFG, plato_w=max(ARM_SPEC[arm]["w30"], ARM_SPEC[arm]["w50"]),
             w30=ARM_SPEC[arm]["w30"], w50=ARM_SPEC[arm]["w50"], arm=arm, seed=seed, tokens=float(tokens))
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed + 24600)
    aux_rng = np.random.default_rng(seed + 24650)

    tk = exp228.tokenizer()
    V = tk.get_vocab_size()
    tr, held, jung, proc_tr, chat_tr, chat_he = exp229.streams()
    bad = exp228.bad_runs(tr)
    with open(exp228.MID / "proc_held.jsonl") as f:
        proc_held = [json.loads(l) for l in f]

    d_tr = np.load(D50_CACHE / "targets_dual30_50_train.npz")
    n_aux = len(d_tr["ids_off"]) - 1
    ids_f, ids_o = d_tr["ids_flat"], d_tr["ids_off"]
    pts_f, pts_o = d_tr["pts_flat"], d_tr["pts_off"]
    t30_arr, t50_arr = d_tr["c4_30"], d_tr["c4_50"]

    class PlatoToyDual(exp228.PlatoToy):
        def __init__(self, cfg, vocab):
            super().__init__(dict(cfg, plato_w=0.0), vocab)
            D = cfg["D"]
            self.head30 = nn.Linear(D, D, bias=True) if cfg["w30"] > 0 else None
            self.head50 = nn.Linear(D, D, bias=True) if cfg["w50"] > 0 else None

        def forward_dual(self, x):
            T = x.shape[1]
            h = self.e(x)
            rot = (self.cos[:T], self.sin[:T])
            h2 = h3 = None
            for li, blk in enumerate(self.b):
                h = blk(h, rot)
                if li == 2:
                    h2 = h
                elif li == 3:
                    h3 = h
            h_last = self.n(h)
            return self.h(h_last), h2, h3

    m = PlatoToyDual(c, V).to(dev)
    infer_p = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("e.", "h.", "head30.", "head50.")))
    assert infer_p == 4722304, f"Param count mismatch: {infer_p}"

    B, ctx, mb = c["batch"], c["ctx"], c["mb"]
    steps = max(1, int(tokens // (B * ctx)))
    backbone_params = [p for n, p in m.named_parameters() if not n.startswith(("head30.", "head50."))]
    head_params = [p for n, p in m.named_parameters() if n.startswith(("head30.", "head50."))]
    dec = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": c["wd"]}, {"params": nod, "weight_decay": 0.0}], c["lr"], betas=(0.9, 0.95))

    print(f"TRAIN {name}: arm={arm} seed={seed} tokens={tokens/1e6:.0f}M steps={steps} dev={dev}", flush=True)
    t0 = time.time()
    for s in range(steps):
        lr_now = exp228.lr_at(c["lr"], s, steps)
        for gr in opt.param_groups:
            gr["lr"] = lr_now
        w30_now = exp229.plato_w_at(c["w30"], s, steps)
        w50_now = exp229.plato_w_at(c["w50"], s, steps)

        i_g = exp228.draw_clean(data_rng, B // 2, len(tr) - ctx - 1, ctx, bad)
        i_p = data_rng.integers(0, len(proc_tr) - ctx - 1, B // 4)
        i_c = data_rng.integers(0, len(chat_tr) - ctx - 1, B // 4)
        xb = np.concatenate([
            np.stack([tr[j:j + ctx + 1] for j in i_g]),
            np.stack([proc_tr[j:j + ctx + 1] for j in i_p]),
            np.stack([chat_tr[j:j + ctx + 1] for j in i_c]),
        ], 0).astype(np.int64)
        data_rng.shuffle(xb, axis=0)

        opt.zero_grad(set_to_none=True)
        tot_ntp, lp30_v, lp50_v = 0.0, 0.0, 0.0
        for i in range(0, B, mb):
            x = torch.from_numpy(xb[i:i + mb, :-1]).to(dev)
            y = torch.from_numpy(xb[i:i + mb, 1:]).to(dev)
            lg = m(x).float()
            l_ntp = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
            l = l_ntp + c["zloss"] * torch.logsumexp(lg, -1).pow(2).mean()
            (l * (x.shape[0] / B)).backward()
            tot_ntp += l_ntp.item() * (x.shape[0] / B)

        # Identical aux batch across all 4 arms!
        chosen = aux_rng.integers(0, n_aux, c["aux_bs"])
        seqs, b_idx, t_idx, tg30, tg50 = [], [], [], [], []
        for bi, pi in enumerate(chosen):
            ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
            seqs.append(ids)
            pa, pb = int(pts_o[pi]), int(pts_o[pi + 1])
            for ki, pt in enumerate(pts_f[pa:pb]):
                b_idx.append(bi); t_idx.append(int(pt))
                tg30.append(t30_arr[pa + ki]); tg50.append(t50_arr[pa + ki])
        L = max(len(sq) for sq in seqs)
        xa = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
        ya = torch.zeros((len(seqs), L - 1), dtype=torch.long, device=dev)
        mk = torch.zeros((len(seqs), L - 1), dtype=torch.float32, device=dev)
        for bi, ids in enumerate(seqs):
            xa[bi, :len(ids)] = torch.from_numpy(ids)
            ya[bi, :len(ids) - 1] = torch.from_numpy(ids[1:])
            mk[bi, :len(ids) - 1] = 1.0
        b_t = torch.tensor(b_idx, dtype=torch.long, device=dev)
        t_t = torch.tensor(t_idx, dtype=torch.long, device=dev)

        lg_a, h2, h3 = m.forward_dual(xa)
        ce_a = F.cross_entropy(lg_a[:, :-1, :].float().reshape(-1, V), ya.reshape(-1), reduction="none").view_as(ya)
        l_aux = c["aux_ntp_w"] * ((ce_a * mk).sum() / mk.sum().clamp_min(1.0))
        if w30_now > 0 and m.head30 is not None:
            t30 = torch.from_numpy(np.stack(tg30)).to(dev)
            lp30 = exp228.plato_alignment_loss(m.head30(h2[b_t, t_t, :].float()), t30)
            l_aux = l_aux + w30_now * lp30
            lp30_v = float(lp30.item())
        if w50_now > 0 and m.head50 is not None:
            t50 = torch.from_numpy(np.stack(tg50)).to(dev)
            lp50 = exp228.plato_alignment_loss(m.head50(h3[b_t, t_t, :].float()), t50)
            l_aux = l_aux + w50_now * lp50
            lp50_v = float(lp50.item())
        l_aux.backward()

        torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
        if head_params:
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        assert math.isfinite(tot_ntp)
        if s % 150 == 0 or s == steps - 1:
            print(f"  step {s}/{steps} ntp={tot_ntp:.4f} p30={lp30_v:.4f} p50={lp50_v:.4f} ({(time.time()-t0)/(s+1):.3f}s/st)", flush=True)

    train_s = round(time.time() - t0, 1)
    torch.save(m.state_dict(), CKPT_DIR / f"{name}.pt")

    # Full blind evaluation!
    m.eval()
    ev_books = exp228.eval_bpb(m, c, tk, [held], nb=1_000_000)
    ev_chat = exp228.eval_bpb(m, c, tk, [chat_he])
    ev_proc = exp228.score_mcq_items(m, c, tk, proc_held)

    # Blind Procrustes alignment to unseen Phi-4 & OLMo 3 at BOTH 30% (L2) and 50% (L3)!
    d_ev = np.load(D50_CACHE / "targets_dual30_50_eval.npz", allow_pickle=True)
    e_ids_f, e_ids_o = d_ev["ids_flat"], d_ev["ids_off"]
    e_pts_f, e_pts_o = d_ev["pts_flat"], d_ev["pts_off"]
    S_L2, S_L3 = [], []
    with torch.no_grad():
        for pi in range(len(e_ids_o) - 1):
            ids = e_ids_f[int(e_ids_o[pi]):int(e_ids_o[pi + 1])].astype(np.int64)
            pts = e_pts_f[int(e_pts_o[pi]):int(e_pts_o[pi + 1])]
            _, h2, h3 = m.forward_dual(torch.tensor([ids], dtype=torch.long, device=dev))
            h2_np = h2[0].float().cpu().numpy()
            h3_np = h3[0].float().cpu().numpy()
            S_L2.append(h2_np[pts]); S_L3.append(h3_np[pts])
    S_L2 = np.concatenate(S_L2, 0).astype(np.float64)
    S_L3 = np.concatenate(S_L3, 0).astype(np.float64)
    grps = d_ev["pt_groups"].tolist()
    doms = d_ev["pt_domains"].tolist()
    rmsk = d_ev["pt_ret"]

    blind_align = {}
    for s_lbl, S_mat in (("L2_33pct", S_L2), ("L3_50pct", S_L3)):
        blind_align[s_lbl] = {}
        for t_lbl in ("phi4_30", "phi4_50", "olmo3_30", "olmo3_50"):
            r = PD.eval_procrustes_transfer(S_mat, d_ev[t_lbl], group_ids=grps, domains=doms, ret_mask=rmsk)
            blind_align[s_lbl][t_lbl] = {"cos": r["cos"], "r2_opt": r["r2_opt"], "top1": r["top1"], "top5": r["top5"]}

    # Internal HOUSE & Compositional Binding (bm120, bm_rel242, bm_neut245) 2-fold CV inside student L2 & L3!
    @torch.no_grad()
    def enc_pair_diffs(pairs, k_pos, k_neg):
        d2, d3 = [], []
        for r in pairs:
            for side in (k_pos, k_neg):
                ids = tk.encode(r[side].strip()).ids[:c["ctx"]]
                if len(ids) < 2:
                    ids = ids + [0]
                _, h2, h3 = m.forward_dual(torch.tensor([ids], dtype=torch.long, device=dev))
                v2 = h2[0, -1].float().cpu().numpy()
                v3 = h3[0, -1].float().cpu().numpy()
                if side == k_pos:
                    p2, p3 = v2, v3
                else:
                    d2.append(p2 - v2); d3.append(p3 - v3)
        return np.stack(d2, 0).astype(np.float64), np.stack(d3, 0).astype(np.float64)

    def setting_cv2(D_mat, sids):
        uniq = sorted(set(sids))
        sid2i = {s: i for i, s in enumerate(uniq)}
        si = np.array([sid2i[s] for s in sids])
        accs = []
        for sd in range(10):
            rng = np.random.default_rng(24680 + sd)
            f0 = set(int(x) for x in rng.permutation(len(uniq))[: len(uniq) // 2])
            fl = np.array([0 if int(x) in f0 else 1 for x in si])
            oof = np.zeros(len(si))
            for f in (0, 1):
                tr_, te_ = (fl != f), (fl == f)
                u = AR.unit(D_mat[tr_].mean(0))
                oof[te_] = D_mat[te_] @ u
            accs.append(float(np.mean(oof > 0)))
        return round(float(np.mean(accs)), 4)

    probes = {}
    house_pairs = [p for p in AR.load_arch_pairs() if p["cat"] in AR.HOUSE_CATS]
    hd2, hd3 = enc_pair_diffs(house_pairs, "done", "neglected")
    hsids = [str(p.get("setting", p.get("id", i))) for i, p in enumerate(house_pairs)]
    probes["house_disjoint"] = {"L2": setting_cv2(hd2, hsids), "L3": setting_cv2(hd3, hsids)}

    bm_sets = [("bm120", BM238.load_mirror_data())]
    for bnm, pth, k_j in [
        ("bm_rel242", SCRATCH_DIR / "exp240_242/bm_rel242_data.json", "bm_rel_120"),
        ("bm_neut245", SCRATCH_DIR / "exp245_controls/bm_neut245_data.json", "bm_neut_120"),
    ]:
        if pth.exists():
            bm_sets.append((bnm, json.loads(pth.read_text())[k_j]))
    for bnm, rows in bm_sets:
        bd2, bd3 = enc_pair_diffs(rows, "pos", "neg")
        bsids = [str(r["setting_id"]) for r in rows]
        probes[bnm] = {"L2": setting_cv2(bd2, bsids), "L3": setting_cv2(bd3, bsids)}

    # Also score BLiMP + COPA
    blimp_p = exp228.COMP / "blimp.jsonl"
    copa_p = exp228.COMP / "copa.jsonl"
    comp_res = {}
    start = tk.encode("\n\n").ids
    @torch.no_grad()
    def lp(ctx_s, cont):
        a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
        b = tk.encode(cont).ids
        ids = (a + b)[-(c["ctx"] + 1):]
        k = len(b)
        lg = m(torch.tensor([ids[:-1]], device=dev))[0].float()
        l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=dev)[:, None])[:, 0]
        return float(l.sum()), float(l.mean())
    if blimp_p.exists():
        bl = [json.loads(l) for l in open(blimp_p)]
        comp_res["blimp"] = round(float(np.mean([lp("", r["good"])[0] > lp("", r["bad"])[0] for r in bl])), 4)
    if copa_p.exists():
        cp = [json.loads(l) for l in open(copa_p)]
        ok = []
        for r in cp:
            link = " because" if r["question"] == "cause" else " so"
            pre = r["premise"].rstrip(".") + link
            sc = [lp(pre, " " + ch[0].lower() + ch[1:])[1] for ch in r["choices"]]
            ok.append(1.0 if sc[r["label"]] > sc[1 - r["label"]] else 0.0)
        comp_res["copa"] = round(float(np.mean(ok)), 4)

    info = {
        "name": name, "arm": arm, "seed": seed, "tokens": tokens, "train_s": train_s,
        "books_bpb": round(ev_books["bpb"], 4), "chat_bpb": round(ev_chat["bpb"], 4),
        "proc_overall": ev_proc["overall"], "comp": comp_res,
        "blind_align": blind_align, "student_probes": probes,
    }
    (EXP_DIR / f"{name}.json").write_text(json.dumps(info, indent=2))
    print(f"DONE {name}: books={info['books_bpb']} chat={info['chat_bpb']} proc_gain={ev_proc['overall']['gain']:+.3f} "
          f"phi4_30_L2={blind_align['L2_33pct']['phi4_30']['r2_opt']} phi4_50_L3={blind_align['L3_50pct']['phi4_50']['r2_opt']} "
          f"probes={probes}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract50":
        keys = tuple(sys.argv[2:]) if len(sys.argv) > 2 else ALL_6
        extract50(keys)
    elif cmd == "consensus50":
        consensus50()
    elif cmd == "train":
        name = sys.argv[2]
        kv = dict(a.split("=", 1) for a in sys.argv[3:])
        train_5m(name, kv["arm"], seed=int(kv.get("seed", 1)), tokens=int(float(kv.get("tokens", 25e6))))
    else:
        raise SystemExit(f"Unknown cmd {cmd}")
