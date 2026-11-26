"""Addendum 228 (Amendment 228a): Multi-Family Platonic Consensus Distillation & Myth/Archetype Curriculum (5M scale, 100M tokens, 3 seeds).
Trains a plain modern 5M transformer (identical 4,722,304 inference non-embed params to Addendum 225 X0!) with a
training-only linear projection on layer 2 (33% depth — Plato's Key functional peak, dropped at inference -> 0 extra params):
  P0: Plain Control                    — 50% Gutenberg + 25% proc + 25% chat + aux NTP on base annotated passages (plato_w=0)
  P1: Single-Teacher Latent            — P0 + layer-2 alignment (plato_w=0.25) to Gemma-4-E4B only
  P2: 5-Family Platonic Consensus      — P0 + layer-2 alignment (plato_w=0.25) to 5-family GPA Consensus (Gemma4+Qwen3.5+Llama3.1+Mistral+Inkling2)
  P3: Platonic Consensus + Myth/Folk   — P2 with 25% Gutenberg + 25% public Myth/Folklore + 25% proc + 25% chat & myth-enriched annotated pool

Blind evaluation (`score`):
  1. Domain-stratified, group-aware 2-fold CV Procrustes transfer to UNSEEN 6th family (Phi-4-mini): cosine, R^2, Top-1 & Top-5 retrieval
  2. Held-out Archetype Causal Shape (ALL 200 items3 myths & mechanisms): mean-centered (primary) + raw TF & TL + per-item arrays
  3. Core fluent/reasoning suite: books bpb, local Jung bpb, chat bpb, BLiMP, COPA, lastword, old MCQ, test v2, proc_held (25M/50M/100M)

Usage:
  FL_DEV=cpu python fluent/exp228.py test
  python fluent/exp228.py train NAME arm=P0..P3 seed=1 [tokens=100e6]
  python fluent/exp228.py score NAME..."""
import json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from fluent import plato_data

RUNS = HERE / "runs"
COMP = HERE / "comp"
MID = HERE / "mid"
PLATO = HERE / "plato"
PRIV = Path(os.environ.get("PLATONIC_CKPT_DIR", str(HERE / "local_ckpt")))
GUT = HERE.parent / "smallworld"
JUNG8 = Path(os.environ.get("PLATONIC_EXTRA_EVAL_DIR", str(HERE / "local_eval")))
TOK_PATH = HERE.parent / "transfer" / "tok" / "bpe8k.json"
DEV = os.environ.get("FL_DEV") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = os.environ.get("FL_AMP", "bf16" if DEV.startswith("cuda") else "")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

for p in (RUNS, RUNS / "comp", PRIV / "ckpt"):
    p.mkdir(parents=True, exist_ok=True)


def amp():
    return torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16) if AMP == "bf16" else torch.autocast(DEV.split(":")[0], enabled=False)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def streams():
    tr = np.memmap(GUT / "gut_tok2" / "train.bin", dtype=np.uint16, mode="r")
    held = np.memmap(GUT / "gut_tok2" / "held.bin", dtype=np.uint16, mode="r")
    jung_p = JUNG8 / "held.npz"
    if jung_p.exists():
        z = np.load(jung_p)
        jt, off = z["toks"], z["off"]
        jung = [jt[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    else:
        jung = []
    proc = np.memmap(MID / "proc_train.bin", dtype=np.uint16, mode="r")
    chat_tr = np.memmap(MID / "chat_train.bin", dtype=np.uint16, mode="r")
    chat_he = np.memmap(MID / "chat_held.bin", dtype=np.uint16, mode="r")
    myth = np.memmap(PLATO / "myth_train.bin", dtype=np.uint16, mode="r") if (PLATO / "myth_train.bin").exists() else tr
    return tr, held, jung, proc, chat_tr, chat_he, myth


ARM_SPEC = {
    "P0": dict(plato_w=0.0,  target="none",      use_myth=False),
    "P1": dict(plato_w=0.25, target="single",    use_myth=False),
    "P2": dict(plato_w=0.25, target="consensus", use_myth=False),
    "P3": dict(plato_w=0.25, target="consensus", use_myth=True),
}

BASE_CFG = dict(D=256, L=6, H=8, mlp=1024, ctx=1024, batch=32, mb=8, lr=3e-3, wd=0.1, zloss=1e-4,
                tokens=100e6, plato_layer=2, aux_ntp_w=0.10, aux_bs=8)


# ---------------------------------------------------------------- Model (identical 4,722,304 inference params to X0!)

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


def rope(x, cos, sin):
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


_MASK = {}


def causal_mask(n, dev):
    k = (n, str(dev))
    if k not in _MASK:
        _MASK[k] = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
    return _MASK[k]


class Block228(nn.Module):
    def __init__(self, c):
        super().__init__()
        D, self.H, mlp = c["D"], c["H"], c["mlp"]
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.qn, self.kn = RMSNorm(D // self.H), RMSNorm(D // self.H)
        self.up = nn.Linear(D, mlp, bias=False)
        self.down = nn.Linear(mlp, D, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.down.weight)

    def forward(self, x, rot):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q), *rot), rope(self.kn(k), *rot)
        if x.device.type == "cuda":
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(T, x.device))
        h = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return h + self.down(F.relu(self.up(self.n2(h))).square())


class PlatoToy(nn.Module):
    def __init__(self, c, V):
        super().__init__()
        self.c = c
        D = c["D"]
        self.e = nn.Embedding(V, D)
        nn.init.normal_(self.e.weight, 0, D ** -0.5)
        self.b = nn.ModuleList(Block228(c) for _ in range(c["L"]))
        self.n = RMSNorm(D)
        self.h = nn.Linear(D, V, bias=False)
        self.plato_head = nn.Linear(D, D, bias=True) if c["plato_w"] > 0 else None
        dh = D // c["H"]
        freq = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        t = torch.arange(c["ctx"]).float()[:, None] * freq[None]
        self.register_buffer("cos", t.cos(), persistent=False)
        self.register_buffer("sin", t.sin(), persistent=False)

    def forward_full(self, x):
        T = x.shape[1]
        assert T <= self.cos.shape[0]
        h = self.e(x)
        rot = (self.cos[:T], self.sin[:T])
        h_mid = None
        tgt_l = self.c["plato_layer"]
        for li, blk in enumerate(self.b):
            h = blk(h, rot)
            if li == tgt_l:
                h_mid = h
        h_last = self.n(h)
        return self.h(h_last), h_mid, h_last

    def forward(self, x):
        lg, _, _ = self.forward_full(x)
        return lg


def nonembed_inference(m):
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("e.", "h.", "plato_head.")))


def plato_alignment_loss(pred, tgt):
    """Exact fp32 unit-sphere cosine + RMS-normalized MSE loss."""
    with torch.autocast(pred.device.type, enabled=False):
        p = pred.float()
        t = tgt.float()
        pn = p / (p.pow(2).sum(dim=-1, keepdim=True) + 1e-8).sqrt()
        tn = t / (t.pow(2).sum(dim=-1, keepdim=True) + 1e-8).sqrt()
        l_cos = (1.0 - (pn * tn).sum(dim=-1)).mean()
        p_rms = (p.pow(2).mean() + 1e-8).sqrt()
        t_rms = (t.pow(2).mean() + 1e-8).sqrt()
        l_mse = F.mse_loss(p / p_rms, t / t_rms)
        return 0.5 * l_cos + 0.5 * l_mse


# ---------------------------------------------------------------- Shared data & eval helpers

def bad_runs(tr, tok=188, min_len=64):
    idx = np.flatnonzero(np.asarray(tr) == tok)
    if len(idx) == 0:
        return np.zeros((0, 2), np.int64)
    br = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]] + 1
    keep = ends - starts >= min_len
    return np.stack([starts[keep], ends[keep]], 1)


def draw_clean(rng, n, hi, T, bad):
    out = rng.integers(0, hi, n)
    for k in range(n):
        while any(out[k] < b and out[k] + T + 1 > a for a, b in bad):
            out[k] = rng.integers(0, hi)
    return out


def load_plato_train_pool(use_myth):
    """Return exact equal-sized subset of passage indices (Amendment 228a: strict equal-size assertion)."""
    d = np.load(PLATO / "targets_train.npz")
    doms = d["domains"]
    i_gut = np.flatnonzero(doms == "gut")
    i_myth = np.flatnonzero(doms == "myth")
    i_proc = np.flatnonzero(doms == "proc")
    i_chat = np.flatnonzero(doms == "chat")
    if use_myth:
        n_m = min(len(i_gut) // 2, len(i_myth))
        keep = np.concatenate([i_gut[:len(i_gut) - n_m], i_myth[:n_m], i_proc, i_chat])
    else:
        keep = np.concatenate([i_gut, i_proc, i_chat])
    assert len(keep) == len(i_gut) + len(i_proc) + len(i_chat), "pool size mismatch!"
    return {
        "ids_flat": d["ids_flat"], "ids_off": d["ids_off"],
        "pts_flat": d["pts_flat"], "pts_off": d["pts_off"],
        "consensus": d["consensus_256"],
        "single": d["single_gemma_256"],
        "pool_idx": keep,
    }


def make_aux_batch(pool, rng, bs, target_key, dev):
    """Pack FULL `ids` (length <= 512 <= ctx) into x_full (bs, L) so h_mid[b, pt] reads the EXACT final token
    len(ids)-1 without off-by-one clamping (Amendment 228a!), while NTP CE scores lg[:, :-1] vs ids[1:]."""
    chosen = pool["pool_idx"][rng.integers(0, len(pool["pool_idx"]), bs)]
    seqs = []
    b_idx, t_idx, tgts = [], [], []
    t_arr = pool[target_key] if target_key != "none" else None
    for bi, pi in enumerate(chosen):
        a, b = int(pool["ids_off"][pi]), int(pool["ids_off"][pi + 1])
        ids = pool["ids_flat"][a:b].astype(np.int64)
        seqs.append(ids)
        pa, pb = int(pool["pts_off"][pi]), int(pool["pts_off"][pi + 1])
        pts = pool["pts_flat"][pa:pb]
        for k_i, pt in enumerate(pts):
            pt_i = int(pt)
            assert 1 <= pt_i < len(ids), f"invalid pt {pt_i} for len {len(ids)}"
            b_idx.append(bi)
            t_idx.append(pt_i)
            if t_arr is not None:
                tgts.append(t_arr[pa + k_i])
    L = max(len(s) for s in seqs)
    x_full = torch.zeros((bs, L), dtype=torch.long, device=dev)
    y_ntp = torch.zeros((bs, L - 1), dtype=torch.long, device=dev)
    msk = torch.zeros((bs, L - 1), dtype=torch.float32, device=dev)
    for bi, ids in enumerate(seqs):
        n = len(ids)
        x_full[bi, :n] = torch.from_numpy(ids)
        y_ntp[bi, :n - 1] = torch.from_numpy(ids[1:])
        msk[bi, :n - 1] = 1.0
    b_t = torch.tensor(b_idx, dtype=torch.long, device=dev)
    t_t = torch.tensor(t_idx, dtype=torch.long, device=dev)
    tgt_t = torch.from_numpy(np.stack(tgts).astype(np.float32)).to(dev) if tgts else None
    return x_full, y_ntp, msk, b_t, t_t, tgt_t


@torch.no_grad()
def eval_bpb(m, c, tk, seqs, nb=None):
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    ctx = c["ctx"]
    arrs = [np.asarray(s[:nb] if nb else s) for s in seqs if len(s) > ctx]
    if not arrs:
        return {"bpb": None, "nats_per_token": None, "chunks": 0, "ctx_gain_768_vs_256": 0.0}
    chunks = np.stack([s[i:i + ctx + 1] for s in arrs for i in range(0, len(s) - ctx, ctx)]).astype(np.int64)
    tot, last_full, last_short = 0.0, [], []
    for i in range(0, len(chunks), c["mb"]):
        x = torch.from_numpy(chunks[i:i + c["mb"], :-1]).to(m_dev)
        y = torch.from_numpy(chunks[i:i + c["mb"], 1:]).to(m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg = m(x)
        L = F.cross_entropy(lg.float().transpose(1, 2), y, reduction="none")
        tot += L.sum().item()
        if ctx >= 512:
            last_full.append(L[:, -256:].mean(1).cpu())
            xs, ys = x[:, ctx - 512:], y[:, ctx - 512:]
            with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                lgs = m(xs)
            last_short.append(F.cross_entropy(lgs.float().transpose(1, 2), ys, reduction="none")[:, -256:].mean(1).cpu())
    nbytes = sum(len(tk.decode(ch[1:].tolist()).encode("utf-8")) for ch in chunks)
    m.train()
    out = {"bpb": tot / math.log(2) / nbytes, "nats_per_token": tot / chunks[:, 1:].size, "chunks": len(chunks)}
    if last_full:
        out["ctx_gain_768_vs_256"] = float(torch.cat(last_short).mean() - torch.cat(last_full).mean())
    return out


@torch.no_grad()
def score_mcq_items(m, c, tk, items, save_items=False):
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    start = tk.encode("\n\n").ids
    kinds = sorted({r["kind"] for r in items})

    def score_one(ctx_s, cont):
        a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
        b = tk.encode(cont).ids
        k = len(b)
        ids = (a + b)[-(c["ctx"] + 1):]
        x = torch.tensor([ids[:-1]], device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg = m(x)[0].float()
        l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=m_dev)[:, None])[:, 0]
        return float(l.mean())

    def credit(scores):
        b = max(scores)
        tied = [i for i, x in enumerate(scores) if x == b]
        return (1.0 / len(tied)) if 0 in tied else 0.0

    res = {}
    all_w, all_wo, per_item = [], [], []
    for kind in kinds:
        sub = [r for r in items if r["kind"] == kind]
        w = [credit([score_one(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in sub]
        wo = [credit([score_one(f"Question: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in sub]
        all_w.extend(w)
        all_wo.extend(wo)
        if save_items:
            for r, wi, woi in zip(sub, w, wo):
                per_item.append([r["kind"], r.get("slot", -1), wi, woi])
        res[kind] = {"n": len(sub), "with": float(np.mean(w)), "without": float(np.mean(wo)), "gain": float(np.mean(w) - np.mean(wo))}
    res["overall"] = {"n": len(all_w), "with": float(np.mean(all_w)), "without": float(np.mean(all_wo)), "gain": float(np.mean(all_w) - np.mean(all_wo))}
    if save_items:
        res["items"] = per_item
    m.train()
    return res


@torch.no_grad()
def score_platonic_and_archetypes(m, c):
    """Blind evaluation on held-out targets_eval.npz (Amendment 228a: domain-stratified group-aware 2-fold CV + mean-centered TF/TL + per-item arrays!)."""
    ev_path = PLATO / "targets_eval.npz"
    if not ev_path.exists():
        return {}
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    d = np.load(ev_path)
    ids_f, ids_o = d["ids_flat"], d["ids_off"]
    pts_f, pts_o = d["pts_flat"], d["pts_off"]
    metas = [json.loads(s) for s in d["meta_json"]]

    stu_pts_mid, stu_story_mid, stu_story_last = [], [], []
    for i in range(len(metas)):
        ids = ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64)
        pts = pts_f[int(pts_o[i]):int(pts_o[i + 1])]
        x = torch.from_numpy(ids[None, :]).to(m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, h_mid, h_last = m.forward_full(x)
        hm = h_mid[0].float().cpu().numpy()
        hl = h_last[0].float().cpu().numpy()
        stu_pts_mid.append(hm[pts])
        stu_story_mid.append(hm[1:].mean(axis=0) if len(hm) > 1 else hm[0])
        stu_story_last.append(hl[1:].mean(axis=0) if len(hl) > 1 else hl[0])

    S_pts = np.concatenate(stu_pts_mid, axis=0)
    S_mid = np.stack(stu_story_mid, axis=0)
    S_last = np.stack(stu_story_last, axis=0)

    pt_groups = d["pt_groups"].tolist()
    pt_domains = d["pt_domains"].tolist()
    pt_ret_mask = d["pt_ret_mask"]
    st_groups = [m_r["group_id"] for m_r in metas]
    st_domains = [m_r["domain"] for m_r in metas]
    st_ret_mask = np.array([m_r.get("role", "") != "foil" for m_r in metas], dtype=bool)

    out = {
        "blind_phi4_pts": plato_data.eval_procrustes_transfer(S_pts, d["blind_phi4_256"].astype(np.float64), group_ids=pt_groups, domains=pt_domains, ret_mask=pt_ret_mask),
        "blind_phi4_story": plato_data.eval_procrustes_transfer(S_mid, d["blind_phi4_story_256"].astype(np.float64), group_ids=st_groups, domains=st_domains, ret_mask=st_ret_mask),
        "consensus_eval_pts": plato_data.eval_procrustes_transfer(S_pts, d["consensus_256"].astype(np.float64), group_ids=pt_groups, domains=pt_domains, ret_mask=pt_ret_mask),
    }
    if "blind_olmo3_256" in d.files:
        out["blind_olmo3_pts"] = plato_data.eval_procrustes_transfer(S_pts, d["blind_olmo3_256"].astype(np.float64), group_ids=pt_groups, domains=pt_domains, ret_mask=pt_ret_mask)
        out["blind_olmo3_story"] = plato_data.eval_procrustes_transfer(S_mid, d["blind_olmo3_story_256"].astype(np.float64), group_ids=st_groups, domains=st_domains, ret_mask=st_ret_mask)

    # Held-out items3 Archetype Causal Shape (mean-centered PRIMARY matching statedpat.py/cross_analyze.py, plus raw & per-item arrays!)
    def cos_sim(a, b):
        return float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))

    i3_idx = [idx for idx, meta in enumerate(metas) if meta.get("domain", "").startswith("items3_")]
    for lname, mat in (("layer2", S_mid), ("final", S_last)):
        mat_c = mat.copy()
        if i3_idx:
            mat_c[i3_idx] = mat[i3_idx] - mat[i3_idx].mean(axis=0, keepdims=True)
        for mode_lbl, M_use in (("centered", mat_c), ("raw", mat)):
            by_item = defaultdict(dict)
            for idx in i3_idx:
                meta = metas[idx]
                by_item[(meta["kind"], meta["item_id"])][meta["role"]] = M_use[idx]
            arch_res = {}
            for kind in ("myth", "mech"):
                tfs, tls, per_it = [], [], []
                for (k, iid), roles in sorted(by_item.items()):
                    if k == kind and all(r in roles for r in ("anchor", "true", "foil", "lure")):
                        c_tr = cos_sim(roles["anchor"], roles["true"])
                        tf = float(c_tr > cos_sim(roles["anchor"], roles["foil"]))
                        tl = float(c_tr > cos_sim(roles["anchor"], roles["lure"]))
                        tfs.append(tf); tls.append(tl)
                        if mode_lbl == "centered":
                            per_it.append([iid, tf, tl])
                if tfs:
                    entry = {"n": len(tfs), "TF": round(float(np.mean(tfs)), 4), "TL": round(float(np.mean(tls)), 4)}
                    if mode_lbl == "centered":
                        entry["items"] = per_it
                    arch_res[kind] = entry
            out[f"items3_{lname}_{mode_lbl}"] = arch_res
    m.train()
    return out


# ---------------------------------------------------------------- Training & scoring

def lr_at(lr, s, steps):
    w = max(1, steps // 50)
    d0 = int(steps * 0.8)
    return lr * ((s + 1) / w if s < w else 1.0 if s < d0 else max(0.0, (steps - s) / (steps - d0)))


def train(name, arm, seed, tokens=100_000_000, overwrite=False):
    assert arm in ARM_SPEC, f"unknown arm {arm}"
    assert not name.startswith(("C", "B", "OLD", "M", "T", "S", "D", "X", "O", "W")), "name collision guard"
    if not overwrite:
        assert not (RUNS / f"{name}.json").exists(), f"run already exists: {name}.json (pass overwrite=1 or choose a new run name)"

    c = dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, seed=seed, tokens=float(tokens))
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed + 22500)
    aux_rng = np.random.default_rng(seed + 22800)

    tk = tokenizer()
    V = tk.get_vocab_size()
    tr, held, jung, proc_tr, chat_tr, chat_he, myth_tr = streams()
    bad = bad_runs(tr)
    bad_m = bad_runs(myth_tr)
    with open(MID / "proc_held.jsonl") as f:
        proc_held = [json.loads(l) for l in f]
    aux_pool = load_plato_train_pool(use_myth=c["use_myth"])

    m = PlatoToy(c, V).to(DEV)
    B, ctx, mb = c["batch"], c["ctx"], c["mb"]
    assert B % 4 == 0
    steps = max(1, int(tokens // (B * ctx)))
    marks = {max(1, steps // 4): "25M", max(2, steps // 2): "50M"}

    backbone_params = [p for n, p in m.named_parameters() if not n.startswith("plato_head.")]
    head_params = [p for n, p in m.named_parameters() if n.startswith("plato_head.")]
    dec = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": c["wd"]}, {"params": nod, "weight_decay": 0.0}], c["lr"], betas=(0.9, 0.95))

    print(f"{name}: arm={arm} target={c['target']} myth={c['use_myth']} seed={seed} infer_nonembed={nonembed_inference(m)} steps={steps} dev={DEV} amp={AMP}", flush=True)
    t0 = time.time()
    curve = {}

    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(c["lr"], s, steps)

        if c["use_myth"]:
            i_g = draw_clean(data_rng, B // 4, len(tr) - ctx - 1, ctx, bad)
            i_m = draw_clean(data_rng, B // 4, len(myth_tr) - ctx - 1, ctx, bad_m)
            i_p = data_rng.integers(0, len(proc_tr) - ctx - 1, B // 4)
            i_c = data_rng.integers(0, len(chat_tr) - ctx - 1, B // 4)
            xb = np.concatenate([
                np.stack([tr[j:j + ctx + 1] for j in i_g]),
                np.stack([myth_tr[j:j + ctx + 1] for j in i_m]),
                np.stack([proc_tr[j:j + ctx + 1] for j in i_p]),
                np.stack([chat_tr[j:j + ctx + 1] for j in i_c]),
            ], 0).astype(np.int64)
        else:
            i_g = draw_clean(data_rng, B // 2, len(tr) - ctx - 1, ctx, bad)
            i_p = data_rng.integers(0, len(proc_tr) - ctx - 1, B // 4)
            i_c = data_rng.integers(0, len(chat_tr) - ctx - 1, B // 4)
            xb = np.concatenate([
                np.stack([tr[j:j + ctx + 1] for j in i_g]),
                np.stack([proc_tr[j:j + ctx + 1] for j in i_p]),
                np.stack([chat_tr[j:j + ctx + 1] for j in i_c]),
            ], 0).astype(np.int64)
        data_rng.shuffle(xb, axis=0)

        opt.zero_grad(set_to_none=True)
        tot_ntp, tot_plato = 0.0, 0.0
        for i in range(0, B, mb):
            x = torch.from_numpy(xb[i:i + mb, :-1]).to(DEV)
            y = torch.from_numpy(xb[i:i + mb, 1:]).to(DEV)
            with amp():
                lg = m(x)
            lg_f = lg.float()
            l_ntp = F.cross_entropy(lg_f.reshape(-1, V), y.reshape(-1))
            l = l_ntp
            if c["zloss"]:
                l = l + c["zloss"] * torch.logsumexp(lg_f, -1).pow(2).mean()
            (l * (x.shape[0] / B)).backward()
            tot_ntp += l_ntp.item() * (x.shape[0] / B)

        # Auxiliary annotated passage batch (full sequence -> exact final-token alignment without clamping!)
        xa_full, ya_ntp, mska, b_t, t_t, tgt_t = make_aux_batch(aux_pool, aux_rng, c["aux_bs"], c["target"], DEV)
        with amp():
            lg_a, h_mid, _ = m.forward_full(xa_full)
        lg_af = lg_a[:, :-1, :].float()
        ce_a = F.cross_entropy(lg_af.reshape(-1, V), ya_ntp.reshape(-1), reduction="none").view_as(ya_ntp)
        l_aux = c["aux_ntp_w"] * ((ce_a * mska).sum() / mska.sum().clamp_min(1.0))
        if c["plato_w"] > 0 and tgt_t is not None and len(b_t) > 0:
            pred = m.plato_head(h_mid[b_t, t_t, :].float())
            l_p = plato_alignment_loss(pred, tgt_t)
            l_aux = l_aux + c["plato_w"] * l_p
            tot_plato = float(l_p.item())
        l_aux.backward()

        # Clip backbone and training-only plato_head separately so P0 vs P1..P3 have identical backbone clip semantics
        gn = float(torch.nn.utils.clip_grad_norm_(backbone_params, 1.0))
        if head_params:
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        assert math.isfinite(tot_ntp), "diverged"

        if s % 100 == 0 or s == steps - 1:
            p_str = f" plato {tot_plato:.4f}" if c["plato_w"] > 0 else ""
            print(f"step {s}/{steps} ntp {tot_ntp:.4f}{p_str} gnorm {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} s/step {(time.time() - t0) / (s + 1):.3f}", flush=True)
        if (s + 1) in marks:
            lbl = marks[s + 1]
            pr = score_mcq_items(m, c, tk, proc_held)
            curve[lbl] = pr
            print(f"CHECKPOINT {name} {lbl} proc_held overall {pr['overall']}", flush=True)

    train_s = round(time.time() - t0)
    ev_books = eval_bpb(m, c, tk, [held], nb=2_000_000)
    ev_jung = eval_bpb(m, c, tk, jung)
    ev_chat = eval_bpb(m, c, tk, [chat_he])
    pr_end = score_mcq_items(m, c, tk, proc_held)
    curve["100M"] = pr_end
    torch.save(m.state_dict(), PRIV / "ckpt" / f"{name}.pt")
    info = {
        "name": name, "config": c, "dev": DEV, "amp": AMP, "steps": steps,
        "infer_nonembed": nonembed_inference(m),
        "train_s": train_s, "s_per_step": round(train_s / steps, 4), "last_ntp": tot_ntp,
        "books": ev_books, "jung": ev_jung, "chat_held": ev_chat, "proc_curve": curve,
    }
    with open(RUNS / f"{name}.json", "w") as f:
        json.dump(info, f, indent=1)
    _fb = lambda ev: f"{ev['bpb']:.4f}" if ev.get("bpb") is not None else "n/a"
    print(f"EXP228_DONE {name} books={_fb(ev_books)} jung={_fb(ev_jung)} chat={_fb(ev_chat)} proc_100M={pr_end['overall']['gain']:+.3f} ({train_s} s)", flush=True)


def score(names):
    tk = tokenizer()
    V = tk.get_vocab_size()
    tests = {}
    for k, path in (
        ("blimp", COMP / "blimp.jsonl"),
        ("copa", COMP / "copa.jsonl"),
        ("lastword", COMP / "lastword.jsonl"),
        ("mcq", COMP / "mcq.jsonl"),
        ("mcq2", COMP / "mcq2.jsonl"),
        ("proc", MID / "proc_held.jsonl"),
    ):
        if path.exists():
            with open(path) as f:
                tests[k] = [json.loads(l) for l in f]

    for name in names:
        with open(RUNS / f"{name}.json") as f:
            c = json.load(f)["config"]
        m = PlatoToy(c, V).to(DEV)
        m.load_state_dict(torch.load(PRIV / "ckpt" / f"{name}.pt", map_location=DEV, weights_only=True))
        m.eval()
        start = tk.encode("\n\n").ids
        t0 = time.time()

        @torch.no_grad()
        def lp(ctx_s, cont):
            a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
            b = tk.encode(cont).ids
            ids = (a + b)[-(c["ctx"] + 1):]
            k = len(b)
            x = torch.tensor([ids[:-1]], device=DEV)
            with amp():
                lg = m(x)[0].float()
            l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=DEV)[:, None])[:, 0]
            return float(l.sum()), float(l.mean())

        def credit(scores):
            b = max(scores)
            tied = [i for i, x in enumerate(scores) if x == b]
            return (1.0 / len(tied)) if 0 in tied else 0.0

        @torch.no_grad()
        def top1(ctx_s):
            ids = (start + tk.encode(ctx_s).ids)[-c["ctx"]:]
            with amp():
                return int(m(torch.tensor([ids], device=DEV))[0, -1].argmax())

        res = {}
        if "blimp" in tests:
            ok = [lp("", r["good"])[0] > lp("", r["bad"])[0] for r in tests["blimp"]]
            res["blimp"] = float(np.mean(ok))
            by = defaultdict(list)
            for r, o in zip(tests["blimp"], ok):
                by[r["set"]].append(o)
            res["blimp_sets_below_chance"] = int(sum(np.mean(v) < 0.5 for v in by.values()))
        if "copa" in tests:
            ok = []
            for r in tests["copa"]:
                link = " because" if r["question"] == "cause" else " so"
                pre = r["premise"].rstrip(".") + link
                sc = [lp(pre, " " + ch[0].lower() + ch[1:])[1] for ch in r["choices"]]
                ok.append(0.5 if sc[0] == sc[1] else float(int(np.argmax(sc)) == r["label"]))
            res["copa"] = float(np.mean(ok))
        if "lastword" in tests:
            res["lastword_full"] = float(np.mean([top1(r["context"]) == tk.encode(" " + r["word"]).ids[0] for r in tests["lastword"]]))
            res["lastword_last_sentence"] = float(np.mean([top1(r["last_sentence"]) == tk.encode(" " + r["word"]).ids[0] for r in tests["lastword"]]))
        if "mcq" in tests:
            w, wo = [], []
            for r in tests["mcq"]:
                opts = [r["answer"]] + r["wrong"]
                w.append(credit([lp(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o)[1] for o in opts]))
                wo.append(credit([lp(f"Question: {r['question']}\nAnswer:", " " + o)[1] for o in opts]))
            res["mcq_with_passage"] = float(np.mean(w))
            res["mcq_without"] = float(np.mean(wo))
            res["mcq_gain"] = res["mcq_with_passage"] - res["mcq_without"]
            res["mcq_items"] = [[r["pid"], a, b] for r, a, b in zip(tests["mcq"], w, wo)]
        if "mcq2" in tests:
            res["v2"] = score_mcq_items(m, c, tk, tests["mcq2"], save_items=True)
        if "proc" in tests:
            res["proc"] = score_mcq_items(m, c, tk, tests["proc"], save_items=True)
        res["platonic"] = score_platonic_and_archetypes(m, c)
        res["s"] = round(time.time() - t0)
        with open(RUNS / "comp" / f"{name}.json", "w") as f:
            json.dump(res, f, indent=1)
        pg = res.get("proc", {}).get("overall", {}).get("gain", float("nan"))
        vg = res.get("v2", {}).get("overall", {}).get("gain", float("nan"))
        pcos = res.get("platonic", {}).get("blind_phi4_pts", {}).get("cos", float("nan"))
        mtf = res.get("platonic", {}).get("items3_layer2_centered", {}).get("myth", {}).get("TF", float("nan"))
        print(f"SCORE_DONE {name}: blimp={res.get('blimp', 0):.3f} proc={pg:+.3f} v2={vg:+.3f} blind_phi4_cos={pcos:.3f} myth_TF={mtf:.3f} ({res['s']} s)", flush=True)


def self_test():
    """Verify exact 4,722,304 inference param parity, strict causality, unclamped final-token aux alignment, and finite grads on P0..P3."""
    torch.manual_seed(0)
    V = 8192
    for arm in ("P0", "P1", "P2", "P3"):
        c = dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, ctx=256)
        m = PlatoToy(c, V)
        ne = nonembed_inference(m)
        assert ne == 4_722_304, f"{arm} infer_nonembed={ne} != 4722304"
        m.eval()
        with torch.no_grad():
            for _, p in m.named_parameters():
                if p.dim() >= 2 and p.abs().max() == 0:
                    nn.init.normal_(p, 0, 0.03)
            x = torch.randint(0, V, (2, 80))
            lg0, hm0, _ = m.forward_full(x)
            x2 = x.clone()
            x2[:, 40:] = torch.randint(0, V, (2, 40))
            lg1, hm1, _ = m.forward_full(x2)
            assert (lg1[:, :40] - lg0[:, :40]).abs().max().item() < 1e-4, f"{arm} logit leak!"
            assert (hm1[:, :40] - hm0[:, :40]).abs().max().item() < 1e-4, f"{arm} layer2 leak!"
        m.train()
        lg, hm, _ = m.forward_full(x)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        if c["plato_w"] > 0:
            tgt = torch.randn(2, 5, c["D"])
            # Include exact final token index 79 (no clamping!)
            pred = m.plato_head(hm[:, [10, 20, 30, 40, 79], :].float())
            loss = loss + c["plato_w"] * plato_alignment_loss(pred, tgt)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), f"{arm} bad grad!"
        print(f"  {arm}: infer_nonembed={ne} leak=OK final_tok_align=OK grads=OK", flush=True)
    print("EXP228_TEST_OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "train":
        kw = dict(kv.split("=") for kv in sys.argv[3:])
        train(
            sys.argv[2], kw["arm"], int(kw.get("seed", 1)),
            tokens=float(kw.get("tokens", 100e6)),
            overwrite=bool(int(kw.get("overwrite", 0))),
        )
    elif cmd == "score":
        score(sys.argv[2:])
    else:
        raise ValueError(f"unknown cmd {cmd}")
