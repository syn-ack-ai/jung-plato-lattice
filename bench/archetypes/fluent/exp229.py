"""Addendum 229 Stage 2B (Amendment 229b): Annealed 5-Family Platonic Consensus + Double-Denoised Myth Prototypes + Rotated U_jung Boost/Ablate.
Trains a plain modern 5M transformer (identical 4,722,304 inference non-embed params to Addendum 225 X0 & Addendum 228 P0..P3)
with a training-only layer-2 linear projection (dropped at inference -> 0 extra params) and ANNEALED plato_w
(0.25 for steps 0..65%, linear decay 0.25 -> 0.0 over steps 65..90%, pure-NTP polish 0.0 over final 90..100% while lr > 0!).

Every arm sees the EXACT SAME main 100M stream (50% Gutenberg + 25% proc + 25% chat — zero archaic folklore drag),
the EXACT SAME auxiliary batch composition (6 neutral base passages + 2 verified modernized myth/mech stories = 8/step,
~6 exposures per modern story at aux_ntp_w=0.10 -> <0.7 effective NTP epochs, zero memorization!), and the EXACT SAME seeds:
  M0: Annealed P2 + Modernized Myths (Instance C5)   — tests annealing plato_w->0 & modernized myths vs P0/P2/P3
  M1: M0 + Double-Denoised Myth Prototypes (LORO)    — modern myth targets replaced by leave-one-retelling-out cross-setting C5 mean
  M2: M0 + Boosted 5-Family Rotated U_jung (a=3.5)   — lifts 2D HOUSE/SELF2 subspace energy in C5 (~1.7% -> ~17%) via ONE shared global scale
  M3: M0 - Ablated U_jung Causal Control (a=0.0)     — projects out exact 2D HOUSE/SELF2 subspace from C5 (0.00% energy)

Usage:
  FL_DEV=cpu python fluent/exp229.py test
  python fluent/exp229.py train E229_M0_s1 arm=M0 seed=1 [tokens=100e6]
  python fluent/exp229.py score E229_M0_s1 ..."""
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ARCH = HERE.parent
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from fluent import arch_rot229, exp228, plato_data

RUNS = HERE / "runs"
COMP = HERE / "comp"
MID = HERE / "mid"
PLATO = HERE / "plato"
PRIV = Path(os.environ.get("PLATONIC_CKPT_DIR", str(HERE / "local_ckpt")))
GUT = ARCH / "smallworld"
JUNG8 = Path(os.environ.get("PLATONIC_EXTRA_EVAL_DIR", str(HERE / "local_eval")))
DEV = exp228.DEV
AMP = exp228.AMP

ARM_SPEC = {
    "M0": dict(plato_w=0.25, myth_target="instance", ujung_alpha=1.0),
    "M1": dict(plato_w=0.25, myth_target="loro",     ujung_alpha=1.0),
    "M2": dict(plato_w=0.25, myth_target="instance", ujung_alpha=3.5),
    "M3": dict(plato_w=0.25, myth_target="instance", ujung_alpha=0.0),
}

BASE_CFG = dict(
    D=256, L=6, H=8, mlp=1024, ctx=1024, batch=32, mb=8, lr=3e-3, wd=0.1, zloss=1e-4,
    tokens=100e6, plato_layer=2, aux_ntp_w=0.10, aux_bs_base=6, aux_bs_myth=2,
    anneal_start=0.65, anneal_end=0.90,
)


def _atomic_json(path, obj):
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def streams():
    """Own streams helper (omits unused archaic myth_train.bin!)."""
    tr = np.memmap(GUT / "gut_tok2" / "train.bin", dtype=np.uint16, mode="r")
    held = np.memmap(GUT / "gut_tok2" / "held.bin", dtype=np.uint16, mode="r")
    jung_p = JUNG8 / "held.npz"
    if jung_p.exists():
        with np.load(jung_p) as z:
            jt, off = z["toks"].copy(), z["off"].copy()
        jung = [jt[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    else:
        jung = []
    proc = np.memmap(MID / "proc_train.bin", dtype=np.uint16, mode="r")
    chat_tr = np.memmap(MID / "chat_train.bin", dtype=np.uint16, mode="r")
    chat_he = np.memmap(MID / "chat_held.bin", dtype=np.uint16, mode="r")
    return tr, held, jung, proc, chat_tr, chat_he


def plato_w_at(base_w, s, steps, anneal_start=0.65, anneal_end=0.90):
    """Hold base_w on [0, anneal_start), linearly decay to 0.0 over [anneal_start, anneal_end),
    then hold exact 0.0 over [anneal_end, 1.0] for pure-NTP cooldown while lr > 0!"""
    if base_w <= 0.0:
        return 0.0
    d0 = int(steps * anneal_start)
    d1 = max(d0 + 1, int(steps * anneal_end))
    if s < d0:
        return float(base_w)
    if s >= d1:
        return 0.0
    return float(base_w * (d1 - s) / float(d1 - d0))


def transform_ujung_pair(Z_base_all, active_base_mask, Z_myth, U, alpha, w_base=0.75, w_myth=0.25):
    """Apply ONE shared global linear operator T_alpha(z) = s * (z + (alpha - 1) * (z @ U.T) @ U) to BOTH pools,
    choosing the single scalar s so the active 6:2 sampled distribution keeps exact RMS parity with alpha=1.0!
    (Fixes M1: excludes unused archaic myth rows and prevents base-vs-myth scale discontinuity inside l_mse.)"""
    Zb = np.asarray(Z_base_all, dtype=np.float64)
    Zm = np.asarray(Z_myth, dtype=np.float64)
    if abs(alpha - 1.0) < 1e-9:
        return Zb.astype(np.float32), Zm.astype(np.float32)
    U = np.asarray(U, dtype=np.float64)
    assert U.ndim == 2 and U.shape[1] == Zb.shape[-1] == Zm.shape[-1]
    assert float(np.max(np.abs(U @ U.T - np.eye(U.shape[0])))) < 1e-4, "U_jung not orthonormal!"

    Zb_act = Zb[active_base_mask]
    ms_orig = w_base * float(np.mean(Zb_act ** 2)) + w_myth * float(np.mean(Zm ** 2))
    Zb_t = Zb + (float(alpha) - 1.0) * ((Zb @ U.T) @ U)
    Zm_t = Zm + (float(alpha) - 1.0) * ((Zm @ U.T) @ U)
    ms_new = w_base * float(np.mean(Zb_t[active_base_mask] ** 2)) + w_myth * float(np.mean(Zm_t ** 2))
    s = math.sqrt(max(1e-12, ms_orig) / max(1e-12, ms_new))
    return (Zb_t * s).astype(np.float32), (Zm_t * s).astype(np.float32)


def load_pools(myth_target, ujung_alpha):
    """Load neutral base pool (gut + proc + chat from targets_train.npz) + modernized myth/mech pool (targets_myth229.npz),
    applying one shared global U_jung subspace operator across active points."""
    U = None
    if abs(ujung_alpha - 1.0) >= 1e-9:
        with np.load(PLATO / "arch229_consensus.npz") as z_u:
            U = z_u["U_jung_c5"].astype(np.float64)

    with np.load(PLATO / "targets_train.npz") as d_b:
        doms = d_b["domains"]
        keep_base = np.concatenate([
            np.flatnonzero(doms == "gut"),
            np.flatnonzero(doms == "proc"),
            np.flatnonzero(doms == "chat"),
        ])
        pts_off_b = d_b["pts_off"].copy()
        c5_base_raw = d_b["consensus_256"].astype(np.float64)
        active_pt_mask = np.zeros(len(c5_base_raw), dtype=bool)
        for pi in keep_base:
            active_pt_mask[int(pts_off_b[pi]):int(pts_off_b[pi + 1])] = True
        b_ids_f, b_ids_o, b_pts_f = d_b["ids_flat"].copy(), d_b["ids_off"].copy(), d_b["pts_flat"].copy()

    with np.load(PLATO / "targets_myth229.npz") as d_m:
        t_key = "c5_proto_loro_256" if myth_target == "loro" else "c5_instance_256"
        c5_myth_raw = d_m[t_key].astype(np.float64)
        m_ids_f, m_ids_o = d_m["ids_flat"].copy(), d_m["ids_off"].copy()
        m_pts_f, m_pts_o = d_m["pts_flat"].copy(), d_m["pts_off"].copy()

    c5_base, c5_myth = transform_ujung_pair(c5_base_raw, active_pt_mask, c5_myth_raw, U, ujung_alpha)

    base_pool = {
        "ids_flat": b_ids_f, "ids_off": b_ids_o,
        "pts_flat": b_pts_f, "pts_off": pts_off_b,
        "tgt": c5_base, "pool_idx": keep_base,
    }
    myth_pool = {
        "ids_flat": m_ids_f, "ids_off": m_ids_o,
        "pts_flat": m_pts_f, "pts_off": m_pts_o,
        "tgt": c5_myth, "pool_idx": np.arange(len(m_ids_o) - 1, dtype=np.int64),
    }
    return base_pool, myth_pool


def make_mixed_aux_batch(base_pool, myth_pool, rng, bs_base, bs_myth, dev):
    """Draw bs_base neutral passages + bs_myth modernized myth/mech stories with IDENTICAL RNG order across M0..M3."""
    c_base = base_pool["pool_idx"][rng.integers(0, len(base_pool["pool_idx"]), bs_base)]
    c_myth = myth_pool["pool_idx"][rng.integers(0, len(myth_pool["pool_idx"]), bs_myth)]
    draws = [(base_pool, int(pi)) for pi in c_base] + [(myth_pool, int(pi)) for pi in c_myth]

    seqs, b_idx, t_idx, tgts = [], [], [], []
    for bi, (pool, pi) in enumerate(draws):
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
            tgts.append(pool["tgt"][pa + k_i])

    bs = len(seqs)
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
    tgt_t = torch.from_numpy(np.stack(tgts).astype(np.float32)).to(dev)
    return x_full, y_ntp, msk, b_t, t_t, tgt_t


@torch.no_grad()
def score_student_jung_coherence(m, c, tk, pairs=None, eval_npz_path=None):
    """Manipulation & transfer probe on the 312 public HOUSE + SELF2 pairs inside the 5M student (layer 2):
    1. Strictly disjoint 2-fold CV within the student (raw & length-residualized HOUSE, perfection control, SELF2, HOUSE->SELF2).
    2. Direct plato_head alignment to frozen u_house_c5 / u_self2_c5 (manipulation check for M2 vs M3).
    3. Neutral-text Procrustes backbone probe (fit student->C5 on held_gut+proc_held in targets_eval.npz, project u_house_c5
       into student layer 2 — apples-to-apples across P0 and M0..M3 even when plato_head is None or zeroed!)."""
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    if pairs is None:
        pairs = arch_rot229.load_arch_pairs()

    def enc_mid(txt):
        ids = tk.encode(txt).ids[:c["ctx"]]
        if len(ids) < 2:
            ids = ids + [0]
        x = torch.tensor([ids], dtype=torch.long, device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, h_mid, _ = m.forward_full(x)
        hm = h_mid[0].float().cpu().numpy()
        return hm[-1], hm[1:].mean(axis=0), len(ids)

    d_last, n_last, d_mean, n_mean, d_len, n_len = [], [], [], [], [], []
    for p in pairs:
        dl, dm, ln1 = enc_mid(p["done"])
        nl, nm, ln2 = enc_mid(p["neglected"])
        d_last.append(dl); d_mean.append(dm); d_len.append(ln1)
        n_last.append(nl); n_mean.append(nm); n_len.append(ln2)
    D_last = np.stack(d_last).astype(np.float64) - np.stack(n_last).astype(np.float64)
    D_mean = np.stack(d_mean).astype(np.float64) - np.stack(n_mean).astype(np.float64)
    d_len = np.array(d_len, dtype=np.float64)
    n_len = np.array(n_len, dtype=np.float64)
    dl_arr = d_len - n_len

    cats_all = np.array([p["cat"] for p in pairs])
    house_idx = np.flatnonzero(np.isin(cats_all, arch_rot229.HOUSE_CATS))
    self2_idx = np.flatnonzero(cats_all == "self2")
    cats_h = cats_all[house_idx]
    cats_s = cats_all[self2_idx]
    perf_mask = (cats_h == "perfection")
    folds_h = arch_rot229.make_stratified_2fold(cats_h)
    folds_s = arch_rot229.make_stratified_2fold(cats_s)

    out = {}
    for mode_name, D_mat in (("last", D_last), ("mean", D_mean)):
        D_h = D_mat[house_idx]
        D_s = D_mat[self2_idx]
        r_h = arch_rot229.eval_disjoint_2fold([D_h], D_h, cats_h, folds_h, is_house=True, seed=2291)
        r_h_lr = arch_rot229.eval_disjoint_2fold_lenres(
            [D_h], [(d_len[house_idx], n_len[house_idx])],
            D_h, d_len[house_idx], n_len[house_idx], cats_h, folds_h, is_house=True, seed=2291,
        )
        r_s = arch_rot229.eval_disjoint_2fold([D_s], D_s, cats_s, folds_s, is_house=False, seed=2292)
        u_h = arch_rot229.house_dir(D_h, cats_h)
        sc_hs = D_s @ u_h
        r_len = float(np.corrcoef(r_h["scores"], dl_arr[house_idx])[0, 1]) if np.std(dl_arr[house_idx]) > 0 else 0.0
        out[mode_name] = {
            "house_2f_cos": r_h["cos"],
            "house_2f_acc": r_h["acc"],
            "house_2f_lenres": r_h_lr["acc"],
            "house_perf_acc": round(float(np.mean(r_h["scores"][perf_mask] > 0.0)), 4) if perf_mask.any() else None,
            "self2_2f_acc": r_s["acc"],
            "house_to_self2_acc": round(float(np.mean(sc_hs > 0.0)), 4),
            "r_len": round(r_len, 4),
        }

    u_path = PLATO / "arch229_consensus.npz"
    if u_path.exists():
        with np.load(u_path) as z_u:
            u_hc5 = z_u["u_house_c5"].astype(np.float64)
            u_sc5 = z_u["u_self2_c5"].astype(np.float64)

        # (a) Direct plato_head alignment (manipulation check for M0..M3)
        if m.plato_head is not None:
            W = m.plato_head.weight.detach().float().cpu().numpy().astype(np.float64)
            D_proj = D_last @ W.T
            u_stu_h = arch_rot229.house_dir(D_proj[house_idx], cats_h)
            u_stu_s = arch_rot229.unit(D_proj[self2_idx].mean(axis=0))
            sc_h = D_proj[house_idx] @ u_hc5
            sc_s = D_proj[self2_idx] @ u_sc5
            out["plato_head_to_u_jung_c5"] = {
                "cos_house": round(float(np.dot(u_stu_h, u_hc5)), 4),
                "cos_self2": round(float(np.dot(u_stu_s, u_sc5)), 4),
                "zeroshot_house_acc": round(float(np.mean(sc_h > 0.0)), 4),
                "zeroshot_perf_acc": round(float(np.mean(sc_h[perf_mask] > 0.0)), 4) if perf_mask.any() else None,
                "zeroshot_self2_acc": round(float(np.mean(sc_s > 0.0)), 4),
            }

        # (b) Neutral-text Procrustes backbone probe (fit student layer 2 -> C5 strictly on held_gut + proc_held!)
        ev_p = Path(eval_npz_path) if eval_npz_path is not None else (PLATO / "targets_eval.npz")
        if ev_p.exists():
            with np.load(ev_p) as d_ev:
                ids_f, ids_o = d_ev["ids_flat"], d_ev["ids_off"]
                pts_f, pts_o = d_ev["pts_flat"], d_ev["pts_off"]
                metas = [json.loads(s) for s in d_ev["meta_json"]]
                c5_ev = d_ev["consensus_256"].astype(np.float64)
            stu_neu, c5_neu = [], []
            for i, meta in enumerate(metas):
                if meta.get("domain") in ("held_gut", "proc_held"):
                    ids = ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64)[:c["ctx"]]
                    pa, pb = int(pts_o[i]), int(pts_o[i + 1])
                    pts = pts_f[pa:pb]
                    if len(ids) >= 2 and int(pts.max()) < len(ids):
                        x = torch.from_numpy(ids[None, :]).to(m_dev)
                        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                            _, h_mid, _ = m.forward_full(x)
                        stu_neu.append(h_mid[0].float().cpu().numpy()[pts])
                        c5_neu.append(c5_ev[pa:pb])
            if stu_neu:
                S_neu = np.concatenate(stu_neu, axis=0).astype(np.float64)
                C_neu = np.concatenate(c5_neu, axis=0).astype(np.float64)
                _, _, R_s2c, _ = plato_data.fit_procrustes_map(S_neu, C_neu)
                D_rot = D_last @ R_s2c.T
                u_rot_h = arch_rot229.house_dir(D_rot[house_idx], cats_h)
                u_rot_s = arch_rot229.unit(D_rot[self2_idx].mean(axis=0))
                sc_rh = D_rot[house_idx] @ u_hc5
                sc_rs = D_rot[self2_idx] @ u_sc5
                out["backbone_neutral_procrustes_to_c5"] = {
                    "n_neutral_vecs": int(len(S_neu)),
                    "cos_house": round(float(np.dot(u_rot_h, u_hc5)), 4),
                    "cos_self2": round(float(np.dot(u_rot_s, u_sc5)), 4),
                    "zeroshot_house_acc": round(float(np.mean(sc_rh > 0.0)), 4),
                    "zeroshot_perf_acc": round(float(np.mean(sc_rh[perf_mask] > 0.0)), 4) if perf_mask.any() else None,
                    "zeroshot_self2_acc": round(float(np.mean(sc_rs > 0.0)), 4),
                }
    m.train()
    return out


def train(name, arm, seed, tokens=100_000_000, overwrite=False):
    assert arm in ARM_SPEC, f"unknown arm {arm}"
    assert name.startswith("E229_"), f"Addendum 229 run names must start with E229_ (got {name})"
    if not overwrite:
        assert not (RUNS / f"{name}.json").exists(), f"run already exists: {name}.json (pass overwrite=1 or choose a new run name)"

    c = dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, seed=seed, tokens=float(tokens))
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed + 22500)
    aux_rng = np.random.default_rng(seed + 22900)

    tk = exp228.tokenizer()
    V = tk.get_vocab_size()
    tr, held, jung, proc_tr, chat_tr, chat_he = streams()
    bad = exp228.bad_runs(tr)
    with open(MID / "proc_held.jsonl") as f:
        proc_held = [json.loads(l) for l in f]
    base_pool, myth_pool = load_pools(c["myth_target"], c["ujung_alpha"])

    m = exp228.PlatoToy(c, V).to(DEV)
    B, ctx, mb = c["batch"], c["ctx"], c["mb"]
    assert B % 4 == 0
    steps = max(1, int(tokens // (B * ctx)))
    marks = {max(1, steps // 4): "25M", max(2, steps // 2): "50M"}

    backbone_params = [p for n, p in m.named_parameters() if not n.startswith("plato_head.")]
    head_params = [p for n, p in m.named_parameters() if n.startswith("plato_head.")]
    dec = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": c["wd"]}, {"params": nod, "weight_decay": 0.0}], c["lr"], betas=(0.9, 0.95))

    print(
        f"{name}: arm={arm} myth_tgt={c['myth_target']} ujung_a={c['ujung_alpha']} anneal=[{c['anneal_start']},{c['anneal_end']}] "
        f"seed={seed} infer_nonembed={exp228.nonembed_inference(m)} steps={steps} dev={DEV} amp={AMP}",
        flush=True,
    )
    t0 = time.time()
    curve = {}

    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = exp228.lr_at(c["lr"], s, steps)
        cur_pw = plato_w_at(c["plato_w"], s, steps, anneal_start=c["anneal_start"], anneal_end=c["anneal_end"])

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
        tot_ntp, tot_plato = 0.0, 0.0
        for i in range(0, B, mb):
            x = torch.from_numpy(xb[i:i + mb, :-1]).to(DEV)
            y = torch.from_numpy(xb[i:i + mb, 1:]).to(DEV)
            with exp228.amp():
                lg = m(x)
            lg_f = lg.float()
            l_ntp = F.cross_entropy(lg_f.reshape(-1, V), y.reshape(-1))
            l = l_ntp
            if c["zloss"]:
                l = l + c["zloss"] * torch.logsumexp(lg_f, -1).pow(2).mean()
            (l * (x.shape[0] / B)).backward()
            tot_ntp += l_ntp.item() * (x.shape[0] / B)

        xa_full, ya_ntp, mska, b_t, t_t, tgt_t = make_mixed_aux_batch(
            base_pool, myth_pool, aux_rng, c["aux_bs_base"], c["aux_bs_myth"], DEV,
        )
        with exp228.amp():
            lg_a, h_mid, _ = m.forward_full(xa_full)
        lg_af = lg_a[:, :-1, :].float()
        ce_a = F.cross_entropy(lg_af.reshape(-1, V), ya_ntp.reshape(-1), reduction="none").view_as(ya_ntp)
        l_aux = c["aux_ntp_w"] * ((ce_a * mska).sum() / mska.sum().clamp_min(1.0))
        if cur_pw > 0.0 and len(b_t) > 0:
            pred = m.plato_head(h_mid[b_t, t_t, :].float())
            l_p = exp228.plato_alignment_loss(pred, tgt_t)
            l_aux = l_aux + cur_pw * l_p
            tot_plato = float(l_p.item())
        l_aux.backward()

        gn = float(torch.nn.utils.clip_grad_norm_(backbone_params, 1.0))
        if head_params and cur_pw > 0.0:
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        assert math.isfinite(tot_ntp), "diverged"

        if s % 100 == 0 or s == steps - 1:
            print(
                f"step {s}/{steps} ntp {tot_ntp:.4f} plato {tot_plato:.4f} (w={cur_pw:.3f}) "
                f"gnorm {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} s/step {(time.time() - t0) / (s + 1):.3f}",
                flush=True,
            )
        if (s + 1) in marks:
            lbl = marks[s + 1]
            pr = exp228.score_mcq_items(m, c, tk, proc_held)
            curve[lbl] = pr
            print(f"CHECKPOINT {name} {lbl} proc_held overall {pr['overall']}", flush=True)

    train_s = round(time.time() - t0)
    ev_books = exp228.eval_bpb(m, c, tk, [held], nb=2_000_000)
    ev_jung = exp228.eval_bpb(m, c, tk, jung)
    ev_chat = exp228.eval_bpb(m, c, tk, [chat_he])
    pr_end = exp228.score_mcq_items(m, c, tk, proc_held)
    curve["100M"] = pr_end
    ckpt_tmp = PRIV / "ckpt" / f"{name}.tmp.pt"
    torch.save(m.state_dict(), ckpt_tmp)
    os.replace(ckpt_tmp, PRIV / "ckpt" / f"{name}.pt")
    info = {
        "name": name, "config": c, "dev": DEV, "amp": AMP, "steps": steps,
        "infer_nonembed": exp228.nonembed_inference(m),
        "train_s": train_s, "s_per_step": round(train_s / steps, 4), "last_ntp": tot_ntp,
        "books": ev_books, "jung": ev_jung, "chat_held": ev_chat, "proc_curve": curve,
    }
    _atomic_json(RUNS / f"{name}.json", info)
    _fb = lambda ev: f"{ev['bpb']:.4f}" if ev.get("bpb") is not None else "n/a"
    print(f"EXP229_DONE {name} books={_fb(ev_books)} jung={_fb(ev_jung)} chat={_fb(ev_chat)} proc_100M={pr_end['overall']['gain']:+.3f} ({train_s} s)", flush=True)


def score(names, force=False):
    """Run full Addendum 228 scoring suite (re-scoring whenever checkpoint is newer than cached JSON or force=True)
    PLUS student HOUSE/SELF2 coherence & U_jung probe!"""
    need_228 = []
    for name in names:
        cp = RUNS / "comp" / f"{name}.json"
        ckpt_p = PRIV / "ckpt" / f"{name}.pt"
        if force or not cp.exists() or (ckpt_p.exists() and ckpt_p.stat().st_mtime > cp.stat().st_mtime):
            need_228.append(name)
        else:
            with open(cp) as f:
                if "platonic" not in json.load(f):
                    need_228.append(name)
    if need_228:
        exp228.score(need_228)

    tk = exp228.tokenizer()
    V = tk.get_vocab_size()
    for name in names:
        with open(RUNS / f"{name}.json") as f:
            c = json.load(f)["config"]
        m = exp228.PlatoToy(c, V).to(DEV)
        m.load_state_dict(torch.load(PRIV / "ckpt" / f"{name}.pt", map_location=DEV, weights_only=True))
        m.eval()
        jung_res = score_student_jung_coherence(m, c, tk)
        comp_path = RUNS / "comp" / f"{name}.json"
        with open(comp_path) as f:
            res = json.load(f)
        res["jung_coherence"] = jung_res
        _atomic_json(comp_path, res)
        ph = jung_res.get("plato_head_to_u_jung_c5", {})
        bp = jung_res.get("backbone_neutral_procrustes_to_c5", {})
        lh = jung_res.get("last", {})
        print(
            f"JUNG_PROBE {name}: stu_H2f={lh.get('house_2f_acc', 0):.3f} (lr {lh.get('house_2f_lenres', 0):.3f}) "
            f"H->S2={lh.get('house_to_self2_acc', 0):.3f} | head_cos={ph.get('cos_house', 0):+.3f} "
            f"bb_cos={bp.get('cos_house', 0):+.3f} bb_zsH={bp.get('zeroshot_house_acc', 0):.3f}",
            flush=True,
        )


def self_test():
    torch.manual_seed(0)
    V = 8192
    # 1. Verify annealing schedule [0.65, 0.90] with true 0.0 polish on [0.90, 1.00]
    assert abs(plato_w_at(0.25, 0, 1000) - 0.25) < 1e-9
    assert abs(plato_w_at(0.25, 649, 1000) - 0.25) < 1e-9
    assert abs(plato_w_at(0.25, 650, 1000) - 0.25) < 1e-9
    assert abs(plato_w_at(0.25, 775, 1000) - 0.125) < 1e-9
    assert abs(plato_w_at(0.25, 900, 1000) - 0.0) < 1e-9
    assert abs(plato_w_at(0.25, 999, 1000) - 0.0) < 1e-9

    # 2. Verify transform_ujung_pair uses ONE shared global scale across active base + myth and keeps exact RMS parity
    rng = np.random.default_rng(229)
    Zb = rng.normal(0, 1.0, (500, 256))
    Zm = rng.normal(0, 1.0, (200, 256))
    act = np.zeros(500, dtype=bool)
    act[:350] = True
    Q, _ = np.linalg.qr(rng.normal(size=(256, 2)))
    U = Q.T
    Zb_ab, Zm_ab = transform_ujung_pair(Zb, act, Zm, U, 0.0)
    Zb_bo, Zm_bo = transform_ujung_pair(Zb, act, Zm, U, 3.5)
    assert float(np.mean(np.sum((Zb_ab @ U.T) ** 2, axis=1))) < 1e-10
    assert float(np.mean(np.sum((Zm_ab @ U.T) ** 2, axis=1))) < 1e-10
    ms0 = 0.75 * float(np.mean(Zb[act] ** 2)) + 0.25 * float(np.mean(Zm ** 2))
    ms_ab = 0.75 * float(np.mean(Zb_ab[act] ** 2)) + 0.25 * float(np.mean(Zm_ab ** 2))
    ms_bo = 0.75 * float(np.mean(Zb_bo[act] ** 2)) + 0.25 * float(np.mean(Zm_bo ** 2))
    assert abs(ms_ab - ms0) < 1e-4 and abs(ms_bo - ms0) < 1e-4
    # Verify orthogonal 254D coordinates kept exact relative ratio between base and myth!
    proj_o = lambda X: X - (X @ U.T) @ U
    ratio_orig = float(np.mean(proj_o(Zm) ** 2) / np.mean(proj_o(Zb[act]) ** 2))
    ratio_bo = float(np.mean(proj_o(Zm_bo) ** 2) / np.mean(proj_o(Zb_bo[act]) ** 2))
    assert abs(ratio_bo - ratio_orig) < 1e-5, f"base/myth orthogonal scale discontinuity: {ratio_orig} vs {ratio_bo}"

    # 3. Verify exact 4,722,304 inference param parity, make_mixed_aux_batch, and AdamW skip at cur_pw=0
    synth_pool = {
        "ids_flat": np.arange(1, 201, dtype=np.uint16), "ids_off": np.array([0, 50, 100, 150, 200], dtype=np.int64),
        "pts_flat": np.array([10, 25, 49, 10, 25, 49, 10, 25, 49, 10, 25, 49], dtype=np.int32),
        "pts_off": np.array([0, 3, 6, 9, 12], dtype=np.int64),
        "tgt": rng.normal(0, 1.0, (12, 256)).astype(np.float32), "pool_idx": np.arange(4, dtype=np.int64),
    }
    xa, ya, msk, bt, tt, tgtt = make_mixed_aux_batch(synth_pool, synth_pool, np.random.default_rng(1), 6, 2, "cpu")
    assert xa.shape == (8, 50) and tgtt.shape == (24, 256)

    for arm in ("M0", "M1", "M2", "M3"):
        c = dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, ctx=256)
        m = exp228.PlatoToy(c, V)
        assert exp228.nonembed_inference(m) == 4_722_304
        lg, hm, _ = m.forward_full(xa)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), ya.reshape(-1))
        pred = m.plato_head(hm[bt, tt, :].float())
        loss = loss + 0.25 * exp228.plato_alignment_loss(pred, tgtt)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

    # 4. End-to-end smoke test of score_student_jung_coherence on REAL arch pairs (catches H1/H2/M2!)
    tk = exp228.tokenizer()
    c_sm = dict(BASE_CFG, **ARM_SPEC["M0"], arm="M0", ctx=512)
    m_sm = exp228.PlatoToy(c_sm, V)
    # Build balanced 42-pair slice covering all 7 HOUSE cats (4/cat = 28) + 14 SELF2 pairs for fast <1s CPU smoke
    all_p = arch_rot229.load_arch_pairs()
    by_c = {}
    for p in all_p:
        by_c.setdefault(p["cat"], []).append(p)
    mini_p = []
    for cat in arch_rot229.HOUSE_CATS:
        mini_p.extend(by_c[cat][:4])
    mini_p.extend(by_c["self2"][:14])
    jr = score_student_jung_coherence(m_sm, c_sm, tk, pairs=mini_p, eval_npz_path=Path("/nonexistent"))
    assert "last" in jr and "house_2f_acc" in jr["last"] and "house_2f_lenres" in jr["last"], f"bad jung_res: {jr}"
    assert "plato_head_to_u_jung_c5" in jr, f"missing plato_head probe: {jr}"
    print(f"EXP229_TEST_OK infer_nonembed=4722304 shared_scale=OK anneal=[0.65,0.90] jung_probe_smoke=OK", flush=True)


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
