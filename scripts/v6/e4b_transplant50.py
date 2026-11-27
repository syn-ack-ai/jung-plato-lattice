"""Experiment 251: 50%-Depth Cross-Family Platonic Consensus Transplant onto Gemma 4 E4B (Aion 8B Backbone).

Strictly paired causal experiment updating ONLY full-rank layers 19-20 MLPs (157,286,400 params,
fp32 master during AdamW training cast back to bf16 after; training-only 2560->256 plato_head
dropped at inference -> ZERO extra inference parameters!) targeting layer 20 (20/41 = 48.8%,
matching `raw50_gemma4_e4b.npz` and `depth240_gemma4_e4b.npz` slot 2) on strictly NEUTRAL
`gut + proc + chat` passages (ZERO task labels, ZERO bm_actor2x2_247, ZERO HOUSE, ZERO aft.csv!).

Four arms (strictly matched seed=251, steps=450, identical 4 neutral passages/step, identical
0.25*CE + 1.5*KL anchored to untouched Base Gemma 4 E4B logits via chunked lm_head):
  - base   : Untouched Base Gemma 4 E4B
  - ctrl50 : plato_w = 0.0 (matched 0.25*CE + 1.5*KL control on L19-20 MLPs)
  - c3_50  : plato_w = 0.40 -> 0.0 targeting RIVAL-ONLY C3_50 (Qwen3.5 + Llama3.1 + Mistral;
             ZERO Gemma 4 in teacher, ZERO Phi-4-mini, ZERO OLMo 3 -> zero self-distillation!)
  - c4_50  : plato_w = 0.40 -> 0.0 targeting FULL C4_50 (Gemma4 + Qwen3.5 + Llama3.1 + Mistral;
             ZERO Phi-4-mini, ZERO OLMo 3)

Runs 100% locally in MLX on Apple Silicon (zero interference with any running daemon).

Usage:
  python e4b_transplant50.py test
  python e4b_transplant50.py build_targets
  python e4b_transplant50.py train ctrl50|c3_50|c4_50
  python e4b_transplant50.py extract_eval base|ctrl50|c3_50|c4_50
  python e4b_transplant50.py analyze
  python e4b_transplant50.py run_all
"""
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = Path(os.environ.get("EXP251_DIR", os.environ.get("JPL_EXP251_DIR", str(SCRATCH_DIR / "exp251_e4b_c4_50")))).resolve()
SLICE_DIR = EXP_DIR / "slices"
EVAL_DIR = EXP_DIR / "eval_cache"

REPO_ARCH = SCRIPT_DIR.parents[1] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(REPO_FLUENT / "plato"))).resolve()
D50_CACHE = PLATO_CACHE / "depth50_distill"
DEPTH_DIR = Path(os.environ.get("EXP251_DEPTH_DIR", os.environ.get("JPL_DEPTH_DIR", str(PLATO_CACHE / "depth240")))).resolve()
PROC_HELD_PATH = Path(os.environ.get("JPL_PROC_HELD_PATH", str(REPO_FLUENT / "mid" / "proc_held.jsonl")))
MODEL_PATH = os.environ.get("GEMMA4_MLX_MODEL", "mlx-community/gemma-4-E4B-it-bf16")

for _p in (str(REPO_ARCH), str(REPO_FLUENT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_dirs():
    for _d in (EXP_DIR, SLICE_DIR, EVAL_DIR):
        _d.mkdir(parents=True, exist_ok=True)


def _targets_path():
    p = EXP_DIR / "targets_e4b_50.npz"
    if p.exists():
        return p
    alt = EXP_DIR / "data" / "targets_e4b_50.npz"
    return alt if alt.exists() else p

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import exp245_binding as E245  # noqa: E402
import exp247_actor_binding as E247  # noqa: E402

C3_KEYS = ("qwen35_9b", "llama31_8b", "mistral7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
BLIND_KEYS = ("phi4_mini", "olmo3_7b")
ALL_6 = C4_KEYS + BLIND_KEYS

PLATO50_LAYER = 20          # exact int(round(0.50 * 41)) == 20 (matches raw50 & depth240 slot 2!)
MLP_LAYERS = (19, 20)       # 157,286,400 full-rank params directly feeding layer 20
STAGE_A_LAYER = 12          # 30% depth sanity invariance check


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, p)


def _plato_w_at(base_w, step, total_steps, anneal_start=0.65, anneal_end=0.90):
    if base_w <= 0.0:
        return 0.0
    frac = step / max(1, total_steps)
    if frac <= anneal_start:
        return float(base_w)
    if frac >= anneal_end:
        return 0.0
    t = (frac - anneal_start) / max(1e-9, anneal_end - anneal_start)
    return float(base_w * 0.5 * (1.0 + math.cos(math.pi * t)))


# ---------------------------------------------------------------- Step 1: Build Rival-Only C3_50 and Full C4_50 Targets

def build_targets(d_plato=256):
    """Build exact Gemma-4-tokenized neutral training & eval packages for BOTH C3_50 (ZERO Gemma!)
    and C4_50, plus blind unseen Phi-4-mini & OLMo 3 7B ground-truth eval targets."""
    from mlx_lm import load
    t0 = time.time()
    _, tok = load(MODEL_PATH)
    g_enc = lambda s: PD._text_ids_mlx(tok, s)

    train_rows = [json.loads(l) for l in open(PLATO_CACHE / "passages.jsonl")]
    eval_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    tr_neut_idx = [i for i, r in enumerate(train_rows) if r["domain"] in ("gut", "proc", "chat")]

    raw50 = {k: dict(np.load(D50_CACHE / f"raw50_{k}.npz")) for k in ALL_6}
    for k in ALL_6:
        assert len(raw50[k]["tr_cnt"]) == len(tr_neut_idx), f"tr_cnt mismatch for {k}"
        assert len(raw50[k]["ev_cnt"]) == len(eval_rows), f"ev_cnt mismatch for {k}"
    assert int(raw50["gemma4_e4b"]["layer_idx"]) == PLATO50_LAYER

    off50_tr = {k: np.concatenate([[0], np.cumsum(raw50[k]["tr_cnt"])]) for k in ALL_6}
    off50_ev = {k: np.concatenate([[0], np.cumsum(raw50[k]["ev_cnt"])]) for k in ALL_6}

    tr_ids, tr_pts, tr_doms, tr_orig_idx = [], [], [], []
    tr_sl = {k: [] for k in ALL_6}
    for local_i, orig_i in enumerate(tr_neut_idx):
        r = train_rows[orig_i]
        n_seg = len(r["segs"])
        g_ids, g_pts = PD.boundary_token_indices(g_enc, r["segs"])
        ok = (
            2 <= len(g_ids) <= 512 and len(g_pts) == n_seg
            and all(bool(raw50[k]["tr_ok"][local_i]) and int(raw50[k]["tr_cnt"][local_i]) == n_seg for k in ALL_6)
        )
        if ok:
            tr_ids.append(g_ids); tr_pts.append(g_pts); tr_doms.append(r["domain"]); tr_orig_idx.append(orig_i)
            for k in ALL_6:
                tr_sl[k].append(raw50[k]["tr_pts"][off50_tr[k][local_i]:off50_tr[k][local_i + 1]])

    ev_ids, ev_pts, ev_orig_idx, ev_pt_groups, ev_pt_doms, ev_pt_ret = [], [], [], [], [], []
    ev_sl = {k: [] for k in ALL_6}
    for r_i, r in enumerate(eval_rows):
        n_seg = len(r["segs"])
        g_ids, g_pts = PD.boundary_token_indices(g_enc, r["segs"])
        ok = (
            2 <= len(g_ids) <= 512 and len(g_pts) == n_seg
            and all(bool(raw50[k]["ev_ok"][r_i]) and int(raw50[k]["ev_cnt"][r_i]) == n_seg for k in ALL_6)
        )
        if ok:
            ev_ids.append(g_ids); ev_pts.append(g_pts); ev_orig_idx.append(r_i)
            is_ret = (r.get("role", "") != "foil")
            for _ in range(n_seg):
                ev_pt_groups.append(r.get("group_id", str(r_i)))
                ev_pt_doms.append(r.get("domain", ""))
                ev_pt_ret.append(is_ret)
            for k in ALL_6:
                ev_sl[k].append(raw50[k]["ev_pts"][off50_ev[k][r_i]:off50_ev[k][r_i + 1]])

    print(f"E4B_50 ALIGNED: train={len(tr_ids)}/{len(tr_neut_idx)} eval={len(ev_ids)}/{len(eval_rows)}", flush=True)

    # Strict train-fitted 256D PCA per model
    Y_tr, Y_ev = {}, {}
    for k in ALL_6:
        Y_tr[k], Y_ev[k], _ = PD.pca_project(
            np.concatenate(tr_sl[k], 0).astype(np.float64),
            np.concatenate(ev_sl[k], 0).astype(np.float64),
            d_plato,
        )

    assert "gemma4_e4b" not in C3_KEYS and not (set(C3_KEYS) & set(BLIND_KEYS))
    assert not (set(C4_KEYS) & set(BLIND_KEYS))
    C3_tr, C3_ev, _, _, _ = PD.generalized_procrustes_consensus([Y_tr[k] for k in C3_KEYS], [Y_ev[k] for k in C3_KEYS], n_iters=10)
    C4_tr, C4_ev, _, _, _ = PD.generalized_procrustes_consensus([Y_tr[k] for k in C4_KEYS], [Y_ev[k] for k in C4_KEYS], n_iters=10)

    diag = {"n_train_passages": len(tr_ids), "n_train_pts": int(len(C3_tr)), "n_eval_passages": len(ev_ids), "n_eval_pts": int(len(C3_ev)), "teachers_vs_blind": {}}
    for bk in BLIND_KEYS:
        diag["teachers_vs_blind"][bk] = {}
        for lbl, mat in (("base_gemma4", Y_ev["gemma4_e4b"]), ("C3_50_rival_only", C3_ev), ("C4_50_full", C4_ev), ("qwen35_9b", Y_ev["qwen35_9b"])):
            r = PD.eval_procrustes_transfer(mat, Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_doms, ret_mask=ev_pt_ret)
            diag["teachers_vs_blind"][bk][lbl] = r
            print(f"  -> {bk:10s} | {lbl:16s}: cos={r['cos']:.4f} r2_frob={r['r2_frob']:+.4f} r2_opt={r['r2_opt']:.4f} top1={r['top1']:.4f}", flush=True)

    def pack_seq(seq_list):
        off = np.zeros(len(seq_list) + 1, dtype=np.int64)
        for i, s in enumerate(seq_list):
            off[i + 1] = off[i] + len(s)
        flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in seq_list], axis=0)
        return flat, off

    tr_ids_f, tr_ids_o = pack_seq(tr_ids)
    tr_pts_f, tr_pts_o = pack_seq(tr_pts)
    ev_ids_f, ev_ids_o = pack_seq(ev_ids)
    ev_pts_f, ev_pts_o = pack_seq(ev_pts)

    PD._atomic_savez(
        EXP_DIR / "targets_e4b_50.npz",
        tr_ids_flat=tr_ids_f, tr_ids_off=tr_ids_o, tr_pts_flat=tr_pts_f, tr_pts_off=tr_pts_o,
        tr_domains=np.array(tr_doms), tr_orig_idx=np.array(tr_orig_idx, dtype=np.int32),
        c3_50_tr=C3_tr.astype(np.float32), c4_50_tr=C4_tr.astype(np.float32),
        ev_ids_flat=ev_ids_f, ev_ids_off=ev_ids_o, ev_pts_flat=ev_pts_f, ev_pts_off=ev_pts_o,
        ev_orig_idx=np.array(ev_orig_idx, dtype=np.int32),
        ev_pt_groups=np.array(ev_pt_groups), ev_pt_domains=np.array(ev_pt_doms), ev_pt_ret=np.array(ev_pt_ret, dtype=bool),
        c3_50_ev=C3_ev.astype(np.float32), c4_50_ev=C4_ev.astype(np.float32),
        phi4_50_ev=Y_ev["phi4_mini"].astype(np.float32), olmo3_50_ev=Y_ev["olmo3_7b"].astype(np.float32),
    )
    _atomic_json(EXP_DIR / "teacher50_diag.json", diag)
    print(f"BUILD_TARGETS_DONE in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- MLX Forward Helpers & Loss

def _mlx_fwd_capture(model, x, cap_layers=(PLATO50_LAYER,), stop_grad_before=None):
    """Exact Gemma 4 MLX forward pass matching `lm.__call__` while capturing hidden states at `cap_layers`.
    When `stop_grad_before` is set (e.g. 19), applies `mx.stop_gradient` to `h` and cached KVs right before
    that layer so frozen layers 0..18 incur zero backward overhead while keeping layers 19..41 100% exact!"""
    import mlx.core as mx
    inner = model.language_model.model
    inp_emb = inner.embed_tokens(x)
    h = inp_emb * inner.embed_scale
    if inner.hidden_size_per_layer_input:
        pli = inner._get_per_layer_inputs(x, inp_emb)
        pli = inner._project_per_layer_inputs(h, pli)
        pli = [pli[:, :, i, :] for i in range(len(inner.layers))]
    else:
        pli = [None] * len(inner.layers)

    cache = [None] * len(inner.layers)
    masks = inner._make_masks(h, cache)
    intermediates = [(None, None)] * len(inner.layers)
    caps = {}
    cap_set = set(cap_layers)

    for idx, (layer, c, mask, prev_idx, per_layer_input) in enumerate(
        zip(inner.layers, cache, masks, inner.previous_kvs, pli)
    ):
        if stop_grad_before is not None and idx == stop_grad_before:
            h = mx.stop_gradient(h)
        kvs, offset = intermediates[prev_idx]
        if stop_grad_before is not None and prev_idx < stop_grad_before and kvs is not None:
            kvs = (mx.stop_gradient(kvs[0]), mx.stop_gradient(kvs[1]))
        h, kvs, offset = layer(h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
        intermediates[idx] = (kvs, offset)
        if idx in cap_set:
            caps[idx] = h
    return inner.norm(h), caps


def _mlx_chunked_ce_kl(model, h_cur, y_ids, h_ref=None, ce_w=0.25, kl_w=1.5, chunk=32):
    """Exact chunked tied-embedding lm_head + softcap + CE + KL (caps peak vocab VRAM cleanly!)."""
    import mlx.core as mx
    lm_wrap = model.language_model
    emb = lm_wrap.model.embed_tokens
    softcap = float(lm_wrap.args.final_logit_softcapping)
    T = h_cur.shape[1]
    sum_ce = mx.array(0.0, dtype=mx.float32)
    sum_kl = mx.array(0.0, dtype=mx.float32)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        hc = h_cur[:, s:e, :]
        lg_c = emb.as_linear(hc).astype(mx.float32)
        lg_c = mx.tanh(lg_c / softcap) * softcap
        lse_c = mx.logsumexp(lg_c, axis=-1, keepdims=True)
        log_pc = lg_c - lse_c
        if y_ids is not None and ce_w > 0.0:
            yc = y_ids[:, s:e]
            ce_c = -mx.take_along_axis(log_pc, yc[:, :, None], axis=-1).squeeze(-1)
            sum_ce = sum_ce + mx.sum(ce_c)
        if h_ref is not None and kl_w > 0.0:
            hr = mx.stop_gradient(h_ref[:, s:e, :])
            lg_r = emb.as_linear(hr).astype(mx.float32)
            lg_r = mx.tanh(lg_r / softcap) * softcap
            log_pr = lg_r - mx.logsumexp(lg_r, axis=-1, keepdims=True)
            pr = mx.exp(log_pr)
            kl_c = mx.sum(pr * (log_pr - log_pc), axis=-1)
            sum_kl = sum_kl + mx.sum(kl_c)
    mean_ce = sum_ce / float(T)
    mean_kl = sum_kl / float(T)
    loss = ce_w * mean_ce + kl_w * mean_kl
    return loss, mean_ce, mean_kl


def _mlx_plato_alignment_loss(pred, tgt):
    """Bit-for-bit match to `exp228.plato_alignment_loss`: 0.5 * cosine_loss + 0.5 * RMS-normalized MSE."""
    import mlx.core as mx
    p = pred.astype(mx.float32)
    t = tgt.astype(mx.float32)
    pn = p / mx.sqrt(mx.sum(p * p, axis=-1, keepdims=True) + 1e-8)
    tn = t / mx.sqrt(mx.sum(t * t, axis=-1, keepdims=True) + 1e-8)
    l_cos = mx.mean(1.0 - mx.sum(pn * tn, axis=-1))
    p_rms = mx.sqrt(mx.mean(p * p) + 1e-8)
    t_rms = mx.sqrt(mx.mean(t * t) + 1e-8)
    l_mse = mx.mean((p / p_rms - t / t_rms) ** 2)
    return 0.5 * l_cos + 0.5 * l_mse


def _snapshot_mlp_weights(model, layers=MLP_LAYERS):
    from mlx.utils import tree_map
    inner = model.language_model.model
    return {li: tree_map(lambda a: a, inner.layers[li].mlp.parameters()) for li in layers}


def _apply_mlp_weights(model, snap):
    inner = model.language_model.model
    for li, params in snap.items():
        inner.layers[li].mlp.update(params)


def _save_mlx_slice(model, path, layers=MLP_LAYERS):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    inner = model.language_model.model
    flat = {}
    for li in layers:
        for k, v in tree_flatten(inner.layers[li].mlp.parameters()):
            flat[f"layers.{li}.mlp.{k}"] = v.astype(mx.bfloat16)
    tmp = Path(path).with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), flat)
    os.replace(tmp, path)


def _load_mlx_slice(model, path):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    flat = mx.load(str(path))
    by_l = defaultdict(list)
    for k, v in flat.items():
        # key format: layers.<li >.mlp.<subkey>
        parts = k.split(".", 3)
        li = int(parts[1])
        by_l[li].append((parts[3], v))
    inner = model.language_model.model
    for li, items in by_l.items():
        inner.layers[li].mlp.update(tree_unflatten(items))


def _export_pytorch_living_slice(safetensors_path, out_pt_path, meta):
    """Export an Option-1 Zero-LoRA bf16 PyTorch slice (`mlp_slices: {19: ..., 20: ...}`) compatible with
    `LivingE4B.load_living_slice` in `aion_8b/e4b_living.py` using ggufenv's torch if available."""
    import subprocess
    gguf_py = os.environ.get("JPL_TORCH_PYTHON", sys.executable)
    if not Path(gguf_py).exists():
        return False
    code = (
        "import json, os, torch; from safetensors.torch import load_file; "
        f"raw = load_file({str(safetensors_path)!r}); "
        "mlp_slices = {}; "
        "for k, v in raw.items(): "
        "  _, li, _, sub = k.split('.', 3); li = int(li); "
        "  mlp_slices.setdefault(li, {})[sub] = v.to(dtype=torch.bfloat16, device='cpu'); "
        f"payload = {{'model_id': 'google/gemma-4-E4B-it', 'mlp_layers': tuple(sorted(mlp_slices.keys())), 'mlp_slices': mlp_slices, 'meta': {json.dumps(meta)}}}; "
        f"tmp = {str(out_pt_path)!r} + '.tmp'; torch.save(payload, tmp); os.replace(tmp, {str(out_pt_path)!r}); "
        "print('PT_SLICE_OK', sorted(mlp_slices.keys()), {k: list(v.keys()) for k, v in mlp_slices.items()})"
    )
    r = subprocess.run([gguf_py, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        print(" ", r.stdout.strip(), flush=True)
        return True
    print("  PT slice export skipped/failed:", r.stderr.strip()[:200], flush=True)
    return False


# ---------------------------------------------------------------- Step 2: Strictly Paired Causal Training

ARM_CFG = {
    "ctrl50": dict(plato_w=0.0,  target_key=None),
    "c3_50":  dict(plato_w=0.40, target_key="c3_50_tr"),
    "c4_50":  dict(plato_w=0.40, target_key="c4_50_tr"),
}


def train_arm(arm, steps=450, lr_mlp=1.5e-5, lr_head=1e-3, ce_w=0.25, kl_w=1.5, n_acc=4, seed=251):
    assert arm in ARM_CFG, f"Unknown arm {arm}"
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load

    cfg = ARM_CFG[arm]
    out_sf = SLICE_DIR / f"{arm}_L19_20.safetensors"
    if out_sf.exists():
        print(f"TRAIN {arm}: slice {out_sf} already exists; skipping!", flush=True)
        return

    t0 = time.time()
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)

    d_t = np.load(_targets_path())
    ids_f, ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    pts_f, pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
    tgt_arr = d_t[cfg["target_key"]] if cfg["target_key"] else None
    n_tr = len(ids_o) - 1

    model, _ = load(MODEL_PATH)
    inner = model.language_model.model

    # Snapshot exact untouched bf16 reference weights for layers 19-20 (zero extra memory overhead!)
    ref_mlp_bf16 = {li: tree_map(lambda a: mx.array(a), inner.layers[li].mlp.parameters()) for li in MLP_LAYERS}
    mx.eval(ref_mlp_bf16)

    # Upcast ONLY L19-20 MLPs to fp32 master during AdamW training so lr=1.5e-5 never stalls under bf16 ULP!
    model.freeze()
    for li in MLP_LAYERS:
        inner.layers[li].mlp.apply(lambda w: w.astype(mx.float32))
        inner.layers[li].mlp.unfreeze()

    class TrainWrap(nn.Module):
        def __init__(self, m, use_head):
            super().__init__()
            self.m = m
            self.head = nn.Linear(2560, 256, bias=True) if use_head else None

    wrap = TrainWrap(model, use_head=(cfg["plato_w"] > 0.0))
    if wrap.head is not None:
        wrap.head.apply(lambda w: w.astype(mx.float32))
        wrap.head.unfreeze()

    n_mlp_p = sum(v.size for li in MLP_LAYERS for _, v in tree_flatten(inner.layers[li].mlp.trainable_parameters()))
    n_head_p = sum(v.size for _, v in tree_flatten(wrap.head.trainable_parameters())) if wrap.head is not None else 0
    assert n_mlp_p == 157_286_400, f"Unexpected MLP param count: {n_mlp_p}"
    print(f"TRAIN {arm} start: steps={steps} n_acc={n_acc} plato_w={cfg['plato_w']} target={cfg['target_key']} "
          f"mlp_params={n_mlp_p:,} head_params={n_head_p:,}", flush=True)

    opt_mlp = optim.AdamW(learning_rate=lr_mlp, betas=[0.9, 0.95], weight_decay=0.01)
    opt_head = optim.AdamW(learning_rate=lr_head, betas=[0.9, 0.95], weight_decay=0.0) if wrap.head is not None else None

    def single_passage_loss(w_mod, x, y, h_ref, pts_mx, tgt_mx, cur_pw):
        h_out, caps = _mlx_fwd_capture(w_mod.m, x, cap_layers=(PLATO50_LAYER,), stop_grad_before=min(MLP_LAYERS))
        l_ck, ce_v, kl_v = _mlx_chunked_ce_kl(w_mod.m, h_out[:, :-1, :], y, h_ref=h_ref, ce_w=ce_w, kl_w=kl_w, chunk=32)
        if cur_pw > 0.0 and w_mod.head is not None:
            pred = w_mod.head(caps[PLATO50_LAYER][0, pts_mx, :].astype(mx.float32))
            l_p = _mlx_plato_alignment_loss(pred, tgt_mx)
            tot = l_ck + cur_pw * l_p
        else:
            l_p = mx.array(0.0, dtype=mx.float32)
            tot = l_ck
        return tot, (ce_v, kl_v, l_p)

    vg_fn = nn.value_and_grad(wrap, single_passage_loss)

    history = []
    for s in range(steps):
        w_s = max(1, steps // 10)
        lr_mul = (s + 1) / w_s if s < w_s else 0.5 * (1.0 + math.cos(math.pi * (s - w_s) / max(1, steps - w_s)))
        opt_mlp.learning_rate = lr_mlp * lr_mul
        if opt_head is not None:
            opt_head.learning_rate = lr_head * lr_mul
        cur_pw = _plato_w_at(cfg["plato_w"], s, steps, anneal_start=0.65, anneal_end=0.90)

        # Identical passage draws across ALL 3 arms for identical seed=251!
        picks = rng.integers(0, n_tr, size=n_acc)
        batch = []
        for pi in picks:
            g_ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int32)
            pa, pb = int(pts_o[pi]), int(pts_o[pi + 1])
            g_pts = pts_f[pa:pb].astype(np.int32)
            g_tgt = tgt_arr[pa:pb] if tgt_arr is not None else None
            batch.append((g_ids, g_pts, g_tgt))

        # 1. Exact Base E4B reference hidden states via zero-copy parameter reference swap!
        cur_mlp_fp32 = _snapshot_mlp_weights(model, MLP_LAYERS)
        _apply_mlp_weights(model, ref_mlp_bf16)
        ref_hs = []
        for g_ids, _, _ in batch:
            x = mx.array(g_ids[None, :])
            h_out_r, _ = _mlx_fwd_capture(model, x, cap_layers=(), stop_grad_before=None)
            hr = mx.stop_gradient(h_out_r[:, :-1, :].astype(mx.bfloat16))
            mx.eval(hr)
            ref_hs.append(hr)
        _apply_mlp_weights(model, cur_mlp_fp32)
        del cur_mlp_fp32

        # 2. Accumulate gradients sequentially across n_acc passages (keeps peak unified RAM low!)
        acc_grads = None
        sum_ce = sum_kl = sum_plato = 0.0
        for (g_ids, g_pts, g_tgt), h_ref in zip(batch, ref_hs):
            x = mx.array(g_ids[None, :])
            y = x[:, 1:]
            pts_mx = mx.array(g_pts)
            tgt_mx = mx.array(g_tgt, dtype=mx.float32) if g_tgt is not None else mx.zeros((1, 256), dtype=mx.float32)
            (_, (ce_v, kl_v, lp_v)), grads = vg_fn(wrap, x, y, h_ref, pts_mx, tgt_mx, cur_pw)
            scale = 1.0 / float(n_acc)
            grads_s = tree_map(lambda g: g * scale, grads)
            acc_grads = grads_s if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads_s)
            mx.eval(acc_grads, ce_v, kl_v, lp_v)
            sum_ce += float(ce_v.item()) * scale
            sum_kl += float(kl_v.item()) * scale
            sum_plato += float(lp_v.item()) * scale
        del ref_hs

        # Split gradients cleanly for MLP vs training-only plato_head, clip each group to global norm 1.0
        mlp_grads = acc_grads["m"]
        mlp_gn = math.sqrt(sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for _, v in tree_flatten(mlp_grads)))
        if mlp_gn > 1.0:
            mlp_grads = tree_map(lambda g: g * (1.0 / mlp_gn), mlp_grads)
        opt_mlp.update(wrap.m, mlp_grads)

        if opt_head is not None and "head" in acc_grads and cur_pw > 0.0:
            h_grads = acc_grads["head"]
            h_gn = math.sqrt(sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for _, v in tree_flatten(h_grads)))
            if h_gn > 1.0:
                h_grads = tree_map(lambda g: g * (1.0 / h_gn), h_grads)
            opt_head.update(wrap.head, h_grads)

        mx.eval(wrap.trainable_parameters(), opt_mlp.state)
        if opt_head is not None:
            mx.eval(opt_head.state)
        del acc_grads, mlp_grads

        assert math.isfinite(sum_ce) and math.isfinite(sum_kl) and math.isfinite(sum_plato), f"Diverged at step {s}!"
        if s % 10 == 9:
            mx.clear_cache()
        if s % 50 == 0 or s == steps - 1:
            rec = {
                "step": s, "ce": round(sum_ce, 4), "kl": round(sum_kl, 5), "plato": round(sum_plato, 5),
                "w": round(cur_pw, 4), "gn": round(mlp_gn, 3),
                "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "sec": round(time.time() - t0, 1),
            }
            history.append(rec)
            print(f"  [{arm}] step {s:3d}/{steps} ce={rec['ce']:.3f} kl={rec['kl']:.4f} plato={rec['plato']:.4f} "
                  f"(w={rec['w']:.3f}) gn={rec['gn']:.2f} peak={rec['peak_gb']:.1f}GB ({rec['sec']:.0f}s)", flush=True)

    # Compute exact L2 weight drift vs untouched base before saving bf16 slice (and DROPPING plato_head!)
    deltas = {}
    for li in MLP_LAYERS:
        cur_flat = dict(tree_flatten(inner.layers[li].mlp.parameters()))
        ref_flat = dict(tree_flatten(ref_mlp_bf16[li]))
        d2 = sum(float(mx.sum((cur_flat[k].astype(mx.float32) - ref_flat[k].astype(mx.float32)) ** 2).item()) for k in ref_flat)
        r2 = sum(float(mx.sum(ref_flat[k].astype(mx.float32) ** 2).item()) for k in ref_flat)
        deltas[f"L{li}"] = {"l2_delta": round(math.sqrt(d2), 4), "rel_delta": round(math.sqrt(d2 / max(1e-12, r2)), 6)}

    _save_mlx_slice(model, out_sf, MLP_LAYERS)
    meta = {
        "arm": arm, "mlp_layers": list(MLP_LAYERS), "plato_layer": PLATO50_LAYER, "steps": steps,
        "seed": seed, "plato_w": cfg["plato_w"], "target_key": cfg["target_key"],
        "lr_mlp": lr_mlp, "ce_w": ce_w, "kl_w": kl_w, "n_acc": n_acc,
        "weight_deltas": deltas, "elapsed_sec": round(time.time() - t0, 1),
        "slice_sha256": _sha256(out_sf), "history": history,
    }
    _export_pytorch_living_slice(out_sf, SLICE_DIR / f"{arm}_L19_20_living_slice.pt", meta)
    _atomic_json(EXP_DIR / f"train_{arm}_log.json", meta)
    print(f"TRAIN_DONE {arm}: deltas={deltas} elapsed={meta['elapsed_sec']}s", flush=True)

    del wrap, model, opt_mlp, opt_head, ref_mlp_bf16
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------- Step 3: Extract Zero-Leakage Eval Activations & LM Guardrails

def extract_eval(arm):
    """Extract L20 (50% depth) hidden states on ALL neutral train/eval checkpoints + benchmarks + NLL + proc_held MCQ
    using ONLY the bf16 model (plato_head DROPPED -> 0 extra inference params!)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    acts_npz = EVAL_DIR / f"acts_{arm}.npz"
    lm_json = EVAL_DIR / f"lm_{arm}.json"
    if acts_npz.exists() and lm_json.exists():
        print(f"EXTRACT_EVAL {arm}: already cached, skipping!", flush=True)
        return

    t0 = time.time()
    mx.set_cache_limit(2 << 30)
    model, tok = load(MODEL_PATH)
    if arm != "base":
        sf = SLICE_DIR / f"{arm}_L19_20.safetensors"
        assert sf.exists(), f"Missing trained slice {sf}"
        _load_mlx_slice(model, sf)
    print(f"EXTRACT_EVAL {arm}: loaded model in {time.time() - t0:.1f}s", flush=True)

    d_t = np.load(_targets_path())

    # 1. Neutral train & eval checkpoint states at L20 (for 100% zero-leakage PCA & blind Phi-4 / OLMo 3 Procrustes!)
    #    Also capture L12 on first 16 eval passages to assert exact 0.0 upstream invariance!
    def run_packed_pts(ids_f, ids_o, pts_f, pts_o, label, check_l12=False):
        pts_list, st_list, l12_probe = [], [], []
        n_p = len(ids_o) - 1
        for i in range(n_p):
            ids = ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int32)
            pts = pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int32)
            caps_want = (STAGE_A_LAYER, PLATO50_LAYER) if (check_l12 and i < 16) else (PLATO50_LAYER,)
            _, caps = _mlx_fwd_capture(model, mx.array(ids[None, :]), cap_layers=caps_want)
            h20 = caps[PLATO50_LAYER][0].astype(mx.float32)
            sel = h20[mx.array(pts)]
            hm = mx.mean(h20[1:], axis=0) if h20.shape[0] > 1 else h20[0]
            if check_l12 and i < 16:
                h12_m = mx.mean(caps[STAGE_A_LAYER][0].astype(mx.float32), axis=0)
                mx.eval(sel, hm, h12_m)
                l12_probe.append(np.array(h12_m, dtype=np.float32))
            else:
                mx.eval(sel, hm)
            pts_list.append(np.array(sel, dtype=np.float32))
            st_list.append(np.array(hm, dtype=np.float32))
            if (i + 1) % 600 == 0:
                print(f"  [{arm}] {label}: {i + 1}/{n_p} ({time.time() - t0:.0f}s)", flush=True)
        return np.concatenate(pts_list, 0), np.stack(st_list, 0), (np.stack(l12_probe, 0) if l12_probe else None)

    tr_pts, _, _ = run_packed_pts(d_t["tr_ids_flat"], d_t["tr_ids_off"], d_t["tr_pts_flat"], d_t["tr_pts_off"], "tr_pts")
    ev_pts, ev_story, l12_probe = run_packed_pts(d_t["ev_ids_flat"], d_t["ev_ids_off"], d_t["ev_pts_flat"], d_t["ev_pts_off"], "ev_pts", check_l12=True)

    # 2. Neutral depth240 corpus (mean & last at L20 — exact match to depth240_gemma4_e4b.npz!)
    tr_rows = [json.loads(l) for l in open(PLATO_CACHE / "passages.jsonl")]
    neut_texts = [r["text"].strip() for r in tr_rows if r["domain"] in ("gut", "proc", "chat")]

    def extract_texts_ml(texts, label):
        ms, ls = [], []
        for i, txt in enumerate(texts):
            ids = PD._text_ids_mlx(tok, txt)
            _, caps = _mlx_fwd_capture(model, mx.array([ids]), cap_layers=(PLATO50_LAYER,))
            h = caps[PLATO50_LAYER][0].astype(mx.float32)
            hm = mx.mean(h[1:], axis=0) if h.shape[0] > 1 else h[0]
            hl = h[-1]
            mx.eval(hm, hl)
            ms.append(np.array(hm, dtype=np.float32))
            ls.append(np.array(hl, dtype=np.float32))
            if (i + 1) % 800 == 0:
                print(f"  [{arm}] {label}: {i + 1}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        return np.stack(ms, 0), np.stack(ls, 0)

    neut_mean, neut_last = extract_texts_ml(neut_texts, "neut_2394")

    # 3. Evaluative VAL5 & HOUSE/SELF2 + benchmarks (bm120, bm_rel242, bm_neut245, bm_actor2x2_247 with NLL!)
    house_pairs = AR.load_arch_pairs()
    sg, ov, _ = V234.load_valence_data()
    val_pairs = sg + ov
    bm120_rows = BM238.load_mirror_data()
    bmrel_rows = json.loads((EXP242_DIR / "bm_rel242_data.json").read_text())["bm_rel_120"]
    bmneut_rows = json.loads((EXP245_DIR / "bm_neut245_data.json").read_text())["bm_neut_120"]
    actor_rows = json.loads((EXP247_DIR / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]

    house_pos_m, house_pos_l = extract_texts_ml([p["done"].strip() for p in house_pairs], "house_pos")
    house_neg_m, house_neg_l = extract_texts_ml([p["neglected"].strip() for p in house_pairs], "house_neg")
    val_pos_m, val_pos_l = extract_texts_ml([p["pos"].strip() for p in val_pairs], "val_pos")
    val_neg_m, val_neg_l = extract_texts_ml([p["neg"].strip() for p in val_pairs], "val_neg")
    bm120_pos_m, bm120_pos_l = extract_texts_ml([r["pos"].strip() for r in bm120_rows], "bm120_pos")
    bm120_neg_m, bm120_neg_l = extract_texts_ml([r["neg"].strip() for r in bm120_rows], "bm120_neg")
    bmrel_pos_m, bmrel_pos_l = extract_texts_ml([r["pos"].strip() for r in bmrel_rows], "bmrel_pos")
    bmrel_neg_m, bmrel_neg_l = extract_texts_ml([r["neg"].strip() for r in bmrel_rows], "bmrel_neg")
    bmneut_pos_m, bmneut_pos_l = extract_texts_ml([r["pos"].strip() for r in bmneut_rows], "bmneut_pos")
    bmneut_neg_m, bmneut_neg_l = extract_texts_ml([r["neg"].strip() for r in bmneut_rows], "bmneut_neg")

    # bm_actor2x2_247 with exact whole-sentence AND disjoint-changed-token NLL (matching exp247_actor_binding.py!)
    ap_m, ap_l, an_m, an_l, ap_nll, an_nll, ap_chg, an_chg = [], [], [], [], [], [], [], []
    lm_wrap = model.language_model
    emb = lm_wrap.model.embed_tokens
    softcap = float(lm_wrap.args.final_logit_softcapping)

    def seq_nll_per_tok(ids):
        x = mx.array([ids])
        h_out, caps = _mlx_fwd_capture(model, x, cap_layers=(PLATO50_LAYER,))
        # Chunked exact per-token CE (length <= 512, fast!)
        hc = h_out[:, :-1, :]
        T = hc.shape[1]
        ces = []
        for s in range(0, T, 64):
            e = min(T, s + 64)
            lg = emb.as_linear(hc[:, s:e, :]).astype(mx.float32)
            lg = mx.tanh(lg / softcap) * softcap
            ce_c = nn.losses.cross_entropy(lg[0], x[0, 1 + s:1 + e], reduction="none")
            ces.append(ce_c)
        ce_all = mx.concatenate(ces, axis=0)
        h20 = caps[PLATO50_LAYER][0].astype(mx.float32)
        hm = mx.mean(h20[1:], axis=0) if h20.shape[0] > 1 else h20[0]
        hl = h20[-1]
        mx.eval(ce_all, hm, hl)
        return np.array(ce_all, dtype=np.float32), np.array(hm, dtype=np.float32), np.array(hl, dtype=np.float32)

    for r in actor_rows:
        p_txt, n_txt = r["pos"].strip(), r["neg"].strip()
        ids_p = PD._text_ids_mlx(tok, p_txt)
        ids_n = PD._text_ids_mlx(tok, n_txt)
        mp, mn = E247._disjoint_changed_token_masks(tok, p_txt, n_txt, ids_p, ids_n)
        ce_p, hmp, hlp = seq_nll_per_tok(ids_p)
        ce_n, hmn, hln = seq_nll_per_tok(ids_n)
        ap_m.append(hmp); ap_l.append(hlp); an_m.append(hmn); an_l.append(hln)
        ap_nll.append(float(ce_p.mean())); an_nll.append(float(ce_n.mean()))
        ap_chg.append(float(ce_p[mp].mean())); an_chg.append(float(ce_n[mn].mean()))

    # 4. Held-out LM Guardrails: held_gut NLL (exact 40 passages) + proc_held 400-item MCQ!
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    gut_sub = [r for r in ev_rows if r.get("domain") == "held_gut"][:40]
    gut_nlls = []
    for r in gut_sub:
        ids = PD._text_ids_mlx(tok, "".join(r["segs"]))[:512]
        if len(ids) >= 4:
            ce_all, _, _ = seq_nll_per_tok(ids)
            gut_nlls.append(float(ce_all.mean()))

    # proc_held MCQ (400 items, length-normalized mean log-prob on answer tokens, matching exp228.score_mcq_items!)
    proc_items = [json.loads(l) for l in open(PROC_HELD_PATH)]

    def score_continuation_mean_lp(prefix, cont):
        ids_pre = PD._text_ids_mlx(tok, prefix)
        ids_full = PD._text_ids_mlx(tok, prefix + cont)
        k = max(1, len(ids_full) - len(ids_pre))
        x = mx.array([ids_full])
        h_out, _ = _mlx_fwd_capture(model, x, cap_layers=())
        hc = h_out[:, -k - 1:-1, :]
        lg = emb.as_linear(hc).astype(mx.float32)
        lg = mx.tanh(lg / softcap) * softcap
        ce = nn.losses.cross_entropy(lg[0], x[0, -k:], reduction="mean")
        mx.eval(ce)
        return -float(ce.item())

    def mcq_credit(scores):
        b = max(scores)
        tied = [i for i, v in enumerate(scores) if v == b]
        return (1.0 / len(tied)) if 0 in tied else 0.0

    proc_by_kind = defaultdict(lambda: {"w": [], "wo": []})
    for idx_p, r in enumerate(proc_items):
        opts = [r["answer"]] + r["wrong"]
        pre_w = f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:"
        pre_wo = f"Question: {r['question']}\nAnswer:"
        sw = [score_continuation_mean_lp(pre_w, " " + o) for o in opts]
        swo = [score_continuation_mean_lp(pre_wo, " " + o) for o in opts]
        proc_by_kind[r["kind"]]["w"].append(mcq_credit(sw))
        proc_by_kind[r["kind"]]["wo"].append(mcq_credit(swo))
        if (idx_p + 1) % 200 == 0:
            print(f"  [{arm}] proc_held MCQ: {idx_p + 1}/{len(proc_items)} ({time.time() - t0:.0f}s)", flush=True)

    proc_res = {}
    all_w, all_wo = [], []
    for k, d in sorted(proc_by_kind.items()):
        all_w.extend(d["w"]); all_wo.extend(d["wo"])
        proc_res[k] = {"n": len(d["w"]), "with": round(float(np.mean(d["w"])), 4), "without": round(float(np.mean(d["wo"])), 4), "gain": round(float(np.mean(d["w"]) - np.mean(d["wo"])), 4)}
    proc_res["overall"] = {"n": len(all_w), "with": round(float(np.mean(all_w)), 4), "without": round(float(np.mean(all_wo)), 4), "gain": round(float(np.mean(all_w) - np.mean(all_wo)), 4)}

    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    PD._atomic_savez(
        acts_npz,
        tr_pts=pack(tr_pts), ev_pts=pack(ev_pts), ev_story=pack(ev_story), l12_probe=l12_probe.astype(np.float32),
        neut_mean=pack(neut_mean), neut_last=pack(neut_last),
        house_pos_mean=pack(house_pos_m), house_pos_last=pack(house_pos_l),
        house_neg_mean=pack(house_neg_m), house_neg_last=pack(house_neg_l),
        val_pos_mean=pack(val_pos_m), val_pos_last=pack(val_pos_l),
        val_neg_mean=pack(val_neg_m), val_neg_last=pack(val_neg_l),
        bm120_pos_mean=pack(bm120_pos_m), bm120_pos_last=pack(bm120_pos_l),
        bm120_neg_mean=pack(bm120_neg_m), bm120_neg_last=pack(bm120_neg_l),
        bmrel_pos_mean=pack(bmrel_pos_m), bmrel_pos_last=pack(bmrel_pos_l),
        bmrel_neg_mean=pack(bmrel_neg_m), bmrel_neg_last=pack(bmrel_neg_l),
        bmneut_pos_mean=pack(bmneut_pos_m), bmneut_pos_last=pack(bmneut_pos_l),
        bmneut_neg_mean=pack(bmneut_neg_m), bmneut_neg_last=pack(bmneut_neg_l),
        actor_pos_mean=pack(np.stack(ap_m, 0)), actor_pos_last=pack(np.stack(ap_l, 0)),
        actor_neg_mean=pack(np.stack(an_m, 0)), actor_neg_last=pack(np.stack(an_l, 0)),
        actor_pos_nll=np.array(ap_nll, dtype=np.float32), actor_neg_nll=np.array(an_nll, dtype=np.float32),
        actor_pos_chg_nll=np.array(ap_chg, dtype=np.float32), actor_neg_chg_nll=np.array(an_chg, dtype=np.float32),
        gut_nlls=np.array(gut_nlls, dtype=np.float32),
        proc_with_items=np.array(all_w, dtype=np.float32),
        proc_without_items=np.array(all_wo, dtype=np.float32),
    )
    lm_out = {
        "arm": arm,
        "held_gut_nll_mean": round(float(np.mean(gut_nlls)), 4),
        "held_gut_n": len(gut_nlls),
        "proc_held": proc_res,
        "actor_nll_whole_acc": round(float(np.mean(np.array(an_nll) > np.array(ap_nll))), 4),
        "actor_nll_changed_acc": round(float(np.mean(np.array(an_chg) > np.array(ap_chg))), 4),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    _atomic_json(lm_json, lm_out)
    print(f"EXTRACT_EVAL_DONE {arm}: gut_nll={lm_out['held_gut_nll_mean']} proc_with={proc_res['overall']['with']} "
          f"(gain={proc_res['overall']['gain']:+.4f}) actor_nll=({lm_out['actor_nll_whole_acc']}/{lm_out['actor_nll_changed_acc']}) "
          f"in {lm_out['elapsed_sec']}s", flush=True)
    del model, tok
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------- Step 4: Full Zero-Leakage Paired Analysis

def _paired_boot_ci(diff_per_unit, n_boot=2000, seed=2510):
    arr = np.asarray(diff_per_unit, dtype=np.float64)
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    bs = arr[idx].mean(axis=1)
    p_lo, p_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    b_lo, b_hi = 2.0 * obs - p_hi, 2.0 * obs - p_lo
    return {
        "obs": round(obs, 5),
        "ci95_pct": [round(p_lo, 5), round(p_hi, 5)],
        "ci95_basic": [round(b_lo, 5), round(b_hi, 5)],
    }


def _procrustes_oof_preds(X, Y, group_ids):
    """Exact 2-fold group-disjoint orthogonal Procrustes OOF predictions matching `plato_data.eval_procrustes_transfer`."""
    uniq = sorted(set(group_ids))
    g2f = {g: (i % 2) for i, g in enumerate(uniq)}
    folds = np.array([g2f[g] for g in group_ids], dtype=int)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    preds = np.zeros_like(Yc)
    for f in (0, 1):
        tr, te = (folds != f), (folds == f)
        M = Xc[tr].T @ Yc[tr]
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        preds[te] = Xc[te] @ R
    return preds, Yc


def _group_cos_per_unit(preds, Yc, group_ids):
    pn = preds / np.maximum(1e-12, np.linalg.norm(preds, axis=1, keepdims=True))
    yn = Yc / np.maximum(1e-12, np.linalg.norm(Yc, axis=1, keepdims=True))
    pt_cos = np.sum(pn * yn, axis=1)
    by_g = defaultdict(list)
    for g, c in zip(group_ids, pt_cos):
        by_g[g].append(float(c))
    keys = sorted(by_g.keys())
    return keys, np.array([np.mean(by_g[g]) for g in keys], dtype=np.float64)


def analyze_all(arms=("base", "ctrl50", "c3_50", "c4_50")):
    t0 = time.time()
    d_t = np.load(_targets_path(), allow_pickle=True)
    ev_groups = d_t["ev_pt_groups"].tolist()
    ev_doms = d_t["ev_pt_domains"].tolist()
    ev_ret = d_t["ev_pt_ret"]
    phi4_ev = d_t["phi4_50_ev"].astype(np.float64)
    olmo3_ev = d_t["olmo3_50_ev"].astype(np.float64)
    c3_ev = d_t["c3_50_ev"].astype(np.float64)
    c4_ev = d_t["c4_50_ev"].astype(np.float64)

    acts = {a: dict(np.load(EVAL_DIR / f"acts_{a}.npz")) for a in arms}
    lms = {a: json.loads((EVAL_DIR / f"lm_{a}.json").read_text()) for a in arms}

    # 1. Exact L12 invariance check!
    l12_invariance = {}
    base_l12 = acts["base"]["l12_probe"].astype(np.float64)
    for a in arms:
        diff = float(np.max(np.abs(acts[a]["l12_probe"].astype(np.float64) - base_l12)))
        l12_invariance[a] = diff
        assert diff < 1e-5, f"L12 changed in arm {a}: {diff}"

    # 2. Blind unseen-family 50%-depth Procrustes alignment + paired group bootstrap!
    blind_align = {}
    Z_ev_by_arm = {}
    grp_cos = {bk: {} for bk in BLIND_KEYS}
    for a in arms:
        _, Z_ev, _ = PD.pca_project(acts[a]["tr_pts"].astype(np.float64), acts[a]["ev_pts"].astype(np.float64), 256)
        Z_ev_by_arm[a] = Z_ev
        blind_align[a] = {
            "blind_phi4_50": PD.eval_procrustes_transfer(Z_ev, phi4_ev, group_ids=ev_groups, domains=ev_doms, ret_mask=ev_ret),
            "blind_olmo3_50": PD.eval_procrustes_transfer(Z_ev, olmo3_ev, group_ids=ev_groups, domains=ev_doms, ret_mask=ev_ret),
            "diag_c3_50_ev": PD.eval_procrustes_transfer(Z_ev, c3_ev, group_ids=ev_groups, domains=ev_doms, ret_mask=ev_ret),
            "diag_c4_50_ev": PD.eval_procrustes_transfer(Z_ev, c4_ev, group_ids=ev_groups, domains=ev_doms, ret_mask=ev_ret),
        }
        for bk, Yt in (("phi4_mini", phi4_ev), ("olmo3_7b", olmo3_ev)):
            pr, yc = _procrustes_oof_preds(Z_ev, Yt, ev_groups)
            _, gvals = _group_cos_per_unit(pr, yc, ev_groups)
            grp_cos[bk][a] = gvals

    blind_paired_deltas = {}
    for cmp_arm in ("c3_50", "c4_50"):
        if cmp_arm not in arms:
            continue
        blind_paired_deltas[f"{cmp_arm}_minus_ctrl50"] = {}
        blind_paired_deltas[f"{cmp_arm}_minus_base"] = {}
        for bk in BLIND_KEYS:
            bk_key = "blind_phi4_50" if bk == "phi4_mini" else "blind_olmo3_50"
            blind_paired_deltas[f"{cmp_arm}_minus_ctrl50"][bk] = {
                "delta_cos_pt": round(blind_align[cmp_arm][bk_key]["cos"] - blind_align["ctrl50"][bk_key]["cos"], 4),
                "delta_r2_frob": round(blind_align[cmp_arm][bk_key]["r2_frob"] - blind_align["ctrl50"][bk_key]["r2_frob"], 4),
                "delta_r2_opt": round(blind_align[cmp_arm][bk_key]["r2_opt"] - blind_align["ctrl50"][bk_key]["r2_opt"], 4),
                "delta_top1": round(blind_align[cmp_arm][bk_key]["top1"] - blind_align["ctrl50"][bk_key]["top1"], 4),
                "group_cos_paired_boot": _paired_boot_ci(grp_cos[bk][cmp_arm] - grp_cos[bk]["ctrl50"], n_boot=2000, seed=2511),
            }
            blind_paired_deltas[f"{cmp_arm}_minus_base"][bk] = {
                "delta_cos_pt": round(blind_align[cmp_arm][bk_key]["cos"] - blind_align["base"][bk_key]["cos"], 4),
                "delta_r2_frob": round(blind_align[cmp_arm][bk_key]["r2_frob"] - blind_align["base"][bk_key]["r2_frob"], 4),
                "delta_r2_opt": round(blind_align[cmp_arm][bk_key]["r2_opt"] - blind_align["base"][bk_key]["r2_opt"], 4),
                "delta_top1": round(blind_align[cmp_arm][bk_key]["top1"] - blind_align["base"][bk_key]["top1"], 4),
                "group_cos_paired_boot": _paired_boot_ci(grp_cos[bk][cmp_arm] - grp_cos[bk]["base"], n_boot=2000, seed=2512),
            }

    # 3. Zero-Task-Label 50%-Depth Relational & Compositional Probing (Within-Model Native & 256D + Cross-Family!)
    actor_rows = json.loads((EXP247_DIR / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    bm120_rows = BM238.load_mirror_data()
    bmrel_rows = json.loads((EXP242_DIR / "bm_rel242_data.json").read_text())["bm_rel_120"]
    bmneut_rows = json.loads((EXP245_DIR / "bm_neut245_data.json").read_text())["bm_neut_120"]
    ben_mask = np.array([r["actor_type"] == "benign" for r in actor_rows], dtype=bool)
    adv_mask = ~ben_mask

    sid_actor, sp_actor = E245._sid_and_splits(actor_rows, 24700)
    sid_bm120, sp_bm120 = E245._sid_and_splits(bm120_rows, 23800)
    sid_bmrel, sp_bmrel = E245._sid_and_splits(bmrel_rows, 24200)
    sid_bmneut, sp_bmneut = E245._sid_and_splits(bmneut_rows, 24500)

    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    self2_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] == "self2"])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    sg_rows, ov_rows, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])

    # Load frozen 5 rival models at slot=2 (50% depth) once!
    RIVALS = ("qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
    SLOT50 = 2
    r_d240 = {k: dict(np.load(DEPTH_DIR / f"depth240_{k}.npz")) for k in RIVALS}
    r_actor = {k: dict(np.load(DEPTH_DIR / f"bmactor247_{k}.npz")) for k in RIVALS}
    r_bmrel = {k: dict(np.load(DEPTH_DIR / f"bmrel242_{k}.npz")) for k in RIVALS}
    r_bmneut = {k: dict(np.load(DEPTH_DIR / f"bmneut245_{k}.npz")) for k in RIVALS}

    def eval_cv2_20split(Ds, Dt, Qs, Qt, splits, mask_ben=None, sid_i=None):
        Ds_p = V234._project_out_rows(Ds, Qs) if Qs is not None and Qs.shape[1] > 0 else Ds
        Dt_p = V234._project_out_rows(Dt, Qt) if Qt is not None and Qt.shape[1] > 0 else Dt
        sp_acc, sp_ben, sp_adv, sp_all4 = [], [], [], []
        oof_sum = np.zeros(len(Dt), dtype=np.float64)
        for fl in splits:
            oof = np.zeros(len(Dt), dtype=np.float64)
            for f in (0, 1):
                tr, te = (fl != f), (fl == f)
                u = AR.unit(Ds_p[tr].mean(0))
                if Qt is not None and Qt.shape[1] > 0:
                    u = V234._project_out(u, Qt)
                oof[te] = Dt_p[te] @ u
            oof_sum += (oof > 0).astype(np.float64)
            sp_acc.append(float(np.mean(oof > 0)))
            if mask_ben is not None:
                sp_ben.append(float(np.mean(oof[mask_ben] > 0)))
                sp_adv.append(float(np.mean(oof[~mask_ben] > 0)))
            if sid_i is not None:
                u_s = sorted(set(int(x) for x in sid_i))
                a4 = [all(oof[sid_i == s] > 0) for s in u_s]
                sp_all4.append(float(np.mean(a4)))
        item_rate = oof_sum / float(len(splits))
        out = {"acc": round(float(np.mean(sp_acc)), 4), "_item_rate": item_rate}
        if mask_ben is not None:
            out["benign"] = round(float(np.mean(sp_ben)), 4)
            out["adverse"] = round(float(np.mean(sp_adv)), 4)
        if sid_i is not None:
            out["all4"] = round(float(np.mean(sp_all4)), 4)
        return out

    def fit_rot_for_arm(arm_acts, pool_key, d_plato=256):
        """Strict neutral-text PCA + 4-family (or C3-anchored!) orthogonal Procrustes into shared 256D frame.
        Note: To evaluate cross-family transfer between frozen rivals and `Gemma_{arm}` cleanly WITHOUT shifting
        rival reference coordinates across arms, we anchor the 256D reference frame on the FROZEN 3-rival GPA
        consensus (`C3_neut`) and align EVERY model (all 5 frozen rivals AND `Gemma_{arm}`) strictly via
        neutral-passage orthogonal Procrustes to that identical frozen frame!"""
        mats_n = {}
        for k in RIVALS:
            mats_n[k] = r_d240[k][f"neut_{pool_key}"][SLOT50].astype(np.float64)
        mats_n["gemma_arm"] = arm_acts[f"neut_{pool_key}"].astype(np.float64)

        pca_p = {}
        Y_n = {}
        for k, M in mats_n.items():
            mu = M.mean(axis=0, keepdims=True)
            Mc = M - mu
            _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
            V = Vt[:d_plato].T
            Z = Mc @ V
            sc = float(np.sqrt(np.mean(np.sum(Z * Z, axis=1))) + 1e-12)
            Y_n[k] = Z / sc
            pca_p[k] = (mu, V, sc)

        # Frozen 3-rival consensus anchor (100% IDENTICAL across all 4 Gemma arms!)
        C_ref, _, _, _, _ = PD.generalized_procrustes_consensus([Y_n[k] for k in C3_KEYS], [Y_n[k] for k in C3_KEYS], n_iters=10)
        R_map = {}
        for k in list(RIVALS) + ["gemma_arm"]:
            M = Y_n[k].T @ C_ref
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R_map[k] = U @ Vt
        return {"pca": pca_p, "R": R_map}

    def map_vec(M, rot, k):
        mu, V, sc = rot["pca"][k]
        return (((M.astype(np.float64) - mu) @ V) / sc) @ rot["R"][k]

    probing = {}
    setting_rates_store = {}  # for paired 60-setting bootstrap CIs!

    for a in arms:
        aa = acts[a]
        probing[a] = {"within_native": {}, "within_256d": {}, "rival_to_gemma": {}, "gemma_to_rival": {}, "house_L20": {}}

        # HOUSE / SELF2 setting-disjoint 2-fold CV at L20 (last token, matching Stage A!)
        Dh_last = aa["house_pos_last"].astype(np.float64) - aa["house_neg_last"].astype(np.float64)
        fh = arch_rot229_folds = AR.make_stratified_2fold(house_cats)
        fs = AR.make_stratified_2fold(np.array(["self2"] * len(self2_idx)))
        rh = AR.eval_disjoint_2fold([Dh_last[house_idx]], Dh_last[house_idx], house_cats, fh, is_house=True, seed=2511)
        rs = AR.eval_disjoint_2fold([Dh_last[self2_idx]], Dh_last[self2_idx], np.array(["self2"] * len(self2_idx)), fs, is_house=False, seed=2512)
        u_h = AR.house_dir(Dh_last[house_idx], house_cats)
        probing[a]["house_L20"] = {
            "house_2f_acc": rh["acc"], "self2_2f_acc": rs["acc"],
            "house_to_self2_acc": round(float(np.mean((Dh_last[self2_idx] @ u_h) > 0.0)), 4),
        }

        for pool_key in ("mean", "last"):
            # Native full-dim (2560-D) within-model probing!
            Dh_nat = aa[f"house_pos_{pool_key}"][house_idx].astype(np.float64) - aa[f"house_neg_{pool_key}"][house_idx].astype(np.float64)
            Dv_nat = aa[f"val_pos_{pool_key}"].astype(np.float64) - aa[f"val_neg_{pool_key}"].astype(np.float64)
            b_nat, _, _ = E245._build_valence_erasure_bases(Dv_nat, Dh_nat, sg_doms, house_cats)
            Q5_nat = b_nat["val5"]
            D_act_nat = aa[f"actor_pos_{pool_key}"].astype(np.float64) - aa[f"actor_neg_{pool_key}"].astype(np.float64)
            D_nt_nat = aa[f"bmneut_pos_{pool_key}"].astype(np.float64) - aa[f"bmneut_neg_{pool_key}"].astype(np.float64)
            D_120_nat = aa[f"bm120_pos_{pool_key}"].astype(np.float64) - aa[f"bm120_neg_{pool_key}"].astype(np.float64)
            D_rel_nat = aa[f"bmrel_pos_{pool_key}"].astype(np.float64) - aa[f"bmrel_neg_{pool_key}"].astype(np.float64)

            wn_act = eval_cv2_20split(D_act_nat, D_act_nat, Q5_nat, Q5_nat, sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
            probing[a]["within_native"][pool_key] = {
                "bm_actor2x2_247": {k: v for k, v in wn_act.items() if not k.startswith("_")},
                "bm120": eval_cv2_20split(D_120_nat, D_120_nat, Q5_nat, Q5_nat, sp_bm120)["acc"],
                "bm_rel242": eval_cv2_20split(D_rel_nat, D_rel_nat, Q5_nat, Q5_nat, sp_bmrel)["acc"],
                "bm_neut245": eval_cv2_20split(D_nt_nat, D_nt_nat, Q5_nat, Q5_nat, sp_bmneut)["acc"],
            }
            if pool_key == "mean":
                setting_rates_store.setdefault("within_native_actor_mean", {})[a] = np.array([wn_act["_item_rate"][sid_actor == s].mean() for s in sorted(set(sid_actor))])

            # Shared 256-D neutral-Procrustes frame (anchored on frozen C3_neut!)
            rot = fit_rot_for_arm(aa, pool_key, d_plato=256)
            Z_act, Z_nt, Z_120, Z_rel, Q5 = {}, {}, {}, {}, {}
            for k in RIVALS:
                Z_act[k] = map_vec(r_actor[k][f"pos_{pool_key}"][SLOT50], rot, k) - map_vec(r_actor[k][f"neg_{pool_key}"][SLOT50], rot, k)
                Z_nt[k] = map_vec(r_bmneut[k][f"pos_{pool_key}"][SLOT50], rot, k) - map_vec(r_bmneut[k][f"neg_{pool_key}"][SLOT50], rot, k)
                Z_120[k] = map_vec(r_d240[k][f"bm120_pos_{pool_key}"][SLOT50], rot, k) - map_vec(r_d240[k][f"bm120_neg_{pool_key}"][SLOT50], rot, k)
                Z_rel[k] = map_vec(r_bmrel[k][f"pos_{pool_key}"][SLOT50], rot, k) - map_vec(r_bmrel[k][f"neg_{pool_key}"][SLOT50], rot, k)
                Dh_k = map_vec(r_d240[k][f"house_pos_{pool_key}"][SLOT50][house_idx], rot, k) - map_vec(r_d240[k][f"house_neg_{pool_key}"][SLOT50][house_idx], rot, k)
                Dv_k = map_vec(r_d240[k][f"val_pos_{pool_key}"][SLOT50], rot, k) - map_vec(r_d240[k][f"val_neg_{pool_key}"][SLOT50], rot, k)
                bk, _, _ = E245._build_valence_erasure_bases(Dv_k, Dh_k, sg_doms, house_cats)
                Q5[k] = bk["val5"]

            Z_act["gemma_arm"] = map_vec(aa[f"actor_pos_{pool_key}"], rot, "gemma_arm") - map_vec(aa[f"actor_neg_{pool_key}"], rot, "gemma_arm")
            Z_nt["gemma_arm"] = map_vec(aa[f"bmneut_pos_{pool_key}"], rot, "gemma_arm") - map_vec(aa[f"bmneut_neg_{pool_key}"], rot, "gemma_arm")
            Z_120["gemma_arm"] = map_vec(aa[f"bm120_pos_{pool_key}"], rot, "gemma_arm") - map_vec(aa[f"bm120_neg_{pool_key}"], rot, "gemma_arm")
            Z_rel["gemma_arm"] = map_vec(aa[f"bmrel_pos_{pool_key}"], rot, "gemma_arm") - map_vec(aa[f"bmrel_neg_{pool_key}"], rot, "gemma_arm")
            Dh_g = map_vec(aa[f"house_pos_{pool_key}"][house_idx], rot, "gemma_arm") - map_vec(aa[f"house_neg_{pool_key}"][house_idx], rot, "gemma_arm")
            Dv_g = map_vec(aa[f"val_pos_{pool_key}"], rot, "gemma_arm") - map_vec(aa[f"val_neg_{pool_key}"], rot, "gemma_arm")
            bg, _, _ = E245._build_valence_erasure_bases(Dv_g, Dh_g, sg_doms, house_cats)
            Q5["gemma_arm"] = bg["val5"]

            # Within-model 256D
            w256 = eval_cv2_20split(Z_act["gemma_arm"], Z_act["gemma_arm"], Q5["gemma_arm"], Q5["gemma_arm"], sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
            probing[a]["within_256d"][pool_key] = {k: v for k, v in w256.items() if not k.startswith("_")}

            # Cross-family: ALL 5 frozen rivals -> Gemma_{arm} AND Gemma_{arm} -> ALL 5 frozen rivals!
            r2g = {"per_source": {}, "summary": {}}
            g2r = {"per_target": {}, "summary": {}}
            r2g_cv_items, g2r_cv_items, r2g_zs_items, g2r_zs_items = {}, {}, {}, {}

            Dt_g_p = V234._project_out_rows(Z_act["gemma_arm"], Q5["gemma_arm"])
            u_nt_g = V234._project_out(AR.unit(Z_nt["gemma_arm"].mean(0)), Q5["gemma_arm"])

            for rk in RIVALS:
                # rival -> gemma
                cv_in = eval_cv2_20split(Z_act[rk], Z_act["gemma_arm"], Q5[rk], Q5["gemma_arm"], sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
                u_nt_r = V234._project_out(V234._project_out(AR.unit(Z_nt[rk].mean(0)), Q5[rk]), Q5["gemma_arm"])
                zs_in_sc = Dt_g_p @ u_nt_r
                r2g["per_source"][rk] = {
                    "actor_cv2": {k: v for k, v in cv_in.items() if not k.startswith("_")},
                    "zs_bmneut_to_actor": {
                        "acc": round(float(np.mean(zs_in_sc > 0)), 4),
                        "benign": round(float(np.mean(zs_in_sc[ben_mask] > 0)), 4),
                        "adverse": round(float(np.mean(zs_in_sc[adv_mask] > 0)), 4),
                    },
                    "bm120_cv2": eval_cv2_20split(Z_120[rk], Z_120["gemma_arm"], Q5[rk], Q5["gemma_arm"], sp_bm120)["acc"],
                    "bmrel_cv2": eval_cv2_20split(Z_rel[rk], Z_rel["gemma_arm"], Q5[rk], Q5["gemma_arm"], sp_bmrel)["acc"],
                }
                r2g_cv_items[rk] = cv_in["_item_rate"]
                r2g_zs_items[rk] = (zs_in_sc > 0).astype(np.float64)

                # gemma -> rival
                cv_out = eval_cv2_20split(Z_act["gemma_arm"], Z_act[rk], Q5["gemma_arm"], Q5[rk], sp_actor, mask_ben=ben_mask, sid_i=sid_actor)
                Dt_r_p = V234._project_out_rows(Z_act[rk], Q5[rk])
                u_nt_go = V234._project_out(u_nt_g, Q5[rk])
                zs_out_sc = Dt_r_p @ u_nt_go
                g2r["per_target"][rk] = {
                    "actor_cv2": {k: v for k, v in cv_out.items() if not k.startswith("_")},
                    "zs_bmneut_to_actor": {
                        "acc": round(float(np.mean(zs_out_sc > 0)), 4),
                        "benign": round(float(np.mean(zs_out_sc[ben_mask] > 0)), 4),
                        "adverse": round(float(np.mean(zs_out_sc[adv_mask] > 0)), 4),
                    },
                }
                g2r_cv_items[rk] = cv_out["_item_rate"]
                g2r_zs_items[rk] = (zs_out_sc > 0).astype(np.float64)

            for grp_nm, keys_sub in (("all5", RIVALS), ("teacher3", C3_KEYS), ("blind2_phi4_olmo", BLIND_KEYS)):
                r2g["summary"][grp_nm] = {
                    "actor_cv2_acc": round(float(np.mean([r2g["per_source"][k]["actor_cv2"]["acc"] for k in keys_sub])), 4),
                    "actor_cv2_adv": round(float(np.mean([r2g["per_source"][k]["actor_cv2"]["adverse"] for k in keys_sub])), 4),
                    "actor_cv2_all4": round(float(np.mean([r2g["per_source"][k]["actor_cv2"]["all4"] for k in keys_sub])), 4),
                    "zs_bmneut_acc": round(float(np.mean([r2g["per_source"][k]["zs_bmneut_to_actor"]["acc"] for k in keys_sub])), 4),
                }
                g2r["summary"][grp_nm] = {
                    "actor_cv2_acc": round(float(np.mean([g2r["per_target"][k]["actor_cv2"]["acc"] for k in keys_sub])), 4),
                    "actor_cv2_adv": round(float(np.mean([g2r["per_target"][k]["actor_cv2"]["adverse"] for k in keys_sub])), 4),
                    "actor_cv2_all4": round(float(np.mean([g2r["per_target"][k]["actor_cv2"]["all4"] for k in keys_sub])), 4),
                    "zs_bmneut_acc": round(float(np.mean([g2r["per_target"][k]["zs_bmneut_to_actor"]["acc"] for k in keys_sub])), 4),
                }
                if pool_key == "mean":
                    avg_in = np.mean([r2g_cv_items[k] for k in keys_sub], axis=0)
                    avg_out = np.mean([g2r_cv_items[k] for k in keys_sub], axis=0)
                    avg_zsin = np.mean([r2g_zs_items[k] for k in keys_sub], axis=0)
                    avg_zsout = np.mean([g2r_zs_items[k] for k in keys_sub], axis=0)
                    for tag, vec in ((f"r2g_cv2_{grp_nm}", avg_in), (f"g2r_cv2_{grp_nm}", avg_out), (f"r2g_zs_{grp_nm}", avg_zsin), (f"g2r_zs_{grp_nm}", avg_zsout)):
                        setting_rates_store.setdefault(tag, {})[a] = np.array([vec[sid_actor == s].mean() for s in sorted(set(sid_actor))])

            probing[a]["rival_to_gemma"][pool_key] = r2g
            probing[a]["gemma_to_rival"][pool_key] = g2r

    # Paired 60-setting bootstrap deltas for probing & LM guardrails!
    probing_paired_deltas = {}
    for cmp_arm in ("c3_50", "c4_50"):
        if cmp_arm not in arms:
            continue
        probing_paired_deltas[f"{cmp_arm}_minus_ctrl50"] = {
            tag: _paired_boot_ci(by_a[cmp_arm] - by_a["ctrl50"], n_boot=2000, seed=2520)
            for tag, by_a in setting_rates_store.items()
        }
        probing_paired_deltas[f"{cmp_arm}_minus_base"] = {
            tag: _paired_boot_ci(by_a[cmp_arm] - by_a["base"], n_boot=2000, seed=2521)
            for tag, by_a in setting_rates_store.items()
        }
        # LM guardrail paired CIs (40 held_gut passages + 400 proc_held items)
        probing_paired_deltas[f"{cmp_arm}_minus_ctrl50"]["held_gut_nll_delta"] = _paired_boot_ci(
            acts[cmp_arm]["gut_nlls"].astype(np.float64) - acts["ctrl50"]["gut_nlls"].astype(np.float64), n_boot=2000, seed=2522
        )
        probing_paired_deltas[f"{cmp_arm}_minus_ctrl50"]["proc_held_with_delta"] = _paired_boot_ci(
            acts[cmp_arm]["proc_with_items"].astype(np.float64) - acts["ctrl50"]["proc_with_items"].astype(np.float64), n_boot=2000, seed=2523
        )

    out = {
        "script_sha256": _sha256(__file__),
        "targets_sha256": _sha256(_targets_path()),
        "plato50_layer": PLATO50_LAYER,
        "mlp_layers": list(MLP_LAYERS),
        "c3_teacher_families": list(C3_KEYS),
        "c4_teacher_families": list(C4_KEYS),
        "blind_unseen_families": list(BLIND_KEYS),
        "l12_invariance_max_abs_diff": l12_invariance,
        "lm_guardrails": lms,
        "blind_50pct_alignment": blind_align,
        "blind_50pct_paired_deltas": blind_paired_deltas,
        "probing_50pct": probing,
        "probing_50pct_paired_deltas": probing_paired_deltas,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    _atomic_json(EXP_DIR / "exp251_e4b_transplant50_results.json", out)
    print("\n=== EXP 251 SUMMARY ===", flush=True)
    for a in arms:
        p4 = blind_align[a]["blind_phi4_50"]
        ol = blind_align[a]["blind_olmo3_50"]
        r2g_b = probing[a]["rival_to_gemma"]["mean"]["summary"]["blind2_phi4_olmo"]
        g2r_b = probing[a]["gemma_to_rival"]["mean"]["summary"]["blind2_phi4_olmo"]
        nat = probing[a]["within_native"]["mean"]["bm_actor2x2_247"]
        print(
            f"  {a:7s} | Phi4(50%): cos={p4['cos']:.4f} R2f={p4['r2_frob']:+.4f} R2o={p4['r2_opt']:.4f} top1={p4['top1']:.4f} "
            f"| OLMo(50%): cos={ol['cos']:.4f} R2f={ol['r2_frob']:+.4f} R2o={ol['r2_opt']:.4f} top1={ol['top1']:.4f} "
            f"| gutNLL={lms[a]['held_gut_nll_mean']:.4f} proc={lms[a]['proc_held']['overall']['with']:.4f} "
            f"| natAct={nat['acc']:.4f} | blind2->G={r2g_b['actor_cv2_acc']:.4f}(zs={r2g_b['zs_bmneut_acc']:.4f}) "
            f"| G->blind2={g2r_b['actor_cv2_acc']:.4f}(zs={g2r_b['zs_bmneut_acc']:.4f})",
            flush=True,
        )
    print(f"SAVED {EXP_DIR / 'exp251_e4b_transplant50_results.json'} in {out['elapsed_sec']}s", flush=True)


# ---------------------------------------------------------------- Unit Self-Test

def self_test():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load

    # 1. Exact plato_alignment_loss parity check against numpy formula
    rng = np.random.default_rng(0)
    p_np = rng.normal(size=(8, 256)).astype(np.float32)
    t_np = rng.normal(size=(8, 256)).astype(np.float32)
    lp_mx = float(_mlx_plato_alignment_loss(mx.array(p_np), mx.array(t_np)).item())
    pn = p_np / np.sqrt((p_np ** 2).sum(-1, keepdims=True) + 1e-8)
    tn = t_np / np.sqrt((t_np ** 2).sum(-1, keepdims=True) + 1e-8)
    lp_np = 0.5 * float(np.mean(1.0 - (pn * tn).sum(-1))) + 0.5 * float(np.mean((p_np / np.sqrt(np.mean(p_np ** 2) + 1e-8) - t_np / np.sqrt(np.mean(t_np ** 2) + 1e-8)) ** 2))
    assert abs(lp_mx - lp_np) < 1e-6, f"Loss parity mismatch: {lp_mx} vs {lp_np}"

    # 2. Load real E4B, verify step-0 KL == 0, verify BOTH L19 & L20 MLP + head get finite non-zero grads,
    #    verify stop_gradient_before=19 matches full no-stop_grad forward output bit-for-bit!
    model, _ = load(MODEL_PATH)
    inner = model.language_model.model
    x = mx.array([[2] + list(range(100, 164))])
    y = x[:, 1:]
    h_full, c_full = _mlx_fwd_capture(model, x, cap_layers=(PLATO50_LAYER,), stop_grad_before=None)
    h_sg, c_sg = _mlx_fwd_capture(model, x, cap_layers=(PLATO50_LAYER,), stop_grad_before=min(MLP_LAYERS))
    mx.eval(h_full, h_sg, c_full[PLATO50_LAYER], c_sg[PLATO50_LAYER])
    fwd_err = float(mx.max(mx.abs(h_full.astype(mx.float32) - h_sg.astype(mx.float32))).item())
    assert fwd_err < 1e-5, f"stop_grad forward mismatch: {fwd_err}"

    _, ce0, kl0 = _mlx_chunked_ce_kl(model, h_full[:, :-1, :], y, h_ref=h_full[:, :-1, :], ce_w=0.25, kl_w=1.5, chunk=32)
    mx.eval(ce0, kl0)
    assert abs(float(kl0.item())) < 1e-5, f"Step-0 KL not zero: {float(kl0.item())}"

    model.freeze()
    for li in MLP_LAYERS:
        inner.layers[li].mlp.apply(lambda w: w.astype(mx.float32))
        inner.layers[li].mlp.unfreeze()

    class W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.head = nn.Linear(2560, 256)

    w = W(model)
    pts_mx = mx.array([10, 30, 50])
    tgt_mx = mx.array(rng.normal(size=(3, 256)).astype(np.float32))
    h_ref = mx.stop_gradient(h_full[:, :-1, :].astype(mx.bfloat16))

    def lfn(wm):
        ho, cs = _mlx_fwd_capture(wm.m, x, cap_layers=(PLATO50_LAYER,), stop_grad_before=min(MLP_LAYERS))
        lck, _, _ = _mlx_chunked_ce_kl(wm.m, ho[:, :-1, :], y, h_ref=h_ref, ce_w=0.25, kl_w=1.5, chunk=32)
        lp = _mlx_plato_alignment_loss(wm.head(cs[PLATO50_LAYER][0, pts_mx, :].astype(mx.float32)), tgt_mx)
        return lck + 0.4 * lp

    val, gr = nn.value_and_grad(w, lfn)(w)
    mx.eval(val, gr)
    flat_g = dict(tree_flatten(gr))
    l19_gn = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for k, v in flat_g.items() if "layers.19.mlp" in k)
    l20_gn = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for k, v in flat_g.items() if "layers.20.mlp" in k)
    hd_gn = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for k, v in flat_g.items() if k.startswith("head."))
    assert l19_gn > 0 and l20_gn > 0 and hd_gn > 0, f"Zero grad! l19={l19_gn} l20={l20_gn} head={hd_gn}"
    assert "gemma4_e4b" not in C3_KEYS
    print(f"EXP251_SELF_TEST_OK loss_parity={abs(lp_mx - lp_np):.1e} fwd_err={fwd_err:.1e} kl0={float(kl0.item()):.1e} "
          f"gn(L19,L20,head)=({math.sqrt(l19_gn):.3f},{math.sqrt(l20_gn):.3f},{math.sqrt(hd_gn):.3f})", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        if "--run-model-test" not in sys.argv[2:]:
            print("NOTE: `e4b_transplant50.py test` loads the 8B Gemma 4 MLX checkpoint (~16 GB download if not cached). Pass `--run-model-test` to execute it.")
            raise SystemExit(0)
        self_test()
    elif cmd == "build_targets":
        _ensure_dirs()
        build_targets()
    elif cmd == "train":
        _ensure_dirs()
        train_arm(sys.argv[2])
    elif cmd == "extract_eval":
        _ensure_dirs()
        extract_eval(sys.argv[2])
    elif cmd == "analyze":
        _ensure_dirs()
        analyze_all()
    elif cmd == "run_all":
        _ensure_dirs()
        build_targets()
        for _a in ("ctrl50", "c3_50", "c4_50"):
            train_arm(_a)
        for _a in ("base", "ctrl50", "c3_50", "c4_50"):
            extract_eval(_a)
        analyze_all()
    else:
        raise SystemExit(f"Unknown cmd {cmd}")
