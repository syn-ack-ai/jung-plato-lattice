"""Experiment 251 (CUDA RTX 4090): 50%-Depth Cross-Family Platonic Consensus Transplant on
official multimodal `google/gemma-4-E4B-it` AND live Aion E4B upgrade.

Trains ONLY full-rank layers 19-20 MLPs (157,286,400 params, fp32 master during AdamW cast
back to bf16 after; training-only 2560->256 plato_head dropped at inference -> 0 extra
inference parameters!), hooking layer 20 (`20/41 = 48.8%`, matching `raw50` & `depth240`
slot 2) on strictly NEUTRAL `gut + proc + chat` passages (ZERO task labels!).

Strictly paired causal arms on unmodified official `google/gemma-4-E4B-it`:
  - base   : Untouched official `google/gemma-4-E4B-it`
  - ctrl50 : plato_w = 0.0 (matched 0.25*CE + 1.5*KL on L19-20, seed=251, 450 steps)
  - c3_50  : plato_w = 0.40 -> 0.0 targeting RIVAL-ONLY C3_50 (Qwen3.5 + Llama3.1 + Mistral;
             ZERO Gemma 4 in teacher -> zero self-distillation; ZERO Phi-4 / OLMo 3!)
  - c4_50  : plato_w = 0.40 -> 0.0 targeting FULL C4_50 (Gemma4 + Qwen3.5 + Llama3.1 + Mistral)

Plus `upgrade_aion`: grafts L19-20 onto live Aion's active checkpoint (keeping L11-12, L40-41,
attn_slices, mm_proj_slices, and O3 Titans 100% frozen & KL-anchored to live Aion!) and runs
full BEFORE vs AFTER blind 50%-depth alignment + Buddy behavior + Vision & Audio checks.
"""
import difflib
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
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO = ARCH / "fluent" / "plato"
MID = ARCH / "fluent" / "mid"
SELFSHAPE = ARCH / "selfshape"
EXP_DIR = Path(os.environ.get("EXP251_DIR", os.environ.get("JPL_EXP251_DIR", "./exp251_work"))).resolve()
SLICE_DIR = EXP_DIR / "slices"
EVAL_DIR = EXP_DIR / "eval_cache"
for _d in (EXP_DIR, SLICE_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from aion_8b import e4b_living as el  # noqa: E402
from fluent import arch_rot229, bow_mirror238, exp228, exp229, plato_data, valence_ctrl234  # noqa: E402

SEEDS_3 = (251, 252, 253)


def _targets_path():
    p = EXP_DIR / "targets_e4b_50.npz"
    if p.exists():
        return p
    legacy = EXP_DIR / "data" / "targets_e4b_50.npz"
    return legacy if legacy.exists() else p


def _eval_texts_path():
    p = EXP_DIR / "eval_texts_251.json"
    if p.exists():
        return p
    legacy = EXP_DIR / "data" / "eval_texts_251.json"
    return legacy if legacy.exists() else p


def _tag_for(arm, seed):
    if arm == "base":
        return "base"
    return arm if int(seed) == 251 else f"{arm}_s{int(seed)}"


def build_eval_texts(out_path=None):
    """Build `eval_texts_251.json` directly from committed repository benchmark datasets."""
    dst = Path(out_path) if out_path else (EXP_DIR / "eval_texts_251.json")
    tr_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    neut_texts = [r["text"].strip() for r in tr_rows if r["domain"] in ("gut", "proc", "chat")]
    house_pairs = arch_rot229.load_arch_pairs()
    sg, ov, _ = valence_ctrl234.load_valence_data()
    val_pairs = sg + ov
    bm120_rows = bow_mirror238.load_mirror_data()
    bmrel_rows = json.loads((SELFSHAPE / "bm_rel242_data.json").read_text())["bm_rel_120"]
    bmneut_rows = json.loads((SELFSHAPE / "bm_neut245_data.json").read_text())["bm_neut_120"]
    actor_rows = json.loads((SELFSHAPE / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    payload = {
        "neut": neut_texts,
        "house_pos": [p["done"].strip() for p in house_pairs],
        "house_neg": [p["neglected"].strip() for p in house_pairs],
        "val_pos": [p["pos"].strip() for p in val_pairs],
        "val_neg": [p["neg"].strip() for p in val_pairs],
        "bm120_pos": [r["pos"].strip() for r in bm120_rows],
        "bm120_neg": [r["neg"].strip() for r in bm120_rows],
        "bmrel_pos": [r["pos"].strip() for r in bmrel_rows],
        "bmrel_neg": [r["neg"].strip() for r in bmrel_rows],
        "bmneut_pos": [r["pos"].strip() for r in bmneut_rows],
        "bmneut_neg": [r["neg"].strip() for r in bmneut_rows],
        "actor_pos": [r["pos"].strip() for r in actor_rows],
        "actor_neg": [r["neg"].strip() for r in actor_rows],
    }
    _atomic_json(dst, payload)
    print(f"BUILD_EVAL_TEXTS_DONE: {dst} (neut={len(neut_texts)}, actor={len(actor_rows)})", flush=True)
    return dst

PLATO50_LAYER = 20          # exact int(round(0.50 * 41)) == 20
MLP_LAYERS = (19, 20)       # 157,286,400 full-rank params directly feeding layer 20
STAGE_A_LAYER = 12          # 30% depth invariance check

ARM_CFG = {
    "ctrl50": dict(plato_w=0.0,  target_key=None),
    "c3_50":  dict(plato_w=0.40, target_key="c3_50_tr"),
    "c4_50":  dict(plato_w=0.40, target_key="c4_50_tr"),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, p)


def _atomic_savez(path, **arrays):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p.with_suffix(".tmp"))
    np.savez(tmp, **arrays)
    os.replace(tmp + ".npz", p)


def _load_e4b_hook20(slice_path=None, with_plato_head=False, offload_mm=True):
    """Load official multimodal `google/gemma-4-E4B-it` wrapped in LivingE4B, ensuring layer 20 is hooked!"""
    model = el.load_living_e4b(slice_path=slice_path, with_plato_head=with_plato_head, offload_mm=offload_mm)
    if PLATO50_LAYER not in model._hooked_layers:
        model._hooks.append(model.lm.layers[PLATO50_LAYER].register_forward_hook(model._make_hook(PLATO50_LAYER)))
        model._hooked_layers.add(PLATO50_LAYER)
    return model


def _disjoint_changed_token_masks(ids_p, ids_n):
    sm = difflib.SequenceMatcher(None, ids_p[1:], ids_n[1:])
    mp = np.zeros(max(1, len(ids_p) - 1), dtype=bool)
    mn = np.zeros(max(1, len(ids_n) - 1), dtype=bool)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            mp[i1:i2] = True
            mn[j1:j2] = True
    if not mp.any():
        mp[:] = True
    if not mn.any():
        mn[:] = True
    return mp, mn


# ---------------------------------------------------------------- Step 1: Strictly Paired Causal Training (CUDA RTX 4090)

def train_arm(arm, steps=450, lr_mlp=1.5e-5, lr_head=1e-3, ce_w=0.25, kl_w=1.5, n_acc=4, seed=251, base_slice=None, out_name=None, keep_titans_in_ref=False):
    assert arm in ARM_CFG, f"Unknown arm {arm}"
    cfg = ARM_CFG[arm]
    tag = out_name or arm
    out_pt = SLICE_DIR / f"{tag}_L19_20.pt"
    if out_pt.exists():
        print(f"TRAIN {tag}: {out_pt} already exists, skipping!", flush=True)
        return out_pt

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    d_t = np.load(_targets_path())
    ids_f, ids_o = d_t["tr_ids_flat"], d_t["tr_ids_off"]
    pts_f, pts_o = d_t["tr_pts_flat"], d_t["tr_pts_off"]
    tgt_arr = d_t[cfg["target_key"]] if cfg["target_key"] else None
    n_tr = len(ids_o) - 1

    model = _load_e4b_hook20(slice_path=base_slice, with_plato_head=(cfg["plato_w"] > 0.0), offload_mm=True)
    dev = next(model.lm.parameters()).device

    # Verify exact tokenizer parity against targets_e4b_50.npz on first neutral passage!
    tr_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    orig0 = int(d_t["tr_orig_idx"][0])
    ids0_hf = el._gemma_encode_with_bos(model.tok, "".join(tr_rows[orig0]["segs"]))
    ids0_npz = ids_f[int(ids_o[0]):int(ids_o[1])].tolist()
    assert ids0_hf == ids0_npz, f"Tokenizer mismatch! hf[:10]={ids0_hf[:10]} npz[:10]={ids0_npz[:10]}"

    ref_mlp_cpu = model.snapshot_mlp_weights_cpu(MLP_LAYERS)
    backup_buf = {}
    use_titans_live = bool(keep_titans_in_ref and base_slice is not None)

    model.set_grad_checkpointing(True)
    opt = None
    history = []
    try:
        with model.fp32_mlps_for_training(MLP_LAYERS):
            model.unfreeze_stage_a(MLP_LAYERS)
            mlp_params = [p for li in MLP_LAYERS for p in model.lm.layers[li].mlp.parameters()]
            head_params = list(model.plato_head.parameters()) if model.plato_head is not None else []
            n_mlp_p = sum(p.numel() for p in mlp_params)
            assert n_mlp_p == 157_286_400, f"Unexpected MLP param count: {n_mlp_p}"
            pgs = [{"params": mlp_params, "lr": lr_mlp, "weight_decay": 0.01}]
            if head_params:
                pgs.append({"params": head_params, "lr": lr_head, "weight_decay": 0.0})
            opt = torch.optim.AdamW(pgs, betas=(0.9, 0.95))

            print(
                f"TRAIN {tag} start: steps={steps} n_acc={n_acc} plato_w={cfg['plato_w']} target={cfg['target_key']} "
                f"mlp_params={n_mlp_p:,} head_params={sum(p.numel() for p in head_params):,} base_slice={base_slice} "
                f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
                flush=True,
            )
            model.train()

            for s in range(steps):
                w_s = max(1, steps // 10)
                lr_mul = (s + 1) / w_s if s < w_s else 0.5 * (1.0 + math.cos(math.pi * (s - w_s) / max(1, steps - w_s)))
                opt.param_groups[0]["lr"] = lr_mlp * lr_mul
                if head_params:
                    opt.param_groups[1]["lr"] = lr_head * lr_mul
                cur_pw = exp229.plato_w_at(cfg["plato_w"], s, steps, anneal_start=0.65, anneal_end=0.90)

                picks = rng.integers(0, n_tr, size=n_acc)
                batch = []
                for pi in picks:
                    g_ids = ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].astype(np.int64)
                    pa, pb = int(pts_o[pi]), int(pts_o[pi + 1])
                    g_pts = pts_f[pa:pb].astype(np.int64)
                    g_tgt = tgt_arr[pa:pb] if tgt_arr is not None else None
                    batch.append((g_ids, g_pts, g_tgt))

                # Precompute exact reference hidden states via ONE CPU weight swap per step
                ref_hs = []
                with torch.no_grad(), model.swapped_to_ref_weights(ref_mlp_cpu, disable_titans=not use_titans_live, backup_buf=backup_buf):
                    with el._autocast_ctx(dev):
                        for g_ids, _, _ in batch:
                            x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                            hr = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :].detach()
                            ref_hs.append(hr)

                opt.zero_grad(set_to_none=True)
                sum_ce = sum_kl = sum_plato = 0.0
                for (g_ids, g_pts, g_tgt), h_ref in zip(batch, ref_hs):
                    x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                    with el._autocast_ctx(dev), model.run_ctx(enabled=use_titans_live, state=None, update=False, capture_layers=(PLATO50_LAYER,)) as ctx:
                        h_cur = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
                        h20 = ctx["capture"][PLATO50_LAYER]
                        ce_v, kl_v, h_grad = el.chunked_lm_head_ce_kl(
                            model.base, h_cur, y_ids=x[:, 1:], h_ref=h_ref,
                            ce_w=ce_w / n_acc, kl_w=kl_w / n_acc, chunk_t=64, return_grad=True,
                        )
                        sum_ce += ce_v / n_acc
                        sum_kl += kl_v / n_acc

                        if cur_pw > 0.0 and model.plato_head is not None:
                            t_idx = torch.from_numpy(g_pts).to(dev)
                            t_tgt = torch.from_numpy(g_tgt).to(dev, dtype=torch.float32)
                            pred = model.plato_head(h20[0, t_idx, :].float())
                            l_p = exp228.plato_alignment_loss(pred, t_tgt)
                            sum_plato += float(l_p.item()) / n_acc
                            torch.autograd.backward([h_cur, (cur_pw / n_acc) * l_p], [h_grad, None])
                        else:
                            h_cur.backward(h_grad)
                    del h_cur, h20, h_grad
                del ref_hs

                gn = float(torch.nn.utils.clip_grad_norm_(mlp_params, 1.0))
                if head_params and cur_pw > 0.0:
                    torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                opt.step()
                assert math.isfinite(sum_ce) and math.isfinite(sum_kl) and math.isfinite(sum_plato), f"{tag} diverged at step {s}!"
                if s % 50 == 0 or s == steps - 1:
                    rec = {
                        "step": s, "ce": round(sum_ce, 4), "kl": round(sum_kl, 5), "plato": round(sum_plato, 5),
                        "w": round(cur_pw, 4), "gn": round(gn, 3),
                        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                        "sec": round(time.time() - t0, 1),
                    }
                    history.append(rec)
                    print(
                        f"  [{tag}] step {s:3d}/{steps} ce={rec['ce']:.3f} kl={rec['kl']:.4f} plato={rec['plato']:.4f} "
                        f"(w={rec['w']:.3f}) gn={rec['gn']:.2f} vram={rec['vram_gb']:.1f}GB ({rec['sec']:.0f}s)",
                        flush=True,
                    )
    finally:
        model.set_grad_checkpointing(False)
        model.zero_grad(set_to_none=True)
        model.freeze_all()
        model.eval()
        del opt, backup_buf
        gc.collect()
        torch.cuda.empty_cache()

    # Exact L2 weight drift vs reference before saving compact bf16 slice (plato_head DROPPED -> 0 extra params!)
    deltas = {}
    with torch.no_grad():
        for li in MLP_LAYERS:
            d2 = r2 = 0.0
            for n, p in model.lm.layers[li].mlp.named_parameters():
                pf = p.detach().float().cpu()
                rf = ref_mlp_cpu[li][n].float()
                d2 += float((pf - rf).pow(2).sum().item())
                r2 += float(rf.pow(2).sum().item())
            deltas[f"L{li}"] = {"l2_delta": round(math.sqrt(d2), 4), "rel_delta": round(math.sqrt(d2 / max(1e-12, r2)), 6)}
    del ref_mlp_cpu

    # Determine which MLP layers to persist (ONLY 19-20 for clean Base arms, OR union with loaded slice for live Aion!)
    if base_slice is not None:
        base_payload = torch.load(base_slice, map_location="cpu", weights_only=True)
        save_mlps = sorted(set(int(k) for k in base_payload.get("mlp_slices", {}).keys()) | set(MLP_LAYERS))
        base_meta = dict(base_payload.get("meta", {}))
    else:
        save_mlps = list(MLP_LAYERS)
        base_meta = {}

    meta = {
        **base_meta,
        "plato50_arm": arm, "plato50_tag": tag, "plato50_mlp_layers": list(MLP_LAYERS),
        "plato50_layer": PLATO50_LAYER, "plato50_steps": steps, "plato50_seed": seed,
        "plato50_w": cfg["plato_w"], "plato50_target": cfg["target_key"],
        "plato50_deltas": deltas, "plato50_elapsed_sec": round(time.time() - t0, 1),
    }
    model.save_living_slice(out_pt, meta=meta, mlp_layers=save_mlps)
    meta["slice_sha256"] = _sha256(out_pt)
    meta["history"] = history
    _atomic_json(EXP_DIR / f"train_{tag}_log.json", meta)
    print(f"TRAIN_DONE {tag}: saved={out_pt} mlps={save_mlps} deltas={deltas} elapsed={meta['plato50_elapsed_sec']}s", flush=True)

    model.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_pt


# ---------------------------------------------------------------- Step 2: Comprehensive Extraction + Multimodal & LM Guardrails

@torch.no_grad()
def extract_eval(arm, slice_path=None, run_buddy=False):
    """Extract L20 (50% depth) hidden states on ALL neutral train/eval checkpoints + benchmarks + NLL +
    proc_held MCQ + Multimodal Vision/Audio sanity check (and optional Buddy behavior eval!)."""
    acts_npz = EVAL_DIR / f"acts_{arm}.npz"
    lm_json = EVAL_DIR / f"lm_{arm}.json"
    if acts_npz.exists() and lm_json.exists():
        print(f"EXTRACT_EVAL {arm}: already cached, skipping!", flush=True)
        return

    t0 = time.time()
    if slice_path is None and arm != "base":
        slice_path = SLICE_DIR / f"{arm}_L19_20.pt"
        assert Path(slice_path).exists(), f"Missing trained slice {slice_path}"

    # Always reload freshly WITHOUT plato_head -> guarantees 0 extra inference params!
    model = _load_e4b_hook20(slice_path=slice_path, with_plato_head=False, offload_mm=True)
    model.eval()
    dev = next(model.lm.parameters()).device
    tok = model.tok
    use_titans = bool(run_buddy)  # keep Titans disabled for clean Base-E4B arms (o==0 anyway), enabled for live Aion!

    d_t = np.load(_targets_path())
    ev_texts_p = _eval_texts_path()
    if not ev_texts_p.exists():
        ev_texts_p = build_eval_texts(ev_texts_p)
    texts = json.loads(ev_texts_p.read_text())

    def run_packed_pts(ids_f, ids_o, pts_f, pts_o, label, check_l12=False):
        pts_list, st_list, l12_probe = [], [], []
        n_p = len(ids_o) - 1
        for i in range(n_p):
            ids = ids_f[int(ids_o[i]):int(ids_o[i + 1])].astype(np.int64)
            pts = pts_f[int(pts_o[i]):int(pts_o[i + 1])].astype(np.int64)
            x = torch.from_numpy(ids).unsqueeze(0).to(dev)
            caps_want = (STAGE_A_LAYER, PLATO50_LAYER) if (check_l12 and i < 16) else (PLATO50_LAYER,)
            with el._autocast_ctx(dev), model.run_ctx(enabled=use_titans, state=None, update=False, capture_layers=caps_want) as ctx:
                _ = model.base.model(input_ids=x, use_cache=False)
                h20 = ctx["capture"][PLATO50_LAYER][0].float().cpu().numpy()
                if check_l12 and i < 16:
                    l12_probe.append(ctx["capture"][STAGE_A_LAYER][0].float().mean(dim=0).cpu().numpy())
            pts_list.append(h20[pts])
            st_list.append(h20[1:].mean(axis=0) if len(h20) > 1 else h20[0])
            if (i + 1) % 800 == 0:
                print(f"  [{arm}] {label}: {i + 1}/{n_p} ({time.time() - t0:.0f}s)", flush=True)
        return np.concatenate(pts_list, 0), np.stack(st_list, 0), (np.stack(l12_probe, 0) if l12_probe else None)

    tr_pts, _, _ = run_packed_pts(d_t["tr_ids_flat"], d_t["tr_ids_off"], d_t["tr_pts_flat"], d_t["tr_pts_off"], "tr_pts")
    ev_pts, ev_story, l12_probe = run_packed_pts(d_t["ev_ids_flat"], d_t["ev_ids_off"], d_t["ev_pts_flat"], d_t["ev_pts_off"], "ev_pts", check_l12=True)

    def extract_texts_ml(text_list, label):
        ms, ls = [], []
        for i, txt in enumerate(text_list):
            ids = el._gemma_encode_with_bos(tok, txt)[:768]
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            with el._autocast_ctx(dev), model.run_ctx(enabled=use_titans, state=None, update=False, capture_layers=(PLATO50_LAYER,)) as ctx:
                _ = model.base.model(input_ids=x, use_cache=False)
                h = ctx["capture"][PLATO50_LAYER][0].float()
                hm = h[1:].mean(dim=0) if h.shape[0] > 1 else h[0]
                hl = h[-1]
            ms.append(hm.cpu().numpy())
            ls.append(hl.cpu().numpy())
            if (i + 1) % 1000 == 0:
                print(f"  [{arm}] {label}: {i + 1}/{len(text_list)} ({time.time() - t0:.0f}s)", flush=True)
        return np.stack(ms, 0), np.stack(ls, 0)

    neut_m, neut_l = extract_texts_ml(texts["neut"], "neut_2394")
    house_pm, house_pl = extract_texts_ml(texts["house_pos"], "house_pos")
    house_nm, house_nl = extract_texts_ml(texts["house_neg"], "house_neg")
    val_pm, val_pl = extract_texts_ml(texts["val_pos"], "val_pos")
    val_nm, val_nl = extract_texts_ml(texts["val_neg"], "val_neg")
    bm120_pm, bm120_pl = extract_texts_ml(texts["bm120_pos"], "bm120_pos")
    bm120_nm, bm120_nl = extract_texts_ml(texts["bm120_neg"], "bm120_neg")
    bmrel_pm, bmrel_pl = extract_texts_ml(texts["bmrel_pos"], "bmrel_pos")
    bmrel_nm, bmrel_nl = extract_texts_ml(texts["bmrel_neg"], "bmrel_neg")
    bmneut_pm, bmneut_pl = extract_texts_ml(texts["bmneut_pos"], "bmneut_pos")
    bmneut_nm, bmneut_nl = extract_texts_ml(texts["bmneut_neg"], "bmneut_neg")

    def seq_ce_and_l20(ids):
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with el._autocast_ctx(dev), model.run_ctx(enabled=use_titans, state=None, update=False, capture_layers=(PLATO50_LAYER,)) as ctx:
            h_out = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
            h20 = ctx["capture"][PLATO50_LAYER][0].float()
            hm = h20[1:].mean(dim=0) if h20.shape[0] > 1 else h20[0]
            hl = h20[-1]
        ces = []
        T = h_out.shape[1]
        y = x[:, 1:]
        for s in range(0, T, 64):
            e = min(T, s + 64)
            lg = el._lm_head_softcap(model.base, h_out[:, s:e, :])
            ce_c = F.cross_entropy(lg[0], y[0, s:e], reduction="none")
            ces.append(ce_c.cpu().numpy())
        return np.concatenate(ces, 0).astype(np.float32), hm.cpu().numpy(), hl.cpu().numpy()

    ap_m, ap_l, an_m, an_l, ap_nll, an_nll, ap_chg, an_chg = [], [], [], [], [], [], [], []
    for p_txt, n_txt in zip(texts["actor_pos"], texts["actor_neg"]):
        ids_p = el._gemma_encode_with_bos(tok, p_txt)
        ids_n = el._gemma_encode_with_bos(tok, n_txt)
        mp, mn = _disjoint_changed_token_masks(ids_p, ids_n)
        ce_p, hmp, hlp = seq_ce_and_l20(ids_p)
        ce_n, hmn, hln = seq_ce_and_l20(ids_n)
        ap_m.append(hmp); ap_l.append(hlp); an_m.append(hmn); an_l.append(hln)
        ap_nll.append(float(ce_p.mean())); an_nll.append(float(ce_n.mean()))
        ap_chg.append(float(ce_p[mp].mean())); an_chg.append(float(ce_n[mn].mean()))

    # Held-out gut NLL (exact 40 passages)
    ev_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl")]
    gut_sub = [r for r in ev_rows if r.get("domain") == "held_gut"][:40]
    gut_nlls = []
    for r in gut_sub:
        ids = el._gemma_encode_with_bos(tok, "".join(r["segs"]))[:512]
        if len(ids) >= 4:
            ce_all, _, _ = seq_ce_and_l20(ids)
            gut_nlls.append(float(ce_all.mean()))

    # Held-out proc_held 400-item MCQ!
    proc_items = [json.loads(l) for l in open(MID / "proc_held.jsonl")]

    def score_cont(prefix, cont):
        ids_pre = el._gemma_encode_with_bos(tok, prefix)
        ids_full = el._gemma_encode_with_bos(tok, prefix + cont)
        k = max(1, len(ids_full) - len(ids_pre))
        x = torch.tensor([ids_full], dtype=torch.long, device=dev)
        with el._autocast_ctx(dev), model.run_ctx(enabled=use_titans, state=None, update=False):
            h_out = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, -k - 1:-1, :]
        lg = el._lm_head_softcap(model.base, h_out)[0]
        ce = F.cross_entropy(lg, x[0, -k:], reduction="mean")
        return -float(ce.item())

    def credit(scores):
        b = max(scores)
        tied = [i for i, v in enumerate(scores) if v == b]
        return (1.0 / len(tied)) if 0 in tied else 0.0

    proc_by_kind = defaultdict(lambda: {"w": [], "wo": []})
    for idx_p, r in enumerate(proc_items):
        opts = [r["answer"]] + r["wrong"]
        pre_w = f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:"
        pre_wo = f"Question: {r['question']}\nAnswer:"
        proc_by_kind[r["kind"]]["w"].append(credit([score_cont(pre_w, " " + o) for o in opts]))
        proc_by_kind[r["kind"]]["wo"].append(credit([score_cont(pre_wo, " " + o) for o in opts]))

    proc_res, all_w, all_wo = {}, [], []
    for k, d in sorted(proc_by_kind.items()):
        all_w.extend(d["w"]); all_wo.extend(d["wo"])
        proc_res[k] = {"n": len(d["w"]), "with": round(float(np.mean(d["w"])), 4), "without": round(float(np.mean(d["wo"])), 4), "gain": round(float(np.mean(d["w"]) - np.mean(d["wo"])), 4)}
    proc_res["overall"] = {"n": len(all_w), "with": round(float(np.mean(all_w)), 4), "without": round(float(np.mean(all_wo)), 4), "gain": round(float(np.mean(all_w) - np.mean(all_wo)), 4)}

    # Quick blind Phi-4 & OLMo 3 50%-depth printout right on cuda_host!
    _, Z_ev, _ = plato_data.pca_project(tr_pts.astype(np.float64), ev_pts.astype(np.float64), 256)
    ev_g = d_t["ev_pt_groups"].tolist()
    ev_d = d_t["ev_pt_domains"].tolist()
    ev_r = d_t["ev_pt_ret"]
    p4_q = plato_data.eval_procrustes_transfer(Z_ev, d_t["phi4_50_ev"].astype(np.float64), group_ids=ev_g, domains=ev_d, ret_mask=ev_r)
    ol_q = plato_data.eval_procrustes_transfer(Z_ev, d_t["olmo3_50_ev"].astype(np.float64), group_ids=ev_g, domains=ev_d, ret_mask=ev_r)

    # Multimodal Vision ("Red") + Audio ("blue pelican") sanity check!
    mm_res = el.eval_multimodal_sanity(model)
    buddy_res = el.eval_buddy_behavior(model) if run_buddy else None

    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    _atomic_savez(
        acts_npz,
        tr_pts=pack(tr_pts), ev_pts=pack(ev_pts), ev_story=pack(ev_story), l12_probe=l12_probe.astype(np.float32),
        neut_mean=pack(neut_m), neut_last=pack(neut_l),
        house_pos_mean=pack(house_pm), house_pos_last=pack(house_pl),
        house_neg_mean=pack(house_nm), house_neg_last=pack(house_nl),
        val_pos_mean=pack(val_pm), val_pos_last=pack(val_pl),
        val_neg_mean=pack(val_nm), val_neg_last=pack(val_nl),
        bm120_pos_mean=pack(bm120_pm), bm120_pos_last=pack(bm120_pl),
        bm120_neg_mean=pack(bm120_nm), bm120_neg_last=pack(bm120_nl),
        bmrel_pos_mean=pack(bmrel_pm), bmrel_pos_last=pack(bmrel_pl),
        bmrel_neg_mean=pack(bmrel_nm), bmrel_neg_last=pack(bmrel_nl),
        bmneut_pos_mean=pack(bmneut_pm), bmneut_pos_last=pack(bmneut_pl),
        bmneut_neg_mean=pack(bmneut_nm), bmneut_neg_last=pack(bmneut_nl),
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
        "quick_blind_phi4_50": p4_q,
        "quick_blind_olmo3_50": ol_q,
        "multimodal": mm_res,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if buddy_res is not None:
        lm_out["buddy"] = buddy_res
    _atomic_json(lm_json, lm_out)
    print(
        f"EXTRACT_EVAL_DONE {arm}: Phi4(50%)={p4_q['cos']:.4f}(R2f={p4_q['r2_frob']:+.4f},top1={p4_q['top1']:.4f}) "
        f"OLMo(50%)={ol_q['cos']:.4f}(R2f={ol_q['r2_frob']:+.4f},top1={ol_q['top1']:.4f}) "
        f"gutNLL={lm_out['held_gut_nll_mean']} proc={proc_res['overall']['with']}(+{proc_res['overall']['gain']:.4f}) "
        f"actNLL=({lm_out['actor_nll_whole_acc']}/{lm_out['actor_nll_changed_acc']}) "
        f"MM=(vis={mm_res['vision_ok']},aud={mm_res['audio_ok']}) in {lm_out['elapsed_sec']}s",
        flush=True,
    )
    model.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build_eval_texts":
        build_eval_texts()
    elif cmd == "train":
        arm_arg = sys.argv[2]
        seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 251
        train_arm(arm_arg, seed=seed_arg, out_name=_tag_for(arm_arg, seed_arg))
    elif cmd == "extract_eval":
        extract_eval(sys.argv[2])
    elif cmd == "run_base_study":
        for _a in ("ctrl50", "c3_50", "c4_50"):
            train_arm(_a, seed=251, out_name=_tag_for(_a, 251))
        for _a in ("base", "ctrl50", "c3_50", "c4_50"):
            extract_eval(_a)
    elif cmd == "run_3seed":
        build_eval_texts()
        extract_eval("base")
        for _s in SEEDS_3:
            for _a in ("ctrl50", "c3_50", "c4_50"):
                _tag = _tag_for(_a, _s)
                train_arm(_a, seed=_s, out_name=_tag)
                extract_eval(_tag)
    elif cmd == "upgrade_aion":
        live_slice = sys.argv[2]
        win_arm = sys.argv[3]  # e.g., c3_50 or c4_50
        extract_eval("aion_before", slice_path=live_slice, run_buddy=True)
        out_pt = train_arm(win_arm, base_slice=live_slice, out_name="aion_after_50", keep_titans_in_ref=True)
        extract_eval("aion_after", slice_path=out_pt, run_buddy=True)
    else:
        raise SystemExit(f"Unknown cmd {cmd}")
