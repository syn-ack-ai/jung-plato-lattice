"""Multi-Family Platonic Python-Coding Lattice Builder (`Exp 258`, Local Mac Studio MLX).

Extracts aligned semantic code-checkpoint hidden states at THREE depths (30%, 50%, 70%)
in ONE forward pass per model (strictly 1 model in RAM at a time) across 5 independent
NON-Qwen open LLM families:
  - 3 Rival Teachers (for GPA Consensus):
      1. Google Gemma 4 E4B (`mlx-community/gemma-4-E4B-it-bf16`)
      2. Meta Llama 3.1 8B  (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`)
      3. Mistral 7B v0.3    (`mlx-community/Mistral-7B-Instruct-v0.3-4bit`)
  - 2 Strictly Blind Target Families (NEVER in consensus construction!):
      4. Microsoft Phi-4-mini (`mlx-community/Phi-4-mini-instruct-4bit`)
      5. AllenAI OLMo 3 7B    (`mlx-community/Olmo-3-7B-Instruct-4bit`)

Then computes 10-iteration Generalized Procrustes Analysis (GPA) 256-dim Python-Coding
Consensus (`C_code_30`, `C_code_50`, `C_code_70`) strictly on `code_train.jsonl`, evaluates
exact `plato_data.py` OOF 2-fold Procrustes transfer (`r2_frob`, `r2_opt`, `cos`, `top1`, `top5`)
on `code_eval.jsonl` comparing each single teacher vs. the 3-family GPA consensus against
blind Phi-4-mini and OLMo 3 7B, and packs `targets_qwen600m_code_lattice_30_50_70.npz`
tokenized cleanly with `Qwen/Qwen3-0.6B`!
"""
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

DEPTHS = (30, 50, 70)
DEPTH_FRACS = {30: 0.30, 50: 0.50, 70: 0.70}

TEACHER_KEYS = ("gemma4_e4b", "llama31_8b", "mistral7b")
BLIND_KEYS = ("phi4_mini", "olmo3_7b")
ALL_KEYS = TEACHER_KEYS + BLIND_KEYS

MODEL_SPECS = {
    "gemma4_e4b": os.environ.get("GEMMA4_MLX_MODEL", "mlx-community/gemma-4-E4B-it-bf16"),
    "llama31_8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mistral7b":  "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "phi4_mini":  "mlx-community/Phi-4-mini-instruct-4bit",
    "olmo3_7b":   "mlx-community/Olmo-3-7B-Instruct-4bit",
}


def _corpus_sha256():
    h = hashlib.sha256()
    for p in (DATA / "code_train.jsonl", DATA / "code_eval.jsonl"):
        h.update(p.read_bytes())
    return h.hexdigest()


def _text_ids_mlx(tok, t):
    ids = tok.encode(t, add_special_tokens=False)
    if getattr(tok, "_starts_bos", None) is None:
        b = tok.bos_token
        try:
            tmpl = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
            tok._starts_bos = bool(b) and tmpl.startswith(b)
        except Exception:
            tok._starts_bos = bool(b)
    return [tok.bos_token_id] + ids if tok._starts_bos and (not ids or ids[0] != tok.bos_token_id) else ids


def _find_layers(model):
    for obj in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
        if obj is not None:
            if hasattr(obj, "layers"):
                return obj.layers
            if hasattr(obj, "model") and hasattr(obj.model, "layers"):
                return obj.model.layers
    raise RuntimeError(f"Could not locate decoder layers on {type(model)}")


def boundary_token_indices(encode_fn, segs, allow_merge_tie=False):
    full_text = "".join(segs)
    full_ids = encode_fn(full_text)
    if len(full_ids) < 2:
        return full_ids, []
    pts = []
    acc = ""
    for j, s in enumerate(segs):
        acc += s
        if j == len(segs) - 1:
            idx = len(full_ids) - 1
        else:
            p_ids = encode_fn(acc.rstrip())
            idx = max(1, min(len(full_ids) - 1, len(p_ids) - 1))
        if not pts or idx > pts[-1]:
            pts.append(idx)
        elif allow_merge_tie:
            pts.append(min(len(full_ids) - 1, pts[-1]))
        else:
            return full_ids, []
    return full_ids, pts


def _atomic_savez(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def extract_model_3strata(key: str):
    out_npz = CACHE / f"raw_code_{key}.npz"
    corpus_sig = _corpus_sha256()
    model_path = MODEL_SPECS[key]
    if out_npz.exists():
        try:
            with np.load(out_npz) as d:
                if str(d["corpus_sha256"].item()) == corpus_sig and str(d["model_path"].item()) == model_path:
                    print(f"[CODE_LATTICE] Verified cache for {key}, skipping.", flush=True)
                    return
        except Exception:
            pass

    train_rows = [json.loads(l) for l in open(DATA / "code_train.jsonl") if l.strip()]
    eval_rows = [json.loads(l) for l in open(DATA / "code_eval.jsonl") if l.strip()]

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"[CODE_LATTICE] Loading {key} ({model_path})...", flush=True)
    model, tok = load(model_path)
    layers = _find_layers(model)
    nl = len(layers)
    tgt_idx = {d: max(1, min(nl - 2, int(round(DEPTH_FRACS[d] * (nl - 1))))) for d in DEPTHS}
    blk_to_depth = {id(layers[li]): d for d, li in tgt_idx.items()}
    print(f"[CODE_LATTICE] {key}: L={nl}, hooked layers={tgt_idx}", flush=True)

    rec_buf = {}
    patched = {}
    for blk in layers:
        cls = type(blk)
        if cls not in patched:
            orig = cls.__call__
            patched[cls] = orig

            def make_hook(orig_fn):
                def hooked(self, *a, **k):
                    out = orig_fn(self, *a, **k)
                    d_hit = blk_to_depth.get(id(self))
                    if d_hit is not None:
                        h = out[0] if isinstance(out, tuple) else out
                        rec_buf[d_hit] = h[0]
                    return out
                return hooked

            cls.__call__ = make_hook(orig)

    enc_fn = lambda s: _text_ids_mlx(tok, s)

    def run_pool(rows, tag):
        pts_vecs = {d: [] for d in DEPTHS}
        pt_cnts, row_ok = [], []
        for idx, r in enumerate(rows):
            full_ids, pt_idxs = boundary_token_indices(enc_fn, r["segs"], allow_merge_tie=False)
            good = (4 <= len(full_ids) <= 512 and len(pt_idxs) == len(r["segs"]))
            if not good:
                pt_cnts.append(0)
                row_ok.append(False)
                continue
            rec_buf.clear()
            _ = model(mx.array([full_ids]))
            assert len(rec_buf) == len(DEPTHS), f"{key} captured {list(rec_buf.keys())}"
            arr_idx = mx.array(pt_idxs)
            eval_list = []
            tmp_h = {}
            for d in DEPTHS:
                hd = rec_buf[d].astype(mx.float32)[arr_idx]
                tmp_h[d] = hd
                eval_list.append(hd)
            mx.eval(*eval_list)
            for d in DEPTHS:
                pts_vecs[d].append(np.array(tmp_h[d], dtype=np.float32))
            pt_cnts.append(len(pt_idxs))
            row_ok.append(True)
            if (idx + 1) % 400 == 0 or idx == len(rows) - 1:
                print(f"[CODE_LATTICE] {key} [{tag}] {idx + 1}/{len(rows)} ok={sum(row_ok)} ({time.time() - t0:.1f}s)", flush=True)
        out = {d: np.concatenate(pts_vecs[d], axis=0) for d in DEPTHS}
        return out, np.array(pt_cnts, dtype=np.int32), np.array(row_ok, dtype=bool)

    try:
        tr_pts, tr_cnt, tr_ok = run_pool(train_rows, "train")
        ev_pts, ev_cnt, ev_ok = run_pool(eval_rows, "eval")
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        del model, tok
        gc.collect()
        mx.clear_cache()

    save_d = {
        "tr_cnt": tr_cnt, "tr_ok": tr_ok,
        "ev_cnt": ev_cnt, "ev_ok": ev_ok,
        "num_layers": np.int32(nl),
        "corpus_sha256": np.array(corpus_sig),
        "model_path": np.array(model_path),
    }
    for d in DEPTHS:
        assert np.isfinite(tr_pts[d]).all() and np.isfinite(ev_pts[d]).all(), f"{key} non-finite at {d}%"
        pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
        save_d[f"tr_pts_{d}"] = pack(tr_pts[d])
        save_d[f"ev_pts_{d}"] = pack(ev_pts[d])
        save_d[f"layer_{d}"] = np.int32(tgt_idx[d])
    _atomic_savez(out_npz, **save_d)
    print(f"[CODE_LATTICE] SAVED {out_npz.name}: tr_ok={int(tr_ok.sum())}/{len(tr_ok)} ({tr_pts[30].shape}) ev_ok={int(ev_ok.sum())}/{len(ev_ok)} in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Generalized Procrustes Analysis & Exact OOF Paper Metric (`plato_data.py`)
# ---------------------------------------------------------------------------

def fit_procrustes_map(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mu_a = A.mean(axis=0, keepdims=True)
    mu_b = B.mean(axis=0, keepdims=True)
    Ac, Bc = A - mu_a, B - mu_b
    U, _, Vt = np.linalg.svd(Bc.T @ Ac, full_matrices=False)
    R = U @ Vt
    s = float(np.linalg.norm(Bc, "fro") / max(1e-12, np.linalg.norm(Ac @ R.T, "fro")))
    return mu_a, mu_b, R, s


def apply_procrustes_map(A, params):
    mu_a, mu_b, R, s = params
    return mu_b + s * ((np.asarray(A, dtype=np.float64) - mu_a) @ R.T)


def pca_project(X_tr, X_ev, d_out=256):
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_ev = np.asarray(X_ev, dtype=np.float64)
    mu = X_tr.mean(axis=0, keepdims=True)
    Xc = X_tr - mu
    cov = (Xc.T @ Xc) / float(len(Xc))
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:d_out]
    V = vecs[:, idx]
    Z_tr = Xc @ V
    Z_ev = (X_ev - mu) @ V
    rms = float(np.sqrt(np.mean(Z_tr ** 2)))
    return Z_tr / max(1e-12, rms), Z_ev / max(1e-12, rms), float(vals[idx].sum() / max(1e-12, vals.sum()))


def gpa_consensus(Y_tr_list, Y_ev_list, n_iters=10):
    K = len(Y_tr_list)
    C = Y_tr_list[0].copy()
    for _ in range(n_iters):
        aligned = []
        for k in range(K):
            params = fit_procrustes_map(Y_tr_list[k], C)
            aligned.append(apply_procrustes_map(Y_tr_list[k], params))
        C = np.mean(aligned, axis=0)
        C = C - C.mean(axis=0, keepdims=True)
        C = C / max(1e-12, float(np.sqrt(np.mean(C ** 2))))
    tr_aligned, ev_aligned = [], []
    for k in range(K):
        params = fit_procrustes_map(Y_tr_list[k], C)
        tr_aligned.append(apply_procrustes_map(Y_tr_list[k], params))
        ev_aligned.append(apply_procrustes_map(Y_ev_list[k], params))
    tr_raw = np.mean(tr_aligned, axis=0)
    ev_raw = np.mean(ev_aligned, axis=0)
    tr_mu = tr_raw.mean(axis=0, keepdims=True)
    tr_rms = max(1e-12, float(np.sqrt(np.mean((tr_raw - tr_mu) ** 2))))
    return (tr_raw - tr_mu) / tr_rms, (ev_raw - tr_mu) / tr_rms


def eval_oof_2fold(src_ev, tgt_ev, group_ids):
    src_ev = np.asarray(src_ev, dtype=np.float64)
    tgt_ev = np.asarray(tgt_ev, dtype=np.float64)
    uniq = sorted(set(group_ids))
    rng = random.Random(2588)
    rng.shuffle(uniq)
    half = len(uniq) // 2
    g_fold = {g: (0 if i < half else 1) for i, g in enumerate(uniq)}
    folds = np.array([g_fold[g] for g in group_ids], dtype=np.int32)

    pred_all, pred_opt_all, true_all, cos_all = [], [], [], []
    top1_l, top5_l = [], []
    for test_f in (0, 1):
        m_tr, m_te = (folds != test_f), (folds == test_f)
        params = fit_procrustes_map(src_ev[m_tr], tgt_ev[m_tr])
        pred_tr = apply_procrustes_map(src_ev[m_tr], params)
        pred = apply_procrustes_map(src_ev[m_te], params)
        true_tr, true = tgt_ev[m_tr], tgt_ev[m_te]
        ym_tr, pm_tr = true_tr.mean(axis=0, keepdims=True), pred_tr.mean(axis=0, keepdims=True)
        alpha_f = float(np.sum((true_tr - ym_tr) * (pred_tr - pm_tr)) / max(1e-12, np.sum((pred_tr - pm_tr) ** 2)))
        pred_opt_all.append(ym_tr + alpha_f * (pred - pm_tr))
        pn = pred / np.maximum(1e-12, np.linalg.norm(pred, axis=1, keepdims=True))
        tn = true / np.maximum(1e-12, np.linalg.norm(true, axis=1, keepdims=True))
        cos_all.append(np.sum(pn * tn, axis=1))
        pred_all.append(pred); true_all.append(true)
        sim = pn @ tn.T
        diag = np.diag(sim)[:, None]
        sb = (sim > diag + 1e-12).sum(axis=1)
        tied = (np.abs(sim - diag) <= 1e-12).sum(axis=1)
        top1_l.append(float(np.mean(np.where(sb == 0, 1.0 / np.maximum(1, tied), 0.0))))
        top5_l.append(float(np.mean(np.minimum(1.0, np.maximum(0, 5 - sb) / np.maximum(1, tied)))))

    pred_cat = np.concatenate(pred_all, axis=0)
    pred_opt_cat = np.concatenate(pred_opt_all, axis=0)
    true_cat = np.concatenate(true_all, axis=0)
    ss_tot = float(np.sum((true_cat - true_cat.mean(axis=0, keepdims=True)) ** 2))
    return {
        "r2_frob": round(1.0 - float(np.sum((true_cat - pred_cat) ** 2)) / max(1e-12, ss_tot), 5),
        "r2_opt": round(1.0 - float(np.sum((true_cat - pred_opt_cat) ** 2)) / max(1e-12, ss_tot), 5),
        "cos": round(float(np.concatenate(cos_all).mean()), 5),
        "top1": round(float(np.mean(top1_l)), 5),
        "top5": round(float(np.mean(top5_l)), 5),
    }


def _slice_keep(pts, cnt, ok, keep_mask):
    pos = 0
    out = []
    for i in range(len(cnt)):
        c = int(cnt[i])
        if ok[i]:
            if keep_mask[i]:
                out.append(pts[pos : pos + c])
            pos += c
    assert pos == len(pts), f"pos={pos} != len={len(pts)}"
    return np.concatenate(out, axis=0).astype(np.float64)


def build_consensus_and_pack_qwen600m(d_plato: int = 256):
    from transformers import AutoTokenizer

    train_rows = [json.loads(l) for l in open(DATA / "code_train.jsonl") if l.strip()]
    eval_rows = [json.loads(l) for l in open(DATA / "code_eval.jsonl") if l.strip()]
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    q_enc = lambda s: qtok.encode(s, add_special_tokens=False)

    raw = {k: dict(np.load(CACHE / f"raw_code_{k}.npz", allow_pickle=True)) for k in ALL_KEYS}

    # Strict joint intersection across ALL 5 rival/blind models AND Qwen3-0.6B tokenizer!
    def build_keep(rows, split_prefix):
        keep = []
        q_ids_list, q_pts_list, q_ids_off, q_pts_off, pt_groups = [], [], [0], [0], []
        for i, r in enumerate(rows):
            n_seg = len(r["segs"])
            all_ok = all(bool(raw[k][f"{split_prefix}_ok"][i]) and int(raw[k][f"{split_prefix}_cnt"][i]) == n_seg for k in ALL_KEYS)
            q_ids, q_pts = boundary_token_indices(q_enc, r["segs"], allow_merge_tie=True)
            ok = all_ok and (4 <= len(q_ids) <= 512) and (len(q_pts) == n_seg)
            keep.append(ok)
            if ok:
                q_ids_list.extend(q_ids)
                q_pts_list.extend(q_pts)
                q_ids_off.append(len(q_ids_list))
                q_pts_off.append(len(q_pts_list))
                pt_groups.extend([r["id"]] * n_seg)
        return (
            np.array(keep, dtype=bool),
            np.array(q_ids_list, dtype=np.int32),
            np.array(q_ids_off, dtype=np.int64),
            np.array(q_pts_list, dtype=np.int32),
            np.array(q_pts_off, dtype=np.int64),
            np.array(pt_groups),
        )

    tr_keep, tr_ids_f, tr_ids_o, tr_pts_f, tr_pts_o, tr_grps = build_keep(train_rows, "tr")
    ev_keep, ev_ids_f, ev_ids_o, ev_pts_f, ev_pts_o, ev_grps = build_keep(eval_rows, "ev")
    print(f"[CODE_LATTICE] Jointly aligned across 5 non-Qwen families + Qwen3-0.6B: train={int(tr_keep.sum())}/{len(tr_keep)} ({len(tr_pts_f)} pts), eval={int(ev_keep.sum())}/{len(ev_keep)} ({len(ev_pts_f)} pts)", flush=True)

    pack = {
        "tr_ids_flat": tr_ids_f, "tr_ids_off": tr_ids_o,
        "tr_pts_flat": tr_pts_f, "tr_pts_off": tr_pts_o,
        "ev_ids_flat": ev_ids_f, "ev_ids_off": ev_ids_o,
        "ev_pts_flat": ev_pts_f, "ev_pts_off": ev_pts_o,
        "ev_pt_groups": ev_grps,
        "corpus_sha256": np.array(_corpus_sha256()),
    }
    report = {
        "n_train_examples": int(tr_keep.sum()),
        "n_train_pts": int(len(tr_pts_f)),
        "n_eval_examples": int(ev_keep.sum()),
        "n_eval_pts": int(len(ev_pts_f)),
        "teachers": list(TEACHER_KEYS),
        "blind_judges": list(BLIND_KEYS),
        "by_depth": {},
    }

    for d in DEPTHS:
        Z_tr, Z_ev, var_ex = {}, {}, {}
        for k in ALL_KEYS:
            X_tr = _slice_keep(raw[k][f"tr_pts_{d}"], raw[k]["tr_cnt"], raw[k]["tr_ok"], tr_keep)
            X_ev = _slice_keep(raw[k][f"ev_pts_{d}"], raw[k]["ev_cnt"], raw[k]["ev_ok"], ev_keep)
            Z_tr[k], Z_ev[k], var_ex[k] = pca_project(X_tr, X_ev, d_out=d_plato)

        # GPA Consensus strictly on 3 rival teachers (ZERO Qwen, ZERO Phi-4, ZERO OLMo 3!)
        C_tr, C_ev = gpa_consensus([Z_tr[k] for k in TEACHER_KEYS], [Z_ev[k] for k in TEACHER_KEYS], n_iters=10)

        pack[f"code_r3_{d}_tr"] = C_tr.astype(np.float32)
        pack[f"code_r3_{d}_ev"] = C_ev.astype(np.float32)
        # Also save single-teacher Gemma 4 targets so we can ablate 3-family GPA Consensus vs Single Teacher!
        pack[f"code_gemma4_{d}_tr"] = Z_tr["gemma4_e4b"].astype(np.float32)
        pack[f"code_gemma4_{d}_ev"] = Z_ev["gemma4_e4b"].astype(np.float32)
        for bk in BLIND_KEYS:
            pack[f"{bk}_{d}_ev"] = Z_ev[bk].astype(np.float32)

        # Evaluate EVERY single teacher vs 3-Family GPA Consensus against strictly BLIND Phi-4 & OLMo 3!
        depth_rep = {"pca_var_exp": {k: round(v, 4) for k, v in var_ex.items()}, "blind_oof": {}}
        for src_name, src_arr in [("GPA_R3_Consensus", C_ev)] + [(tk, Z_ev[tk]) for tk in TEACHER_KEYS]:
            res_b = {}
            for bk in BLIND_KEYS:
                res_b[bk] = eval_oof_2fold(src_arr, Z_ev[bk], ev_grps.tolist())
            res_b["mean_r2_frob"] = round(float(np.mean([res_b[bk]["r2_frob"] for bk in BLIND_KEYS])), 5)
            res_b["mean_r2_opt"] = round(float(np.mean([res_b[bk]["r2_opt"] for bk in BLIND_KEYS])), 5)
            res_b["mean_cos"] = round(float(np.mean([res_b[bk]["cos"] for bk in BLIND_KEYS])), 5)
            res_b["mean_top1"] = round(float(np.mean([res_b[bk]["top1"] for bk in BLIND_KEYS])), 5)
            depth_rep["blind_oof"][src_name] = res_b
        report["by_depth"][f"depth_{d}"] = depth_rep
        print(
            f"[CODE_LATTICE] DEPTH {d}% BLIND OOF (Phi-4 + OLMo 3): "
            f"GPA_R3={depth_rep['blind_oof']['GPA_R3_Consensus']['mean_r2_frob']:.4f} (opt={depth_rep['blind_oof']['GPA_R3_Consensus']['mean_r2_opt']:.4f}, top1={depth_rep['blind_oof']['GPA_R3_Consensus']['mean_top1']:.4f}) | "
            + " | ".join(f"{tk}={depth_rep['blind_oof'][tk]['mean_r2_frob']:.4f}" for tk in TEACHER_KEYS),
            flush=True,
        )

    out_npz = ROOT / "targets_qwen600m_code_lattice_30_50_70.npz"
    _atomic_savez(out_npz, **pack)
    rep_path = ROOT / "code_lattice_blind_oof_report.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"[CODE_LATTICE] DONE! Packed {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB) and {rep_path.name}", flush=True)


def main():
    for key in ALL_KEYS:
        extract_model_3strata(key)
    build_consensus_and_pack_qwen600m(d_plato=256)


if __name__ == "__main__":
    main()
