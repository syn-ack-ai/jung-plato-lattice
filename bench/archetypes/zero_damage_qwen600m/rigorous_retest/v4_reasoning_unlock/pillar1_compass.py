"""Pillar 1 Engine: Test-Time Platonic Trajectory Compass (`Exp 263`).

Zero weight friction (`WikiText-2 drift = 0.00000` guaranteed) and ZERO test-time 31B inference!
Strict Scientific Protocol:
  - Universal space (`Whitened GCCA`, `Standard GPA`, or `31B-only`) is built EXCLUSIVELY from
    rival teachers (`Gemma 4 31B`, `Llama 3.1 8B`, `Mistral 7B`) on `train` split of
    `verifier_pairs_code.jsonl` (strictly decontaminated `code_train.jsonl`).
  - Cross-family student map (`Qwen3-0.6B -> Universal Space`) is 100% UNSUPERVISED (Procrustes,
    Ridge, or 2-layer MLP trained purely on hidden-state alignment MSE; ZERO correctness labels!).
  - Universal Trajectory Verifier (`V_univ`) is trained 100% EXCLUSIVELY on rival teachers'
    universal coordinates (ZERO `0.6B` labels, ZERO `0.6B` activations!).
  - Every Compass variant is evaluated on BOTH `K=8` stochastic candidates AND `Greedy + K=8` (9 cands)
    on disjoint `dev_mbpp_170` (already graded ONCE in `bwrap`), with 10,000-resample paired bootstrap
    CIs against ALL fair baselines:
      1. Greedy pass@1
      2. Random@1 (mean K=8)
      3. Self-LogProb (mean & sum)
      4. AST-Valid + Self-LogProb
      5. Self-Consistency (AST majority + Self-LogProb)
      6. Prompted 0.6B Self-Audit (`Yes - No` logit)
      7. Own-0.6B Native Probe (same verifier trained directly on 0.6B native states!)
      8. Shuffled-Map Negative Control
"""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
DEPTHS = (30, 50, 70, 85)
TEACHERS = ("gemma4_31b", "llama31_8b", "mistral7b")


def _whiten_pca(X_tr, X_all, d_out=256, reg=0.05, whiten=True):
    mu = X_tr.mean(0, keepdims=True)
    Xc = X_tr - mu
    n, D = Xc.shape
    k = min(d_out, D, n - 1)
    if D <= 1200:
        cov = (Xc.T @ Xc) / float(n)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1][:k]
        V = vecs[:, idx]
        ev = np.maximum(vals[idx], 1e-8)
    else:
        # Fast exact 3-step randomized subspace iteration for top-k right singular vectors of (n, D)
        rng = np.random.default_rng(26300)
        q_dim = min(D, k + 32)
        Omega = rng.standard_normal((D, q_dim))
        Y = Xc @ Omega
        Q, _ = np.linalg.qr(Y, mode="reduced")
        for _ in range(3):
            Z = Xc.T @ Q
            Qz, _ = np.linalg.qr(Z, mode="reduced")
            Y = Xc @ Qz
            Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ Xc  # (q_dim, D)
        _, S, Vt = np.linalg.svd(B, full_matrices=False)
        V = Vt[:k].T
        ev = np.maximum((S[:k] ** 2) / float(n), 1e-8)
    if whiten:
        scale = 1.0 / np.sqrt(ev + reg * float(np.mean(ev)))
        P = V * scale[None, :]
    else:
        P = V
    Z_tr = Xc @ P
    rms = float(np.sqrt(np.mean(np.sum(Z_tr ** 2, axis=1)))) + 1e-12
    return (X_all - mu) @ (P / rms), {"mu": mu, "P": P / rms}


def _apply_pca(X, params):
    return (X - params["mu"]) @ params["P"]


def _fit_procrustes(A, B):
    ma, mb = A.mean(0, keepdims=True), B.mean(0, keepdims=True)
    Ac, Bc = A - ma, B - mb
    U, _, Vt = np.linalg.svd(Bc.T @ Ac, full_matrices=False)
    R = U @ Vt
    s = float(np.linalg.norm(Bc, "fro") / max(1e-12, np.linalg.norm(Ac @ R.T, "fro")))
    return {"ma": ma, "mb": mb, "R": R, "s": s}


def _apply_procrustes(A, p):
    return p["mb"] + p["s"] * ((A - p["ma"]) @ p["R"].T)


def _fit_ridge(A, B, lam=10.0):
    ma, mb = A.mean(0, keepdims=True), B.mean(0, keepdims=True)
    Ac, Bc = A - ma, B - mb
    d = Ac.shape[1]
    W = np.linalg.solve(Ac.T @ Ac + lam * np.eye(d), Ac.T @ Bc)
    return {"ma": ma, "mb": mb, "W": W}


def _apply_ridge(A, p):
    return p["mb"] + (A - p["ma"]) @ p["W"]


def fit_universal_teacher_space(t_raw: dict, ok_mask: np.ndarray, tr_row_mask: np.ndarray,
                                depth: int, space_mode: str = "gcca", d_out: int = 256, reg: float = 0.05):
    """Build universal teacher space at `depth` strictly from `tr_row_mask` rows.
    Returns:
      - `C_all`: (N_ok, 6, d_out) consensus coordinates
      - `T_aligned`: dict[teacher -> (N_ok, 6, d_out)] each teacher mapped into universal space
      - `diag`: alignment fidelity diagnostics
    """
    n_ok = int(ok_mask.sum())
    n_seg = 6
    tr_pts = np.repeat(tr_row_mask, n_seg)  # (N_ok * 6,)

    use_teachers = ("gemma4_31b",) if space_mode == "31b_only" else TEACHERS
    whiten = (space_mode == "gcca")

    Z_t = {}
    for tk in use_teachers:
        X = t_raw[tk][f"h_{depth}"].astype(np.float64).reshape(-1, t_raw[tk][f"h_{depth}"].shape[-1])
        Z, _ = _whiten_pca(X[tr_pts], X, d_out=d_out, reg=reg, whiten=whiten)
        Z_t[tk] = Z

    if len(use_teachers) == 1:
        C_flat = Z_t[use_teachers[0]]
        T_al = {use_teachers[0]: C_flat.reshape(n_ok, n_seg, -1)}
        return C_flat.reshape(n_ok, n_seg, -1), T_al, {"mode": space_mode, "d_out": C_flat.shape[1]}

    if space_mode == "gcca":
        # Exact Whitened Multi-View GCCA (MAXVAR SVD on concatenated whitened views, fit on tr_pts!)
        M_tr = np.concatenate([Z_t[tk][tr_pts] for tk in use_teachers], axis=1)
        # Compute top d_out right singular vectors via covariance (fast & exact!)
        # Actually left singular vectors G_tr satisfy G_tr = M_tr @ V / S; use covariance of views:
        cov_m = M_tr.T @ M_tr
        vals, vecs = np.linalg.eigh(cov_m)
        idx = np.argsort(vals)[::-1][:d_out]
        V_top = vecs[:, idx]
        inv_s = 1.0 / np.sqrt(np.maximum(vals[idx], 1e-8))
        # Consensus on all points: average of linear projections of each view into GCCA frame!
        M_all = np.concatenate([Z_t[tk] for tk in use_teachers], axis=1)
        C_flat = (M_all @ V_top) * inv_s[None, :]
        # Map each individual teacher into C_flat via train-only Procrustes
        T_al = {}
        for tk in use_teachers:
            pr = _fit_procrustes(Z_t[tk][tr_pts], C_flat[tr_pts])
            T_al[tk] = _apply_procrustes(Z_t[tk], pr).reshape(n_ok, n_seg, d_out)
        # Re-average aligned teachers for exact symmetry
        C_flat = np.mean([T_al[tk].reshape(-1, d_out) for tk in use_teachers], axis=0)
        return C_flat.reshape(n_ok, n_seg, d_out), T_al, {"mode": "gcca", "d_out": d_out}

    # Standard 10-iteration Generalized Procrustes Analysis (GPA) fit strictly on tr_pts
    C_tr = Z_t[use_teachers[0]][tr_pts].copy()
    for _ in range(10):
        aligned_tr = []
        for tk in use_teachers:
            pr = _fit_procrustes(Z_t[tk][tr_pts], C_tr)
            aligned_tr.append(_apply_procrustes(Z_t[tk][tr_pts], pr))
        C_tr = np.mean(aligned_tr, axis=0)
        C_tr = C_tr / (np.sqrt(np.mean(np.sum(C_tr ** 2, axis=1))) + 1e-12)

    T_al = {}
    for tk in use_teachers:
        pr = _fit_procrustes(Z_t[tk][tr_pts], C_tr)
        T_al[tk] = _apply_procrustes(Z_t[tk], pr).reshape(n_ok, n_seg, d_out)
    C_all = np.mean([T_al[tk] for tk in use_teachers], axis=0)
    return C_all, T_al, {"mode": "gpa", "d_out": d_out}


def fit_unsupervised_student_map(X_s_pairs, C_univ_pairs, tr_row_mask, X_s_cands,
                                 map_type="procrustes", d_out=256, whiten=True, ridge_lam=25.0, seed=263):
    """100% UNSUPERVISED alignment map from Qwen3-0.6B hidden states -> Universal 31B space.
    ZERO correctness labels used!"""
    n_p, n_seg, D_s = X_s_pairs.shape
    n_c = X_s_cands.shape[0]
    tr_pts = np.repeat(tr_row_mask, n_seg)

    Xp_flat = X_s_pairs.reshape(-1, D_s).astype(np.float64)
    Xc_flat = X_s_cands.reshape(-1, D_s).astype(np.float64)
    Cp_flat = C_univ_pairs.reshape(-1, C_univ_pairs.shape[-1]).astype(np.float64)

    if map_type == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Cp_flat))
        Cp_fit = Cp_flat[perm]
    else:
        Cp_fit = Cp_flat

    if map_type in ("procrustes", "shuffled"):
        Zp, pca_p = _whiten_pca(Xp_flat[tr_pts], Xp_flat, d_out=d_out, reg=0.05, whiten=whiten)
        Zc = _apply_pca(Xc_flat, pca_p)
        pr = _fit_procrustes(Zp[tr_pts], Cp_fit[tr_pts])
        Sp_al = _apply_procrustes(Zp, pr).reshape(n_p, n_seg, -1)
        Sc_al = _apply_procrustes(Zc, pr).reshape(n_c, n_seg, -1)
    elif map_type == "ridge":
        # Direct regularized linear map from full 1024-dim student residual stream -> universal 256-dim space!
        rd = _fit_ridge(Xp_flat[tr_pts], Cp_fit[tr_pts], lam=ridge_lam)
        Sp_al = _apply_ridge(Xp_flat, rd).reshape(n_p, n_seg, -1)
        Sc_al = _apply_ridge(Xc_flat, rd).reshape(n_c, n_seg, -1)
    elif map_type == "ridge_whitened":
        Zp, pca_p = _whiten_pca(Xp_flat[tr_pts], Xp_flat, d_out=min(512, D_s), reg=0.05, whiten=True)
        Zc = _apply_pca(Xc_flat, pca_p)
        rd = _fit_ridge(Zp[tr_pts], Cp_fit[tr_pts], lam=ridge_lam)
        Sp_al = _apply_ridge(Zp, rd).reshape(n_p, n_seg, -1)
        Sc_al = _apply_ridge(Zc, rd).reshape(n_c, n_seg, -1)
    else:
        raise ValueError(f"Unknown map_type: {map_type}")

    # Cal-split OOF alignment R^2 diagnostic (purely unsupervised!)
    cal_pts = ~tr_pts
    tc = Cp_flat[cal_pts]
    pc = Sp_al.reshape(-1, Cp_flat.shape[-1])[cal_pts]
    r2_cal = 1.0 - float(np.sum((tc - pc) ** 2)) / max(1e-12, float(np.sum((tc - tc.mean(0, keepdims=True)) ** 2)))
    # Top-1 retrieval on cal split (subset up to 1000 points for speed)
    m_sub = min(len(tc), 1000)
    pn = pc[:m_sub] / (np.linalg.norm(pc[:m_sub], axis=1, keepdims=True) + 1e-12)
    tn = tc[:m_sub] / (np.linalg.norm(tc[:m_sub], axis=1, keepdims=True) + 1e-12)
    top1_cal = float(np.mean(np.argmax(pn @ tn.T, axis=1) == np.arange(m_sub)))
    return Sp_al, Sc_al, {"cal_r2": round(r2_cal, 4), "cal_top1": round(top1_cal, 4)}


def extract_trajectory_features(Z_by_depth: dict, depths, feat_mode: str = "audit_plus_code"):
    """Build per-trajectory feature vector from universal coordinates Z[d] of shape (N, 6, d_out).
    Checkpoints: 0=prob_end, 1=def_sig, 2=body1, 3=body2, 4=code_end, 5=audit_end.
    """
    parts = []
    for d in depths:
        Z = Z_by_depth[d]  # (N, 6, dim)
        if feat_mode == "code_end":
            f = Z[:, 4, :]
        elif feat_mode == "audit_end":
            f = Z[:, 5, :]
        elif feat_mode == "audit_plus_code":
            f = np.concatenate([Z[:, 4, :], Z[:, 5, :]], axis=1)
        elif feat_mode == "deltas_plan_body_audit":
            # Reasoning transition vectors: problem->sig, sig->end, end->audit!
            d1 = Z[:, 1, :] - Z[:, 0, :]
            d2 = Z[:, 4, :] - Z[:, 1, :]
            d3 = Z[:, 5, :] - Z[:, 4, :]
            f = np.concatenate([d1, d2, d3], axis=1)
        elif feat_mode == "full_trajectory_deltas":
            # All 5 step transitions + final states
            ds = [Z[:, k + 1, :] - Z[:, k, :] for k in range(5)]
            f = np.concatenate(ds + [Z[:, 4, :], Z[:, 5, :]], axis=1)
        elif feat_mode == "code_deltas_only":
            # Pure reasoning transitions inside code (NO audit prompt at all!)
            ds = [Z[:, k + 1, :] - Z[:, k, :] for k in range(4)]
            f = np.concatenate(ds + [Z[:, 4, :]], axis=1)
        elif feat_mode == "audit_minus_code":
            # Pure audit shift relative to code_end + audit_end
            f = np.concatenate([Z[:, 5, :] - Z[:, 4, :], Z[:, 5, :]], axis=1)
        else:
            raise ValueError(f"Unknown feat_mode: {feat_mode}")
        # L2-normalize each depth block so depths contribute balanced scales
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        parts.append(f)
    return np.concatenate(parts, axis=1)


def fit_pairwise_verifier(F_train_views: list, pair_ids_tr, labels_tr, clf_type: str = "pairwise_ridge", lam: float = 50.0):
    """Train universal trajectory verifier strictly on teacher views (ZERO 0.6B labels!).
    `F_train_views`: list of (N_tr, F_dim) feature matrices (one per rival teacher or consensus).
    """
    diffs, X_all, y_all = [], [], []
    # Pair up clean (1) and buggy (0) for each pair_id (ignoring unlabeled alignment-only rows with label < 0)
    by_pid = {}
    for i, (pid, lab) in enumerate(zip(pair_ids_tr, labels_tr)):
        if int(lab) not in (0, 1) or int(pid) < 0:
            continue
        by_pid.setdefault(int(pid), {})[int(lab)] = i
    valid_pairs = [(d[1], d[0]) for d in by_pid.values() if 1 in d and 0 in d]
    lbl_mask = (labels_tr == 0) | (labels_tr == 1)

    for Fv in F_train_views:
        X_all.append(Fv[lbl_mask])
        y_all.append(labels_tr[lbl_mask])
        pos_i = [p[0] for p in valid_pairs]
        neg_i = [p[1] for p in valid_pairs]
        diffs.append(Fv[pos_i] - Fv[neg_i])

    D_mat = np.concatenate(diffs, axis=0)  # (N_pairs * n_views, F)
    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0).astype(np.float64)

    if clf_type == "mean_diff":
        w = D_mat.mean(axis=0)
        w = w / (np.linalg.norm(w) + 1e-12)
        return {"type": "linear", "w": w}

    if clf_type == "pairwise_ridge":
        # Exact Ridge regression on pairwise diffs: w = (D^T D + lam I)^{-1} D^T 1
        # Uses exact Woodbury dual identity D^T (D D^T + lam I)^{-1} 1 when F_dim > N_rows (1000x faster!)
        N_r, F_dim = D_mat.shape
        if F_dim > N_r:
            G = D_mat @ D_mat.T + lam * np.eye(N_r)
            alpha = np.linalg.solve(G, np.ones(N_r, dtype=np.float64))
            w = D_mat.T @ alpha
        else:
            A = D_mat.T @ D_mat + lam * np.eye(F_dim)
            b = D_mat.sum(axis=0)
            w = np.linalg.solve(A, b)
        w = w / (np.linalg.norm(w) + 1e-12)
        return {"type": "linear", "w": w}

    if clf_type == "pairwise_logistic":
        # Exact L2-regularized Newton/L-BFGS logistic loss on pairwise diffs (minimizes sum log(1+exp(-w.d)) + 0.5*lam*||w||^2)
        from scipy.optimize import minimize
        F_dim = D_mat.shape[1]
        def loss_grad(w):
            z = np.clip(D_mat @ w, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-z))
            loss = float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * lam * np.dot(w, w) / len(D_mat))
            grad = -(D_mat.T @ (1.0 - sig)) / len(D_mat) + (lam / len(D_mat)) * w
            return loss, grad
        res = minimize(loss_grad, np.zeros(F_dim), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        w = res.x / (np.linalg.norm(res.x) + 1e-12)
        return {"type": "linear", "w": w}

    if clf_type == "mahalanobis_llr":
        # Shrinkage Gaussian log-likelihood ratio between valid and flawed universal trajectories
        X_pos = X_cat[y_cat == 1]
        X_neg = X_cat[y_cat == 0]
        mu_p, mu_n = X_pos.mean(0), X_neg.mean(0)
        # Shared shrinkage covariance -> Linear Discriminant Analysis with exact ridge regularization
        Sp = np.cov(X_pos.T) if len(X_pos) > 2 else np.eye(X_cat.shape[1])
        Sn = np.cov(X_neg.T) if len(X_neg) > 2 else np.eye(X_cat.shape[1])
        S = 0.5 * (Sp + Sn) + lam * np.eye(X_cat.shape[1]) / len(X_cat)
        w = np.linalg.solve(S, mu_p - mu_n)
        w = w / (np.linalg.norm(w) + 1e-12)
        return {"type": "linear", "w": w}

    raise ValueError(f"Unknown clf_type: {clf_type}")


def score_verifier(F_mat, model):
    return F_mat @ model["w"]


def eval_pairwise_cal_acc(scores, pair_ids, labels):
    by_pid = {}
    for s, pid, lab in zip(scores, pair_ids, labels):
        by_pid.setdefault(int(pid), {})[int(lab)] = float(s)
    hits = [1.0 if d[1] > d[0] else (0.5 if d[1] == d[0] else 0.0) for d in by_pid.values() if 1 in d and 0 in d]
    return float(np.mean(hits)) if hits else 0.0


def zscore_within_task(scores_mat):
    """Standardize candidate scores within each task (170, K) to zero mean, unit std."""
    mu = scores_mat.mean(axis=1, keepdims=True)
    sd = scores_mat.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (scores_mat - mu) / sd


def eval_reranker_on_mbpp(scores_flat, ok_flat, ast_ok_flat, n_tasks=170, K_total=9,
                          include_greedy=False, ast_gate=False):
    """Evaluate candidate reranking on disjoint dev_mbpp_170 (`K=8` stochastic OR `Greedy+8`)."""
    S = scores_flat.reshape(n_tasks, K_total).copy()
    OK = ok_flat.reshape(n_tasks, K_total)
    AST = ast_ok_flat.reshape(n_tasks, K_total)
    start_k = 0 if include_greedy else 1
    S = S[:, start_k:]
    OK = OK[:, start_k:]
    AST = AST[:, start_k:]
    if ast_gate:
        # Penalize syntax-invalid candidates unless ALL candidates fail AST parse
        any_ast = AST.any(axis=1, keepdims=True)
        S = np.where(any_ast & (~AST), -1e9, S)
    pick = np.argmax(S, axis=1)
    chosen_ok = OK[np.arange(n_tasks), pick]
    hits = int(chosen_ok.sum())
    return hits, hits / float(n_tasks), chosen_ok


def paired_bootstrap_bool(a_bool, b_bool, n_boot=10000, seed=263):
    rng = np.random.default_rng(seed)
    diff = a_bool.astype(np.float64) - b_bool.astype(np.float64)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    b = diff[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p_two = float(2.0 * min(np.mean(b <= 0.0), np.mean(b >= 0.0)))
    return {
        "delta_pp": round(100.0 * float(diff.mean()), 2),
        "ci95_lo_pp": round(100.0 * float(lo), 2),
        "ci95_hi_pp": round(100.0 * float(hi), 2),
        "p_two_sided": round(min(1.0, p_two), 5),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
    }


def load_pillar1_bundle():
    t_raw = {tk: dict(np.load(CACHE / f"p1_teacher_{tk}.npz", allow_pickle=True)) for tk in TEACHERS}
    s_pairs = dict(np.load(CACHE / "p1_student_qwen06b_pairs.npz", allow_pickle=True))
    s_cands = dict(np.load(CACHE / "dev_mbpp_170_qwen06b_k8_states.npz", allow_pickle=True))
    c_meta = json.loads((CACHE / "dev_mbpp_170_qwen06b_k8_cands.json").read_text())

    # Intersect valid teacher rows (all 3 teachers ok)
    ok_mask = np.ones_like(t_raw[TEACHERS[0]]["row_ok"], dtype=bool)
    for tk in TEACHERS:
        ok_mask &= t_raw[tk]["row_ok"].astype(bool)

    # Align teacher arrays (which only stored row_ok rows for each teacher!) to common ok_mask
    t_aligned = {}
    for tk in TEACHERS:
        tk_ok = t_raw[tk]["row_ok"].astype(bool)
        # Map global index -> local stored index in teacher tk
        local_idx = np.cumsum(tk_ok) - 1
        keep_local = local_idx[ok_mask]
        t_aligned[tk] = {f"h_{d}": t_raw[tk][f"h_{d}"][keep_local] for d in DEPTHS}

    s_pairs_ok = {f"h_{d}": s_pairs[f"h_{d}"][ok_mask] for d in DEPTHS}
    for k in ("labels", "pair_ids", "is_cal", "code_sum_lp", "code_mean_lp", "audit_yes_no_diff"):
        s_pairs_ok[k] = s_pairs[k][ok_mask]

    return t_aligned, s_pairs_ok, s_cands, c_meta, ok_mask


def compute_all_baselines(s_cands, c_meta, s_pairs_ok):
    """Compute EVERY fair non-Platonic baseline on BOTH K=8 and Greedy+8 pools."""
    ok = s_cands["ok"].astype(bool)
    ast_ok = s_cands["ast_ok"].astype(bool)
    n_tasks = int(c_meta["n_tasks"])
    K_tot = int(c_meta["K_stoch"]) + 1

    # Self-Consistency score: frequency of candidate's canonical AST hash within its task (+ tiny mean_lp tie-break)
    sc_scores = np.zeros(len(ok), dtype=np.float64)
    cands = c_meta["candidates"]
    for ti in range(n_tasks):
        sl = slice(ti * K_tot, (ti + 1) * K_tot)
        hashes = [cands[j]["ast_hash"] for j in range(ti * K_tot + 1, (ti + 1) * K_tot)]
        cnt = Counter(h for h in hashes if h is not None)
        for j in range(ti * K_tot, (ti + 1) * K_tot):
            h = cands[j]["ast_hash"]
            freq = cnt.get(h, 0) if h is not None else 0
            sc_scores[j] = float(freq) + 0.01 * float(s_cands["code_mean_lp"][j])

    base_signals = {
        "self_logprob_mean": s_cands["code_mean_lp"].astype(np.float64),
        "self_logprob_sum": s_cands["code_sum_lp"].astype(np.float64),
        "self_consistency": sc_scores,
        "self_audit_yes_no": s_cands["audit_yes_no_diff"].astype(np.float64),
    }

    out = {
        "greedy": {
            "hits": int(ok.reshape(n_tasks, K_tot)[:, 0].sum()),
            "pass1": round(float(ok.reshape(n_tasks, K_tot)[:, 0].mean()), 5),
            "bools": ok.reshape(n_tasks, K_tot)[:, 0],
        },
        "random_k8_mean": round(float(ok.reshape(n_tasks, K_tot)[:, 1:].mean()), 5),
        "oracle_k8_hits": int(ok.reshape(n_tasks, K_tot)[:, 1:].any(axis=1).sum()),
        "oracle_k9_hits": int(ok.reshape(n_tasks, K_tot).any(axis=1).sum()),
    }
    for name, sig in base_signals.items():
        for ast_g in (False, True):
            tag = f"{name}{'_ast' if ast_g else ''}"
            h8, p8, b8 = eval_reranker_on_mbpp(sig, ok, ast_ok, n_tasks, K_tot, include_greedy=False, ast_gate=ast_g)
            h9, p9, b9 = eval_reranker_on_mbpp(sig, ok, ast_ok, n_tasks, K_tot, include_greedy=True, ast_gate=ast_g)
            out[tag] = {"k8_hits": h8, "k8_pass1": round(p8, 5), "k8_bools": b8,
                        "k9_hits": h9, "k9_pass1": round(p9, 5), "k9_bools": b9}

    # Own-0.6B Native Probe baseline (same pairwise_ridge verifier trained DIRECTLY on 0.6B's own native states!)
    tr_mask = ~s_pairs_ok["is_cal"]
    cal_mask = s_pairs_ok["is_cal"]
    for feat_mode in ("audit_plus_code", "full_trajectory_deltas"):
        F_tr = extract_trajectory_features({d: s_pairs_ok[f"h_{d}"].astype(np.float64) for d in DEPTHS}, DEPTHS, feat_mode)
        F_c = extract_trajectory_features({d: s_cands[f"h_{d}"].astype(np.float64) for d in DEPTHS}, DEPTHS, feat_mode)
        v_own = fit_pairwise_verifier([F_tr[tr_mask]], s_pairs_ok["pair_ids"][tr_mask], s_pairs_ok["labels"][tr_mask],
                                      clf_type="pairwise_ridge", lam=50.0)
        cal_acc = eval_pairwise_cal_acc(score_verifier(F_tr[cal_mask], v_own), s_pairs_ok["pair_ids"][cal_mask], s_pairs_ok["labels"][cal_mask])
        sc_c = score_verifier(F_c, v_own)
        h8, p8, b8 = eval_reranker_on_mbpp(sc_c, ok, ast_ok, n_tasks, K_tot, include_greedy=False, ast_gate=True)
        h9, p9, b9 = eval_reranker_on_mbpp(sc_c, ok, ast_ok, n_tasks, K_tot, include_greedy=True, ast_gate=True)
        out[f"own_06b_native_{feat_mode}"] = {
            "cal_pair_acc": round(cal_acc, 4),
            "k8_hits": h8, "k8_pass1": round(p8, 5), "k8_bools": b8,
            "k9_hits": h9, "k9_pass1": round(p9, 5), "k9_bools": b9,
            "scores": sc_c,
        }
    return out


def run_compass_experiment(t_aligned, s_pairs_ok, s_cands, ok_mask, cfg: dict):
    """Run a single Pillar 1 Platonic Compass configuration and return full metrics + candidate scores."""
    depths = tuple(cfg.get("depths", (30, 50, 70, 85)))
    space_mode = cfg.get("space_mode", "gcca")         # gcca | gpa | 31b_only
    map_type = cfg.get("map_type", "procrustes")       # procrustes | ridge | ridge_whitened | shuffled
    d_out = int(cfg.get("d_out", 256))
    ridge_lam = float(cfg.get("ridge_lam", 25.0))
    feat_mode = cfg.get("feat_mode", "audit_plus_code")
    clf_type = cfg.get("clf_type", "pairwise_ridge")   # mean_diff | pairwise_ridge | pairwise_logistic | mahalanobis_llr
    clf_lam = float(cfg.get("clf_lam", 50.0))
    train_views = cfg.get("train_views", "all_teachers")  # consensus | all_teachers | 31b_only

    tr_mask = ~s_pairs_ok["is_cal"]
    cal_mask = s_pairs_ok["is_cal"]

    C_by_d, T_by_d, Sp_by_d, Sc_by_d = {}, {}, {}, {}
    align_diag = {}
    for d in depths:
        C_all, T_al, _ = fit_universal_teacher_space(t_aligned, ok_mask, tr_mask, d, space_mode=space_mode, d_out=d_out)
        Sp_al, Sc_al, adiag = fit_unsupervised_student_map(
            s_pairs_ok[f"h_{d}"], C_all, tr_mask, s_cands[f"h_{d}"],
            map_type=map_type, d_out=d_out, whiten=(space_mode == "gcca"), ridge_lam=ridge_lam,
        )
        C_by_d[d] = C_all
        T_by_d[d] = T_al
        Sp_by_d[d] = Sp_al
        Sc_by_d[d] = Sc_al
        align_diag[f"d{d}"] = adiag

    # Build teacher feature views (STRICTLY ZERO 0.6B labels/activations!)
    if train_views == "consensus":
        F_views = [extract_trajectory_features(C_by_d, depths, feat_mode)[tr_mask]]
    elif train_views == "31b_only":
        F_views = [extract_trajectory_features({d: T_by_d[d]["gemma4_31b"] for d in depths}, depths, feat_mode)[tr_mask]]
    else:
        tkeys = list(T_by_d[depths[0]].keys())
        F_views = [
            extract_trajectory_features({d: T_by_d[d][tk] for d in depths}, depths, feat_mode)[tr_mask]
            for tk in tkeys
        ]
        if len(tkeys) > 1:
            F_views.append(extract_trajectory_features(C_by_d, depths, feat_mode)[tr_mask])

    verifier = fit_pairwise_verifier(F_views, s_pairs_ok["pair_ids"][tr_mask], s_pairs_ok["labels"][tr_mask],
                                     clf_type=clf_type, lam=clf_lam)

    # Diagnostic 1: Teacher cal-split pairwise accuracy (ceiling of verifier on rival models themselves)
    F_t_cal = extract_trajectory_features(C_by_d, depths, feat_mode)[cal_mask]
    t_cal_acc = eval_pairwise_cal_acc(score_verifier(F_t_cal, verifier), s_pairs_ok["pair_ids"][cal_mask], s_pairs_ok["labels"][cal_mask])

    # Diagnostic 2: Zero-shot Student cal-split pairwise accuracy (0.6B projected -> universal verifier!)
    F_s_cal = extract_trajectory_features(Sp_by_d, depths, feat_mode)[cal_mask]
    s_cal_acc = eval_pairwise_cal_acc(score_verifier(F_s_cal, verifier), s_pairs_ok["pair_ids"][cal_mask], s_pairs_ok["labels"][cal_mask])

    # Test-time Compass score on all 1,530 dev_mbpp_170 candidates!
    F_s_cand = extract_trajectory_features(Sc_by_d, depths, feat_mode)
    compass_scores = score_verifier(F_s_cand, verifier)

    ok = s_cands["ok"].astype(bool)
    ast_ok = s_cands["ast_ok"].astype(bool)
    h8_raw, p8_raw, b8_raw = eval_reranker_on_mbpp(compass_scores, ok, ast_ok, include_greedy=False, ast_gate=False)
    h8_ast, p8_ast, b8_ast = eval_reranker_on_mbpp(compass_scores, ok, ast_ok, include_greedy=False, ast_gate=True)
    h9_ast, p9_ast, b9_ast = eval_reranker_on_mbpp(compass_scores, ok, ast_ok, include_greedy=True, ast_gate=True)

    return {
        "cfg": cfg,
        "align_diag": align_diag,
        "teacher_cal_pair_acc": round(t_cal_acc, 4),
        "student_zs_cal_pair_acc": round(s_cal_acc, 4),
        "k8_raw_hits": h8_raw, "k8_raw_pass1": round(p8_raw, 5), "k8_raw_bools": b8_raw,
        "k8_ast_hits": h8_ast, "k8_ast_pass1": round(p8_ast, 5), "k8_ast_bools": b8_ast,
        "k9_ast_hits": h9_ast, "k9_ast_pass1": round(p9_ast, 5), "k9_ast_bools": b9_ast,
        "compass_scores": compass_scores,
    }
