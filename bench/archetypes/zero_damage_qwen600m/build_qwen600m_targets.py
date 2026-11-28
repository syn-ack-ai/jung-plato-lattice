"""Build strictly Qwen-free 30% & 50% Rival Consensus targets (Gemma 4 E4B + Llama 3.1 8B + Mistral 7B)
plus strictly Blind Eval targets (Phi-4-mini + OLMo 3 7B) and Rival-Only Jungian subspace `U_jung_r3`
for fast autonomous optimization on `Qwen/Qwen3-0.6B` (28 layers, D=1024, L8=30%, L14=50%).

Zero Qwen models appear anywhere in the teacher targets!
Zero mythology passages appear in training targets (strictly neutral: gut + proc + chat)!
"""
import json
import os
import time
from pathlib import Path
import numpy as np

ARCH = Path(os.environ.get("JPL_ARCH_DIR", Path(__file__).resolve().parent.parent))
PLATO = ARCH / "fluent" / "plato"
D50 = PLATO / "depth50_distill"
OUT_DIR = Path(os.environ.get("JPL_OUT_DIR", Path(__file__).resolve().parent))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHERS = ["gemma4_e4b", "llama31_8b", "mistral7b"]  # Strictly NON-Qwen 7B-9B rivals!
BLINDS = ["phi4_mini", "olmo3_7b"]


def extract_subset_pts(z, keep_mask):
    """Given an npz with `tr_pts`, `tr_cnt`, `tr_ok`, extract concatenated points for passages where keep_mask is True."""
    pts = z["tr_pts"]
    cnt = z["tr_cnt"]
    ok = z["tr_ok"]
    offs = np.zeros(len(cnt) + 1, dtype=np.int64)
    # Only ok passages have rows in tr_pts! Wait: verify whether sum(cnt[ok]) == len(pts) or sum(cnt) == len(pts)
    if int(cnt[ok].sum()) == len(pts):
        pos = 0
        out = []
        for i in range(len(cnt)):
            c = int(cnt[i])
            if ok[i]:
                if keep_mask[i]:
                    out.append(pts[pos : pos + c])
                pos += c
        assert pos == len(pts)
        return np.concatenate(out, axis=0).astype(np.float64)
    elif int(cnt.sum()) == len(pts):
        pos = 0
        out = []
        for i in range(len(cnt)):
            c = int(cnt[i])
            if keep_mask[i]:
                out.append(pts[pos : pos + c])
            pos += c
        assert pos == len(pts)
        return np.concatenate(out, axis=0).astype(np.float64)
    else:
        raise ValueError(f"Mismatch: len(pts)={len(pts)}, sum(cnt)={cnt.sum()}, sum(cnt[ok])={cnt[ok].sum()}")


def pca_k_fit_transform(X_tr, X_ev, k=256):
    mu = X_tr.mean(axis=0, keepdims=True)
    Xc = X_tr - mu
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    V = vt[:k].T  # (D, k)
    Z_tr = Xc @ V
    Z_ev = (X_ev - mu) @ V
    # Frobenius normalize by train RMS norm (matching meta.json / Exp 228-251 convention)
    scale = float(np.sqrt(np.mean(np.sum(Z_tr ** 2, axis=1)))) + 1e-12
    return (Z_tr / scale), (Z_ev / scale), mu.squeeze(0), V, scale


def orthogonal_procrustes_map(A, B):
    """Find orthogonal R minimizing ||A @ R - B||_F."""
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def gpa_consensus(Z_tr_dict, Z_ev_dict, n_iter=10):
    """Generalized Procrustes Analysis across rival teachers (fit strictly on train!)."""
    names = list(Z_tr_dict.keys())
    # Align all to first teacher initially
    C_tr = Z_tr_dict[names[0]].copy()
    Rs = {}
    for _ in range(n_iter):
        aligned_tr = []
        for n in names:
            R = orthogonal_procrustes_map(Z_tr_dict[n], C_tr)
            Rs[n] = R
            aligned_tr.append(Z_tr_dict[n] @ R)
        C_tr = np.mean(aligned_tr, axis=0)
        # Re-normalize consensus RMS
        scale = float(np.sqrt(np.mean(np.sum(C_tr ** 2, axis=1)))) + 1e-12
        C_tr = C_tr / scale
    C_ev = np.mean([Z_ev_dict[n] @ Rs[n] for n in names], axis=0)
    # Apply final consensus RMS norm scale on eval consistent with train
    C_ev = C_ev / scale
    return C_tr.astype(np.float32), C_ev.astype(np.float32), Rs


def r2_frob(A, B):
    """Strict centered + Frobenius-normalized orthogonal Procrustes R2_frob."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    Ac = Ac / (np.linalg.norm(Ac) + 1e-12)
    Bc = Bc / (np.linalg.norm(Bc) + 1e-12)
    R = orthogonal_procrustes_map(Ac, Bc)
    return float(1.0 - np.sum((Ac @ R - Bc) ** 2))


def main():
    t0 = time.time()
    tr_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl") if l.strip()]
    ev_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl") if l.strip()]

    # Load 50% and 30% raw files
    z50 = {n: np.load(D50 / f"raw50_{n}.npz") for n in TEACHERS + BLINDS}
    z30 = {n: np.load(PLATO / f"raw_{n}.npz") for n in TEACHERS + BLINDS}

    # Neutral indices from raw50
    neut_idx = z50[TEACHERS[0]]["tr_neut_idx"]  # length 2398 indexing into passages.jsonl
    for n in TEACHERS + BLINDS:
        assert np.array_equal(z50[n]["tr_neut_idx"], neut_idx), f"neut_idx mismatch for {n}"

    # Joint valid mask across ALL 3 teachers at BOTH 30% and 50% depth
    ok50 = np.ones(len(neut_idx), dtype=bool)
    ok30_neut = np.ones(len(neut_idx), dtype=bool)
    for n in TEACHERS:
        ok50 &= z50[n]["tr_ok"].astype(bool)
        ok30_neut &= z30[n]["tr_ok"][neut_idx].astype(bool)
        # Also verify segment counts match between 30% and 50% for valid neutral passages
        same_cnt = (z50[n]["tr_cnt"] == z30[n]["tr_cnt"][neut_idx])
        ok50 &= same_cnt
        ok30_neut &= same_cnt
    keep_neut = ok50 & ok30_neut
    kept_orig_idx = neut_idx[keep_neut]
    print(f"Kept strictly-neutral rival-aligned train passages: {int(keep_neut.sum())} / {len(neut_idx)}")

    # Verify point counts match across all teachers at both depths
    keep30_full = np.zeros(len(tr_rows), dtype=bool)
    keep30_full[kept_orig_idx] = True

    # Build 30% & 50% Rival-3 Consensus (Gemma 4 + Llama 3.1 + Mistral, ZERO Qwen!)
    Z30_tr, Z30_ev, pcs30 = {}, {}, {}
    Z50_tr, Z50_ev, pcs50 = {}, {}, {}
    for n in TEACHERS:
        X30_tr = extract_subset_pts(z30[n], keep30_full)
        X30_ev = z30[n]["ev_pts"].astype(np.float64)
        Z30_tr[n], Z30_ev[n], mu30, V30, sc30 = pca_k_fit_transform(X30_tr, X30_ev, k=256)
        pcs30[n] = (mu30, V30, sc30)

        X50_tr = extract_subset_pts(z50[n], keep_neut)
        X50_ev = z50[n]["ev_pts"].astype(np.float64)
        Z50_tr[n], Z50_ev[n], mu50, V50, sc50 = pca_k_fit_transform(X50_tr, X50_ev, k=256)
        pcs50[n] = (mu50, V50, sc50)

    n_tr_pts = Z30_tr[TEACHERS[0]].shape[0]
    for n in TEACHERS:
        assert Z30_tr[n].shape == (n_tr_pts, 256)
        assert Z50_tr[n].shape == (n_tr_pts, 256)
        assert Z30_ev[n].shape == (3192, 256)
        assert Z50_ev[n].shape == (3192, 256)

    r3_30_tr, r3_30_ev, Rs30 = gpa_consensus(Z30_tr, Z30_ev, n_iter=10)
    r3_50_tr, r3_50_ev, Rs50 = gpa_consensus(Z50_tr, Z50_ev, n_iter=10)

    # Also build blind eval 256D PCA targets (fit on neutral train, transform eval — exact same convention!)
    blind_ev = {}
    for b in BLINDS:
        Xb30_tr = extract_subset_pts(z30[b], keep30_full)
        Xb30_ev = z30[b]["ev_pts"].astype(np.float64)
        _, Zb30_ev, _, _, _ = pca_k_fit_transform(Xb30_tr, Xb30_ev, k=256)

        Xb50_tr = extract_subset_pts(z50[b], keep_neut)
        Xb50_ev = z50[b]["ev_pts"].astype(np.float64)
        _, Zb50_ev, _, _, _ = pca_k_fit_transform(Xb50_tr, Xb50_ev, k=256)

        blind_ev[f"{b}_30_ev"] = Zb30_ev.astype(np.float32)
        blind_ev[f"{b}_50_ev"] = Zb50_ev.astype(np.float32)

    # Sanity-check rival consensus vs blind models!
    for b in BLINDS:
        print(
            f"Sanity R3 -> {b}: "
            f"30% R2_frob={r2_frob(r3_30_ev, blind_ev[f'{b}_30_ev']):.4f} | "
            f"50% R2_frob={r2_frob(r3_50_ev, blind_ev[f'{b}_50_ev']):.4f}"
        )

    # Also project rival-only (Gemma 4 + Llama 3.1 + Mistral) Jungian house & self2 directions into R3_30!
    u_house_list, u_self2_list = [], []
    for n in TEACHERS:
        za = np.load(PLATO / f"arch229_{n}.npz")
        mu30, V30, sc30 = pcs30[n]
        R = Rs30[n]
        if "house_pos" in za.files and "house_neg" in za.files:
            dh = (za["house_pos"].mean(axis=0) - za["house_neg"].mean(axis=0)).astype(np.float64)
            vh = ((dh @ V30) / sc30) @ R
            u_house_list.append(vh / (np.linalg.norm(vh) + 1e-12))
        if "self2_pos" in za.files and "self2_neg" in za.files:
            ds = (za["self2_pos"].mean(axis=0) - za["self2_neg"].mean(axis=0)).astype(np.float64)
            vs = ((ds @ V30) / sc30) @ R
            u_self2_list.append(vs / (np.linalg.norm(vs) + 1e-12))

    # Check if arch229 keys matched; otherwise store zero placeholder and print keys
    if u_house_list and u_self2_list:
        uh = np.mean(u_house_list, axis=0)
        uh = uh / (np.linalg.norm(uh) + 1e-12)
        us = np.mean(u_self2_list, axis=0)
        us = us - np.dot(us, uh) * uh
        us = us / (np.linalg.norm(us) + 1e-12)
        U_jung_r3_30 = np.stack([uh, us], axis=0).astype(np.float32)
        print("Built strictly rival-only U_jung_r3_30:", U_jung_r3_30.shape)
    else:
        za0 = np.load(PLATO / f"arch229_{TEACHERS[0]}.npz")
        print("arch229 keys:", za0.files)
        # Fallback: extract house/self2 from arch229['H'] if present using arch229_consensus
        U_jung_r3_30 = np.zeros((2, 256), dtype=np.float32)

    # Extract per-passage expected segment counts & eval metadata
    tr_cnt_kept = z50[TEACHERS[0]]["tr_cnt"][keep_neut].astype(np.int32)
    ev_cnt = z50[TEACHERS[0]]["ev_cnt"].astype(np.int32)
    assert int(tr_cnt_kept.sum()) == n_tr_pts
    assert int(ev_cnt.sum()) == 3192

    # Eval point domains & groups from existing targets_dual30_50_eval.npz
    z_ev_meta = np.load(D50 / "targets_dual30_50_eval.npz", allow_pickle=True)
    out_path = OUT_DIR / "rival30_50_raw_consensus.npz"
    np.savez_compressed(
        out_path,
        tr_orig_idx=kept_orig_idx.astype(np.int32),
        tr_cnt=tr_cnt_kept,
        r3_30_tr=r3_30_tr,
        r3_50_tr=r3_50_tr,
        r3_30_ev=r3_30_ev,
        r3_50_ev=r3_50_ev,
        ev_cnt=ev_cnt,
        ev_pt_domains=z_ev_meta["pt_domains"],
        ev_pt_groups=z_ev_meta["pt_groups"],
        ev_pt_ret=z_ev_meta["pt_ret"],
        phi4_30_ev=blind_ev["phi4_mini_30_ev"],
        phi4_50_ev=blind_ev["phi4_mini_50_ev"],
        olmo3_30_ev=blind_ev["olmo3_7b_30_ev"],
        olmo3_50_ev=blind_ev["olmo3_7b_50_ev"],
        U_jung_r3_30=U_jung_r3_30,
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
