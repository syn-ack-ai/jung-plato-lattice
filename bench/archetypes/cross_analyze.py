"""Cross-company analysis (README addenda 7-8) on cross/NAME.npz from cross.py.
  uv run --with numpy --with scikit-learn --with scipy python cross_analyze.py NAME=COMPANY ...
One layer per model, fixed in advance: 60% depth. Every comparison asserts the texts match by hash.
X1  shared map: representational similarity (Spearman between two models' 30x30 cosine-similarity matrices) on myth
    retellings, mechanism retellings, myth anchors, mechanism anchors, neutral passages, and per-item key-step
    differences (true retelling - foil). Item bootstrap drops pairs of duplicated items (review fix).
X2  direction transfer: model 1's "same pattern" direction (learned on held-out folds of items, standardized in model
    1's space) scored on model 2's claims mapped into model 1's space by a ridge map fitted only on the 150 neutral
    prompts. Nulls (review fix): 200 within-item label shuffles of the direction (one shuffle shared by all pairs per
    draw, for the overall test) and 50 row permutations of the neutral fit (map null). Per-pair p = share of null
    draws >= the real AUC. Ceiling: model 2's own held-out probe.
X3  item agreement: Spearman across models of per-item log-odds margin, logit(yes|true) - mean logit(yes|decoys)
    (review fix: probability margins saturate near 0).
Fresh item set (addendum 9): pass names ending in _f (cross.py --fresh captures). Adds: item-bootstrap CI for X3; for
the key step, a lexical control (tf-idf vectors of the same texts as a model with no understanding: each model's
agreement with it, and model-model agreement with the lexical similarity, foil type and writer partialled out); 2000
bootstrap draws instead of 1000."""
import hashlib, itertools, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

CROSS = Path(__file__).parent / "cross"
RNG = np.random.default_rng(0)
DEPTH = 0.6
N_SHUF, N_MAP = 200, 200


def load(name):
    d = dict(np.load(CROSS / f"{name}.npz"))
    L = d["story"].shape[1]; l = int(round(DEPTH * (L - 1)))
    d["S"] = d["story"][:, l].astype(np.float64); d["C"] = d["claim"][:, l].astype(np.float64); d["layer"] = (l, L)
    d["logit"] = d["logp_yes"] - d["logp_no"]
    return d


def stories(d, s, role):
    m = (d["s_set"] == s) & (d["s_role"] == role)
    o = np.argsort(d["s_item"][m], kind="stable")
    return d["S"][m][o], d["s_hash"][m][o]


def x1_sets(d):
    out, hashes = {}, {}
    for s, lab in (("A", "myth"), ("B", "mechanism")):
        (anc, ha), (tru, ht), (foil, hf) = (stories(d, s, r) for r in ("anchor", "true", "foil"))
        out[f"{lab} retellings"], hashes[f"{lab} retellings"] = tru, ht
        out[f"{lab} anchors"], hashes[f"{lab} anchors"] = anc, ha
        out[f"{lab} key step (true - foil)"], hashes[f"{lab} key step (true - foil)"] = tru - foil, np.char.add(ht, hf)
    out["neutral passages"], hashes["neutral passages"] = stories(d, "N", "neutral")
    return out, hashes


def simmat(X):
    X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X @ X.T


def rsa_full(M1, M2):
    iu = np.triu_indices(len(M1), 1)
    return spearmanr(M1[iu], M2[iu]).statistic


def rsa_boot(M1, M2, idx):
    a, c = np.triu_indices(len(idx), 1); keep = idx[a] != idx[c]
    i, j = idx[a][keep], idx[c][keep]
    return spearmanr(M1[i, j], M2[i, j]).statistic


def rsa_partial(M1, M2, controls, idx=None):
    """Spearman between two similarity matrices with control matrices partialled out (residuals of ranks on the
    controls' ranks); idx: bootstrap resample of items (pairs of the same item dropped)"""
    if idx is None:
        i, j = np.triu_indices(len(M1), 1)
    else:
        a, c = np.triu_indices(len(idx), 1); keep = idx[a] != idx[c]; i, j = idx[a][keep], idx[c][keep]
    Z = np.column_stack([np.ones(len(i))] + [rankdata(C[i, j]) for C in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
    e1, e2 = res(rankdata(M1[i, j])), res(rankdata(M2[i, j]))
    return float(e1 @ e2 / np.sqrt((e1 @ e1) * (e2 @ e2)))


def lexical(d, fresh):
    """key-step (true - foil) tf-idf differences per set, ordered like stories(), texts checked against the capture's
    hashes; plus same-foil-type and same-writer matrices (fresh set only)"""
    sys.path.insert(0, str(Path(__file__).parent))
    if fresh:
        import fresh as F
        src = {"A": (F.A, F.A2), "B": (F.B, F.B2)}
    else:
        from items import A, B
        from items_v2 import A2, B2
        src = {"A": (A, A2), "B": (B, B2)}
    text = {}
    for s, (items, ex) in src.items():
        for iid, _, anchor, true, _ in items:
            text[s, iid, "anchor"], text[s, iid, "true"], text[s, iid, "foil"] = anchor, true, ex[iid][1]
    vec = TfidfVectorizer(sublinear_tf=True).fit(list(text.values()))
    out = {}
    for s in "AB":
        rows = {}
        for role in ("true", "foil"):
            m = (d["s_set"] == s) & (d["s_role"] == role); o = np.argsort(d["s_item"][m], kind="stable")
            ids, hs = d["s_item"][m][o], d["s_hash"][m][o]
            t = [text[s, i, role] for i in ids]
            assert [hashlib.sha1(x.encode()).hexdigest()[:16] for x in t] == list(hs), f"texts differ from capture ({s} {role})"
            rows[role] = vec.transform(t).toarray()
        ctl = [simmat(rows["true"] - rows["foil"])]
        if fresh:
            for attr in ("BRK", "WRITER"):
                lab = np.array([getattr(F, attr)[i] for i in ids]); ctl.append((lab[:, None] == lab[None, :]).astype(float))
        out[s] = ctl
    return out


def mknn(M1, M2, k):
    """mutual k-nearest-neighbour alignment (Huh et al. 2024; Balestriero & LeCun 2026): mean over items of the share
    of an item's k nearest neighbours (cosine, itself excluded) that the two models have in common"""
    nn = lambda M: np.argsort(-(M - 1e9 * np.eye(len(M))), axis=1)[:, :k]  # noqa: E731
    a, b = nn(M1), nn(M2)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def dom(Z, y):
    return Z[y == 1].mean(0) - Z[y == 0].mean(0)


def own_probe_auc(C, y, g):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(C, y, g):
        mu, sd = C[tr].mean(0), C[tr].std(0) + 1e-6
        s[te] = ((C[te] - mu) / sd) @ dom((C[tr] - mu) / sd, y[tr]); s[te] -= s[te].mean()
    return roc_auc_score(y, s)


class Transfer:
    """model 2 claims of one set -> model 1 space via a ridge map fitted on the neutral claims; exact dual solution, so
    scores for any model-1 direction v are K @ (Yc @ v) + const"""

    def __init__(self, d1, d2, s):
        m1, m2, f1, f2 = (d["c_set"] == x for d, x in ((d1, s), (d2, s), (d1, "N"), (d2, "N")))
        assert (d1["c_hash"][m1] == d2["c_hash"][m2]).all() and (d1["c_hash"][f1] == d2["c_hash"][f2]).all()
        self.y = (d1["c_cand"][m1] == "true").astype(int); self.g = d1["c_item"][m1]
        self.C1 = d1["C"][m1]
        X, Y = d2["C"][f2], d1["C"][f1]
        xm, xs = X.mean(0), X.std(0) + 1e-6; self.ym = Y.mean(0); self.Yc = Y - self.ym
        Xz = (X - xm) / xs
        alpha = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Xz, self.Yc).alpha_
        Kfit = Xz @ Xz.T
        self.K = ((d2["C"][m2] - xm) / xs) @ Xz.T @ np.linalg.inv(Kfit + alpha * np.eye(len(Kfit)))
        self.folds = list(GroupKFold(5).split(self.C1, self.y, self.g))

    def auc(self, ylab=None, perm=None, scores=False):
        """ylab: labels used to learn the direction (default the true ones); perm: row permutation of the neutral fit"""
        ylab = self.y if ylab is None else ylab
        Yc = self.Yc if perm is None else self.Yc[perm]
        s = np.zeros(len(self.y))
        for tr, te in self.folds:
            mu, sd = self.C1[tr].mean(0), self.C1[tr].std(0) + 1e-6
            v = dom((self.C1[tr] - mu) / sd, ylab[tr]) / sd
            s[te] = self.K[te] @ (Yc @ v); s[te] -= s[te].mean()   # centred within fold (review 2: fold offsets)
        return s if scores else roc_auc_score(self.y, s)


def shuffle_within(y):
    return np.concatenate([RNG.permutation(y[i: i + 4]) for i in range(0, len(y), 4)])


def main(args):
    names = [a.split("=")[0] for a in args]; comp = dict(a.split("=") for a in args)
    fresh = names[0].endswith("_f"); assert all(n.endswith("_f") == fresh for n in names), "mix of fresh and original captures"
    NB = 2000 if fresh else 1000
    D = {n: load(n) for n in names}
    print("## Models (layer used = 60% depth)\n")
    print("| model | company | layer | myth answer AUC | mechanism answer AUC | own probe AUC myths | own probe AUC mechanisms | item claims with yes/no mass < 0.5 |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        d = D[n]; row = []
        for s in "AB":
            m = d["c_set"] == s; row.append(roc_auc_score((d["c_cand"][m] == "true").astype(int), d["logit"][m]))
        for s in "AB":
            m = d["c_set"] == s
            row.append(own_probe_auc(d["C"][m], (d["c_cand"][m] == "true").astype(int), d["c_item"][m]))
        print(f"| {n} | {comp[n]} | {d['layer'][0]}/{d['layer'][1]} | " + " | ".join(f"{v:.3f}" for v in row)
              + f" | {int((d['mass'][d['c_set'] != 'N'] < 0.5).sum())} |")
    pairs = list(itertools.combinations(names, 2)); cross_pairs = [p for p in pairs if comp[p[0]] != comp[p[1]]]
    print(f"\n## X1. Shared map: representational similarity between models ({len(cross_pairs)} cross-company pairs)\n")
    SM, H = {}, {}
    for n in names:
        sets, hs = x1_sets(D[n]); SM[n] = {k: simmat(v) for k, v in sets.items()}; H[n] = hs
    keys = list(SM[names[0]])
    for p in pairs:
        for k in keys:
            assert (H[p[0]][k] == H[p[1]][k]).all(), (p, k)
    R = {k: {p: rsa_full(SM[p[0]][k], SM[p[1]][k]) for p in pairs} for k in keys}
    for k in keys:
        v = [R[k][p] for p in cross_pairs]; same = [R[k][p] for p in pairs if comp[p[0]] == comp[p[1]]]
        print(f"- {k}: cross-company mean {np.mean(v):.3f} (min {np.min(v):.3f}, max {np.max(v):.3f})"
              + (f"; same company {np.mean(same):.3f}" if same else ""))
    for a, b in (("myth retellings", "mechanism retellings"), ("myth key step (true - foil)", "mechanism key step (true - foil)"),
                 ("myth anchors", "mechanism anchors")):
        diff = [R[a][p] - R[b][p] for p in cross_pairs]; boots = []
        na, nb = len(SM[names[0]][a]), len(SM[names[0]][b])
        for _ in range(NB):
            ia, ib = RNG.integers(0, na, na), RNG.integers(0, nb, nb)
            boots.append(np.mean([rsa_boot(SM[p[0]][a], SM[p[1]][a], ia) - rsa_boot(SM[p[0]][b], SM[p[1]][b], ib) for p in cross_pairs]))
        print(f"- {a} minus {b}: mean {np.mean(diff):+.3f}, item-bootstrap 95% CI [{np.percentile(boots, 2.5):+.3f}, "
              f"{np.percentile(boots, 97.5):+.3f}]; higher for myths in {sum(x > 0 for x in diff)} of {len(diff)} pairs")
    if fresh:
        K = 10
        print(f"\n### X1 mutual {K}-nearest-neighbour alignment (addendum 12; chance {K / (len(SM[names[0]][keys[0]]) - 1):.3f})\n")
        MK = {k: {p: mknn(SM[p[0]][k], SM[p[1]][k], K) for p in cross_pairs} for k in keys}
        for k in keys:
            print(f"- {k}: cross-company mean {np.mean(list(MK[k].values())):.3f}")
        for a, b in (("myth retellings", "mechanism retellings"), ("myth key step (true - foil)", "mechanism key step (true - foil)")):
            dd = [MK[a][p] - MK[b][p] for p in cross_pairs]
            print(f"- {a} minus {b}: mean {np.mean(dd):+.3f}; higher for myths in {sum(x > 0 for x in dd)} of {len(dd)} pairs")
        print("\n### X1 key step: lexical control (addendum 9)\n")
        LX = lexical(D[names[0]], fresh)
        ka, kb = "myth key step (true - foil)", "mechanism key step (true - foil)"
        na, nb = len(SM[names[0]][ka]), len(SM[names[0]][kb])
        lex_r = {s: [rsa_full(SM[n][k], LX[s][0]) for n in names] for s, k in (("A", ka), ("B", kb))}
        boots = []
        for _ in range(NB):
            ia, ib = RNG.integers(0, na, na), RNG.integers(0, nb, nb)
            boots.append(np.mean([rsa_boot(SM[n][ka], LX["A"][0], ia) - rsa_boot(SM[n][kb], LX["B"][0], ib) for n in names]))
        print(f"- each model vs the tf-idf model: myths {np.mean(lex_r['A']):.3f}, mechanisms {np.mean(lex_r['B']):.3f}; "
              f"myths minus mechanisms {np.mean(lex_r['A']) - np.mean(lex_r['B']):+.3f}, 95% CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")
        part = {p: rsa_partial(SM[p[0]][ka], SM[p[1]][ka], LX["A"]) - rsa_partial(SM[p[0]][kb], SM[p[1]][kb], LX["B"]) for p in cross_pairs}
        boots = []
        for _ in range(NB):
            ia, ib = RNG.integers(0, na, na), RNG.integers(0, nb, nb)
            boots.append(np.mean([rsa_partial(SM[p[0]][ka], SM[p[1]][ka], LX["A"], ia) - rsa_partial(SM[p[0]][kb], SM[p[1]][kb], LX["B"], ib) for p in cross_pairs]))
        pa = np.mean([rsa_partial(SM[p[0]][ka], SM[p[1]][ka], LX["A"]) for p in cross_pairs])
        print(f"- model-model, with tf-idf similarity, same foil type and same writer partialled out: myths {pa:.3f}; "
              f"myths minus mechanisms {np.mean(list(part.values())):+.3f}, 95% CI [{np.percentile(boots, 2.5):+.3f}, "
              f"{np.percentile(boots, 97.5):+.3f}]; higher for myths in {sum(v > 0 for v in part.values())} of {len(part)} pairs")
    print("\n## X2. Direction transfer across companies (held-out-fold direction; map fitted on neutral prompts only)\n")
    for s, lab in (("A", "myths"), ("B", "mechanisms")):
        T = [Transfer(D[a], D[b], s) for p in cross_pairs for a, b in (p, p[::-1])]
        real = np.array([t.auc() for t in T])
        ys = T[0].y
        shuf = np.array([[t.auc(ylab=yl) for t in T] for yl in (shuffle_within(ys) for _ in range(N_SHUF))])   # draws x pairs
        perms = [RNG.permutation(len(T[0].Yc)) for _ in range(N_MAP)]
        mapn = np.array([[t.auc(perm=pm) for t in T] for pm in perms])
        pv = lambda null, r: ((null >= r).sum(0) + 1) / (len(null) + 1)  # noqa: E731
        p_pair, p_map = pv(shuf, real), pv(mapn, real)
        print(f"- {lab}: mean AUC {real.mean():.3f} over {len(T)} ordered pairs (min {real.min():.3f}, max {real.max():.3f})")
        print(f"  - map-permutation null (primary): mean {mapn.mean():.3f}; overall p {pv(mapn.mean(1), real.mean()):.4f}; "
              f"pairs with p < 0.05: {(p_map < 0.05).sum()} of {len(T)}")
        print(f"  - label-shuffle null: mean {shuf.mean():.3f}; overall p {pv(shuf.mean(1), real.mean()):.4f}; pairs with p < 0.05: {(p_pair < 0.05).sum()} of {len(T)}")
        SC = np.array([t.auc(scores=True) for t in T]); ylab = T[0].y; groups = np.arange(len(ylab)) // 4
        if s == "A":
            realA, SCA, yA = real, SC, ylab
    def boot_mean_auc(SCx, yx):
        it = RNG.integers(0, len(yx) // 4, len(yx) // 4); rows = (it[:, None] * 4 + np.arange(4)).ravel()
        return np.mean([roc_auc_score(yx[rows], sc[rows]) for sc in SCx])
    bd = [boot_mean_auc(SCA, yA) - boot_mean_auc(SC, ylab) for _ in range(NB)]
    print(f"- myths minus mechanisms: mean {realA.mean() - real.mean():+.3f}, item-bootstrap 95% CI [{np.percentile(bd, 2.5):+.3f}, "
          f"{np.percentile(bd, 97.5):+.3f}] (higher for myths in {(realA > real).sum()} of {len(real)} ordered pairs; pairs share models, so that count is descriptive)")
    print("\n## X3. Item agreement: Spearman across models of per-item log-odds margin\n")
    MG = {}
    for s, lab in (("A", "myths"), ("B", "mechanisms")):
        marg = {}
        for n in names:
            d = D[n]; m = d["c_set"] == s; items = d["c_item"][m]; c = d["c_cand"][m]; z = d["logit"][m]
            marg[n] = np.array([z[(items == i) & (c == "true")].mean() - z[(items == i) & (c != "true")].mean() for i in dict.fromkeys(items)])
        MG[s] = marg
        v = [spearmanr(marg[a], marg[b]).statistic for a, b in cross_pairs]
        print(f"- {lab}: cross-company mean {np.nanmean(v):.3f} (min {np.nanmin(v):.3f}, max {np.nanmax(v):.3f}); NaN pairs {int(np.isnan(v).sum())}")
    if fresh:
        mean_r = lambda s, idx: np.nanmean([spearmanr(MG[s][a][idx], MG[s][b][idx]).statistic for a, b in cross_pairs])  # noqa: E731
        na, nb = len(MG["A"][names[0]]), len(MG["B"][names[0]])
        boots = [mean_r("A", RNG.integers(0, na, na)) - mean_r("B", RNG.integers(0, nb, nb)) for _ in range(NB)]
        real = mean_r("A", np.arange(na)) - mean_r("B", np.arange(nb))
        print(f"- myths minus mechanisms: {real:+.3f}, item-bootstrap 95% CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
