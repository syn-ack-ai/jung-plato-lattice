"""What the ending axis carries (README addendum 94, part B). Saved captures plus labels.
  python moral.py label   (DeepSeek) lesson type of each item's true version, blind, random order -> moral/labels.jsonl
  python moral.py report  (numpy, scipy, scikit-learn) -> moral/report.md, report.json"""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
OUT = HERE / "moral"; OUT.mkdir(exist_ok=True)
LAB = OUT / "labels.jsonl"
NB, NN = int(os.environ.get("NB_MORAL", 2000)), int(os.environ.get("NN_MORAL", 500))
TYPES = {"a": "wrongdoing punished", "b": "virtue rewarded", "c": "another moral lesson", "d": "no moral lesson"}
ASK = "Here is a short story:\n{t}\nWhich fits it best?\n" + "\n".join(f"{k}: {v}" for k, v in TYPES.items()) + '\nReply with JSON only: {{"type": "a letter"}}.'
SETS = {"fresh": "items3", "items7": "items7", "items9": "items9_d31"}


def label():
    import crosslang as CL, inv79
    k = CL.key()
    done = {(r["set"], r["kind"], r["id"]) for r in map(json.loads, open(LAB)) if r["type"] != "failed"} if LAB.exists() else set()
    todo = [(n, r[0], i, r[1]) for n, b in SETS.items() for i, r in inv79.items(b).items() if (n, r[0], i) not in done]
    random.Random(94).shuffle(todo)

    def one(x):
        for _ in range(3):
            try:
                y, _u = CL.call(k, ASK.format(t=x[3]))
                v = str(y.get("type")).strip().lower()
                if v in TYPES:
                    return {"set": x[0], "kind": x[1], "id": x[2], "type": TYPES[v]}
            except Exception:
                pass
        return {"set": x[0], "kind": x[1], "id": x[2], "type": "failed"}

    with ThreadPoolExecutor(8) as ex, open(LAB, "a") as f:
        for row in ex.map(one, todo):
            f.write(json.dumps(row) + "\n"); f.flush()
    print("labelled", len(todo))


def auc(x, y):
    """P(x[y] > x[~y]) by ranks; y boolean"""
    r = rankdata(x); n1 = y.sum(); n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def folds(y, k=5, seed=94):
    """fold number per item, stratified by label"""
    g = np.random.default_rng(seed); out = np.zeros(len(y), int)
    for v in set(y):
        i = np.where(y == v)[0]; g.shuffle(i); out[i] = np.arange(len(i)) % k
    return out


def crossfit(X, ends, batch, fd):
    """position on the ending axis, each fold scored with an axis from the others, centred by batch inside the training folds"""
    s = np.zeros(len(X))
    for f in set(fd):
        te, tr = fd == f, fd != f
        XT = X[tr].copy()
        for b in set(batch[tr]):
            mb = batch[tr] == b
            XT[mb] -= XT[mb].mean(0)
        et = ends[tr]
        v = XT[et == "true"].mean(0) - XT[et == "foil"].mean(0); v = v / (np.linalg.norm(v) + 1e-12)
        s[te] = (X[te] - X[tr].mean(0)) @ v
    return s


def auc_batch(s, y, batch, mask):
    v = []
    for b in set(batch):
        m = mask & (batch == b)
        if y[m].sum() >= 2 and (~y[m]).sum() >= 2:
            v.append(auc(s[m], y[m]))
    return float(np.mean(v)) if v else float("nan")


def report():
    import cross_analyze as CA, inv79, freq_analyze as FA
    M = {(r["set"], r["kind"], r["id"]): r["type"] for r in map(json.loads, open(LAB))}
    E = {(r["set"], r["kind"], r["id"]): r["better"] for r in map(json.loads, open(HERE / "endings" / "labels.jsonl"))}
    name = {"items7": "items7", "items9_d31": "items9"}
    X, meta = {}, None
    for n in inv79.MODELS:
        d = CA.load(n + "_f"); k = inv79.keys(n, "joint")
        rows, x = {}, {}
        for s in "AB":
            F = CA.stories(d, s, "true")[0] - CA.stories(d, s, "foil")[0]
            rows[s] = [("fresh", s, str(i)) for i in FA.mats(d, s)[1]] + [(name[b], s, str(i)) for b, i in k[s][1]]
            x[s] = np.vstack([F, k[s][0]])
        assert meta is None or meta == rows
        meta, X[n] = rows, x
    types = {s: np.array([M[r] for r in meta[s]]) for s in "AB"}
    ends = {s: np.array([E[r] for r in meta[s]]) for s in "AB"}
    batch = {s: np.array([r[0] for r in meta[s]]) for s in "AB"}
    if os.environ.get("MORAL_SHUFFLE"):                          # a check on the method: endings shuffled within batch, expect about 0.5
        g0 = np.random.default_rng(int(os.environ["MORAL_SHUFFLE"]))
        for s in "AB":
            for b in set(batch[s]):
                m = batch[s] == b
                ends[s][m] = g0.permutation(ends[s][m])
    fd = {s: folds(ends[s]) for s in "AB"}
    S = {s: {n: crossfit(X[n][s], ends[s], batch[s], fd[s]) for n in X} for s in "AB"}
    g = np.random.default_rng(94)

    def test(s, mask, y):
        est = float(np.mean([auc_batch(S[s][n], y, batch[s], mask) for n in X]))
        idx = np.where(mask)[0]
        boots = []
        for _ in range(NB):
            i = np.concatenate([g.choice(np.where(batch[s][idx] == b)[0], (batch[s][idx] == b).sum()) for b in set(batch[s][idx])])
            j = idx[i]
            boots.append(np.nanmean([auc_batch(S[s][n][j], y[j], batch[s][j], np.ones(len(j), bool)) for n in X]))
        null = []
        for _ in range(NN):
            e2 = ends[s].copy()
            for b in set(batch[s]):
                m = batch[s] == b
                e2[m] = g.permutation(e2[m])
            yy = (e2 == "true") if y is YEND[s] else y
            mk = (mask & np.isin(e2, CLEAR)) if y is YEND[s] else mask
            f2 = folds(e2, seed=int(g.integers(1 << 30)))
            null.append(np.nanmean([auc_batch(crossfit(X[n][s], e2, batch[s], f2), yy, batch[s], mk) for n in X]))
        return {"n": [int(y[mask].sum()), int((~y[mask]).sum())], "auc": est, "ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))], "p": float((1 + np.sum(np.array(null) >= est)) / (NN + 1)), "null_mean": float(np.nanmean(null))}

    CLEAR = ("true", "foil")
    YEND = {s: ends[s] == "true" for s in "AB"}
    res = {"types": {s: {t: int((types[s] == t).sum()) for t in list(TYPES.values()) + ["failed"]} for s in "AB"}}
    res["moral_share"] = {s: float(np.isin(types[s], ["wrongdoing punished", "virtue rewarded", "another moral lesson"]).mean()) for s in "AB"}
    t = types["A"]
    res["B1"] = test("A", np.isin(t, ["wrongdoing punished", "virtue rewarded"]), t == "virtue rewarded")
    for s, kn in (("A", "myths"), ("B", "mechanisms")):
        m = (types[s] == "no moral lesson") & np.isin(ends[s], CLEAR)
        y = YEND[s]
        ok = y[m].sum() >= 5 and (~y[m]).sum() >= 5
        res[f"B3_{kn}"] = test(s, m, y) if ok else {"n": [int(y[m].sum()), int((~y[m]).sum())], "auc": None}
    f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    ci = lambda r: f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r.get("ci") else ""  # noqa: E731
    L = ["# What the ending axis carries (addendum 94, part B, cross-fitted)", "", "| test | groups | AUC | 95% | p, endings shuffled and axis refitted |", "|---|---|---|---|---|",
         f"| B1 myths: virtue rewarded against wrongdoing punished | {res['B1']['n']} | {f(res['B1']['auc'])} | {ci(res['B1'])} | {res['B1']['p']:.4f} (null mean {res['B1']['null_mean']:.3f}) |"]
    for kn in ("myths", "mechanisms"):
        r = res[f"B3_{kn}"]
        L.append(f"| B3 {kn} with no moral lesson: true ends better against foil ends better | {r['n']} | {f(r['auc'])} | {ci(r)} | {'' if r['auc'] is None else format(r['p'], '.4f') + ' (null mean ' + format(r['null_mean'], '.3f') + ')'} |")
    L += ["", "Lesson types: " + "; ".join(f"{'myths' if s == 'A' else 'mechanisms'} {res['types'][s]}" for s in "AB"), "", f"Share with a moral lesson: myths {res['moral_share']['A']:.3f}, mechanisms {res['moral_share']['B']:.3f}."]
    above = lambda r: r["ci"][0] > 0.5 and r["p"] < 0.05  # noqa: E731    # the interval is conditional on one axis; shuffled-label runs passed it alone
    b1 = above(res["B1"])
    tested = [res[f"B3_{k}"] for k in ("myths", "mechanisms") if res[f"B3_{k}"]["auc"] is not None]
    if not tested:
        res["verdict"] = "B3 not testable"
    elif any(above(r) for r in tested):
        res["verdict"] = "outcome in general, which also sorts kinds of lesson" if b1 else "outcome in general, and it does not sort kinds of lesson"
    else:
        res["verdict"] = "specifically moral" if b1 else "the moral reading fails"
    L += ["", f"Reading by the registered rule: **{res['verdict']}**."]
    if not os.environ.get("MORAL_SHUFFLE"):
        (OUT / "report.json").write_text(json.dumps(res, indent=1)); (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print("MORAL_DONE")


if __name__ == "__main__":
    {"label": label, "report": report}[sys.argv[1]]()
