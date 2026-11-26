"""Is the pattern an attractor? (README addendum 88, plans/attractor_test.md; user: "this seems to be a reifnorced attractor inspite of the language"; "add the attractor test to the plan".)
Addendum 87's invented stories, vectors only, nothing asked. Tests 1 and 2 of the plan; test 3 (settling in a loop) waits for the retrofit track.
  capture NAME PATH   (Mac, MLX) cut, swap and shuffled versions at five depths, and pushes followed upward -> attractor/caps_NAME.npz
  report              -> attractor/report.md"""
import json, os, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent; OUT = HERE / "attractor"; sys.path.insert(0, str(HERE))
SEED = 88; FRAC = (0.2, 0.4, 0.6, 0.8, 1.0); DEPTH = 2; LANGS = ("en", "el", "ja", "zh")


def sentences(t, lang):
    return [q.strip() for q in (re.split(r"(?<=[.!?;·])\s+", t) if lang in ("en", "el") else re.split(r"(?<=[。！？])", t)) if q.strip()]


def join(S, lang):
    return (" " if lang in ("en", "el") else "").join(S)


def versions(rows):
    """per story: intact true and foil; cut (the sentences before the first that differs between true and foil); swap (the sentence before the cut replaced by one from another pattern's story in that language); shuffled (the true story's sentences in a fixed random order) and shuffled + the same swap"""
    rng = np.random.default_rng(SEED); by = {}; out = []
    for r in rows:
        by.setdefault(r["lang"], []).append(r)
    for r in rows:
        lg = r["lang"]; A, B = sentences(r["true"], lg), sentences(r["foil"], lg); j = next((i for i, (x, y) in enumerate(zip(A, B)) if x != y), None)
        if j is None or j < 1 or len(A) < 3:
            continue
        others = [x for x in by[lg] if x["pattern_id"] != r["pattern_id"]]; donor = sentences(others[int(rng.integers(len(others)))]["true"], lg); filler = donor[int(rng.integers(len(donor)))]; k = int(rng.integers(0, j)); sw = A[:k] + [filler] + A[k + 1:]; rest = [i for i in range(len(A)) if i != k]; rest = [rest[i] for i in rng.permutation(len(rest))]; o = rest[:k] + [k] + rest[k:]; sh = [A[i] for i in o]; shs = sh[:]; shs[k] = filler   # review: the swapped sentence stays at the same position in the shuffled text
        m = int(rng.choice([i for i in range(len(A)) if i != j]))   # the user's prediction: remove the key sentence, or another one
        out.append((r, {"true": r["true"], "foil": r["foil"], "cut": join(A[:j], lg), "swap": join(sw, lg), "shuf": join(sh, lg), "shuf_swap": join(shs, lg), "drop_key": join(A[:j] + A[j + 1:], lg), "drop_other": join(A[:m] + A[m + 1:], lg), "_j": j, "_n": len(A)}))
    return out


def capture(name, path):
    import mlx.core as mx
    import cross
    from mlx_lm import load
    OUT.mkdir(exist_ok=True); f = OUT / f"caps_{name}.npz"
    if f.exists():
        print(name, "already captured", flush=True); return
    model, tok = load(path); cls = type(model.layers[0]); orig = cls.__call__; REC = []; ST = {"add": None, "at": None, "n": 0}

    def rec(self, *a, **k):
        o = orig(self, *a, **k); h = o[0] if isinstance(o, tuple) else o
        if ST["add"] is not None and ST["n"] == ST["at"]:
            h = h + ST["add"].astype(h.dtype); o = (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        ST["n"] += 1; REC.append(h[0]); return o
    cls.__call__ = rec; nl = len(model.layers); idx = [int(round(fr * (nl - 1))) for fr in FRAC]
    def states(text, add=None, at=None):
        REC.clear(); ST.update(add=add, at=at, n=0); model(mx.array([cross.text_ids(tok, text)])); assert len(REC) == nl, (len(REC), nl); h = mx.stack([REC[j][1:].astype(mx.float32).mean(0) for j in idx]); mx.eval(h); a = np.array(h); assert np.isfinite(a).all() and np.isfinite(a.astype(np.float16)).all(), (text[:30],); return a
    C = {(c["kind"], c["pattern_id"], c["lang"]): c for c in map(json.loads, open(HERE / "crosslang" / "checked.jsonl")) if c["set"] == "I"}
    R = [r for r in map(json.loads, open(HERE / "crosslang" / "items.jsonl")) if r.get("ok") and r["set"] == "I" and (lambda c: bool(c) and c["p31"] >= 0.7 and c["mass"] >= 0.5)(C.get((r["kind"], r["pattern_id"], r["lang"])))]
    V = versions(R); meta, H = [], []; rng = np.random.default_rng(SEED); L0 = idx[0]; nz = []
    print(name, "stories", len(V), "of", len(R), "| by kind", {k: sum(1 for r, _ in V if r["kind"] == k) for k in ("myth", "mechanism")}, flush=True)
    keys = {}
    for n, (r, vs) in enumerate(V):
        kept = {}
        for ver, text in vs.items():
            if ver.startswith("_"):
                continue
            kept[ver] = states(text); H.append(kept[ver].astype(np.float16)); meta.append((r["kind"], r["pattern_id"], r["lang"], ver))
        d = kept["true"][0] - kept["foil"][0]; size = float(np.linalg.norm(kept["true"][0]) * 0.1); u = d / (np.linalg.norm(d) + 1e-9); w = rng.standard_normal(d.shape); w -= (w @ u) * u; w /= np.linalg.norm(w); others = [x for (g, q), x in keys.items() if g == r["lang"] and q != r["pattern_id"]]
        keys[r["lang"], r["pattern_id"]] = u; dirs = [("key", u), ("rand", w)] + ([("other", others[int(rng.integers(len(others)))])] if others else [])      # review: another pattern's key step is the fair control; a random direction sits off the data
        for lab, v in dirs:
            ad = mx.array((v * size).astype(np.float32)); pushed = states(vs["true"], ad, L0); nz.append((r["kind"], r["pattern_id"], r["lang"], lab, *[float(np.linalg.norm(pushed[q] - kept["true"][q]) / (np.linalg.norm(kept["true"][q]) + 1e-9)) for q in range(len(idx))]))
        if (n + 1) % 25 == 0:
            print(name, n + 1, "/", len(V), flush=True)
    tmp = OUT / f"caps_{name}.tmp.npz"; np.savez(tmp, H=np.stack(H), meta=np.array(meta), noise=np.array([x[4:] for x in nz], dtype=np.float64), noise_meta=np.array([x[:4] for x in nz]), frac=np.array(FRAC)); os.replace(tmp, f); print(name, "saved", np.stack(H).shape, flush=True)


def report():
    rng = np.random.default_rng(SEED); Fs = sorted(f for f in OUT.glob("caps_*.npz") if ".tmp" not in f.name); assert Fs, "no captures"; Sg = {(r["kind"], r["pattern_id"], r["lang"]): r["better"] for r in map(json.loads, open(HERE / "crosslang" / "sign.jsonl"))}; pairs = [(a, b) for i, a in enumerate(LANGS) for b in LANGS[i + 1:]]
    lines = ["# Is the pattern an attractor? Tests 1 and 2 (addendum 88 with its amendment)", "", f"- models: {[f.stem[5:] for f in Fs]}. Only test 3, settling in a loop, would license the word attractor; these are feed-forward signs.", ""]

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

    def ci(v, n=10000):
        v = np.array(list(v), dtype=float); v = v[np.isfinite(v)]; assert len(v), "empty"; b = v[rng.integers(0, len(v), (n, len(v)))].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)
    f3 = lambda v: "{:+.4f} [{:+.4f}, {:+.4f}] ({})".format(*ci(v))  # noqa: E731
    T1, T1b, T2d, T2c, T2b, used, psize = {}, {}, {}, {}, {}, {}, []
    for f in Fs:
        name = f.stem[5:]; z = np.load(f, allow_pickle=True); H = z["H"].astype(np.float64); M = z["meta"].astype(str); X = {tuple(m): h for m, h in zip(M, H)}; ids = {k: sorted({m[1] for m in M if m[0] == k}) for k in ("myth", "mechanism")}; a1, a1b, ad, ac = {}, {}, {}, {}
        for kind in ids:
            for a, b in pairs:
                P = [p for p in ids[kind] if all((kind, p, g, "true") in X for g in (a, b))]; used[kind, a, b] = len(P)
                if len(P) < 8:
                    continue
                n = len(P)
                for src, dst in ((a, b), (b, a)):
                    Ca = np.array([X[kind, p, src, "cut"][DEPTH] for p in P]); Ts = np.array([X[kind, p, src, "true"][DEPTH] for p in P]); Td = np.array([X[kind, p, dst, "true"][DEPTH] for p in P]); Fd = np.array([X[kind, p, dst, "foil"][DEPTH] for p in P]); ms, md = Ts.mean(0), Td.mean(0)
                    S = unit(Ca - ms) @ unit(Td - md).T; sc = np.diag(S) - (S.sum(1) - np.diag(S)) / (n - 1)        # T1: the cut telling finds its pattern in another language
                    mid = (Td + Fd).mean(0) / 2; L = unit(Ca - Ca.mean(0)) @ (unit(Td - mid) - unit(Fd - mid)).T; lean = np.diag(L) - (L.sum(1) - np.diag(L)) / (n - 1)     # T1b, with the review's control: the lean toward the intact telling, minus the same lean against other patterns' pairs
                    for j, p in enumerate(P):
                        a1.setdefault((kind, p), []).append(sc[j]); a1b.setdefault((kind, p, str(Sg.get((kind, p, dst)))), []).append(lean[j])
            for p in ids[kind]:
                for g in LANGS:
                    if (kind, p, g, "swap") in X:
                        n1 = np.linalg.norm(X[kind, p, g, "swap"] - X[kind, p, g, "true"], axis=1); n0 = np.linalg.norm(X[kind, p, g, "shuf_swap"] - X[kind, p, g, "shuf"], axis=1); ad.setdefault((kind, p), []).append(n1[3] / max(n1[0], 1e-9) - n0[3] / max(n0[0], 1e-9))      # T2 (amended): the size of the swap's change, 80% over 20% depth, in the story minus in its shuffled sentences
                        gap = lambda u, v: 1 - np.sum(unit(X[kind, p, g, u]) * unit(X[kind, p, g, v]), 1); g1, g0 = gap("swap", "true"), gap("shuf_swap", "shuf"); ac.setdefault((kind, p), []).append(g1[3] / max(g1[0], 1e-9) - g0[3] / max(g0[0], 1e-9))  # noqa: E731
        T1[name] = {k: float(np.mean(v)) for k, v in a1.items()}; T1b[name] = {k: float(np.mean(v)) for k, v in a1b.items()}; T2d[name] = {k: float(np.mean(v)) for k, v in ad.items()}; T2c[name] = {k: float(np.mean(v)) for k, v in ac.items()}
        nm = z["noise_meta"].astype(str); nv = z["noise"]; d = {}
        for m, v in zip(nm, nv):
            d.setdefault((m[0], m[1], m[2]), {})[m[3]] = v[3] / max(v[1], 1e-9); psize.append(v[1])                                    # T2b: the push's size at 80% depth over 40%
        T2b[name] = {}
        for k, v in d.items():
            if "key" in v and "other" in v:
                T2b[name].setdefault((k[0], k[1]), []).append(v["key"] - v["other"])
        T2b[name] = {k: float(np.mean(v)) for k, v in T2b[name].items()}
    pool = lambda D: {k: float(np.mean([D[n][k] for n in D if k in D[n]])) for k in set().union(*[set(D[n]) for n in D])}  # noqa: E731
    # T3, the user's prediction: a story with part of its pattern missing matches less; the key sentence's absence costs more than another sentence's
    T3 = {}
    for f in Fs:
        name = f.stem[5:]; z = np.load(f, allow_pickle=True); H = z["H"].astype(np.float64); M = z["meta"].astype(str); X = {tuple(m): h for m, h in zip(M, H)}; ids = {k: sorted({m[1] for m in M if m[0] == k}) for k in ("myth", "mechanism")}; acc = {}
        for kind in ids:
            for a, b in pairs:
                Q = [p for p in ids[kind] if all((kind, p, g, "true") in X for g in (a, b))]
                if len(Q) < 8:
                    continue
                n = len(Q)
                for src, dst in ((a, b), (b, a)):
                    Td = np.array([X[kind, p, dst, "true"][DEPTH] for p in Q]); md = Td.mean(0); Ts = np.array([X[kind, p, src, "true"][DEPTH] for p in Q]); ms = Ts.mean(0); r = {}
                    for ver in ("true", "drop_other", "drop_key"):
                        V = np.array([X[kind, p, src, ver][DEPTH] for p in Q]); Sx = unit(V - ms) @ unit(Td - md).T; r[ver] = np.diag(Sx) - (Sx.sum(1) - np.diag(Sx)) / (n - 1)
                    for j, p in enumerate(Q):
                        acc.setdefault((kind, p), []).append((r["true"][j], r["drop_other"][j], r["drop_key"][j]))
        T3[name] = {k: np.mean(v, 0) for k, v in acc.items()}
    K3 = set().union(*[set(T3[n]) for n in T3]); Y = {k: np.mean([T3[n][k] for n in T3 if k in T3[n]], 0) for k in K3}; Ym = np.array([v for k, v in Y.items() if k[0] == "myth"]); Yk = np.array([v for k, v in Y.items() if k[0] == "mechanism"]); Ya = np.array(list(Y.values()))
    P1 = pool(T1); a, lo, hi, n = ci(list(P1.values())); lines += [f"- **T1 (primary), completion: a telling cut before its key step finds its own pattern's whole telling in another language, same pattern minus other patterns, 60% depth, pooled: {a:+.4f} [{lo:+.4f}, {hi:+.4f}], {n} patterns -> {'supported' if lo > 0 else 'not supported'}**", f"- By kind: myths {f3(v for k, v in P1.items() if k[0] == 'myth')}; mechanisms {f3(v for k, v in P1.items() if k[0] == 'mechanism')}. Per model: " + "; ".join(f"{m} {np.mean(list(T1[m].values())):+.4f}" for m in T1) + ". Language pairs used, by kind: " + ", ".join(f"{k[0][:4]} {k[1]}-{k[2]} {v}" for k, v in sorted(used.items())) + "."]
    Pb = pool(T1b); G = {s: np.array([v for k, v in Pb.items() if k[2] == s]) for s in ("true", "foil")}
    def two(fn):
        g = {s: np.array([v for k, v in Pb.items() if k[2] == s and fn(k)]) for s in ("true", "foil")}; b = (g["true"][rng.integers(0, len(g["true"]), (10000, len(g["true"])))].mean(1) + g["foil"][rng.integers(0, len(g["foil"]), (10000, len(g["foil"])))].mean(1)) / 2; return (g["true"].mean() + g["foil"].mean()) / 2, b
    m_, b_ = two(lambda k: True); lines.append(f"- T1b, the cut telling leans to the intact telling over the broken one in another language, minus the same lean against other patterns' pairs, mean of the two outcome groups: {m_:+.4f} [{np.percentile(b_, 2.5):+.4f}, {np.percentile(b_, 97.5):+.4f}] (true ends better {G['true'].mean():+.4f}, foil ends better {G['foil'].mean():+.4f}).")
    mm, bm = two(lambda k: k[0] == "myth"); me, be = two(lambda k: k[0] == "mechanism"); dd = bm - be; lines.append(f"- T1b by kind: myths {mm:+.4f}; mechanisms {me:+.4f}; difference {mm - me:+.4f} [{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}].")
    P2 = pool(T2d); P2c = pool(T2c); lines.append(f"- T2 (amended), contraction: the size of the swap's change, 80% over 20% depth, in the story minus in its shuffled sentences, position pinned (below 0: the story pulls the damage back more): {f3(P2.values())}. Myths {f3(v for k, v in P2.items() if k[0] == 'myth')}; mechanisms {f3(v for k, v in P2.items() if k[0] == 'mechanism')}. The cosine gap version: {f3(P2c.values())}.")
    P3 = pool(T2b); lines.append(f"- T2b, a push followed upward: its size at 80% depth over 40%, along the story's own key step minus along another pattern's (below 0: its own key step's push fades faster): {f3(P3.values())}. The push's relative size at 40% depth, median {float(np.median(psize)):.4f}: near 0 means too small to read.")
    lines.append(f"- T3, the mythologem's parts: the same pattern's match in another language, whole story {Ya[:, 0].mean():+.4f}, another sentence removed {Ya[:, 1].mean():+.4f}, the key sentence removed {Ya[:, 2].mean():+.4f}. **Removing the key sentence costs more than removing another: {f3(Ya[:, 1] - Ya[:, 2])}.** Myths {f3(Ym[:, 1] - Ym[:, 2])}; mechanisms {f3(Yk[:, 1] - Yk[:, 2])}. Removing any sentence costs: {f3(Ya[:, 0] - Ya[:, 1])}.")
    (OUT / "report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    {"report": report, "capture": lambda: capture(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
