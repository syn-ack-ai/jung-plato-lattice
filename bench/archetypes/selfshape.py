"""Is the shared fix shape the Self, perfection, or plain correctness? (README addendum 57; user: "what if the shared one
is somethig akin to the 'self' or perfection?"; "yes pre-register and run it"). Reuses archstory.py (pairs), fixlib.py
(library stories, ranking) and agentshape.py (the agent's reasoning states and labels).
  write      SELF and PERFECTION pairs (archstory's two-step writing) -> selfshape/pairs/stories.json; Jung's core
             archetypes as library stories (fixlib's writing) -> selfshape/lib/library.json
  capture    base E4B (MLX): the pairs, the library stories and the templated right/wrong pairs of result 27 (xcode.items),
             mean state over tokens after the first, layers 17, 25, 34 -> selfshape/caps_e4b.npz
  analyze    W1-W4 -> selfshape/report.md
  python selfshape.py write | capture;  uv run --no-project --with numpy --with scipy --with scikit-learn python selfshape.py analyze"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "selfshape"
sys.path.insert(0, str(HERE))
LAYERS, PRIMARY = (17, 25, 34), 25
PAIRS = {
    "self": {"desc": "something or someone is divided into parts that do not fit together, and one part is pushed away",
             "done": "The parts are brought together into one whole, the rejected part included, and it works as one.",
             "neglected": "the rejected part stays pushed away and it all stays divided",
             "key": "the parts come together into one whole, including the part that was pushed away", "names": ["wholeness", "integrat"]},
    "perfection": {"desc": "someone is finishing a piece of work",
                   "done": "They keep at it until no flaw is left anywhere.",
                   "neglected": "they leave one tiny flaw in it, though the rest is fine",
                   "key": "the work is finished with no flaw left anywhere", "names": ["perfect", "flawless"]},
}
JUNG = [
    ("the Self", "a divided person or group becomes whole by accepting the part they had rejected", ["wholeness", "integrat"]),
    ("the Shadow", "someone meets the hidden dark side of themselves and must own it", ["shadow"]),
    ("the Persona", "someone hides behind a public mask until it cracks", ["persona"]),
    ("anima and animus", "someone is drawn to a figure who embodies what they lack, and grows through it", ["anima", "animus"]),
    ("the Great Mother", "a figure who nourishes and protects others but also holds them back", ["great mother"]),
    ("the Wise Old Man", "an old guide appears with counsel at the moment of need", ["wise old"]),
    ("the Child", "a small, vulnerable new beginning carries the future", ["divine child"]),
    ("perfection", "someone works on a piece of work until it has no flaw at all", ["perfect", "flawless"]),
]
GROUPS = ("SELF", "PERFECTION", "CORRECTNESS", "LINE")


def unit(v):
    return v / np.linalg.norm(v)


def write():
    import archstory as AST, fixlib as FL
    AST.ARCH = PAIRS; AST.OUT = OUT / "pairs"; AST.OUT.mkdir(parents=True, exist_ok=True); AST.write()
    FL.LIB = JUNG; FL.OUT = OUT / "lib"; FL.OUT.mkdir(parents=True, exist_ok=True); FL.write()


def capture():
    import mlx.core as mx
    import cross, rewrite, xcode
    model, tok = rewrite.load_model("e4b"); cls = type(model.layers[0]); orig = cls.__call__; REC = []

    def rec(self, *a, **k):
        o = orig(self, *a, **k); REC.append((o[0] if isinstance(o, tuple) else o)[0]); return o
    cls.__call__ = rec

    def state(t):
        REC.clear(); model(mx.array([cross.text_ids(tok, t)])); assert len(REC) == 42
        return [np.array(REC[L][1:].astype(mx.float32).mean(0)) for L in LAYERS]
    rows, lab = {}, {}

    def add(key, texts, labels):
        S = [state(t) for t in texts]
        for j, L in enumerate(LAYERS):
            rows[f"{key}_L{L}"] = np.stack([s[j] for s in S]).astype(np.float32)
        lab[key] = np.array(labels)
    P = json.load(open(OUT / "pairs" / "stories.json")); its = [(a, it) for a, v in P.items() for it in v]
    add("DONE", [it["done"] for _, it in its], [a for a, _ in its]); add("NEG", [it["neglected"] for _, it in its], [a for a, _ in its])
    lib = json.load(open(OUT / "lib" / "library.json")); add("LIB", [v["story"] for v in lib], [v["arch"] for v in lib])
    X = xcode.items(); xs = [(d, r, w) for d, v in X.items() for r, w in v]
    add("RIGHT", [r for _, r, _ in xs], [d for d, _, _ in xs]); add("WRONG", [w for _, _, w in xs], [d for d, _, _ in xs])
    np.savez(OUT / "caps_e4b.npz", **rows, **{f"lab_{k}": v for k, v in lab.items()})
    print("captured", {k: len(v) for k, v in lab.items()}, flush=True)


def groups(l, sd):
    """{direction: [item matrices]} in the rooms' z-space; a direction = unit mean of its groups' unit means"""
    import agentshape as AS
    K, _ = AS.rooms(l); C = np.load(OUT / "caps_e4b.npz")
    DN = (C[f"DONE_L{l}"].astype(np.float64) - C[f"NEG_L{l}"].astype(np.float64)) / sd; RW = (C[f"RIGHT_L{l}"].astype(np.float64) - C[f"WRONG_L{l}"].astype(np.float64)) / sd
    return {"SELF": [DN[C["lab_DONE"] == "self"]], "PERFECTION": [DN[C["lab_DONE"] == "perfection"]],
            "CORRECTNESS": [RW[C["lab_RIGHT"] == d] for d in sorted(set(C["lab_RIGHT"]))], "LINE": [K["myths"], K["mechanisms"], K["bugstory"]]}


def direction(G):
    return unit(np.mean([unit(g.mean(0)) for g in G], 0))


def fix_items(l, sd):
    """STATIC (result 47) and AGENT fix signatures as item matrices in the rooms' z-space"""
    import agentshape as AS, archroom as AR
    C = np.load(HERE / "fixlib" / "caps_e4b.npz"); FB = (C[f"FIX_L{l}"] - C[f"BUG_L{l}"]).astype(np.float64) / sd; CB = (C[f"COS_L{l}"] - C[f"BUG_L{l}"]).astype(np.float64) / sd
    c = unit(CB.mean(0)); out = {"STATIC": FB - np.outer(FB @ c, c)}
    Y, run, *_, keep, _ = AS.labels(); Z = np.load(AS.D / "steps.npz"); H = Z[f"REASON_L{l}"].astype(np.float64); ok = keep & ~np.isnan(H).any(1); mv = AR.move_labels(); d = []
    for r_ in np.unique(run[ok]):
        g = ok & (run == r_); a = g & (Y["Q2"] == 1); b = g & np.array([m == "investigate" for m in mv])
        if a.any() and b.any():
            d.append(H[a].mean(0) - H[b].mean(0))
    out["AGENT"] = np.array(d) / sd
    return out


def analyze():
    import agentshape as AS, fixlib as FL
    rng = np.random.default_rng(0); lines = ["# The Self, perfection or plain correctness? (addendum 57)", ""]
    P = json.load(open(OUT / "pairs" / "stories.json")); lib = json.load(open(OUT / "lib" / "library.json"))
    from collections import Counter
    lines += [f"- pairs kept: " + ", ".join(f"{a} {len(v)}{' (under 25)' if len(v) < 25 else ''}" for a, v in P.items()) + f"; Jung library stories: {dict(Counter(v['arch'] for v in lib))}",
              "- the SELF minus columns are built at matched size (each direction from as many items as SELF has; amendment); p is two-sided, a verdict also needs a positive cosine", ""]
    verdict = {}
    for l in LAYERS:
        _, sd = AS.rooms(l); G = groups(l, sd); F = fix_items(l, sd); D = {k: direction(v) for k, v in G.items()}
        lines += [f"## Layer {l}{' (primary)' if l == PRIMARY else ''}: cosine of the fix signatures with each direction", "",
                  "| signature | " + " | ".join(GROUPS) + " | SELF minus PERFECTION | SELF minus CORRECTNESS |", "|---|" + "---|" * (len(GROUPS) + 2)]
        for sname, X in F.items():
            f = unit(X.mean(0)); cells = []
            for g in GROUPS:
                obs = f @ D[g]; null = [abs(f @ direction([rng.choice([-1.0, 1.0], len(x))[:, None] * x for x in G[g]])) for _ in range(1000)]
                cells.append(f"{obs:+.3f} (p {(np.sum(np.array(null) >= abs(obs)) + 1) / 1001:.3f})")
            ns = len(G["SELF"][0]); bs = {("SELF", "PERFECTION"): [], ("SELF", "CORRECTNESS"): []}; flat = {g: np.concatenate(G[g]) for g in G}
            lab = {g: np.concatenate([[k] * len(x) for k, x in enumerate(G[g])]) for g in G}

            def matched(g, r):                                               # review: every direction built from n_SELF items
                ix = r.choice(len(flat[g]), ns, replace=True); return direction([flat[g][ix][lab[g][ix] == k] for k in np.unique(lab[g][ix])])
            for _ in range(2000):
                fb = unit(X[rng.integers(0, len(X), len(X))].mean(0)); Db = {g: matched(g, rng) for g in ("SELF", "PERFECTION", "CORRECTNESS")}
                for a, b in bs:
                    bs[a, b].append(fb @ Db[a] - fb @ Db[b])
            for a, b in bs:
                v = np.array(bs[a, b]); cells.append(f"{np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
            lines.append(f"| {sname} ({len(X)}) | " + " | ".join(cells) + " |")
            mc = {g: np.mean([f @ matched(g, rng) for _ in range(200)]) for g in ("SELF", "PERFECTION", "CORRECTNESS", "LINE")}
            sh = {}
            for g in G:                                                       # reliability: cosine between directions from two random halves
                vals = []
                for _ in range(50):
                    pm = rng.permutation(len(flat[g])); h1, h2 = pm[: len(pm) // 2], pm[len(pm) // 2:]
                    vals.append(direction([flat[g][h1][lab[g][h1] == k] for k in np.unique(lab[g][h1])]) @ direction([flat[g][h2][lab[g][h2] == k] for k in np.unique(lab[g][h2])]))
                sh[g] = float(np.mean(vals))
            verdict[l, sname] = {g: (float(f @ D[g]), cells[k]) for k, g in enumerate(GROUPS)}
            lines.append(f"|   matched size ({ns} items each), mean cosine | " + " | ".join(f"{mc[g]:+.3f}" for g in GROUPS) + " | | |")
            lines.append(f"|   split-half cosine of the direction | " + " | ".join(f"{sh[g]:.2f}" for g in GROUPS) + " | | |")
        lines += ["", "- directions with each other: " + ", ".join(f"{a} . {b} {D[a] @ D[b]:+.2f}" for i, a in enumerate(GROUPS) for b in GROUPS[i + 1:]), ""]

    v25 = verdict[PRIMARY, "STATIC"]
    def above(g):
        c, cell = v25[g]; p = float(cell.split("p ")[1].rstrip(")")); return c > 0 and p < 0.05
    lines += ["## W1 verdicts (static fix signature, layer 25)", ""] + [f"- {g} above its null (prediction {pr}): {'yes' if above(g) else 'no'} ({v25[g][1]})"
                                                                          for g, pr in (("SELF", "40%"), ("PERFECTION", "35%"), ("CORRECTNESS", "60%"))] + [""]
    # W3: the three directions on the agent's reasoning states, as addendum 52
    Y, run, repo, cov, resolved, keep, info = AS.labels(); Z = np.load(AS.D / "steps.npz"); avail = ~np.isnan(Z[f"REASON_L{PRIMARY}"].astype(np.float32)).any(1)
    for q in Y:
        Y[q][~avail] = np.nan
    cov = np.column_stack([cov, np.log(np.maximum(Z["n_reason"], 1))]); W = {q: AS.WithinRun(Y[q], run, cov) for q in AS.QS}; res = {}
    _, sd = AS.rooms(PRIMARY); G = groups(PRIMARY, sd); H = np.nan_to_num(Z[f"REASON_L{PRIMARY}"].astype(np.float32))
    R = rng.standard_normal((AS.NRAND, H.shape[1])).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True); PR = H @ R.T
    rnd = {q: np.array([W[q].auc(W[q].score(PR[:, j])) for j in range(AS.NRAND)]) for q in AS.QS}
    for g in ("SELF", "PERFECTION", "CORRECTNESS"):
        v = unit(direction(G[g]) / sd).astype(np.float32); proj = H @ v
        for q in AS.QS:
            s = W[q].score(proj); obs, pp = W[q].perm_p(s, rng); pr = (np.sum(rnd[q] >= obs) + 1) / (AS.NRAND + 1); res[g, q] = (obs, pp, pr, max(pp, pr))
    cells = sorted(res, key=lambda c: res[c][3]); adj, mx_ = {}, 0.0
    for j, c in enumerate(cells):
        mx_ = max(mx_, min(1.0, (len(cells) - j) * res[c][3])); adj[c] = mx_
    lines += [f"## W3: on the agent's reasoning states (layer {PRIMARY}; within-run AUROC; p permutation / random; Holm over 12)", "",
              "| direction | " + " | ".join(f"{q} {AS.QNAME[q]}" for q in AS.QS) + " |", "|---|" + "---|" * 4]
    for g in ("SELF", "PERFECTION", "CORRECTNESS"):
        lines.append(f"| {g} | " + " | ".join(f"{res[g, q][0]:.3f}; {res[g, q][1]:.3f} / {res[g, q][2]:.3f}; Holm {adj[g, q]:.3f}{' *' if adj[g, q] < 0.05 else ''}" for q in AS.QS) + " |")
    lines += ["", f"- random directions, 95th percentile: " + ", ".join(f"{q} {np.percentile(rnd[q], 95):.3f}" for q in AS.QS), ""]

    # W4: the open ranking over 58 archetypes (addendum 54's library + Jung's core archetypes), library z-space as fixlib
    L = PRIMARY; C0 = np.load(HERE / "fixlib" / "caps_e4b.npz"); C1 = np.load(OUT / "caps_e4b.npz")
    X = np.concatenate([C0[f"LIB_L{L}"], C1[f"LIB_L{L}"]]).astype(np.float64); arch = np.concatenate([C0["arch"], C1["lab_LIB"]]); sdl = X.std(0) + 1e-6; X = X / sdl
    n = Counter(arch); names = [a for a in dict.fromkeys(arch) if n[a] >= 8]; keep_ = np.isin(arch, names); X, arch = X[keep_], arch[keep_]
    FB = (C0[f"FIX_L{L}"] - C0[f"BUG_L{L}"]).astype(np.float64) / sdl; CB = (C0[f"COS_L{L}"] - C0[f"BUG_L{L}"]).astype(np.float64) / sdl; c = unit(CB.mean(0))
    sigs = {"STATIC": FB - np.outer(FB @ c, c), "AGENT": fix_items(L, sdl)["AGENT"]}
    lines += [f"## W4: open ranking over {len(names)} archetypes (layer {L})", ""]
    for sname, Dm in sigs.items():
        sc = FL.ranking(Dm.mean(0), X, arch, names); order = np.argsort(-sc); null = [FL.ranking(Dm.mean(0), X, rng.permutation(arch), names).max() for _ in range(2000)]
        ranks = []
        for _ in range(1000):
            Xb = np.concatenate([X[rng.choice(np.flatnonzero(arch == a), (arch == a).sum())] for a in names]); ab = np.concatenate([[a] * int((arch == a).sum()) for a in names])
            sb = FL.ranking(Dm[rng.integers(0, len(Dm), len(Dm))].mean(0), Xb, ab, names); ranks.append(list(np.argsort(-sb)).index(names.index("the Self")) + 1)
        if "the Self" not in names:
            lines.append(f"- {sname}: the Self dropped (under 8 stories)"); continue
        rs = list(order).index(names.index("the Self")) + 1
        lines += [f"- {sname}: top 5 " + ", ".join(f"{names[j]} {sc[j]:+.3f}" for j in order[:5]) + f"; top score against the label shuffle p {(np.sum(np.array(null) >= sc.max()) + 1) / 2001:.4f}",
                  f"  - the Self: rank {rs} of {len(names)} (bootstrap 95% {np.percentile(ranks, 2.5):.0f}-{np.percentile(ranks, 97.5):.0f}), cosine {sc[names.index('the Self')]:+.3f}; "
                  + "; ".join(f"{a} rank {list(order).index(names.index(a)) + 1}" for a in ("the Shadow", "the Persona", "anima and animus", "the Great Mother", "the Wise Old Man", "the Child", "perfection") if a in names)]
    (OUT / "report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


# ---------- addendum 58: confirmation at layer 17 with new SELF stories ----------
SETTINGS2 = ["a fishing village", "a city hospital", "a high school band", "a construction site", "a family restaurant", "a rowing club",
             "a mountain village", "a newspaper office", "a train station", "a dance troupe", "a farm cooperative", "a fire station",
             "a university lab", "a small airline", "a football team", "a quilting circle", "a car repair garage", "a ferry crew",
             "a summer camp", "a monastery garden", "a tailor's shop", "a radio station", "a film crew", "a vineyard cooperative",
             "a hiking club", "a coal mine", "a chess club", "a toy workshop", "a city council", "a cricket club", "a circus",
             "a book club", "a lifeboat crew", "a bakery chain", "a museum", "a recycling yard", "a jazz band", "a refugee camp kitchen",
             "a wildlife reserve", "a shipyard", "a village choir", "a hotel kitchen", "a tea plantation", "a printing press",
             "a cycling team", "a garden nursery", "a carpentry shop", "a fishing fleet", "a marching band", "a student newspaper",
             "a theater company", "a bus company", "a family orchard", "a pottery village", "a horse stable", "a market town",
             "a research ship", "a sports club", "a weaving mill", "a youth orchestra", "a harbor pilot station", "a ski school",
             "a cheese dairy", "a postal sorting office", "a river barge", "a desert caravan", "a lighthouse crew", "a bee farm",
             "a hospice", "a small factory", "a robotics club for children", "a surf school", "a monastery", "a rice farm",
             "a puppet theater", "a glass workshop", "a mountain hut crew", "a fishing cooperative", "a school kitchen", "a flower market"]
PAIRS2 = {"self2": {"desc": "a group, a place or a person has pushed one of its own parts away, and it works badly",
                    "done": "The pushed-away part is brought back and accepted, and the whole works as one again.",
                    "neglected": "the part stays pushed away, and the whole keeps working badly",
                    "key": "the pushed-away part is brought back and the whole works as one", "names": ["wholeness", "integrat"]}}


def write2():
    import archstory as AST
    AST.ARCH = PAIRS2; AST.SETTINGS = SETTINGS2; AST.OUT = OUT / "pairs2"; AST.OUT.mkdir(parents=True, exist_ok=True); AST.write()


def capture2():
    import mlx.core as mx
    import cross, rewrite
    model, tok = rewrite.load_model("e4b"); cls = type(model.layers[0]); orig = cls.__call__; REC = []

    def rec(self, *a, **k):
        o = orig(self, *a, **k); REC.append((o[0] if isinstance(o, tuple) else o)[0]); return o
    cls.__call__ = rec
    rows = {}; P = json.load(open(OUT / "pairs2" / "stories.json"))["self2"]
    for key, field in (("DONE", "done"), ("NEG", "neglected")):
        S = []
        for it in P:
            REC.clear(); model(mx.array([cross.text_ids(tok, it[field])])); assert len(REC) == 42; S.append([np.array(REC[L][1:].astype(mx.float32).mean(0)) for L in LAYERS])
        for j, L in enumerate(LAYERS):
            rows[f"{key}_L{L}"] = np.stack([s[j] for s in S]).astype(np.float32)
    np.savez(OUT / "caps2_e4b.npz", **rows); print("captured", len(P), "SELF2 pairs", flush=True)


def analyze2():
    import agentshape as AS
    rng = np.random.default_rng(58); P = json.load(open(OUT / "pairs2" / "stories.json"))["self2"]; C2 = np.load(OUT / "caps2_e4b.npz")
    lines = ["# Confirming the Self lean at layer 17 (addendum 58)", "", f"- SELF2 pairs: {len(P)} (new settings; result 49's pairs not in the primary test)", ""]
    for l in (17, 25, 34):
        _, sd = AS.rooms(l); G = groups(l, sd); F = fix_items(l, sd)
        S2 = (C2[f"DONE_L{l}"].astype(np.float64) - C2[f"NEG_L{l}"].astype(np.float64)) / sd; G["SELF2"] = [S2]; G["SELF_POOLED"] = [G["SELF"][0], S2]       # two groups, equal weight (review)
        flat = {g: np.concatenate(G[g]) for g in G}; lab = {g: np.concatenate([[k] * len(x) for k, x in enumerate(G[g])]) for g in G}; ns = len(S2)

        def matched(g, m):
            ix = rng.choice(len(flat[g]), m, replace=True); return direction([flat[g][ix][lab[g][ix] == k] for k in np.unique(lab[g][ix])])
        lines += [f"## Layer {l}{' (primary)' if l == 17 else ''}", "", "| signature | SELF2 | SELF (result 49) | SELF pooled | PERFECTION | CORRECTNESS | LINE | SELF2 minus PERFECTION | SELF2 minus CORRECTNESS |",
                  "|---|" + "---|" * 8]
        for sname, X in F.items():
            f = unit(X.mean(0)); cells = []
            for g in ("SELF2", "SELF", "SELF_POOLED", "PERFECTION", "CORRECTNESS", "LINE"):
                obs = f @ direction(G[g]); null = np.array([abs(f @ direction([rng.choice([-1.0, 1.0], len(x))[:, None] * x for x in G[g]])) for _ in range(1000)])
                p = (np.sum(null >= abs(obs)) + 1) / 1001; cells.append(f"{obs:+.3f} (p {p:.3f})")
                if l == 17 and sname == "STATIC" and g == "SELF2":
                    c1_null = obs > 0 and p < 0.05
            diffs = {}
            for b in ("PERFECTION", "CORRECTNESS"):
                v = []
                for _ in range(2000):
                    m = min(ns, len(flat[b])); Ss = S2[rng.choice(ns, m, replace=False)]     # review: both sides at the smaller size
                    fb = unit(X[rng.integers(0, len(X), len(X))].mean(0)); v.append(fb @ direction([Ss[rng.integers(0, m, m)]]) - fb @ matched(b, m))
                v = np.array(v); diffs[b] = (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)); cells.append(f"{diffs[b][0]:+.3f} [{diffs[b][1]:+.3f}, {diffs[b][2]:+.3f}]")
            if l == 17 and sname == "STATIC":
                c1 = c1_null and all(d[1] > 0 for d in diffs.values())
            lines.append(f"| {sname} ({len(X)}) | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"- C1 (layer 17, static): SELF2 beyond its null and above PERFECTION and CORRECTNESS -> {'supported' if c1 else 'not supported'} (beyond its null alone: {'yes' if c1_null else 'no'})", ""]
    # C3: SELF2 at layer 17 on the agent's reasoning states
    Y, run, repo, cov, resolved, keep, info = AS.labels(); Z = np.load(AS.D / "steps.npz"); avail = ~np.isnan(Z["REASON_L17"].astype(np.float32)).any(1)
    for q in Y:
        Y[q][~avail] = np.nan
    cov = np.column_stack([cov, np.log(np.maximum(Z["n_reason"], 1))]); W = {q: AS.WithinRun(Y[q], run, cov) for q in AS.QS}
    _, sd = AS.rooms(17); H = np.nan_to_num(Z["REASON_L17"].astype(np.float32)); S2 = (C2["DONE_L17"].astype(np.float64) - C2["NEG_L17"].astype(np.float64)) / sd
    v = unit(direction([S2]) / sd).astype(np.float32); proj = H @ v
    R = rng.standard_normal((AS.NRAND, H.shape[1])).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True); PR = H @ R.T; res = {}
    for q in AS.QS:
        s = W[q].score(proj); obs, pp = W[q].perm_p(s, rng); rnd = np.array([W[q].auc(W[q].score(PR[:, j])) for j in range(AS.NRAND)]); pr = (np.sum(rnd >= obs) + 1) / (AS.NRAND + 1)
        res[q] = (obs, pp, pr, max(pp, pr), np.percentile(rnd, 95))
    order = sorted(res, key=lambda q: res[q][3]); adj, mx_ = {}, 0.0
    for j, q in enumerate(order):
        mx_ = max(mx_, min(1.0, (len(order) - j) * res[q][3])); adj[q] = mx_
    lines += ["## C3: SELF2 on the agent's reasoning states (layer 17)", ""] + [f"- {q} {AS.QNAME[q]}: {res[q][0]:.3f} (p permutation {res[q][1]:.3f}, random {res[q][2]:.3f}; random 95th {res[q][4]:.3f}); Holm {adj[q]:.3f}" for q in AS.QS]
    (OUT / "report2.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    {"write": write, "capture": capture, "analyze": analyze, "write2": write2, "capture2": capture2, "analyze2": analyze2}[sys.argv[1]]()
