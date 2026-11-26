"""The archetypes' shared house vs the fix signature (README addendum 59, part A; user: "It could also be that the
direction is pointing to a shared house of all the arcehtypes involved, OR just the direction all those arcehtypes
cluster in"). Ten directions in the rooms' z-space (the rooms' right-minus-wrong for myths, mechanisms and bug stories;
DONE-minus-NEGLECTED for duty, craft, diagnosis, order, healing, SELF, PERFECTION); the static and agent fix signatures
as selfshape.fix_items.
  A1  HOUSE = unit mean of the 10 unit directions; cosine with each fix signature; sign-flip null (1,000)
  A2  share of the fix signature's direction inside the span of the 10 (squared projection length); same null
  A3  mean pairwise cosine of the 10; HOUSE minus the best single direction (paired bootstrap, 2,000)
  also: the steering vector for part B (unit(sd * HOUSE) at layer 17, raw stream) -> house/HOUSE_L17.npz
  uv run --no-project --with numpy --with scipy --with scikit-learn python house.py     -> house/report.md"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "house"
sys.path.insert(0, str(HERE))
import agentshape as AS  # noqa: E402
import selfshape as SS  # noqa: E402


def unit(v):
    return v / np.linalg.norm(v)


def items(l, sd):
    K, _ = AS.rooms(l); A = np.load(HERE / "archstory" / "caps_e4b.npz"); S = np.load(HERE / "selfshape" / "caps_e4b.npz")
    DA = (A[f"DONE_L{l}"].astype(np.float64) - A[f"NEG_L{l}"].astype(np.float64)) / sd; DS = (S[f"DONE_L{l}"].astype(np.float64) - S[f"NEG_L{l}"].astype(np.float64)) / sd
    out = {r: K[r] for r in ("myths", "mechanisms", "bugstory")}
    out.update({a: DA[A["arch"] == a] for a in ("duty", "craft", "diagnosis", "order", "healing")})
    out.update({a: DS[S["lab_DONE"] == a] for a in ("self", "perfection")})
    return out


def dirs(I, rng=None):
    """unit direction per set; with rng, every set rebuilt from its items with random signs (the null)"""
    return {k: unit(((rng.choice([-1.0, 1.0], len(x))[:, None] * x) if rng is not None else x).mean(0)) for k, x in I.items()}


def house(D):
    return unit(np.mean(list(D.values()), 0))


def span_share(f, D):
    Q = np.linalg.qr(np.array(list(D.values())).T)[0]; return float(np.sum((Q.T @ f) ** 2))


def main():
    rng = np.random.default_rng(59); OUT.mkdir(exist_ok=True); lines = ["# The archetypes' shared house vs the fix signature (addendum 59, part A)", ""]
    for l in (17, 25, 34):
        _, sd = AS.rooms(l); I = items(l, sd); D = dirs(I); Hd = house(D); F = SS.fix_items(l, sd)
        names = list(D); pc = [D[a] @ D[b] for i, a in enumerate(names) for b in names[i + 1:]]
        lines += [f"## Layer {l}{' (primary)' if l == 25 else ''}", "", f"- the 10 directions' mean pairwise cosine {np.mean(pc):+.3f} (range {min(pc):+.2f} to {max(pc):+.2f}); "
                  f"each with HOUSE: " + ", ".join(f"{k} {D[k] @ Hd:+.2f}" for k in names), ""]
        for sname, X in F.items():
            f = unit(X.mean(0)); c = f @ Hd; s = span_share(f, D); nc, ns = [], []
            for _ in range(1000):
                Dn = dirs(I, rng); nc.append(abs(f @ house(Dn))); ns.append(span_share(f, Dn))
            pc_, ps_ = (np.sum(np.array(nc) >= abs(c)) + 1) / 1001, (np.sum(np.array(ns) >= s) + 1) / 1001
            single = {k: f @ v for k, v in D.items()}; best = max(single, key=single.get); bs = []
            for _ in range(2000):
                fb = unit(X[rng.integers(0, len(X), len(X))].mean(0)); Db = {k: unit(x[rng.integers(0, len(x), len(x))].mean(0)) for k, x in I.items()}
                bs.append(fb @ house(Db) - max(fb @ v_ for v_ in Db.values()))         # review: the best single chosen in every draw
            lines += [f"### {sname} fix signature ({len(X)})",
                      f"- A1: cosine with HOUSE {c:+.3f}, sign-flip p {pc_:.4f} -> {'beyond its null' if c > 0 and pc_ < 0.05 else 'not beyond its null'}",
                      f"- A2: share inside the span of the 10 directions {s:.4f} (chance about {len(D) / len(f):.4f}; null 95th {np.percentile(ns, 95):.4f}), p {ps_:.4f}",
                      f"- A3: best single direction {best} {single[best]:+.3f}; HOUSE minus the best single {c - single[best]:+.3f} (bootstrap mean {np.mean(bs):+.3f}; basic interval "
                      f"[{2 * (c - single[best]) - np.percentile(bs, 97.5):+.3f}, {2 * (c - single[best]) - np.percentile(bs, 2.5):+.3f}])",
                      "- each single direction: " + ", ".join(f"{k} {v:+.3f}" for k, v in sorted(single.items(), key=lambda x: -x[1])), ""]
        if l == 17:                                                            # part B's steering vector: move along HOUSE in z-space
            keep = AS.labels()[5]; v = unit(sd * Hd); rn = float(np.linalg.norm(np.load(AS.D / "h_017.npy").astype(np.float32)[keep], axis=1).mean())
            np.savez(OUT / "HOUSE_L17.npz", v=v.astype(np.float32), layer=17, resid_norm=np.float32(rn)); lines.append(f"- saved the layer-17 steering vector (resid norm {rn:.1f})")
    (OUT / "report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
