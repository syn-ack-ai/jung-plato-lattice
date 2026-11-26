"""Cross-company test (README addendum 7): capture each model's layer activations on
  stories: myth anchors / true retellings / foils, mechanism anchors / true / foils, 30 neutral passages
           (raw text, mean over tokens after the first)
  claims:  the yes/no "same pattern?" prompt for every A and B item x 4 candidates (240), plus 150 prompts in the same
           template whose two "stories" are unrelated standard-library docstring passages (for fitting maps between
           models on content with no story pattern); last token, and P(yes)
Works for any mlx_lm decoder: records the output of every block of type(model.layers[0]).
  python3 cross.py MODEL_PATH NAME [--fresh]   ->  cross/NAME.npz
--fresh (addendum 9): the fresh item set (fresh.py) instead of the original 30 + 30; NAME must end in _f.
--confirm (addendum 16): the confirmation set (confirm.py, items4); NAME must end in _c."""
import hashlib, importlib, inspect, sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
FRESH = "--fresh" in sys.argv
CONFIRM = "--confirm" in sys.argv
assert not (FRESH and CONFIRM)
if FRESH:
    from fresh import A, A2, B, B2  # noqa: E402
elif CONFIRM:
    from confirm import A, A2, B, B2  # noqa: E402
else:
    from items import A, B  # noqa: E402
    from items_v2 import A2, B2  # noqa: E402

CLAIM = ("Story 1:\n{anchor}\n\nStory 2:\n{opt}\n\nDo these two stories follow the same underlying pattern? "
         "Answer Yes or No.")
MODULES = ["json", "argparse", "collections", "itertools", "functools", "pathlib", "re", "random", "datetime", "csv",
           "logging", "subprocess", "threading", "unittest", "typing", "heapq", "bisect", "shutil", "tarfile", "zipfile",
           "sqlite3", "statistics", "textwrap", "string", "fractions", "decimal", "contextlib", "abc"]


def neutral_chunks(words=40):
    """(module, chunk) taken round-robin across the modules, so consecutive chunks come from different modules"""
    per = []
    for m in MODULES:
        mod = importlib.import_module(m)
        docs = [inspect.getdoc(mod) or ""] + [inspect.getdoc(v) or "" for k, v in sorted(vars(mod).items())
                                              if not k.startswith("_") and (inspect.isfunction(v) or inspect.isclass(v))
                                              and getattr(v, "__module__", "") == mod.__name__]
        w = " ".join(" ".join(docs).split()).split(" ")
        per.append([(m, " ".join(w[i: i + words])) for i in range(0, len(w) - words, words)])
    out = []
    for k in range(max(len(p) for p in per)):
        out += [p[k] for p in per if k < len(p)]
    return out


def neutral_pairs(chunks, n):
    """n pairs of chunks from different modules, no chunk used twice"""
    pairs, i = [], 0
    while len(pairs) < n:
        a, b = chunks[i], chunks[i + 1]
        if a[0] != b[0]:
            pairs.append((a[1], b[1])); i += 2
        else:
            i += 1
    return pairs


def texts():
    """(kind, set, item, role, text); kind story or claim"""
    for s, items, ex in (("A", A, A2), ("B", B, B2)):
        for iid, _, anchor, true, _ in items:
            for role, t in (("anchor", anchor), ("true", true), ("foil", ex[iid][1])):
                yield "story", s, iid, role, t
    ch = neutral_chunks()
    for i, (_, t) in enumerate(ch[:30]):
        yield "story", "N", f"n{i:02d}", "neutral", t
    for s, items, ex in (("A", A, A2), ("B", B, B2)):
        for iid, _, anchor, true, surface in items:
            for cand, opt in (("true", true), ("surface", surface), ("lure", ex[iid][0]), ("foil", ex[iid][1])):
                yield "claim", s, iid, cand, CLAIM.format(anchor=anchor, opt=opt)
    for i, (a, b) in enumerate(neutral_pairs(ch[30:], 150)):
        yield "claim", "N", f"p{i:03d}", "neutral", CLAIM.format(anchor=a, opt=b)



def text_ids(tok, t):
    """token ids for a bare text (correction, 2026-11-01): tok.encode() leaves out the start token for Gemma 4 and Llama
    3.1, whose chat templates begin with it, so captures before this fix ran those models without it (MLX with <bos>
    matches PyTorch at cosine 0.999; without it 0.12-0.38). Prepend it when the chat template uses it; every other
    model gets exactly tok.encode(t) as before. Position 0 is dropped downstream as before."""
    ids = tok.encode(t)
    if getattr(tok, "_starts_bos", None) is None:
        b = tok.bos_token
        tok._starts_bos = bool(b) and tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                                              add_generation_prompt=True).startswith(b)
    return [tok.bos_token_id] + ids if tok._starts_bos and (not ids or ids[0] != tok.bos_token_id) else ids

REC = []


def main():
    path, name = [a for a in sys.argv[1:] if a not in ("--fresh", "--confirm")]
    assert name.endswith("_f") == FRESH, "fresh-set captures are named NAME_f, original ones not"
    assert name.endswith("_c") == CONFIRM, "confirmation-set captures are named NAME_c"
    model, tok = load(path)
    cls = type(model.layers[0]); orig = cls.__call__

    def rec(self, *a, **k):
        out = orig(self, *a, **k)
        REC.append((out[0] if isinstance(out, tuple) else out)[0])
        return out

    cls.__call__ = rec
    first = lambda w: {tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w)}  # noqa: E731
    yes, no = sorted(first("Yes")), sorted(first("No"))
    assert not set(yes) & set(no) and all(tok.decode([t]).strip() for t in yes + no), (yes, no)
    nl = len(model.layers)
    meta, story, claim, lyes, lno, mass, hashes = [], [], [], [], [], [], []
    for kind, s, iid, role, t in texts():
        REC.clear(); hashes.append(hashlib.sha1(t.encode()).hexdigest()[:16])
        if kind == "story":
            x = mx.array([text_ids(tok, t)])
            model(x); assert len(REC) == nl, (len(REC), nl)
            h = mx.stack([r[1:].astype(mx.float32).mean(0) for r in REC]); mx.eval(h)
            story.append(np.array(h))
        else:
            p = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
            logits = model(mx.array([tok.encode(p, add_special_tokens=False)]))[0, -1].astype(mx.float32)
            assert len(REC) == nl, (len(REC), nl)
            lp = logits - mx.logsumexp(logits); h = mx.stack([r[-1].astype(mx.float32) for r in REC]); mx.eval(h, lp)
            ly, ln = mx.logsumexp(lp[mx.array(yes)]).item(), mx.logsumexp(lp[mx.array(no)]).item()
            claim.append(np.array(h)); lyes.append(ly); lno.append(ln); mass.append(float(np.exp(ly) + np.exp(ln)))
        meta.append((kind, s, iid, role))
    out = HERE / "cross"; out.mkdir(exist_ok=True)
    col = lambda j, k: np.array([m[j] for m in meta if m[0] == k])  # noqa: E731
    hk = lambda k: np.array([h for h, m in zip(hashes, meta) if m[0] == k])  # noqa: E731
    S, C = np.stack(story), np.stack(claim)
    assert np.isfinite(S).all() and np.isfinite(C).all()
    small = lambda x: x.astype(np.float16) if np.abs(x).max() < 6e4 else x  # noqa: E731  (float16 only if it fits)
    ly, ln = np.array(lyes), np.array(lno)
    np.savez(out / f"{name}.npz", story=small(S), claim=small(C), logp_yes=ly, logp_no=ln,
             p_yes=np.exp(ly) / (np.exp(ly) + np.exp(ln)), mass=np.array(mass), s_hash=hk("story"), c_hash=hk("claim"),
             s_set=col(1, "story"), s_item=col(2, "story"), s_role=col(3, "story"),
             c_set=col(1, "claim"), c_item=col(2, "claim"), c_cand=col(3, "claim"))
    item = col(1, "claim") != "N"
    print(f"{name}: {len(story)} stories, {len(claim)} claims, {nl} layers x {S.shape[2]}; yes/no mass < 0.5 on "
          f"{int((np.array(mass)[item] < 0.5).sum())} of {int(item.sum())} item claims; max |activation| {max(np.abs(S).max(), np.abs(C).max()):.0f}")


if __name__ == "__main__":
    main()
