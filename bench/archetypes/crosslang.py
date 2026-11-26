"""Does the hidden pattern look alike across languages? (README addendum 87.) DeepSeek writes separately in each language, the 31B checks, the models read; nothing is asked.
  write [N]   (Mac) sets I, T, R, resumable -> crosslang/items.jsonl (N limits the patterns, for a test)
  check       (Mac, MLX) the 31B: which version shows the key act -> crosslang/checked.jsonl
  label       (Mac) DeepSeek: which version ends better -> crosslang/sign.jsonl (the outcome control)
  capture NAME PATH   (Mac, MLX) states per model, also of the edited sentences alone -> crosslang/caps_NAME.npz;   report -> crosslang/report.md"""
import difflib, importlib.util, json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent; OUT = HERE / "crosslang"; sys.path.insert(0, str(HERE))
MODEL = "deepseek/deepseek-v4.1-flash-1017"; SEED = 87
LANG = {"en": ("English", "medieval England or the English countryside", "a harbour town in modern England"), "el": ("Modern Greek", "an ancient Greek village or island", "a workshop or town in modern Greece"),
        "ja": ("Japanese", "old Japan, a mountain village or the coast", "a workshop or town in modern Japan"), "zh": ("Modern Standard Chinese, simplified characters", "old China, with its Taoist hermits and mountains", "a workshop or town in modern China")}
SIZE = {"en": "about 90 words", "el": "about 90 words", "ja": "about 220 characters", "zh": "about 170 characters"}
SCRIPT = {"en": r"[A-Za-z]", "el": r"[Ͱ-Ͽἀ-῿]", "ja": r"[぀-ヿ一-鿿]", "zh": r"[一-鿿]"}
FAMILIES = {"forbidden_look": {"el": "Orpheus and Eurydice", "ja": "Izanagi and Izanami in Yomi", "en": "Lot's wife"}, "time_lost": {"el": "Epimenides sleeping in the cave", "ja": "Urashima Taro", "zh": "Wang Zhi and the rotted axe handle", "en": "Rip Van Winkle"},
            "goddess_withdraws": {"el": "Demeter withdrawing after Persephone is taken", "ja": "Amaterasu hiding in the cave"}, "monster_slain": {"el": "Perseus and Andromeda", "ja": "Susanoo and Yamata no Orochi", "zh": "Li Ji slaying the serpent", "en": "Saint George and the dragon"},
            "flood": {"el": "Deucalion and Pyrrha", "zh": "Yu the Great taming the flood", "en": "Noah's ark"}, "lost_bride": {"el": "Cupid and Psyche", "ja": "the Crane Wife", "zh": "the Cowherd and the Weaver Girl", "en": "the Selkie wife"}}
ASK = """Write in {lang} only. Reply with JSON only: {{"true": "...", "foil": "..."}}.
"true": {kind}, {size}, set in {setting}, with names natural to that place. It must follow this underlying pattern, every step in order: {pattern}
Do not state the pattern or a moral. Do not translate from another language: write it as someone from there would tell it.
"foil": the same story with the same people and setting and nearly all the same wording, changing only the key causal step, so that the pattern no longer holds.{fails}"""
ASK_REAL = """Write in {lang} only. Reply with JSON only: {{"true": "...", "foil": "...", "key": "the key act in one English sentence"}}.
"true": the story of {story}, told plainly as it is told there, {size}.
"foil": the same story with the same people and nearly all the same wording, changing only the key causal step, so that the turning point of the story does not happen.{fails}"""
ASK_TR = """Translate these two short texts into {lang}, faithfully, keeping the difference between them. Reply with JSON only: {{"true": "...", "foil": "..."}}.
true: {t}
foil: {f}"""


def key():
    return next(l.split("=", 1)[1].strip().strip("\"'") for l in open(HERE.parent.parent / ".env") if l.strip().startswith("OPENROUTER_KEY"))


def call(k, prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 6000, "response_format": {"type": "json_object"}, "provider": {"data_collection": "deny"}}
    x = json.loads(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + k}), timeout=240).read())
    return json.loads(re.search(r"\{.*\}", x["choices"][0]["message"]["content"] or "", re.S).group(0)), x.get("usage") or {}


def gate(lang, t, f):
    bad = []; share = lambda s: len(re.findall(SCRIPT[lang], s)) / max(len(re.findall(r"[^\W\d_]", s)), 1)  # noqa: E731
    for name, s in (("true", t), ("foil", f)):
        if share(s) < 0.85:
            bad.append(f"{name} is not written in the requested language")
    n = (lambda s: len(s.split())) if lang in ("en", "el") else len; lo, hi = (45, 170) if lang in ("en", "el") else (100, 450)
    if not lo <= n(t) <= hi:
        bad.append(f"true has length {n(t)}, outside {lo}-{hi}")
    if max(n(t), n(f)) > 1.3 * max(min(n(t), n(f)), 1):
        bad.append("true and foil differ too much in length")
    r = difflib.SequenceMatcher(None, t, f, autojunk=False).ratio()
    if r < 0.6:
        bad.append(f"the foil changed too much (similarity {r:.2f}; keep nearly all the wording)")
    if t.strip() == f.strip():
        bad.append("the foil is identical to the true story")
    return bad


def write():
    spec = importlib.util.spec_from_file_location("w0", HERE / "items9_d31" / "w0.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(SEED); n = int(sys.argv[2]) if len(sys.argv) > 2 else 40; A = [m.A[j] for j in sorted(rng.permutation(len(m.A))[:n])]; B = [m.B[j] for j in sorted(rng.permutation(len(m.B))[:n])]; OUT.mkdir(exist_ok=True); f = OUT / "items.jsonl"; k = key()
    done = {(r["set"], r["kind"], r["pattern_id"], r["lang"]) for r in map(json.loads, open(f))} if f.exists() else set(); jobs = []
    for kind, items in (("myth", A), ("mechanism", B)):
        for it in items:
            for lg in LANG:
                jobs.append(("I", kind, it[0], lg, it[1], None))
    for kind, items in (("myth", A[:5]), ("mechanism", B[:5])):
        for it in items:
            for lg in ("el", "ja", "zh"):
                jobs.append(("T", kind, it[0], lg, it[1], None))
    for fam, mem in FAMILIES.items():
        for lg, story in mem.items():
            jobs.append(("R", "myth", fam, lg, story, None))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]; base = {}
    if f.exists():
        for r in map(json.loads, open(f)):
            if r["set"] == "I" and r["lang"] == "en" and r.get("ok"):
                base[r["pattern_id"]] = r
    def one(j):
        st, kind, pid, lg, pat, _ = j; fails = ""; last = None; use = {"prompt_tokens": 0, "completion_tokens": 0}
        for rnd in range(3):
            if st == "T":
                if pid not in base:
                    return dict(set=st, kind=kind, pattern_id=pid, lang=lg, ok=False, why="no English base yet", rounds=0)
                p = ASK_TR.format(lang=LANG[lg][0], t=base[pid]["true"], f=base[pid]["foil"])
            elif st == "R":
                p = ASK_REAL.format(lang=LANG[lg][0], story=pat, size=SIZE[lg], fails=fails)
            else:
                p = ASK.format(lang=LANG[lg][0], kind="a short tale told as a myth" if kind == "myth" else "a plain short account of how something works or went wrong", size=SIZE[lg], setting=LANG[lg][1 if kind == "myth" else 2], pattern=pat, fails=fails)
            try:
                y, u = call(k, p); use["prompt_tokens"] += u.get("prompt_tokens", 0); use["completion_tokens"] += u.get("completion_tokens", 0); t, fo = str(y["true"]).strip(), str(y["foil"]).strip(); bad = gate(lg, t, fo)
            except Exception as e:
                bad = [f"error {type(e).__name__}"]; t = fo = ""; y = {}
            last = dict(set=st, kind=kind, pattern_id=pid, pattern=pat, lang=lg, true=t, foil=fo, key=y.get("key"), ok=not bad, why=bad, rounds=rnd + 1, usage=use)
            if not bad:
                break
            fails = "\nYour last draft failed these checks; fix them: " + "; ".join(bad)
        return last
    first = [j for j in jobs if j[0] != "T"]; second = [j for j in jobs if j[0] == "T"]
    with open(f, "a") as out, ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, first):
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
            if r["set"] == "I" and r["lang"] == "en" and r.get("ok"):
                base[r["pattern_id"]] = r
        for r in ex.map(one, second):
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
    R = [json.loads(l) for l in open(f)]; from collections import Counter
    print("rows", len(R), "ok", sum(r.get("ok", False) for r in R)); print("ok by set and language", sorted(Counter((r["set"], r["lang"]) for r in R if r.get("ok")).items())); print("failed by language", sorted(Counter(r["lang"] for r in R if not r.get("ok")).items()), "reasons", Counter(w.split(" (")[0][:40] for r in R if not r.get("ok") for w in (r.get("why") or []) if isinstance(r.get("why"), list)).most_common(5))
    tok, ot = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in R), sum((r.get("usage") or {}).get("completion_tokens", 0) for r in R); print(f"tokens in {tok} out {ot}, cost about ${tok * 0.15e-6 + ot * 0.6e-6:.2f}")

SIGN_ASK = """Two versions of a short text, written in {lang}. Which version ends better for the people or the thing it describes? Reply with JSON only: {{"better": "A", "B" or "same"}}.

A: {a}

B: {b}"""


def label():
    """(Mac) DeepSeek: which version ends better, both orders -> crosslang/sign.jsonl; the third amendment's control. "mixed" if the two orders disagree"""
    k = key(); f = OUT / "sign.jsonl"; done = {(r["kind"], r["pattern_id"], r["lang"]) for r in map(json.loads, open(f))} if f.exists() else set(); R = [r for r in map(json.loads, open(OUT / "items.jsonl")) if r.get("ok") and r["set"] == "I" and (r["kind"], r["pattern_id"], r["lang"]) not in done]

    def one(r):
        got = []
        for a, b, flip in ((r["true"], r["foil"], False), (r["foil"], r["true"], True)):
            for t in range(3):
                try:
                    y, _ = call(k, SIGN_ASK.format(lang=LANG[r["lang"]][0], a=a, b=b)); v = str(y.get("better")).strip().lower()
                    if v in ("a", "b", "same"):
                        got.append("same" if v == "same" else ("true" if (v == "a") != flip else "foil")); break
                except Exception:
                    pass
            else:
                got.append("failed")
        return {"kind": r["kind"], "pattern_id": r["pattern_id"], "lang": r["lang"], "better": got[0] if got[0] == got[1] else "mixed"}
    with ThreadPoolExecutor(8) as ex, open(f, "a") as out:
        for r in ex.map(one, R):
            out.write(json.dumps(r) + "\n"); out.flush()
    from collections import Counter
    S = [json.loads(l) for l in open(f)]; print("labels", len(S), Counter(r["better"] for r in S)); print("by kind", sorted(Counter((r["kind"], r["better"]) for r in S).items())); by = {}
    for r in S:
        by.setdefault((r["kind"], r["pattern_id"]), []).append(r["better"])
    print("patterns whose four languages agree", sum(len(set(v)) == 1 for v in by.values()), "of", len(by))


CHECK_ASK = """Below are two versions of a story, written in {lang}. The underlying pattern is: {pattern}
Which version follows this pattern, with its key step intact? Answer with one letter.

A: {a}

B: {b}"""


def check():
    """(Mac, MLX) Gemma 4 31B must tell which version follows the pattern, p at least 0.7 over both orders (as addendum 53); Set T is not checked -> crosslang/checked.jsonl"""
    import mlx.core as mx
    import rewrite
    model, tok = rewrite.load_model("d31"); AB = {x: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (x, " " + x)}) for x in "AB"}; f = OUT / "checked.jsonl"
    done = {(r["set"], r["kind"], r["pattern_id"], r["lang"]) for r in map(json.loads, open(f))} if f.exists() else set(); R = [r for r in map(json.loads, open(OUT / "items.jsonl")) if r.get("ok") and r["set"] != "T" and (r["set"], r["kind"], r["pattern_id"], r["lang"]) not in done]
    print("to check", len(R), flush=True); out = open(f, "a")
    for n, r in enumerate(R):
        ps, ms = [], []; key = r.get("key") or r["pattern"]
        for first in (True, False):
            a, b = (r["true"], r["foil"]) if first else (r["foil"], r["true"])
            q = tok.apply_chat_template([{"role": "user", "content": CHECK_ASK.format(lang=LANG[r["lang"]][0], pattern=key, a=a, b=b)}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            pr = mx.softmax(model(mx.array([tok.encode(q, add_special_tokens=False)]))[0, -1].astype(mx.float32)); pa, pb = (sum(pr[t].item() for t in AB[x]) for x in "AB"); ps.append((pa if first else pb) / max(pa + pb, 1e-9)); ms.append(pa + pb)
        out.write(json.dumps({"set": r["set"], "kind": r["kind"], "pattern_id": r["pattern_id"], "lang": r["lang"], "p31": sum(ps) / 2, "mass": min(ms)}) + "\n"); out.flush()
        if (n + 1) % 50 == 0:
            print(n + 1, "/", len(R), flush=True)
    C = [json.loads(l) for l in open(f)]; from collections import Counter
    print("passed (p at least 0.7, letter mass at least 0.5)", Counter(c["lang"] for c in C if c["p31"] >= 0.7 and c["mass"] >= 0.5), "of", Counter(c["lang"] for c in C), "| median letter mass by language", {g: round(float(np.median([c["mass"] for c in C if c["lang"] == g])), 3) for g in LANG}, flush=True); print("CROSSLANG_CHECK_DONE", flush=True)


FRAC = (0.2, 0.4, 0.6, 0.8, 1.0)


def capture(name, path):
    """(Mac, MLX) the mean state over a story's tokens after the first, at five depths, as cross.py -> crosslang/caps_NAME.npz; resumable by model"""
    import mlx.core as mx
    import cross
    from mlx_lm import load
    f = OUT / f"caps_{name}.npz"
    if f.exists():
        print(name, "already captured", flush=True); return
    model, tok = load(path); cls = type(model.layers[0]); orig = cls.__call__; REC = []

    def rec(self, *a, **k):
        o = orig(self, *a, **k); REC.append((o[0] if isinstance(o, tuple) else o)[0]); return o
    cls.__call__ = rec; nl = len(model.layers); idx = [int(round(fr * (nl - 1))) for fr in FRAC]; meta, H = [], []
    R = [r for r in map(json.loads, open(OUT / "items.jsonl")) if r.get("ok")]
    for n, r in enumerate(R):
        vers = {"true": r["true"], "foil": r["foil"]}
        if r["set"] == "I":
            tc, fc = clauses(r["true"], r["foil"], r["lang"])
            if tc and fc:
                vers.update(true_clause=tc, foil_clause=fc)
        for ver, text in vers.items():
            REC.clear(); model(mx.array([cross.text_ids(tok, text)])); assert len(REC) == nl, (len(REC), nl)
            h = mx.stack([REC[j][1:].astype(mx.float32).mean(0) for j in idx]); mx.eval(h); a32 = np.array(h); a16 = a32.astype(np.float16); assert np.isfinite(a32).all() and np.isfinite(a16).all(), (name, r["pattern_id"], ver)
            H.append(a16); meta.append((r["set"], r["kind"], r["pattern_id"], r["lang"], ver))
        if (n + 1) % 50 == 0:
            print(name, n + 1, "/", len(R), flush=True)
    tmp = OUT / f"caps_{name}.tmp.npz"; np.savez(tmp, H=np.stack(H), meta=np.array(meta), frac=np.array(FRAC)); os.replace(tmp, f); print(name, "saved", np.stack(H).shape, flush=True)


def clauses(t, f, lang):
    """the sentences that differ between true and foil, alone: the review's control. If they alone match across languages as well as the stories do, the match is word meaning"""
    cut = (lambda x: [q.strip() for q in re.split(r"(?<=[.!?;·])\s+", x) if q.strip()]) if lang in ("en", "el") else (lambda x: [q.strip() for q in re.split(r"(?<=[。！？])", x) if q.strip()])
    A, B = cut(t), cut(f)
    return " ".join(q for q in A if q not in B), " ".join(q for q in B if q not in A)


def report():
    """C1-C5 and the controls of addendum 87 with its second amendment (requires the checks and the captures)"""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED); P = {(c["set"], c["kind"], c["pattern_id"], c["lang"]): c for c in map(json.loads, open(OUT / "checked.jsonl"))}; asked = {"d31": 1.000, "a4b26": 0.982, "e4b": 0.941, "qwen3_8b": 0.872, "llama31_8b": 0.714}; DEPTH = 2
    passed = lambda st, kind, p, g: (lambda c: bool(c) and c["p31"] >= 0.7 and c["mass"] >= 0.5)(P.get((st, kind, p, g)))  # noqa: E731
    Fs = sorted(f for f in OUT.glob("caps_*.npz") if ".tmp" not in f.name); L = list(LANG); pairs = [(a, b) for i, a in enumerate(L) for b in L[i + 1:]]; from collections import Counter
    lines = ["# The hidden pattern across languages (addendum 87)", "", f"- models: {[f.stem[5:] for f in Fs]}", f"- stories passing the 31B's check, Set I, by language: {dict(sorted(Counter(k[3] for k in P if k[0] == 'I' and passed(*k)).items()))} of {dict(sorted(Counter(k[3] for k in P if k[0] == 'I').items()))}", ""]; Z = {}

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

    def table(z, st):
        H = z["H"][:, DEPTH].astype(np.float64); return {tuple(m[1:]): x for x, m in zip(H, z["meta"].astype(str)) if m[0] == st}
    for f in Fs:
        z = np.load(f, allow_pickle=True); Z[f.stem[5:]] = (table(z, "I"), table(z, "T"), table(z, "R"))
    # the language check: the English story finds its own translation, centred within each language
    flag = {}; rows = []
    for name, (T, Tt, _) in Z.items():
        out = {}
        for lg in ("el", "ja", "zh"):
            kinds = {k[1]: k[0] for k in Tt if k[2] == lg}; ids = sorted(p for p in kinds if (kinds[p], p, "en", "true") in T)
            cen = lambda g: np.mean([v for k, v in T.items() if k[2] == g and k[3] == "true"], 0)  # noqa: E731
            A = unit(np.array([T[kinds[p], p, "en", "true"] for p in ids]) - cen("en")); B = unit(np.array([Tt[kinds[p], p, lg, "true"] for p in ids]) - cen(lg)); out[lg] = float(np.mean((A @ B.T).argmax(1) == np.arange(len(ids)))) if ids else float("nan")
        flag[name] = out; rows.append(f"| {name} | " + " | ".join(f"{out[g]:.2f}" + ("" if out[g] >= 0.5 else " FLAG") for g in ("el", "ja", "zh")) + " |")
    lines += ["## Language check: the English story finds its own translation (Set T, centred; chance about 0.1; under 0.5 is flagged)", "", "| model | el | ja | zh |", "|---|---|---|---|"] + rows + [""]

    def key_steps(T, kind, a, b, ver, filt):
        ids = sorted({k[1] for k in T if k[0] == kind and k[2] == a and k[3] == ver[0]} & {k[1] for k in T if k[0] == kind and k[2] == b and k[3] == ver[0]})
        ids = [p for p in ids if (kind, p, a, ver[1]) in T and (kind, p, b, ver[1]) in T and (not filt or (passed("I", kind, p, a) and passed("I", kind, p, b)))]
        if len(ids) < 8:
            return ids, None, None
        return ids, unit(np.array([T[kind, p, a, ver[0]] - T[kind, p, a, ver[1]] for p in ids])), unit(np.array([T[kind, p, b, ver[0]] - T[kind, p, b, ver[1]] for p in ids]))

    Sg = {(r["kind"], r["pattern_id"], r["lang"]): r["better"] for r in map(json.loads, open(OUT / "sign.jsonl"))}

    def pat_scores(Da, Db, remove, sa=None, sb=None):
        """per pattern: same pattern across a and b, minus other patterns, both ways. remove: k leading axes fitted on the other folds' patterns, five folds. sa, sb: outcome labels; others are then only patterns whose outcome runs the same way (third amendment)"""
        n = len(Da)
        def s_of(M, rows):
            if sa is None:
                return M[rows, rows] - (M[rows].sum(1) - M[rows, rows] + M[:, rows].sum(0) - M[rows, rows]) / (2 * (n - 1))
            o = []
            for p in rows:
                q1 = [q for q in range(n) if q != p and sb[q] == sa[p]]; q2 = [q for q in range(n) if q != p and sa[q] == sb[p]]
                o.append(M[p, p] - (M[p, q1].mean() + M[q2, p].mean()) / 2 if len(q1) >= 3 and len(q2) >= 3 else np.nan)
            return np.array(o)
        if not remove:
            return s_of(Da @ Db.T, np.arange(n))
        order = np.random.default_rng(SEED).permutation(n); out = np.zeros(n)
        for fold in np.array_split(order, 5):
            rest = np.setdiff1d(np.arange(n), fold); _, _, Vt = np.linalg.svd(np.vstack([Da[rest], Db[rest]]), full_matrices=False); V = Vt[:remove]; pa, pb = unit(Da - (Da @ V.T) @ V), unit(Db - (Db @ V.T) @ V); out[fold] = s_of(pa @ pb.T, fold)
        return out

    def collect(T, kind, ver=("true", "foil"), remove=0, filt=True, skip=(), signed=True):
        acc = {}; np_ = {}
        for a, b in pairs:
            if (a, b) in skip:
                continue
            ids, Da, Db = key_steps(T, kind, a, b, ver, filt)
            if Da is None:
                continue
            sa, sb = ([Sg.get((kind, p, a)) for p in ids], [Sg.get((kind, p, b)) for p in ids]) if signed else (None, None); s = pat_scores(Da, Db, remove, sa, sb); np_[a, b] = (len(ids), float(np.nanmean(s)))
            for p, x in zip(ids, s):
                if np.isfinite(x):
                    acc.setdefault(p, []).append(x)
        return {p: float(np.mean(v)) for p, v in acc.items()}, np_

    def ci(v, n=10000):
        v = np.array(list(v)); bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1); return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)
    f3 = lambda v: "{:+.4f} [{:+.4f}, {:+.4f}]".format(*ci(v)[:3])  # noqa: E731
    C = {name: {kind: collect(T[0], kind) for kind in ("myth", "mechanism")} for name, T in Z.items()}; Craw = {name: {kind: collect(T[0], kind, signed=False) for kind in ("myth", "mechanism")} for name, T in Z.items()}; Cax = {name: {kind: collect(T[0], kind, signed=False, remove=1) for kind in ("myth", "mechanism")} for name, T in Z.items()}; Cls = {name: {kind: collect(T[0], kind, ver=("true_clause", "foil_clause")) for kind in ("myth", "mechanism")} for name, T in Z.items()}
    lines += ["## C1 per model: the key step across languages, same pattern minus other patterns whose outcome runs the same way, mean over language pairs, 60% depth", "", "| model | myths | mechanisms | raw, no outcome control | the edited sentences alone |", "|---|---|---|---|---|"]
    for n in Z:
        lines.append(f"| {n} | {f3(C[n]['myth'][0].values())} | {f3(C[n]['mechanism'][0].values())} | {f3(list(Craw[n]['myth'][0].values()) + list(Craw[n]['mechanism'][0].values()))} | {f3(list(Cls[n]['myth'][0].values()) + list(Cls[n]['mechanism'][0].values()))} |")

    def pool(D, kind, names=None):
        names = list(names or Z); ids = set.intersection(*[set(D[n][kind][0]) for n in names]); return {p: float(np.mean([D[n][kind][0][p] for n in names])) for p in ids}
    A1, B1 = pool(C, "myth"), pool(C, "mechanism"); m, lo, hi, n = ci(list(A1.values()) + list(B1.values())); R1, R2 = pool(Craw, "myth"), pool(Craw, "mechanism"); C1a, C1b = pool(Cls, "myth"), pool(Cls, "mechanism"); Xa, Xb = pool(Cax, "myth"), pool(Cax, "mechanism"); lines += ["", f"- **C1 (primary, third amendment), others limited to the same outcome direction, pooled over models and language pairs: {m:+.4f} [{lo:+.4f}, {hi:+.4f}], {n} patterns -> {'supported' if lo > 0 else 'not supported'}**", f"- C1 raw, no outcome control: {f3(list(R1.values()) + list(R2.values()))}. C1 with the leading axis removed instead: {f3(list(Xa.values()) + list(Xb.values()))}. These run above the primary if a shared outcome axis carries the match.", f"- The review's control, the edited sentences alone: {f3(list(C1a.values()) + list(C1b.values()))}. If it matches C1, the match is word meaning, not the story's pattern."]
    ex = [A1[q] - C1a[q] for q in A1 if q in C1a] + [B1[q] - C1b[q] for q in B1 if q in C1b]; lines.append(f"- The story beyond its edited sentences, same patterns, paired: {f3(ex)}. Above 0 means the whole story aligns more than the changed sentences alone.")
    my, me = np.array(list(A1.values())), np.array(list(B1.values())); bs = [my[rng.integers(0, len(my), len(my))].mean() - me[rng.integers(0, len(me), len(me))].mean() for _ in range(10000)]; lines.append(f"- C4, myths minus mechanisms, same outcome direction: {my.mean() - me.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (myths {my.mean():+.4f}, mechanisms {me.mean():+.4f})")
    pp = {}
    for n in Z:
        for kind in ("myth", "mechanism"):
            for k, (c, v) in C[n][kind][1].items():
                pp.setdefault(k, []).append(v)
    lines.append("- C1 by language pair, mean over models and kinds: " + "; ".join(f"{a}-{b} {np.mean(v):+.4f}" for (a, b), v in sorted(pp.items())))
    Nj = {name: {kind: collect(T[0], kind, skip={("ja", "zh")}) for kind in ("myth", "mechanism")} for name, T in Z.items()}; lines.append(f"- C1 without the Japanese-Chinese pair, whose characters overlap: {f3(list(pool(Nj, 'myth').values()) + list(pool(Nj, 'mechanism').values()))}")
    bad = [n for n, o in flag.items() if any(v < 0.5 for v in o.values())]
    if bad and len(bad) < len(Z):
        keep = [n for n in Z if n not in bad]; lines.append(f"- C1 without flagged models ({bad}): {f3(list(pool(C, 'myth', keep).values()) + list(pool(C, 'mechanism', keep).values()))}")
    un = {name: {kind: collect(T[0], kind, filt=False) for kind in ("myth", "mechanism")} for name, T in Z.items()}; lines.append(f"- C1 with no check filter (the 31B both selects and is scored): {f3(list(pool(un, 'myth').values()) + list(pool(un, 'mechanism').values()))}")
    fl = []
    for name, T in Z.items():
        for kind in ("myth", "mechanism"):
            for a, b in pairs:
                ids, Da, Db = key_steps(T[0], kind, a, b, ("true", "foil"), True)
                if Da is not None:
                    sa, sb = [Sg.get((kind, p, a)) for p in ids], [Sg.get((kind, p, b)) for p in ids]
                    for _ in range(10):
                        o = rng.permutation(len(ids)); fl.append(np.nanmean(pat_scores(Da, Db[o], 0, sa, [sb[j] for j in o])))
    if fl:
        lines.append(f"- Floor: pattern labels shuffled, same outcome control, mean {np.mean(fl):+.4f}, 95% of shuffles within [{np.percentile(fl, 2.5):+.4f}, {np.percentile(fl, 97.5):+.4f}]")
    # ceiling: on the same footing as C1: within a kind, others from Set I, the same axis removal
    tc, cc = [], []
    for name, (T, Tt, _) in Z.items():
        for lg in ("el", "ja", "zh"):
            for kind in ("myth", "mechanism"):
                ids, Da, Db = key_steps(T, kind, "en", lg, ("true", "foil"), False)
                if Da is None:
                    continue
                for j, p in enumerate(ids):
                    if (kind, p, lg, "true") not in Tt or (kind, p, lg, "foil") not in Tt:
                        continue
                    rest = np.delete(np.arange(len(ids)), j); _, _, Vt = np.linalg.svd(np.vstack([Da[rest], Db[rest]]), full_matrices=False); V = Vt[:1]; pr = lambda x: unit(x - (x @ V.T) @ V); e = pr(Da[j:j + 1])[0]; base = float((e @ pr(Db[rest]).T).mean()); tr = pr(unit(np.array([Tt[kind, p, lg, "true"] - Tt[kind, p, lg, "foil"]])))[0]; tc.append(float(e @ tr) - base); cc.append(float(e @ pr(Db[j:j + 1])[0]) - base)  # noqa: E731
    if tc:
        lines.append(f"- Ceiling, Set T, English key step against its own translation, on C1's footing: {np.mean(tc):+.4f}; the separately written story for the same patterns: {np.mean(cc):+.4f} ({np.mean(cc) / np.mean(tc):.0%} of the ceiling), {len(tc)} comparisons")
    # lexical check on the raw text
    R = [r for r in map(json.loads, open(OUT / "items.jsonl")) if r.get("ok") and r["set"] == "I"]; grams = lambda x: {x[i:i + 3] for i in range(max(len(x) - 2, 0))}; G = {(r["pattern_id"], r["lang"]): grams(r["true"]) for r in R}; ls = []  # noqa: E731
    for a, b in pairs:
        ids = sorted({p for p, g in G if g == a} & {p for p, g in G if g == b}); J = lambda u, v: len(u & v) / max(len(u | v), 1); same = np.mean([J(G[p, a], G[p, b]) for p in ids]); oth = np.mean([J(G[p, a], G[q, b]) for p in ids[:30] for q in ids[:30] if p != q]); ls.append(f"{a}-{b} {same - oth:+.4f}")  # noqa: E731
    lines.append("- Lexical check, character 3-gram overlap, same pattern minus other, no model: " + "; ".join(ls))
    # whole stories
    rows = ["", "## C2 and C3, whole stories", "", "| model | C2 same pattern minus other, across languages, centred | C3 same pattern in another language minus another pattern in the same language |", "|---|---|---|"]
    for name, (T, _, _) in Z.items():
        c2, c3 = [], []
        for kind in ("myth", "mechanism"):
            for a, b in pairs:
                ids = [p for p in sorted({k[1] for k in T if k[0] == kind and k[2] == a and k[3] == "true"} & {k[1] for k in T if k[0] == kind and k[2] == b and k[3] == "true"}) if passed("I", kind, p, a) and passed("I", kind, p, b)]
                if len(ids) < 8:
                    continue
                Ta, Tb = np.array([T[kind, p, a, "true"] for p in ids]), np.array([T[kind, p, b, "true"] for p in ids]); M = unit(Ta - Ta.mean(0)) @ unit(Tb - Tb.mean(0)).T; n_ = len(ids); c2.append(np.diag(M) - (M.sum(1) - np.diag(M) + M.sum(0) - np.diag(M)) / (2 * (n_ - 1))); X = unit(Ta) @ unit(Tb).T; Aa, Bb = unit(Ta) @ unit(Ta).T, unit(Tb) @ unit(Tb).T; c3.append(np.diag(X) - ((Aa.sum(1) - 1) + (Bb.sum(1) - 1)) / (2 * (n_ - 1)))
        rows.append(f"| {name} | {f3(np.concatenate(c2))} | {f3(np.concatenate(c3))} |")
    lines += rows
    rows = []
    for name, (T, _, _) in Z.items():
        o = {}
        for kind in ("myth", "mechanism"):
            Rr = []
            for a, b in pairs:
                ids, Da, Db = key_steps(T, kind, a, b, ("true", "foil"), True)
                if Da is not None:
                    u = np.triu_indices(len(ids), 1); Rr.append(spearmanr((Da @ Da.T)[u], (Db @ Db.T)[u]).statistic)
            o[kind] = np.mean(Rr) if Rr else float("nan")
        rows.append(f"{name} myths {o['myth']:+.3f}, mechanisms {o['mechanism']:+.3f}")
    lines += ["", "- C5, within a model, Spearman between two languages' key-step similarity matrices (about 40 x 40 per kind): " + "; ".join(rows)]
    if len(Z) >= 3:
        c1 = {n: float(np.mean(list(C[n]["myth"][0].values()) + list(C[n]["mechanism"][0].values()))) for n in Z}; ks = [n for n in c1 if n in asked]; lines.append(f"- C1 per model against the asked score: Spearman {spearmanr([asked[n] for n in ks], [c1[n] for n in ks]).statistic:+.2f} over {len(ks)} models" + (f"; without the 31B, which also selected the stories, {spearmanr([asked[n] for n in ks if n != 'd31'], [c1[n] for n in ks if n != 'd31']).statistic:+.2f}" if len(ks) >= 4 else "") + ": " + "; ".join(f"{n} {c1[n]:+.4f}" for n in ks))
    lines += ["", "## Set R, real parallels, descriptive: is the nearest other-language story from the same family?", "", "Family labels are shuffled within each language, keeping one story per family and language. Models may know these parallels, and some spread by contact.", ""]
    for name, (_, _, Tr) in Z.items():
        for lab, use in (("all stories", None), ("passing the check", True)):
            ks = sorted({(k[1], k[2]) for k in Tr if k[3] == "true" and (use is None or passed("R", "myth", k[1], k[2]))}); fam = np.array([k[0] for k in ks]); lang = np.array([k[1] for k in ks])
            if len(ks) < 6:
                continue
            Wh = np.array([Tr["myth", k[0], k[1], "true"] for k in ks]); Ke = np.array([Tr["myth", k[0], k[1], "true"] - Tr["myth", k[0], k[1], "foil"] for k in ks])

            def hit(V, lab_):
                V = unit(V - V.mean(0)); C_ = V @ V.T; np.fill_diagonal(C_, -9); C_[lang[:, None] == lang[None, :]] = -9; return float(np.mean(lab_[C_.argmax(1)] == lab_))

            def shuf():
                o = fam.copy()
                for g in set(lang):
                    w = np.where(lang == g)[0]; o[w] = rng.permutation(o[w])
                return o
            nw, nk = [hit(Wh, shuf()) for _ in range(1000)], [hit(Ke, shuf()) for _ in range(1000)]; lines.append(f"- {name}, {lab}, {len(ks)} stories: whole story {hit(Wh, fam):.2f} (shuffled {np.mean(nw):.2f}, shuffles at or above {np.mean(np.array(nw) >= hit(Wh, fam)):.3f}); key step {hit(Ke, fam):.2f} (shuffled {np.mean(nk):.2f}, {np.mean(np.array(nk) >= hit(Ke, fam)):.3f})")
    (OUT / "report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    {"write": write, "label": label, "check": check, "report": report, "capture": lambda: capture(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
