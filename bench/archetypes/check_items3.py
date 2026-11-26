"""Automated checks for the fresh item set (README addendum 9).
  python3 check_items3.py items3/w1.py ...     one or more writer files (writers run this on their own file)
  python3 check_items3.py                      all of items3/w*.py, plus whole-set balance and baselines
  python3 check_items3.py --dir items4 [FILES]  the neuron-670 confirmation set (addendum 15; assignment items4/assign4.py)
  python3 check_items3.py --dir items5 [FILES]  the alpha = 1 test set (addendum 23; assignment items5/assign5.py)
  python3 check_items3.py --dir items7 [FILES]  invented myths and mechanisms (addendum 36; assignment items7/assign7.py)
  python3 check_items3.py --dir items8 [FILES]  the second invented batch (addendum 41; assignment items8/assign8.py)
Each file defines A (myths) and B (mechanisms): lists of (id, pattern, anchor, true, surface, lure, foil, brk),
brk in {"cause", "link"}. FAIL lines must be fixed; WARN lines must be looked at."""
import difflib, importlib.util, math, re, sys, collections
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from freq_terms import TERMS  # noqa: E402
sys.path.insert(0, str(HERE / "items4"))
sys.path.insert(0, str(HERE / "items5"))
from assign4 import TERMS as TERMS4  # noqa: E402
from assign5 import TERMS as TERMS5  # noqa: E402
sys.path.insert(0, str(HERE / "items7"))
from assign7 import TERMS as TERMS7  # noqa: E402   (addendum 36: invented myths and mechanisms)
sys.path.insert(0, str(HERE / "items8"))
try:                                                     # addendum 41: the second invented batch (absent until written)
    from assign8 import TERMS as TERMS8  # noqa: E402
except ImportError:
    TERMS8 = {}
sys.path.insert(0, str(HERE / "items9"))
try:                                                     # addendum 79: the 31B-invented batch (absent until invented)
    from assign9 import TERMS as TERMS9  # noqa: E402
except ImportError:
    TERMS9 = {}
sys.path.insert(0, str(HERE / "obscure"))
try:                                                     # addendum 93: little-seen tales (absent until proposed)
    from assign_obs import TERMS as TERMS_OBS  # noqa: E402
except ImportError:
    TERMS_OBS = {}
sys.path.insert(0, str(HERE / "novels"))
try:                                                     # addendum 93, part C
    from assign_nov import TERMS as TERMS_NOV  # noqa: E402
except ImportError:
    TERMS_NOV = {}
sys.path.insert(0, str(HERE / "novelsB"))
try:                                                     # addendum 93, crossed arm
    from assign_novb import TERMS as TERMS_NOVB  # noqa: E402
except ImportError:
    TERMS_NOVB = {}
TERMS = {**TERMS, **TERMS4, **TERMS5, **TERMS7, **TERMS8, **TERMS9, **TERMS_OBS, **TERMS_NOV, **TERMS_NOVB}
for _g in ("grfiction", "grreal", "grevery", "grhippo"):             # addendum 96
    sys.path.insert(0, str(HERE / _g))
    try:
        TERMS.update(__import__(f"assign_{_g}").TERMS)
    except ImportError:
        pass
from items import A as A1, B as B1  # noqa: E402

LIMITS = {"anchor": (20, 75), "surface": (20, 70), "true": (30, 75), "lure": (30, 75), "foil": (30, 75)}
FIELDS = ("id", "pattern", "anchor", "true", "surface", "lure", "foil", "brk")
STOP = set("""a an the and or but if then than that this these those there their they them he she it its his her hers him
we our you your i me my of in on at to from by for with without into onto over under after before again about as is are was
were be been being has have had do does did not no nor so too very can could would should will just only also each every
one two three all some any more most much many other another such own same what which who whom when where why how while
until because though although whether up down out off back first last next year years time times day days people""".split())
# names of the source stories and terms that name a pattern: none may appear in true, lure, foil or surface options
ANCHOR_WARN_OK = False                                       # addendum 96: rewrite.py sets this for its groups
GIVEAWAY = ["trojan", "achilles", "pandora", "midas", "sour grape", "cry wolf", "cried wolf", "crying wolf", "goldilocks",
            "streisand", "domino", "vicious circle", "vicious cycle", "feedback loop", "catch-22", "golden goose", "goose that",
            "ugly duckling", "new clothes", "damocles", "gordian", "procrustean", "pyrrhic", "herculean", "odyssey", "nemesis",
            "narciss", "sisyph", "promethe", "tantaliz", "cassandra", "judas", "samaritan", "prodigal", "goliath", "babel",
            "writing on the wall", "eleventh hour", "lion's share", "bottleneck", "placebo", "herd immunity", "keystone",
            "butterfly effect", "tipping point", "chain reaction", "compound interest", "snowball", "self-fulfilling",
            "prophecy", "karma", "poetic justice", "irony", "ironic", "moral of", "lesson", "learned that", "learns that",
            "just like", "modern-day", "modern day", "fairy tale", "fairytale", "myth", "legend", "fable", "parable", "bible",
            "biblical"]
FANTASY = ["gods", "spell", "curse", "magic", "wizard", "witch", "dragon", "giant", "monster", "oracle", "fairy"]   # warn in true/foil
STEMS = {"narciss", "sisyph", "promethe", "tantaliz", "procrustean", "herculean", "biblical"}   # prefix matches
SECTION_TERMS = ["law", "paradox", "effect", "principle", "syndrome", "fallacy", "bias", "theorem"]   # warn only


def words(t):
    return re.findall(r"[A-Za-z][A-Za-z'-]*", t)


def content(t):
    return {w.lower().strip("'") for w in words(t) if len(w) >= 5 and w.lower() not in STOP}


STARTERS = STOP | set("""once later soon then there here every after before when while one two three years so but
finally meanwhile eventually instead""".split())


def bare(w):
    return re.sub(r"('s|')$", "", w)


CAPS = {w.strip() for w in open("/usr/share/dict/words") if w[:1].isupper()}   # proper nouns of the system word list


def names(t):
    """capitalized words of the anchor (possessive stripped): every one inside a sentence, and a sentence's first word
    when the word list knows it as a proper noun (Joseph, Samson; not Curious); compared with case kept, so the name
    Job or Lot does not match the words job or lot"""
    out = set()
    for sent in re.split(r"(?<=[.!?:;])\s+|\n", t):
        for k, x in enumerate(words(sent)):
            b = bare(x)
            if x[0].isupper() and len(b) > 1 and b.lower() not in STARTERS and (k > 0 or b in CAPS):
                out.add(b)
    return out


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, "A", []), getattr(m, "B", [])


def check_item(s, it, out):
    tag = f"{s}:{it[0] if it else '?'}"
    if not (isinstance(it, tuple) and len(it) == 8 and all(isinstance(x, str) and x.strip() for x in it)):
        out.append(("FAIL", tag, "not an 8-tuple of non-empty strings")); return
    d = dict(zip(FIELDS, it))
    if d["brk"] not in ("cause", "link"):
        out.append(("FAIL", tag, f"brk must be 'cause' or 'link', got {d['brk']!r}"))
    n = {k: len(words(d[k])) for k in LIMITS}
    for k, (lo, hi) in LIMITS.items():
        if not lo <= n[k] <= hi:
            out.append(("FAIL", tag, f"{k} has {n[k]} words (allowed {lo}-{hi})"))
    tri = [n["true"], n["lure"], n["foil"]]
    if min(tri) == 0 or max(tri) / min(tri) > 1.3:
        out.append(("FAIL", tag, f"true/lure/foil lengths {tri}: longest more than 1.3x the shortest"))
    r = difflib.SequenceMatcher(None, [w.lower() for w in words(d["true"])], [w.lower() for w in words(d["foil"])]).ratio()
    if r < 0.6:
        out.append(("FAIL", tag, f"foil is not a minimal edit of true (word-sequence similarity {r:.2f} < 0.60)"))
    if r > 0.95:
        out.append(("WARN", tag, f"foil is almost identical to true (similarity {r:.2f}); is the key step really broken?"))
    an = names(d["anchor"])
    for k in ("true", "foil"):
        leak = an & {bare(w) for w in words(d[k])}
        if leak:
            out.append(("FAIL", tag, f"{k} reuses names from the anchor: {sorted(leak)}"))
        shared = content(d["anchor"]) & content(d[k])
        if len(shared) >= 3:
            out.append(("WARN", tag, f"{k} shares {len(shared)} content words with the anchor: {sorted(shared)}"))
    for k in ("true", "lure", "foil", "surface"):
        low = d[k].lower()
        hits = [g for g in GIVEAWAY if re.search(rf"\b{re.escape(g)}" + ("" if g in STEMS else r"\b"), low)]
        soft = [g for g in FANTASY if re.search(rf"\b{g}", low)]
        if soft and k in ("true", "foil"):
            out.append(("WARN", tag, f"{k} contains {soft}: the true match and foil should be ordinary modern stories"))
        if hits and k in ("true", "foil"):
            out.append(("FAIL", tag, f"{k} contains a giveaway word: {hits}"))
        elif hits and not (ANCHOR_WARN_OK and all(g in d["anchor"].lower() for g in hits)):
            out.append(("WARN", tag, f"{k} contains {hits} (fine only if it is part of the anchor's own surface)"))
        terms = [t for t in SECTION_TERMS if re.search(rf"\b{t}\b", low)]
        if terms:
            out.append(("WARN", tag, f"{k} contains {terms}: make sure no pattern is named"))
    if s == "B" and any(g in d["anchor"].lower() for g in ("bottleneck", "feedback", "placebo", "leverage", "paradox", "law of")):
        out.append(("WARN", tag, "mechanism anchor names its mechanism; describe it instead"))
    # the phrases that name this item (freq_terms.py): never in true or foil; for mechanisms nowhere
    for k in (("anchor", "true", "surface", "lure", "foil") if s == "B" else ("true", "foil")):
        own = [t for t in TERMS.get(d["id"], []) if re.search(rf"\b{re.escape(t.lower()).replace(chr(39), '[' + chr(39) + chr(8217) + ']')}\b", d[k].lower())]
        if own:
            out.append(("FAIL", tag, f"{k} names the item itself: {own}"))


def balance(s, items, out, label):
    if len(items) < 10:
        return
    longest = shortest = 0
    for it in items:
        n = [len(words(it[3])), len(words(it[5])), len(words(it[6]))]
        longest += n[0] == max(n) and n.count(max(n)) == 1; shortest += n[0] == min(n) and n.count(min(n)) == 1
    k = len(items)
    msg = f"{label} set {s}: true is the strictly longest of true/lure/foil in {longest}/{k}, strictly shortest in {shortest}/{k}"
    out.append(("FAIL" if longest > 0.4 * k or shortest < 0.2 * k else "ok", f"{label}:{s}", msg + " (need longest <= 40%, shortest >= 20%)"))
    brk = collections.Counter(it[7] for it in items)
    if k >= 10 and min(brk.get("cause", 0), brk.get("link", 0)) < 0.4 * k:
        out.append(("FAIL", f"{label}:{s}", f"brk split {dict(brk)}: keep each type between 40% and 60%"))


def baselines(sets):
    """lexical (tf-idf cosine with the anchor) and length pickers on the 3-way choice; ideal: true near 1/3 or below"""
    docs = [t for items in sets.values() for it in items for t in (it[2], it[3], it[5], it[6], it[4])]
    df = collections.Counter(w for t in docs for w in set(x.lower() for x in words(t)))
    N = len(docs)

    def vec(t):
        c = collections.Counter(x.lower() for x in words(t))
        v = {w: (1 + math.log(n)) * math.log(N / df[w]) for w, n in c.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1
        return {w: x / z for w, x in v.items()}

    cos = lambda a, b: sum(x * b.get(w, 0) for w, x in a.items())  # noqa: E731
    for s, items in sets.items():
        lex, ln, pair = collections.Counter(), collections.Counter(), 0
        for it in items:
            a = vec(it[2]); opts = {"true": it[3], "lure": it[5], "foil": it[6]}
            sc = {k: cos(a, vec(t)) for k, t in opts.items()}; lex[max(sc, key=sc.get)] += 1
            L = {k: len(words(t)) for k, t in opts.items()}; top = [k for k, v in L.items() if v == max(L.values())]
            ln[top[0] if len(top) == 1 else "tie"] += 1
            pair += cos(a, vec(it[3])) > cos(a, vec(it[4]))
        k = len(items)
        print(f"baseline set {s} ({k} items): tf-idf picks true {lex['true']}, lure {lex['lure']}, foil {lex['foil']}; "
              f"strictly longest is true {ln['true']}, lure {ln['lure']}, foil {ln['foil']}, tie {ln['tie']}; tf-idf prefers true over surface in {pair}/{k}")


def main(paths):
    d = "items3"
    if paths[:1] == ["--dir"]:
        d, paths = paths[1], paths[2:]
    whole = not paths
    paths = paths or sorted(str(p) for p in (HERE / d).glob("w*.py"))
    d = d.split("_")[0] if d.startswith(("items7_", "items8_")) else d     # addendum 50: rewritten sets use their batch's assignment
    out, sets, seen = [], {"A": [], "B": []}, collections.Counter()
    old = {it[0] for it in A1 + B1}
    if d != "items3":
        import fresh
        old |= {it[0] for it in fresh.A + fresh.B}
    if d in ("items5", "items7"):
        from assign4 import MECHANISMS as M4, MYTHS as Y4
        old |= {r[0] for r in Y4 + M4}
    if d in ("items7", "items8"):
        from assign5 import MECHANISMS as M5, MYTHS as Y5
        old |= {r[0] for r in Y5 + M5}
    if d == "items8":
        from assign7 import MECHANISMS as M7, MYTHS as Y7
        old |= {r[0] for r in Y7 + M7}
    for p in paths:
        A, B = load(p)
        for s, items in (("A", A), ("B", B)):
            for it in items:
                check_item(s, it, out)
                if isinstance(it, tuple) and it:
                    seen[it[0]] += 1
                    if it[0] in old:
                        out.append(("FAIL", f"{s}:{it[0]}", "id already used in the original set"))
            balance(s, items, out, Path(p).stem)
            sets[s] += [it for it in items if isinstance(it, tuple) and len(it) == 8]
    for i, c in seen.items():
        if c > 1:
            out.append(("FAIL", i, f"id used {c} times"))
    if whole:
        if d == "items3":
            sys.path.insert(0, str(HERE / "items3"))
            from assign import MECHANISMS, MYTHS
        elif d == "items5":
            from assign5 import MECHANISMS, MYTHS
        elif d == "items7":
            from assign7 import MECHANISMS, MYTHS
        elif d == "items8":
            from assign8 import MECHANISMS, MYTHS
        else:
            from assign4 import MECHANISMS, MYTHS
        for s, ref in (("A", MYTHS), ("B", MECHANISMS)):
            missing = {r[0] for r in ref} - {it[0] for it in sets[s]}
            if missing:
                out.append(("FAIL", f"all:{s}", f"{len(missing)} assigned ids missing: {sorted(missing)[:8]}"))
        for s in "AB":
            balance(s, sets[s], out, "all")
        mean = {s: sum(len(words(t)) for it in sets[s] for t in (it[3], it[5], it[6])) / max(1, 3 * len(sets[s])) for s in "AB"}
        out.append(("FAIL" if max(mean.values()) > 1.15 * min(mean.values()) else "ok", "all",
                    f"mean option length: myths {mean['A']:.1f} words, mechanisms {mean['B']:.1f} (must be within 15%)"))
    for lvl in ("FAIL", "WARN", "ok"):
        for l, tag, msg in out:
            if l == lvl:
                print(f"{l:4s} {tag}: {msg}")
    print(f"items: A {len(sets['A'])}, B {len(sets['B'])}; FAIL {sum(l == 'FAIL' for l, *_ in out)}, WARN {sum(l == 'WARN' for l, *_ in out)}")
    baselines(sets)
    return any(l == "FAIL" for l, *_ in out)


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
