"""Addendum 221 rung 2b: comprehension tests, scored from log-probabilities so a base model needs no answer format.
python fluent/comp.py static          BLiMP, COPA, last-word items and the question passages, from local public files only -> fluent/comp/
python fluent/comp.py write           Inkling 2 Mini (Mac, serve_mac.sh on 127.0.0.1:8177) writes one question with 1 right and 3 wrong answers a passage
python fluent/comp.py verify          Qwen3.8 27B (MLX, the Mac) must pick the written answer from the four; failures go back to `write` (at most 2 rewrites)
python fluent/comp.py score NAME...   on the 4090: every test for each checkpoint -> fluent/runs/comp/NAME.json
python fluent/comp.py write2 | verify2 test v2 (addendum 222): invented worlds, paraphrase, distractors in the passage, two-sentence inference
Everything here is public text (Gutenberg held-out books, BLiMP, COPA); nothing from Jung."""
import glob, json, random, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; COMP = HERE / "comp"; COMP.mkdir(exist_ok=True); SW = HERE.parent / "smallworld"
MCQ_ASK = ("Read the passage. Write ONE multiple-choice question that can only be answered by reading this passage (not from general knowledge), "
           "with the right answer and three wrong answers. All four answers must be short (1 to 6 words), the same kind of thing, and plausible; "
           "the wrong ones must be clearly wrong given the passage. Reply with JSON only: {{\"question\": \"...\", \"answer\": \"...\", \"wrong\": [\"...\", \"...\", \"...\"]}}"
           "{extra}\n\nPassage:\n{p}")


def held_text():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(HERE.parent / "transfer" / "tok" / "bpe8k.json")); h = np.fromfile(SW / "gut_tok" / "held.bin", dtype=np.uint16)
    return tk.decode(h.tolist())


def static():
    import pyarrow.parquet as pq
    rng = random.Random(221); out = {}
    rows = []
    for f in sorted(glob.glob(str(SW / "bench" / "blimp" / "*.parquet"))):
        t = pq.read_table(f).to_pylist()[:100]; name = Path(f).name.split("__")[1]
        rows += [{"set": name, "good": r["sentence_good"], "bad": r["sentence_bad"]} for r in t]
    out["blimp"] = rows
    t = pq.read_table(str(SW / "bench" / "copa" / "copa_balanced__default__test__0000.parquet")).to_pylist()
    out["copa"] = [{"premise": r["premise"], "question": r["question"], "choices": [r["choice1"], r["choice2"]], "label": int(r["label"])} for r in t]
    paras = [p for p in held_text().split("\n\n") if p.strip()]
    lw = []
    for p in paras:
        w = p.split()
        if not 60 <= len(w) <= 250:
            continue
        last = re.sub(r"[^A-Za-z]", "", w[-1])
        if len(last) >= 4 and re.search(r"\b" + re.escape(last) + r"\b", " ".join(w[:-1]), re.I) and w[-1].rstrip(".!?\"'”’);:,").endswith(last):
            ctx = p[:p.rstrip().rfind(w[-1])].rstrip(); sents = re.split(r"(?<=[.!?])\s+", ctx)
            lw.append({"context": ctx, "last_sentence": sents[-1], "word": last})
    rng.shuffle(lw); out["lastword"] = lw[:1000]
    ps = [p for p in paras if 150 <= len(p.split()) <= 300]; rng.shuffle(ps); out["passages"] = [{"pid": i, "passage": p} for i, p in enumerate(ps[:300])]
    for k, v in out.items():
        with open(COMP / f"{k}.jsonl", "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
        print(k, len(v), flush=True)


def ink(prompt):
    body = {"model": "inkling", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200, "temperature": 0.7}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8177/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return r["choices"][0]["message"]["content"]


def lint(q, p):
    try:
        d = json.loads(re.search(r"\{.*\}", q, re.S).group(0)); opts = [d["answer"]] + list(d["wrong"])
    except Exception:
        return None, "unparsable"
    if len(d["wrong"]) != 3 or any(not isinstance(o, str) or not o.strip() or len(o.split()) > 6 for o in opts):
        return None, "answer form"
    if len({o.strip().lower() for o in opts}) != 4:
        return None, "answers repeat"
    return {"question": d["question"].strip(), "answer": d["answer"].strip(), "wrong": [o.strip() for o in d["wrong"]]}, "ok"


def write():
    """one item a passage, written by Inkling; 4 at a time (the server's slots). Passages already holding an accepted item are skipped"""
    from concurrent.futures import ThreadPoolExecutor
    P = [json.loads(l) for l in open(COMP / "passages.jsonl")]; f = COMP / "mcq_raw.jsonl"
    have = {}
    if f.exists():
        for l in open(f):
            r = json.loads(l); have.setdefault(r["pid"], []).append(r)
    todo = [p for p in P if not any(r.get("verified") for r in have.get(p["pid"], [])) and len(have.get(p["pid"], [])) < 3
            and not (have.get(p["pid"]) and have[p["pid"]][-1].get("verified") is None)]
    def one(p):
        tries = have.get(p["pid"], []); extra = "" if not tries else " Write a different question from the one before; make sure exactly one answer is right."
        for _ in range(2):
            item, why = lint(ink(MCQ_ASK.format(p=p["passage"], extra=extra)), p["passage"])
            if item:
                return {"pid": p["pid"], **item, "attempt": len(tries) + 1, "verified": None}
        return {"pid": p["pid"], "attempt": len(tries) + 1, "failed_lint": why, "verified": False}
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex, open(f, "a") as out:
        for i, r in enumerate(ex.map(one, todo)):
            out.write(json.dumps(r) + "\n"); out.flush()
            if i % 25 == 0:
                print(f"{i + 1}/{len(todo)} written, {time.time() - t0:.0f} s", flush=True)
    print("WRITE_DONE", len(todo), flush=True)


def verify():
    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(4 * 2 ** 30); m, tk = load(str(Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit")); L = [tk.encode(c)[-1] for c in "ABCD"]
    P = {json.loads(l)["pid"]: json.loads(l)["passage"] for l in open(COMP / "passages.jsonl")}; f = COMP / "mcq_raw.jsonl"; rows = [json.loads(l) for l in open(f)]; rng = random.Random(7)
    for r in rows:
        if r.get("verified") is not None or "question" not in r:
            continue
        opts = [r["answer"]] + r["wrong"]; order = list(range(4)); rng.shuffle(order)
        msg = (f"Passage:\n{P[r['pid']]}\n\nQuestion: {r['question']}\n" + "".join(f"{'ABCD'[k]}. {opts[o]}\n" for k, o in enumerate(order)) + "\nAnswer with one letter.")
        try:
            s = tk.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        except TypeError:
            s = tk.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
        lg = m(mx.array([tk.encode(s)]))[0, -1]; pick = int(np.argmax([float(lg[i]) for i in L])); r["verified"] = order[pick] == 0
    with open(f, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r.get("verified")]; print(f"verified {len(ok)} of {len(rows)} attempts; passages with an item: {len({r['pid'] for r in ok})}", flush=True)
    with open(COMP / "mcq.jsonl", "w") as out:
        seen = set()
        for r in ok:
            if r["pid"] not in seen:
                seen.add(r["pid"]); out.write(json.dumps({"pid": r["pid"], "passage": P[r["pid"]], "question": r["question"], "answer": r["answer"], "wrong": r["wrong"]}) + "\n")


# ---------------------------------------------------------------- test v2 (addendum 222): understanding, not matching

KINDS = {
 "invented": ("Invent a short passage (150 to 250 words) about a FICTIONAL place and its people: invented names, customs and facts, with at least five concrete facts, "
              "and at least two facts that go against common knowledge (for example, a river that runs uphill). Then write ONE question about it that can only be "
              "answered from the passage, the right answer and three wrong answers (each 1 to 6 words, the same kind of thing), ALL FOUR being names or things the passage mentions word for word. "
              "Reply with JSON only: {{\"passage\": \"...\", \"question\": \"...\", \"answer\": \"...\", \"wrong\": [\"...\", \"...\", \"...\"]}}{extra}"),
 "paraphrase": ("Read the passage. Write ONE question that can only be answered from it, the right answer and three wrong answers (each 1 to 8 words). The right answer must be "
                "REWORDED: it must not copy any run of three or more words from the passage, though it means what the passage says. The wrong answers are also not copied "
                "from the passage, and each wrong answer uses about as many of the passage's own words as the right answer does, so that matching words cannot pick the "
                "right one. Reply with JSON only: {{\"question\": \"...\", \"answer\": \"...\", \"wrong\": [\"...\", \"...\", \"...\"]}}{extra}\n\nPassage:\n{p}"),
 "inpassage": ("Read the passage. Write ONE question that can only be answered from it, with four answer options that ALL appear word for word in the passage "
               "(names, places, things or short phrases copied exactly), of which exactly one answers the question and three do not. Each option 1 to 6 words. "
               "Reply with JSON only: {{\"question\": \"...\", \"answer\": \"...\", \"wrong\": [\"...\", \"...\", \"...\"]}}{extra}\n\nPassage:\n{p}"),
 "inference": ("Read the passage. Write ONE question whose answer is NOT stated in any single sentence but follows from combining two facts the passage states in different "
               "sentences. Give the right answer and three wrong answers (each 1 to 6 words); the right answer must not appear word for word in the passage. "
               "Reply with JSON only: {{\"question\": \"...\", \"answer\": \"...\", \"wrong\": [\"...\", \"...\", \"...\"], \"facts\": [\"first fact\", \"second fact\"]}}{extra}\n\nPassage:\n{p}"),
}
N2 = 110                                                         # items attempted per kind (75 or more kept is the aim)


def runs3(a, b):
    """True if a and b share a run of three or more words"""
    wa, wb = re.findall(r"[a-z0-9']+", norm(a).lower()), re.findall(r"[a-z0-9']+", norm(b).lower()); tri = {tuple(wb[i:i + 3]) for i in range(len(wb) - 2)}
    return any(tuple(wa[i:i + 3]) in tri for i in range(len(wa) - 2))


def norm(t):
    return t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


def has_words(text, phrase):
    """phrase occurs in text as whole words (curly quotes normalised)"""
    return re.search(r"(?<![a-z0-9])" + re.escape(norm(phrase).lower()) + r"(?![a-z0-9])", norm(text).lower()) is not None


def lint2(kind, q, passage):
    try:
        d = json.loads(re.search(r"\{.*\}", q, re.S).group(0)); opts = [d["answer"]] + list(d["wrong"]); question = d["question"]
        if kind == "invented":
            passage = d["passage"]
        if not (isinstance(question, str) and isinstance(passage, str) and len(d["wrong"]) == 3 and all(isinstance(o, str) for o in opts)):
            return None, "answer form"
    except Exception:
        return None, "unparsable"
    if kind == "invented" and not 120 <= len(passage.split()) <= 300:
        return None, "passage length"
    most = 8 if kind == "paraphrase" else 6
    if any(not o.strip() or len(o.split()) > most for o in opts) or len({norm(o).strip().lower() for o in opts}) != 4:
        return None, "answer form"
    if kind == "paraphrase" and any(runs3(norm(o), norm(passage)) or has_words(passage, o) for o in opts):
        return None, "copied"
    if kind == "inpassage" and not all(has_words(passage, o) for o in opts):
        return None, "option not in passage"
    if kind == "invented" and not all(has_words(passage, o) for o in opts):
        return None, "invented options not all in passage"
    if kind == "inference" and has_words(passage, d["answer"]):
        return None, "answer stated"
    return {"kind": kind, "passage": passage, "question": question.strip(), "answer": d["answer"].strip(), "wrong": [o.strip() for o in d["wrong"]],
            "facts": d.get("facts") if kind == "inference" else None}, "ok"


def prose_ok(p):
    """amendment 222a: a passage counts only as prose: at most 4% digits and at least 2 sentences (no indexes, tables, genealogies or contents pages)"""
    return sum(ch.isdigit() for ch in p) / max(1, len(p)) <= 0.04 and len(re.findall(r"[a-z][.!?](\s|$)", p)) >= 2


def v2_spares():
    """replacement prose passages for slots whose assigned passage is not prose: held-out paragraphs of 120-149 or 301-340 words (outside the first
    pool, so no existing slot changes), prose only, in a seeded order"""
    used = {json.loads(l)["passage"] for l in open(COMP / "passages.jsonl")}; rng = random.Random(2222)
    ps = [p for p in held_text().split("\n\n") if (120 <= len(p.split()) < 150 or 300 < len(p.split()) <= 340) and p not in used and prose_ok(p)]; rng.shuffle(ps); return ps


def v2_passages():
    """real held-out passages for the three real-text kinds, none of them in the first passage test"""
    used = {json.loads(l)["passage"] for l in open(COMP / "passages.jsonl")}; rng = random.Random(222)
    ps = [p for p in held_text().split("\n\n") if 150 <= len(p.split()) <= 300 and p not in used]; rng.shuffle(ps); return ps


def write2():
    from concurrent.futures import ThreadPoolExecutor
    f = COMP / "mcq2_raw.jsonl"; rows = [json.loads(l) for l in open(f)] if f.exists() else []
    done = {}                                                    # (kind, slot) -> attempts so far; a slot with an accepted or pending item is skipped
    for r in rows:
        done.setdefault((r["kind"], r["slot"]), []).append(r)
    ps = v2_passages(); spares = iter(v2_spares()); jobs = []
    for k, kind in enumerate(KINDS):
        for slot in range(N2):
            tries = done.get((kind, slot), [])
            if any(t.get("verified") for t in tries) or (tries and tries[-1].get("verified") is None) or len(tries) >= 3:
                continue
            p = ps[(k - 1) * N2 + slot] if kind != "invented" else ""      # invented (k = 0) uses no real passage
            if kind != "invented" and not prose_ok(p):
                p = next(spares)                                         # amendment 222a: a prose passage instead of an index, table or list
            jobs.append((kind, slot, p, len(tries)))
    def one(job):
        kind, slot, p, n = job; extra = "" if not n else " Make sure exactly one answer is right."; why = "no reply"
        for _ in range(2):
            try:
                item, why = lint2(kind, ink(KINDS[kind].format(p=p, extra=extra)), p)
            except Exception as e:                               # a network error or an odd reply must not stop the run
                item, why = None, f"error {type(e).__name__}"
            if item:
                return {"slot": slot, **item, "attempt": n + 1, "verified": None}
        return {"kind": kind, "slot": slot, "attempt": n + 1, "failed_lint": why, "verified": False}
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex, open(f, "a") as out:
        for i, r in enumerate(ex.map(one, jobs)):
            out.write(json.dumps(r) + "\n"); out.flush()
            if i % 40 == 0:
                print(f"{i + 1}/{len(jobs)} written, {time.time() - t0:.0f} s", flush=True)
    print("WRITE2_DONE", len(jobs), flush=True)


def verify2():
    """Qwen3.8 27B must pick the written answer with the passage; for invented worlds it must not pick it without the passage"""
    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(4 * 2 ** 30); m, tk = load(str(Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit")); L = [tk.encode(c)[-1] for c in "ABCD"]; rng = random.Random(8)
    def pick(passage, r, order):
        opts = [r["answer"]] + r["wrong"]; head = f"Passage:\n{passage}\n\n" if passage else ""
        msg = head + f"Question: {r['question']}\n" + "".join(f"{'ABCD'[k]}. {opts[o]}\n" for k, o in enumerate(order)) + "\nAnswer with one letter."
        try:
            s = tk.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        except TypeError:
            s = tk.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
        lg = m(mx.array([tk.encode(s)]))[0, -1]; return order[int(np.argmax([float(lg[i]) for i in L]))]
    f = COMP / "mcq2_raw.jsonl"; rows = [json.loads(l) for l in open(f)]
    for r in rows:
        if r.get("verified") and (r["kind"] != "invented" and not prose_ok(r["passage"]) or r["kind"] == "inference" and "needs_combining" not in r or overlap_baseline([r]) == 1.0):
            r["verified"] = None                                         # amendment 222a: round-1 items face the new checks
        if r.get("verified") is not None or "question" not in r:
            continue
        if r["kind"] != "invented" and not prose_ok(r["passage"]):
            r["verified"] = False; r["rejected"] = "not prose"; continue         # amendment 222a
        if overlap_baseline([r]) == 1.0:                                 # amendment 222b: word matching alone picks the answer uniquely
            r["verified"] = False; r["rejected"] = "overlap answers it"; continue
        order = list(range(4)); rng.shuffle(order); ok = pick(r["passage"], r, order) == 0
        if ok and r["kind"] == "inference":                               # amendment 222a: the answer must need two statements combined
            msg = (f"Passage:\n{r['passage']}\n\nQuestion: {r['question']}\nAnswer: {r['answer']}\n\nIs this answer stated directly in a single sentence of the passage (A), "
                   "or does reaching it require combining information from two or more different sentences (B)? Answer with one letter.")
            try:
                s_ = tk.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
            except TypeError:
                s_ = tk.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
            lg = m(mx.array([tk.encode(s_)]))[0, -1]; ok = float(lg[L[1]]) > float(lg[L[0]]); r["needs_combining"] = ok
        if ok and r["kind"] == "invented":
            order = list(range(4)); rng.shuffle(order); ok = pick("", r, order) != 0; r["closed_book_right"] = not ok
        r["verified"] = ok
    with open(f, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    keep, seen = [], set()
    for r in sorted(rows, key=lambda r: (r["kind"], r["slot"], r["attempt"])):
        if r.get("verified") and (r["kind"], r["slot"]) not in seen and sum(x["kind"] == r["kind"] for x in keep) < 75:   # the registered 300: 75 a kind
            seen.add((r["kind"], r["slot"])); keep.append({k: r[k] for k in ("kind", "slot", "passage", "question", "answer", "wrong")})
    with open(COMP / "mcq2.jsonl", "w") as out:
        for r in keep:
            out.write(json.dumps(r) + "\n")
    by = {}
    for r in keep:
        by[r["kind"]] = by.get(r["kind"], 0) + 1
    print("kept by kind", by, "of", N2, "slots each; overlap rule:", {k: round(overlap_baseline([r for r in keep if r["kind"] == k]), 3) for k in by}, flush=True)


def overlap_baseline(items):
    """the no-model word-overlap rule (stop words out, ties split evenly), as in addendum 221's check"""
    stop = set("the a an of to in and or is was were be by for on at with as that this it its from his her their he she they".split()); w = lambda t: set(re.findall(r"[a-z']+", t.lower())) - stop  # noqa: E731
    tot = 0.0
    for r in items:
        P = w(r["passage"]); opts = [r["answer"]] + r["wrong"]; sc = [len(w(o) & P) / max(1, len(w(o))) for o in opts]; b = max(sc); tied = [i for i, x in enumerate(sc) if x == b]
        tot += (1 / len(tied)) if 0 in tied else 0
    return tot / max(1, len(items))


# ---------------------------------------------------------------- scoring on the 4090

def score(names):
    import torch, torch.nn.functional as F
    sys.path.insert(0, str(HERE)); import train as T
    tests = {k: [json.loads(l) for l in open(COMP / f"{k}.jsonl")] for k in ("blimp", "copa", "lastword", "mcq", "mcq2") if (COMP / f"{k}.jsonl").exists()}
    for name in names:
        c = json.load(open(T.RUNS / f"{name}.json"))["config"]; tk = T.tokenizer(c["tok"]); m = T.Toy(c, tk.get_vocab_size()).to(T.DEV)
        m.load_state_dict(torch.load(T.PRIV / "ckpt" / f"{name}.pt", map_location=T.DEV, weights_only=True)); m.eval(); start = tk.encode("\n\n").ids; t0 = time.time()

        @torch.no_grad()
        def lp(ctx, cont):
            """sum and mean log-probability of the tokens of cont after ctx (both strings; context cut on the left to fit)"""
            a = start + tk.encode(ctx).ids if ctx else list(start); b = tk.encode(cont).ids; ids = (a + b)[-(c["ctx"] + 1):]; k = len(b)
            x = torch.tensor([ids[:-1]], device=T.DEV)
            with T.amp():
                lg = m(x)[0].float()
            l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=T.DEV)[:, None])[:, 0]
            return float(l.sum()), float(l.mean())

        def credit(scores):
            """expected accuracy when ties are broken at random: the right option is index 0"""
            b = max(scores); tied = [i for i, x in enumerate(scores) if x == b]; return (1.0 / len(tied)) if 0 in tied else 0.0

        @torch.no_grad()
        def top1(ctx):
            ids = (start + tk.encode(ctx).ids)[-c["ctx"]:]
            with T.amp():
                return int(m(torch.tensor([ids], device=T.DEV))[0, -1].argmax())

        res = {}
        if "blimp" in tests:
            ok = [lp("", r["good"])[0] > lp("", r["bad"])[0] for r in tests["blimp"]]; res["blimp"] = float(np.mean(ok))
            by = {}
            for r, o in zip(tests["blimp"], ok):
                by.setdefault(r["set"], []).append(o)
            res["blimp_sets_below_chance"] = int(sum(np.mean(v) < 0.5 for v in by.values()))
        if "copa" in tests:
            ok = []
            for r in tests["copa"]:
                link = " because" if r["question"] == "cause" else " so"; pre = r["premise"].rstrip(".") + link
                sc = [lp(pre, " " + ch[0].lower() + ch[1:])[1] for ch in r["choices"]]; ok.append(0.5 if sc[0] == sc[1] else float(int(np.argmax(sc)) == r["label"]))
            res["copa"] = float(np.mean(ok))
        if "lastword" in tests:
            full = [top1(r["context"]) == tk.encode(" " + r["word"]).ids[0] for r in tests["lastword"]]
            short = [top1(r["last_sentence"]) == tk.encode(" " + r["word"]).ids[0] for r in tests["lastword"]]
            res["lastword_full"] = float(np.mean(full)); res["lastword_last_sentence"] = float(np.mean(short))
        if "mcq" in tests:
            w, wo = [], []
            for r in tests["mcq"]:
                opts = [r["answer"]] + r["wrong"]
                w.append(credit([lp(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o)[1] for o in opts]))
                wo.append(credit([lp(f"Question: {r['question']}\nAnswer:", " " + o)[1] for o in opts]))
            res["mcq_with_passage"] = float(np.mean(w)); res["mcq_without"] = float(np.mean(wo)); res["mcq_gain"] = res["mcq_with_passage"] - res["mcq_without"]; res["mcq_n"] = len(w)
            res["mcq_items"] = [[r["pid"], a, b] for r, a, b in zip(tests["mcq"], w, wo)]      # per question: with, without (for subset reads)
        if "mcq2" in tests:                                          # test v2, by kind: with the passage, without it, gain
            for kind in KINDS:
                items = [r for r in tests["mcq2"] if r["kind"] == kind]
                if not items:
                    continue
                w = [credit([lp(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o)[1] for o in [r["answer"]] + r["wrong"]]) for r in items]
                wo = [credit([lp(f"Question: {r['question']}\nAnswer:", " " + o)[1] for o in [r["answer"]] + r["wrong"]]) for r in items]
                res[f"v2_{kind}"] = {"n": len(items), "with": float(np.mean(w)), "without": float(np.mean(wo)), "gain": float(np.mean(w) - np.mean(wo)), "overlap_rule": overlap_baseline(items)}
        res["s"] = round(time.time() - t0); (T.RUNS / "comp").mkdir(exist_ok=True); json.dump(res, open(T.RUNS / "comp" / f"{name}.json", "w"), indent=1); print(name, res, flush=True)


if __name__ == "__main__":
    {"static": static, "write": write, "verify": verify, "write2": write2, "verify2": verify2}.get(sys.argv[1], lambda: score(sys.argv[2:]))()
