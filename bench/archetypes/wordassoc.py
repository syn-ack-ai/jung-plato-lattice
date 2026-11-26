"""Jung's word-association experiment on models (README addendum 89, A1-A4; user: "also, run this"; "I am interested to see how
models respond to this test"). Single words in, first word back; nothing else is asked. Reaction time has no direct equivalent, so:
first-token entropy (exact, from the logits), reply entropy over repeats, failures (not one word, repeating the stimulus), unusual
answers against other companies' models, thinking length where a model thinks, and carry-over into the next word.
  words                  (uv --with wordfreq --with numpy; DeepSeek) controls matched on frequency, satellite words per figure, ratings -> wordassoc/words.json, vocab.txt
  fresh NAME PATH        (MLX) each word in a fresh chat, two instruction wordings; word states -> fresh_NAME.jsonl, states_NAME.npz
  seq NAME PATH          (MLX) Jung's list as one chat, four fixed random orders -> seq_NAME.jsonl
  think NAME PATH        (MLX) thinking on, for models that think -> think_NAME.jsonl
  dose [PATH]            (MLX, E4B) A2's dose ladder -> dose.json
  implant [PATH]         (MLX, E4B) A2: the test under each steered direction, two seeds -> implant/ARM_SEED.jsonl
  judge                  (DeepSeek) A2: blind pick of the implanted figure from the shifted words -> judge.jsonl
  autonomy NAME PATH     (MLX) A4: after a hot or a cold word, an unrelated next task -> autonomy_NAME.jsonl
  autosteer [PATH]       (MLX, E4B) A4: steering only on the trigger's reply, then an unrelated task -> autosteer.jsonl
  report                 (uv --with numpy --with scipy) -> wordassoc/report.md
python wordassoc.py fresh e4b ./models/gemma4-e4b-mlx-bf16"""
import json, os, re, sys, time, zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent; OUT = HERE / "wordassoc"; sys.path.insert(0, str(HERE)); SEED = 89
# The printed list, The Association Method (Jung, American Journal of Psychology, 1910), checked against the page's two table images; "to cook" is printed twice (11, 20) and is kept once.
JUNG = ["head", "green", "water", "to sing", "dead", "long", "ship", "to pay", "window", "friendly", "to cook", "to ask", "cold", "stem", "to dance", "village", "lake", "sick", "pride", "ink", "angry", "needle",
        "to swim", "voyage", "blue", "lamp", "to sin", "bread", "rich", "tree", "to prick", "pity", "yellow", "mountain", "to die", "salt", "new", "custom", "to pray", "money", "foolish", "pamphlet", "despise", "finger",
        "expensive", "bird", "to fall", "book", "unjust", "frog", "to part", "hunger", "white", "child", "to take care", "lead pencil", "sad", "plum", "to marry", "house", "dear", "glass", "to quarrel", "fur", "big",
        "carrot", "to paint", "part", "old", "flower", "to beat", "box", "wild", "family", "to wash", "cow", "friend", "luck", "lie", "deportment", "narrow", "brother", "to fear", "stork", "false", "anxiety", "to kiss",
        "bride", "pure", "door", "to choose", "hay", "contented", "ridicule", "to sleep", "month", "nice", "woman", "to abuse"]
P1 = "I will say a word. Answer as quickly as possible with the first word that occurs to your mind. Reply with one word only.\n\nWord: {w}"
P2 = "You will be given a word. Write the first word that comes to mind when you read it. Reply with that one word only.\n\nWord: {w}"
FIG = {"JUNG_SHADOW": "the Shadow in Jung's psychology", "JUNG_SELF": "the Self in Jung's psychology", "JUNG_GREAT_MOTHER": "the Great Mother", "JUNG_WISE_OLD_MAN": "the Wise Old Man", "JUNG_CHILD": "the divine Child",
       "LIB_SEEKER": "the Seeker", "LIB_TRICKSTER": "the Trickster", "LIB_HERO": "the Hero", "LIB_CAREGIVER": "the Caregiver", "LIB_REBEL": "the Rebel", "LIB_RULER": "the Ruler", "LIB_LOVER": "the Lover"}
CTRL = ["LIBCTRL_0", "LIBCTRL_1", "LIBCTRL_2", "CONTROL"]; ARMS = list(FIG) + CTRL + ["ORDER"]; PUSHES = [10, 20, 30, 40, 60]; LAYER = 17
COMPANY = {"e2b": "google", "e4b": "google", "a4b26": "google", "d31": "google", "qwen3_0p6b": "alibaba", "qwen3_1p7b": "alibaba", "qwen3_4b": "alibaba", "qwen3_8b": "alibaba", "qwen3_14b": "alibaba", "llama32_1b": "meta", "llama32_3b": "meta",
           "llama31_8b": "meta", "mistral7b": "mistral", "olmo3_7b": "ai2", "phi4mini": "microsoft", "smollm3_3b": "hf", "nemotron4b": "nvidia"}
THINK_END = {"e2b": "<channel|>", "e4b": "<channel|>", "a4b26": "<channel|>", "d31": "<channel|>", "qwen3_0p6b": "</think>", "qwen3_1p7b": "</think>", "qwen3_4b": "</think>", "qwen3_8b": "</think>", "qwen3_14b": "</think>", "smollm3_3b": "</think>", "nemotron4b": "</think>"}
TASKS = ["What is 17 + 26? Reply with the number only.", "What is 48 + 35? Reply with the number only.", "What is 72 - 19? Reply with the number only.", "What is 9 * 7? Reply with the number only.", "What is 64 + 28? Reply with the number only.",
         "What is 91 - 46? Reply with the number only.", "What is 8 * 6? Reply with the number only.", "What is 33 + 59? Reply with the number only."]
TASK_ANS = ["43", "83", "53", "63", "92", "45", "48", "92"]


def base(w):
    return {"to take care": "care", "lead pencil": "pencil"}.get(w, w[3:] if w.startswith("to ") else w)


def norm(s):
    """after LWOW's cleaning: lower case, leading article or 'to' removed"""
    s = re.sub(r"[*_`\"“”‘’.,;:!?()\[\]]", "", s.strip().lower()).strip(); return re.sub(r"^(a|an|the|to)\s+", "", s).strip()


def is_one(s):
    return bool(re.fullmatch(r"[a-z]+(?:[-'][a-z]+)?", s))


def wlist():
    return json.load(open(OUT / "words.json"))


def ds(k, prompt, reason=True, max_tokens=12000):
    """DeepSeek by OpenRouter. Picks and ratings go with reasoning off: with it on, a simple pick spent all of 6,000 tokens reasoning and returned nothing (first attempt)"""
    import urllib.request
    body = {"model": "deepseek/deepseek-v4.1-flash-1017", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": max_tokens, "response_format": {"type": "json_object"}, "provider": {"data_collection": "deny"}}
    if not reason:
        body["reasoning"] = {"enabled": False}
    x = json.loads(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + k}), timeout=300).read())
    return json.loads(re.search(r"\{.*\}", x["choices"][0]["message"]["content"] or "", re.S).group(0))


def seed_for(*a):
    return (SEED * 1000 + zlib.crc32("|".join(map(str, a)).encode())) % (2 ** 31)


# ---------- word lists (DeepSeek only; nothing from the private corpus leaves this machine: Jung's list here is the public 1910 printing)
def words():
    import crosslang
    from wordfreq import top_n_list, zipf_frequency
    OUT.mkdir(exist_ok=True); f = OUT / "words.json"; assert not f.exists(), "words.json exists: the lists are fixed before any run"
    k = crosslang.key(); rng = np.random.default_rng(SEED); Z = lambda w: zipf_frequency(w, "en")  # noqa: E731
    def ask(p, reason=False):
        for t in range(5):
            try:
                return ds(k, p, reason)
            except Exception as e:
                print("retry", str(e)[:120], flush=True); time.sleep(5)
        raise RuntimeError("DeepSeek failed")
    J = list(dict.fromkeys(JUNG)); assert len(J) == 99; jb = {base(w) for w in J}
    S = {}
    for a, desc in FIG.items():
        r = ask(f'Reply with JSON only: {{"words": ["...", ...]}}. List 20 common single English words for the behaviours, feelings and objects that circle {desc}: what a person caught up in this figure does, feels and is surrounded by. Single lowercase words only, no names, not the figure\'s own name.', True)
        S[a] = [x.strip().lower() for x in r["words"] if re.fullmatch(r"[a-z]+", x.strip().lower()) and Z(x.strip().lower()) >= 3.0]
    count = {}; [count.__setitem__(x, count.get(x, 0) + 1) for v in S.values() for x in set(v)]
    S = {a: [x for x in dict.fromkeys(v) if count[x] == 1 and x not in jb][:12] for a, v in S.items()}
    assert all(len(v) >= 8 for v in S.values()), {a: len(v) for a, v in S.items()}
    used = jb | {x for v in S.values() for x in v}
    pool = [w for w in top_n_list("en", 120000) if re.fullmatch(r"[a-z]{3,}", w) and w not in used]; zp = np.array([Z(w) for w in pool]); cands = {}
    for w in J:
        ok = np.where(np.abs(zp - Z(base(w))) <= 0.2)[0]; ok = ok if len(ok) >= 40 else np.argsort(np.abs(zp - Z(base(w))))[:40]; cands[w] = [pool[i] for i in rng.choice(ok, size=40, replace=False)]    # rare words (deportment) have few neighbours within 0.2: then the 40 nearest
    uniq = sorted({x for v in cands.values() for x in v} | {base(w) for w in J}); tag = {}
    for i in range(0, len(uniq), 50):                                         # tagging apart from choosing: the choice below is random among same-tagged words, so no writer prefers a meaning (first attempt: asked to pick, it mismatched parts of speech)
        tag.update(ask('Tag each word with its most common part of speech in English: noun, verb, adjective or other. Use other for proper nouns, abbreviations, foreign words, misspellings, function words, adverbs, and any inflected form (plurals, -ing, -ed, -s verb forms). Reply with JSON only: {"tags": {"word": "tag", ...}}.\n\n' + json.dumps(uniq[i:i + 50]))["tags"])
    want = {w: "verb" if w.startswith("to ") else tag.get(base(w), "other") for w in J}; print("J tags", {t: sum(v == t for v in want.values()) for t in ("noun", "verb", "adjective", "other")}, [w for w in J if want[w] == "other"], flush=True); C = {}; taken = set()
    for w in J:
        t = want[w] if want[w] != "other" else "noun"; opts = [x for x in cands[w] if tag.get(x) == t and x not in taken]; assert opts, ("no candidate", w, t); C[w] = str(rng.choice(opts)); taken.add(C[w])
    assert len(C) == 99, 99 - len(C)
    Cw = ["to " + C[w] if w.startswith("to ") else C[w] for w in J]; allw = J + Cw + [x for v in S.values() for x in v]; R = {}
    for i in range(0, len(allw), 30):
        chunk = allw[i:i + 30]
        r = ask('Rate each English word. arousal: how emotionally stirring it is, 1 calm to 9 stirring. valence: 1 unpleasant to 9 pleasant. concreteness: 1 abstract to 5 concrete. Reply with JSON only: {"ratings": {"word": [arousal, valence, concreteness], ...}}.\n\n' + json.dumps(chunk))["ratings"]
        R.update({w: r[w] for w in chunk if w in r and len(r[w]) == 3})
    assert len(R) == len(allw), sorted(set(allw) - set(R))[:10]
    json.dump({"J": J, "C": Cw, "S": S, "zipf": {w: Z(base(w)) for w in allw}, "rate": R}, open(f, "w"), indent=0)
    (OUT / "vocab.txt").write_text("\n".join(w for w in top_n_list("en", 150000) if re.fullmatch(r"[a-z]+(?:[-'][a-z]+)?", w)))
    gap = np.mean([Z(base(a)) - Z(base(b)) for a, b in zip(J, Cw)]); print("words", len(J), len(Cw), {a: len(v) for a, v in S.items()}, "mean zipf gap J-C", round(float(gap), 3)); print("control sample", Cw[:15])


# ---------- the model
class M:
    """a loaded model with a counting patch on its decoder layers; nothing is added unless ST['add'] is set"""
    def __init__(self, name, path):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx; self.name = name; self.model, self.tok = load(os.path.expanduser(path)); self.nl = len(self.model.layers); self.cap_at = LAYER if name == "e4b" else int(round(LAYER / 41 * (self.nl - 1)))
        ST = {"add": None, "start": 0, "n": 0, "at": LAYER, "cap": False, "h": None}; self.ST = ST; cls = type(self.model.layers[0]); orig = cls.__call__; me = self

        def rec(self_, *a, **k):
            o = orig(self_, *a, **k); i = ST["n"]; ST["n"] += 1; h = o[0] if isinstance(o, tuple) else o
            if ST["add"] is not None and i == ST["at"]:
                s = ST["start"]; h = mx.concatenate([h[:, :s], h[:, s:] + ST["add"].astype(h.dtype)], axis=1) if s > 0 else h + ST["add"].astype(h.dtype)
                o = (h,) + tuple(o[1:]) if isinstance(o, tuple) else h; ST["hits"] = ST.get("hits", 0) + 1
            if ST["cap"] and i == me.cap_at:
                ST["h"] = h[0]
            return o
        cls.__call__ = rec
        t = self.tok; self.stop = set(getattr(t, "eos_token_ids", None) or []) | ({t.eos_token_id} if t.eos_token_id is not None else set())
        for x in ("<turn|>", "<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>", "<|end|>", "<|return|>"):
            i = t.convert_tokens_to_ids(x)
            if isinstance(i, int) and i >= 0 and i != getattr(t, "unk_token_id", None):
                self.stop.add(i)
        self.check_cache()

    def ids(self, msgs, think=False):
        return self.tok.encode(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think), add_special_tokens=False)

    def header(self, msgs):
        """how many tokens the model's turn header adds; A2 steers from there, since the plugin's rule (generated tokens only) would leave a one-word reply's first token unsteered"""
        a = self.ids(msgs); b = self.tok.encode(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False), add_special_tokens=False); n = 0
        while n < len(b) and n < len(a) and a[n] == b[n]:
            n += 1
        k = len(a) - n; assert 1 <= k <= 12, ("header size", k); return k

    def forward(self, x, cache, add=None, start=0):
        self.ST.update(add=add, start=start, n=0, hits=0); o = self.model(x, cache=cache); assert self.ST["n"] == self.nl, (self.ST["n"], self.nl); assert add is None or self.ST["hits"] == 1, "steering did not fire"
        self.ST["add"] = None; return o[:, -1, :].astype(self.mx.float32)

    def expand(self, cache, N):
        for c in cache:
            assert hasattr(c, "keys") and hasattr(c, "values"), type(c)
            c.keys = self.mx.repeat(c.keys, N, axis=0); c.values = self.mx.repeat(c.values, N, axis=0)

    def check_cache(self):
        """expanding one prompt's cache to a batch must give the same next-token logits as running the batch whole"""
        from mlx_lm.models.cache import make_prompt_cache
        mx = self.mx; ids = self.ids([{"role": "user", "content": P1.format(w="water")}]); c = make_prompt_cache(self.model); self.can_expand = all(hasattr(q, "keys") and hasattr(q, "values") for q in c)
        if not self.can_expand:                                               # Nemotron's hybrid layers keep other state: run() then sends the whole prompt as a batch, which is slower and needs no check
            print(self.name, "whole-batch sampling, cache", {type(q).__name__ for q in c}, "| stop ids", sorted(self.stop)[:6], flush=True); return
        lg = self.forward(mx.array([ids]), c); y = int(mx.argmax(lg, axis=-1).item()); self.expand(c, 3); a = self.forward(mx.array([[y]] * 3), c); b = self.forward(mx.array([ids + [y]] * 3), make_prompt_cache(self.model)); e = self.forward(mx.array([ids + [y]]), make_prompt_cache(self.model))
        soft = lambda v: (lambda P: P / P.sum())(np.array(mx.softmax(v[0] / 5, -1), dtype=np.float64))  # noqa: E731
        gap = js(soft(a), soft(b)); noise = js(soft(e), soft(b)); assert gap < 1e-3, ("cache expansion differs", gap, noise)    # at temperature 5, so that a peaked next token cannot hide a difference; rounding on these weights moves single logits by up to 0.5 even between batch sizes with no cache
        print(self.name, "cache ok, JS", round(gap, 7), "rounding", round(noise, 7), "| stop ids", sorted(self.stop)[:6], flush=True)

    def run(self, ids, N, max_new=12, add=None, k=0, T=1.0, specials=False):
        """-> full first-token distribution (numpy), [token id lists]; each reply is drawn at temperature T; steering, if any, starts k tokens before the end of the prompt and stays on"""
        from mlx_lm.models.cache import make_prompt_cache
        mx = self.mx; c = make_prompt_cache(self.model); whole = N > 1 and not self.can_expand; lg = self.forward(mx.array([ids] * N if whole else [ids]), c, add=add, start=len(ids) - k if add is not None else 0); P = np.array(mx.softmax(lg[0], axis=-1), dtype=np.float64); P = P / P.sum()
        if N == 0:
            return P, []
        y = mx.random.categorical(lg[:1] / T, num_samples=N)[0]; mx.eval(y); self.expand(c, N) if not whole and self.can_expand else None; out = [[int(t)] for t in np.array(y)]; done = [o[0] in self.stop for o in out]
        for _ in range(max_new - 1):
            if all(done):
                break
            lg = self.forward(y[:, None], c, add=add, start=0); y = mx.random.categorical(lg / T); mx.eval(y); ys = np.array(y)
            for i, t in enumerate(ys):
                if not done[i]:
                    out[i].append(int(t)); done[i] = int(t) in self.stop
        return P, out

    def text(self, toks):
        return self.tok.decode([t for t in toks if t not in self.stop], skip_special_tokens=True)

    def top(self, P, n=30):
        i = np.argpartition(-P, min(n, len(P) - 1))[:n]; i = i[np.argsort(-P[i])]; return [[self.tok.decode([int(t)]), round(float(P[t]), 5)] for t in i]


def entropy(P):
    p = P[P > 1e-12]; return float(-(p * np.log(p)).sum())


def js(P, Q):
    m = (P + Q) / 2; f = lambda a: float((a[a > 1e-12] * np.log2(a[a > 1e-12] / m[a > 1e-12])).sum())  # noqa: E731
    return max(0.5 * f(P) + 0.5 * f(Q), 0.0)


def rec_of(m, P, toks, w):
    R = {}
    for t in toks:
        s = norm(m.text(t)); R[s] = R.get(s, 0) + 1
    n = max(len(toks), 1); return {"h1": round(entropy(P), 5), "top": m.top(P), "replies": R, "fail": round(sum(v for s, v in R.items() if not is_one(s)) / n, 4), "echo": round(sum(v for s, v in R.items() if s == base(w)) / n, 4)}


def rows(f):
    """the file's whole lines; a kill can leave an unterminated last line, which is dropped here (review)"""
    t = Path(f).read_text(); return [json.loads(l) for l in t.split("\n")[:-1] if l.strip()]


def done_keys(f, *ks):
    """what is finished; also cuts an unterminated tail, so that the next append starts on its own line"""
    if not f.exists():
        return set()
    t = f.read_text()
    if t and not t.endswith("\n"):
        f.write_text(t[:t.rfind("\n") + 1])
    return {tuple(x[k] for k in ks) for x in rows(f)}


# ---------- A1
def fresh(name, path, only=None):
    OUT.mkdir(exist_ok=True); W = wlist(); m = M(name, path); mx = m.mx; f = OUT / f"fresh_{name}.jsonl"; seen = done_keys(f, "prompt", "word"); sets = {"J": W["J"], "C": W["C"], "S": [x for v in W["S"].values() for x in v]}
    if only:
        sets = {k: v[:only] for k, v in sets.items()}
    todo = [("P1", P1, s, w, 40) for s in sets for w in sets[s]] + [("P2", P2, s, w, 20) for s in ("J", "C") for w in sets[s]]; t0 = time.time(); fo = open(f, "a")
    for i, (pn, pr, s, w, N) in enumerate(todo):
        if (pn, w) in seen:
            continue
        mx.random.seed(seed_for(name, pn, w)); P, toks = m.run(m.ids([{"role": "user", "content": pr.format(w=w)}]), N)
        fo.write(json.dumps({"prompt": pn, "set": s, "word": w, "n": N, **rec_of(m, P, toks, w)}, ensure_ascii=False) + "\n"); fo.flush()
        if i % 50 == 0:
            mx.clear_cache(); print(name, i, "of", len(todo), round(time.time() - t0), "s", w, "->", sorted(json.loads(json.dumps(rec_of(m, P, toks, w)))["replies"].items(), key=lambda a: -a[1])[:3], flush=True)
    g = OUT / f"states_{name}.npz"
    if not g.exists() and not only:                                           # a smoke run saves no states (review)
        Hw, Hl, K = [], [], []; empty = m.ids([{"role": "user", "content": P1.format(w="")}])
        for s in sets:
            for w in sets[s]:
                ids = m.ids([{"role": "user", "content": P1.format(w=w)}]); p = 0
                while p < min(len(ids), len(empty)) and ids[p] == empty[p]:
                    p += 1
                q = 0
                while q < min(len(ids), len(empty)) - p and ids[-1 - q] == empty[-1 - q]:
                    q += 1
                assert len(ids) - q > p, w
                m.ST["cap"] = True; m.forward(mx.array([ids]), None); m.ST["cap"] = False; h = m.ST["h"]; a = h[p:len(ids) - q].astype(mx.float32).mean(0); b = h[-1].astype(mx.float32); mx.eval(a, b); Hw.append(np.array(a)); Hl.append(np.array(b)); K.append(w)
        np.savez_compressed(g, Hw=np.array(Hw), Hl=np.array(Hl), words=np.array(K), layer=m.cap_at)
    print(name, "FRESH_DONE", round(time.time() - t0), "s", flush=True)


def orders(W, n=4):
    r = np.random.default_rng(SEED); return [list(r.permutation(W["J"])) for _ in range(n)]


def seq(name, path, n_orders=4, limit=None):
    W = wlist(); m = M(name, path); mx = m.mx; f = OUT / f"seq_{name}.jsonl"; seen = {o for o, in done_keys(f, "order")} if f.exists() else set(); t0 = time.time()
    if seen and any(json.loads(l)["pos"] == 98 for l in open(f)):
        rows = list(rows(f)); full = {o for o in seen if sum(r["order"] == o for r in rows) == 99}; rows = [r for r in rows if r["order"] in full]; open(f, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); seen = full     # an unfinished order restarts
    else:
        seen = set(); open(f, "w").close()
    fo = open(f, "a")
    for o, order in enumerate(orders(W, n_orders)):
        if o in seen:
            continue
        msgs = []; mx.random.seed(seed_for(name, "seq", o)); prev = None
        for pos, w in enumerate(order[:limit] if limit else order):
            msgs.append({"role": "user", "content": P1.format(w=w) if pos == 0 else f"Word: {w}"}); ids = m.ids(msgs); P, toks = m.run(ids, 1); raw = m.text(toks[0]).strip(); r = norm(raw)
            fo.write(json.dumps({"order": o, "pos": pos, "word": w, "prev": prev, "h1": round(entropy(P), 5), "top": m.top(P), "first": m.tok.decode([toks[0][0]]), "reply": r, "fail": not is_one(r), "len": len(ids)}, ensure_ascii=False) + "\n"); fo.flush()
            msgs.append({"role": "assistant", "content": raw or "-"}); prev = w
        mx.clear_cache(); print(name, "order", o, "done", round(time.time() - t0), "s", flush=True)
    print(name, "SEQ_DONE", round(time.time() - t0), "s", flush=True)


def think(name, path, M_=6, max_new=300, only=None):
    if name not in THINK_END:
        print(name, "does not think"); return
    W = wlist(); m = M(name, path); mx = m.mx; f = OUT / f"think_{name}.jsonl"; seen = done_keys(f, "word"); tok = m.tok; end = tok.convert_tokens_to_ids(THINK_END[name]); assert isinstance(end, int) and end >= 0 and end != getattr(tok, "unk_token_id", None), ("thinking end", end); t0 = time.time(); fo = open(f, "a"); todo = W["J"] + W["C"]
    for i, w in enumerate(todo[:only] if only else todo):
        if w in seen:
            continue
        mx.random.seed(seed_for(name, "think", w)); P, toks = m.run(m.ids([{"role": "user", "content": P1.format(w=w)}], think=True), M_, max_new=max_new); out = []
        for t in toks:
            e = t.index(end) if end in t else None; closed = t[-1] in m.stop; body = t[e + 1:] if e is not None else t
            out.append({"think": e if e is not None else (0 if closed else len(t)), "ended": e is not None or closed, "reply": norm(m.text(body))})
        fo.write(json.dumps({"word": w, "h1": round(entropy(P), 5), "samples": out}, ensure_ascii=False) + "\n"); fo.flush()
        if i % 25 == 0:
            mx.clear_cache(); print(name, i, round(time.time() - t0), "s", w, [(x["think"], x["reply"]) for x in out][:3], flush=True)
    print(name, "THINK_DONE", round(time.time() - t0), "s", flush=True)


# ---------- A2, on E4B, whose directions these are
def vec(arm):
    z = np.load(HERE / "consteer" / "vecs" / f"{arm}.npz"); assert int(z["layer"]) == LAYER; v = z["v"].astype(np.float32); assert abs(float(np.linalg.norm(v)) - 1) < 1e-3; return v


E4B = "./models/gemma4-e4b-mlx-bf16"


def dose(path=E4B, N=20):
    """the largest push that keeps replies real single words: validity at least 0.9 of plain's. Chosen per arm, on 24 words that the test also uses, before any figure is scored"""
    W = wlist(); vocab = set((OUT / "vocab.txt").read_text().split()); m = M("e4b", path); mx = m.mx; r = np.random.default_rng(SEED); sub = list(r.choice(W["J"], 12, replace=False)) + list(r.choice(W["C"], 12, replace=False)); res = {}; base_P = {}
    def valid(toks):
        s = [norm(m.text(t)) for t in toks]; return sum(is_one(x) and all(y in vocab for y in re.split(r"[-']", x)) for x in s) / len(s)
    v0 = []
    for w in sub:
        msgs = [{"role": "user", "content": P1.format(w=w)}]; mx.random.seed(seed_for("dose", w)); P, t = m.run(m.ids(msgs), N); base_P[w] = P; v0.append(valid(t))
    plain = float(np.mean(v0)); print("plain validity", round(plain, 3), flush=True)
    k = m.header([{"role": "user", "content": P1.format(w="x")}])
    for arm in ARMS:
        a = mx.array(vec(arm)); res[arm] = {}
        for push in PUSHES:
            vs, jj = [], []
            for w in sub:
                mx.random.seed(seed_for("dose", w)); P, t = m.run(m.ids([{"role": "user", "content": P1.format(w=w)}]), N, add=a * push, k=k); vs.append(valid(t)); jj.append(js(base_P[w], P))
            res[arm][push] = {"valid": round(float(np.mean(vs)), 4), "js": round(float(np.mean(jj)), 4)}
        ok = [p for p in PUSHES if res[arm][p]["valid"] >= 0.9 * plain]; res[arm]["push"] = max(ok) if ok else None; mx.clear_cache(); print(arm, res[arm], flush=True)
    res["plain"] = plain; res["k"] = k; json.dump(res, open(OUT / "dose.json", "w"), indent=0); print("DOSE_DONE", flush=True)


def implant(path=E4B, N=20):
    """every word under every arm, two seeds; first-token JS against plain is exact and needs no seeds, the replies are for the fingerprint, the drift and the judge"""
    W = wlist(); D = json.load(open(OUT / "dose.json")); m = M("e4b", path); mx = m.mx; (OUT / "implant").mkdir(exist_ok=True); k = D["k"]; todo = W["J"] + W["C"] + [x for v in W["S"].values() for x in v]; P0 = {}; t0 = time.time()
    for w in todo:
        mx.random.seed(seed_for("plain0", w)); P0[w], _ = m.run(m.ids([{"role": "user", "content": P1.format(w=w)}]), 0)
    for arm in ["plain"] + [a for a in ARMS if D[a]["push"] is not None]:
        add = None if arm == "plain" else mx.array(vec(arm)) * D[arm]["push"]
        for s in (0, 1):
            f = OUT / "implant" / f"{arm}_{s}.jsonl"; seen = done_keys(f, "word"); fo = open(f, "a")
            for w in todo:
                if w in seen:
                    continue
                mx.random.seed(seed_for("implant", s, w)); P, t = m.run(m.ids([{"role": "user", "content": P1.format(w=w)}]), N, add=add, k=k if add is not None else 0); x = rec_of(m, P, t, w); x.update(word=w, js=round(js(P0[w], P), 5)); fo.write(json.dumps(x, ensure_ascii=False) + "\n"); fo.flush()
            mx.clear_cache(); print(arm, s, "done", round(time.time() - t0), "s", flush=True)
    print("IMPLANT_DONE", round(time.time() - t0), "s", flush=True)


def judge(n=5, top=30):
    """DeepSeek, blind: arms under random letters, the words whose replies moved most, as stimulus, plain's replies, the arm's replies. It picks a figure or none. Words only; no corpus text"""
    import crosslang
    k = crosslang.key(); rng = np.random.default_rng(SEED); D = {}; f = OUT / "judge.jsonl"; seen = done_keys(f, "arm", "i"); fo = open(f, "a")
    load = lambda a: {x["word"]: x for s in (0, 1) for x in rows(OUT / "implant" / f"{a}_{s}.jsonl")}  # noqa: E731
    pl = load("plain"); names = {a: FIG[a] for a in FIG}
    for arm in list(FIG) + CTRL:
        g = OUT / "implant" / f"{arm}_0.jsonl"
        if not g.exists():
            continue
        X = load(arm); rank = sorted(X, key=lambda w: -X[w]["js"])[:top]; lines = []
        for w in rng.permutation(rank):
            tp = lambda d: ", ".join(a for a, _ in sorted(d.items(), key=lambda a: -a[1])[:3])  # noqa: E731
            lines.append(f"{w}: usually [{tp(pl[w]['replies'])}]; now [{tp(X[w]['replies'])}]")
        for i in range(n):
            if (arm, i) in seen:
                continue
            opts = [names[a] for a in rng.permutation(list(names))] + ["none of these"]
            p = "A language model answered a word-association test twice. Something was changed inside it before the second time. Here are the words whose answers moved most, with the usual answers and the new ones.\n\n" + "\n".join(lines) + "\n\nWhich one figure below best explains the change? If nothing stands out, say none of these.\n" + "\n".join(f"- {o}" for o in opts) + '\n\nReply with JSON only: {"pick": "the figure, exactly as written", "why": "one short sentence"}'
            for t in range(4):
                try:
                    r = ds(k, p, reason=False); break
                except Exception as e:
                    print("retry", str(e)[:100], flush=True); time.sleep(5)
            else:
                raise RuntimeError("the judge failed: nothing is written for this arm")
            fo.write(json.dumps({"arm": arm, "i": i, "pick": r.get("pick"), "why": r.get("why")}, ensure_ascii=False) + "\n"); fo.flush()
    print("JUDGE_DONE", flush=True)


# ---------- A4
def next_tasks(W):
    r = np.random.default_rng(SEED + 4); return [(t, "math", a) for t, a in zip(TASKS, TASK_ANS)] + [(f"Word: {w}", "word", None) for w in r.choice(W["C"], 8, replace=False)]


def hot_cold(name, W, n=15):
    F = [x for x in rows(OUT / f"fresh_{name}.jsonl") if x["prompt"] == "P1" and x["set"] == "J"]; assert len(F) == 99, (name, len(F)); s = sorted(F, key=lambda x: x["h1"]); return [x["word"] for x in s[-n:]], [x["word"] for x in s[:n]]


def autonomy(name, path, N=10):
    """after a hot or a cold word of this model's own, the same unrelated tasks. A chat is the model's only memory, so anything that carries over goes through the reply's text"""
    W = wlist(); m = M(name, path); mx = m.mx; hot, cold = hot_cold(name, W); f = OUT / f"autonomy_{name}.jsonl"; seen = done_keys(f, "trigger", "task"); fo = open(f, "a"); t0 = time.time()
    for grp, L in (("hot", hot), ("cold", cold)):
        for w in L:
            mx.random.seed(seed_for(name, "auto", w)); first = [{"role": "user", "content": P1.format(w=w)}]; _, t = m.run(m.ids(first), 1); raw = m.text(t[0]).strip()
            for task, kind, ans in next_tasks(W):
                if (w, task) in seen:
                    continue
                mx.random.seed(seed_for(name, "auto", w, task)); P, toks = m.run(m.ids(first + [{"role": "assistant", "content": raw or "-"}, {"role": "user", "content": task}]), N); R = [norm(m.text(x)) for x in toks]
                fo.write(json.dumps({"group": grp, "trigger": w, "first_reply": norm(raw), "task": task, "kind": kind, "h1": round(entropy(P), 5), "right": (sum(x == ans for x in R) / N if ans else None), "replies": R}, ensure_ascii=False) + "\n"); fo.flush()
        mx.clear_cache()
    print(name, "AUTONOMY_DONE", round(time.time() - t0), "s", flush=True)


def autosteer(path=E4B):
    """arms: steering on only for the trigger's reply (carry), and on for the next task as well (the positive control); the JS against plain with the plain reply, on the next task's first token"""
    W = wlist(); D = json.load(open(OUT / "dose.json")); m = M("e4b", path); mx = m.mx; k = D["k"]; r = np.random.default_rng(SEED + 5); trig = list(r.choice(W["J"], 20, replace=False)); tasks = next_tasks(W); f = OUT / "autosteer.jsonl"; seen = done_keys(f, "arm", "trigger", "task"); fo = open(f, "a"); t0 = time.time()
    plainR = {}; plainP = {}
    for w in trig:
        first = [{"role": "user", "content": P1.format(w=w)}]; mx.random.seed(seed_for("auto", w)); _, t = m.run(m.ids(first), 1); plainR[w] = m.text(t[0]).strip()
        for task, _, _ in tasks:
            plainP[w, task], _ = m.run(m.ids(first + [{"role": "assistant", "content": plainR[w] or "-"}, {"role": "user", "content": task}]), 0)
    for arm in [a for a in ARMS if D[a]["push"] is not None]:
        add = mx.array(vec(arm)) * D[arm]["push"]
        for w in trig:
            first = [{"role": "user", "content": P1.format(w=w)}]; mx.random.seed(seed_for("auto", w)); _, t = m.run(m.ids(first), 1, add=add, k=k); raw = m.text(t[0]).strip()
            for task, kind, _ in tasks:
                if (arm, w, task) in seen:
                    continue
                ids = m.ids(first + [{"role": "assistant", "content": raw or "-"}, {"role": "user", "content": task}]); Pc, _ = m.run(ids, 0); Ps, _ = m.run(m.ids(first + [{"role": "assistant", "content": plainR[w] or "-"}, {"role": "user", "content": task}]), 0, add=add, k=k)
                fo.write(json.dumps({"arm": arm, "trigger": w, "task": task, "kind": kind, "reply": norm(raw), "plain_reply": norm(plainR[w]), "same_reply": norm(raw) == norm(plainR[w]), "carry": round(js(plainP[w, task], Pc), 5), "direct": round(js(plainP[w, task], Ps), 5)}, ensure_ascii=False) + "\n"); fo.flush()
        mx.clear_cache(); print(arm, round(time.time() - t0), "s", flush=True)
    print("AUTOSTEER_DONE", round(time.time() - t0), "s", flush=True)


# ---------- report
def load_fresh(prompt="P1"):
    D = {}
    for f in sorted(OUT.glob("fresh_*.jsonl")):
        n = f.stem[6:]; X = {x["word"]: x for x in rows(f) if x["prompt"] == prompt}; W = wlist(); need = W["J"] + W["C"] + ([y for v in W["S"].values() for y in v] if prompt == "P1" else [])
        if n in COMPANY and all(w in X for w in need):                        # a model counts only when this wording is complete (review)
            D[n] = X
    return D


def rep_entropy(R):
    v = np.array(list(R.values()), float); v = v / v.sum(); return float(-(v * np.log(v)).sum())


def dist_js(A, B):
    k = sorted(set(A) | set(B)); a = np.array([A.get(x, 0.0) for x in k], float); b = np.array([B.get(x, 0.0) for x in k], float); return js(a / a.sum(), b / b.sum())


def z(a):
    a = np.asarray(a, float); s = a.std(); return (a - a.mean()) / s if s > 0 else a * 0


def indices(D, words):
    """-> {model: {sign: array over words}}, DI = mean z of the first-token entropy, the reply entropy and unusualness; unusual = JS of the replies against the mean of other companies' models (one model's company has none to compare: no unusual sign)"""
    out = {}
    for n, X in D.items():
        ws = [w for w in words if w in X]; assert len(ws) == len(words), (n, len(ws))
        others = [o for o in D if COMPANY[o] != COMPANY[n]]; un = []
        for w in words:
            pool = {}
            for o in others:
                t = sum(D[o][w]["replies"].values())
                for s, c in D[o][w]["replies"].items():
                    pool[s] = pool.get(s, 0) + c / t
            un.append(dist_js(X[w]["replies"], pool) if pool else 0.0)
        h1 = np.array([X[w]["h1"] for w in words]); hr = np.array([rep_entropy(X[w]["replies"]) for w in words]); un = np.array(un)
        out[n] = {"h1": h1, "hr": hr, "un": un, "DI": (z(h1) + z(hr) + z(un)) / (3 if un.any() else 2), "DI2": (z(h1) + z(hr)) / 2, "fail": np.array([X[w]["fail"] for w in words]), "echo": np.array([X[w]["echo"] for w in words])}
    return out


def residual(y, *xs):
    A = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in xs]); return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def boot(f, n, B=2000, seed=SEED):
    r = np.random.default_rng(seed); v = np.array([f(r.integers(0, n, n)) for _ in range(B)]); return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def report():
    from scipy.stats import rankdata, spearmanr
    W = wlist(); L = []; D = load_fresh(); assert D, "no model has finished"; words = W["J"] + W["C"]; J, C = W["J"], W["C"]; nJ = len(J); order = sorted(D, key=lambda n: list(COMPANY).index(n)); D = {n: D[n] for n in order}; I = indices(D, words); zf = np.array([W["zipf"][w] for w in words]); ln = np.array([len(base(w)) for w in words]); rate = np.array([W["rate"][w] for w in words], float)
    p = lambda *a: (L.append(" ".join(str(x) for x in a)), print(*a))  # noqa: E731
    p("# Word association on models (addendum 89)\n\n## A1. How the models respond\n")
    p("| model | one word | repeats the stimulus | first-token entropy | reply entropy | mean most common reply share | hottest words | coldest words |\n|---|---|---|---|---|---|---|---|")
    for n, S in I.items():
        X = D[n]; o = np.argsort(-S["DI"]); top = np.mean([max(X[w]["replies"].values()) / sum(X[w]["replies"].values()) for w in words])
        p(f"| {n} | {1 - S['fail'].mean():.3f} | {S['echo'].mean():.3f} | {S['h1'].mean():.2f} | {S['hr'].mean():.2f} | {top:.2f} | {', '.join(words[i] for i in o[:8])} | {', '.join(words[i] for i in o[-6:])} |")
    show = ["to sin", "to die", "to pray", "money", "to marry", "woman", "child", "anxiety", "to kiss", "dead", "pride", "to abuse"]
    p("\nMost common reply per model, for some of Jung's charged words:\n\n| model | " + " | ".join(show) + " |\n|---|" + "---|" * len(show))
    for n in D:
        p(f"| {n} | " + " | ".join(max(D[n][w]["replies"], key=D[n][w]["replies"].get) or "(empty)" for w in show) + " |")
    R = np.column_stack([residual(rankdata(I[n]["DI"]), zf, ln) for n in D]); names = list(D); pairs = [(a, b) for a in range(len(names)) for b in range(a + 1, len(names)) if COMPANY[names[a]] != COMPANY[names[b]]]
    if pairs:
        cc = lambda idx: float(np.mean([np.corrcoef(R[idx, a], R[idx, b])[0, 1] for a, b in pairs]))  # noqa: E731
        p(f"\nDisturbance shared across companies (mean over {len(pairs)} pairs of models, after word frequency and length): {cc(np.arange(len(words))):+.3f} {boot(cc, len(words), 500)}")
        raw = np.mean([spearmanr(I[names[a]]["h1"], I[names[b]]["h1"])[0] for a, b in pairs]); p(f"First-token entropy alone, raw Spearman: {raw:+.3f}")
        for sg in ("h1", "hr", "un", "DI2"):                                  # review: words on which models disagree raise "un" for every model, so the index is partly shared by construction
            Rs = np.column_stack([residual(rankdata(I[n][sg]), zf, ln) for n in D]); p(f"  shared, after frequency and length, {dict(h1='first-token entropy', hr='reply entropy', un='unusual (shared by construction)', DI2='index without unusual')[sg]}: {np.mean([np.corrcoef(Rs[:, a], Rs[:, b])[0, 1] for a, b in pairs]):+.3f}")
    mean_di = np.mean([I[n]["DI"] for n in D], 0); dj = mean_di[:nJ] - mean_di[nJ:]; f = lambda i: float(dj[i].mean())  # noqa: E731
    p(f"\nJung's list against words matched on frequency (J minus C, disturbance in sd, pooled, {nJ} pairs): {f(np.arange(nJ)):+.3f} {boot(f, nJ)}; zipf gap {np.mean(zf[:nJ] - zf[nJ:]):+.3f}")
    p("\n| model | J minus C |\n|---|---|")
    for n in D:
        d = I[n]["DI"][:nJ] - I[n]["DI"][nJ:]; p(f"| {n} | {d.mean():+.3f} {tuple(round(x, 3) for x in boot(lambda i: float(d[i].mean()), nJ))} |")
    for lab, c in (("arousal", 0), ("valence", 1), ("concreteness", 2)):
        others = [rate[:, j] for j in range(3) if j != c]; y = residual(rankdata(mean_di), zf, ln, *others); x = residual(rankdata(rate[:, c]), zf, ln, *others); g = lambda i: float(np.corrcoef(x[i], y[i])[0, 1])  # noqa: E731
        p(f"Disturbance with {lab} (DeepSeek's ratings; after frequency, length and the other two): {g(np.arange(len(words))):+.3f} {boot(g, len(words))}")
    D2 = load_fresh("P2"); both = [n for n in D if n in D2]
    if both:
        p("\nThe same word's first-token entropy under the two instruction wordings, Spearman: " + ", ".join(f"{n} {spearmanr([D[n][w]['h1'] for w in words], [D2[n][w]['h1'] for w in words])[0]:+.2f}" for n in both))
    # thinking
    T = {f.stem[6:]: {x["word"]: x for x in rows(f)} for f in sorted(OUT.glob("think_*.jsonl"))}
    if T:
        p("\n## Thinking length\n\n| model | words | median thinking tokens | not finished | J minus C (sd), matched | with first-token entropy (no-thinking run) |\n|---|---|---|---|---|---|")
        for n, X in T.items():
            ok = [w for w in words if w in X]
            if len(ok) < len(words) or n not in D:
                p(f"| {n} | {len(ok)} | | | | |"); continue
            v = np.array([np.mean([s["think"] for s in X[w]["samples"]]) for w in words]); u = np.mean([not s["ended"] for w in words for s in X[w]["samples"]]); zv = z(v); d = zv[:nJ] - zv[nJ:]
            p(f"| {n} | {len(ok)} | {np.median(v):.0f} | {u:.3f} | {d.mean():+.3f} {tuple(round(x, 3) for x in boot(lambda i: float(d[i].mean()), nJ))} | {spearmanr(v, I[n]['h1'])[0]:+.2f} |")
    # carry-over
    S = {f.stem[4:]: list(rows(f)) for f in sorted(OUT.glob("seq_*.jsonl"))}
    if S:
        p("\n## Carry-over, Jung's list as one chat (4 random orders)\n\nHot previous word: the correlation of how disturbed the previous word was (fresh chat) with this word's first-token entropy against its own mean over orders. Echo: how much more likely the previous reply's first token is here than at the same word in other orders.\n\n| model | turns | one word | same reply as earlier in chat | entropy, chain minus fresh | hot previous word | shuffle p | echo (log) |\n|---|---|---|---|---|---|---|---|")
        rng = np.random.default_rng(SEED)
        for n, X in S.items():
            full = [o for o in {r["order"] for r in X} if sum(r["order"] == o for r in X) == len(J)]; X = [r for r in X if r["order"] in full]
            if n not in D or len(full) < 2:                                   # one order cannot be centred on its own mean (review: it printed a false p)
                p(f"| {n} | {len(X)} | n/a, fewer than 2 complete orders | | | | | |"); continue
            di = dict(zip(words, I[n]["DI"])); by = {}
            for r in X:
                by.setdefault(r["word"], []).append(r)
            xs, ys, prev_w = [], [], []
            for w, rs in by.items():
                m_ = np.mean([r["h1"] for r in rs])
                for r in rs:
                    if r["prev"] is not None:
                        xs.append(di[r["prev"]]); ys.append(r["h1"] - m_); prev_w.append(r["prev"])
            xs, ys, prev_w = np.array(xs), np.array(ys), np.array(prev_w); rho = spearmanr(xs, ys)[0]; uw = sorted(set(prev_w)); nul = []
            for _ in range(500):
                sh = dict(zip(uw, rng.permutation([di[u] for u in uw]))); nul.append(spearmanr([sh[u] for u in prev_w], ys)[0])
            pv = float((np.sum(np.abs(nul) >= abs(rho)) + 1) / 501); ech = []; seen_r = {}; rep = []
            for r in X:
                seen_r.setdefault(r["order"], [])
                rep.append(r["reply"] in seen_r[r["order"]]); seen_r[r["order"]].append(r["reply"])
            tk = lambda s: s.strip().lower()  # noqa: E731
            prev_first = {(r["order"], r["pos"]): tk(r["first"]) for r in X}
            for r in X:
                if r["pos"] == 0:
                    continue
                t = prev_first[(r["order"], r["pos"] - 1)]; g = lambda q: max(sum(pp for tt, pp in q["top"] if tk(tt) == t), 1e-6)  # noqa: E731
                oth = [g(q) for q in by[r["word"]] if q["order"] != r["order"]]
                if oth:
                    ech.append(np.log(g(r)) - np.mean(np.log(oth)))
            p(f"| {n} | {len(X)} | {1 - np.mean([r['fail'] for r in X]):.3f} | {np.mean(rep):.3f} | {np.mean([r['h1'] - D[n][r['word']]['h1'] for r in X]):+.2f} | {rho:+.3f} | {pv:.3f} | {np.mean(ech):+.2f} |")
    # A3
    p("\n## A3. Natural clusters\n")
    if "e4b" in D and (OUT / "states_e4b.npz").exists():
        Z = np.load(OUT / "states_e4b.npz"); K = list(Z["words"].astype(str)); H = Z["Hw"].astype(np.float64); ix = [K.index(w) for w in words]; H = H[ix] - H[ix].mean(0); H = H / np.linalg.norm(H, axis=1, keepdims=True); G = H @ H.T; np.fill_diagonal(G, -9); nn = np.argsort(-G, axis=1)[:, :8]; rng = np.random.default_rng(SEED)
        moran = lambda y: float(np.mean((y[:, None] * y[nn]).sum(1) / 8))  # noqa: E731
        p("Do semantically near words share their disturbance (E4B's word states, 8 nearest; the index is the mean product of z-scores; shuffle over words)?\n\n| model | index | shuffle p |\n|---|---|---|")
        for n in D:
            y = z(residual(rankdata(I[n]["DI"]), zf, ln)); o = moran(y); nul = [moran(rng.permutation(y)) for _ in range(2000)]; p(f"| {n} | {o:+.3f} | {(np.sum(np.array(nul) >= o) + 1) / 2001:.4f} |")
        V = {f.stem: np.load(f)["v"].astype(np.float64) for f in (HERE / "consteer" / "vecs").glob("*.npz")}; lib = [a for a in V if a.startswith(("LIB_", "JUNG_")) and not a.startswith("LIBCTRL")]; ctr = ["LIBCTRL_0", "LIBCTRL_1", "LIBCTRL_2", "CONTROL"]
        for lab, y0 in (("E4B's own disturbance", I["e4b"]["DI"]), ("all models' mean disturbance", mean_di)):
            y = residual(rankdata(y0), zf, ln); Hr = Z["Hw"].astype(np.float64)[ix]; Hr = Hr - Hr.mean(0); cs = {a: float(np.corrcoef(Hr @ V[a], y)[0, 1]) for a in lib + ctr}; top = sorted(lib, key=lambda a: -abs(cs[a]))[:6]; M_ = Hr @ np.array([V[a] for a in lib]).T; M_ = (M_ - M_.mean(0)) / M_.std(0); yz = z(y); obs = np.max(np.abs(M_.T @ yz / len(yz))); nul = [np.max(np.abs(M_.T @ rng.permutation(yz) / len(yz))) for _ in range(2000)]
            p(f"\n{lab} against the word's position on each of {len(lib)} named one-against-the-rest directions: largest |r| {obs:.3f}, word shuffle p {(np.sum(np.array(nul) >= obs) + 1) / 2001:.4f}; top {[(a, round(cs[a], 3)) for a in top]}; random-label and setting controls {[(a, round(cs[a], 3)) for a in ctr]}")
    # A2
    if (OUT / "dose.json").exists() and (OUT / "implant").exists():
        a2(W, p)
    A = {f.stem[9:]: list(rows(f)) for f in sorted(OUT.glob("autonomy_*.jsonl"))}
    if A:
        p("\n## A4. After a hot or a cold word, the same unrelated tasks\n\n| model | hot minus cold, next task's first-token entropy | hot minus cold, arithmetic right |\n|---|---|---|")
        for n, X in A.items():
            by = {}
            for r in X:
                by.setdefault(r["group"], {}).setdefault(r["trigger"], []).append(r)
            if len(by) < 2:
                continue
            tm = {g: np.array([np.mean([r["h1"] for r in v]) for v in by[g].values()]) for g in by}; am = {g: np.array([np.mean([r["right"] for r in v if r["kind"] == "math"]) for v in by[g].values()]) for g in by}; rb = np.random.default_rng(SEED)   # the triggers are the unit (review: resampling tasks gave false intervals)
            bs = sorted(tm["hot"][rb.integers(0, len(tm["hot"]), len(tm["hot"]))].mean() - tm["cold"][rb.integers(0, len(tm["cold"]), len(tm["cold"]))].mean() for _ in range(5000))
            p(f"| {n} | {tm['hot'].mean() - tm['cold'].mean():+.3f} ({bs[125]:+.3f}, {bs[4875]:+.3f}) | {am['hot'].mean() - am['cold'].mean():+.3f} |")
    if (OUT / "autosteer.jsonl").exists():
        X = list(rows(OUT / "autosteer.jsonl")); p("\n## A4. Steering only on the trigger's reply, then an unrelated task (E4B)\n\n| arm | reply changed | next task, JS: carried by the steered reply, task unsteered | task steered, plain reply | share carried |\n|---|---|---|---|---|")
        for arm in dict.fromkeys(r["arm"] for r in X):
            R_ = [r for r in X if r["arm"] == arm]; c = np.mean([r["carry"] for r in R_]); dd = np.mean([r["direct"] for r in R_]); p(f"| {arm} | {1 - np.mean([r['same_reply'] for r in R_]):.2f} | {c:.4f} | {dd:.4f} | {c / dd if dd else 0:.2f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


def a2(W, p):
    D = json.load(open(OUT / "dose.json")); need = len(W["J"]) + len(W["C"]) + sum(len(v) for v in W["S"].values()); whole = lambda a: all((OUT / "implant" / f"{a}_{s}.jsonl").exists() and len(rows(OUT / "implant" / f"{a}_{s}.jsonl")) == need for s in (0, 1)); arms = [a for a in ARMS if whole(a)]    # an arm counts only when both seeds are complete (review)
    if not whole("plain") or not arms:
        return
    X = {a: [{x["word"]: x for x in rows(OUT / "implant" / f"{a}_{s}.jsonl")} for s in (0, 1)] for a in arms + ["plain"]}; figs = [a for a in FIG if a in arms]; Sw = {a: W["S"][a] for a in figs}; allS = [x for a in figs for x in Sw[a]]; rng = np.random.default_rng(SEED); J = W["J"] + W["C"]; everyw = J + allS
    p("\n## A2. An implanted complex (E4B, steering from the model's turn on)\n\n| arm | push | one word | mean JS, first token | same reply as plain (replies) |\n|---|---|---|---|---|")
    for a in arms:
        x = X[a][0]; p(f"| {a} | {D[a]['push']} | {1 - np.mean([x[w]['fail'] for w in everyw]):.3f} | {np.mean([x[w]['js'] for w in everyw]):.3f} | {1 - np.mean([dist_js(x[w]['replies'], X['plain'][1][w]['replies']) for w in everyw]):.3f} |")
    def two_way(M_):
        return M_ - M_.mean(0, keepdims=True) - M_.mean(1, keepdims=True) + M_.mean()
    A_ = np.array([[np.log(X[a][0][w]["js"] + 1e-5) for w in allS] for a in figs]); R = two_way(A_); idx = {a: [allS.index(w) for w in Sw[a]] for a in figs}
    def stat(R_, perm):
        return np.array([R_[i, idx[figs[perm[i]]]].mean() for i in range(len(figs))])
    o = stat(R, np.arange(len(figs))); nul = [stat(R, rng.permutation(len(figs))).mean() for _ in range(10000)]; pv = (np.sum(np.array(nul) >= o.mean()) + 1) / 10001; lo, hi = boot(lambda i: float(o[i].mean()), len(o))
    p(f"\nDoes a figure disturb its own satellite words most? Own minus expected, log JS, after each implant's strength and each word's sensitivity: {o.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], label shuffle p {pv:.4f}. Per figure: " + ", ".join(f"{a} {v:+.2f}" for a, v in zip(figs, o)))
    def share(a, L_):
        c = t = 0
        for s in (0, 1):
            for w in J:
                for r, v in X[a][s][w]["replies"].items():
                    t += v; c += v * (r in L_)
        return c / t
    Sh = np.array([[share(a, set(Sw[b])) for b in figs] for a in figs]); P0 = np.array([share("plain", set(Sw[b])) for b in figs]); Dr = two_way(Sh - P0[None, :]); dg = np.diag(Dr); nul = [np.mean(Dr[np.arange(len(figs)), rng.permutation(len(figs))]) for _ in range(10000)]; pv = (np.sum(np.array(nul) >= dg.mean()) + 1) / 10001
    p(f"Do replies to other words drift toward the figure's own satellites? Own minus expected share: {dg.mean():+.4f}, label shuffle p {pv:.4f}. Own share against plain: " + ", ".join(f"{a} {Sh[i, i]:.3f} vs {P0[i]:.3f}" for i, a in enumerate(figs)))
    every = arms + ["plain"]
    def ll(r, q):
        t = sum(q.values()); n_ = len(q) + 1; return sum(v * np.log((q.get(s, 0) + 0.5) / (t + 0.5 * n_)) for s, v in r.items())
    right = {}
    for a in every:
        sc = {b: sum(ll(X[a][1][w]["replies"], X[b][0][w]["replies"]) for w in everyw) for b in every}; right[a] = max(sc, key=sc.get)
    p(f"Fingerprint: seed 1's replies matched to seed 0's arms, {sum(a == b for a, b in right.items())} of {len(every)} right; wrong: {[(a, b) for a, b in right.items() if a != b]}")
    if (OUT / "judge.jsonl").exists():
        Jd = list(rows(OUT / "judge.jsonl")); nf = len(FIG); same = lambda r: str(r["pick"]).strip().casefold() == FIG.get(r["arm"], "").casefold(); none = lambda r: str(r["pick"]).strip().casefold().startswith("none"); hits = [same(r) for r in Jd if r["arm"] in FIG]; none_f = np.mean([none(r) for r in Jd if r["arm"] in FIG]); none_c = np.mean([none(r) for r in Jd if r["arm"] in CTRL]) if any(r["arm"] in CTRL for r in Jd) else float("nan")
        by = {}
        for r in Jd:
            if r["arm"] in FIG:
                by.setdefault(r["arm"], []).append(same(r))
        p(f"Blind judge: right {np.mean(hits):.3f} (chance {1 / nf:.3f}); says none: figures {none_f:.3f}, controls {none_c:.3f}. Per figure: " + ", ".join(f"{a} {np.mean(v):.2f}" for a, v in by.items()))


if __name__ == "__main__":
    c = sys.argv[1]; a = sys.argv[2:]
    {"words": words, "fresh": lambda: fresh(a[0], a[1], int(a[2]) if len(a) > 2 else None), "seq": lambda: seq(a[0], a[1], limit=int(a[2]) if len(a) > 2 else None), "think": lambda: think(a[0], a[1], only=int(a[2]) if len(a) > 2 else None),
     "dose": lambda: dose(*a[:1]), "implant": lambda: implant(*a[:1]), "judge": judge, "autonomy": lambda: autonomy(a[0], a[1]), "autosteer": lambda: autosteer(*a[:1]), "report": report}[c]()
