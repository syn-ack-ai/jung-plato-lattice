"""Two small models from scratch: archetype views (LeJEPA) vs next-token (README addendum 55; user: "yes create the two
small models and see if you can test this faster"). After Balestriero and LeCun (2026, arXiv 2610.39302): views of
the same thing train toward the representation other models share with far fewer samples than next-token prediction.
Here the shared thing is a causal pattern and the views are retellings of it.
  evalset           (Mac) the famous (items3) and invented (items7) story texts with the 8 capture models' states at 60%
                    depth (texts checked against the captures' hashes), and the 3-way items -> tinyarch/evalsets.npz
  patterns          (CUDA GPU, vLLM E4B) new causal patterns -> tinyarch/patterns.json
  views             (CUDA GPU, vLLM E4B) 8 framed retellings per pattern -> tinyarch/views.json
  tok               BPE (8,192) on the training views -> tinyarch/tok.json
  train ARM SEED    ARM in jepa, ntp, ntp_grouped -> tinyarch/curve_ARM_sSEED.json
  report            -> tinyarch/report.md
  ~/g4agent/swe/bin/python tinyarch.py patterns      (CUDA GPU)"""
import hashlib, json, math, re, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "tinyarch"
sys.path.insert(0, str(HERE))
AREAS = ["family", "work", "farming", "the sea", "school", "travel", "building", "cooking", "weather", "animals", "music", "trade",
         "law and rules", "games", "crafts", "medicine", "money", "neighbours", "friendship", "sport", "rivers", "forests", "cities",
         "villages", "markets", "ships", "roads", "harvests", "festivals", "letters and news", "teaching", "inheritance", "promises",
         "rumours", "tools", "water", "fire", "seasons", "journeys", "old age"]
FRAMES = ["as a myth or fable set in a world a little apart from ours, with invented names", "as an ordinary modern story about a family",
          "as a story at a workplace", "as a story at a school or among children", "as a story about farming, animals or nature (no talking animals)",
          "as a story from a historical time", "as a story about a sport or a game", "as a story about trade, money or a market"]
PATTERN_Q = ("Invent a new underlying causal pattern for a story: a short chain of causes and effects in general terms (what someone "
             "does, what follows from it, how it turns out), with no names, no known story, no idiom or saying. Area: {area}. "
             "Answer with one sentence of 15-35 words and nothing else.")
VIEW_Q = ("Here is an underlying pattern: {pattern}\nTell one short story (60-90 words, present tense, plain English) that follows "
          "this pattern, {frame}. Every causal step must be there, in order. Do not state the pattern or a moral, do not name it, "
          "nothing about computers or software. Answer with the story only.")
ABS_FACETS = ["its places", "its objects and tools", "its people and roles", "its materials", "its sounds and colours", "its customs",
              "its parts and layout", "its size and shape"]                                   # addendum 71 amendment: one facet per sample (variety)
ABS_Q = ("Write one general statement about {area}, about {facet}: what it is like, how it is arranged or what it "
         "consists of, in general terms. Describe a state, not events: no causes and effects, nothing that happens over time. "
         "No names, no known story, no idiom or saying. Answer with one sentence of 15-35 words and nothing else.")   # addendum 71
CAUSAL = re.compile(r"\b(because|since|so|thus|therefore|hence|then|after|afterwards|before|until|unless|if|when|whenever|once|while|during|only to|"
                    r"leads?|led|leading|caus\w*|result\w*|consequen\w*|effects?|due|thanks|makes?|making|made(?!\s+(of|from|up))|turns?|turned|"
                    r"becom\w*|became|eventually|finally|soon|later|whereupon|ultimately|gradually|culminat\w*|over time|in turn|subsequent\w*|"
                    r"unexpected\w*|unintentional\w*|trigger\w*|prompt\w*|forc\w*|allow\w*|enabl\w*|encourag\w*|prevent\w*|yield\w*|breed\w*|bred|"
                    r"foster\w*|spur\w*|ensur\w*|(?<!fresh )produc\w*|driv\w*|drove|bring\w*|brought|let|lets|letting|leav\w*|left|creat(e|es|ed|ing|ion|ions|or|ors)|generat\w*|"
                    r"provok\w*|induc\w*|spark\w*|fuel\w*|invit\w*|risk\w*|threat\w*|undermin\w*|erod\w*|damag\w*|ruin\w*|destroy\w*|collaps\w*|"
                    r"fail\w*|succeed\w*|reward\w*|punish\w*|consequently|accordingly|inevitabl\w*|whereby|thereby|which means|so that|in order|"
                    r"alter\w*|reveal\w*|initiat\w*|cascad\w*|escalat\w*|reshap\w*|spread\w*|strain\w*|attract\w*|delay\w*|divert\w*|redirect\w*|"
                    r"diminish\w*|manifest\w*|spawn\w*|overflow\w*|submerg\w*|introduc\w*|releas\w*|halt\w*|chang\w*|shift\w*|transform\w*|"
                    r"increas\w*|decreas\w*|grow\w*|grew|rais\w*|lower\w*|reduc\w*|improv\w*|worsen\w*|weaken\w*|strengthen\w*|deepen\w*)\b", re.I)
STATE = re.compile(r"\b(is|are|has|have|holds?|contains?|consists?|stands?|sits?|lies|lie|comprises?|constitutes?|encompass\w*|involves?|includes?|features?|composed|made of)\b", re.I)   # addendum 71 (review): together they pass 0 of the 2,200 patterns
SOFTWARE = re.compile(r"\b(comput|software|program|code[sd]?\b|coding|bug|debug|script|algorithm|digital)", re.I)
V, MAXLEN, D, LAYERS_N, HEADS, STEPS, BATCH, EVERY, LR = 8192, 384, 384, 6, 6, 6000, 128, 250, 3e-4
SIZES = {"S": (384, 6, 6), "M": (512, 8, 8), "L": (768, 12, 12)}               # addendum 70: width, layers, heads
D, LAYERS_N, HEADS = SIZES[__import__("os").environ.get("TINY_SIZE", "S")]
DATA = __import__("os").environ.get("TINY_DATA", "")                          # addendum 70: "" = addendum 55's data, "_big" = the new set
DEV = __import__("os").environ.get("TINY_DEVICE", "cuda")                    # addendum 66: the Mac runs with mps (float32)


def amp(torch, enabled=True):
    """bf16 autocast on CUDA as before; float32 elsewhere"""
    import contextlib
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled) if DEV.startswith("cuda") else contextlib.nullcontext()
TOKBUDGET = 12800                                                            # addendum 63: training tokens per step
GRADLOG = __import__("os").environ.get("TINY_GRADLOG") == "1"                  # addendum 67: log the two losses' gradient cosine


# ---------- evaluation sets (Mac) ----------
def evalset():
    import cross, rewrite
    models = ("e4b", "d31", "qwen3_8b", "llama31_8b", "mistral7b", "phi4mini", "olmo3_7b", "smollm3_3b")    # invented.MODELS
    out, meta = {}, {}
    for tag, folder, suf in (("items3", "items3", "_f"), ("items7", "items7", "_inv_f")):
        F = rewrite.folder(folder); neutral = [t for _, t in cross.neutral_chunks()[:30]]; texts = None
        for n in models:
            d = np.load(HERE / "cross" / f"{n}{suf}.npz"); L = d["story"].shape[1]; l = int(round(0.6 * (L - 1)))
            if texts is None:
                texts = []
                for s, i, r in zip(d["s_set"], d["s_item"], d["s_role"]):
                    if s == "N":
                        texts.append(neutral[int(i[1:])])
                    else:
                        it = F[i]; texts.append({"anchor": it[2], "true": it[3], "foil": it[6]}[r])
                assert [hashlib.sha1(t.encode()).hexdigest()[:16] for t in texts] == list(d["s_hash"]), f"{tag}: texts differ from the capture"
                meta[tag] = {"texts": texts, "set": list(d["s_set"]), "item": list(d["s_item"]), "role": list(d["s_role"])}
            out[f"{tag}__{n}"] = d["story"][:, l].astype(np.float16)
        meta[tag]["three_way"] = [{"id": i, "anchor": it[2], "true": it[3], "lure": it[5], "foil": it[6]} for i, it in sorted(F.items())]
    OUT.mkdir(exist_ok=True); np.savez(OUT / "evalsets.npz", **out); json.dump(meta, open(OUT / "evalsets.json", "w"))
    print({k: v.shape for k, v in out.items()})


# ---------- data (CUDA GPU, vLLM E4B) ----------
def _llm():
    import r1
    from transformers import AutoTokenizer
    from vllm import LLM
    p = r1.path_of("base"); return LLM(model=p, max_model_len=2048, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "audio": 0}), AutoTokenizer.from_pretrained(p)


def _chat(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _words(t):
    return set(re.findall(r"[a-z]{4,}", t.lower()))


def patterns(target=2200, big=False):
    from vllm import SamplingParams
    llm, tok = _llm(); kept, seen, rnd = [], [], 0
    if big:                                                                    # addendum 70: new patterns, none near addendum 55's
        seen = [_words(q["pattern"]) for q in json.load(open(OUT / "patterns.json"))]
        M_ = json.load(open(OUT / "evalsets.json")); evw = [_words(t["anchor"] + " " + t["true"]) for g in ("items3", "items7") for t in M_[g]["three_way"]]   # review
    while len(kept) < target and rnd < (40 if big else 12):
        outs = llm.generate([_chat(tok, PATTERN_Q.format(area=a)) for a in AREAS],                 # review: a seed per request and round
                            [SamplingParams(n=16, temperature=1.0, top_p=0.95, top_k=-1, max_tokens=80, seed=(10**7 if big else 0) + 100000 * rnd + 100 * k) for k in range(len(AREAS))])   # addendum 70: seeds apart from addendum 55's
        for a, o in zip(AREAS, outs):
            for c in o.outputs:
                t = c.text.strip().strip('"').strip()
                if c.finish_reason != "stop" or not 12 <= len(t.split()) <= 45 or "\n" in t or SOFTWARE.search(t):
                    continue
                w = _words(t)
                if any(len(w & x) / max(len(w | x), 1) >= 0.5 for x in seen):        # near-duplicates removed
                    continue
                if big and any(len(w & e) / max(len(w), 1) >= 0.4 for e in evw):   # addendum 70 (review): none near an evaluation item
                    continue
                seen.append(w); kept.append({"id": f"{'b' if big else 'p'}{len(kept):04d}", "area": a, "pattern": t})
        rnd += 1; print(f"round {rnd}: {len(kept)} patterns", flush=True)
    if len(kept) < (0.75 if big else 0.8) * target:                               # review: keep what was made
        json.dump(kept, open(OUT / f"patterns{'_big' if big else ''}_partial.json", "w"), indent=1)
    assert len(kept) >= (0.75 if big else 0.8) * target, f"only {len(kept)} patterns"; OUT.mkdir(exist_ok=True); json.dump(kept[:target], open(OUT / f"patterns{'_big' if big else ''}.json", "w"), indent=1)


def views(big=False):
    from vllm import SamplingParams
    P = json.load(open(OUT / f"patterns{'_big' if big else ''}.json")); llm, tok = _llm(); out = {}
    for rnd in range(3):
        todo = [(p, f) for p in P for f in range(len(FRAMES)) if f not in out.get(p["id"], {})]
        if not todo:
            break
        outs = llm.generate([_chat(tok, VIEW_Q.format(pattern=p["pattern"], frame=FRAMES[f])) for p, f in todo],
                            [SamplingParams(n=1, temperature=0.8, top_p=0.95, top_k=-1, max_tokens=220, seed=1000000 * rnd + 10 * int(p["id"][1:]) + f) for p, f in todo])
        for (p, f), o in zip(todo, outs):
            t = o.outputs[0].text.strip().strip('"').strip(); n = len(t.split()); pw = re.findall(r"[a-z']+", p["pattern"].lower())
            tw = " ".join(re.findall(r"[a-z']+", t.lower())); copied = any(" ".join(pw[k:k + 6]) in tw for k in range(max(len(pw) - 5, 0)))
            if o.outputs[0].finish_reason == "stop" and 40 <= n <= 130 and not copied and not SOFTWARE.search(t):
                out.setdefault(p["id"], {})[f] = t
        print(f"round {rnd}: {sum(len(v) for v in out.values())} views", flush=True)
    rng = np.random.default_rng(55); ids = sorted(k for k, v in out.items() if len(v) >= 4); held = set(rng.choice(ids, len(ids) // 10, replace=False).tolist())
    json.dump({k: {"split": "heldout" if k in held else "train", "frames": sorted(out[k]), "views": [out[k][f] for f in sorted(out[k])]} for k in ids}, open(OUT / f"views{'_big' if big else ''}.json", "w"), indent=1)
    print(f"{len(ids)} patterns with 4+ views, {len(held)} held out", flush=True)


def abstract(pilot=False):
    """addendum 71 (Mac, MLX): base E4B writes general, non-causal statements per area; each training pattern gets one from
    its own area, matched on token length (nearest unused, longest patterns first) -> tinyarch/abstract.json; pilot: one
    round, yield per area -> tinyarch/abstract_pilot.jsonl"""
    import mlx.core as mx
    import rewrite
    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler
    from tokenizers import Tokenizer
    assert pilot or not (OUT / "abstract.json").exists(), "abstract.json exists"
    mx.random.seed(70 if pilot else 71); e4b, etok = rewrite.load_model("e4b"); btok = Tokenizer.from_file(str(OUT / "tok.json"))
    P = {q["id"]: q for q in json.load(open(OUT / "patterns.json"))}; Vw = json.load(open(OUT / "views.json"))
    tr = sorted(k for k, v in Vw.items() if v["split"] == "train"); need = {}
    for k in tr:
        need[P[k]["area"]] = need.get(P[k]["area"], 0) + 1
    seen = [_words(q["pattern"]) for q in P.values()]; pool = {a: [] for a in AREAS}; raw = open(OUT / ("abstract_pilot.jsonl" if pilot else "abstract_raw.jsonl"), "w")
    chat = lambda a, f: etok.apply_chat_template([{"role": "user", "content": ABS_Q.format(area=a, facet=f)}], add_generation_prompt=True, enable_thinking=False, tokenize=True)  # noqa: E731
    for rnd in range(1 if pilot else 60):
        todo = [(a, f) for a in AREAS for f in ABS_FACETS for _ in range(1 if pilot else 2) if len(pool[a]) < 1.5 * need.get(a, 0)]   # amendment: 2 per facet
        if not todo:
            break
        texts = []
        for j in range(0, len(todo), 128):                                    # amendment: batches of 128 (640 at once ran the Mac out of GPU memory)
            texts += batch_generate(e4b, etok, [chat(a, f) for a, f in todo[j:j + 128]], max_tokens=80, sampler=make_sampler(temp=1.0, top_p=0.95)).texts
        for (a, f), t in zip(todo, texts):
            t = t.strip().strip('"').strip(); w = _words(t)
            ok = t[-1:] in (".", "!", "?") and 12 <= len(t.split()) <= 45 and "\n" not in t and not SOFTWARE.search(t) and not CAUSAL.search(t) and bool(STATE.search(t))
            ok = ok and not any(len(w & x) / max(len(w | x), 1) >= 0.5 for x in seen)          # near-duplicates of each other or of a pattern
            raw.write(json.dumps({"round": rnd, "area": a, "facet": f, "text": t, "kept": ok}) + "\n")
            if ok:
                seen.append(w); pool[a].append(t)
        raw.flush(); print(f"round {rnd}: {sum(min(len(pool[a]), need.get(a, 0)) for a in AREAS)} of {len(tr)} patterns covered", flush=True)
    if pilot:
        print("kept of 8 per area:", {a: len(pool[a]) for a in AREAS}, f"; overall {sum(map(len, pool.values())) / (8 * len(AREAS)):.0%}", flush=True); return
    by = {}
    for a in AREAS:
        ks = sorted((k for k in tr if P[k]["area"] == a), key=lambda k: -len(btok.encode("Pattern: " + P[k]["pattern"]).ids))
        cand = [(t, len(btok.encode("Pattern: " + t).ids)) for t in pool[a]]; used = set()
        assert len(cand) >= len(ks), f"{a}: {len(cand)} sentences for {len(ks)} patterns"
        for k in ks:
            n = len(btok.encode("Pattern: " + P[k]["pattern"]).ids); j = min((j for j in range(len(cand)) if j not in used), key=lambda j: abs(cand[j][1] - n))
            used.add(j); by[k] = cand[j][0]
    json.dump({"by_pattern": by, "pool": pool}, open(OUT / "abstract.json", "w"), indent=1)
    ln = lambda xs: np.mean([len(btok.encode("Pattern: " + x).ids) for x in xs])  # noqa: E731
    lp_, la_ = ln([P[k]["pattern"] for k in tr]), ln(list(by.values()))
    print(f"{len(by)} sentences; mean tokens with the prefix: patterns {lp_:.1f}, abstract {la_:.1f}", flush=True)
    assert abs(lp_ - la_) <= 1, "mean token length differs by more than 1 (addendum 71 amendment)"


def abstract_rematch(lam=1.0):
    """addendum 71 amendment: nearest-length matching left the abstract sentences 1.14 tokens longer on average (tolerance
    1); per area, a min-cost assignment with cost |length difference| + lam x the sentence's length, the smallest lam
    of (0, 0.1, 0.25, 0.5, 1.0) that meets the tolerance -> tinyarch/abstract.json (the nearest-match file kept)"""
    from scipy.optimize import linear_sum_assignment
    from tokenizers import Tokenizer
    btok = Tokenizer.from_file(str(OUT / "tok.json")); A = json.load(open(OUT / "abstract.json")); P = {q["id"]: q for q in json.load(open(OUT / "patterns.json"))}
    Vw = json.load(open(OUT / "views.json")); tr = sorted(k for k, v in Vw.items() if v["split"] == "train"); ln = lambda t: len(btok.encode("Pattern: " + t).ids)  # noqa: E731
    if not (OUT / "abstract_nearest.json").exists():
        json.dump(A, open(OUT / "abstract_nearest.json", "w"), indent=1)
    by = {}
    for a in AREAS:
        ks = sorted(k for k in tr if P[k]["area"] == a); x = np.array([ln(P[k]["pattern"]) for k in ks]); y = np.array([ln(t) for t in A["pool"][a]])
        r, c = linear_sum_assignment(np.abs(x[:, None] - y[None, :]) + lam * y[None, :]); by.update({ks[i]: A["pool"][a][j] for i, j in zip(r, c)})
    lp_, la_ = np.mean([ln(P[k]["pattern"]) for k in tr]), np.mean([ln(v) for v in by.values()]); assert len(by) == len(tr)
    print(f"lambda {lam}: mean tokens with the prefix, patterns {lp_:.2f}, abstract {la_:.2f}", flush=True)
    assert abs(lp_ - la_) <= 1, "mean token length differs by more than 1 (addendum 71 amendment)"
    json.dump({"by_pattern": by, "pool": A["pool"], "lambda": lam}, open(OUT / "abstract.json", "w"), indent=1)

def content_replay(arm, seed, stmt, tok, Vw, steps=None):
    """addendum 71 (review): the content arms' batch sampler replayed without a model (the same generator calls as train);
    -> (sequences per step, share of training tokens in statement sequences, total sequences)"""
    assert arm in ("patmix", "absmix"); rng = np.random.default_rng(seed); tr_ids = [k for k, v in Vw.items() if v["split"] == "train"]; cache = {}
    enc = lambda t: cache[t] if t in cache else cache.setdefault(t, tok.encode(t).ids)  # noqa: E731
    seqs = stok = tot = 0; steps = steps or STEPS
    for _ in range(steps):
        n_tok = 0
        while n_tok < TOKBUDGET:
            q = tr_ids[rng.integers(len(tr_ids))]; story = enc(Vw[q]["views"][rng.integers(len(Vw[q]["views"]))])
            for k, part in enumerate((enc("Pattern: " + stmt[q]), story)):
                part = part[: MAXLEN - 1]; seqs += 1; n_tok += len(part); tot += len(part); stok += len(part) if k == 0 else 0
    return seqs / steps, stok / tot, seqs


def check71():
    """addendum 71 (Mac, before the runs): replayed batches of PATMIX and ABSMIX, seeds 0-4; statement-token share within 1 point"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(OUT / "tok.json")); Vw = json.load(open(OUT / "views.json")); P = {q["id"]: q["pattern"] for q in json.load(open(OUT / "patterns.json"))}
    AB = json.load(open(OUT / "abstract.json"))["by_pattern"]; out = {}
    for a, st in (("patmix", P), ("absmix", AB)):
        out[a] = [content_replay(a, s, st, tok, Vw) for s in range(5)]
        print(a, "sequences per step", " ".join(f"{x[0]:.1f}" for x in out[a]), "| statement-token share", " ".join(f"{x[1]:.4f}" for x in out[a]), flush=True)
    d = max(abs(x[1] - y[1]) for x, y in zip(out["patmix"], out["absmix"])); print(f"largest share difference {d:.4f}", flush=True)
    json.dump(out, open(OUT / "check71.json", "w")); assert d <= 0.01, "statement-token share differs by more than 1 point (addendum 71 amendment)"


def tok_train():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    Vw = json.load(open(OUT / "views.json")); texts = [t for v in Vw.values() if v["split"] == "train" for t in v["views"]]
    t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True); t.decoder = decoders.ByteLevel()
    t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=V, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[SEP]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    t.save(str(OUT / "tok.json")); print("tokenizer", t.get_vocab_size(), "on", len(texts), "texts", flush=True)


# ---------- model ----------
def build(torch, causal):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(MAXLEN, D)
            layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
            self.body = nn.TransformerEncoder(layer, LAYERS_N); self.norm = nn.LayerNorm(D); self.causal = causal
            self.proj = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, 128))     # JEPA projector (loss only)
            nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)   # review: N(0,1) handicapped the tied output

        def hidden(self, x, pad):
            T = x.shape[1]; h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1) if self.causal else None
            return self.norm(self.body(h, mask=mask, src_key_padding_mask=pad, is_causal=self.causal))

        def pooled(self, x, pad):
            h = self.hidden(x, pad); m = (~pad).float()[..., None]; return (h * m).sum(1) / m.sum(1).clamp_min(1)

        def logits(self, h):
            return h @ self.emb.weight.T                                                      # tied output layer

        def run_layers(self, x, pad, upto=None, detach_below=None):
            """addendum 75: the blocks one by one; upto=k returns the raw state after k blocks, detach_below=k cuts the
            gradient below block k; otherwise as hidden()"""
            T_ = x.shape[1]; h = self.emb(x) + self.pos(torch.arange(T_, device=x.device))[None]
            mask = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=x.device), 1) if self.causal else None
            for i, layer in enumerate(self.body.layers):
                if upto is not None and i == upto:
                    return h
                if detach_below is not None and i == detach_below:
                    h = h.detach()
                h = layer(h, src_mask=mask, src_key_padding_mask=pad, is_causal=self.causal)
            return h if upto is not None else self.norm(h)

        def pooled_mid(self, x, pad, upto=None):
            """addendum 75: the mean-pooled state after half the blocks (or after `upto` blocks), layer-normalized without parameters"""
            h = self.run_layers(x, pad, upto=LAYERS_N // 2 if upto is None else upto); m = (~pad).float()[..., None]
            return torch.nn.functional.layer_norm((h * m).sum(1) / m.sum(1).clamp_min(1), (h.shape[-1],))
    return Net()


def sigreg(torch, z, m=256):
    """LeJEPA's SIGReg: Epps-Pulley distance of random 1-D projections of the batch from N(0, 1), times the batch size"""
    a = torch.randn(z.shape[1], m, device=z.device); a = a / a.norm(dim=0, keepdim=True); p = (z @ a).float()
    t = torch.linspace(-3, 3, 17, device=z.device); x = p[..., None] * t
    err = (torch.cos(x).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(x).mean(0) ** 2
    return torch.trapezoid(err * torch.exp(-t ** 2 / 2), t, dim=-1).mean() * z.shape[0]


def encode(tok, texts, maxlen=MAXLEN):
    ids = [[2] + tok.encode(t).ids[: maxlen - 1] for t in texts]; n = max(len(i) for i in ids)
    return np.array([i + [0] * (n - len(i)) for i in ids]), np.array([[k >= len(i) for k in range(n)] for i in ids])


def mutual_knn(A, B, k=10, clamp=True):
    """Huh et al.: k nearest neighbours by cosine (self excluded), mean overlap; clamp: values clipped at the 0.95 quantile of
    |x| of each matrix before normalizing (their outlier step; review: the capture states have huge outlier dimensions)"""
    def nn_(X):
        X = X.astype(np.float64)
        if clamp:
            q = np.quantile(np.abs(X), 0.95); X = np.clip(X, -q, q)
        X = X / np.linalg.norm(X, axis=1, keepdims=True); S = X @ X.T; np.fill_diagonal(S, -np.inf); return np.argsort(-S, 1)[:, :k]
    a, b = nn_(A), nn_(B); return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


class Evaluator:
    def __init__(self, tok, Vw):
        E = np.load(OUT / "evalsets.npz"); M = json.load(open(OUT / "evalsets.json")); self.tok = tok; self.sets = {}
        for tag in ("items3", "items7"):
            self.sets[tag] = {"texts": M[tag]["texts"], "big": {k.split("__")[1]: E[k] for k in E.files if k.startswith(tag)}, "three": M[tag]["three_way"]}
        self.held = [v["views"] for v in Vw.values() if v["split"] == "heldout"]

    def embed(self, torch, net, texts):
        out = []
        for j in range(0, len(texts), 128):
            x, pad = encode(self.tok, texts[j:j + 128])
            with torch.no_grad(), amp(torch):
                out.append(net.pooled(torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)).float().cpu().numpy())
        return np.concatenate(out)

    def __call__(self, torch, net, rng):
        net.eval(); r = {}
        for tag, S in self.sets.items():
            Z = self.embed(torch, net, S["texts"]); r[f"align_{tag}"] = float(np.mean([mutual_knn(Z, B) for B in S["big"].values()]))
            r[f"align_raw_{tag}"] = float(np.mean([mutual_knn(Z, B, clamp=False) for B in S["big"].values()]))
            th = S["three"]; flat = [x for t in th for x in (t["anchor"], t["true"], t["lure"], t["foil"])]; Qraw = self.embed(torch, net, flat)
            Q = Qraw / np.linalg.norm(Qraw, axis=1, keepdims=True); Q = Q.reshape(len(th), 4, -1)
            if getattr(net, "p_half", False):                                 # addendum 68 (review): the pattern half alone, descriptive
                Qp = Qraw[:, : D // 2]; Qp = (Qp / np.linalg.norm(Qp, axis=1, keepdims=True)).reshape(len(th), 4, -1)
                r[f"tfP_{tag}"] = [int(Qp[i, 0] @ Qp[i, 1] > Qp[i, 0] @ Qp[i, 3]) for i in range(len(th))]
                r[f"tlP_{tag}"] = [int(Qp[i, 0] @ Qp[i, 1] > Qp[i, 0] @ Qp[i, 2]) for i in range(len(th))]
            r[f"match_{tag}"] = [int(Q[i, 0] @ Q[i, 1] > max(Q[i, 0] @ Q[i, 2], Q[i, 0] @ Q[i, 3])) for i in range(len(th))]   # ties are misses
            r[f"tf_{tag}"] = [int(Q[i, 0] @ Q[i, 1] > Q[i, 0] @ Q[i, 3]) for i in range(len(th))]      # addendum 61: true vs foil (same opening)
            r[f"tl_{tag}"] = [int(Q[i, 0] @ Q[i, 1] > Q[i, 0] @ Q[i, 2]) for i in range(len(th))]      # true vs lure
            r[f"cos_{tag}"] = [[round(float(Q[i, 0] @ Q[i, j]), 5) for j in (1, 2, 3)] for i in range(len(th))]   # true, lure, foil
        a = self.embed(torch, net, [v[0] for v in self.held]); b = self.embed(torch, net, [v[1] for v in self.held])
        a, b = a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True); hits = []
        for i in range(len(a)):
            others = rng.choice([j for j in range(len(a)) if j != i], 4, replace=False); hits.append(int(a[i] @ b[i] > max(a[i] @ b[j] for j in others)))
        r["viewret"] = float(np.mean(hits)); r["viewret_hits"] = hits                    # addendum 70 (review): per pattern, for a seeds x items bootstrap
        if getattr(net, "lm", False):                                          # review: memorization check for the next-token arms
            x, pad = encode(self.tok, [v[2] for v in self.held if len(v) > 2], MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
            tot, cnt = 0.0, 0                                                  # addendum 70 (review): logits in chunks of 64 rows (about 800 held-out patterns in _big)
            with torch.no_grad(), amp(torch):
                h = net.hidden(x[:, :-1], pad[:, :-1])
                for j in range(0, len(h), 64):
                    lg = net.logits(h[j:j + 64]).float(); y = x[j:j + 64, 1:].masked_fill(pad[j:j + 64, 1:], -100)
                    tot += float(torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1), ignore_index=-100, reduction="sum")); cnt += int((y != -100).sum())
            r["heldout_loss"] = tot / cnt
        net.train(); return r


def train(arm, seed, prefix="curve"):
    import torch
    from tokenizers import Tokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tok = Tokenizer.from_file(str(OUT / "tok.json"))
    Vw = json.load(open(OUT / f"views{DATA}.json")); tr = [v["views"] for v in Vw.values() if v["split"] == "train"]
    CONTENT = ("inst", "patex", "patmix", "mix50", "fade", "absmix")          # addenda 63, 69, 71: content arms, equal tokens per step
    assert arm in ("jepa", "ntp", "ntp_grouped", "jepa_causal", "mlm", "mlm_jepa", "ntp_jepa", "wm", "wm_x", "mlm_jepa_split", "ntp_jepa_split", "twospace", "twospace_notok", "mlm_jepa_mid", "mlm_jepa_top", "mlm_jepa_w01", "mlm_jepa_curr", "mlm_jepa_low") + CONTENT, arm
    net = build(torch, causal=arm in ("ntp", "ntp_grouped", "jepa_causal", "ntp_jepa", "wm", "wm_x", "ntp_jepa_split") + CONTENT).to(DEV); net.lm = arm in ("ntp", "ntp_grouped", "ntp_jepa") + CONTENT   # same init per seed within an architecture
    wmlog = {}
    if arm in ("twospace", "twospace_notok"):                                 # addendum 68: heads built after the encoder, without shifting the random stream
        with torch.random.fork_rng(devices=[]):
            net.add_module("proj_p", torch.nn.Sequential(torch.nn.Linear(D // 2, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)).to(DEV))
            net.add_module("cons", torch.nn.Sequential(torch.nn.Linear(128 + D - D // 2, 256), torch.nn.GELU(), torch.nn.Linear(256, 128)).to(DEV))
        split2 = {}
        for pi, vs in enumerate(tr):
            for vi, t in enumerate(vs):
                sn = re.split(r"(?<=[.!?])\s+", t.strip())
                if len(sn) >= 3:
                    split2[pi, vi] = (" ".join(sn[:-1]), sn[-1])
        pats2 = {}
        for pi, vi in split2:
            pats2.setdefault(pi, []).append(vi)
        pats2_ok = sorted(pi for pi, v in pats2.items() if len(v) >= 2); print("twospace: patterns with two split retellings", len(pats2_ok), flush=True)
        net.p_half = True
        ho_pairs = []                                                         # held-out consequence pairs (review): one per held-out pattern, fixed
        for k_, v_ in Vw.items():
            if v_["split"] != "heldout":
                continue
            sp = [re.split(r"(?<=[.!?])\s+", t.strip()) for t in v_["views"]]; ok = [x for x in sp if len(x) >= 3]
            if len(ok) >= 2:
                ho_pairs.append((" ".join(ok[0][:-1]), ok[1][-1]))

        def cons_eval():
            net.eval()
            with torch.no_grad(), amp(torch):
                xc, pc = encode(tok, [c_ for c_, _ in ho_pairs], MAXLEN // 2); xo, po = encode(tok, [o_ for _, o_ in ho_pairs], MAXLEN // 2)
                hc = net.pooled(torch.tensor(xc, device=DEV), torch.tensor(pc, device=DEV)).float(); hoo = net.pooled(torch.tensor(xo, device=DEV), torch.tensor(po, device=DEV)).float()
                pcz = net.proj_p(hc[:, : D // 2]).float(); tg = net.proj_p(hoo[:, : D // 2]).float(); pr = net.cons(torch.cat([pcz, hc[:, D // 2:]], -1)).float()
                var = float(tg.var(0).mean()) + 1e-12; rolled = torch.roll(tg, 1, 0)
                out = {"ho_cons": float(((pr - tg) ** 2).mean()) / var, "ho_copy": float(((pcz - tg) ** 2).mean()) / var,
                       "ho_shuffled": float(((pr - rolled) ** 2).mean()) / var, "ho_sigreg": float(sigreg(torch, tg)), "ho_n": len(ho_pairs)}
            net.train(); return out
    if arm.endswith("_split"):                                                # addendum 67: the views loss's own projector, built after the encoder
        with torch.random.fork_rng(devices=[]):                               # review: no shift of the random stream (masks match the combined arms)
            net.add_module("proj_s", torch.nn.Sequential(torch.nn.Linear(D // 2, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)).to(DEV))

    def grad_cos(lt, lj):
        """cosine between the two losses' gradients on the shared encoder parameters, overall and per block (no update)"""
        named = [(n, q) for n, q in net.named_parameters() if not n.startswith(("proj", "wm_pred"))]
        gt = torch.autograd.grad(lt, [q for _, q in named], retain_graph=True, allow_unused=True)
        gj = torch.autograd.grad(lj, [q for _, q in named], retain_graph=True, allow_unused=True)
        groups = {}
        for (n, q), a, b in zip(named, gt, gj):
            if a is None or b is None:
                continue
            key = "emb" if n.startswith(("emb", "pos")) else n.split(".")[2] if n.startswith("body.layers") else "norm"
            g = groups.setdefault(key, [0.0, 0.0, 0.0]); g[0] += float((a * b).sum()); g[1] += float((a * a).sum()); g[2] += float((b * b).sum())
        cos = lambda ks: (lambda t: t[0] / (t[1] ** 0.5 * t[2] ** 0.5 + 1e-12))([sum(groups[k][i] for k in ks if k in groups) for i in range(3)])  # noqa: E731
        body = [k for k in groups if k.isdigit()]; low = [k for k in body if int(k) < LAYERS_N - 1]
        out = {"gcos": cos(list(groups)), "gcos_body": cos(body), "gcos_low": cos(low)}       # review: body = encoder layers only (the tied
        out.update({f"gcos_{k}": cos([k]) for k in groups})                                  # output layer is a token-only head); low = layers 0-4
        out.update({f"graw_{k}": v for k, v in groups.items()})                              # raw dot product and squared norms, for any regrouping
        return out
    if arm in ("wm", "wm_x"):                                                # addendum 66: the predictor is built after the encoder (same encoder init)
        pred = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 128)).to(DEV); net.add_module("wm_pred", pred)
        split = {}
        for pi, vs in enumerate(tr):
            for vi, t in enumerate(vs):
                sn = re.split(r"(?<=[.!?])\s+", t.strip())
                if len(sn) >= 3:
                    split[pi, vi] = (" ".join(sn[:-1]), sn[-1])
        wm_keys = sorted(split); print("wm: retellings with 3+ sentences", len(wm_keys), "of", sum(len(v) for v in tr), flush=True)
        by_pat = {}
        for pi, vi in wm_keys:
            by_pat.setdefault(pi, []).append(vi)
        x_pats = sorted(pi for pi, v in by_pat.items() if len(v) >= 2)            # wm_x: cause of one retelling -> outcome of another
    if arm in CONTENT:
        tr_ids = [k for k, v in Vw.items() if v["split"] == "train"]; stmt = {q["id"]: q["pattern"] for q in json.load(open(OUT / f"patterns{DATA}.json"))}
        if arm == "absmix":                                                   # addendum 71: the statement replaced by an abstract, non-causal sentence
            assert DATA == "", "abstract sentences exist for addendum 55's data only"
            ab = json.load(open(OUT / "abstract.json"))["by_pattern"]; assert all(q in ab for q in tr_ids), "abstract sentences missing"; stmt = {q: ab[q] for q in tr_ids}
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    ev = Evaluator(tok, Vw); curve = []; t0 = time.time()
    print(arm, seed, sum(p.numel() for p in net.parameters()) / 1e6, "M parameters,", len(tr), "train patterns", flush=True)
    for step in range(STEPS + 1):
        if step % EVERY == 0:
            r = ev(torch, net, np.random.default_rng(1000 + step)); r.update(step=step, samples=step * BATCH, tokens=step * TOKBUDGET if arm in CONTENT else None, **wmlog); curve.append(r)
            if arm in ("twospace", "twospace_notok"):
                curve[-1].update(cons_eval())
            print(arm, seed, step, {k: (round(v, 4) if isinstance(v, float) else round(float(np.mean(v)), 4)) for k, v in r.items() if k not in ("step", "samples", "tokens") and v is not None}, f"{time.time() - t0:.0f}s", flush=True)
        if step == STEPS:
            break
        lr = LR * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        if arm not in CONTENT and arm not in ("wm", "wm_x"):              # review: twospace draws its batch exactly as mlm does
            P = rng.choice(len(tr), BATCH // 2, replace=False); pairs = [rng.choice(len(tr[p]), 2, replace=False) for p in P]
        with amp(torch):
            if arm in CONTENT:                                  # addendum 63: fill the batch with sequences up to 12,800 training tokens
                seqs, n_tok = [], 0
                while n_tok < TOKBUDGET:
                    q = tr_ids[rng.integers(len(tr_ids))]; story = tok.encode(Vw[q]["views"][rng.integers(len(Vw[q]["views"]))]).ids
                    if arm in ("mix50", "fade"):                       # addendum 69: the pattern line on some stories only
                        said = rng.random() < (0.5 if arm == "mix50" else max(0.0, 1 - step / STEPS))
                        parts = [tok.encode("Pattern: " + stmt[q] + "\n").ids + story] if said else [story]
                    elif arm == "inst":
                        parts = [story]
                    elif arm == "patex":
                        parts = [tok.encode("Pattern: " + stmt[q] + "\n").ids + story]
                    else:                                               # patmix; absmix (addendum 71) with its own sentences
                        parts = [tok.encode("Pattern: " + stmt[q]).ids, story]
                    for part in parts:
                        part = part[: MAXLEN - 1]; seqs.append([2] + part); n_tok += len(part)
                seqs.sort(key=len); chunks, cur = [], []                         # review: micro-batches of at most about 8k padded positions
                for q_ in seqs:
                    if cur and len(q_) * (len(cur) + 1) > 8192:
                        chunks.append(cur); cur = []
                    cur.append(q_)
                chunks.append(cur); tot = 0.0; wmlog = {"c_seqs": wmlog.get("c_seqs", 0) + len(seqs), "c_tok": wmlog.get("c_tok", 0) + n_tok}   # addendum 71: batch counters
                for ch in chunks:
                    n = max(len(q_) for q_ in ch); xc = np.array([q_ + [0] * (n - len(q_)) for q_ in ch]); pc = xc == 0; pc[:, 0] = False
                    xc, pc = torch.tensor(xc, device=DEV), torch.tensor(pc, device=DEV)
                    h = net.hidden(xc[:, :-1], pc[:, :-1]); m = ~pc[:, 1:]
                    lc = torch.nn.functional.cross_entropy(net.logits(h[m]).float(), xc[:, 1:][m], reduction="sum") / n_tok
                    lc.backward(); tot += float(lc)
                loss = None
            elif arm in ("twospace", "twospace_notok"):            # addendum 68: pattern space (views + consequence), detail space (masked tokens)
                full = [tr[p_][k_] for p_, ks_ in zip(P, pairs) for k_ in ks_]          # the same texts as mlm's batch (review)
                x, pad = encode(tok, full, MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                zp = net.proj_p(net.pooled(x, pad)[:, : D // 2]).float(); z2 = zp.view(BATCH // 2, 2, -1)
                cp = [(int(p_), int(ks_[0]), int(ks_[1])) for p_, ks_ in zip(P, pairs) if (int(p_), int(ks_[0])) in split2 and (int(p_), int(ks_[1])) in split2]
                xc, pc = encode(tok, [split2[p_, a_][0] for p_, a_, _ in cp], MAXLEN // 2); xo, po = encode(tok, [split2[p_, b_][1] for p_, _, b_ in cp], MAXLEN // 2)
                hc = net.pooled(torch.tensor(xc, device=DEV), torch.tensor(pc, device=DEV)); ho = net.pooled(torch.tensor(xo, device=DEV), torch.tensor(po, device=DEV))
                pc_ = net.proj_p(hc[:, : D // 2]).float(); target = net.proj_p(ho[:, : D // 2]).float()
                with amp(torch, False):
                    c = z2.mean(1, keepdim=True); lv = 0.95 * ((z2 - c) ** 2).mean() + 0.05 * (sigreg(torch, z2[:, 0]) + sigreg(torch, z2[:, 1])) / 2
                    predc = net.cons(torch.cat([pc_, hc[:, D // 2:].float()], -1)); cmse = ((predc - target) ** 2).mean(); lc = 0.95 * cmse + 0.05 * sigreg(torch, target)
                loss = lv + lc
                if arm == "twospace":
                    keep_b = torch.cat([torch.zeros(D // 2, device=DEV), torch.ones(D - D // 2, device=DEV)])
                    pick = (torch.rand(x.shape, device=DEV) < 0.15) & ~pad; pick[:, 0] = False; u = torch.rand(x.shape, device=DEV)
                    xm = torch.where(pick & (u < 0.8), torch.ones_like(x), x); xm = torch.where(pick & (u >= 0.8) & (u < 0.9), torch.randint(4, V, x.shape, device=DEV), xm)
                    loss = loss + torch.nn.functional.cross_entropy(net.logits(net.hidden(xm, pad) * keep_b).float().reshape(-1, V), x.masked_fill(~pick, -100).reshape(-1), ignore_index=-100)
                wmlog = {"cons_mse": float(cmse.detach())}
            elif arm in ("wm", "wm_x"):                            # addendum 66: predict the OUTCOME's embedding from the CAUSE's
                if arm == "wm":
                    ks = [wm_keys[j] for j in rng.choice(len(wm_keys), BATCH, replace=False)]; causes = [split[k][0] for k in ks]; outs = [split[k][1] for k in ks]
                else:                                              # amendment (review): another retelling's outcome, so names and setting cannot carry it
                    pp = [x_pats[j] for j in rng.choice(len(x_pats), BATCH, replace=False)]; ab = [rng.choice(by_pat[pi], 2, replace=False) for pi in pp]
                    causes = [split[pi, a][0] for pi, (a, b) in zip(pp, ab)]; outs = [split[pi, b][1] for pi, (a, b) in zip(pp, ab)]
                xc, pc = encode(tok, causes, MAXLEN // 2); xo, po = encode(tok, outs, MAXLEN // 2)
                ec = net.proj(net.pooled(torch.tensor(xc, device=DEV), torch.tensor(pc, device=DEV))).float()
                eo = net.proj(net.pooled(torch.tensor(xo, device=DEV), torch.tensor(po, device=DEV))).float()
                with amp(torch, False):
                    mse = ((net.wm_pred(ec) - eo) ** 2).mean(); sg = (sigreg(torch, ec) + sigreg(torch, eo)) / 2; loss = 0.95 * mse + 0.05 * sg
                wmlog = {"wm_mse": float(mse.detach()), "wm_sigreg": float(sg.detach())}            # review: the two terms logged apart
            elif arm in ("mlm_jepa", "ntp_jepa"):                  # addendum 65: token loss plus views loss on the same 128 texts, weight 1 each
                x, pad = encode(tok, [tr[p][k] for p, ks in zip(P, pairs) for k in ks], MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                z = net.proj(net.pooled(x, pad)).float(); z2 = z.view(BATCH // 2, 2, -1)
                with amp(torch, False):
                    c = z2.mean(1, keepdim=True); lj = 0.95 * ((z2 - c) ** 2).mean() + 0.05 * (sigreg(torch, z2[:, 0]) + sigreg(torch, z2[:, 1])) / 2
                if arm == "mlm_jepa":
                    pick = (torch.rand(x.shape, device=DEV) < 0.15) & ~pad; pick[:, 0] = False; u = torch.rand(x.shape, device=DEV)
                    xm = torch.where(pick & (u < 0.8), torch.ones_like(x), x); xm = torch.where(pick & (u >= 0.8) & (u < 0.9), torch.randint(4, V, x.shape, device=DEV), xm)
                    lt = torch.nn.functional.cross_entropy(net.logits(net.hidden(xm, pad)).float().reshape(-1, V), x.masked_fill(~pick, -100).reshape(-1), ignore_index=-100)
                else:
                    lt = torch.nn.functional.cross_entropy(net.logits(net.hidden(x[:, :-1], pad[:, :-1])).float().reshape(-1, V), x[:, 1:].masked_fill(pad[:, 1:], -100).reshape(-1), ignore_index=-100)
                loss = lt + lj
                if GRADLOG and step % EVERY == 0:
                    curve[-1].update(grad_cos(lt, lj))
            elif arm in ("mlm_jepa_mid", "mlm_jepa_top", "mlm_jepa_w01", "mlm_jepa_curr", "mlm_jepa_low"):   # addendum 75: where and how strongly the views loss enters
                x, pad = encode(tok, [tr[p][k] for p, ks in zip(P, pairs) for k in ks], MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                if arm == "mlm_jepa_mid":
                    hv = net.pooled_mid(x, pad)
                elif arm == "mlm_jepa_low":                                     # amendment 2: after block 2 of 6
                    hv = net.pooled_mid(x, pad, upto=LAYERS_N // 3)
                elif arm == "mlm_jepa_top":
                    hz = net.run_layers(x, pad, detach_below=LAYERS_N - 1); m_ = (~pad).float()[..., None]; hv = (hz * m_).sum(1) / m_.sum(1).clamp_min(1)
                else:
                    hv = net.pooled(x, pad)
                z = net.proj(hv).float(); z2 = z.view(BATCH // 2, 2, -1)
                with amp(torch, False):
                    c = z2.mean(1, keepdim=True); lj = 0.95 * ((z2 - c) ** 2).mean() + 0.05 * (sigreg(torch, z2[:, 0]) + sigreg(torch, z2[:, 1])) / 2
                pick = (torch.rand(x.shape, device=DEV) < 0.15) & ~pad; pick[:, 0] = False; u = torch.rand(x.shape, device=DEV)
                xm = torch.where(pick & (u < 0.8), torch.ones_like(x), x); xm = torch.where(pick & (u >= 0.8) & (u < 0.9), torch.randint(4, V, x.shape, device=DEV), xm)
                lt = torch.nn.functional.cross_entropy(net.logits(net.hidden(xm, pad)).float().reshape(-1, V), x.masked_fill(~pick, -100).reshape(-1), ignore_index=-100)
                wv = 0.1 if arm == "mlm_jepa_w01" else 0.0 if (arm == "mlm_jepa_curr" and step < STEPS // 2) else 1.0
                loss = lt + wv * lj
            elif arm in ("mlm_jepa_split", "ntp_jepa_split"):            # addendum 67: views on the first half of the final state, tokens on the second
                x, pad = encode(tok, [tr[p][k] for p, ks in zip(P, pairs) for k in ks], MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                z = net.proj_s(net.pooled(x, pad)[:, : D // 2]).float(); z2 = z.view(BATCH // 2, 2, -1)
                with amp(torch, False):
                    c = z2.mean(1, keepdim=True); lj = 0.95 * ((z2 - c) ** 2).mean() + 0.05 * (sigreg(torch, z2[:, 0]) + sigreg(torch, z2[:, 1])) / 2
                keep_b = torch.cat([torch.zeros(D // 2, device=DEV), torch.ones(D - D // 2, device=DEV)])
                if arm == "mlm_jepa_split":
                    pick = (torch.rand(x.shape, device=DEV) < 0.15) & ~pad; pick[:, 0] = False; u = torch.rand(x.shape, device=DEV)
                    xm = torch.where(pick & (u < 0.8), torch.ones_like(x), x); xm = torch.where(pick & (u >= 0.8) & (u < 0.9), torch.randint(4, V, x.shape, device=DEV), xm)
                    lt = torch.nn.functional.cross_entropy(net.logits(net.hidden(xm, pad) * keep_b).float().reshape(-1, V), x.masked_fill(~pick, -100).reshape(-1), ignore_index=-100)
                else:
                    lt = torch.nn.functional.cross_entropy(net.logits(net.hidden(x[:, :-1], pad[:, :-1]) * keep_b).float().reshape(-1, V), x[:, 1:].masked_fill(pad[:, 1:], -100).reshape(-1), ignore_index=-100)
                loss = lt + lj
                if GRADLOG and step % EVERY == 0:
                    curve[-1].update(grad_cos(lt, lj))
            elif arm in ("jepa", "jepa_causal"):                  # 64 patterns x 2 views = 128 texts (addendum 61: also on the causal encoder)
                x, pad = encode(tok, [tr[p][k] for p, ks in zip(P, pairs) for k in ks], MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                z = net.proj(net.pooled(x, pad)).float(); z2 = z.view(BATCH // 2, 2, -1)
                with amp(torch, False):                  # review: LeJEPA's scale (means; SIGReg per view, float32)
                    c = z2.mean(1, keepdim=True); loss = 0.95 * ((z2 - c) ** 2).mean() + 0.05 * (sigreg(torch, z2[:, 0]) + sigreg(torch, z2[:, 1])) / 2
            elif arm == "mlm":                                  # addendum 61: BERT-style masked tokens on the bidirectional encoder, the same 128 texts
                x, pad = encode(tok, [tr[p][k] for p, ks in zip(P, pairs) for k in ks], MAXLEN // 2); x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                pick = (torch.rand(x.shape, device=DEV) < 0.15) & ~pad; pick[:, 0] = False; u = torch.rand(x.shape, device=DEV)
                xm = torch.where(pick & (u < 0.8), torch.ones_like(x), x)                              # [UNK] (id 1) serves as the mask token
                xm = torch.where(pick & (u >= 0.8) & (u < 0.9), torch.randint(4, V, x.shape, device=DEV), xm)
                lg = net.logits(net.hidden(xm, pad)).float(); y = x.masked_fill(~pick, -100)
                loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1), ignore_index=-100)
            else:
                if arm == "ntp":                                # the same 128 texts, one per sequence
                    texts = [tr[p][k] for p, ks in zip(P, pairs) for k in ks]; x, pad = encode(tok, texts, MAXLEN // 2)
                else:                                           # ntp_grouped: the two views of a pattern in one sequence
                    seqs = [tok.encode(tr[p][ks[0]]).ids[: MAXLEN // 2 - 2] + [3] + tok.encode(tr[p][ks[1]]).ids[: MAXLEN // 2 - 1] for p, ks in zip(P, pairs)]
                    n = max(len(s) for s in seqs) + 1; x = np.array([[2] + s + [0] * (n - 1 - len(s)) for s in seqs]); pad = x == 0; pad[:, 0] = False
                x, pad = torch.tensor(x, device=DEV), torch.tensor(pad, device=DEV)
                lg = net.logits(net.hidden(x[:, :-1], pad[:, :-1])).float(); y = x[:, 1:].masked_fill(pad[:, 1:], -100)
                loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1), ignore_index=-100)
        if loss is not None:                                                   # content arms backpropagate per micro-batch above
            loss.backward()
        else:
            loss = torch.tensor(tot)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
        if step % 500 == 0:
            print(arm, seed, step, "loss", round(float(loss), 4), flush=True)
    json.dump({"arm": arm, "seed": seed, "device": DEV, "size": [D, LAYERS_N, HEADS], "data": DATA, "params": sum(p.numel() for p in net.parameters()),
               "host": __import__("socket").gethostname(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if DEV.startswith("cuda") else DEV,     # addendum 70 (review): which machine and files
               "files_md5": {n: hashlib.md5(open(OUT / n, "rb").read()).hexdigest() for n in ("tok.json", "evalsets.npz", "evalsets.json", f"views{DATA}.json")}, "curve": curve}, open(OUT / f"{prefix}_{arm}_s{seed}.json", "w"))
    torch.save(net.state_dict(), OUT / f"{prefix}_{arm}_s{seed}.pt")                     # addendum 61: weights kept


def report():
    C = {}
    for f in sorted(OUT.glob("curve_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):     # review: pair arms by seed
        d = json.load(open(f)); C.setdefault(d["arm"], []).append(d["curve"])
    align = lambda c: np.array([(r["align_items3"] + r["align_items7"]) / 2 for r in c])  # noqa: E731
    lines = ["# Two small models from scratch: archetype views vs next-token (addendum 55)", "",
             "| arm (seeds) | ALIGN last step (items3 / items7) | MATCH items3 / items7 | VIEWRET |", "|---|---|---|---|"]
    for arm, cs in C.items():
        last = [c[-1] for c in cs]
        lines.append(f"| {arm} ({len(cs)}) | {np.mean([(r['align_items3'] + r['align_items7']) / 2 for r in last]):.4f} ({np.mean([r['align_items3'] for r in last]):.4f} / "
                     f"{np.mean([r['align_items7'] for r in last]):.4f}) | {np.mean([np.mean(r['match_items3']) for r in last]):.3f} / {np.mean([np.mean(r['match_items7']) for r in last]):.3f} | "
                     f"{np.mean([r['viewret'] for r in last]):.3f} |")
    if "jepa" in C and "ntp" in C:
        assert len(C["jepa"]) == len(C["ntp"]), "seeds differ between arms"
        J, N = np.array([align(c) for c in C["jepa"]]), np.array([align(c) for c in C["ntp"]]); samples = [r["samples"] for r in C["ntp"][0]]
        lines.append("")
        lines.append("- ALIGN by samples, seed means (JEPA / NTP" + (" / NTP_GROUPED" if "ntp_grouped" in C else "") + "): " + "; ".join(
            f"{s // 1000}k {J.mean(0)[k]:.4f} / {N.mean(0)[k]:.4f}" + (f" / {np.array([align(c) for c in C['ntp_grouped']]).mean(0)[k]:.4f}" if "ntp_grouped" in C else "")
            for k, s in enumerate(samples) if k % 4 == 0 or k == len(samples) - 1))
        n_last = N.mean(0)[-1]; reach = next((samples[k] for k in range(len(samples)) if J.mean(0)[k] >= n_last), None)
        all_seeds = all(J[i, -1] > N[i, -1] for i in range(min(len(J), len(N))))
        ratio = (reach / samples[-1]) if reach is not None else None
        lines.append(f"- H1: JEPA above NTP at the last step in every seed: {'yes' if all_seeds else 'no'} ({', '.join(f'{a:+.4f}' for a in (J[:, -1] - N[:, -1]))}); "
                     f"JEPA reaches NTP's last ALIGN ({n_last:.4f}) at {('%dk samples, %.3f of NTP' % (reach // 1000, ratio)) if reach is not None else 'never'} -> "
                     f"{'supported' if all_seeds and ratio is not None and ratio <= 1 / 3 else 'not supported'}")
        for tag in ("items3", "items7"):
            mj = np.mean([c[-1][f"match_{tag}"] for c in C["jepa"]], 0); mn = np.mean([c[-1][f"match_{tag}"] for c in C["ntp"]], 0); rng = np.random.default_rng(0)
            bs = [np.mean((mj - mn)[ix]) for ix in (rng.integers(0, len(mj), len(mj)) for _ in range(2000))]
            lines.append(f"- H2 {tag}: MATCH JEPA minus NTP {np.mean(mj - mn):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
        if "ntp_grouped" in C:
            G = np.array([align(c) for c in C["ntp_grouped"]]); lines.append(f"- H3 (descriptive): NTP_GROUPED minus NTP, last-step ALIGN {G.mean(0)[-1] - N.mean(0)[-1]:+.4f}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report61():
    """addendum 61: P1 JEPA_CAUSAL minus NTP and P2 JEPA minus MLM on last-step true-vs-foil (mean of items3 and items7), P3
    views minus token prediction on the TF gain from step 0 (pooled over the two architectures); items and seeds resampled"""
    C = {}
    for f in sorted(OUT.glob("curve61_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]
    arms = [a for a in ("ntp", "jepa_causal", "mlm", "jepa") if a in C]; seeds = sorted(set.intersection(*(set(C[a]) for a in arms)))
    T = ("items3", "items7"); rng = np.random.default_rng(61)
    same0 = {(a, b): all(np.allclose(C[a][s][0][f"cos_{t}"], C[b][s][0][f"cos_{t}"], atol=1e-4) for s in seeds for t in T)    # review: identical start
             for a, b in (("ntp", "jepa_causal"), ("mlm", "jepa")) if a in C and b in C}
    M = lambda a, m, t, k: np.array([C[a][s][k][f"{m}_{t}"] for s in seeds], dtype=float)  # noqa: E731   seeds x items
    lines = ["# Small models, same architecture per comparison (addendum 61)", "", f"- seeds {seeds}; arms {arms}; identical step-0 states within each architecture: "
             + ", ".join(f"{a}/{b} {'yes' if v else 'NO'}" for (a, b), v in same0.items()) + " (so P3 equals the mean of P1 and P2)", "",
             "| arm | TF step 0 -> last (items3 / items7) | TL last | MATCH last | ALIGN last | TF peak (samples) |", "|---|---|---|---|---|---|"]
    for a in arms:
        tf0 = [M(a, "tf", t, 0).mean() for t in T]; tf1 = [M(a, "tf", t, -1).mean() for t in T]
        curve = np.mean([[np.mean([np.mean(r[f"tf_{t}"]) for t in T]) for r in C[a][s]] for s in seeds], 0); kpk = int(np.argmax(curve))
        lines.append(f"| {a} | {tf0[0]:.3f} -> {tf1[0]:.3f} / {tf0[1]:.3f} -> {tf1[1]:.3f} | {np.mean([M(a, 'tl', t, -1).mean() for t in T]):.3f} | "
                     f"{np.mean([M(a, 'match', t, -1).mean() for t in T]):.3f} | {np.mean([np.mean([C[a][s][-1][f'align_{t}'] for s in seeds]) for t in T]):.4f} | "
                     f"{curve[kpk]:.3f} ({C[a][seeds[0]][kpk]['samples'] // 1000}k) |")

    def boot(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, len(seeds), len(seeds)); ii = {t: rng.integers(0, M(arms[0], "tf", t, -1).shape[1], M(arms[0], "tf", t, -1).shape[1]) for t in T}
            out.append(fn(si, ii))
        return np.array(out)
    def last(a, b):
        return lambda si, ii: np.mean([M(a, "tf", t, -1)[si][:, ii[t]].mean() - M(b, "tf", t, -1)[si][:, ii[t]].mean() for t in T])
    def gain(a):
        return lambda si, ii, t: (M(a, "tf", t, -1) - M(a, "tf", t, 0))[si][:, ii[t]].mean()
    full = {t: np.arange(M(arms[0], "tf", t, -1).shape[1]) for t in T}; allseeds = np.arange(len(seeds))
    lines.append("")
    for lab, a, b in (("P1 (primary): JEPA_CAUSAL minus NTP, last-step TF", "jepa_causal", "ntp"), ("P2: JEPA minus MLM, last-step TF", "jepa", "mlm")):
        if a in C and b in C:
            pt = last(a, b)(allseeds, full); bs = boot(last(a, b)); lo, hi = np.percentile(bs, [2.5, 97.5])
            per = ", ".join(f"{t} {np.mean(M(a, 'tf', t, -1)) - np.mean(M(b, 'tf', t, -1)):+.3f}" for t in T)
            lines.append(f"- {lab}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] ({per}) -> {'supported' if lo > 0 else 'not supported'}")
    if all(a in C for a in ("jepa", "jepa_causal", "mlm", "ntp")):
        f3 = lambda si, ii: np.mean([np.mean([gain(v)(si, ii, t) - gain(k)(si, ii, t) for t in T]) for v, k in (("jepa_causal", "ntp"), ("jepa", "mlm"))])  # noqa: E731
        pt = f3(allseeds, full); bs = boot(f3); lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"- P3: TF gain from step 0, views minus token prediction (both architectures): {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'supported' if lo > 0 else 'not supported'}")
        lines.append("- TL, last step, views minus token prediction: causal " + f"{np.mean([M('jepa_causal', 'tl', t, -1).mean() - M('ntp', 'tl', t, -1).mean() for t in T]):+.3f}, bidirectional "
                     f"{np.mean([M('jepa', 'tl', t, -1).mean() - M('mlm', 'tl', t, -1).mean() for t in T]):+.3f}")
    (OUT / "report61.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report63():
    """addendum 63: E1 efficiency k (tokens PATEX needs to reach INST's last-step TF), E2 PATEX minus INST, E3 PATEX minus
    PATMIX; bootstrap over seeds and items (5,000)"""
    C = {}
    for f in sorted(OUT.glob("curve63_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]
    arms = [a for a in ("inst", "patex", "patmix") if a in C]; seeds = sorted(set.intersection(*(set(C[a]) for a in arms))); T = ("items3", "items7")
    rng = np.random.default_rng(63); tok_per = TOKBUDGET * EVERY; ncp = len(C[arms[0]][seeds[0]])
    TF = {a: {t: np.array([[C[a][s][k][f"tf_{t}"] for k in range(ncp)] for s in seeds], dtype=float) for t in T} for a in arms}   # seeds x checkpoints x items
    curve = lambda a, si, ii: np.mean([TF[a][t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    full_s = np.arange(len(seeds)); full_i = {t: np.arange(TF[arms[0]][t].shape[2]) for t in T}
    same0 = all(np.allclose(C[a][s][0]["cos_items3"], C[arms[0]][s][0]["cos_items3"], atol=1e-4) for a in arms for s in seeds)
    lines = ["# Teaching the pattern vs deriving it from examples (addendum 63)", "", f"- seeds {seeds}; arms {arms}; equal tokens per step ({TOKBUDGET}); identical step 0 across arms: {'yes' if same0 else 'NO'}", "",
             "| arm | TF step 0 -> last (mean of items3 and items7) | TF peak (tokens) | TL last | VIEWRET last | ALIGN last | held-out loss last |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        c = curve(a, full_s, full_i); kpk = int(np.argmax(c))
        lines.append(f"| {a} | {c[0]:.3f} -> {c[-1]:.3f} | {c[kpk]:.3f} ({kpk * tok_per / 1e6:.1f}M) | {np.mean([np.mean([np.mean(C[a][s][-1][f'tl_{t}']) for t in T]) for s in seeds]):.3f} | "
                     f"{np.mean([C[a][s][-1]['viewret'] for s in seeds]):.3f} | {np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in T]) for s in seeds]):.4f} | "
                     f"{np.mean([C[a][s][-1].get('heldout_loss', np.nan) for s in seeds]):.2f} |")

    def draws(fn, n=5000):
        out = []
        for _ in range(n):
            si = rng.integers(0, len(seeds), len(seeds)); ii = {t: rng.integers(0, len(full_i[t]), len(full_i[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out, dtype=float)
    lines.append("")
    if "inst" in C:
        g = draws(lambda si, ii: curve("inst", si, ii)[-1] - curve("inst", si, ii)[0]); g0 = curve("inst", full_s, full_i); glo = np.percentile(g, 2.5)
        lines.append(f"- INST's own TF gain from step 0: {g0[-1] - g0[0]:+.3f} [{glo:+.3f}, {np.percentile(g, 97.5):+.3f}]")
    if "inst" in C and "patex" in C:
        def stable_hit(c, target):                                          # review: the first checkpoint after which the curve stays >= target
            return next((j for j in range(1, ncp) if (c[j:] >= target).all()), None)

        def k_of(si, ii):
            target = curve("inst", si, ii)[-1]; hi = stable_hit(curve("inst", si, ii), target); hp = stable_hit(curve("patex", si, ii), target)
            return 0.0 if hp is None else hi / hp                               # never reached: less efficient (review)
        if glo > 0:
            kd = draws(k_of); kp = k_of(full_s, full_i)
            lines.append(f"- E1 (primary): PATEX reaches INST's last-step TF with k = {kp:.2f} times fewer tokens; 95% CI [{np.percentile(kd, 2.5):.2f}, {np.percentile(kd, 97.5):.2f}] "
                         f"(PATEX never stays at INST's final level in {np.mean(kd == 0):.1%} of draws) -> {'supported' if np.percentile(kd, 2.5) > 1 else 'not supported'}")
        else:
            lines.append("- E1: undefined (INST's own TF gain does not clear 0); E2 decides")
        for lab, a, b in (("E2: PATEX minus INST", "patex", "inst"), ("E3: PATEX minus PATMIX", "patex", "patmix")):
            if a in C and b in C:
                d = draws(lambda si, ii: curve(a, si, ii)[-1] - curve(b, si, ii)[-1]); pt = curve(a, full_s, full_i)[-1] - curve(b, full_s, full_i)[-1]
                per = ", ".join(f"{t} {TF[a][t][:, -1].mean() - TF[b][t][:, -1].mean():+.3f}" for t in T)
                lines.append(f"- {lab}, last-step TF: {pt:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] ({per})" + (f" -> {'supported' if np.percentile(d, 2.5) > 0 else 'not supported'}" if a == "patex" and b == "inst" else ""))
    (OUT / "report63.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report65():
    """addendum 65: the combined arms against addendum 61's arms (same seeds and init): C1 TL and TF against MLM, C2 the
    stable-reach k for MLM's last-step TF; the causal pair descriptive"""
    C = {}
    for f in sorted(list(OUT.glob("curve61_*_s*.json")) + list(OUT.glob("curve65_*_s*.json")), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]
    T = ("items3", "items7"); rng = np.random.default_rng(65); seeds = sorted(set.intersection(*(set(C[a]) for a in C))); ncp = len(C["mlm"][seeds[0]])
    assert seeds == [0, 1, 2], f"seeds present in every arm: {seeds}"                # review: a missing seed must not change C1 silently
    same0 = {(a, b): all(np.allclose(C[a][s][0][f"cos_{t}"], C[b][s][0][f"cos_{t}"], atol=1e-4) for s in seeds for t in T)
             for a, b in (("mlm_jepa", "mlm"), ("mlm_jepa", "jepa"), ("ntp_jepa", "ntp"), ("ntp_jepa", "jepa_causal")) if a in C and b in C}
    A = {a: {(m, t): np.array([[C[a][s][k][f"{m}_{t}"] for k in range(ncp)] for s in seeds], dtype=float) for m in ("tf", "tl") for t in T} for a in C}
    cur = lambda a, m, si, ii: np.mean([A[a][m, t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    fs = np.arange(len(seeds)); fi = {t: np.arange(A["mlm"]["tf", t].shape[2]) for t in T}

    def draws(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, len(seeds), len(seeds)); ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out, dtype=float)

    def stable_hit(c, target):
        return next((j for j in range(1, ncp) if (c[j:] >= target).all()), None)
    lines = ["# Views plus token prediction (addendum 65)", "", f"- seeds {seeds}; arms {sorted(C)}; identical step 0: " + ", ".join(f"{a}/{b} {'yes' if v else 'NO'}" for (a, b), v in same0.items()), "",
             "| arm | TF step 0 -> last | TL step 0 -> last | VIEWRET last | ALIGN last | held-out loss last |", "|---|---|---|---|---|---|"]
    for a in ("ntp", "jepa_causal", "ntp_jepa", "mlm", "jepa", "mlm_jepa"):
        if a in C:
            tf, tl = cur(a, "tf", fs, fi), cur(a, "tl", fs, fi)
            lines.append(f"| {a} | {tf[0]:.3f} -> {tf[-1]:.3f} | {tl[0]:.3f} -> {tl[-1]:.3f} | {np.mean([C[a][s][-1]['viewret'] for s in seeds]):.3f} | "
                         f"{np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in T]) for s in seeds]):.4f} | {np.mean([C[a][s][-1].get('heldout_loss') or np.nan for s in seeds]):.2f} |")
    lines.append("")
    for comb, tok_arm, view_arm, lab in (("mlm_jepa", "mlm", "jepa", "bidirectional"), ("ntp_jepa", "ntp", "jepa_causal", "causal")):
        if comb not in C:
            continue
        D_ = {}
        for m, b in (("tl", tok_arm), ("tf", tok_arm), ("tl", view_arm), ("tf", view_arm)):
            d = draws(lambda si, ii: cur(comb, m, si, ii)[-1] - cur(b, m, si, ii)[-1]); pt = cur(comb, m, fs, fi)[-1] - cur(b, m, fs, fi)[-1]; D_[m, b] = d
            lines.append(f"- {lab}: {comb} minus {b}, last-step {m.upper()}: {pt:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        def k_of(si, ii):
            target = cur(tok_arm, "tf", si, ii)[-1]; ht = stable_hit(cur(tok_arm, "tf", si, ii), target); hc = stable_hit(cur(comb, "tf", si, ii), target)
            return 0.0 if hc is None else ht / hc
        kd = draws(k_of, 5000); lines.append(f"- {lab}: k (tokens {tok_arm} needs to stably reach its own last TF / {comb}'s) {k_of(fs, fi):.2f} [{np.percentile(kd, 2.5):.2f}, {np.percentile(kd, 97.5):.2f}]")
        if comb == "mlm_jepa":
            tl, tf = D_["tl", tok_arm], D_["tf", tok_arm]                       # review: the same draws as the lines above
            lines.append(f"- C1 (primary): TL CI above 0 ({np.percentile(tl, 2.5):+.3f}) and TF CI lower bound above -0.03 ({np.percentile(tf, 2.5):+.3f}) -> "
                         f"{'supported' if np.percentile(tl, 2.5) > 0 and np.percentile(tf, 2.5) > -0.03 else 'not supported'}; C2: {'supported' if np.percentile(kd, 2.5) > 1 else 'not supported'}")
    (OUT / "report65.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report66():
    """addendum 66: WM vs NTP (both on the same device): W1 last-step TF, W2 mean TF over checkpoints after step 0"""
    C = {}
    devs = set()
    for f in sorted(OUT.glob("curve66_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]; devs.add(d.get("device"))
    assert len(devs) == 1, f"arms trained on different devices: {devs}"
    T = ("items3", "items7"); rng = np.random.default_rng(66); seeds = sorted(set(C["wm"]) & set(C["ntp"])); assert seeds == [0, 1, 2], seeds
    ncp = len(C["ntp"][0]); arms = [a for a in ("ntp", "wm", "wm_x") if a in C]
    assert all(len(C[a][s]) == ncp for a in arms for s in seeds), "checkpoint counts differ"
    A = {a: {(m, t): np.array([[C[a][s][k][f"{m}_{t}"] for k in range(ncp)] for s in seeds], dtype=float) for m in ("tf", "tl") for t in T} for a in arms}
    cur = lambda a, m, si, ii: np.mean([A[a][m, t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    fs = np.arange(len(seeds)); fi = {t: np.arange(A["ntp"]["tf", t].shape[2]) for t in T}
    same0 = all(np.allclose(C["wm"][s][0][f"cos_{t}"], C["ntp"][s][0][f"cos_{t}"], atol=1e-4) for s in seeds for t in T)

    def draws(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, len(seeds), len(seeds)); ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out)
    lines = ["# A world-model objective vs next-token (addendum 66)", "", f"- seeds {seeds}; trained on {devs.pop()}; identical step 0: {'yes' if same0 else 'NO'}", "",
             "| arm | TF step 0 -> last | TF mean after step 0 | TL last | VIEWRET last | ALIGN last | NTP held-out loss / WM mse, sigreg (last) |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        tf = cur(a, "tf", fs, fi)
        lines.append(f"| {a} | {tf[0]:.3f} -> {tf[-1]:.3f} | {tf[1:].mean():.3f} | {cur(a, 'tl', fs, fi)[-1]:.3f} | {np.mean([C[a][s][-1]['viewret'] for s in seeds]):.3f} | "
                     f"{np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in T]) for s in seeds]):.4f} | "
                     + (f"{np.mean([C[a][s][-1].get('heldout_loss') for s in seeds]):.2f}" if a == "ntp" else f"{np.mean([C[a][s][-1].get('wm_mse', np.nan) for s in seeds]):.4f}, {np.mean([C[a][s][-1].get('wm_sigreg', np.nan) for s in seeds]):.3f}") + " |")
    lines.append("")
    if "wm_x" in C:
        for lab, fn in (("W3 (secondary): last-step TF, WM_X minus NTP", lambda si, ii: cur("wm_x", "tf", si, ii)[-1] - cur("ntp", "tf", si, ii)[-1]),
                        ("W3: mean TF after step 0, WM_X minus NTP", lambda si, ii: cur("wm_x", "tf", si, ii)[1:].mean() - cur("ntp", "tf", si, ii)[1:].mean())):
            d = draws(fn); lines.append(f"- {lab}: {fn(fs, fi):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
    for lab, fn in (("W1 (primary): last-step TF, WM minus NTP", lambda si, ii: cur("wm", "tf", si, ii)[-1] - cur("ntp", "tf", si, ii)[-1]),
                    ("W2: mean TF after step 0, WM minus NTP", lambda si, ii: cur("wm", "tf", si, ii)[1:].mean() - cur("ntp", "tf", si, ii)[1:].mean()),
                    ("TL last step, WM minus NTP", lambda si, ii: cur("wm", "tl", si, ii)[-1] - cur("ntp", "tl", si, ii)[-1])):
        d = draws(fn); pt = fn(fs, fi); lo, hi = np.percentile(d, [2.5, 97.5])
        lines.append(f"- {lab}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (f" -> {'supported' if lo > 0 else 'not supported'}" if lab.startswith("W") else ""))
    for t in T:
        lines.append(f"- {t}: last-step TF WM {A['wm']['tf', t][:, -1].mean():.3f}, NTP {A['ntp']['tf', t][:, -1].mean():.3f}")
    (OUT / "report66.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report67():
    """addendum 67: G1 gradient conflict (MLM_JEPA), S1/S2 split vs combined on the mean of TF and TL"""
    C = {}; devs = set()
    for f in sorted(OUT.glob("curve67_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]; devs.add(d.get("device"))
    assert len(devs) == 1, devs
    T = ("items3", "items7"); rng = np.random.default_rng(67); seeds = sorted(set.intersection(*(set(C[a]) for a in C))); assert seeds == [0, 1, 2], seeds
    ncp = len(C["mlm_jepa"][0]); arms = [a for a in ("mlm_jepa", "mlm_jepa_split", "ntp_jepa", "ntp_jepa_split") if a in C]
    assert all(len(C[a][s]) == ncp for a in arms for s in seeds), "checkpoint counts differ"
    assert all("gcos_body" in C[a][s][1] for a in arms for s in seeds), "gradient cosines were not logged (TINY_GRADLOG=1)"
    same0 = {(a, b): all(np.allclose(C[a][s][0][f"cos_{t}"], C[b][s][0][f"cos_{t}"], atol=1e-4) for s in seeds for t in ("items3", "items7"))
             for a, b in (("mlm_jepa_split", "mlm_jepa"), ("ntp_jepa_split", "ntp_jepa")) if a in C and b in C}
    A = {a: {(m, t): np.array([[C[a][s][k][f"{m}_{t}"] for k in range(ncp)] for s in seeds], dtype=float) for m in ("tf", "tl") for t in T} for a in arms}
    both = lambda a, si, ii: np.mean([(A[a]["tf", t][si][:, :, ii[t]].mean(axis=(0, 2)) + A[a]["tl", t][si][:, :, ii[t]].mean(axis=(0, 2))) / 2 for t in T], 0)  # noqa: E731
    one = lambda a, m, si, ii: np.mean([A[a][m, t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    fs = np.arange(3); fi = {t: np.arange(A[arms[0]]["tf", t].shape[2]) for t in T}

    def draws(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, 3, 3); ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out)
    lines = ["# Do the two kinds of learning cancel, and do separate spaces help? (addendum 67)", "", f"- seeds {seeds}; trained on {devs.pop()}; identical step 0: "
             + ", ".join(f"{a}/{b} {'yes' if v else 'NO'}" for (a, b), v in same0.items()), "",
             "| arm | TF last | TL last | (TF+TL)/2 last | mean after step 0 | VIEWRET last | ALIGN last | gradient cosine, encoder layers (per seed) | layers 0-4 |", "|---|---|---|---|---|---|---|---|---|"]
    gc, gl = {}, {}
    for a in arms:
        g = np.array([[C[a][s][k].get("gcos_body", np.nan) for k in range(1, ncp - 1)] for s in seeds]); gc[a] = g   # review: step 6000 has no gradient
        gl[a] = np.array([[C[a][s][k].get("gcos_low", np.nan) for k in range(1, ncp - 1)] for s in seeds])
        lines.append(f"| {a} | {one(a, 'tf', fs, fi)[-1]:.3f} | {one(a, 'tl', fs, fi)[-1]:.3f} | {both(a, fs, fi)[-1]:.3f} | {both(a, fs, fi)[1:].mean():.3f} | "
                     f"{np.mean([C[a][s][-1]['viewret'] for s in seeds]):.3f} | {np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in ('items3', 'items7')]) for s in seeds]):.4f} | "
                     f"{np.nanmean(g):+.3f} ({', '.join(f'{v:+.3f}' for v in np.nanmean(g, 1))}) | {np.nanmean(gl[a]):+.3f} |")
    lines.append("")
    g = gc["mlm_jepa"]; gd = []
    for _ in range(10000):
        si = rng.integers(0, 3, 3); ki = rng.integers(0, g.shape[1], g.shape[1]); gd.append(np.nanmean(g[si][:, ki]))
    lo, hi = np.percentile(gd, [2.5, 97.5]); g1 = np.nanmean(g) < 0 and hi < 0
    lines.append(f"- G1 (conflict, MLM_JEPA, encoder layers): mean gradient cosine {np.nanmean(g):+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'supported: the losses fight' if g1 else 'not supported'}")
    blocks = sorted({k[5:] for a in arms for s in seeds for k in C[a][s][1] if k.startswith("gcos_")})
    lines.append("- per block (MLM_JEPA, mean after step 0): " + ", ".join(f"{b} {np.nanmean([[C['mlm_jepa'][s][k].get('gcos_' + b, np.nan) for k in range(1, ncp - 1)] for s in seeds]):+.3f}" for b in blocks))
    for sa, ca in (("mlm_jepa_split", "mlm_jepa"), ("ntp_jepa_split", "ntp_jepa")):
        if sa in gl and ca in gl:
            lines.append(f"- conflict on layers 0-4 (separated layers left out): {sa} {np.nanmean(gl[sa]):+.3f}, {ca} {np.nanmean(gl[ca]):+.3f}")
    res = {}
    for split_a, comb_a, lab in (("mlm_jepa_split", "mlm_jepa", "bidirectional"), ("ntp_jepa_split", "ntp_jepa", "causal")):
        if split_a in C and comb_a in C:
            for nm, fn in (("S1 last step", lambda si, ii: both(split_a, si, ii)[-1] - both(comb_a, si, ii)[-1]),
                           ("S2 mean after step 0", lambda si, ii: both(split_a, si, ii)[1:].mean() - both(comb_a, si, ii)[1:].mean()),
                           ("TF last", lambda si, ii: one(split_a, "tf", si, ii)[-1] - one(comb_a, "tf", si, ii)[-1]),
                           ("TL last", lambda si, ii: one(split_a, "tl", si, ii)[-1] - one(comb_a, "tl", si, ii)[-1])):
                d = draws(fn); lo, hi = np.percentile(d, [2.5, 97.5]); res[lab, nm] = lo > 0
                lines.append(f"- {lab}: {split_a} minus {comb_a}, {nm}: {fn(fs, fi):+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (primary)" if lab == "bidirectional" and nm.startswith("S1") else ""))
    s1 = res.get(("bidirectional", "S1 last step"), False)
    lines.append(f"- S1 -> {'supported' if s1 else 'not supported'}; S2 -> {'supported' if res.get(('bidirectional', 'S2 mean after step 0'), False) else 'not supported'}; "
                 f"reading: {'support for the hypothesis at this scale' if g1 and s1 else 'separation helps for another reason' if s1 else 'conflict without this fix' if g1 else 'neither'}")
    (OUT / "report67.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report68():
    """addendum 68: TWOSPACE vs MLM_JEPA (T1) and vs MLM (T2), TWOSPACE_NOTOK vs MLM on TF (T3), the consequence loss (L1)"""
    C = {}; devs = set()
    for f in sorted(OUT.glob("curve68_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]; devs.add(d.get("device"))
    assert len(devs) == 1, devs
    T = ("items3", "items7"); rng = np.random.default_rng(68); seeds = sorted(set.intersection(*(set(C[a]) for a in C))); assert seeds == [0, 1, 2], seeds
    arms = [a for a in ("mlm", "mlm_jepa", "mlm_jepa_split", "twospace", "twospace_notok") if a in C]; ncp = len(C["mlm"][0])
    assert all(len(C[a][s]) == ncp for a in arms for s in seeds), "checkpoint counts differ"
    same0 = all(np.allclose(C[a][s][0][f"cos_{t}"], C["mlm"][s][0][f"cos_{t}"], atol=1e-4) for a in arms for s in seeds for t in T)
    A = {a: {(m, t): np.array([[C[a][s][k][f"{m}_{t}"] for k in range(ncp)] for s in seeds], dtype=float) for m in ("tf", "tl") for t in T} for a in arms}
    one = lambda a, m, si, ii: np.mean([A[a][m, t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    both = lambda a, si, ii: (one(a, "tf", si, ii) + one(a, "tl", si, ii)) / 2  # noqa: E731
    fs = np.arange(3); fi = {t: np.arange(A["mlm"]["tf", t].shape[2]) for t in T}

    def draws(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, 3, 3); ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out)
    lines = ["# A world model with two connected spaces (addendum 68)", "", f"- seeds {seeds}; trained on {devs.pop()}; identical step 0 across arms: {'yes' if same0 else 'NO'}", "",
             "| arm | TF last | TL last | (TF+TL)/2 last | mean after step 0 | VIEWRET last | ALIGN last | consequence loss 32k -> last |", "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        cl = [np.mean([C[a][s][k].get("cons_mse", np.nan) for s in seeds]) for k in (1, ncp - 1)]
        lines.append(f"| {a} | {one(a, 'tf', fs, fi)[-1]:.3f} | {one(a, 'tl', fs, fi)[-1]:.3f} | {both(a, fs, fi)[-1]:.3f} | {both(a, fs, fi)[1:].mean():.3f} | "
                     f"{np.mean([C[a][s][-1]['viewret'] for s in seeds]):.3f} | {np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in T]) for s in seeds]):.4f} | "
                     + (f"{cl[0]:.3f} -> {cl[1]:.3f}" if not np.isnan(cl[0]) else "") + " |")
    lines.append("")
    tests = [("T1 (primary): (TF+TL)/2 last, TWOSPACE minus MLM_JEPA", "twospace", "mlm_jepa", lambda a, b: lambda si, ii: both(a, si, ii)[-1] - both(b, si, ii)[-1]),
             ("T2: TL last, TWOSPACE minus MLM", "twospace", "mlm", lambda a, b: lambda si, ii: one(a, "tl", si, ii)[-1] - one(b, "tl", si, ii)[-1]),
             ("T2: TF last, TWOSPACE minus MLM", "twospace", "mlm", lambda a, b: lambda si, ii: one(a, "tf", si, ii)[-1] - one(b, "tf", si, ii)[-1]),
             ("T3: TF last, TWOSPACE_NOTOK minus MLM", "twospace_notok", "mlm", lambda a, b: lambda si, ii: one(a, "tf", si, ii)[-1] - one(b, "tf", si, ii)[-1]),
             ("T1b: (TF+TL)/2 last, TWOSPACE minus MLM_JEPA_SPLIT (the consequence step alone)", "twospace", "mlm_jepa_split", lambda a, b: lambda si, ii: both(a, si, ii)[-1] - both(b, si, ii)[-1]),
             ("TWOSPACE_NOTOK's own TF gain from step 0", "twospace_notok", "twospace_notok", lambda a, b: lambda si, ii: one(a, "tf", si, ii)[-1] - one(a, "tf", si, ii)[0]),
             ("(TF+TL)/2 mean after step 0, TWOSPACE minus MLM_JEPA", "twospace", "mlm_jepa", lambda a, b: lambda si, ii: both(a, si, ii)[1:].mean() - both(b, si, ii)[1:].mean())]
    R = {}
    for lab, a, b, mk in tests:
        if a in C and b in C:
            fn = mk(a, b); d = draws(fn); lo, hi = np.percentile(d, [2.5, 97.5]); R[lab] = (lo, hi)
            lines.append(f"- {lab}: {fn(fs, fi):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    v = lambda k, ok: ("not run" if k not in R else "supported" if ok else "not supported")  # noqa: E731
    k1, k2, k3 = "T1 (primary): (TF+TL)/2 last, TWOSPACE minus MLM_JEPA", "T2: TL last, TWOSPACE minus MLM", "T3: TF last, TWOSPACE_NOTOK minus MLM"
    t1 = k1 in R and R[k1][0] > 0; t2 = k2 in R and R[k2][0] > 0 and R["T2: TF last, TWOSPACE minus MLM"][0] > -0.03; t3 = k3 in R and R[k3][0] > 0
    lines.append(f"- verdicts: T1 {v(k1, t1)}; T2 {v(k2, t2)}; T3 (superiority, CI above 0) {v(k3, t3)}"
                 + (f"; T3 non-inferiority (lower bound above -0.03): {'yes' if R[k3][0] > -0.03 else 'no'}" if k3 in R else ""))
    for a in ("twospace", "twospace_notok"):                                   # L1 on held-out patterns, normalized, against the shuffled pairing
        if a in C:
            hc = [np.mean([C[a][s][k]["ho_cons"] for s in seeds]) for k in (1, ncp - 1)]; hs = np.mean([C[a][s][ncp - 1]["ho_shuffled"] for s in seeds])
            hcp = np.mean([C[a][s][ncp - 1]["ho_copy"] for s in seeds])
            lines.append(f"- L1 {a}: held-out consequence error (normalized) {hc[0]:.3f} -> {hc[1]:.3f}; shuffled-pattern baseline {hs:.3f}; copy baseline {hcp:.3f} -> "
                         + ("learned beyond pattern identity (at most 0.7 x shuffled)" if hc[1] <= 0.7 * hs else "not beyond the shuffled baseline"))
            tfp = np.mean([np.mean([np.mean(C[a][s][-1][f"tfP_{t}"]) for t in T]) for s in seeds]); tlp = np.mean([np.mean([np.mean(C[a][s][-1][f"tlP_{t}"]) for t in T]) for s in seeds])
            lines.append(f"- {a}, pattern half alone (descriptive): TF {tfp:.3f}, TL {tlp:.3f}")
    (OUT / "report68.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report69():
    """addendum 69: MIX50 and FADE against addendum 63's INST and PATEX (same device, seeds, start)"""
    C = {}
    for f in sorted(list(OUT.glob("curve63_*_s*.json")) + list(OUT.glob("curve69_*_s*.json")), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]
    T = ("items3", "items7"); rng = np.random.default_rng(69); S5 = [0, 1, 2, 3, 4]
    for a in ("inst", "patex", "mix50"):                                      # review: M1 needs these three complete
        assert a in C and sorted(C[a]) == S5, f"{a}: seeds {sorted(C.get(a, {}))}"
    arms = [a for a in ("inst", "patex", "patmix", "mix50", "fade") if a in C and sorted(C[a]) == S5]
    ncp = len(C["inst"][0]); assert all(len(C[a][s]) == ncp for a in arms for s in S5)
    same0 = all(np.allclose(C[a][s][0]["cos_items3"], C["inst"][s][0]["cos_items3"], atol=1e-4) for a in arms for s in S5)
    TF = {a: {t: np.array([[C[a][s][k][f"tf_{t}"] for k in range(ncp)] for s in S5], dtype=float) for t in T} for a in arms}
    VR = {a: np.array([[C[a][s][k]["viewret"] for k in range(ncp)] for s in S5]) for a in arms}
    cur = lambda a, si, ii: np.mean([TF[a][t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    fs = np.arange(5); fi = {t: np.arange(TF["inst"][t].shape[2]) for t in T}

    def draws(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, 5, 5); ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out, dtype=float)

    def stable_hit(c, target):
        return next((j for j in range(1, ncp) if (c[j:] >= target).all()), None)
    lines = ["# The mixed pattern test (addendum 69)", "", f"- seeds {S5}; device: the box (assumed for addendum 63's files); identical step 0 across arms: {'yes' if same0 else 'NO'}"
             + ("" if "fade" in arms else "; FADE incomplete"), "",
             "| arm | TF step 0 -> last (items3 / items7) | TF mean after step 0 | VIEWRET last | TL last | ALIGN last | held-out loss last |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        c = cur(a, fs, fi)
        lines.append(f"| {a} | {c[0]:.3f} -> {c[-1]:.3f} ({TF[a]['items3'][:, -1].mean():.3f} / {TF[a]['items7'][:, -1].mean():.3f}) | {c[1:].mean():.3f} | {VR[a][:, -1].mean():.3f} | "
                     f"{np.mean([np.mean([np.mean(C[a][s][-1][f'tl_{t}']) for t in T]) for s in S5]):.3f} | {np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in T]) for s in S5]):.4f} | "
                     f"{np.mean([C[a][s][-1].get('heldout_loss') or np.nan for s in S5]):.2f} |")
    lines += ["", "TF by checkpoint (every 4th): " + "; ".join(f"{a} " + " ".join(f"{x:.3f}" for x in cur(a, fs, fi)[::4]) for a in arms), ""]
    gi = draws(lambda si, ii: cur("inst", si, ii)[-1] - cur("inst", si, ii)[0]); inst_learns = np.percentile(gi, 2.5) > 0
    for lab, a in (("M1 (primary)", "mix50"), ("M2", "fade")):
        if a not in arms:
            lines.append(f"- {lab}: {a} incomplete or not run"); continue
        d1 = draws(lambda si, ii: cur(a, si, ii)[-1] - cur("inst", si, ii)[-1]); p1 = cur(a, fs, fi)[-1] - cur("inst", fs, fi)[-1]
        per = VR[a][:, -1] - VR["patex"][:, -1]; d2 = np.array([per[rng.integers(0, 5, 5)].mean() for _ in range(10000)])
        ok = np.percentile(d1, 2.5) > 0 and np.percentile(d2, 2.5) > 0
        lines.append(f"- {lab} {a}: TF minus INST {p1:+.3f} [{np.percentile(d1, 2.5):+.3f}, {np.percentile(d1, 97.5):+.3f}]; VIEWRET minus PATEX {per.mean():+.3f} "
                     f"[{np.percentile(d2, 2.5):+.3f}, {np.percentile(d2, 97.5):+.3f}] (per seed {', '.join(f'{x:+.3f}' for x in per)}; {int((per > 0).sum())} of 5 positive; "
                     f"a 5-seed bound is coarse) -> {'supported' if ok else 'not supported'}")
        dp = draws(lambda si, ii: cur(a, si, ii)[-1] - cur("patex", si, ii)[-1]); lines.append(f"  - TF minus PATEX {cur(a, fs, fi)[-1] - cur('patex', fs, fi)[-1]:+.3f} [{np.percentile(dp, 2.5):+.3f}, {np.percentile(dp, 97.5):+.3f}]")
        if inst_learns:                                                       # addendum 63's E1 rule against INST
            def k_of(si, ii, a=a):
                target = cur("inst", si, ii)[-1]; hi = stable_hit(cur("inst", si, ii), target); hp = stable_hit(cur(a, si, ii), target)
                return 0.0 if hp is None else hi / hp
            kd = draws(k_of, 5000); lines.append(f"  - E1-style k against INST: {k_of(fs, fi):.2f} [{np.percentile(kd, 2.5):.2f}, {np.percentile(kd, 97.5):.2f}]")
        else:
            lines.append("  - E1-style k: undefined (INST's own TF gain does not clear 0)")
    (OUT / "report69.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report71():
    """addendum 71: ABSMIX against addendum 63's INST and PATMIX (the box, same seeds and start): A1 PATMIX minus ABSMIX,
    A2 ABSMIX minus INST, last-step TF; bootstrap over seeds and items (10,000)"""
    from tokenizers import Tokenizer
    C = {}
    for f in sorted(list(OUT.glob("curve63_*_s*.json")) + list(OUT.glob("curve71_*_s*.json")), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); C.setdefault(d["arm"], {})[d["seed"]] = d["curve"]
    T = ("items3", "items7"); rng = np.random.default_rng(71); S5 = [0, 1, 2, 3, 4]
    for a in ("inst", "patmix", "absmix"):
        assert a in C and sorted(C[a]) == S5, f"{a}: seeds {sorted(C.get(a, {}))}"
    arms = [a for a in ("inst", "patex", "patmix", "absmix") if a in C and sorted(C[a]) == S5]
    ncp = len(C["inst"][0]); assert all(len(C[a][s]) == ncp for a in arms for s in S5)
    same0 = all(np.allclose(C[a][s][0]["cos_items3"], C["inst"][s][0]["cos_items3"], atol=1e-4) for a in arms for s in S5)
    assert same0, "ABSMIX does not start from addendum 63's weights"                   # review: the README says checked
    TF = {a: {t: np.array([[C[a][s][k][f"tf_{t}"] for k in range(ncp)] for s in S5], dtype=float) for t in T} for a in arms}
    cur = lambda a, si, ii: np.mean([TF[a][t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    fs = np.arange(5); fi = {t: np.arange(TF["inst"][t].shape[2]) for t in T}
    ck = json.load(open(OUT / "check71.json")); rec = [C["absmix"][s][-1].get("c_seqs") for s in S5]
    assert rec == [x[2] for x in ck["absmix"]], f"replayed batches differ from the run: {rec} vs {[x[2] for x in ck['absmix']]}"

    def draws(fn, n=10000):
        out = []
        for _ in range(n):
            si = rng.integers(0, 5, 5); ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; out.append(fn(si, ii))
        return np.array(out, dtype=float)
    lines = ["# Pattern content or any abstract sentence? (addendum 71)", "", f"- seeds {S5}; identical step 0 across arms: {'yes' if same0 else 'NO'}", "",
             "| arm | TF step 0 -> last (items3 / items7) | TF gain over its own step 0 [95% CI] | TL last | VIEWRET last | ALIGN last | held-out loss last |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        c = cur(a, fs, fi); g = draws(lambda si, ii: cur(a, si, ii)[-1] - cur(a, si, ii)[0], 5000)
        lines.append(f"| {a} | {c[0]:.3f} -> {c[-1]:.3f} ({TF[a]['items3'][:, -1].mean():.3f} / {TF[a]['items7'][:, -1].mean():.3f}) | {c[-1] - c[0]:+.3f} "
                     f"[{np.percentile(g, 2.5):+.3f}, {np.percentile(g, 97.5):+.3f}] | {np.mean([np.mean([np.mean(C[a][s][-1][f'tl_{t}']) for t in T]) for s in S5]):.3f} | "
                     f"{np.mean([C[a][s][-1]['viewret'] for s in S5]):.3f} | {np.mean([np.mean([C[a][s][-1][f'align_{t}'] for t in T]) for s in S5]):.4f} | "
                     f"{np.mean([C[a][s][-1].get('heldout_loss') or np.nan for s in S5]):.2f} |")
    lines += ["", "- batches (replayed; ABSMIX's replay equals its recorded counts): sequences per step PATMIX " + " ".join(f"{x[0]:.1f}" for x in ck["patmix"])
              + ", ABSMIX " + " ".join(f"{x[0]:.1f}" for x in ck["absmix"]) + "; statement-token share PATMIX " + " ".join(f"{x[1]:.3f}" for x in ck["patmix"])
              + ", ABSMIX " + " ".join(f"{x[1]:.3f}" for x in ck["absmix"]), ""]
    sup = {}
    for lab, a, b in (("A1 (primary): PATMIX minus ABSMIX", "patmix", "absmix"), ("A2: ABSMIX minus INST", "absmix", "inst"), ("PATMIX minus INST", "patmix", "inst")):
        d = draws(lambda si, ii: cur(a, si, ii)[-1] - cur(b, si, ii)[-1]); pt = cur(a, fs, fi)[-1] - cur(b, fs, fi)[-1]
        per = ", ".join(f"{np.mean([TF[a][t][s, -1].mean() - TF[b][t][s, -1].mean() for t in T]):+.3f}" for s in S5)
        sup[lab[:2]] = np.percentile(d, 2.5) > 0
        lines.append(f"- {lab}, last-step TF: {pt:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] (per seed {per})"
                     + (f" -> {'supported' if np.percentile(d, 2.5) > 0 else 'not supported'}" if lab.startswith("A") else ""))
    lines.append("- reading: " + ("the pattern content carries the gain" if sup["A1"] else "a generic regularizer (abstract sentences alone help)" if sup["A2"] else "undecided at this power"))
    btok = Tokenizer.from_file(str(OUT / "tok.json")); P = {q["id"]: q["pattern"] for q in json.load(open(OUT / "patterns.json"))}; AB = json.load(open(OUT / "abstract.json"))["by_pattern"]
    ln = lambda xs: np.mean([len(btok.encode("Pattern: " + x).ids) for x in xs])  # noqa: E731
    lines += ["", f"- mean tokens with the prefix: patterns {ln([P[k] for k in AB]):.1f}, abstract sentences {ln(list(AB.values())):.1f} ({len(AB)} training patterns)", "",
              "Ten abstract sentences (random):"] + [f"- {AB[k]}" for k in rng.choice(sorted(AB), 10, replace=False)]
    (OUT / "report71.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report70():
    """addendum 70 (amendment: PATMIX primary): SC1 at L, last-step TF, PATMIX minus INST (SC1b PATEX); SC2 each difference's
    slope over log2 parameters, one item draw shared by the sizes, seeds drawn per size; SC3 the grouping cost at L, VIEWRET
    PATEX minus INST (SC3b PATMIX), seeds and held-out patterns resampled; held-out loss; bootstrap 10,000"""
    C, NP, FP = {}, {}, set()
    for f in sorted(OUT.glob("curve70[SML]_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); z = f.stem.split("_")[0][-1]
        assert d["data"] == "_big" and list(d["size"]) == list(SIZES[z]), f"{f.name}: data {d['data']!r}, size {d['size']}"   # every size on the new data
        C.setdefault(z, {}).setdefault(d["arm"], {})[d["seed"]] = d["curve"]; NP.setdefault(z, set()).add(d["params"]); FP.add(tuple(sorted(d["files_md5"].items())))
    Z = ["S", "M", "L"]; assert sorted(C) == sorted(Z), f"sizes present: {sorted(C)} (copy every machine's curve70 files first)"
    assert len(FP) == 1, f"tokenizer, evaluation or data files differ between runs: {FP}"
    T = ("items3", "items7"); rng = np.random.default_rng(70); ARMS = ("inst", "patmix", "fade"); SD = sorted(C["L"]["inst"])     # amendment 2: FADE replaces PATEX
    for z in Z:
        for a in ARMS:
            assert sorted(C[z].get(a, {})) == SD and len(SD) >= 3, f"{z} {a}: seeds {sorted(C[z].get(a, {}))}"
        assert len(NP[z]) == 1, f"{z}: parameter counts {NP[z]}"
    ncp = {z: len(C[z]["inst"][SD[0]]) for z in Z}; assert all(len(C[z][a][s]) == ncp[z] for z in Z for a in ARMS for s in SD); ns = len(SD)
    TF = {(z, a): {t: np.array([[C[z][a][s][k][f"tf_{t}"] for k in range(ncp[z])] for s in SD], dtype=float) for t in T} for z in Z for a in ARMS}
    VH = {(z, a): np.array([C[z][a][s][-1]["viewret_hits"] for s in SD], dtype=float) for z in Z for a in ARMS}       # seeds x held-out patterns
    cur = lambda z, a, si, ii: np.mean([TF[z, a][t][si][:, :, ii[t]].mean(axis=(0, 2)) for t in T], 0)  # noqa: E731
    fs = np.arange(ns); fi = {t: np.arange(TF["S", "inst"][t].shape[2]) for t in T}; nv = VH["S", "inst"].shape[1]; lp = {z: math.log2(next(iter(NP[z]))) for z in Z}

    def items():
        return {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}

    def seeds():
        return rng.integers(0, ns, ns)

    def stable_hit(c, target):
        return next((j for j in range(1, len(c)) if (c[j:] >= target).all()), None)
    lines = ["# The scaling sweep (addendum 70)", "", f"- sizes {', '.join(f'{z} {next(iter(NP[z])) / 1e6:.1f}M parameters' for z in Z)}; seeds {SD}; data _big; files {dict(next(iter(FP)))}",
             f"- training tokens per parameter: " + ", ".join(f"{z} {STEPS * TOKBUDGET / next(iter(NP[z])):.1f}" for z in Z) + " (fixed tokens; L sees the fewest per parameter)",
             f"- the bootstrap over {ns} seeds understates seed variance (few seeds)"]
    for z in Z:
        same0 = all(np.allclose(C[z][a][s][0]["cos_items3"], C[z]["inst"][s][0]["cos_items3"], atol=1e-4) for a in ARMS for s in SD)
        lines += ["", f"## {z}: identical step 0 across arms {'yes' if same0 else 'NO'}", "",
                  "| arm | TF step 0 -> last (items3 / items7) | TF gain over step 0 [95% CI] | TL last | VIEWRET last (per seed) | ALIGN last | held-out loss last |", "|---|---|---|---|---|---|---|"]
        gains = {}
        for a in ARMS:
            c = cur(z, a, fs, fi); g = []
            for _ in range(5000):
                si, ii = seeds(), items(); cc = cur(z, a, si, ii); g.append(cc[-1] - cc[0])
            gains[a] = np.percentile(g, 2.5)
            lines.append(f"| {a} | {c[0]:.3f} -> {c[-1]:.3f} ({TF[z, a]['items3'][:, -1].mean():.3f} / {TF[z, a]['items7'][:, -1].mean():.3f}) | {c[-1] - c[0]:+.3f} [{np.percentile(g, 2.5):+.3f}, "
                         f"{np.percentile(g, 97.5):+.3f}] | {np.mean([np.mean([np.mean(C[z][a][s][-1][f'tl_{t}']) for t in T]) for s in SD]):.3f} | {VH[z, a].mean():.3f} "
                         f"({', '.join(f'{x:.2f}' for x in VH[z, a].mean(1))}) | {np.mean([np.mean([C[z][a][s][-1][f'align_{t}'] for t in T]) for s in SD]):.4f} | "
                         f"{np.mean([C[z][a][s][-1].get('heldout_loss') or np.nan for s in SD]):.2f} |")
        lines.append("")
        for a in ("patmix", "fade"):
            d, dv = [], []
            for _ in range(10000):
                si, ii = seeds(), items(); jj = rng.integers(0, nv, nv); d.append(cur(z, a, si, ii)[-1] - cur(z, "inst", si, ii)[-1])
                dv.append(VH[z, a][si][:, jj].mean() - VH[z, "inst"][si][:, jj].mean())
            pt = cur(z, a, fs, fi)[-1] - cur(z, "inst", fs, fi)[-1]; lo, hi = np.percentile(d, [2.5, 97.5]); vlo, vhi = np.percentile(dv, [2.5, 97.5])
            dm = []                                                           # amendment (addendum 63's curves rise, then fall): mean TF after step 0, secondary
            for _ in range(5000):
                si, ii = seeds(), items(); dm.append(cur(z, a, si, ii)[1:].mean() - cur(z, "inst", si, ii)[1:].mean())
            per = ", ".join(f"{np.mean([TF[z, a][t][k, -1].mean() - TF[z, 'inst'][t][k, -1].mean() for t in T]):+.3f}" for k in range(ns))
            tag = ("SC1 (primary): " if a == "patmix" else "SC1b: ") if z == "L" else ""
            lines.append(f"- {tag}{z} {a.upper()} minus INST, last-step TF: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] (per seed {per})" + (f" -> {'supported' if lo > 0 else 'not supported'}" if z == "L" else ""))
            lines.append(f"  - mean TF after step 0 (secondary): {cur(z, a, fs, fi)[1:].mean() - cur(z, 'inst', fs, fi)[1:].mean():+.3f} [{np.percentile(dm, 2.5):+.3f}, {np.percentile(dm, 97.5):+.3f}]")
            if gains["inst"] > 0:                                                 # addendum 63's rule: k only if INST learns
                ks = []
                for _ in range(5000):
                    si, ii = seeds(), items(); ci = cur(z, "inst", si, ii); hp = stable_hit(cur(z, a, si, ii), ci[-1]); ks.append(0.0 if hp is None else stable_hit(ci, ci[-1]) / hp)
                ci = cur(z, "inst", fs, fi); hp = stable_hit(cur(z, a, fs, fi), ci[-1])
                lines.append(f"  - E1-style k against INST: {0.0 if hp is None else stable_hit(ci, ci[-1]) / hp:.2f} [{np.percentile(ks, 2.5):.2f}, {np.percentile(ks, 97.5):.2f}]")
            else:
                lines.append("  - E1-style k: undefined (INST's own TF gain does not clear 0)")
            hl = np.mean([(C[z][a][s][-1].get("heldout_loss") or np.nan) - (C[z]["inst"][s][-1].get("heldout_loss") or np.nan) for s in SD])
            lines.append(f"  - VIEWRET minus INST {VH[z, a].mean() - VH[z, 'inst'].mean():+.3f} [{vlo:+.3f}, {vhi:+.3f}]"
                         + (f" -> {'SC3' if a == 'fade' else 'SC3b'}: grouping cost {'gone' if vhi >= 0 else 'remains'}" if z == "L" else "") + f"; held-out loss minus INST {hl:+.2f}")
    lines.append("")
    x = np.array([lp[z] for z in Z])
    for a, lab in (("patmix", "SC2 (PATMIX)"), ("fade", "SC2b (FADE)")):                # one item draw for all sizes (the same items), seeds per size
        pt = np.polyfit(x, [cur(z, a, fs, fi)[-1] - cur(z, "inst", fs, fi)[-1] for z in Z], 1)[0]; bs = []
        for _ in range(10000):
            ii = items(); y = []
            for z in Z:
                si = seeds(); y.append(cur(z, a, si, ii)[-1] - cur(z, "inst", si, ii)[-1])
            bs.append(np.polyfit(x, y, 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"- {lab}: TF difference per doubling of parameters {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {'grows' if lo > 0 else 'shrinks' if hi < 0 else 'holds'}")
    (OUT / "report70.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

def report73():
    """addendum 73: the clash between views and masked words by size. cost = last-step TF, MLM_JEPA minus MLM; gain = TL;
    conflict = MLM_JEPA's gradient cosine (encoder layers). R gate at S, K1 cost(L) - cost(S), K2 both kept at L, K3 slope
    of the cosine; bootstrap 10,000 (items drawn once and shared by the sizes, seeds per size)"""
    C, NP, FP = {}, {}, set()
    for f in sorted(OUT.glob("curve73[SML]_*_s*.json"), key=lambda f: int(f.stem.split("_s")[-1])):
        d = json.load(open(f)); z = f.stem.split("_")[0][-1]
        assert d["data"] == "_big" and list(d["size"]) == list(SIZES[z]), f"{f.name}: data {d['data']!r}, size {d['size']}"
        C.setdefault(z, {}).setdefault(d["arm"], {})[d["seed"]] = d["curve"]; NP.setdefault(z, set()).add(d["params"]); FP.add(tuple(sorted(d["files_md5"].items())))
    Z = ["S", "M", "L"]; assert sorted(C) == sorted(Z), f"sizes present: {sorted(C)} (copy every machine's curve73 files first)"
    assert len(FP) == 1, f"tokenizer, evaluation or data files differ between runs: {FP}"
    T = ("items3", "items7"); rng = np.random.default_rng(73); ARMS = ("mlm", "mlm_jepa"); SD = [0, 1, 2]
    for z in Z:
        for a in ARMS:
            assert sorted(C[z].get(a, {})) == SD, f"{z} {a}: seeds {sorted(C[z].get(a, {}))}"
        assert len(NP[z]) == 1 and all("gcos_body" in C[z]["mlm_jepa"][s][1] for s in SD), f"{z}: parameter counts or gradient logging"
    ncp = {z: len(C[z]["mlm"][0]) for z in Z}; assert all(len(C[z][a][s]) == ncp[z] for z in Z for a in ARMS for s in SD)
    A = {(z, a, m): {t: np.array([[C[z][a][s][k][f"{m}_{t}"] for k in range(ncp[z])] for s in SD], dtype=float) for t in T} for z in Z for a in ARMS for m in ("tf", "tl")}
    last = lambda z, a, m, si, ii: np.mean([A[z, a, m][t][si][:, -1][:, ii[t]].mean() for t in T])  # noqa: E731
    G = {z: np.array([[C[z]["mlm_jepa"][s][k].get("gcos_body", np.nan) for k in range(1, ncp[z] - 1)] for s in SD]) for z in Z}   # step 6000 has no gradient
    fs = np.arange(3); fi = {t: np.arange(A["S", "mlm", "tf"][t].shape[2]) for t in T}; lp = np.array([math.log2(next(iter(NP[z]))) for z in Z])
    cost = lambda z, si, ii: last(z, "mlm_jepa", "tf", si, ii) - last(z, "mlm", "tf", si, ii)  # noqa: E731
    gain = lambda z, si, ii: last(z, "mlm_jepa", "tl", si, ii) - last(z, "mlm", "tl", si, ii)  # noqa: E731
    D = {k: [] for k in ("costS", "costM", "costL", "gainS", "gainM", "gainL", "k1", "cslope", "gslope")}
    for _ in range(10000):
        ii = {t: rng.integers(0, len(fi[t]), len(fi[t])) for t in T}; si = {z: rng.integers(0, 3, 3) for z in Z}
        c = {z: cost(z, si[z], ii) for z in Z}; gcs = []
        for z in Z:
            D["cost" + z].append(c[z]); D["gain" + z].append(gain(z, si[z], ii))
            ki = rng.integers(0, G[z].shape[1], G[z].shape[1]); gcs.append(np.nanmean(G[z][si[z]][:, ki]))
        D["k1"].append(c["L"] - c["S"]); D["cslope"].append(np.polyfit(lp, [c[z] for z in Z], 1)[0]); D["gslope"].append(np.polyfit(lp, gcs, 1)[0])
    ci = lambda k: np.percentile(D[k], [2.5, 97.5])  # noqa: E731
    pt = {z: (cost(z, fs, fi), gain(z, fs, fi)) for z in Z}
    lines = ["# Does the clash shrink as the model grows? (addendum 73)", "", f"- sizes {', '.join(f'{z} {next(iter(NP[z])) / 1e6:.1f}M' for z in Z)}; seeds {SD}; data _big; files {dict(next(iter(FP)))}",
             "- tokens per step are fixed: L sees the fewest per parameter", "",
             "| size | arm | TF step 0 -> last | TL step 0 -> last | VIEWRET last | ALIGN last | gradient cosine, encoder layers (per seed) |", "|---|---|---|---|---|---|---|"]
    for z in Z:
        same0 = all(np.allclose(C[z]["mlm_jepa"][s][0]["cos_items3"], C[z]["mlm"][s][0]["cos_items3"], atol=1e-4) for s in SD)
        for a in ARMS:
            tf = [np.mean([A[z, a, "tf"][t][:, k].mean() for t in T]) for k in (0, -1)]; tl = [np.mean([A[z, a, "tl"][t][:, k].mean() for t in T]) for k in (0, -1)]
            g = f"{np.nanmean(G[z]):+.3f} ({', '.join(f'{v:+.3f}' for v in np.nanmean(G[z], 1))})" if a == "mlm_jepa" else ""
            lines.append(f"| {z}{'' if same0 else ' (step 0 differs!)'} | {a} | {tf[0]:.3f} -> {tf[1]:.3f} | {tl[0]:.3f} -> {tl[1]:.3f} | {np.mean([C[z][a][s][-1]['viewret'] for s in SD]):.3f} | "
                         f"{np.mean([np.mean([C[z][a][s][-1][f'align_{t}'] for t in T]) for s in SD]):.4f} | {g} |")
    lines.append("")
    for z in Z:
        lines.append(f"- {z}: cost (TF, MLM_JEPA minus MLM) {pt[z][0]:+.3f} [{ci('cost' + z)[0]:+.3f}, {ci('cost' + z)[1]:+.3f}]; gain (TL) {pt[z][1]:+.3f} [{ci('gain' + z)[0]:+.3f}, {ci('gain' + z)[1]:+.3f}]")
    r = ci("costS")[1] < 0; lines += ["", f"- R (gate): the clash replicates at S: {'yes' if r else 'NO (no clash to relieve; K1-K3 descriptive)'}"]
    k1 = ci("k1"); lines.append(f"- K1 (primary): cost at L minus cost at S {pt['L'][0] - pt['S'][0]:+.3f} [{k1[0]:+.3f}, {k1[1]:+.3f}]" + (f" -> {'supported' if k1[0] > 0 else 'not supported'}" if r else ""))
    k2 = ci("costL")[1] >= 0 and ci("gainL")[0] > 0; lines.append(f"- K2: at L the cost's CI reaches 0 and the gain's lies above 0" + (f" -> {'supported' if k2 else 'not supported'}" if r else " (descriptive)"))
    gp = np.polyfit(lp, [np.nanmean(G[z]) for z in Z], 1)[0]; gs = ci("gslope")
    lines.append(f"- K3: gradient cosine per doubling of parameters {gp:+.4f} [{gs[0]:+.4f}, {gs[1]:+.4f}]" + (f" -> {'supported' if gs[0] > 0 else 'not supported'}" if r else " (descriptive)")
                 + " (checkpoints of one run are not independent: this CI runs narrow)")
    pl = {z: np.nanmean([[np.nanmean([C[z]["mlm_jepa"][s][k].get(f"gcos_{j}", np.nan) for j in range(SIZES[z][1])]) for k in range(1, ncp[z] - 1)] for s in SD]) for z in Z}   # review
    lines.append("- mean of the per-layer cosines (each layer weighted equally): " + ", ".join(f"{z} {pl[z]:+.3f}" for z in Z))
    cs = ci("cslope"); lines.append(f"- the cost's slope per doubling of parameters {np.polyfit(lp, [pt[z][0] for z in Z], 1)[0]:+.4f} [{cs[0]:+.4f}, {cs[1]:+.4f}]")
    blocks = sorted({k[5:] for k in C["L"]["mlm_jepa"][0][1] if k.startswith("gcos_")}, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0, b))
    for z in Z:
        lines.append(f"- {z} per block (MLM_JEPA, mean after step 0): " + ", ".join(f"{b} {np.nanmean([[C[z]['mlm_jepa'][s][k].get('gcos_' + b, np.nan) for k in range(1, ncp[z] - 1)] for s in SD]):+.3f}" for b in blocks))
    (OUT / "report73.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def report75():
    """addendum 75: the four placements of the views loss against MLM (run here) and MLM_JEPA (addendum 67's Mac runs);
    TF and TL at the top and middle pooled states from the saved last-step weights; bootstrap over items and seeds"""
    import torch
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(OUT / "tok.json")); M = json.load(open(OUT / "evalsets.json")); T = ("items3", "items7"); SD = [0, 1, 2]
    files = {"mlm": "curve75_mlm", "mlm_jepa": "curve67_mlm_jepa", **{a: f"curve75_{a}" for a in ("mlm_jepa_mid", "mlm_jepa_top", "mlm_jepa_w01", "mlm_jepa_curr")}}
    if any((OUT / f"curve75_mlm_jepa_low_s{k}.json").exists() for k in (0, 1, 2)):   # amendment 2 (review: any seed, so a missing one fails loudly)
        files["mlm_jepa_low"] = "curve75_mlm_jepa_low"
    R, C = {}, {}
    for a, pre in files.items():
        for s in SD:
            d = json.load(open(OUT / f"{pre}_s{s}.json")); assert d["arm"] == a and d["device"] == "mps", (pre, d["arm"], d["device"]); C[a, s] = d["curve"]
            net = build(torch, causal=False); net.load_state_dict(torch.load(OUT / f"{pre}_s{s}.pt", map_location="cpu", weights_only=True)); net.eval()
            for t in T:
                th = M[t]["three_way"]; texts = [x for it in th for x in (it["anchor"], it["true"], it["lure"], it["foil"])]; Z = {"top": [], "mid": [], "low": []}
                for j in range(0, len(texts), 128):
                    x, pad = encode(tok, texts[j:j + 128]); x, pad = torch.tensor(x), torch.tensor(pad)
                    with torch.no_grad():
                        Z["top"].append(net.pooled(x, pad).numpy()); Z["mid"].append(net.pooled_mid(x, pad).numpy()); Z["low"].append(net.pooled_mid(x, pad, upto=LAYERS_N // 3).numpy())
                for dep in ("top", "mid", "low"):
                    q = np.concatenate(Z[dep]).astype(np.float64); q = (q / np.linalg.norm(q, axis=1, keepdims=True)).reshape(len(th), 4, -1)
                    an, tr_, lu, fo = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                    R[a, s, t, dep, "tf"] = ((an * tr_).sum(1) > (an * fo).sum(1)).astype(float); R[a, s, t, dep, "tl"] = ((an * tr_).sum(1) > (an * lu).sum(1)).astype(float)
            agree = np.mean([R[a, s, t, "top", "tf"].mean() - np.mean(C[a, s][-1][f"tf_{t}"]) for t in T])
            assert abs(agree) < 0.02, f"{pre} s{s}: top TF from the weights differs from the run's own last step by {agree:+.3f}"
    same0 = all(np.allclose(C[a, s][0]["cos_items3"], C["mlm", s][0]["cos_items3"], atol=1e-4) for a in files for s in SD)
    assert same0, "the six arms do not start from the same weights (a torch change between sessions?)"          # review
    rng = np.random.default_rng(75); n = {t: len(R["mlm", 0, t, "top", "tf"]) for t in T}
    val = lambda a, dep, m, si, ii: np.mean([np.mean([R[a, s, t, dep, m][ii[t]].mean() for s in si]) for t in T])  # noqa: E731
    full = (SD, {t: np.arange(n[t]) for t in T})
    draws = [([SD[k] for k in rng.integers(0, 3, 3)], {t: rng.integers(0, n[t], n[t]) for t in T}) for _ in range(10000)]
    lines = ["# Where should the pattern objective act? (addendum 75)", "", f"- seeds {SD}; the Mac; identical step 0 across the arms: {'yes' if same0 else 'NO'}", "",
             "| arm | TF top | TL top | TF middle | TL middle | TF low | TL low | VIEWRET last | ALIGN last |", "|---|---|---|---|---|---|---|---|---|"]
    for a in files:
        lines.append(f"| {a} | " + " | ".join(f"{val(a, dep, m, *full):.3f}" for dep, m in (("top", "tf"), ("top", "tl"), ("mid", "tf"), ("mid", "tl"), ("low", "tf"), ("low", "tl")))
                     + f" | {np.mean([C[a, s][-1]['viewret'] for s in SD]):.3f} | {np.mean([np.mean([C[a, s][-1][f'align_{t}'] for t in T]) for s in SD]):.4f} |")
    lines.append("")

    def diff(a, b, dep, m, b_dep=None):
        b_dep = b_dep or dep; pt = val(a, dep, m, *full) - val(b, b_dep, m, *full)
        bs = np.array([val(a, dep, m, si, ii) - val(b, b_dep, m, si, ii) for si, ii in draws]); return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    rows75 = [("mlm_jepa_mid", "mid", "F1 (primary)"), ("mlm_jepa_top", "top", "F2"), ("mlm_jepa_w01", "top", "F2"), ("mlm_jepa_curr", "top", "F2"), ("mlm_jepa", "top", "reference")]
    if "mlm_jepa_low" in files:
        rows75.append(("mlm_jepa_low", "low", "amendment 2, reported"))
    for a, tl_dep, lab in rows75:
        tf = diff(a, "mlm", "top", "tf"); tl = diff(a, "mlm", tl_dep, "tl"); ok = tf[1] > -0.03 and tl[1] > 0
        lines.append(f"- {lab} {a}: top TF minus MLM {tf[0]:+.3f} [{tf[1]:+.3f}, {tf[2]:+.3f}]; TL ({tl_dep}) minus MLM {tl[0]:+.3f} [{tl[1]:+.3f}, {tl[2]:+.3f}] -> "
                     + ("both kept" if ok else "not both") + ("" if lab != "reference" else " (addendum 65's clash, here on the Mac)"))
    f3 = diff("mlm_jepa_mid", "mlm_jepa_top", "top", "tf"); lines.append(f"- F3 (above vs below): top TF, MID minus TOP {f3[0]:+.3f} [{f3[1]:+.3f}, {f3[2]:+.3f}]")
    (OUT / "report75.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    {"evalset": evalset, "patterns": patterns, "views": views, "patterns_big": lambda: patterns(8000, True), "views_big": lambda: views(True), "tok": tok_train, "report": report, "report61": report61, "report63": report63, "report65": report65, "report66": report66, "report67": report67, "report68": report68, "report69": report69, "report70": report70, "report71": report71, "report73": report73, "report75": report75, "abstract": abstract, "abstract_rematch": lambda: abstract_rematch(float(sys.argv[2])), "abstract_pilot": lambda: abstract(True), "check71": check71,
     "train": lambda: train(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "curve")}[sys.argv[1]]()
