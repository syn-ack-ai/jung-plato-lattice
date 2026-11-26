"""A general English base for the toy (addendum 220 follow-up): public-domain Gutenberg books (sedthh/gutenberg_english, smallworld/gutenberg) in the toy's
bpe8k tokens. Books by Jung (text ids 48225, 65903, 66041: early translations that overlap the Collected Works) are left out, so the held-out Jung works stay
unseen. Books are drawn in a seeded random order; the first 200 drawn are a held-out English set, the rest fill the training stream up to TARGET tokens.
python smallworld/gut_tokens.py [TARGET_M] [OUT] [CLEAN]   -> smallworld/OUT/{train,held}.bin (uint16), meta.json
OUT defaults to gut_tok (300M, cleaning 1); addendum 221 rung 2b's base uses gut_tok2 (1,500M, cleaning 2)."""
import glob, json, re, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
HERE = Path(__file__).parent; OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "gut_tok"); OUT.mkdir(exist_ok=True); JUNG = {48225, 65903, 66041}
CLEAN = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TARGET = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 300_000_000; HELD = 200
tk = Tokenizer.from_file(str(HERE.parent / "transfer" / "tok" / "bpe8k.json")); sep = tk.encode("\n\n").ids


def clean(t):
    """Windows line ends to \n; the books' hard wrapping undone: lines inside a paragraph joined by a space, blank lines kept as paragraph breaks, runs of spaces cut to one"""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    paras = [q for q in (re.sub(r"[ \t]+", " ", " ".join(l.strip() for l in p.split("\n"))).strip() for p in re.split(r"\n\s*\n", t)) if q]
    if CLEAN >= 2 and paras:                                     # cleaning 2: a double-spaced book (blank lines between printed lines) is rejoined
        n = sorted(len(q.split()) for q in paras); med = n[len(n) // 2]
        if med < 14:                                             # pieces end a paragraph when they end a sentence and are shorter than a full line
            out, cur = [], []
            for q in paras:
                cur.append(q)
                if re.search(r"[.!?][\"'\u201d\u2019)]*$", q) and len(q.split()) < med:
                    out.append(" ".join(cur)); cur = []
            paras = out + ([" ".join(cur)] if cur else [])
    return "\n\n".join(paras)
books = []
for f in sorted(glob.glob(str(HERE / "gutenberg" / "*.parquet"))):
    t = pq.read_table(f, columns=["METADATA"])
    for i, m in enumerate(t.column("METADATA").to_pylist()):
        md = json.loads(m)
        if md.get("language") == "en" and int(md.get("text_id", -1)) not in JUNG:
            books.append((f, i))
order = np.random.default_rng(220).permutation(len(books)); t0 = time.time(); held, train, ntr, used = [], [], 0, {"held": [], "train": []}
# group the draws by file so each parquet is read once per pass, keeping the draw order inside the output
want = [books[j] for j in order]
held_set = set(want[:HELD]); chosen = list(want[:HELD]); k = HELD; est = 0
while est < TARGET * 1.15 and k < len(want):                    # over-draw a little; tokens per book are only known after encoding
    chosen.append(want[k]); k += 1; est += 90_000
by_file = {}
for b in chosen:
    by_file.setdefault(b[0], []).append(b[1])
toks = {}
for f, idx in by_file.items():
    col = pq.read_table(f, columns=["TEXT"]).column("TEXT"); enc = tk.encode_batch([clean(col[i].as_py()) for i in idx])
    for i, e in zip(idx, enc):
        toks[(f, i)] = np.array(e.ids + sep, dtype=np.uint16)          # compact: 2 bytes a token
    print(f"{Path(f).name}: {len(idx)} books, {time.time() - t0:.0f} s", flush=True)
for b in chosen:
    ids = toks.pop(b)
    if b in held_set:
        held.append(ids); used["held"].append(b[1])
    elif ntr < TARGET:
        train.append(ids); ntr += len(ids); used["train"].append(b[1])
train, held = np.concatenate(train), np.concatenate(held); train.tofile(OUT / "train.bin"); held.tofile(OUT / "held.bin")
meta = {"train_tokens": len(train), "held_tokens": len(held), "train_books": len(used["train"]), "held_books": HELD, "excluded_jung_ids": sorted(JUNG), "seed": 220, "tokenizer": "bpe8k", "cleaning": CLEAN, "s": round(time.time() - t0)}
json.dump(meta, open(OUT / "meta.json", "w"), indent=1); print(meta, flush=True)
