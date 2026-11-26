"""Offline 5-Family Platonic Consensus Builder (strictly 1 model loaded in RAM at a time).
Extracts aligned clause/sentence boundary hidden states across 5 independent open LLM families:
  - 4 MLX ~30%-depth teachers: Google Gemma 4 E4B, Alibaba Qwen3.5 9B, Meta Llama 3.1 8B, Mistral 7B v0.3
  - 5th teacher: Thinking Machines Inkling-2-Mini (looped architecture, local Metal llama-server --embeddings --pooling none)
  - 2 held-out target families (NEVER in consensus construction): Microsoft Phi-4-mini (3.8B) and AllenAI OLMo 3 7B
Computes 10-iteration Generalized Procrustes Analysis (GPA) 256-dim Platonic Consensus (C4 & C5) and builds public
myth_train.bin + compact train/eval target packages for exp228.py / exp229.py.

Usage:
  python fluent/plato_data.py test                        unit tests (GPA, boundary align, stratified group CV)
  python fluent/plato_data.py corpus                      build myth_train.bin + passages.jsonl + eval_passages.jsonl
  python fluent/plato_data.py extract [MODEL_KEY...]      extract states (strictly 1 model at a time)
  python fluent/plato_data.py consensus [auto|c5|c4]      fit GPA strictly on train, eval held-out Phi-4-mini & OLMo 3

Strictly public text only (excludes 200 held-out Gutenberg books, Jung IDs {48225,65903,66041}, and all eval 12-grams)."""
import gc, glob, hashlib, json, math, os, random, re, socket, subprocess, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO = HERE / "plato"
MID = HERE / "mid"
COMP = HERE / "comp"
GUT_DIR = ARCH / "smallworld" / "gutenberg"
TOK_PATH = ARCH / "transfer" / "tok" / "bpe8k.json"

PLATO.mkdir(parents=True, exist_ok=True)

JUNG_IDS = {48225, 65903, 66041}
MYTH_PAT = re.compile(
    r"\b(myth|myths|mythology|folklore|folk[- ]?tale|fairy tale|fable|fables|legend|legends|"
    r"aesop|grimm|homer|iliad|odyssey|arabian nights|edda|arthur|bulfinch|andersen|saga|sagas|"
    r"metamorphoses|aeneid)\b",
    re.I,
)

# 5 independent consensus families + 2 blind held-out families (Phi-4-mini & OLMo-3-7B; NEVER in consensus or Procrustes fit!)
MLX_CONSENSUS_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
CONSENSUS_KEYS = MLX_CONSENSUS_KEYS + ("inkling2_mini",)
BLIND_KEYS = ("phi4_mini", "olmo3_7b")
ALL_KEYS = CONSENSUS_KEYS + BLIND_KEYS

MODEL_SPECS = {
    "gemma4_e4b":    dict(backend="mlx", path=os.environ.get("GEMMA4_MLX_MODEL", "mlx-community/gemma-4-E4B-it-bf16"), depth_frac=0.30),
    "qwen35_9b":     dict(backend="mlx", path=os.environ.get("QWEN35_MLX_MODEL", "lmstudio-community/Qwen3.5-9B-MLX-8bit"), depth_frac=0.30),
    "llama31_8b":    dict(backend="mlx", path="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", depth_frac=0.30),
    "mistral7b":     dict(backend="mlx", path="mlx-community/Mistral-7B-Instruct-v0.3-4bit", depth_frac=0.30),
    "inkling2_mini": dict(backend="llamacpp", path=os.environ.get("INKLING_GGUF_PATH", "Inkling-2-Mini-IQ4_NL.gguf"), port=8179),
    "phi4_mini":     dict(backend="mlx", path="mlx-community/Phi-4-mini-instruct-4bit", depth_frac=0.30),
    "olmo3_7b":      dict(backend="mlx", path="mlx-community/Olmo-3-7B-Instruct-4bit", depth_frac=0.30),
}


def load_bpe8k():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def _words(t):
    return re.findall(r"[a-z0-9']+", t.lower().replace("\u2019", "'"))


def _grams12(t):
    w = _words(t)
    return {tuple(w[i:i + 12]) for i in range(len(w) - 11)}


def clean_gut(t):
    """exact cleaning-2 normalization from smallworld/gut_tokens.py"""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    paras = [q for q in (re.sub(r"[ \t]+", " ", " ".join(l.strip() for l in p.split("\n"))).strip() for p in re.split(r"\n\s*\n", t)) if q]
    if paras:
        n = sorted(len(q.split()) for q in paras)
        med = n[len(n) // 2]
        if med < 14:
            out, cur = [], []
            for q in paras:
                cur.append(q)
                if re.search(r"[.!?][\"'\u201d\u2019)]*$", q) and len(q.split()) < med:
                    out.append(" ".join(cur))
                    cur = []
            paras = out + ([" ".join(cur)] if cur else [])
    return "\n\n".join(paras)


def segment_passage(text, n_pts=6):
    """Split text into up to n_pts non-empty chunks ending at natural sentence/clause boundaries.
    Guarantees ''.join(segs) == text."""
    text = text.strip()
    if not text:
        return []
    cands = [m.end() for m in re.finditer(r"(?:[.!?;:]|\n)\s+", text)]
    if len(cands) < n_pts:
        cands = sorted(set(cands) | {m.end() for m in re.finditer(r",\s+", text)})
    L = len(text)
    if not cands:
        cands = [m.end() for m in re.finditer(r"\s+", text)]
    chosen = []
    for k in range(1, n_pts):
        tgt = int(round(L * k / n_pts))
        best = min(cands, key=lambda c: abs(c - tgt)) if cands else tgt
        if 15 <= best <= L - 15 and (not chosen or best - chosen[-1] >= 15):
            chosen.append(best)
    chosen.append(L)
    segs = []
    prev = 0
    for pos in chosen:
        if pos > prev:
            segs.append(text[prev:pos])
            prev = pos
    assert "".join(segs) == text
    return segs


def boundary_token_indices(encode_fn, segs):
    """Return (full_ids, pt_indices) where each pt_idx is the exact last token of segs[0..j], strictly increasing."""
    full_text = "".join(segs)
    full_ids = encode_fn(full_text)
    if len(full_ids) < 2:
        return full_ids, []
    pts = []
    acc = ""
    for j, s in enumerate(segs):
        acc += s
        if j == len(segs) - 1:
            idx = len(full_ids) - 1
        else:
            p_ids = encode_fn(acc.rstrip())
            idx = max(1, min(len(full_ids) - 1, len(p_ids) - 1))
        if not pts or idx > pts[-1]:
            pts.append(idx)
    return full_ids, pts


def _load_eval_guard_12grams():
    """Collect 12-grams from every eval set (mcq, mcq2, proc_held, lastword, passages, C25_books, items3, items7)."""
    g = set()
    for p, key in (
        (COMP / "mcq.jsonl", "passage"),
        (COMP / "mcq2.jsonl", "passage"),
        (COMP / "passages.jsonl", "passage"),
        (COMP / "lastword.jsonl", "context"),
        (MID / "proc_held.jsonl", "passage"),
        (HERE / "runs" / "gen" / "C25_books.jsonl", "prefix"),
    ):
        if p.exists():
            with open(p) as f:
                for l in f:
                    r = json.loads(l)
                    if key in r:
                        g |= _grams12(r[key])
    for folder in (ARCH / "items3", ARCH / "items7"):
        if folder.exists():
            for fp in sorted(folder.glob("w*.py")):
                g |= _grams12(fp.read_text())
    return g


def _load_items3_eval():
    """Load ALL 100 myths + 100 mechanisms (800 texts: anchor, true, foil, lure) from items3 strictly for held-out eval (Amendment 228a)."""
    if str(ARCH) not in sys.path:
        sys.path.insert(0, str(ARCH))
    import importlib
    os.environ.pop("FRESH_DIR", None)
    fresh = importlib.import_module("fresh")
    out = []
    for s_lbl, items, ex in (("myth", fresh.A, fresh.A2), ("mech", fresh.B, fresh.B2)):
        for iid, pat, anchor, true_t, _ in items:
            lure_t, foil_t = ex[iid]
            for role, txt in (("anchor", anchor), ("true", true_t), ("foil", foil_t), ("lure", lure_t)):
                txt_c = " ".join(txt.strip().split())
                out.append({
                    "id": f"i3_{s_lbl}_{iid}_{role}",
                    "domain": f"items3_{s_lbl}",
                    "group_id": f"i3_{s_lbl}_{iid}",
                    "item_id": f"{s_lbl}_{iid}",
                    "kind": s_lbl,
                    "role": role,
                    "text": txt_c,
                    "segs": segment_passage(txt_c, n_pts=3),
                })
    return out


def _sample_paragraphs_from_text(text, rng, min_w=110, max_w=200, max_keep=6):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    good = []
    for p in paras[2:-2]:
        ws = p.split()
        if min_w <= len(ws) <= max_w and p[-1] in ".!?\"'\u201d":
            if sum(c.isalpha() for c in p) / max(1, len(p)) > 0.72:
                good.append(p)
    rng.shuffle(good)
    return good[:max_keep]


# ---------------------------------------------------------------- Step A: Build public myth_train.bin + passages.jsonl + eval_passages.jsonl

def build_corpus():
    import pyarrow.parquet as pq
    t0 = time.time()
    tk = load_bpe8k()
    sep = tk.encode("\n\n").ids
    guard_g = _load_eval_guard_12grams()
    rng = random.Random(2280)

    # 1. Reconstruct EXACT gut_tokens.py held-out 200 books so ZERO held-out books leak into train or myth_train!
    books = []
    book_meta = {}
    for f in sorted(glob.glob(str(GUT_DIR / "*.parquet"))):
        t = pq.read_table(f, columns=["METADATA"])
        for i, m in enumerate(t.column("METADATA").to_pylist()):
            md = json.loads(m)
            tid = int(md.get("text_id", -1))
            if md.get("language") == "en" and tid not in JUNG_IDS:
                books.append((f, i))
                ttl = str(md.get("title") or "") + " " + str(md.get("subjects") or "")
                book_meta[(f, i)] = (tid, bool(MYTH_PAT.search(ttl)))
    order = np.random.default_rng(220).permutation(len(books))
    want = [books[j] for j in order]
    held_books = list(want[:200])
    train_nonmyth = [b for b in want[200:] if not book_meta[b][1]]
    train_myth = [b for b in want[200:] if book_meta[b][1]]
    print(f"Gutenberg split: held={len(held_books)} train_nonmyth={len(train_nonmyth)} train_myth={len(train_myth)}", flush=True)

    # 2. Build myth_train.bin (~25M tokens) + 1,200 myth passages, guarding WHOLE BOOKS against eval 12-grams (Amendment 228a!)
    rng.shuffle(train_myth)
    myth_by_file = {}
    for b in train_myth[:750]:
        myth_by_file.setdefault(b[0], []).append(b[1])

    myth_tok_chunks = []
    myth_tok_total = 0
    myth_passages = []
    myth_skipped_overlap = 0
    target_myth_toks = 25_000_000

    for f, idxs in sorted(myth_by_file.items()):
        col = pq.read_table(f, columns=["TEXT"]).column("TEXT")
        for i in idxs:
            txt = clean_gut(col[i].as_py())
            if _grams12(txt) & guard_g:
                myth_skipped_overlap += 1
                continue
            if myth_tok_total < target_myth_toks:
                ids = tk.encode(txt).ids
                if len(ids) >= 256:
                    myth_tok_chunks.append(np.array(ids + sep, dtype=np.uint16))
                    myth_tok_total += len(ids) + len(sep)
            if len(myth_passages) < 1400:
                myth_passages.extend(_sample_paragraphs_from_text(txt, rng, min_w=110, max_w=200, max_keep=4))
        if myth_tok_total >= target_myth_toks and len(myth_passages) >= 1400:
            break

    myth_bin = np.concatenate(myth_tok_chunks)[:target_myth_toks]
    myth_bin.tofile(PLATO / "myth_train.bin")
    rng.shuffle(myth_passages)
    myth_passages = myth_passages[:1200]
    print(f"myth_train.bin: {len(myth_bin):,} tokens (skipped {myth_skipped_overlap} eval-overlapping books), myth_passages={len(myth_passages)} ({time.time() - t0:.1f} s)", flush=True)

    # 3. Sample 1,200 general Gutenberg train passages (strictly non-held, non-myth, book-level 12-gram guarded) + 100 held-out Gutenberg
    rng.shuffle(train_nonmyth)
    gut_by_file = {}
    for b in train_nonmyth[:500]:
        gut_by_file.setdefault(b[0], []).append(b[1])
    gut_passages = []
    for f, idxs in sorted(gut_by_file.items()):
        col = pq.read_table(f, columns=["TEXT"]).column("TEXT")
        for i in idxs:
            txt = clean_gut(col[i].as_py())
            if _grams12(txt) & guard_g:
                continue
            gut_passages.extend(_sample_paragraphs_from_text(txt, rng, min_w=110, max_w=200, max_keep=4))
        if len(gut_passages) >= 1400:
            break
    rng.shuffle(gut_passages)
    gut_passages = gut_passages[:1200]

    held_by_file = {}
    for b in held_books[:80]:
        held_by_file.setdefault(b[0], []).append(b[1])
    held_gut_passages = []
    for f, idxs in sorted(held_by_file.items()):
        col = pq.read_table(f, columns=["TEXT"]).column("TEXT")
        for i in idxs:
            txt = clean_gut(col[i].as_py())
            held_gut_passages.extend(_sample_paragraphs_from_text(txt, rng, min_w=110, max_w=200, max_keep=3))
        if len(held_gut_passages) >= 120:
            break
    rng.shuffle(held_gut_passages)
    held_gut_passages = held_gut_passages[:100]

    # 4. Sample 600 unique proc_train passages & 600 unique chat_train passages (while-loop + dedup; Amendment 228a!)
    proc_mm = np.memmap(MID / "proc_train.bin", dtype=np.uint16, mode="r")
    proc_rng = np.random.default_rng(2281)
    proc_passages, seen_h = [], set()
    tries = 0
    while len(proc_passages) < 600 and tries < 25_000:
        tries += 1
        off = int(proc_rng.integers(0, len(proc_mm) - 420))
        raw = tk.decode(proc_mm[off:off + 380].tolist())
        m = re.search(r"(Passage:.*?Answer:[^\n]+)", raw, re.S)
        if m:
            p = m.group(1).strip()
            h = hashlib.sha1(p.encode()).hexdigest()
            if h not in seen_h and 55 <= len(p.split()) <= 200 and not (_grams12(p) & guard_g):
                seen_h.add(h)
                proc_passages.append(p)
    assert len(proc_passages) == 600, f"got only {len(proc_passages)} proc passages after {tries} tries"

    chat_mm = np.memmap(MID / "chat_train.bin", dtype=np.uint16, mode="r")
    chat_passages, seen_c = [], set()
    tries = 0
    while len(chat_passages) < 600 and tries < 30_000:
        tries += 1
        off = int(proc_rng.integers(0, len(chat_mm) - 420))
        raw = tk.decode(chat_mm[off:off + 360].tolist())
        m = re.search(r"(User:.*?Assistant:[^\n]+(?:\n[^\n]+)?)", raw, re.S)
        if m:
            p = " ".join(m.group(1).strip().split())
            h = hashlib.sha1(p.encode()).hexdigest()
            if h not in seen_c and 50 <= len(p.split()) <= 195 and not (_grams12(p) & guard_g):
                seen_c.add(h)
                chat_passages.append(p)
    assert len(chat_passages) == 600, f"got only {len(chat_passages)} chat passages after {tries} tries"

    # 5. Assemble passages.jsonl (3,600 train passages with clean segment boundaries)
    train_rows = []
    for dom, pool in (("gut", gut_passages), ("proc", proc_passages), ("chat", chat_passages), ("myth", myth_passages)):
        for i, txt in enumerate(pool):
            segs = segment_passage(txt, n_pts=6)
            if len(segs) >= 3:
                train_rows.append({"id": f"{dom}_{i:04d}", "domain": dom, "text": txt, "segs": segs})
    with open(PLATO / "passages.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    # 6. Assemble eval_passages.jsonl: 100 held_gut + 100 stratified proc_held (20/family!) + ALL 800 items3 texts = 1,000 eval texts!
    eval_rows = []
    for i, txt in enumerate(held_gut_passages):
        eval_rows.append({
            "id": f"ev_gut_{i:03d}", "domain": "held_gut", "group_id": f"ev_gut_{i:03d}",
            "text": txt, "segs": segment_passage(txt, n_pts=4),
        })
    with open(MID / "proc_held.jsonl") as f:
        ph = [json.loads(l) for l in f]
    by_kind = defaultdict(list)
    for r in ph:
        by_kind[r["kind"]].append(r)
    p_idx = 0
    for k in sorted(by_kind):
        for r in by_kind[k][:20]:
            txt = f"Passage: {r['passage']}\n\nQuestion: {r['question']}"
            eval_rows.append({
                "id": f"ev_proc_{p_idx:03d}", "domain": "proc_held", "group_id": f"ev_proc_{p_idx:03d}",
                "kind": k, "text": txt, "segs": segment_passage(txt, n_pts=4),
            })
            p_idx += 1
    eval_rows.extend(_load_items3_eval())

    with open(PLATO / "eval_passages.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    print(f"CORPUS_DONE train_passages={len(train_rows)} eval_passages={len(eval_rows)} ({time.time() - t0:.1f} s)", flush=True)


# ---------------------------------------------------------------- Step B: Teacher hidden-state extraction (MLX + local Inkling llama-server!)

def _text_ids_mlx(tok, t):
    """Exact cross.py <bos> rule so MLX matches PyTorch at cosine 0.999 across Gemma 4, Llama 3.1, Qwen, Mistral, Phi."""
    ids = tok.encode(t, add_special_tokens=False)
    if getattr(tok, "_starts_bos", None) is None:
        b = tok.bos_token
        try:
            tmpl = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
            tok._starts_bos = bool(b) and tmpl.startswith(b)
        except Exception:
            tok._starts_bos = bool(b)
    return [tok.bos_token_id] + ids if tok._starts_bos and (not ids or ids[0] != tok.bos_token_id) else ids


def _find_layers(model):
    for obj in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
        if obj is not None:
            if hasattr(obj, "layers"):
                return obj.layers
            if hasattr(obj, "model") and hasattr(obj.model, "layers"):
                return obj.model.layers
    raise RuntimeError(f"Could not locate decoder layers on {type(model)}")


def _atomic_savez(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _extract_llamacpp(key, spec, train_rows, eval_rows, out_npz, corpus_sig="", model_sig=""):
    """Extract per-token hidden states from local Inkling-2-Mini GGUF via llama-server --embeddings --pooling none --embd-normalize -1."""
    t0 = time.time()
    port = spec["port"]
    srv = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    cmd = [
        srv, "-m", spec["path"], "-ngl", "999", "-c", "2048", "-b", "1024", "-ub", "1024",
        "--embeddings", "--pooling", "none", "--embd-normalize", "-1",
        "--host", "127.0.0.1", "--port", str(port), "--no-webui",
    ]
    # Fail closed if ANY TCP listener already owns the port (regardless of HTTP status code or timeout)!
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            raise RuntimeError(f"Port {port} is already occupied by an existing TCP listener; refusing to attach {key} extractor to an unknown server!")
    except RuntimeError:
        raise
    except OSError:
        pass
    print(f"EXTRACT {key}: starting local Metal llama-server on port {port} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post(ep, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{ep}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    try:
        ok = False
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                raise RuntimeError(f"{key} child llama-server exited prematurely with code {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200 and proc.poll() is None:
                        ok = True
                        break
            except Exception:
                time.sleep(0.5)
        assert ok and proc.poll() is None, f"{key} llama-server failed to become healthy"
        enc_fn = lambda s: post("/tokenize", {"content": s, "add_special": True})["tokens"]

        def run_pool(rows, tag):
            pts_vecs, story_vecs, pt_cnts, row_ok = [], [], [], []
            for idx, r in enumerate(rows):
                full_ids, pt_idxs = boundary_token_indices(enc_fn, r["segs"])
                good = (2 <= len(full_ids) <= 512 and len(pt_idxs) == len(r["segs"]))
                if not good:
                    # Still query 1 tiny dummy or store zero-length row marked row_ok=False!
                    emb = post("/embedding", {"content": r["text"][:32]})
                    arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
                    D = arr.shape[1]
                    story_vecs.append(np.zeros(D, dtype=np.float32))
                    pt_cnts.append(0)
                    row_ok.append(False)
                    continue
                emb = post("/embedding", {"content": r["text"]})
                arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
                if arr.shape[0] != len(full_ids):
                    story_vecs.append(np.zeros(arr.shape[1], dtype=np.float32))
                    pt_cnts.append(0)
                    row_ok.append(False)
                    continue
                pts_vecs.append(arr[pt_idxs])
                story_vecs.append(arr[1:].mean(axis=0) if arr.shape[0] > 1 else arr[0])
                pt_cnts.append(len(pt_idxs))
                row_ok.append(True)
                if (idx + 1) % 400 == 0 or idx == len(rows) - 1:
                    print(f"EXTRACT {key} [{tag}] {idx + 1}/{len(rows)} ok={sum(row_ok)} ({time.time() - t0:.0f} s)", flush=True)
            return np.concatenate(pts_vecs, axis=0), np.stack(story_vecs, axis=0), np.array(pt_cnts, dtype=np.int32), np.array(row_ok, dtype=bool)

        tr_pts, tr_story, tr_cnt, tr_ok = run_pool(train_rows, "train")
        ev_pts, ev_story, ev_cnt, ev_ok = run_pool(eval_rows, "eval")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert np.isfinite(tr_pts).all() and np.isfinite(ev_pts).all(), f"{key} non-finite states!"
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    _atomic_savez(
        out_npz,
        tr_pts=pack(tr_pts), tr_story=pack(tr_story), tr_cnt=tr_cnt, tr_ok=tr_ok,
        ev_pts=pack(ev_pts), ev_story=pack(ev_story), ev_cnt=ev_cnt, ev_ok=ev_ok,
        layer_idx=np.int32(-1), num_layers=np.int32(24),
        corpus_sha256=np.array(corpus_sig), model_path=np.array(model_sig),
    )
    print(f"EXTRACT_DONE {key}: tr_ok={int(tr_ok.sum())}/{len(tr_ok)} tr_pts={tr_pts.shape} ev_ok={int(ev_ok.sum())}/{len(ev_ok)} in {time.time() - t0:.0f} s", flush=True)


def _corpus_sha256():
    import hashlib
    h = hashlib.sha256()
    for p in (PLATO / "passages.jsonl", PLATO / "eval_passages.jsonl"):
        h.update(p.read_bytes())
    return h.hexdigest()


def extract_model(key):
    assert key in MODEL_SPECS, f"unknown model key {key}"
    out_npz = PLATO / f"raw_{key}.npz"
    spec = MODEL_SPECS[key]
    corpus_sig = _corpus_sha256()
    model_sig = str(spec["path"])
    if out_npz.exists():
        try:
            with np.load(out_npz) as d:
                c_ok = ("corpus_sha256" in d.files and str(d["corpus_sha256"].item()) == corpus_sig)
                m_ok = ("model_path" in d.files and str(d["model_path"].item()) == model_sig)
            if c_ok and m_ok:
                print(f"EXTRACT {key}: verified cache ({out_npz}), skipping", flush=True)
                return
            print(f"EXTRACT {key}: cache provenance mismatch or missing metadata ({out_npz}) — re-extracting", flush=True)
        except Exception:
            print(f"EXTRACT {key}: unreadable cache ({out_npz}) — re-extracting", flush=True)

    train_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    eval_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl")]
    if spec["backend"] == "llamacpp":
        _extract_llamacpp(key, spec, train_rows, eval_rows, out_npz, corpus_sig=corpus_sig, model_sig=model_sig)
        return

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(2 << 30)
    t0 = time.time()
    print(f"EXTRACT {key}: loading {spec['path']} ...", flush=True)
    model, tok = load(spec["path"])
    layers = _find_layers(model)
    nl = len(layers)
    # Exact 0.30 * (nl - 1) convention (matches statedpat.py: layer 12/42 for Gemma 4, layer 9/32 for Qwen/Llama/Mistral/Phi!)
    tgt_idx = max(1, min(nl - 2, int(round(spec["depth_frac"] * (nl - 1)))))
    tgt_blk = layers[tgt_idx]
    print(f"EXTRACT {key}: loaded L={nl}, hooking layer {tgt_idx} ({tgt_idx / (nl - 1):.1%} depth, {type(tgt_blk).__name__})", flush=True)

    rec_buf = []
    patched = {}
    for blk in layers:
        cls = type(blk)
        if cls not in patched:
            orig = cls.__call__
            patched[cls] = orig

            def make_hook(orig_fn):
                def hooked(self, *a, **k):
                    out = orig_fn(self, *a, **k)
                    if self is tgt_blk:
                        h = out[0] if isinstance(out, tuple) else out
                        rec_buf.append(h[0])
                    return out
                return hooked

            cls.__call__ = make_hook(orig)

    enc_fn = lambda s: _text_ids_mlx(tok, s)

    def run_pool(rows, tag):
        pts_vecs, story_vecs, pt_cnts, row_ok = [], [], [], []
        D_dim = None
        for idx, r in enumerate(rows):
            full_ids, pt_idxs = boundary_token_indices(enc_fn, r["segs"])
            good = (2 <= len(full_ids) <= 512 and len(pt_idxs) == len(r["segs"]))
            if not good:
                if D_dim is None:
                    rec_buf.clear(); _ = model(mx.array([full_ids[:4]])); mx.eval(rec_buf[0]); D_dim = int(rec_buf[0].shape[-1])
                story_vecs.append(np.zeros(D_dim, dtype=np.float32))
                pt_cnts.append(0)
                row_ok.append(False)
                continue
            rec_buf.clear()
            _ = model(mx.array([full_ids]))
            assert len(rec_buf) == 1, f"{key} hook fired {len(rec_buf)} times!"
            h = rec_buf[0].astype(mx.float32)
            D_dim = int(h.shape[-1])
            h_pts = h[mx.array(pt_idxs)]
            h_mean = h[1:].mean(axis=0) if h.shape[0] > 1 else h[0]
            mx.eval(h_pts, h_mean)
            pts_vecs.append(np.array(h_pts, dtype=np.float32))
            story_vecs.append(np.array(h_mean, dtype=np.float32))
            pt_cnts.append(len(pt_idxs))
            row_ok.append(True)
            if (idx + 1) % 400 == 0 or idx == len(rows) - 1:
                print(f"EXTRACT {key} [{tag}] {idx + 1}/{len(rows)} ok={sum(row_ok)} ({time.time() - t0:.0f} s)", flush=True)
        return np.concatenate(pts_vecs, axis=0), np.stack(story_vecs, axis=0), np.array(pt_cnts, dtype=np.int32), np.array(row_ok, dtype=bool)

    try:
        tr_pts, tr_story, tr_cnt, tr_ok = run_pool(train_rows, "train")
        ev_pts, ev_story, ev_cnt, ev_ok = run_pool(eval_rows, "eval")
    finally:
        for cls, orig in patched.items():
            cls.__call__ = orig
        del model, tok
        gc.collect()
        mx.clear_cache()

    assert np.isfinite(tr_pts).all() and np.isfinite(ev_pts).all(), f"{key} non-finite states!"
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    _atomic_savez(
        out_npz,
        tr_pts=pack(tr_pts), tr_story=pack(tr_story), tr_cnt=tr_cnt, tr_ok=tr_ok,
        ev_pts=pack(ev_pts), ev_story=pack(ev_story), ev_cnt=ev_cnt, ev_ok=ev_ok,
        layer_idx=np.int32(tgt_idx), num_layers=np.int32(nl),
        corpus_sha256=np.array(corpus_sig), model_path=np.array(model_sig),
    )
    print(f"EXTRACT_DONE {key}: tr_ok={int(tr_ok.sum())}/{len(tr_ok)} tr_pts={tr_pts.shape} ev_ok={int(ev_ok.sum())}/{len(ev_ok)} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- Step C: Generalized Procrustes Analysis (Plato's Key math!)

def fit_procrustes_map(A, B):
    """Exact Plato's Key closed-form Orthogonal Procrustes + global Frobenius scale mapping A -> B."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mu_a = A.mean(axis=0, keepdims=True)
    mu_b = B.mean(axis=0, keepdims=True)
    Ac = A - mu_a
    Bc = B - mu_b
    M = Bc.T @ Ac
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    na = np.linalg.norm(Ac @ R.T, "fro")
    nb = np.linalg.norm(Bc, "fro")
    s = float(nb / max(1e-12, na))
    return mu_a, mu_b, R, s


def apply_procrustes_map(A, params):
    mu_a, mu_b, R, s = params
    return mu_b + s * ((np.asarray(A, dtype=np.float64) - mu_a) @ R.T)


def pca_project(X_tr, X_ev, d_out=256):
    """Fit centered PCA to d_out dimensions strictly on X_tr, project X_tr and X_ev, scale so X_tr has entrywise RMS = 1.0."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_ev = np.asarray(X_ev, dtype=np.float64)
    mu = X_tr.mean(axis=0, keepdims=True)
    Xc = X_tr - mu
    cov = (Xc.T @ Xc) / float(len(Xc))
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:d_out]
    V = vecs[:, idx]
    Z_tr = Xc @ V
    Z_ev = (X_ev - mu) @ V
    rms = float(np.sqrt(np.mean(Z_tr ** 2)))
    Z_tr = Z_tr / max(1e-12, rms)
    Z_ev = Z_ev / max(1e-12, rms)
    var_exp = float(vals[idx].sum() / max(1e-12, vals.sum()))
    return Z_tr, Z_ev, var_exp


def generalized_procrustes_consensus(Y_tr_list, Y_ev_list, n_iters=10):
    """Iterative Generalized Procrustes Analysis fit STRICTLY on Y_tr_list (zero eval leakage!)."""
    K = len(Y_tr_list)
    C = Y_tr_list[0].copy()
    rots = [None] * K
    for _ in range(n_iters):
        aligned = []
        for k in range(K):
            params = fit_procrustes_map(Y_tr_list[k], C)
            rots[k] = params
            aligned.append(apply_procrustes_map(Y_tr_list[k], params))
        C = np.mean(aligned, axis=0)
        C = C - C.mean(axis=0, keepdims=True)
        C = C / max(1e-12, float(np.sqrt(np.mean(C ** 2))))
    tr_aligned, ev_aligned = [], []
    for k in range(K):
        params = fit_procrustes_map(Y_tr_list[k], C)
        rots[k] = params
        tr_aligned.append(apply_procrustes_map(Y_tr_list[k], params))
        ev_aligned.append(apply_procrustes_map(Y_ev_list[k], params))
    tr_raw = np.mean(tr_aligned, axis=0)
    ev_raw = np.mean(ev_aligned, axis=0)
    tr_mu = tr_raw.mean(axis=0, keepdims=True)
    tr_rms = max(1e-12, float(np.sqrt(np.mean((tr_raw - tr_mu) ** 2))))
    C_tr = (tr_raw - tr_mu) / tr_rms
    C_ev = (ev_raw - tr_mu) / tr_rms
    return C_tr, C_ev, tr_aligned, ev_aligned, rots


def eval_procrustes_transfer(src_ev, tgt_ev, group_ids=None, domains=None, ret_mask=None):
    """Domain-stratified, group-aware 2-fold CV (Amendment 228a: never splits a passage or items3 quartet across folds!).
    Reports overall cosine, Frobenius R^2, optimal-scale R^2, retrieval Top-1/Top-5 (on ret_mask if provided), and per-domain cosine."""
    src_ev = np.asarray(src_ev, dtype=np.float64)
    tgt_ev = np.asarray(tgt_ev, dtype=np.float64)
    n = len(src_ev)
    if group_ids is None:
        group_ids = [str(i) for i in range(n)]
    if domains is None:
        domains = ["all"] * n

    # Assign each unique group deterministically to fold 0 or fold 1 stratified by domain
    dom_groups = defaultdict(list)
    seen_g = set()
    for g, d in zip(group_ids, domains):
        if g not in seen_g:
            seen_g.add(g)
            dom_groups[d].append(g)
    g_fold = {}
    rng = random.Random(2288)
    for d in sorted(dom_groups):
        gs = list(dom_groups[d])
        rng.shuffle(gs)
        half = len(gs) // 2
        for i, g in enumerate(gs):
            g_fold[g] = 0 if i < half else 1
    folds = np.array([g_fold[g] for g in group_ids], dtype=np.int32)

    cos_all, pred_all, pred_opt_all, true_all = [], [], [], []
    top1_list, top5_list = [], []
    dom_cos = defaultdict(list)

    for test_f in (0, 1):
        m_tr = folds != test_f
        m_te = folds == test_f
        params = fit_procrustes_map(src_ev[m_tr], tgt_ev[m_tr])
        pred_tr = apply_procrustes_map(src_ev[m_tr], params)
        pred = apply_procrustes_map(src_ev[m_te], params)
        true_tr = tgt_ev[m_tr]
        true = tgt_ev[m_te]
        # Strictly train-fold optimal scalar for r2_opt (no test-fold leakage!)
        ym_tr = true_tr.mean(axis=0, keepdims=True)
        pm_tr = pred_tr.mean(axis=0, keepdims=True)
        alpha_f = float(np.sum((true_tr - ym_tr) * (pred_tr - pm_tr)) / max(1e-12, np.sum((pred_tr - pm_tr) ** 2)))
        pred_opt_all.append(ym_tr + alpha_f * (pred - pm_tr))

        pn = pred / np.maximum(1e-12, np.linalg.norm(pred, axis=1, keepdims=True))
        tn = true / np.maximum(1e-12, np.linalg.norm(true, axis=1, keepdims=True))
        cs = np.sum(pn * tn, axis=1)
        cos_all.append(cs)
        pred_all.append(pred)
        true_all.append(true)
        te_doms = [domains[i] for i in np.flatnonzero(m_te)]
        for c_val, d_val in zip(cs, te_doms):
            dom_cos[d_val].append(float(c_val))

        # Retrieval on non-duplicate subset (ret_mask excludes items3 foil whose prefix duplicates true!)
        if ret_mask is not None:
            rm = np.asarray(ret_mask, dtype=bool)[m_te]
            pn_r, tn_r = pn[rm], tn[rm]
        else:
            pn_r, tn_r = pn, tn
        if len(pn_r) >= 6:
            sim = pn_r @ tn_r.T
            diag = np.diag(sim)[:, None]
            # Exact tie-safe fractional credit (constant vector gets 1/N, NOT 1.0!)
            strictly_better = (sim > diag + 1e-12).sum(axis=1)
            tied = (np.abs(sim - diag) <= 1e-12).sum(axis=1)
            top1_list.append(float(np.mean(np.where(strictly_better == 0, 1.0 / np.maximum(1, tied), 0.0))))
            rem5 = np.maximum(0, 5 - strictly_better)
            top5_list.append(float(np.mean(np.minimum(1.0, rem5 / np.maximum(1, tied)))))

    pred_cat = np.concatenate(pred_all, axis=0)
    pred_opt_cat = np.concatenate(pred_opt_all, axis=0)
    true_cat = np.concatenate(true_all, axis=0)
    ss_res = float(np.sum((true_cat - pred_cat) ** 2))
    ss_tot = float(np.sum((true_cat - true_cat.mean(axis=0, keepdims=True)) ** 2))
    r2_frob = 1.0 - ss_res / max(1e-12, ss_tot)
    r2_opt = 1.0 - float(np.sum((true_cat - pred_opt_cat) ** 2)) / max(1e-12, ss_tot)

    out = {
        "cos": round(float(np.concatenate(cos_all).mean()), 4),
        "r2_frob": round(r2_frob, 4),
        "r2_opt": round(r2_opt, 4),
        "top1": round(float(np.mean(top1_list)), 4) if top1_list else None,
        "top5": round(float(np.mean(top5_list)), 4) if top5_list else None,
    }
    if len(dom_cos) > 1:
        out["by_domain_cos"] = {d: round(float(np.mean(v)), 4) for d, v in sorted(dom_cos.items())}
    return out


def build_consensus(d_plato=256, mode="auto"):
    t0 = time.time()
    tk = load_bpe8k()
    train_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl")]
    eval_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl")]

    # Support explicit `c5` (7-model +Inkling), explicit `c4` (pure-MLX 6-model, even if Inkling cache exists!), or `auto`.
    mode = str(mode).lower()
    assert mode in ("auto", "c5", "c4"), f"Invalid consensus mode: {mode} (use auto, c5, or c4)"
    has_inkling = ((PLATO / "raw_inkling2_mini.npz").exists() if mode == "auto" else (mode == "c5"))
    active_consensus_keys = list(CONSENSUS_KEYS) if has_inkling else list(MLX_CONSENSUS_KEYS)
    active_all_keys = active_consensus_keys + list(BLIND_KEYS)
    if not has_inkling:
        print(f"Running pure-MLX 6-model C4-only consensus mode (mode={mode}; canonical C5 meta.json will NOT be touched).", flush=True)

    corpus_sha = _corpus_sha256()
    raw = {}
    for k in active_all_keys:
        p = PLATO / f"raw_{k}.npz"
        assert p.exists(), f"missing {p}; run extract first!"
        d = np.load(p)
        if "corpus_sha256" in d.files:
            got_sha = str(d["corpus_sha256"].item())
            assert got_sha == corpus_sha, f"Stale cache {p}: corpus_sha256={got_sha[:12]} != current {corpus_sha[:12]}; re-extract {k}!"
        if "model_path" in d.files and k in MODELS:
            got_mp = str(d["model_path"].item())
            assert got_mp == MODELS[k]["path"], f"Model mismatch in {p}: {got_mp} != {MODELS[k]['path']}; re-extract {k}!"
        raw[k] = {fld: d[fld] for fld in d.files}

    # Strict cross-teacher alignment (Amendment 228a: EVERY teacher + bpe8k must match exact len(r["segs"]) with zero truncation!)
    bpe_enc = lambda s: tk.encode(s).ids
    tr_bpe_ids, tr_bpe_pts, tr_domains = [], [], []
    tr_story_keep = []
    tr_keep_masks = {k: [] for k in active_all_keys}

    for r_i, r in enumerate(train_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raw[k]["tr_ok"][r_i]) and int(raw[k]["tr_cnt"][r_i]) == n_seg for k in active_all_keys)
        )
        tr_story_keep.append(ok)
        for k in active_all_keys:
            tr_keep_masks[k].extend([ok] * int(raw[k]["tr_cnt"][r_i]))
        if ok:
            tr_bpe_ids.append(b_ids)
            tr_bpe_pts.append(b_pts)
            tr_domains.append(r["domain"])

    ev_bpe_ids, ev_bpe_pts, ev_meta = [], [], []
    ev_pt_groups, ev_pt_domains, ev_pt_ret_mask = [], [], []
    ev_story_keep = []
    ev_keep_masks = {k: [] for k in active_all_keys}

    for r_i, r in enumerate(eval_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raw[k]["ev_ok"][r_i]) and int(raw[k]["ev_cnt"][r_i]) == n_seg for k in active_all_keys)
        )
        ev_story_keep.append(ok)
        for k in active_all_keys:
            ev_keep_masks[k].extend([ok] * int(raw[k]["ev_cnt"][r_i]))
        if ok:
            ev_bpe_ids.append(b_ids)
            ev_bpe_pts.append(b_pts)
            m_row = {k: r.get(k, "") for k in ("id", "domain", "group_id", "item_id", "kind", "role")}
            ev_meta.append(m_row)
            is_ret = (r.get("role", "") != "foil")  # exclude all items3 foil checkpoints by role!
            for _ in range(n_seg):
                ev_pt_groups.append(m_row["group_id"])
                ev_pt_domains.append(m_row["domain"])
                ev_pt_ret_mask.append(is_ret)

    tr_story_mask = np.array(tr_story_keep, dtype=bool)
    ev_story_mask = np.array(ev_story_keep, dtype=bool)
    print(f"CONSENSUS strict-aligned passages: train={int(tr_story_mask.sum())}/{len(train_rows)} eval={int(ev_story_mask.sum())}/{len(eval_rows)}", flush=True)

    # Fit 256-dim PCA strictly on train points for each teacher + blind Phi-4-mini & OLMo-3-7B
    Y_tr, Y_ev, var_exp = {}, {}, {}
    for k in active_all_keys:
        m_tr = np.array(tr_keep_masks[k], dtype=bool)
        m_ev = np.array(ev_keep_masks[k], dtype=bool)
        X_tr_k = raw[k]["tr_pts"][m_tr].astype(np.float64)
        X_ev_k = raw[k]["ev_pts"][m_ev].astype(np.float64)
        Y_tr[k], Y_ev[k], var_exp[k] = pca_project(X_tr_k, X_ev_k, d_out=d_plato)
        print(f"  PCA256 {k:14s}: tr={Y_tr[k].shape} ev={Y_ev[k].shape} var_exp={var_exp[k]:.3f}", flush=True)

    # Fit 4-family (30%-depth MLX) and, when Inkling 2 Mini is present, 5-family (+Inkling-2-Mini) GPA Consensus strictly on train!
    C4_tr, C4_ev, _, ev_al4, _ = generalized_procrustes_consensus([Y_tr[k] for k in MLX_CONSENSUS_KEYS], [Y_ev[k] for k in MLX_CONSENSUS_KEYS], n_iters=10)
    if has_inkling:
        C5_tr, C5_ev, _, ev_al_main, _ = generalized_procrustes_consensus([Y_tr[k] for k in CONSENSUS_KEYS], [Y_ev[k] for k in CONSENSUS_KEYS], n_iters=10)
    else:
        C5_tr, C5_ev, ev_al_main = C4_tr, C4_ev, ev_al4

    # Story-level blind targets (strictly masked by tr_story_mask!)
    s_blind_ev = {}
    for bk in BLIND_KEYS:
        _, s_blind_ev[bk], _ = pca_project(
            raw[bk]["tr_story"][tr_story_mask].astype(np.float64),
            raw[bk]["ev_story"][ev_story_mask].astype(np.float64),
            d_out=d_plato,
        )

    # Stratified, group-aware blind diagnostics against BOTH unseen families (Phi-4-mini & OLMo-3-7B)!
    diag = {"blind_families": list(BLIND_KEYS), "has_inkling": has_inkling, "corpus_sha256": corpus_sha, "var_explained_256": {k: round(v, 4) for k, v in var_exp.items()}, "to_blind_pts": {}}
    for bk in BLIND_KEYS:
        sub = {}
        for k in active_consensus_keys:
            res_k = eval_procrustes_transfer(Y_ev[k], Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_domains, ret_mask=ev_pt_ret_mask)
            sub[k] = res_k
            print(f"  Blind -> {bk:9s} [{k:14s}]: cos={res_k['cos']:.4f} r2o={res_k['r2_opt']:.4f} top1={res_k['top1']:.4f} top5={res_k['top5']:.4f}", flush=True)
        res_c4 = eval_procrustes_transfer(C4_ev, Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_domains, ret_mask=ev_pt_ret_mask)
        sub["CONSENSUS_4FAM_MLX"] = res_c4
        print(f"  Blind -> {bk:9s} [CONSENSUS_4FAM]: cos={res_c4['cos']:.4f} r2o={res_c4['r2_opt']:.4f} top1={res_c4['top1']:.4f} top5={res_c4['top5']:.4f}", flush=True)
        if has_inkling:
            res_c5 = eval_procrustes_transfer(C5_ev, Y_ev[bk], group_ids=ev_pt_groups, domains=ev_pt_domains, ret_mask=ev_pt_ret_mask)
            sub["CONSENSUS_5FAM_INK"] = res_c5
            print(f"  Blind -> {bk:9s} [CONSENSUS_5FAM]: cos={res_c5['cos']:.4f} r2o={res_c5['r2_opt']:.4f} top1={res_c5['top1']:.4f} top5={res_c5['top5']:.4f}", flush=True)
        diag["to_blind_pts"][bk] = sub

    # Pairwise cross-family cosine AFTER GPA alignment on held-out eval points
    pair_cos = {}
    for i, ki in enumerate(active_consensus_keys):
        for j in range(i + 1, len(active_consensus_keys)):
            kj = active_consensus_keys[j]
            a = ev_al_main[i] / np.maximum(1e-12, np.linalg.norm(ev_al_main[i], axis=1, keepdims=True))
            b = ev_al_main[j] / np.maximum(1e-12, np.linalg.norm(ev_al_main[j], axis=1, keepdims=True))
            pair_cos[f"{ki}_vs_{kj}"] = round(float(np.mean(np.sum(a * b, axis=1))), 4)
    diag["post_gpa_eval_pairwise_cos"] = pair_cos
    print(f"  Post-GPA held-out pairwise cos: {pair_cos}", flush=True)

    def pack_ragged(seq_list, pt_list):
        ids_flat = np.concatenate([np.array(s, dtype=np.uint16) for s in seq_list])
        ids_off = np.zeros(len(seq_list) + 1, dtype=np.int64)
        ids_off[1:] = np.cumsum([len(s) for s in seq_list])
        pts_flat = np.concatenate([np.array(p, dtype=np.int32) for p in pt_list])
        pts_off = np.zeros(len(pt_list) + 1, dtype=np.int64)
        pts_off[1:] = np.cumsum([len(p) for p in pt_list])
        return ids_flat, ids_off, pts_flat, pts_off

    tr_ids_f, tr_ids_o, tr_pts_f, tr_pts_o = pack_ragged(tr_bpe_ids, tr_bpe_pts)
    ev_ids_f, ev_ids_o, ev_pts_f, ev_pts_o = pack_ragged(ev_bpe_ids, ev_bpe_pts)
    assert len(tr_pts_f) == len(C4_tr) and len(ev_pts_f) == len(C4_ev)

    diag["n_train_passages"] = int(tr_story_mask.sum())
    diag["n_train_pts"] = int(len(C4_tr))
    diag["n_eval_passages"] = int(ev_story_mask.sum())
    diag["n_eval_pts"] = int(len(C4_ev))
    diag["elapsed_s"] = round(time.time() - t0, 1)

    if has_inkling:
        assert len(tr_pts_f) == len(C5_tr) and len(ev_pts_f) == len(C5_ev)
        _atomic_savez(
            PLATO / "targets_train.npz",
            ids_flat=tr_ids_f, ids_off=tr_ids_o, pts_flat=tr_pts_f, pts_off=tr_pts_o,
            domains=np.array(tr_domains),
            consensus_256=C5_tr.astype(np.float16),
            consensus4_256=C4_tr.astype(np.float16),
            single_gemma_256=Y_tr["gemma4_e4b"].astype(np.float16),
        )
        _atomic_savez(
            PLATO / "targets_eval.npz",
            ids_flat=ev_ids_f, ids_off=ev_ids_o, pts_flat=ev_pts_f, pts_off=ev_pts_o,
            pt_groups=np.array(ev_pt_groups),
            pt_domains=np.array(ev_pt_domains),
            pt_ret_mask=np.array(ev_pt_ret_mask, dtype=bool),
            consensus_256=C5_ev.astype(np.float16),
            consensus4_256=C4_ev.astype(np.float16),
            blind_phi4_256=Y_ev["phi4_mini"].astype(np.float16),
            blind_phi4_story_256=s_blind_ev["phi4_mini"].astype(np.float16),
            blind_olmo3_256=Y_ev["olmo3_7b"].astype(np.float16),
            blind_olmo3_story_256=s_blind_ev["olmo3_7b"].astype(np.float16),
            meta_json=np.array([json.dumps(m) for m in ev_meta]),
        )
        (PLATO / "meta.json").write_text(json.dumps(diag, indent=1))
        print(f"PLATO_DATA_DONE (C5+C4) train_pts={len(C5_tr)} eval_pts={len(C5_ev)} ({diag['elapsed_s']} s)", flush=True)
    else:
        # Write pure-MLX 6-model C4 outputs to dedicated files — NEVER clobber canonical C5 `meta.json` or alias `consensus_256`!
        _atomic_savez(
            PLATO / "targets_train_c4.npz",
            ids_flat=tr_ids_f, ids_off=tr_ids_o, pts_flat=tr_pts_f, pts_off=tr_pts_o,
            domains=np.array(tr_domains),
            consensus_c4_256=C4_tr.astype(np.float16),
            single_gemma_256=Y_tr["gemma4_e4b"].astype(np.float16),
        )
        _atomic_savez(
            PLATO / "targets_eval_c4.npz",
            ids_flat=ev_ids_f, ids_off=ev_ids_o, pts_flat=ev_pts_f, pts_off=ev_pts_o,
            pt_groups=np.array(ev_pt_groups),
            pt_domains=np.array(ev_pt_domains),
            pt_ret_mask=np.array(ev_pt_ret_mask, dtype=bool),
            consensus_c4_256=C4_ev.astype(np.float16),
            blind_phi4_256=Y_ev["phi4_mini"].astype(np.float16),
            blind_phi4_story_256=s_blind_ev["phi4_mini"].astype(np.float16),
            blind_olmo3_256=Y_ev["olmo3_7b"].astype(np.float16),
            blind_olmo3_story_256=s_blind_ev["olmo3_7b"].astype(np.float16),
            meta_json=np.array([json.dumps(m) for m in ev_meta]),
        )
        (PLATO / "meta_c4.json").write_text(json.dumps(diag, indent=1))
        print(f"PLATO_DATA_DONE (C4-only -> meta_c4.json; canonical C5 meta.json preserved) train_pts={len(C4_tr)} eval_pts={len(C4_ev)} ({diag['elapsed_s']} s)", flush=True)


# ---------------------------------------------------------------- Unit tests

def self_test():
    """Verify GPA recovery, stratified group-aware 2-fold CV, and boundary alignment."""
    rng = np.random.default_rng(228)
    N_tr, N_ev, d = 600, 300, 64
    Z_true = rng.normal(0, 1.0, (N_tr + N_ev, d))
    Y_tr, Y_ev = [], []
    for k in range(5):
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        scale = 0.5 + 0.6 * k
        noise = rng.normal(0, 0.55, Z_true.shape)
        Xk = scale * (Z_true + noise) @ Q + rng.normal(0, 2.0, (1, d))
        yk_tr, yk_ev, _ = pca_project(Xk[:N_tr], Xk[N_tr:], d_out=d)
        Y_tr.append(yk_tr)
        Y_ev.append(yk_ev)
    Qb, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Xb = 1.8 * (Z_true + rng.normal(0, 0.25, Z_true.shape)) @ Qb
    _, Y_blind_ev, _ = pca_project(Xb[:N_tr], Xb[N_tr:], d_out=d)

    C_tr, C_ev, _, _, _ = generalized_procrustes_consensus(Y_tr, Y_ev, n_iters=10)
    groups = [f"g{i // 3}" for i in range(N_ev)]
    doms = ["domA" if i < 150 else "domB" for i in range(N_ev)]
    single_cos = [eval_procrustes_transfer(y, Y_blind_ev, group_ids=groups, domains=doms)["cos"] for y in Y_ev]
    cons_res = eval_procrustes_transfer(C_ev, Y_blind_ev, group_ids=groups, domains=doms)
    assert cons_res["cos"] > max(single_cos) + 0.04, f"GPA consensus did not beat single views: {cons_res} vs {single_cos}"
    assert abs(np.sqrt(np.mean(C_tr ** 2)) - 1.0) < 1e-5
    assert "domA" in cons_res["by_domain_cos"] and "domB" in cons_res["by_domain_cos"]
    # Regression test: collapsed constant predictions must score chance (~1/N), NEVER 1.0!
    const_res = eval_procrustes_transfer(np.ones_like(C_ev), Y_blind_ev, group_ids=groups, domains=doms)
    assert const_res["top1"] < 0.05 and const_res["top5"] < 0.10, f"Tie-safe retrieval broken on constant vec: {const_res}"
    print(f"GPA 5-fam synthetic check: single_cos={[round(x, 3) for x in single_cos]} -> CONSENSUS cos={cons_res['cos']:.3f} r2o={cons_res['r2_opt']:.3f} top1={cons_res['top1']:.3f} const_top1={const_res['top1']:.4f} OK", flush=True)

    tk = load_bpe8k()
    txt = "The hero crossed the dark river at dawn. Inside the stone hall, the old watchman waited quietly. Before noon, the silver key vanished without a trace. No one spoke a word until nightfall."
    segs = segment_passage(txt, n_pts=4)
    ids, pts = boundary_token_indices(lambda s: tk.encode(s).ids, segs)
    assert len(pts) == len(segs) and all(pts[i] < pts[i + 1] for i in range(len(pts) - 1)) and pts[-1] == len(ids) - 1
    print(f"Boundary check: {len(segs)} segs -> pts={pts} / len={len(ids)} OK", flush=True)
    print("PLATO_DATA_TEST_OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "corpus":
        build_corpus()
    elif cmd == "extract":
        keys = sys.argv[2:] if len(sys.argv) > 2 else list(ALL_KEYS)
        for k in keys:
            extract_model(k)
    elif cmd == "consensus":
        mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
        build_consensus(mode=mode)
    else:
        raise ValueError(f"unknown cmd {cmd}")
