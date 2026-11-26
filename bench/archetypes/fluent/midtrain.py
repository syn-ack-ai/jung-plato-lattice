"""Addendum 224: mid-training for comprehension and modern chat (D0-D3 from C25, equal 50M tokens, 3 seeds).
(4090) python fluent/midtrain.py train NAME arm=D0|D1|D2|D3 seed=1 [base=C25 tokens=50e6 lr=3e-4]
(4090) python fluent/midtrain.py eval_chat NAME...     held OASST2 chat bpb -> fluent/runs/comp/NAME_chatbpb.json
(4090) python fluent/midtrain.py score_proc NAME...    scores fluent/mid/proc_held.jsonl -> fluent/runs/comp/NAME_proc.json
(4090) python fluent/midtrain.py gen_chat NAME...      100 held OASST2 prompts -> fluent/runs/gen/NAME_chat.jsonl
(Mac)  python fluent/midtrain.py judge_chat A B        Qwen3.8 27B blind both-order chat preference -> fluent/runs/kit/chat_pair_A_vs_B.json
All data here is public (Gutenberg, procedural micro-worlds, OASST2 English); Jung held-out bpb is evaluated locally only and never printed or exported."""
import json, math, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
MID = HERE / "mid"
RUNS = HERE / "runs"
KIT = RUNS / "kit"


def eval_stream_bpb(m, c, tk, arr, dev, amp_fn):
    """evaluate held-out bits per byte on a uint16 token array using non-overlapping ctx windows (same formula as train.evaluate)"""
    import torch, torch.nn.functional as F
    m.eval()
    ctx = c["ctx"]
    arr = np.asarray(arr)
    assert len(arr) > ctx + 1, f"eval stream too short ({len(arr)}) for ctx={ctx}"
    chunks = np.stack([arr[i:i + ctx + 1] for i in range(0, len(arr) - ctx - 1, ctx)]).astype(np.int64)
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(chunks), c["mb"]):
            x = torch.from_numpy(chunks[i:i + c["mb"], :-1]).to(dev)
            y = torch.from_numpy(chunks[i:i + c["mb"], 1:]).to(dev)
            with amp_fn():
                lg = m(x)
            tot += F.cross_entropy(lg.float().transpose(1, 2), y, reduction="sum").item()
    nbytes = sum(len(tk.decode(ch[1:].tolist()).encode("utf-8")) for ch in chunks)
    m.train()
    return {"bpb": tot / math.log(2) / nbytes, "nats_per_token": tot / chunks[:, 1:].size, "chunks": len(chunks)}


def eval_chat(names):
    """score held-out OASST2 chat bits per byte for any existing checkpoint (e.g., C25 baseline)"""
    import torch
    sys.path.insert(0, str(HERE))
    import train as T

    chat_held = np.memmap(MID / "chat_held.bin", dtype=np.uint16, mode="r")
    (T.RUNS / "comp").mkdir(exist_ok=True)
    for name in names:
        with open(T.RUNS / f"{name}.json") as f:
            c = json.load(f)["config"]
        tk = T.tokenizer(c["tok"])
        m = T.Toy(c, tk.get_vocab_size()).to(T.DEV)
        m.load_state_dict(torch.load(T.PRIV / "ckpt" / f"{name}.pt", map_location=T.DEV, weights_only=True))
        res = eval_stream_bpb(m, c, tk, chat_held, T.DEV, T.amp)
        with open(T.RUNS / "comp" / f"{name}_chatbpb.json", "w") as f:
            json.dump(res, f, indent=1)
        print(f"{name} chat_held bpb {res['bpb']:.4f} (chunks={res['chunks']})", flush=True)


def train(name, arm, seed, base="C25", tokens=50_000_000, lr=3e-4):
    import torch, torch.nn.functional as F
    sys.path.insert(0, str(HERE))
    import train as T

    assert arm in ("D0", "D1", "D2", "D3"), f"unknown arm {arm}"
    assert name != base and not name.startswith(("C", "B", "OLD", "M", "T", "S")), "would overwrite a base or earlier-addendum checkpoint"
    assert not (T.PRIV / "ckpt" / f"{name}.pt").exists(), f"checkpoint already exists: {name}.pt"
    with open(T.RUNS / f"{base}.json") as f:
        c0 = json.load(f)["config"]
    c = dict(c0, lr=lr, seed=seed, start=base, mid=arm, tokens=float(tokens))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tk = T.tokenizer(c["tok"])
    V = tk.get_vocab_size()
    tr, held, jung = T.streams(c["tok"], c["data"])
    bad = T.bad_runs(tr)

    proc_tr = np.memmap(MID / "proc_train.bin", dtype=np.uint16, mode="r") if arm in ("D1", "D3") else None
    chat_tr = np.memmap(MID / "chat_train.bin", dtype=np.uint16, mode="r") if arm in ("D2", "D3") else None
    chat_held = np.memmap(MID / "chat_held.bin", dtype=np.uint16, mode="r")

    m = T.Toy(c, V).to(T.DEV)
    m.load_state_dict(torch.load(T.PRIV / "ckpt" / f"{base}.pt", map_location=T.DEV, weights_only=True))
    B, ctx, mb = c["batch"], c["ctx"], c["mb"]
    assert B % 4 == 0, "batch size must be divisible by 4 for 25/25/50 mixture"
    for arr in (proc_tr, chat_tr, chat_held):
        if arr is not None:
            assert len(arr) > ctx + 1, f"stream too short ({len(arr)}) for ctx={ctx}"
    steps = max(1, int(round(tokens / (B * ctx))))
    warm = max(1, steps // 20)

    dec = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": c["wd"]}, {"params": nod, "weight_decay": 0.0}], lr, betas=(0.9, 0.95))

    def draw_stream(arr, n):
        idx = rng.integers(0, len(arr) - ctx - 1, n)
        return np.stack([arr[j:j + ctx + 1] for j in idx]).astype(np.int64)

    def draw_replay(n):
        idx = T.draw(rng, n, len(tr) - ctx - 1, ctx, bad)
        return np.stack([tr[j:j + ctx + 1] for j in idx]).astype(np.int64)

    print(f"{name}: arm {arm} base {base} seed {seed} steps {steps} ({steps * B * ctx / 1e6:.2f}M toks) dev {T.DEV}", flush=True)
    t0 = time.time()
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * ((s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
        if arm == "D0":
            xb = draw_replay(B)
        elif arm == "D1":
            xb = np.concatenate([draw_stream(proc_tr, B // 2), draw_replay(B // 2)], 0)
        elif arm == "D2":
            xb = np.concatenate([draw_stream(chat_tr, B // 2), draw_replay(B // 2)], 0)
        else:
            xb = np.concatenate([draw_stream(proc_tr, B // 4), draw_stream(chat_tr, B // 4), draw_replay(B // 2)], 0)
        rng.shuffle(xb, axis=0)

        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, B, mb):
            x = torch.from_numpy(xb[i:i + mb, :-1]).to(T.DEV)
            y = torch.from_numpy(xb[i:i + mb, 1:]).to(T.DEV)
            with T.amp():
                lg = m(x)
            lg = lg.float()
            l = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
            if c["zloss"]:
                l = l + c["zloss"] * torch.logsumexp(lg, -1).pow(2).mean()
            (l * (x.shape[0] / B)).backward()
            tot += l.item() * x.shape[0] / B
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        opt.step()
        assert math.isfinite(tot), "diverged"
        if s % 50 == 0 or s == steps - 1:
            print(f"step {s}/{steps} loss {tot:.4f} gnorm {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} s/step {(time.time() - t0) / (s + 1):.2f}", flush=True)

    torch.save(m.state_dict(), T.PRIV / "ckpt" / f"{name}.pt")
    ev = T.evaluate(m, c, tk, held, jung)
    ev["chat_held"] = eval_stream_bpb(m, c, tk, chat_held, T.DEV, T.amp)
    samples = T.continuations(m, c, tk, held)
    info = {"name": name, "config": c, "dev": T.DEV, "amp": T.AMP, "steps": steps, "tokens_seen": steps * B * ctx,
            "nonembed": T.nonembed(m), "train_s": round(time.time() - t0), "last_train_loss": tot, **ev, "samples": samples}
    with open(T.RUNS / f"{name}.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"books bpb {ev['books']['bpb']:.4f} | jung bpb {ev['jung']['bpb']:.4f} | chat_held bpb {ev['chat_held']['bpb']:.4f}", flush=True)
    print("MIDTRAIN_DONE", name, flush=True)


def score_proc(names):
    """score held-out procedural reading items (fluent/mid/proc_held.jsonl) with exact same length-normalized log-prob rule as comp.py"""
    import torch
    sys.path.insert(0, str(HERE))
    import train as T

    with open(MID / "proc_held.jsonl") as f:
        items = [json.loads(l) for l in f]
    kinds = sorted({r["kind"] for r in items})
    for name in names:
        with open(T.RUNS / f"{name}.json") as f:
            c = json.load(f)["config"]
        tk = T.tokenizer(c["tok"])
        m = T.Toy(c, tk.get_vocab_size()).to(T.DEV)
        m.load_state_dict(torch.load(T.PRIV / "ckpt" / f"{name}.pt", map_location=T.DEV, weights_only=True))
        m.eval()
        start = tk.encode("\n\n").ids
        t0 = time.time()

        @torch.no_grad()
        def lp_mean(ctx, cont):
            a = start + tk.encode(ctx).ids if ctx else list(start)
            b = tk.encode(cont).ids
            ids = (a + b)[-(c["ctx"] + 1):]
            k = len(b)
            x = torch.tensor([ids[:-1]], device=T.DEV)
            with T.amp():
                lg = m(x)[0].float()
            l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=T.DEV)[:, None])[:, 0]
            return float(l.mean())

        def credit(scores):
            b = max(scores)
            tied = [i for i, x in enumerate(scores) if x == b]
            return (1.0 / len(tied)) if 0 in tied else 0.0

        res = {}
        all_w, all_wo, per_item = [], [], []
        for kind in kinds:
            sub = [r for r in items if r["kind"] == kind]
            w = [credit([lp_mean(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in sub]
            wo = [credit([lp_mean(f"Question: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in sub]
            all_w.extend(w)
            all_wo.extend(wo)
            for r, a_, b_ in zip(sub, w, wo):
                per_item.append([r["kind"], r["slot"], a_, b_])
            res[kind] = {"n": len(sub), "with": float(np.mean(w)), "without": float(np.mean(wo)), "gain": float(np.mean(w) - np.mean(wo))}
        res["overall"] = {"n": len(all_w), "with": float(np.mean(all_w)), "without": float(np.mean(all_wo)), "gain": float(np.mean(all_w) - np.mean(all_wo))}
        res["items"] = per_item
        res["s"] = round(time.time() - t0)
        (T.RUNS / "comp").mkdir(exist_ok=True)
        with open(T.RUNS / "comp" / f"{name}_proc.json", "w") as f:
            json.dump(res, f, indent=1)
        print(name, "proc_held overall", res["overall"], flush=True)


def gen_chat(names, max_new=200, temp=0.8, topk=50, seed=0):
    """generate replies on 100 held OASST2 user prompts; stops at '\\nUser:' or max_new tokens"""
    import torch
    sys.path.insert(0, str(HERE))
    import train as T

    with open(MID / "chat_held_prompts.jsonl") as f:
        prompts = [json.loads(l) for l in f]
    (T.RUNS / "gen").mkdir(parents=True, exist_ok=True)
    for name in names:
        with open(T.RUNS / f"{name}.json") as f:
            c = json.load(f)["config"]
        tk = T.tokenizer(c["tok"])
        m = T.Toy(c, tk.get_vocab_size()).to(T.DEV)
        m.load_state_dict(torch.load(T.PRIV / "ckpt" / f"{name}.pt", map_location=T.DEV, weights_only=True))
        m.eval()
        rows = []
        t0 = time.time()
        with torch.no_grad():
            for r in prompts:
                g = torch.Generator(device="cpu").manual_seed(seed + int(r["pid"]))
                p_ids = tk.encode(r["prompt"]).ids
                ids = list(p_ids)
                ents = []
                for _ in range(max_new):
                    with T.amp():
                        lg = m(torch.tensor([ids[-c["ctx"]:]], device=T.DEV))[0, -1].float().cpu()
                    pr = torch.softmax(lg / temp, -1)
                    ents.append(float(-(pr * pr.clamp_min(1e-12).log()).sum()))
                    top = torch.topk(pr, topk)
                    nxt = int(top.indices[torch.multinomial(top.values / top.values.sum(), 1, generator=g)])
                    ids.append(nxt)
                    if len(ids) - len(p_ids) >= 5 and "\nUser:" in tk.decode(ids[-10:]):
                        break
                reply = tk.decode(ids[len(p_ids):])
                if "\nUser:" in reply:
                    reply = reply.split("\nUser:", 1)[0]
                rows.append({"arm": name, "pid": r["pid"], "user": r["user"], "reply": reply.strip(), "mean_entropy": float(np.mean(ents))})
        with open(T.RUNS / "gen" / f"{name}_chat.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"GEN_CHAT_DONE {name} ({len(rows)} replies in {time.time() - t0:.0f} s)", flush=True)


CHAT_JUDGE_ASK = (
    "Two replies to the same user message are shown. Which reply is more helpful, coherent, relevant to the user's request, "
    "and natural as a conversational assistant? Ignore which one is longer. Penalize repetitive loops or off-topic rambling. "
    "Answer with one letter, A or B.\n\nUser: {u}\n\nA: {a}\n\nB: {b}"
)


def judge_chat(a, b):
    """Mac, MLX: blind both-order pairwise chat judge (Qwen3.8 27B) + rep4/distinct-n; refuses if Inkling llama-server is running"""
    if subprocess.run(["pgrep", "-f", "llama-server.*Inkling"], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("stop Inkling on the Mac before loading Qwen3.8 27B (64 GB memory rule)")
    import mlx.core as mx
    from mlx_lm import load
    sys.path.insert(0, str(HERE))
    import kit

    mx.set_cache_limit(4 * 2 ** 30)
    m, tk = load(str(kit.JUDGE))
    ia, ib = tk.encode("A")[-1], tk.encode("B")[-1]

    def load_c(arm):
        with open(RUNS / "gen" / f"{arm}_chat.jsonl") as f:
            return {r["pid"]: r for r in (json.loads(l) for l in f)}

    def p_first(u, x, y):
        msg = [{"role": "user", "content": CHAT_JUDGE_ASK.format(u=u, a=x, b=y)}]
        try:
            s = tk.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        except TypeError:
            s = tk.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        lg = m(mx.array([tk.encode(s)]))[0, -1].astype(mx.float32)
        diff = max(-60.0, min(60.0, float(lg[ib]) - float(lg[ia])))
        return 1.0 / (1.0 + math.exp(diff))

    RA, RB = load_c(a), load_c(b)
    pids = sorted(set(RA) & set(RB))
    assert len(pids) > 0, f"no shared pids between {a} and {b}"
    pref = []
    t0 = time.time()
    for pid in pids:
        assert RA[pid]["user"] == RB[pid]["user"]
        u, x, y = RA[pid]["user"], RA[pid]["reply"], RB[pid]["reply"]
        pref.append(0.5 * (p_first(u, x, y) + (1.0 - p_first(u, y, x))))

    def surf(R):
        ts = [R[pid]["reply"] for pid in pids]
        return {"rep4": round(float(np.mean([kit.rep4(t) for t in ts])), 4),
                "distinct2": round(kit.distinct(ts, 2), 4),
                "distinct3": round(kit.distinct(ts, 3), 4),
                "mean_words": round(float(np.mean([len(t.split()) for t in ts])), 1),
                "empty": int(sum(not t for t in ts))}

    out = {"a": a, "b": b, "n": len(pref), "p_a_preferred": round(float(np.mean(pref)), 4), "ci": kit.boot(pref),
           "a_wins": int(sum(p > 0.5 for p in pref)), "surf_a": surf(RA), "surf_b": surf(RB), "s": round(time.time() - t0)}
    KIT.mkdir(parents=True, exist_ok=True)
    with open(KIT / f"chat_pair_{a}_vs_{b}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("CHAT_JUDGE_DONE", a, "vs", b, out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        kw = dict(kv.split("=") for kv in sys.argv[3:])
        train(sys.argv[2], kw["arm"], int(kw.get("seed", 1)), base=kw.get("base", "C25"),
              tokens=float(kw.get("tokens", 50e6)), lr=float(kw.get("lr", 3e-4)))
    elif cmd == "eval_chat":
        eval_chat(sys.argv[2:])
    elif cmd == "score_proc":
        score_proc(sys.argv[2:])
    elif cmd == "gen_chat":
        gen_chat(sys.argv[2:])
    elif cmd == "judge_chat":
        judge_chat(sys.argv[2], sys.argv[3])
    else:
        raise ValueError(f"unknown cmd {cmd}")
