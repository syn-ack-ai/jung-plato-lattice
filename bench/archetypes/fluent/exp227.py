"""Addendum 227 (Amendment 227a): Wake -> NREM Sleep -> REM Dream continual learning engine & 5-Day Benchmark ("5 Days in the Life of a Toy Mind").
Builds directly on Addendum 226's unified OnlineToy fast-weight API (O1..O4) and exposes a reusable `WakeSleepEngine`
for both controlled continual-learning benchmarks and real interactive chat/tool sessions.

Arms (paired across all 3 seeds of the winning Addendum 226 online-learning arm):
  W0: Wake-Only (Insomnia)          — fast weights adapt across all 5 days, never sleeps, never resets; slow weights frozen
  W1: Naive Nightly Fine-Tune       — fine-tunes slow weights on today's 10 episodes only (no replay/KL/EWC), resets fast weights
  W2: Wake + NREM Sleep             — interleaved replay (30% surprise-weighted episodic + 70% corpus) + KL anchor + diagonal Fisher EWC + fast-to-slow distill + reset
  W3: Wake + NREM + REM Dreaming    — W2 + perspective dreams + cross-day shared-entity recombination + self-consistency-gated free dreams
  W4: Autonomous Self-Dreaming      — W3 with 100% self-generated general replay AND self-generated Day-0 Fisher (zero access to pretraining corpus files)

Usage:
  FL_DEV=cpu python fluent/exp227.py test              unit tests + 2-day smoke test of W0..W4 across O1..O4
  python fluent/exp227.py pick                         pick winning O1..O4 arm from completed Addendum 226 runs
  python fluent/exp227.py run O4_s1 arm=W3             run full 5-Day continual learning benchmark on a 226 checkpoint

Public text only for replay/training; Jung held-out bpb is evaluated locally only as a forgetting guardrail and never printed or exported."""
import copy, json, math, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from fluent import exp226

RUNS = HERE / "runs"
WRUNS = RUNS / "w227"
COMP = HERE / "comp"
MID = HERE / "mid"
PRIV = Path(os.environ.get("PLATONIC_CKPT_DIR", str(HERE / "local_ckpt")))
DEV = exp226.DEV
AMP = exp226.AMP

WRUNS.mkdir(parents=True, exist_ok=True)

# Freeze Wake's online-plasticity meta-params AND unused-during-sleep O1 v_proj / O4 adapter scale during NREM sleep (Amendment 227a)
WAKE_META_FROZEN_SUBSTR = ("eta_logit", "gamma_logit", "gate_proj", "scale", "v_proj")
FORBIDDEN_EVAL_WORDS = ("leader", "commands", "stronghold", "watch")

# ---------------------------------------------------------------- 5-Day Continual Learning World (100% disjoint from eval sets)

def _words(t):
    return re.findall(r"[a-z0-9']+", t.lower().replace("\u2019", "'"))


def _grams12(t):
    w = _words(t)
    return {tuple(w[i:i + 12]) for i in range(len(w) - 11)}


def _load_banned_words_and_12grams():
    """collect every word and 12-gram in proc_held, mcq2, and mcq so the 5-Day world is strictly disjoint"""
    banned_w = set(FORBIDDEN_EVAL_WORDS)
    guard_g = set()
    for p, keys in (
        (MID / "proc_held.jsonl", ("passage", "question", "answer")),
        (COMP / "mcq2.jsonl", ("passage", "question", "answer")),
        (COMP / "mcq.jsonl", ("passage", "question", "answer")),
    ):
        if p.exists():
            with open(p) as f:
                for l in f:
                    r = json.loads(l)
                    for k in keys:
                        if k in r:
                            banned_w.update(_words(r[k]))
                            guard_g |= _grams12(r[k])
                    for w in r.get("wrong", []):
                        banned_w.update(_words(w))
    return banned_w, guard_g


def _invent_tokens(rng, n, banned, style="person"):
    """generate pronounceable invented names strictly disjoint from banned words, eval cue substrings, and each other"""
    if style == "person":
        onsets = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "k", "l", "m", "n", "p", "r", "s", "t", "tr", "v", "w", "z"]
        vowels = ["a", "e", "i", "o", "u", "ai", "eo"]
        codas = ["n", "r", "l", "m", "s", "t", "k", "d", "v", "th", "nd"]
    elif style == "mediator":
        onsets = ["ch", "cl", "cr", "fl", "gl", "kr", "pl", "pr", "qu", "sh", "sk", "sl", "sm", "sp", "st", "th", "vr", "wr", "zh"]
        vowels = ["a", "e", "i", "o", "u", "ae", "ou"]
        codas = ["n", "r", "l", "x", "z", "rd", "lm", "rn", "st", "rk"]
    else:  # stronghold (Amendment 227a: removed 'watch'/'hold' codas so invented names never contain eval cues!)
        onsets = ["al", "ar", "bel", "cor", "dal", "el", "fal", "gal", "hel", "il", "jor", "kal", "lor", "mar", "nor", "or", "pel", "ral", "sel", "tal", "ul", "val", "wel", "zor"]
        vowels = ["a", "e", "i", "o", "u"]
        codas = ["wick", "crag", "spire", "keep", "haven", "mere", "ford", "vale", "crest", "tor", "gate", "ridge", "moor", "fell", "stead"]

    out = []
    seen = set(banned)
    while len(out) < n:
        if style == "stronghold":
            w = rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas)
        else:
            syls = 2 if rng.random() < 0.7 else 3
            w = "".join(rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas) for _ in range(syls))
        wl = w.lower()
        if 5 <= len(wl) <= 12 and wl not in seen and not any(fw in wl for fw in FORBIDDEN_EVAL_WORDS):
            seen.add(wl)
            out.append(w.capitalize())
    return out


def build_five_day_world(seed=2270):
    """Build 50 disjoint facts across 5 days (10/day) forming 25 cross-day two-hop bridges (P_i -> M_i -> S_i).
    Every bridge has Link 1 on an earlier day and Link 2 on a later day (1- or 2-day gap).
    Eval paraphrase & 2-hop question templates are strictly disjoint from both Wake and REM dream wording!"""
    banned_w, guard_g = _load_banned_words_and_12grams()
    rng = random.Random(seed)
    used = set(banned_w)
    persons = _invent_tokens(rng, 25, used, style="person")
    used.update(w.lower() for w in persons)
    mediators = _invent_tokens(rng, 25, used, style="mediator")
    used.update(w.lower() for w in mediators)
    strongholds = _invent_tokens(rng, 25, used, style="stronghold")
    used.update(w.lower() for w in strongholds)

    link1_day = [0] * 10 + [1] * 5 + [2] * 5 + [3] * 5
    link2_day = [1] * 5 + [2] * 5 + [3] * 5 + [4] * 10

    l1_by_day = defaultdict(list)
    l2_by_day = defaultdict(list)
    for i in range(25):
        l1_by_day[link1_day[i]].append(i)
        l2_by_day[link2_day[i]].append(i)

    def pick_wrongs(idx, pool, arr):
        cands = [arr[j] for j in pool if j != idx]
        return rng.sample(cands, 3)

    days = [{"day": d, "episodes": [], "exact_tests": [], "para_tests": [], "bridge_tests": []} for d in range(5)]

    for i in range(25):
        p, m, s = persons[i], mediators[i], strongholds[i]
        d1, d2 = link1_day[i], link2_day[i]
        assert d2 > d1, "every bridge must cross days!"

        ep1 = {
            "id": f"L1_{i:02d}", "day": d1, "kind": "link1", "chain": i,
            "subj": p, "rel": "answers_to", "obj": m,
            "wake_q": f"Who does {p} answer to?",
            "wake_a": m,
            "wake_text": f"Question: Who does {p} answer to?\nAnswer: {m}\n\n",
            "fact": f"{p} answers to {m}.",
        }
        w1 = pick_wrongs(i, l1_by_day[d1], mediators)
        days[d1]["episodes"].append(ep1)
        days[d1]["exact_tests"].append({
            "id": ep1["id"], "day": d1, "kind": "exact_l1",
            "question": f"Who does {p} answer to?", "answer": m, "wrong": w1,
        })
        days[d1]["para_tests"].append({
            "id": ep1["id"], "day": d1, "kind": "para_l1",
            "question": f"Which leader commands {p}?", "answer": m, "wrong": w1,
        })

        ep2 = {
            "id": f"L2_{i:02d}", "day": d2, "kind": "link2", "chain": i,
            "subj": m, "rel": "guards", "obj": s,
            "wake_q": f"What place does {m} guard?",
            "wake_a": s,
            "wake_text": f"Question: What place does {m} guard?\nAnswer: {s}\n\n",
            "fact": f"{m} guards {s}.",
        }
        w2 = pick_wrongs(i, l2_by_day[d2], strongholds)
        days[d2]["episodes"].append(ep2)
        days[d2]["exact_tests"].append({
            "id": ep2["id"], "day": d2, "kind": "exact_l2",
            "question": f"What place does {m} guard?", "answer": s, "wrong": w2,
        })
        days[d2]["para_tests"].append({
            "id": ep2["id"], "day": d2, "kind": "para_l2",
            "question": f"Which stronghold is under {m}'s watch?", "answer": s, "wrong": w2,
        })

        days[d2]["bridge_tests"].append({
            "id": f"B2_{i:02d}", "day_start": d1, "day_end": d2, "gap": d2 - d1, "chain": i,
            "question": f"Which stronghold is watched by the leader of {p}?",
            "answer": s, "wrong": w2,
            "bridge_entity": m, "subj": p,
        })

    # Verify zero 12-gram overlap with existing eval guards
    for d in days:
        assert len(d["episodes"]) == 10
        for e in d["episodes"]:
            assert not (_grams12(e["wake_text"]) & guard_g), f"12-gram collision on {e['id']}"
        rng.shuffle(d["episodes"])
    return {"seed": seed, "persons": persons, "mediators": mediators, "strongholds": strongholds, "days": days}


# ---------------------------------------------------------------- State utilities

def _detach_state(st):
    """Return a clean detached copy of an OnlineToy fast-weight state (zero lingering autograd graph!)."""
    out = []
    for item in st:
        if item is None:
            out.append(None)
        elif isinstance(item, tuple):
            out.append(tuple(t.detach().clone() for t in item))
        else:
            out.append(item.detach().clone())
    return out


def _expand_state(st, B):
    """Expand a B=1 OnlineToy fast-weight state list to batch size B cleanly."""
    if st is None:
        return None
    out = []
    for item in st:
        if item is None:
            out.append(None)
        elif isinstance(item, tuple):
            out.append(tuple(t.detach().expand(B, *t.shape[1:]).contiguous() for t in item))
        else:
            out.append(item.detach().expand(B, *item.shape[1:]).contiguous())
    return out


# ---------------------------------------------------------------- Reusable Wake-NREM-REM Continual Learning Engine

class WakeSleepEngine:
    """Complementary Learning Systems controller wrapping any OnlineToy model (O1..O4):
      - WAKE: fast-weight online learning (`adapt`) + surprise-tagged episodic buffer (slow weights frozen)
      - REM:  perspective/paraphrase dreams + cross-day shared-entity bridge recombination + self-consistency-gated free dreams
      - NREM: surprise-weighted episodic + neocortical replay + token KL anchor + core diagonal Fisher EWC + fast-to-slow distill + synaptic reset
    Designed so the exact same engine runs both Addendum 227's 5-day benchmark and live interactive user sessions."""

    def __init__(self, model, cfg, tk, dev=DEV, amp_mode=AMP, seed=1):
        self.m = model
        self.c = cfg
        self.tk = tk
        self.V = tk.get_vocab_size()
        self.dev = dev
        self.amp_mode = amp_mode
        self.use_bf16 = (amp_mode == "bf16" and str(dev).startswith("cuda"))
        self.start_ids = tk.encode("\n\n").ids
        self.rng = np.random.default_rng(seed + 22700)
        self.py_rng = random.Random(seed + 22700)

        self.fast_state = None
        self.reset_fast_weights()
        self.episodes = []          # list of dicts with {wake_text, day, surprise, subj, rel, obj, fact, ...}
        self.core_theta0 = None     # frozen Day-0 parameters for Fisher EWC
        self.core_fisher = None     # normalized diagonal Fisher (mean ~ 1.0 on active slow params)

    def _amp(self):
        dev_type = str(self.dev).split(":")[0]
        return torch.autocast(dev_type, dtype=torch.bfloat16, enabled=self.use_bf16)

    def _encode_seq(self, text, max_len=None):
        lim = max_len or self.c["ctx"]
        return (self.start_ids + self.tk.encode(text).ids)[-lim:]

    def _encode_qa(self, q_prefix, ans_cont, max_len=None):
        lim = (max_len or self.c["ctx"]) + 1
        a = self.start_ids + self.tk.encode(q_prefix).ids
        b = self.tk.encode(ans_cont).ids
        ids = (a + b)[-lim:]
        t0 = max(1, len(ids) - len(b))
        return ids, t0

    @torch.no_grad()
    def reset_fast_weights(self):
        """Synaptic homeostasis: wipe fast hippocampal state back to detached baseline (zero autograd graph!)."""
        self.fast_state = _detach_state(self.m.init_state(1))

    @torch.no_grad()
    def compute_core_fisher(self, replay_batch_fn, n_batches=32, batch_size=4, seq_len=256):
        """Compute normalized diagonal Fisher information on general pre-sleep windows (mean F ~ 1.0 on active slow params)."""
        self.m.eval()
        self.core_theta0 = {n: p.detach().float().clone() for n, p in self.m.named_parameters()}
        fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.m.named_parameters()}

        with torch.enable_grad():
            for _ in range(n_batches):
                xb = replay_batch_fn(batch_size, seq_len)
                x = torch.from_numpy(xb[:, :-1]).to(self.dev)
                y = torch.from_numpy(xb[:, 1:]).to(self.dev)
                self.m.zero_grad(set_to_none=True)
                with self._amp():
                    lg, _ = self.m.forward_with_state(x, state=None, update=False)
                loss = F.cross_entropy(lg.float().reshape(-1, self.V), y.reshape(-1))
                loss.backward()
                for n, p in self.m.named_parameters():
                    if p.grad is not None and not any(k in n for k in WAKE_META_FROZEN_SUBSTR):
                        fisher[n].add_(p.grad.detach().float().pow(2) / float(n_batches))
        self.m.zero_grad(set_to_none=True)

        # Normalize over active non-zero slow parameters so mean(F) == 1.0 and ewc_lam is directly interpretable
        active_sum = sum(v.sum().item() for n, v in fisher.items() if not any(k in n for k in WAKE_META_FROZEN_SUBSTR))
        active_cnt = sum(int((v > 0).sum().item()) for n, v in fisher.items() if not any(k in n for k in WAKE_META_FROZEN_SUBSTR))
        scale = max(1e-12, active_sum / max(1, active_cnt))
        self.core_fisher = {n: (v / scale).clamp_max(100.0) for n, v in fisher.items()}
        return scale

    @torch.no_grad()
    def wake_episode(self, ep, record=True):
        """WAKE step: measure surprise (NLL before update), update live fast weights ONLY (slow weights frozen), log to episodic buffer."""
        self.m.eval()
        ids = self._encode_seq(ep["wake_text"])
        x = torch.tensor([ids], device=self.dev)
        with self._amp():
            lg, _ = self.m.forward_with_state(x[:, :-1], state=self.fast_state, update=False)
            nll = float(F.cross_entropy(lg.float().reshape(-1, self.V), x[:, 1:].reshape(-1)).item())
            new_st = self.m.adapt(x, state=self.fast_state)
        self.fast_state = _detach_state(new_st)
        if record:
            rec = dict(ep)
            rec["surprise"] = round(nll, 4)
            self.episodes.append(rec)
        return nll

    @torch.no_grad()
    def score_option_cold(self, question, option, use_fast_state=False):
        """Context-cleared cold scoring of a single option. use_fast_state=False scores pure slow weights (morning after sleep!)."""
        self.m.eval()
        ctx_s = f"Question: {question}\nAnswer:"
        a = self.start_ids + self.tk.encode(ctx_s).ids
        b = self.tk.encode(" " + option).ids
        k = len(b)
        ids = (a + b)[-(self.c["ctx"] + 1):]
        x = torch.tensor([ids[:-1]], device=self.dev)
        st = self.fast_state if use_fast_state else None
        with self._amp():
            lg, _ = self.m.forward_with_state(x, state=st, update=False)
        l = torch.log_softmax(lg[0, -k:].float(), -1).gather(1, torch.tensor(ids[-k:], device=self.dev)[:, None])[:, 0]
        return float(l.mean())

    @torch.no_grad()
    def eval_mcq_list(self, items, use_fast_state=False):
        if not items:
            return None
        scores = []
        for r in items:
            opts = [r["answer"]] + r["wrong"]
            sc = [self.score_option_cold(r["question"], o, use_fast_state=use_fast_state) for o in opts]
            best = max(sc)
            tied = [i for i, v in enumerate(sc) if v == best]
            scores.append((1.0 / len(tied)) if 0 in tied else 0.0)
        return round(float(np.mean(scores)), 4)

    @torch.no_grad()
    def _sample_batch(self, model_ref, prompt_ids_2d, max_new, temp=0.85, top_k=50):
        """Batched autoregressive sampling (Amendment 227a: ~50x faster than B=1 loop for W4 self-replay & REM dreams!)."""
        model_ref.eval()
        x = torch.tensor(prompt_ids_2d, dtype=torch.long, device=self.dev)
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(int(self.rng.integers(0, 2**31 - 1)))
        for _ in range(max_new):
            x_in = x[:, -self.c["ctx"]:]
            with self._amp():
                lg, _ = model_ref.forward_with_state(x_in, state=None, update=False)
            logits = lg[:, -1, :].float() / max(0.1, temp)
            if top_k > 0 and top_k < logits.numel():
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
            x = torch.cat([x, nxt], dim=1)
        return x.cpu().tolist()

    @staticmethod
    def _rep4_rate(ids):
        if len(ids) < 8:
            return 0.0
        g4 = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
        return 1.0 - (len(set(g4)) / len(g4))

    @torch.no_grad()
    def generate_self_replay_pool(self, model_ref, n_seqs=96, seq_len=256, batch_size=32):
        """W4 autonomous generative replay: batched sampling from frozen pre-sleep model (100% zero corpus files!)."""
        # Pick guaranteed single-token natural starters so every row in a batch has identical prompt length
        raw_starters = ["The", "In", "When", "She", "He", "They", "After", "Before", "Once", "A", "As", "It", "There", "One"]
        starters = []
        for s in raw_starters:
            tids = self.tk.encode(s).ids
            if len(tids) == 1:
                starters.append(self.start_ids + tids)
        if not starters:
            starters = [self.start_ids + [self.tk.encode("The").ids[0]]]
        p_len = len(starters[0])
        need = (seq_len + 1) - p_len

        pool = []
        n_tried = 0
        # Pass 1: clean temp=0.85, rep4 < 0.25; Pass 2 fallback if needed: temp=1.0, rep4 < 0.35
        for temp_use, rep_max, max_rounds in ((0.85, 0.25, 4), (1.0, 0.35, 3)):
            for _ in range(max_rounds):
                if len(pool) >= n_seqs:
                    break
                bs = min(batch_size, max(8, (n_seqs - len(pool)) * 2))
                prompts = [starters[int(self.rng.integers(0, len(starters)))] for _ in range(bs)]
                seqs = self._sample_batch(model_ref, prompts, max_new=need, temp=temp_use, top_k=50)
                n_tried += bs
                for ids in seqs:
                    if len(ids) >= seq_len + 1 and self._rep4_rate(ids[p_len:]) < rep_max:
                        pool.append(np.asarray(ids[:seq_len + 1], dtype=np.int64))
                        if len(pool) >= n_seqs:
                            break
            if len(pool) >= n_seqs // 2:
                break

        if not pool:
            prompts = [starters[0]] * min(8, n_seqs)
            seqs = self._sample_batch(model_ref, prompts, max_new=need, temp=1.0, top_k=50)
            n_tried += len(seqs)
            pool = [np.asarray(ids[:seq_len + 1], dtype=np.int64) for ids in seqs]
        return np.stack(pool, axis=0), round(len(pool) / max(1, n_tried), 3)

    @torch.no_grad()
    def _seq_nll(self, model_ref, text, state=None):
        ids = self._encode_seq(text)
        x = torch.tensor([ids], device=self.dev)
        with self._amp():
            lg, _ = model_ref.forward_with_state(x[:, :-1], state=state, update=False)
        return float(F.cross_entropy(lg.float().reshape(-1, self.V), x[:, 1:].reshape(-1)).item())

    @torch.no_grad()
    def build_rem_dreams(self, model_ref, wake_fast_state, current_day, include_free=True):
        """REM Dreaming phase (Amendment 227a):
          1. Perspective/paraphrase dreams using generic REM-only wording (STRICTLY disjoint from FORBIDDEN_EVAL_WORDS!)
          2. Cross-day associative bridge recombination for BOTH newly completed bridges and replay of previously completed bridges
          3. Model-sampled free associative dream continuations gated by strict Wake-adapted all-occurrence swap preference."""
        today_eps = [e for e in self.episodes if e["day"] == current_day]
        past_eps = [e for e in self.episodes if e["day"] < current_day]
        dreams = []
        n_free_kept = 0
        n_free_tried = 0

        # 1. Generic perspective & reversed-relation dreams
        for e in today_eps:
            s, o = e["subj"], e["obj"]
            if e["rel"] == "answers_to":
                dreams.append((f"Record: {s} reports directly to", f" {o}.\n\n"))
                dreams.append((f"Fact: {o} directs", f" {s}.\n\n"))
                dreams.append((f"Question: To whom does {s} report?\nAnswer:", f" {o}\n\n"))
            elif e["rel"] == "guards":
                dreams.append((f"Record: {o} is protected by", f" {s}.\n\n"))
                dreams.append((f"Fact: The keeper of {o} is", f" {s}.\n\n"))
                dreams.append((f"Question: Which place is protected by {s}?\nAnswer:", f" {o}\n\n"))
            else:
                dreams.append((f"Memory: {e['fact']}\nRecall:", f" {e.get('wake_a', o)}\n\n"))

        # 2. Cross-day associative bridge recombination (newly completed bridges + lighter replay of earlier bridges)
        all_objs = sorted({e["obj"] for e in self.episodes if e["rel"] == "guards"})
        all_eps = list(self.episodes)
        by_subj = defaultdict(list)
        for e in all_eps:
            by_subj[e["subj"]].append(e)

        seen_bridges = set()
        new_bridges, old_bridges = [], []
        for ea in all_eps:
            for eb in by_subj.get(ea["obj"], ()):
                if ea["day"] == eb["day"]:
                    continue
                key = (ea["id"], eb["id"])
                if key in seen_bridges:
                    continue
                seen_bridges.add(key)
                if max(ea["day"], eb["day"]) == current_day:
                    new_bridges.append((ea, eb))
                else:
                    old_bridges.append((ea, eb))

        # Dream all newly completed bridges + up to 5 earlier bridges (reinforces Day-2 bridges on Nights 3..5!)
        chosen_bridges = list(new_bridges)
        if old_bridges:
            chosen_bridges.extend(self.py_rng.sample(old_bridges, min(5, len(old_bridges))))

        for ea, eb in chosen_bridges:
            p, mid, dest = ea["subj"], ea["obj"], eb["obj"]
            dreams.append((
                f"Memory link: {ea['fact']} Also, {eb['fact']} Through {mid}, {p} is tied to",
                f" {dest}.\n\n",
            ))
            dreams.append((
                f"Question: Through {mid}, which place is tied to {p}?\nAnswer:",
                f" {dest}\n\n",
            ))
            dreams.append((
                f"Question: Starting from {p} and following who they report to, what place is reached?\nAnswer:",
                f" {dest}\n\n",
            ))

            # 3. Self-consistency-gated free associative dream (only on newly completed bridges; Amendment 227a fixed!)
            if include_free and (ea, eb) in new_bridges:
                n_free_tried += 1
                seed_txt = f"{ea['fact']} {eb['fact']} So {p} and {dest}"
                p_ids = self._encode_seq(seed_txt)
                gen_ids = self._sample_batch(model_ref, [p_ids], max_new=24, temp=0.8, top_k=40)[0]
                cont_ids = gen_ids[len(p_ids):]
                if self._rep4_rate(cont_ids) < 0.25:
                    cont_txt = self.tk.decode(cont_ids)
                    dream_full = f"{seed_txt}{cont_txt}"
                    swaps = [d for d in all_objs if d != dest]
                    if swaps and dest in dream_full:
                        fake_dest = self.py_rng.choice(swaps)
                        # Replace ALL occurrences of dest so self-contradiction never confounds the gate!
                        fake_full = dream_full.replace(dest, fake_dest)
                        if fake_full != dream_full:
                            nll_true = self._seq_nll(model_ref, dream_full, state=wake_fast_state)
                            nll_fake = self._seq_nll(model_ref, fake_full, state=wake_fast_state)
                            if nll_true < nll_fake:
                                # Preserve bridge clause cleanly in target so dest + continuation are trained!
                                dreams.append((f"Dream: {ea['fact']} {eb['fact']} So {p} and", f" {dest}{cont_txt}\n\n"))
                                n_free_kept += 1

        return dreams, {
            "n_dreams": len(dreams),
            "new_bridges": len(new_bridges),
            "old_bridges_replayed": len(chosen_bridges) - len(new_bridges),
            "free_kept": n_free_kept,
            "free_tried": n_free_tried,
        }

    def _pack_qa_batch(self, pairs, max_len=160):
        """pack list of (prefix, target) pairs into padded batch + target loss mask (1 ONLY on target tokens)"""
        seqs, t0s = [], []
        for pre, tgt in pairs:
            ids, t0 = self._encode_qa(pre, tgt, max_len=max_len)
            seqs.append(ids)
            t0s.append(t0)
        L = max(len(s) for s in seqs)
        B = len(seqs)
        x = torch.zeros((B, L - 1), dtype=torch.long, device=self.dev)
        y = torch.zeros((B, L - 1), dtype=torch.long, device=self.dev)
        msk = torch.zeros((B, L - 1), dtype=torch.float32, device=self.dev)
        for i, (ids, t0) in enumerate(zip(seqs, t0s)):
            n = len(ids) - 1
            x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long, device=self.dev)
            y[i, :n] = torch.tensor(ids[1:], dtype=torch.long, device=self.dev)
            msk[i, max(0, t0 - 1):n] = 1.0
        return x, y, msk

    def _sample_weighted_pairs(self, eps, k):
        """Surprise-weighted episodic replay draw (higher Wake surprise gets higher NREM replay weight)."""
        w = [max(0.25, float(e.get("surprise", 1.0))) for e in eps]
        chosen = self.py_rng.choices(eps, weights=w, k=k)
        return [(f"Question: {e['wake_q']}\nAnswer:", f" {e['wake_a']}\n\n") for e in chosen]

    def sleep_night(self, arm, current_day, replay_batch_fn=None, steps=80, lr=3e-4, kl_w=1.0, ewc_lam=0.25, f2s_w=0.25, replay_seq_len=256):
        """Run one nightly consolidation cycle according to arm W0..W4. Returns telemetry dict."""
        if arm == "W0":
            return {"mode": "insomnia", "steps": 0}

        today_eps = [e for e in self.episodes if e["day"] == current_day]
        past_eps = [e for e in self.episodes if e["day"] < current_day]
        assert today_eps, f"no episodes recorded for day {current_day}"

        # Clear any stale gradients before snapshotting pre-sleep slow model (tiny 5M = 20MB)
        self.m.zero_grad(set_to_none=True)
        m_prev = copy.deepcopy(self.m).to(self.dev).eval()
        for p in m_prev.parameters():
            p.requires_grad_(False)
        wake_st = _detach_state(self.fast_state)

        rem_pairs, rem_stats = [], {"n_dreams": 0, "new_bridges": 0, "old_bridges_replayed": 0, "free_kept": 0, "free_tried": 0}
        if arm in ("W3", "W4"):
            rem_pairs, rem_stats = self.build_rem_dreams(m_prev, wake_st, current_day)

        # W4 builds a 100% self-generated general replay pool from frozen pre-sleep model (zero corpus access!)
        self_replay_pool, self_acc_rate = None, 0.0
        if arm == "W4":
            n_gen = max(32, min(96, steps))
            self_replay_pool, self_acc_rate = self.generate_self_replay_pool(m_prev, n_seqs=n_gen, seq_len=replay_seq_len)

        # Freeze Wake's online-plasticity meta-params (and unused-in-sleep O1 v_proj / O4 scale) during sleep!
        saved_req = {}
        opt_dec, opt_nod = [], []
        for n, p in self.m.named_parameters():
            saved_req[n] = p.requires_grad
            if any(k in n for k in WAKE_META_FROZEN_SUBSTR):
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
                (opt_dec if p.dim() >= 2 else opt_nod).append(p)
        opt = torch.optim.AdamW([{"params": opt_dec, "weight_decay": 0.01}, {"params": opt_nod, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

        self.m.train()
        tot_ep, tot_kl, tot_ewc, tot_f2s = 0.0, 0.0, 0.0, 0.0
        temp_kd = 2.0

        for s in range(steps):
            w_st = max(1, steps // 10)
            if s < w_st:
                cur_lr = lr * (s + 1) / w_st
            else:
                prog = (s - w_st) / max(1, steps - w_st)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * prog))
            for gr in opt.param_groups:
                gr["lr"] = cur_lr

            batch_pairs = []
            if arm == "W1":
                batch_pairs = self._sample_weighted_pairs(today_eps, 8)
            else:
                n_rem = 4 if (arm in ("W3", "W4") and rem_pairs) else 0
                n_nrem = 8 - n_rem
                n_past = sum(1 for _ in range(n_nrem) if past_eps and self.py_rng.random() < 0.45)
                n_today = n_nrem - n_past
                if n_today > 0:
                    batch_pairs.extend(self._sample_weighted_pairs(today_eps, n_today))
                if n_past > 0:
                    batch_pairs.extend(self._sample_weighted_pairs(past_eps, n_past))
                for _ in range(n_rem):
                    batch_pairs.append(self.py_rng.choice(rem_pairs))

            xe, ye, msk = self._pack_qa_batch(batch_pairs)
            opt.zero_grad(set_to_none=True)

            with self._amp():
                lg_e, _ = self.m.forward_with_state(xe, state=None, update=False)
            lg_ef = lg_e.float()
            ce_tok = F.cross_entropy(lg_ef.reshape(-1, self.V), ye.reshape(-1), reduction="none").view_as(ye)
            l_ep = (ce_tok * msk).sum() / msk.sum().clamp_min(1.0)
            loss = l_ep

            # Fast-to-slow logit distillation on today's episodes (W2, W3, W4)
            l_f2s_val = 0.0
            if arm in ("W2", "W3", "W4") and f2s_w > 0.0:
                xt, yt, msk_t = self._pack_qa_batch(self._sample_weighted_pairs(today_eps, 4))
                with torch.no_grad():
                    st4 = _expand_state(wake_st, xt.shape[0])
                    with self._amp():
                        lg_w, _ = m_prev.forward_with_state(xt, state=st4, update=False)
                    p_wake = F.softmax(lg_w.float() / temp_kd, dim=-1)
                with self._amp():
                    lg_s, _ = self.m.forward_with_state(xt, state=None, update=False)
                log_p_slow = F.log_softmax(lg_s.float() / temp_kd, dim=-1)
                kl_f2s_tok = F.kl_div(log_p_slow, p_wake, reduction="none").sum(-1) * (temp_kd ** 2)
                l_f2s = (kl_f2s_tok * msk_t).sum() / msk_t.sum().clamp_min(1.0)
                loss = loss + f2s_w * l_f2s
                l_f2s_val = float(l_f2s.item())

            # Interleaved neocortical replay + explicit per-token KL anchor to pre-sleep model (W2, W3, W4)
            l_kl_val = 0.0
            if arm in ("W2", "W3", "W4"):
                if arm == "W4":
                    idx = self.rng.integers(0, len(self_replay_pool), 8)
                    rb = self_replay_pool[idx]
                else:
                    rb = replay_batch_fn(8, replay_seq_len)
                xr = torch.from_numpy(rb[:, :-1]).to(self.dev)
                yr = torch.from_numpy(rb[:, 1:]).to(self.dev)
                with torch.no_grad():
                    with self._amp():
                        lg_ref, _ = m_prev.forward_with_state(xr, state=None, update=False)
                    p_ref = F.softmax(lg_ref.float(), dim=-1)
                with self._amp():
                    lg_cur, _ = self.m.forward_with_state(xr, state=None, update=False)
                lg_cur_f = lg_cur.float()
                l_rep_ce = F.cross_entropy(lg_cur_f.reshape(-1, self.V), yr.reshape(-1))
                l_kl = F.kl_div(F.log_softmax(lg_cur_f, dim=-1), p_ref, reduction="none").sum(-1).mean()
                loss = 0.30 * loss + 0.70 * (0.5 * l_rep_ce + kl_w * l_kl)
                l_kl_val = float(l_kl.item())

            # Core diagonal Fisher EWC shield (Amendment 227a: removed double /N_params division! Mean(F)=1 -> O(0.1..1.0) penalty)
            l_ewc_val = 0.0
            if arm in ("W2", "W3", "W4") and ewc_lam > 0.0 and self.core_fisher is not None:
                ewc = torch.zeros((), device=self.dev, dtype=torch.float32)
                for n, p in self.m.named_parameters():
                    if p.requires_grad and n in self.core_fisher:
                        f_diag = self.core_fisher[n]
                        if f_diag.max().item() > 0.0:
                            ewc = ewc + (f_diag * (p.float() - self.core_theta0[n]).pow(2)).sum()
                l_ewc = 0.5 * ewc_lam * ewc
                loss = loss + l_ewc
                l_ewc_val = float(l_ewc.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
            opt.step()
            assert math.isfinite(float(loss.item())), f"{arm} diverged during night {current_day}"
            tot_ep += float(l_ep.item()) / steps
            tot_kl += l_kl_val / steps
            tot_ewc += l_ewc_val / steps
            tot_f2s += l_f2s_val / steps

        self.m.zero_grad(set_to_none=True)
        for n, p in self.m.named_parameters():
            p.requires_grad_(saved_req[n])
        self.m.eval()
        self.reset_fast_weights()

        return {
            "mode": arm, "steps": steps,
            "ep_ce": round(tot_ep, 4), "kl": round(tot_kl, 4),
            "ewc": round(tot_ewc, 5), "f2s": round(tot_f2s, 4),
            "rem": rem_stats,
            "self_replay_n": 0 if self_replay_pool is None else int(len(self_replay_pool)),
            "self_replay_acc": self_acc_rate,
        }


# ---------------------------------------------------------------- Guardrail evaluation (supports W0 live fast_state!)

@torch.no_grad()
def _eval_bpb_with_state(m, c, tk, seqs, fast_state=None, update=True, nb=None):
    """Exact twin of exp226.eval_bpb that optionally seeds every window from `fast_state` (needed for W0 Day-5 drift!)."""
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    ctx = c["ctx"]
    arrs = [np.asarray(s[:nb] if nb else s) for s in seqs]
    chunks = np.stack([s[i:i + ctx + 1] for s in arrs for i in range(0, len(s) - ctx - 1, ctx)]).astype(np.int64)
    tot = 0.0
    for i in range(0, len(chunks), c["mb"]):
        x = torch.from_numpy(chunks[i:i + c["mb"], :-1]).to(m_dev)
        y = torch.from_numpy(chunks[i:i + c["mb"], 1:]).to(m_dev)
        st = _expand_state(fast_state, x.shape[0]) if fast_state is not None else None
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg, _ = m.forward_with_state(x, state=st, update=update)
        tot += F.cross_entropy(lg.float().transpose(1, 2), y, reduction="sum").item()
    nbytes = sum(len(tk.decode(ch[1:].tolist()).encode("utf-8")) for ch in chunks)
    return float(tot / math.log(2) / nbytes)


@torch.no_grad()
def _score_proc_with_state(m, c, tk, items, fast_state=None):
    """Exact twin of exp226.score_mcq_items that optionally seeds from `fast_state` (for W0 Day-5 drift!)."""
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    start = tk.encode("\n\n").ids

    def score_one(ctx_s, cont):
        a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
        b = tk.encode(cont).ids
        k = len(b)
        ids = (a + b)[-(c["ctx"] + 1):]
        x = torch.tensor([ids[:-1]], device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg, _ = m.forward_with_state(x, state=fast_state, update=True)
        l = torch.log_softmax(lg[0, -k:].float(), -1).gather(1, torch.tensor(ids[-k:], device=m_dev)[:, None])[:, 0]
        return float(l.mean())

    def credit(scores):
        b = max(scores)
        tied = [i for i, x in enumerate(scores) if x == b]
        return (1.0 / len(tied)) if 0 in tied else 0.0

    w = [credit([score_one(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in items]
    wo = [credit([score_one(f"Question: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in items]
    return float(np.mean(w) - np.mean(wo))


@torch.no_grad()
def eval_guardrails(m, c, tk, tr_held, jung, chat_he, blimp_items, proc_items, fast_state=None, fast_mode=False):
    """Evaluate zero-forgetting guardrails (Amendment 227a: accepts fast_state so W0's true Day-5 insomnia drift is measured!)."""
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    nb_books = 200_000 if fast_mode else 1_000_000

    ev_books = _eval_bpb_with_state(m, c, tk, [tr_held], fast_state=fast_state, update=True, nb=nb_books)
    ev_books_cold = _eval_bpb_with_state(m, c, tk, [tr_held], fast_state=fast_state, update=False, nb=min(200_000, nb_books))
    ev_jung = _eval_bpb_with_state(m, c, tk, jung[:4] if fast_mode else jung, fast_state=fast_state, update=True)
    ev_chat = _eval_bpb_with_state(m, c, tk, [chat_he], fast_state=fast_state, update=True)

    start = tk.encode("\n\n").ids
    b_sub = blimp_items[:200] if fast_mode else blimp_items

    def lp(txt):
        ids = (start + tk.encode(txt).ids)[-(c["ctx"] + 1):]
        x = torch.tensor([ids[:-1]], device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg, _ = m.forward_with_state(x, state=fast_state, update=False)
        k = len(ids) - len(start)
        l = torch.log_softmax(lg[0, -k:].float(), -1).gather(1, torch.tensor(ids[-k:], device=m_dev)[:, None])[:, 0]
        return float(l.sum())

    blimp = float(np.mean([lp(r["good"]) > lp(r["bad"]) for r in b_sub])) if b_sub else None
    p_sub = proc_items[:100] if fast_mode else proc_items
    proc = _score_proc_with_state(m, c, tk, p_sub, fast_state=fast_state) if p_sub else None
    m.eval()
    return {
        "books_bpb": round(ev_books, 4),
        "books_bpb_cold": round(ev_books_cold, 4),
        "jung_bpb": round(ev_jung, 4),
        "chat_bpb": round(ev_chat, 4),
        "blimp": None if blimp is None else round(blimp, 4),
        "proc_gain": None if proc is None else round(proc, 4),
    }


def make_replay_fn(tr, chat_tr, bad, seed):
    rng = np.random.default_rng(seed + 22750)

    def draw(bs, seq_len):
        n_g = bs // 2
        n_c = bs - n_g
        ig = exp226.draw_clean(rng, n_g, len(tr) - seq_len - 1, seq_len, bad)
        ic = rng.integers(0, len(chat_tr) - seq_len - 1, n_c)
        xb = np.concatenate([
            np.stack([tr[j:j + seq_len + 1] for j in ig]),
            np.stack([chat_tr[j:j + seq_len + 1] for j in ic]),
        ], 0).astype(np.int64)
        rng.shuffle(xb, axis=0)
        return xb

    return draw


def pick_226_winner():
    """Inspect completed Addendum 226 runs (O1..O4 x 3 seeds) and pick the best online-learning arm."""
    rows = []
    for arm in ("O1", "O2", "O3", "O4"):
        pgs, rgs, bks, jgs = [], [], [], []
        for s in (1, 2, 3):
            p_tr = RUNS / f"{arm}_s{s}.json"
            p_sc = RUNS / "comp" / f"{arm}_s{s}.json"
            if p_tr.exists() and p_sc.exists():
                d_tr = json.loads(p_tr.read_text())
                d_sc = json.loads(p_sc.read_text())
                bks.append(d_tr["books"]["bpb"])
                jgs.append(d_tr["jung"]["bpb"])
                pgs.append(d_sc["proc"]["overall"]["gain"])
                rgs.append(d_sc["retention"]["mcq2"]["oneshot_gain"])
        if len(pgs) == 3:
            score = float(np.mean(pgs)) + float(np.mean(rgs)) - 0.5 * max(0.0, float(np.mean(bks)) - 1.466)
            rows.append((score, arm, float(np.mean(bks)), float(np.mean(jgs)), float(np.mean(pgs)), float(np.mean(rgs))))
    assert rows, "No completed 3-seed Addendum 226 arms found yet!"
    rows.sort(reverse=True)
    for sc, arm, bk, jg, pg, rg in rows:
        print(f"  {arm}: score={sc:+.4f} books={bk:.4f} jung={jg:.4f} proc_gain={pg:+.4f} retain_1shot={rg:+.4f}", flush=True)
    winner = rows[0][1]
    print(f"WINNER_226 {winner}", flush=True)
    return winner


def run_five_days(base_name, arm, sleep_steps=80, lr=3e-4, kl_w=1.0, ewc_lam=0.25, f2s_w=0.25):
    assert arm in ("W0", "W1", "W2", "W3", "W4"), f"unknown arm {arm}"
    out_path = WRUNS / f"{base_name}_{arm}.json"
    assert not out_path.exists(), f"already exists: {out_path}"

    base_info = json.loads((RUNS / f"{base_name}.json").read_text())
    c = base_info["config"]
    seed = int(c.get("seed", 1))
    tk = exp226.tokenizer()
    V = tk.get_vocab_size()
    m = exp226.OnlineToy(c, V).to(DEV)
    m.load_state_dict(torch.load(PRIV / "ckpt" / f"{base_name}.pt", map_location=DEV, weights_only=True))
    m.eval()

    tr, held, jung, _, chat_tr, chat_he = exp226.streams()
    bad = exp226.bad_runs(tr)
    blimp = [json.loads(l) for l in open(COMP / "blimp.jsonl")] if (COMP / "blimp.jsonl").exists() else []
    proc_held = [json.loads(l) for l in open(MID / "proc_held.jsonl")] if (MID / "proc_held.jsonl").exists() else []

    world = build_five_day_world(seed=2270)
    replay_fn = make_replay_fn(tr, chat_tr, bad, seed=seed)
    eng = WakeSleepEngine(m, c, tk, dev=DEV, amp_mode=AMP, seed=seed)

    print(f"=== W227 base={base_name} arm={arm} seed={seed} steps/night={sleep_steps} ewc={ewc_lam} dev={DEV}", flush=True)
    t0 = time.time()

    # Day 0 baseline guardrails + core Fisher (W4 uses 100% self-generated Day-0 pool -> strict ZERO corpus access!)
    guard_d0 = eval_guardrails(m, c, tk, held, jung, chat_he, blimp, proc_held, fast_state=None)
    if arm in ("W2", "W3") and ewc_lam > 0.0:
        f_scale = eng.compute_core_fisher(replay_fn, n_batches=32, batch_size=4, seq_len=256)
    elif arm == "W4" and ewc_lam > 0.0:
        d0_pool, _ = eng.generate_self_replay_pool(m, n_seqs=96, seq_len=256)
        f_rng = np.random.default_rng(seed + 22799)
        f_scale = eng.compute_core_fisher(lambda bs, sl: d0_pool[f_rng.integers(0, len(d0_pool), bs)], n_batches=32, batch_size=4, seq_len=256)
    else:
        f_scale = 0.0

    all_exact = [r for d in world["days"] for r in d["exact_tests"]]
    all_para = [r for d in world["days"] for r in d["para_tests"]]
    all_bridge = [r for d in world["days"] for r in d["bridge_tests"]]
    pre_scores = {
        "exact_all": eng.eval_mcq_list(all_exact, use_fast_state=False),
        "para_all": eng.eval_mcq_list(all_para, use_fast_state=False),
        "bridge_all": eng.eval_mcq_list(all_bridge, use_fast_state=False),
    }

    daily_logs = []
    seen_exact, seen_para, seen_bridge = [], [], []
    use_fast_eval = (arm == "W0")

    for d_idx, day in enumerate(world["days"]):
        surprises = [eng.wake_episode(ep, record=True) for ep in day["episodes"]]
        wake_today_exact = eng.eval_mcq_list(day["exact_tests"], use_fast_state=True)

        night = eng.sleep_night(
            arm, current_day=d_idx,
            replay_batch_fn=None if arm in ("W0", "W1", "W4") else replay_fn,
            steps=sleep_steps, lr=lr, kl_w=kl_w, ewc_lam=ewc_lam, f2s_w=f2s_w,
        )

        seen_exact.extend(day["exact_tests"])
        seen_para.extend(day["para_tests"])
        seen_bridge.extend(day["bridge_tests"])

        m_today_ex = eng.eval_mcq_list(day["exact_tests"], use_fast_state=use_fast_eval)
        m_today_pa = eng.eval_mcq_list(day["para_tests"], use_fast_state=use_fast_eval)
        m_day1_ex = eng.eval_mcq_list(world["days"][0]["exact_tests"], use_fast_state=use_fast_eval)
        m_all_ex = eng.eval_mcq_list(seen_exact, use_fast_state=use_fast_eval)
        m_all_pa = eng.eval_mcq_list(seen_para, use_fast_state=use_fast_eval)
        m_all_br = eng.eval_mcq_list(seen_bridge, use_fast_state=use_fast_eval) if seen_bridge else None

        rec = {
            "day": d_idx + 1,
            "mean_surprise": round(float(np.mean(surprises)), 4),
            "wake_today_exact": wake_today_exact,
            "morning_today_exact": m_today_ex,
            "morning_today_para": m_today_pa,
            "morning_day1_exact": m_day1_ex,
            "morning_cum_exact": m_all_ex,
            "morning_cum_para": m_all_pa,
            "morning_cum_bridge2hop": m_all_br,
            "night": night,
        }
        daily_logs.append(rec)
        print(
            f"DAY {d_idx + 1} ({base_name}_{arm}): wake={wake_today_exact:.2f} | morning today={m_today_ex:.2f} "
            f"day1={m_day1_ex:.2f} cum_ex={m_all_ex:.2f} cum_pa={m_all_pa:.2f} 2hop={m_all_br if m_all_br is not None else -1:.2f}",
            flush=True,
        )

    # Final Day-5 guardrails (passes live eng.fast_state for W0 insomnia so W0's true forgetting is measured!)
    guard_d5 = eval_guardrails(m, c, tk, held, jung, chat_he, blimp, proc_held, fast_state=(eng.fast_state if arm == "W0" else None))
    drift = {k: None if (guard_d0[k] is None or guard_d5[k] is None) else round(guard_d5[k] - guard_d0[k], 4) for k in guard_d0}

    out = {
        "base": base_name, "arm": arm, "seed": seed,
        "hyper": {"sleep_steps": sleep_steps, "lr": lr, "kl_w": kl_w, "ewc_lam": ewc_lam, "f2s_w": f2s_w, "fisher_scale": f_scale},
        "pre_scores": pre_scores,
        "guard_day0": guard_d0,
        "guard_day5": guard_d5,
        "guard_drift": drift,
        "days": daily_logs,
        "summary": {
            "day5_cum_exact": daily_logs[-1]["morning_cum_exact"],
            "day5_day1_retention": daily_logs[-1]["morning_day1_exact"],
            "day5_cum_para": daily_logs[-1]["morning_cum_para"],
            "day5_bridge2hop": daily_logs[-1]["morning_cum_bridge2hop"],
            "books_drift": drift["books_bpb"],
            "books_cold_drift": drift["books_bpb_cold"],
            "jung_drift": drift["jung_bpb"],
            "blimp_drift": drift["blimp"],
            "proc_drift": drift["proc_gain"],
        },
        "elapsed_s": round(time.time() - t0),
    }
    out_path.write_text(json.dumps(out, indent=1))
    s = out["summary"]
    bd_str = "n/a" if s["blimp_drift"] is None else f"{s['blimp_drift']:+.4f}"
    print(
        f"W227_DONE {base_name}_{arm}: cum_ex={s['day5_cum_exact']:.3f} day1={s['day5_day1_retention']:.3f} "
        f"para={s['day5_cum_para']:.3f} 2hop={s['day5_bridge2hop']:.3f} "
        f"d_books={s['books_drift']:+.4f} d_cold={s['books_cold_drift']:+.4f} d_blimp={bd_str} ({out['elapsed_s']} s)",
        flush=True,
    )


# ---------------------------------------------------------------- Unit tests & multi-architecture smoke check

def self_test():
    """Verify (Amendment 227a):
      1. 5-Day world integrity across ALL 5 days: 50 facts, 25 cross-day bridges, 75 disjoint entities, exact 0.25 overlap,
         AND zero FORBIDDEN_EVAL_WORDS in ALL 5 days' deterministic REM template dreams.
      2. Detached fast_state (requires_grad=False), live non-zero EWC penalty (>1e-4) in W2/W3/W4, strict meta-param preservation
         across ALL 4 base architectures (O1, O2, O3, O4), and exact synaptic reset.
      3. Miniature 2-day end-to-end smoke of ALL 5 arms (W0..W4) including batched W4 self-replay."""
    world = build_five_day_world(seed=2270)
    banned_w, _ = _load_banned_words_and_12grams()
    all_ents = [w.lower() for w in world["persons"] + world["mediators"] + world["strongholds"]]
    assert len(all_ents) == 75 and len(set(all_ents)) == 75, "entity collision!"
    assert not (set(all_ents) & banned_w), "entity leaked from proc_held/mcq2!"
    for w in all_ents:
        for fw in FORBIDDEN_EVAL_WORDS:
            assert fw not in w, f"entity '{w}' contains forbidden eval cue '{fw}'!"

    tot_ep = sum(len(d["episodes"]) for d in world["days"])
    tot_br = sum(len(d["bridge_tests"]) for d in world["days"])
    assert tot_ep == 50 and tot_br == 25
    for d in world["days"]:
        for b in d["bridge_tests"]:
            assert b["gap"] >= 1, "bridge did not cross days!"

    def q_overlap(items):
        tot = 0.0
        for r in items:
            Q = set(_words(r["question"]))
            opts = [r["answer"]] + r["wrong"]
            sc = [len(set(_words(o)) & Q) for o in opts]
            mx = max(sc)
            tied = [i for i, v in enumerate(sc) if v == mx]
            tot += (1.0 / len(tied)) if 0 in tied else 0.0
        return tot / len(items)

    for lbl, key in (("exact", "exact_tests"), ("para", "para_tests"), ("bridge", "bridge_tests")):
        its = [r for d in world["days"] for r in d[key]]
        ov = q_overlap(its)
        assert abs(ov - 0.25) < 1e-6, f"{lbl} overlap baseline != 0.25: {ov}"

    tk = exp226.tokenizer()
    V = tk.get_vocab_size()
    dev_t = "cuda" if torch.cuda.is_available() else "cpu"
    amp_t = "bf16" if dev_t == "cuda" else ""

    # Check ALL 5 days' deterministic REM template dreams for zero forbidden eval words
    c_tmp = dict(exp226.BASE_CFG, **exp226.ARM_SPEC["O2"], arm="O2", ctx=256)
    m_tmp = exp226.OnlineToy(c_tmp, V).to(dev_t)
    eng_tmp = WakeSleepEngine(m_tmp, c_tmp, tk, dev=dev_t, amp_mode=amp_t, seed=1)
    for d_idx, day in enumerate(world["days"]):
        for ep in day["episodes"]:
            eng_tmp.episodes.append(dict(ep, surprise=2.0))
        det_dreams, st_d = eng_tmp.build_rem_dreams(m_tmp, eng_tmp.fast_state, current_day=d_idx, include_free=False)
        blob = " ".join(a + b for a, b in det_dreams).lower()
        for fw in FORBIDDEN_EVAL_WORDS:
            assert fw not in blob, f"Day {d_idx+1} deterministic REM dream leaked eval word '{fw}'!"
    print("World & 5-day REM template check: 50 facts, 25 bridges, 75 clean entities, overlap=0.2500, 0 leakage OK", flush=True)

    def dummy_replay(bs, seq_len):
        return np.random.default_rng(7).integers(10, 500, size=(bs, seq_len + 1), dtype=np.int64)

    def assert_state_detached(st):
        for it in st:
            if it is None:
                continue
            for t in (it if isinstance(it, tuple) else (it,)):
                assert not t.requires_grad, "fast_state retained requires_grad=True!"

    # 1. Cross-architecture compatibility check (O1, O2, O3, O4) on W3 (NREM + REM + EWC + f2s + reset)
    for base_arm in ("O1", "O2", "O3", "O4"):
        c = dict(exp226.BASE_CFG, **exp226.ARM_SPEC[base_arm], arm=base_arm, ctx=256)
        torch.manual_seed(5)
        m = exp226.OnlineToy(c, V).to(dev_t)
        eng = WakeSleepEngine(m, c, tk, dev=dev_t, amp_mode=amp_t, seed=5)
        assert_state_detached(eng.fast_state)
        meta_before = {n: p.detach().clone() for n, p in m.named_parameters() if any(k in n for k in WAKE_META_FROZEN_SUBSTR)}
        sc = eng.compute_core_fisher(dummy_replay, n_batches=2, batch_size=2, seq_len=64)
        assert math.isfinite(sc) and sc > 0
        for ep in world["days"][0]["episodes"][:3]:
            eng.wake_episode(ep, record=True)
        assert_state_detached(eng.fast_state)
        info = eng.sleep_night("W3", current_day=0, replay_batch_fn=dummy_replay, steps=4, lr=3e-4, ewc_lam=0.25, replay_seq_len=64)
        assert info["ewc"] > 1e-4, f"{base_arm} EWC still numerically dead: {info['ewc']}"
        assert_state_detached(eng.fast_state)
        for n, p0 in meta_before.items():
            assert torch.equal(p0, dict(m.named_parameters())[n].detach()), f"{base_arm} modified frozen Wake param {n}!"
        print(f"  Arch {base_arm} W3 check ({dev_t}): ewc={info['ewc']:.5f} kl={info['kl']:.4f} f2s={info['f2s']:.4f} meta_frozen=OK", flush=True)

    # 2. All 5 continual-learning arms (W0..W4) miniature 2-day smoke test on O2
    c = dict(exp226.BASE_CFG, **exp226.ARM_SPEC["O2"], arm="O2", ctx=256)
    for arm in ("W0", "W1", "W2", "W3", "W4"):
        torch.manual_seed(3)
        m = exp226.OnlineToy(c, V).to(dev_t)
        eng = WakeSleepEngine(m, c, tk, dev=dev_t, amp_mode=amp_t, seed=3)
        if arm in ("W2", "W3"):
            eng.compute_core_fisher(dummy_replay, n_batches=2, batch_size=2, seq_len=64)
        elif arm == "W4":
            p0, _ = eng.generate_self_replay_pool(m, n_seqs=8, seq_len=64, batch_size=8)
            eng.compute_core_fisher(lambda bs, sl: p0[:bs], n_batches=2, batch_size=2, seq_len=64)

        for d_idx in (0, 1):
            for ep in world["days"][d_idx]["episodes"][:4]:
                assert math.isfinite(eng.wake_episode(ep, record=True))
            info = eng.sleep_night(arm, current_day=d_idx, replay_batch_fn=None if arm in ("W0", "W1", "W4") else dummy_replay, steps=3, lr=3e-4, replay_seq_len=64)
            if arm in ("W2", "W3", "W4"):
                assert info["ewc"] > 1e-4, f"{arm} EWC too small: {info['ewc']}"

        if arm != "W0":
            b0 = _detach_state(m.init_state(1))
            for a_st, b_st in zip(eng.fast_state, b0):
                if a_st is None:
                    assert b_st is None
                elif isinstance(a_st, tuple):
                    for ta, tb in zip(a_st, b_st):
                        assert torch.allclose(ta, tb, atol=1e-6)
                else:
                    assert torch.allclose(a_st, b_st, atol=1e-6)
        ev = eng.eval_mcq_list(world["days"][0]["exact_tests"][:4], use_fast_state=(arm == "W0"))
        assert 0.0 <= ev <= 1.0
        print(f"  Arm {arm} 2-day smoke ({dev_t}): info={info} eval={ev:.2f}", flush=True)

    print("EXP227_TEST_OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "pick":
        pick_226_winner()
    elif cmd == "run":
        kw = dict(kv.split("=") for kv in sys.argv[3:])
        run_five_days(
            sys.argv[2], kw["arm"],
            sleep_steps=int(kw.get("steps", 80)),
            lr=float(kw.get("lr", 3e-4)),
            kl_w=float(kw.get("kl", 1.0)),
            ewc_lam=float(kw.get("ewc", 0.25)),
            f2s_w=float(kw.get("f2s", 0.25)),
        )
    else:
        raise ValueError(f"unknown cmd {cmd}")
