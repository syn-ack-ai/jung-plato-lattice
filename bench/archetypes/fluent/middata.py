"""Addendum 224: mid-training data builder (Mac CPU; public text only, no Jung).
python fluent/middata.py             builds fluent/mid/{proc_train.bin,proc_held.jsonl,chat_train.bin,chat_held.bin,chat_held_prompts.jsonl,meta.json}
python fluent/middata.py --self-test quick verification of disjointness, overlap baseline, anti-shortcut checks, and tokenization."""
import argparse, gzip, json, random, re, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
COMP = HERE / "comp"
MID = HERE / "mid"
CHAT_GZ = HERE / "chatdata" / "oasst2_trees.jsonl.gz"
TOK_PATH = HERE.parent / "transfer" / "tok" / "bpe8k.json"


def load_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower().replace("\u2019", "'"))


def grams12(t):
    w = words(t)
    return {tuple(w[i:i + 12]) for i in range(len(w) - 11)}


def _a(phrase):
    """indefinite article helper avoiding ungrammatical 'a <vowel>'"""
    return f"an {phrase}" if phrase[0].lower() in "aeiou" else f"a {phrase}"


def load_test_guards():
    """collect all test 12-grams and all mcq2 words so procedural invented entities never overlap tests"""
    req = (
        (COMP / "mcq.jsonl", "passage", True),
        (COMP / "mcq2.jsonl", "passage", True),
        (COMP / "passages.jsonl", "passage", True),
        (COMP / "lastword.jsonl", "context", True),
        (HERE / "runs" / "gen" / "C25_books.jsonl", "prefix", False),
    )
    test_g = set()
    for f, key, required in req:
        if not f.exists():
            if required:
                raise FileNotFoundError(f"missing required test guard file: {f}")
            continue
        with open(f) as fh:
            for l in fh:
                r = json.loads(l)
                test_g |= grams12(r[key])
    mcq2_words = set()
    with open(COMP / "mcq2.jsonl") as fh:
        for l in fh:
            r = json.loads(l)
            mcq2_words.update(words(r["passage"]))
            mcq2_words.update(words(r["answer"]))
            for w in r["wrong"]:
                mcq2_words.update(words(w))
    return test_g, mcq2_words


def overlap_baseline(items):
    """exact same no-model word-overlap rule as comp.py"""
    stop = set("the a an of to in and or is was were be by for on at with as that this it its from his her their he she they".split())
    wfn = lambda t: set(re.findall(r"[a-z']+", t.lower())) - stop
    tot = 0.0
    for r in items:
        P = wfn(r["passage"])
        opts = [r["answer"]] + r["wrong"]
        sc = [len(wfn(o) & P) / max(1, len(wfn(o))) for o in opts]
        b = max(sc)
        tied = [i for i, x in enumerate(sc) if x == b]
        tot += (1.0 / len(tied)) if 0 in tied else 0.0
    return tot / max(1, len(items))


# ---------------------------------------------------------------- invented disjoint vocabulary

def make_names(seed, n, banned):
    """generate pronounceable invented proper nouns disjoint from banned words"""
    rng = random.Random(seed)
    onsets = ["b", "br", "c", "cr", "d", "dr", "f", "fl", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr", "v", "w", "z"]
    vowels = ["a", "e", "i", "o", "u", "ai", "ea", "oa"]
    codas = ["", "n", "r", "l", "m", "s", "t", "k", "d", "v", "th", "nd", "rk", "st"]
    out = []
    seen = set(banned)
    while len(out) < n:
        syls = rng.choice([2, 2, 3])
        w = "".join(rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas) for _ in range(syls))
        if 4 <= len(w) <= 10 and w not in seen:
            seen.add(w)
            out.append(w.capitalize())
    return out


TRADES = ["glassblower", "cartographer", "clockmaker", "blacksmith", "weaver", "herbalist", "bookbinder", "mason",
          "potter", "tanner", "dyer", "locksmith", "chandler", "cooper", "miller", "sailmaker",
          "tinsmith", "engraver", "sculptor", "watchmaker", "baker", "vintner", "fletcher", "cobbler"]
OBJECTS = ["brass compass", "silver lantern", "oak chest", "amber pendant", "iron key", "ivory flute",
           "copper astrolabe", "velvet satchel", "glass hourglass", "jade seal", "pewter goblet", "cedar box",
           "obsidian mirror", "bronze bell", "leather ledger", "quartz prism", "tin whistle", "coral brooch",
           "slate tablet", "gilded locket", "ebony staff", "linen map", "steel caliper", "clay amphora"]
PLACES = ["north tower", "harbor cellar", "west courtyard", "cedar greenhouse", "stone archive", "river boathouse",
          "upper gallery", "south granary", "bell loft", "east observatory", "old foundry", "market vault",
          "garden pavilion", "bridge gatehouse", "lower cistern", "clock belfry", "mill attic", "creek storehouse",
          "pine cottage", "cliff watchtower", "stable loft", "orchard shed", "abbey cloister", "salt warehouse"]
DRINKS = ["mint tea", "spiced cider", "ginger broth", "barley coffee", "plum cordial", "fennel infusion",
          "apple shrub", "thyme tonic", "lemon whey", "clover mead", "sage tisane", "pear nectar",
          "birch sap", "fig syrup", "oat milk", "rose water", "nutmeg punch", "elderberry juice"]
ANIMALS = ["red fox", "grey heron", "spotted hound", "barn owl", "white ferret", "black raven",
           "brown badger", "silver carp", "tame magpie", "marsh otter", "striped cat", "wild falcon",
           "mountain goat", "river turtle", "snow hare", "golden finch", "field mouse", "pied jay"]
GATES = ["Iron Gate", "East Arch", "Harbor Lock", "Sun Bridge", "Chalk Steps", "Red Causeway",
         "Wind Tunnel", "Salt Pier", "Ash Portal", "Oak Crossing", "Glass Viaduct", "Stone Ford",
         "High Pass", "Moss Tunnel", "Copper Wharf", "Cinder Gate", "Deep Quay", "Bell Causeway"]

# Combinatorial counterfactual property inversions: (item, everyday_property, inverted_property)
# Grouped by disjoint physical axes so every passage puts 4 items with inverted properties (including the everyday property as a distractor!)
CF_PAIRS = [
    ("ice", "freezing cold", "scorching hot"),
    ("fire", "scorching hot", "freezing cold"),
    ("iron", "sinking straight down", "floating like cork"),
    ("cork", "floating on top", "sinking like lead"),
    ("honey", "sweet as fruit", "bitter as ash"),
    ("soot", "pitch black", "blindingly white"),
    ("snow", "blindingly white", "pitch black"),
    ("granite", "hard as steel", "soft as wool"),
    ("cotton", "soft as wool", "hard as steel"),
    ("thunder", "deafeningly loud", "completely silent"),
    ("rain", "falling downward", "rising upward"),
    ("smoke", "rising upward", "falling downward"),
    ("saltwater", "salty and sharp", "purely sweet"),
    ("summer wind", "warm and dry", "icy and damp"),
    ("winter sleet", "icy and damp", "warm and dry"),
    ("straw", "light as air", "heavier than stone"),
    ("lead", "heavier than stone", "light as air"),
    ("glass", "clear and brittle", "opaque and rubbery"),
    ("clay", "opaque and dull", "crystal clear"),
    ("midnight", "dark as pitch", "bright as noon"),
    ("noon", "bright as noon", "dark as pitch"),
    ("chalk", "dry and dusty", "wet and oily"),
    ("tar", "thick and sticky", "thin as mist"),
    ("vinegar", "sour and sharp", "mild and creamy"),
]


class Pool:
    def __init__(self, split, people, towns, guilds):
        s = 0 if split == "train" else 1
        self.people = people[s::2]
        self.towns = towns[s::2]
        self.guilds = guilds[s::2]
        self.trades = TRADES[s::2]
        self.objects = OBJECTS[s::2]
        self.places = PLACES[s::2]
        self.drinks = DRINKS[s::2]
        self.animals = ANIMALS[s::2]
        self.gates = GATES[s::2]
        self.cf = CF_PAIRS[s::2]


def make_pools(banned):
    """construct train and held Pools once with guaranteed disjoint invented names"""
    people = make_names(22401, 1600, banned)
    towns = make_names(22402, 800, banned | {w.lower() for w in people})
    guilds = make_names(22403, 600, banned | {w.lower() for w in people} | {w.lower() for w in towns})
    return Pool("train", people, towns, guilds), Pool("held", people, towns, guilds)


# ---------------------------------------------------------------- five procedural families

def gen_binding(rng, pool):
    """4 entities with 3 attributes each in shuffled sentences; 3 distractors are the other entities' attributes."""
    ppl = rng.sample(pool.people, 4)
    trd = rng.sample(pool.trades, 4)
    drk = rng.sample(pool.drinks, 4)
    ani = rng.sample(pool.animals, 4)
    sents = []
    for p, t, d, a in zip(ppl, trd, drk, ani):
        sents.append(rng.choice([
            f"{p} works as {_a(t)} near the market square.",
            f"Everyone in the quarter knows {p} is the local {t}.",
            f"For ten years {p} has earned a living as {_a(t)}.",
        ]))
        sents.append(rng.choice([
            f"Each evening {p} drinks {d} beside {_a(a)}.",
            f"At dusk {p} pours a cup of {d} while feeding {_a(a)}.",
            f"{p} always keeps {_a(a)} indoors and prefers {d} after work.",
        ]))
    rng.shuffle(sents)
    passage = " ".join(sents)
    idx = rng.randrange(4)
    attr = rng.choice(["trade", "drink", "animal"])
    if attr == "trade":
        q = rng.choice([f"What is {ppl[idx]}'s trade?", f"What work does {ppl[idx]} do?"])
        ans = trd[idx]
        wrong = [trd[j] for j in range(4) if j != idx]
        note = f"The passage states that {ppl[idx]} works as {_a(ans)}."
    elif attr == "drink":
        q = rng.choice([f"What does {ppl[idx]} drink?", f"Which drink does {ppl[idx]} prefer?"])
        ans = drk[idx]
        wrong = [drk[j] for j in range(4) if j != idx]
        note = f"The passage mentions that {ppl[idx]} takes {ans}."
    else:
        q = rng.choice([f"Which animal does {ppl[idx]} keep?", f"What creature stays with {ppl[idx]}?"])
        ans = ani[idx]
        wrong = [ani[j] for j in range(4) if j != idx]
        note = f"The passage links {ppl[idx]} with {_a(ans)}."
    return {"kind": "binding", "passage": passage, "question": q, "answer": ans, "wrong": wrong, "note": note}


def gen_twohop(rng, pool):
    """4 chains A->B and B->C in randomized block or interleaved order; asks A->C."""
    ppl = rng.sample(pool.people, 4)
    towns = rng.sample(pool.towns, 4)
    gates = rng.sample(pool.gates, 4)
    objs = rng.sample(pool.objects, 4)
    mode = rng.choice(["town_gate", "guild_obj"])
    part1, part2 = [], []
    if mode == "town_gate":
        for p, tw, gt in zip(ppl, towns, gates):
            part1.append(rng.choice([
                f"{p} grew up in {tw}.",
                f"{p} travels down from {tw} every autumn.",
                f"Records show that {p} hails from {tw}.",
            ]))
            part2.append(rng.choice([
                f"Every resident of {tw} guards the {gt}.",
                f"By custom, anyone from {tw} is assigned to the {gt}.",
                f"Travelers from {tw} always report directly to the {gt}.",
            ]))
        rng.shuffle(part1); rng.shuffle(part2)
        order = rng.randrange(3)
        sents = part1 + part2 if order == 0 else (part2 + part1 if order == 1 else [x for ab in zip(part1, part2) for x in ab])
        passage = " ".join(sents)
        idx = rng.randrange(4)
        q = rng.choice([f"Which post is {ppl[idx]} assigned to?", f"Where does {ppl[idx]} report?"])
        ans = gates[idx]
        wrong = [gates[j] for j in range(4) if j != idx]
        note = f"{ppl[idx]} comes from {towns[idx]}, and people from {towns[idx]} go to the {ans}."
    else:
        glds = rng.sample(pool.guilds, 4)
        for p, g, o in zip(ppl, glds, objs):
            part1.append(rng.choice([
                f"{p} joined the Order of {g} last spring.",
                f"{p} serves as a senior clerk of the {g} hall.",
                f"The ledger lists {p} as a member of {g}.",
            ]))
            part2.append(rng.choice([
                f"Every member of {g} carries {_a(o)} as a badge.",
                f"The emblem of {g} is always {_a(o)}.",
                f"Anyone belonging to {g} must present {_a(o)} at the door.",
            ]))
        rng.shuffle(part1); rng.shuffle(part2)
        order = rng.randrange(3)
        sents = part1 + part2 if order == 0 else (part2 + part1 if order == 1 else [x for ab in zip(part1, part2) for x in ab])
        passage = " ".join(sents)
        idx = rng.randrange(4)
        q = rng.choice([f"What emblem must {ppl[idx]} carry?", f"Which object serves as {ppl[idx]}'s badge?"])
        ans = objs[idx]
        wrong = [objs[j] for j in range(4) if j != idx]
        note = f"{ppl[idx]} belongs to {glds[idx]}, whose emblem is {_a(ans)}."
    return {"kind": "twohop", "passage": passage, "question": q, "answer": ans, "wrong": wrong, "note": note}


def gen_tracking(rng, pool):
    """Track 3 objects across varied sequential transfers among 4 people and 4 places; queries any object (often moved BEFORE the final sentence so recency fails)."""
    ppl = rng.sample(pool.people, 4)
    objs = rng.sample(pool.objects, 3)
    plc = rng.sample(pool.places, 4)
    # Initial state: each object starts with a distinct person at a distinct place
    holder = {objs[0]: ppl[0], objs[1]: ppl[1], objs[2]: ppl[2]}
    loc = {objs[0]: plc[0], objs[1]: plc[1], objs[2]: plc[2]}
    sents = [
        rng.choice([
            f"At dawn {ppl[0]} took the {objs[0]} from the {plc[0]}, {ppl[1]} held the {objs[1]} at the {plc[1]}, and {ppl[2]} kept the {objs[2]} in the {plc[2]}.",
            f"Early in the morning the {objs[0]} was with {ppl[0]} in the {plc[0]}, the {objs[1]} was with {ppl[1]} at the {plc[1]}, and {ppl[2]} carried the {objs[2]} near the {plc[2]}.",
        ])
    ]
    # Perform 4 sequential events across objects with varied templates, tracking exact holder and location
    times = ["Before noon", "After midday", "In the afternoon", "At dusk"]
    history = {o: [(holder[o], loc[o])] for o in objs}
    # Ensure all 4 people and all 4 places appear in the passage
    unused_p = [ppl[3]]
    unused_l = [plc[3]]
    event_order = [0, 1, 2, rng.randrange(3)]
    rng.shuffle(event_order)
    for t_lbl, oi in zip(times, event_order):
        obj = objs[oi]
        cur_p = holder[obj]
        nxt_p = unused_p.pop(0) if unused_p else rng.choice([p for p in ppl if p != cur_p])
        nxt_l = unused_l.pop(0) if unused_l else rng.choice([l for l in plc if l != loc[obj]])
        sents.append(rng.choice([
            f"{t_lbl}, {cur_p} passed the {obj} to {nxt_p} inside the {nxt_l}.",
            f"{t_lbl}, {cur_p} handed the {obj} over to {nxt_p}, who carried it to the {nxt_l}.",
            f"{t_lbl}, {nxt_p} collected the {obj} from {cur_p} and stored it in the {nxt_l}.",
            f"{t_lbl}, {cur_p} left the {obj} with {nxt_p} at the {nxt_l}.",
        ]))
        holder[obj] = nxt_p
        loc[obj] = nxt_l
        history[obj].append((nxt_p, nxt_l))

    passage = " ".join(sents)
    # Prefer querying an object that was NOT moved in the very last sentence (so recency fails!)
    non_last = [o for o in objs if o != objs[event_order[-1]]]
    q_obj = rng.choice(non_last)
    if rng.random() < 0.5:
        q = rng.choice([f"Where is the {q_obj} at the end of the day?", f"In which place did the {q_obj} end up?"])
        ans = loc[q_obj]
        wrong = [l for l in plc if l != ans]
        trail = " -> ".join(l for _, l in history[q_obj])
        note = f"Following the {q_obj} ({trail}), it ends up in the {ans}."
    else:
        q = rng.choice([f"Who holds the {q_obj} at the end of the day?", f"Who ended up with the {q_obj}?"])
        ans = holder[q_obj]
        wrong = [p for p in ppl if p != ans]
        trail = " -> ".join(p for p, _ in history[q_obj])
        note = f"Following the {q_obj} ({trail}), {ans} holds it last."
    return {"kind": "tracking", "passage": passage, "question": q, "answer": ans, "wrong": wrong, "note": note}


def gen_counterfactual(rng, pool):
    """Combinatorial counterfactual world rules: asks for an item's inverted property; all 4 properties appear in the passage AND the item's everyday-true property is a distractor."""
    town = rng.choice(pool.towns)
    # Sample 4 items with 4 distinct inverted properties; ensure one other item's inverted property OR a paired contrast includes the queried item's everyday property when possible
    cand = list(pool.cf)
    rng.shuffle(cand)
    chosen = []
    used_inv = set()
    for item, real_p, inv_p in cand:
        if inv_p not in used_inv:
            chosen.append((item, real_p, inv_p))
            used_inv.add(inv_p)
        if len(chosen) == 4:
            break
    # Pick target item; if another item's inv_p isn't target's real_p, include target's real_p as an explicit contrast in the passage so all 4 options occur in the passage
    idx = rng.randrange(4)
    t_item, t_real, t_inv = chosen[idx]
    other_invs = [chosen[j][2] for j in range(4) if j != idx]
    if t_real not in other_invs and t_real != t_inv:
        # swap one distractor property with t_real by explicitly contrasting what outsiders expect!
        distractor_invs = [t_real, other_invs[0], other_invs[1]]
        contrast_sent = rng.choice([
            f"Outsiders expect {t_item} to be {t_real}, yet inside {town} {t_item} is {t_inv}.",
            f"Unlike elsewhere where {t_item} is {t_real}, in {town} {t_item} is always {t_inv}.",
            f"Although travelers assume {t_item} is {t_real}, anyone visiting {town} finds {t_item} is {t_inv}.",
        ])
    else:
        distractor_invs = other_invs
        contrast_sent = rng.choice([
            f"Inside {town}, {t_item} is always {t_inv}.",
            f"Chronicles of {town} record that {t_item} is {t_inv}.",
        ])
    sents = [contrast_sent]
    for j in range(4):
        if j == idx:
            continue
        it_j, _, inv_j = chosen[j]
        sents.append(rng.choice([
            f"In {town}, {it_j} is {inv_j}.",
            f"Locals in {town} observe that {it_j} is always {inv_j}.",
            f"Visitors quickly notice that {it_j} is {inv_j} there.",
        ]))
    rng.shuffle(sents)
    passage = f"The town of {town} follows strange physical laws. " + " ".join(sents)
    q = rng.choice([
        f"According to the passage, how is {t_item} described in {town}?",
        f"In {town}, what is {t_item} like?",
    ])
    ans = t_inv
    wrong = distractor_invs
    note = f"The passage states that in {town}, {t_item} is {t_inv} (not {t_real})."
    return {"kind": "counterfactual", "passage": passage, "question": q, "answer": ans, "wrong": wrong, "note": note}


def gen_ordering(rng, pool):
    """Scrambled transitive ordering of 4 people with an extra non-adjacent edge + all 4 ranks queried (eliminates 1-mention degree shortcut)."""
    ppl = rng.sample(pool.people, 4)  # true order: ppl[0] > ppl[1] > ppl[2] > ppl[3]
    dim = rng.choice([
        ("older", "younger", "oldest", "youngest"),
        ("taller", "shorter", "tallest", "shortest"),
        ("faster", "slower", "fastest", "slowest"),
        ("stronger", "weaker", "strongest", "weakest"),
    ])
    hi, lo, sup_hi, sup_lo = dim
    # Add one non-adjacent redundant edge so endpoints don't uniquely have degree 1
    extra = rng.choice([(0, 2), (1, 3)])
    pairs = [(0, 1), (1, 2), (2, 3), extra]
    rng.shuffle(pairs)
    sents = []
    for a, b in pairs:
        if rng.random() < 0.5:
            sents.append(f"{ppl[a]} is {hi} than {ppl[b]}.")
        else:
            sents.append(f"{ppl[b]} is {lo} than {ppl[a]}.")
    extra_town = rng.choice(pool.towns)
    passage = f"Four people from {extra_town} compared their records before the festival. " + " ".join(sents)
    # Query any of the 4 ranks (including middle ranks where chaining is mandatory)
    rank = rng.randrange(4)
    if rank == 0:
        q = f"Who is the {sup_hi} of the four?"
        lbl = f"the {sup_hi}"
    elif rank == 1:
        q = f"Who is the second {sup_hi} of the four?"
        lbl = f"second {sup_hi}"
    elif rank == 2:
        q = f"Who is the second {sup_lo} of the four?"
        lbl = f"second {sup_lo}"
    else:
        q = f"Who is the {sup_lo} of the four?"
        lbl = f"the {sup_lo}"
    ans = ppl[rank]
    wrong = [ppl[j] for j in range(4) if j != rank]
    note = f"From {sup_hi} to {sup_lo}, the order is {ppl[0]}, {ppl[1]}, {ppl[2]}, {ppl[3]}, so {ans} is {lbl}."
    return {"kind": "ordering", "passage": passage, "question": q, "answer": ans, "wrong": wrong, "note": note}


FAMILIES = [gen_binding, gen_twohop, gen_tracking, gen_counterfactual, gen_ordering]


def format_train_doc(rng, item):
    """render one procedural item in varied formats (plain Q&A, short Note reasoning, or Reader/Guide exchange without leaking User/Assistant chat tags into D1)"""
    style = rng.randrange(4)
    if style == 0:
        return f"Passage: {item['passage']}\n\nQuestion: {item['question']}\nAnswer: {item['answer']}\n\n"
    if style == 1:
        return f"Passage: {item['passage']}\n\nQuestion: {item['question']}\nNote: {item['note']}\nAnswer: {item['answer']}\n\n"
    if style == 2:
        return f"{item['passage']}\n\nReader: {item['question']}\n\nGuide: {item['note']} Therefore, {item['answer']}.\n\n"
    return f"{item['passage']}\n\nQuestion: {item['question']}\nAnswer: {item['answer']}\n\n"


def build_proc(tk, test_g, banned, target_tokens=30_000_000, n_held_per_family=80, out_dir=MID):
    t0 = time.time()
    rng_tr = random.Random(224)
    rng_he = random.Random(22499)
    pool_tr, pool_he = make_pools(banned)

    tr_names = set(pool_tr.people) | set(pool_tr.towns) | set(pool_tr.guilds)
    he_names = set(pool_he.people) | set(pool_he.towns) | set(pool_he.guilds)
    assert not (tr_names & he_names), "train and held invented names overlap!"
    assert not ({w.lower() for w in tr_names | he_names} & banned), "invented names overlap mcq2!"

    # 1. Held-out procedural items (disjoint entities, never in proc_train.bin)
    held = []
    held_g = set()
    for fam in FAMILIES:
        for slot in range(n_held_per_family):
            for _ in range(20):
                it = fam(rng_he, pool_he)
                assert len({o.lower().strip() for o in [it["answer"]] + it["wrong"]}) == 4, f"duplicate options in {fam.__name__}"
                g = grams12(it["passage"])
                if not (g & test_g):
                    it["slot"] = slot
                    held.append(it)
                    held_g |= g
                    break
            else:
                raise RuntimeError(f"failed to generate clean held item for {fam.__name__}")
    ov = {k: round(overlap_baseline([r for r in held if r["kind"] == k]), 3) for k in ("binding", "twohop", "tracking", "counterfactual", "ordering")}
    for k, v in ov.items():
        assert v <= 0.35, f"held procedural family {k} failed overlap check: {v}"
    with open(out_dir / "proc_held.jsonl", "w") as f:
        for r in held:
            f.write(json.dumps({k: r[k] for k in ("kind", "slot", "passage", "question", "answer", "wrong")}) + "\n")

    # 2. Procedural training stream (~target_tokens tokens)
    buf = []
    tok_count = 0
    n_docs = 0
    skipped = 0
    consec_skip = 0
    batch_texts = []
    batch_cap = min(2000, max(50, target_tokens // 150))
    block_g = test_g | held_g
    while tok_count < target_tokens:
        fam = FAMILIES[n_docs % len(FAMILIES)]
        it = fam(rng_tr, pool_tr)
        doc = format_train_doc(rng_tr, it)
        if grams12(doc) & block_g:
            skipped += 1
            consec_skip += 1
            if consec_skip > 500:
                raise RuntimeError("too many consecutive skipped procedural docs")
            continue
        consec_skip = 0
        batch_texts.append(doc)
        n_docs += 1
        if len(batch_texts) >= batch_cap:
            for e in tk.encode_batch(batch_texts):
                buf.append(np.array(e.ids, dtype=np.uint16))
                tok_count += len(e.ids)
            batch_texts.clear()
    if batch_texts:
        for e in tk.encode_batch(batch_texts):
            buf.append(np.array(e.ids, dtype=np.uint16))
            tok_count += len(e.ids)
    arr = np.concatenate(buf)
    arr.tofile(out_dir / "proc_train.bin")
    info = {"proc_train_tokens": int(len(arr)), "proc_train_docs": n_docs, "proc_skipped": skipped,
            "proc_held_n": len(held), "proc_held_overlap": ov, "s": round(time.time() - t0, 1)}
    print("PROC_DONE", info, flush=True)
    return info


# ---------------------------------------------------------------- OASST2 English chat builder

def _good_node(node):
    return (not node.get("deleted")) and node.get("lang") == "en" and bool(node.get("text", "").strip()) and node.get("review_result", True) is not False


def _top_paths(node, max_paths=3):
    """extract up to max_paths highest-ranked complete (prompter->assistant...) English paths; trims dangling User leaves and scores only assistant ranks"""
    def walk(cur):
        reps = [r for r in cur.get("replies", []) if _good_node(r)]
        if not reps:
            return [[cur]]
        reps.sort(key=lambda r: (r.get("rank") is None, r.get("rank", 99)))
        keep = [r for r in reps if r.get("rank") in (0, 1)][:2] or reps[:1]
        out = []
        for r in keep:
            for sub in walk(r):
                out.append([cur] + sub)
        return out

    clean = []
    seen_ids = set()
    for p in walk(node):
        while p and p[-1].get("role") != "assistant":
            p = p[:-1]
        if len(p) < 2 or len(p) % 2 != 0:
            continue
        if any(p[i].get("role") != ("prompter" if i % 2 == 0 else "assistant") for i in range(len(p))):
            continue
        key = tuple(x.get("message_id", x["text"][:40]) for x in p)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        asst_ranks = [x.get("rank") if x.get("rank") is not None else 1 for x in p[1::2]]
        mean_rk = sum(asst_ranks) / len(asst_ranks)
        clean.append(((mean_rk, -len(p)), p))
    clean.sort(key=lambda x: x[0])
    return [p for _, p in clean[:max_paths]]


def _fmt_conv(path):
    return "\n\n".join(f"{'User' if x.get('role') == 'prompter' else 'Assistant'}: {x['text'].strip()}" for x in path) + "\n\n"


def build_chat(tk, test_g, n_held_trees=250, n_eval_prompts=100, out_dir=MID):
    t0 = time.time()
    trees = []
    with gzip.open(CHAT_GZ, "rt") as f:
        for l in f:
            d = json.loads(l)
            if d.get("tree_state") != "ready_for_export":
                continue
            pr = d.get("prompt", {})
            if _good_node(pr):
                paths = _top_paths(pr, max_paths=3)
                if paths:
                    root_norm = " ".join(words(pr["text"]))
                    trees.append((d.get("message_tree_id", str(len(trees))), root_norm, paths))
    rng = random.Random(224)
    rng.shuffle(trees)

    # Select held-out trees (clean against test_g, unique normalized root prompts)
    held_trees = []
    held_roots = set()
    held_texts = []
    train_trees = []
    for tid, root_norm, paths in trees:
        t0_txt = _fmt_conv(paths[0])
        if len(held_trees) < n_held_trees and root_norm not in held_roots and not (grams12(t0_txt) & test_g):
            held_trees.append((tid, paths))
            held_roots.add(root_norm)
            held_texts.append(t0_txt)
        else:
            train_trees.append((tid, root_norm, paths))
    assert len(held_trees) == n_held_trees, f"only found {len(held_trees)} clean held trees"

    held_ids = np.concatenate([np.array(e.ids, dtype=np.uint16) for e in tk.encode_batch(held_texts)])
    held_ids.tofile(out_dir / "chat_held.bin")

    # 100 clean single-turn user prompts from held trees (15..600 chars)
    eval_prompts = []
    for tid, paths in held_trees:
        u = paths[0][0]["text"].strip()
        if 15 <= len(u) <= 600 and "\n\nUser:" not in u and "\n\nAssistant:" not in u:
            eval_prompts.append({"pid": len(eval_prompts), "tree_id": tid, "user": u, "prompt": f"User: {u}\n\nAssistant:"})
            if len(eval_prompts) >= n_eval_prompts:
                break
    assert len(eval_prompts) == n_eval_prompts, f"only found {len(eval_prompts)} held chat prompts"
    with open(out_dir / "chat_held_prompts.jsonl", "w") as f:
        for r in eval_prompts:
            f.write(json.dumps(r) + "\n")

    # Guard against any 12-gram overlap with tests or held chat, AND duplicate short root prompts
    held_g = set()
    for t in held_texts:
        held_g |= grams12(t)
    block_g = test_g | held_g

    train_texts = []
    skipped = 0
    for _, root_norm, paths in train_trees:
        if root_norm in held_roots:
            skipped += len(paths)
            continue
        for p in paths:
            txt = _fmt_conv(p)
            if grams12(txt) & block_g:
                skipped += 1
                continue
            train_texts.append(txt)
    rng.shuffle(train_texts)
    train_ids = np.concatenate([np.array(e.ids, dtype=np.uint16) for e in tk.encode_batch(train_texts)])
    train_ids.tofile(out_dir / "chat_train.bin")

    info = {"chat_trees_total": len(trees), "chat_train_trees": len(train_trees), "chat_train_paths": len(train_texts),
            "chat_train_tokens": int(len(train_ids)), "chat_train_skipped": skipped,
            "chat_held_trees": len(held_trees), "chat_held_tokens": int(len(held_ids)),
            "chat_eval_prompts": len(eval_prompts), "s": round(time.time() - t0, 1)}
    print("CHAT_DONE", info, flush=True)
    return info


def self_test():
    import tempfile
    tk = load_tok()
    test_g, banned = load_test_guards()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p_info = build_proc(tk, test_g, banned, target_tokens=50_000, n_held_per_family=20, out_dir=td)
        assert p_info["proc_train_tokens"] >= 50_000
        assert p_info["proc_held_n"] == 100
        # Verify no ungrammatical "a <vowel>" in held items and verify all 4 ordering ranks appear
        with open(td / "proc_held.jsonl") as f:
            held = [json.loads(l) for l in f]
        for r in held:
            assert not re.search(r"\ba [aeiou]\w+", r["passage"].lower()), f"bad article: {r['passage']}"
        ord_qs = [r["question"] for r in held if r["kind"] == "ordering"]
        assert any("second" in q for q in ord_qs), "ordering should query middle ranks too"
        c_info = build_chat(tk, test_g, n_held_trees=250, n_eval_prompts=100, out_dir=td)
        assert c_info["chat_train_tokens"] > 1_000_000 and c_info["chat_held_tokens"] > 50_000
        # Verify no dangling User: -> User: transitions in chat_held
        h_txt = tk.decode(np.fromfile(td / "chat_held.bin", dtype=np.uint16).tolist())
        assert "\n\nUser:" in h_txt and "\n\nAssistant:" in h_txt
    print("MIDDATA_SELF_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-tokens", type=int, default=30_000_000)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    MID.mkdir(parents=True, exist_ok=True)
    tk = load_tok()
    test_g, banned = load_test_guards()
    print(f"guards: {len(test_g)} test 12-grams, {len(banned)} mcq2 words", flush=True)
    p_info = build_proc(tk, test_g, banned, target_tokens=a.proc_tokens, n_held_per_family=80, out_dir=MID)
    c_info = build_chat(tk, test_g, n_held_trees=250, n_eval_prompts=100, out_dir=MID)
    with open(MID / "meta.json", "w") as f:
        json.dump({"seed": 224, **p_info, **c_info}, f, indent=1)
    print("MIDDATA_DONE", flush=True)


if __name__ == "__main__":
    main()
