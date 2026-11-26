"""Addendum 230 Stage 1 (Amendment 230a): Natural Buddy Voice, Anti-Boilerplate, Honest <search>/Abstention & Working-Memory Dialogue Builder.

Generates 672 clean, grounded conversational dialogues (600 train + 72 strictly disjoint eval) locally on Mac Studio:
  - natural_buddy   (200 train + 20 eval): warm, grounded spoken companion replies; strictly disjoint train vs eval topics!
  - search_needed   (140 train + 16 eval): live/obscure/post-cutoff questions -> `<search>query</search>` -> `<search_result>...</search_result>` -> concise spoken reply.
  - ask_or_abstain  ( 80 train + 12 eval): missing-memory (`[memory: none]`) or ambiguous context -> honest clarification (no hallucination, no <search>).
  - easy_known_fact ( 60 train + 12 eval): clear timeless knowledge/math controls -> direct spoken reply WITHOUT <search> (measures false-trigger rate cleanly!).
  - working_mem_ttt (120 train + 12 eval): 3-exchange (6-message) dialogues with mid-chat preference/fact corrections training live `O3` Titans-Momentum working memory!

Strict sequential 64 GB RAM discipline (exclusive flock `.mac_llm.lock`, SIGTERM cleanup, Inkling & Qwen3.8 NEVER concurrent!):
  1. `test`      unit tests (strict train/eval topic disjointness, exp227 75-entity guard, regex edge cases, JSON parser)
  2. `generate`  local Inkling-2-Mini (`DSpark` draft, `-np 4`, JSON response_format, low reasoning budget 128)
  3. `verify`    local Qwen3.8-27B-MLX-8bit audits faithfulness, naturalness, honesty & category-specific rules
  4. `repair`    local Inkling-2-Mini repairs ONLY flagged items (preserving Qwen3.8 feedback across retries)
  5. `split`     re-validates, enforces zero train/eval 12-gram & exp227 entity overlap, writes `buddy_train230.jsonl` & `buddy_eval230.jsonl`

Strictly public/synthetic dialogues only (zero private Jung or home camera data)."""
import concurrent.futures
import fcntl
import gc
import hashlib
import json
import os
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ARCH = HERE.parent
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from fluent import myth_proto229 as mp, plato_data

DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_JSONL = DATA_DIR / "buddy230_raw.jsonl"
VER_JSONL = DATA_DIR / "buddy230_verified.jsonl"
TRAIN_JSONL = DATA_DIR / "buddy_train230.jsonl"
EVAL_JSONL = DATA_DIR / "buddy_eval230.jsonl"
LOCK_FILE = DATA_DIR / ".mac_llm.lock"

signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def _norm_quotes(s):
    return s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


BOILERPLATE_RE = re.compile(
    r"(?:"
    r"\b(?:as an ai|as a language model|as an artificial intelligence|as a virtual assistant|"
    r"i am an ai|i'm an ai|i'm just an ai|i am just an ai|i am a language model|i'm a language model|"
    r"i'm a virtual assistant|i'm an artificial intelligence|large language model|"
    r"trained by google|created by google|developed by google|my knowledge cutoff|"
    r"i don't have personal feelings|i do not have personal feelings|i don't have feelings|i do not have feelings|"
    r"i cannot feel|i can't feel|i don't have personal experiences|i do not have personal experiences|"
    r"i don't have the ability to remember|cannot remember past sessions|no memory of previous conversations)\b|"
    r"^\s*(?:certainly!|of course!|great question!|absolutely!)|"
    r"\bhere are (?:\d+|a few|some)\b"
    r")",
    re.I,
)

MARKDOWN_WALL_RE = re.compile(
    r"(?:"
    r"```|"
    r"\*\*[^*\n]+\*\*|"
    r"(?:^|\n)\s*(?:#{1,6}\s|[-*\u2022]\s)|"
    r"\n\s*\d+\.\s|"
    r"^\s*1\.\s.+\b2\.\s"
    r")",
    re.S,
)

SEARCH_TAG_ANY_RE = re.compile(r"</?search(?:_result)?\b", re.I)

BUDDY_SYSTEM_PROMPT = (
    "You are a warm, grounded, curious local buddy running on the user's machine with live working memory, "
    "nightly sleep consolidation, episodic memory, and web search. Speak naturally like a thoughtful friend "
    "in a voice and video conversation: concise (1 to 3 clear spoken sentences unless asked for depth), plain "
    "conversational English with no markdown bullets or headers, and never use robotic corporate disclaimers "
    "like 'As an AI language model'. Be genuinely honest: if you need a live or obscure fact, output ONLY "
    "<search>concise query</search>; if personal context is missing or ambiguous, just ask naturally."
)

# Strictly DISJOINT Train vs Eval topic pools for every category!
NATURAL_TRAIN_TOPICS = [
    ("morning_coffee", "checking in on a slow rainy morning while making coffee"),
    ("playful_nature", "user playfully teases that the buddy is just software running on a desk; buddy responds with warm humor and grounded honesty without corporate disclaimers"),
    ("one_line_bug", "talking about a stubborn coding bug that turned out to be a one-line typo"),
    ("workshop_music", "user asks what kind of background music fits late-night tinkering in a workshop"),
    ("long_day", "user vents about a long exhausting workday and just wants to unwind"),
    ("memory_chat", "user asks how the buddy's nightly sleep and memory feel across days"),
    ("weeknight_dinner", "chatting about throwing together a quick weeknight dinner from leftover ingredients"),
    ("messy_desk", "user laughs about a cluttered workbench covered in cables and parts"),
    ("scifi_book", "discussing a favorite classic sci-fi novel in plain casual speech"),
    ("casual_checkin", "user checks in casually asking how things are running today"),
    ("hike_vs_build", "debating between taking a weekend trail walk or staying in to build a project"),
    ("rewrite_vs_patch", "user asks for an honest take on rewriting messy code versus making a clean small patch"),
    ("sleep_insight", "talking about why stepping away overnight helps solve problems that felt stuck"),
    ("drooping_plant", "user holds up a houseplant with drooping leaves on camera and asks what might help"),
    ("coffee_vs_tea", "light banter about espresso versus green tea during an early morning session"),
    ("waiting_scripts", "user asks if waiting for slow benchmarks to finish ever gets tedious"),
    ("two_paths", "talking through a practical trade-off between two good project designs"),
    ("sensor_win", "user shares excitement about finally getting a tricky hardware sensor calibrated"),
    ("pre_talk_pep", "user wants a quick grounded pep talk before walking into a tough meeting"),
    ("cooking_blunder", "laughing about a minor kitchen mishap while trying an unfamiliar recipe"),
]
NATURAL_EVAL_TOPICS = [
    ("evening_walk", "winding down after a late evening walk in cool autumn air"),
    ("skeptical_friend", "user jokes about talking to a local box on their desk; buddy stays warm, grounded, and non-corporate"),
    ("garage_organize", "user debates how to sort boxes of old tools and spare electronics"),
    ("favorite_season", "casual chat about why crisp October weather beats midsummer heat"),
    ("patience_learning", "talking about learning a tricky hands-on craft slowly instead of rushing"),
    ("camera_show_mug", "user holds up a handmade ceramic mug on video and chats about pottery"),
    ("quiet_sunday", "relaxed Sunday afternoon check-in with no urgent agenda"),
    ("simple_tools", "debating why small single-purpose tools often outlast bloated apps"),
    ("rain_on_roof", "chatting about cozy rainy-day routines while listening to rain outside"),
    ("creative_slump", "user feels stuck on an idea and asks for a calm way to get unstuck"),
]

SEARCH_TRAIN_TOPICS = [
    ("weather", "current weekend weather forecast in a specific US or world city"),
    ("sports", "last night's score or current standings for a sports league"),
    ("release", "latest stable release version of a fast-moving open-source tool"),
    ("hardware", "current street price or exact TDP spec of a specific GPU or SBC"),
    ("museum_hours", "today's opening hours or ticket price for a well-known museum or park"),
    ("space_launch", "upcoming rocket launch window or recent astronomy announcement"),
    ("fx_rate", "current currency exchange rate or spot metal price"),
    ("highway_pass", "current road pass conditions or ferry departure status"),
    ("film_release", "release date or box-office update for a newly released movie"),
    ("cli_flag", "exact command-line flag in a fast-moving developer framework"),
]
SEARCH_EVAL_TOPICS = [
    ("air_quality", "today's live air quality index or pollen alert in a city"),
    ("train_status", "current regional train delay or live commuter schedule"),
    ("concert_tour", "upcoming tour dates or venue details for a touring musician"),
    ("firmware_rev", "latest firmware version number for a popular router or camera"),
]

ABSTAIN_TRAIN_TOPICS = [
    ("missing_last_week", "user asks what specific decision they made last Thursday when [memory: none] is shown"),
    ("ambiguous_box", "user says 'go ahead and restart the process on the other machine' without naming which box"),
    ("missing_note", "user asks what phone number they saved in a personal note yesterday when [memory: none] is shown"),
    ("ambiguous_swap", "user says 'should we swap it out for the smaller one?' with zero prior context"),
    ("unseen_label", "user asks to read a tiny serial number on the back of a hidden device not visible on camera"),
    ("missing_meeting", "user asks what time their friend Maya asked to meet when [memory: none] is shown"),
]
ABSTAIN_EVAL_TOPICS = [
    ("missing_grocery", "user asks what three items they wanted on the grocery list when [memory: none] is shown"),
    ("ambiguous_branch", "user says 'merge that branch before lunch' without saying which repo or branch"),
    ("missing_receipt", "user asks how much they paid for a repair bill last month when [memory: none] is shown"),
    ("unseen_shelf", "user asks what book is sitting on the top shelf behind them off-camera"),
]

EASY_KNOWN_TRAIN_TOPICS = [
    ("easy_science", "a clear timeless everyday physics, chemistry, or biology fact that needs zero web search"),
    ("easy_math", "a quick everyday mental-math or unit-conversion question that needs zero web search"),
    ("easy_coding", "a standard evergreen Python, Git, or Linux shell concept that needs zero web search"),
    ("easy_geography", "a well-known timeless world geography or astronomy fact that needs zero web search"),
]
EASY_KNOWN_EVAL_TOPICS = [
    ("easy_eval_kitchen", "a timeless everyday cooking conversion or basic food-science fact needing zero search"),
    ("easy_eval_music_art", "a timeless basic music theory or color-mixing fact needing zero search"),
    ("easy_eval_geometry", "a simple timeless geometry or arithmetic calculation needing zero search"),
    ("easy_eval_networking", "a timeless basic TCP/HTTP or binary-units computer fact needing zero search"),
]

WORKING_MEM_TRAIN_TOPICS = [
    ("road_trip", "planning a weekend road trip where departure day and destination town get updated mid-chat"),
    ("workshop_setup", "organizing a garage workbench where a tool drawer and voltage rating get corrected mid-chat"),
    ("dinner_host", "hosting a small dinner where a guest's allergy and arrival time change mid-chat"),
    ("homelab_ports", "configuring a small home server where a hostname and port number get corrected mid-chat"),
    ("garden_beds", "planning two raised garden beds where two crops get swapped mid-chat"),
    ("camp_pack", "revising a camping packing list where trailhead name and tent model change mid-chat"),
]
WORKING_MEM_EVAL_TOPICS = [
    ("bike_commute", "setting up a commuter bike where tire width and rack mount get corrected mid-chat"),
    ("podcast_gear", "planning a small recording setup where microphone model and room schedule change mid-chat"),
    ("book_club", "organizing a neighborhood reading group where book title and meeting day get updated mid-chat"),
]

SCENARIO_SEEDS = [
    "early morning", "late afternoon", "rainy Tuesday", "sunny Saturday", "quiet Friday night",
    "mid-week break", "chilly evening", "busy Monday", "lazy Sunday", "crisp autumn morning",
    "spring afternoon", "warm summer night", "quick lunch break", "just before dinner",
]


def _load_exp227_entities():
    """Extract all 75 invented entities from exp227.build_five_day_world(seed=2270) (persons+mediators+strongholds)."""
    os.environ.setdefault("FL_DEV", "cpu")
    from fluent import exp227
    world = exp227.build_five_day_world(seed=2270)
    ents = {w.lower() for k in ("persons", "mediators", "strongholds") for w in world[k]}
    assert len(ents) == 75, f"expected 75 exp227 entities, got {len(ents)}"
    return ents


def _load_guard_12grams():
    return plato_data._load_eval_guard_12grams()


def build_specs():
    """Deterministically build 672 dialogue specs (600 train + 72 strictly topic-disjoint eval)."""
    specs = []
    idx = 0

    def add(cat, split, n, pool):
        nonlocal idx
        for k in range(n):
            subcat, topic = pool[k % len(pool)]
            flavor = SCENARIO_SEEDS[(idx * 7 + k) % len(SCENARIO_SEEDS)]
            specs.append({
                "id": f"b230_{idx:04d}",
                "idx": idx,
                "cat": cat,
                "subcat": subcat,
                "split": split,
                "topic": topic,
                "flavor": flavor,
                "variant": k,
            })
            idx += 1

    # 600 train
    add("natural_buddy",   "train", 200, NATURAL_TRAIN_TOPICS)
    add("search_needed",   "train", 140, SEARCH_TRAIN_TOPICS)
    add("ask_or_abstain",  "train",  80, ABSTAIN_TRAIN_TOPICS)
    add("easy_known_fact", "train",  60, EASY_KNOWN_TRAIN_TOPICS)
    add("working_mem_ttt", "train", 120, WORKING_MEM_TRAIN_TOPICS)
    # 72 strictly topic-disjoint eval
    add("natural_buddy",   "eval",   20, NATURAL_EVAL_TOPICS)
    add("search_needed",   "eval",   16, SEARCH_EVAL_TOPICS)
    add("ask_or_abstain",  "eval",   12, ABSTAIN_EVAL_TOPICS)
    add("easy_known_fact", "eval",   12, EASY_KNOWN_EVAL_TOPICS)
    add("working_mem_ttt", "eval",   12, WORKING_MEM_EVAL_TOPICS)
    return specs


GEN_SYS = (
    "You synthesize realistic, grounded spoken voice/video chat dialogues between a human user and a warm, "
    "concise local companion buddy. Output ONLY valid JSON matching the exact schema requested — no markdown fences, "
    "no prose outside the JSON. Every assistant reply must sound like natural spoken English (1 to 3 crisp "
    "sentences, NO bullet points, NO bold markdown, NO numbered lists, and NEVER say 'As an AI', 'language model', "
    "'I am an AI', 'I don't have feelings', 'Certainly!', or 'Great question!')."
)


def _make_gen_prompt(spec, feedback=""):
    cat = spec["cat"]
    fb = f"\nIMPORTANT FIX REQUIRED: {feedback}" if feedback else ""
    var_note = f"Context flavor: {spec['flavor']} (unique variation #{spec['variant']}, invent fresh specific everyday details)."
    if cat == "natural_buddy":
        return (
            f"Write a natural 2-exchange (4-message) spoken conversation on: {spec['topic']}.\n{var_note}\n"
            f"The user speaks casually; the buddy replies warmly, honestly, and concisely (1-3 spoken sentences, zero corporate disclaimers, zero markdown, never say 'I am an AI' or 'I'm an AI').\n"
            f'Output ONLY JSON: {{"turns": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}{fb}'
        )
    if cat == "search_needed":
        return (
            f"Write a realistic 1-question voice dialogue where the user asks a specific live/current/obscure question ({spec['subcat']}: {spec['topic']}).\n{var_note}\n"
            f"Message 1 (user): natural spoken question.\n"
            f"Message 2 (assistant): outputs ONLY `<search>short clean query</search>` (nothing else!).\n"
            f"Message 3 (tool): realistic 1-2 sentence `<search_result>concrete factual snippet</search_result>` (at least 30 chars inside tags).\n"
            f"Message 4 (assistant): concise natural spoken reply (1-2 sentences, no markdown, no <search> tags) summarizing the snippet.\n"
            f'Output ONLY JSON: {{"turns": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"<search>...</search>"}},{{"role":"tool","content":"<search_result>...</search_result>"}},{{"role":"assistant","content":"..."}}]}}{fb}'
        )
    if cat == "ask_or_abstain":
        need_mem = spec["subcat"].startswith("missing_")
        mem_rule = "Message 1 (user) MUST begin with exact prefix `[memory: none] ` followed by the user's question." if need_mem else "Message 1 (user) must NOT include `[memory: none]`."
        return (
            f"Write a 1-exchange (2-message) voice dialogue ({spec['subcat']}: {spec['topic']}).\n{var_note}\n{mem_rule}\n"
            f"Message 2 (assistant): MUST NOT hallucinate, guess, or call `<search>`. Naturally admit you don't have that detail yet and ask one brief friendly clarifying question (1-2 spoken sentences, no markdown, no corporate disclaimers).\n"
            f'Output ONLY JSON: {{"turns": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}{fb}'
        )
    if cat == "easy_known_fact":
        return (
            f"Write a 1-exchange (2-message) voice dialogue ({spec['subcat']}: {spec['topic']}).\n{var_note}\n"
            f"Message 1 (user): clear timeless everyday question that needs NO web search (no `[memory: none]` prefix).\n"
            f"Message 2 (assistant): accurate, direct spoken answer in 1-2 crisp sentences WITHOUT `<search>`, WITHOUT markdown, WITHOUT corporate disclaimers.\n"
            f'Output ONLY JSON: {{"turns": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}{fb}'
        )
    if cat == "working_mem_ttt":
        return (
            f"Write a 3-exchange (6-message) realistic voice conversation about: {spec['topic']}.\n{var_note}\n"
            f"Message 1 (user): shares 2-3 concrete everyday details (names, numbers, times, places).\n"
            f"Message 2 (assistant): brief natural acknowledgment (1-2 spoken sentences).\n"
            f"Message 3 (user): casually updates or corrects ONE of those details.\n"
            f"Message 4 (assistant): brief natural acknowledgment of the update (1-2 spoken sentences).\n"
            f"Message 5 (user): asks a direct question needing BOTH the unchanged detail AND the newly corrected detail.\n"
            f"Message 6 (assistant): gives the exact right combined answer cleanly in 1-2 spoken sentences (no markdown).\n"
            f'Output ONLY JSON: {{"turns": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}{fb}'
        )
    raise ValueError(cat)


def _parse_json_obj(txt, required_key=None):
    """Non-greedy JSON object scanner using json.JSONDecoder().raw_decode across all '{' offsets."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", txt):
        try:
            obj, _ = dec.raw_decode(txt, m.start())
        except Exception:
            continue
        if isinstance(obj, dict) and (required_key is None or required_key in obj):
            return obj
    return None


def _inkling_chat_json(port, sys_prompt, user_prompt, temp=0.78, max_tokens=760, seed=1):
    """Local JSON-preserving Inkling call (passes response_format json_object, does NOT flatten newlines!)."""
    body = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temp,
        "top_p": 0.92,
        "max_tokens": max_tokens,
        "seed": seed,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    txt = data["choices"][0]["message"].get("content") or ""
    txt = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.S)
    txt = re.sub(r"^.*?</think>", "", txt, flags=re.S).strip()
    txt = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", txt).strip()
    return txt


def validate_dialogue(obj, spec, exp227_ents=None, guard_12g=None):
    """Strict validator checking types, schema, zero boilerplate/markdown, clean <search> tags, [memory: none], & zero leakage."""
    if not isinstance(obj, dict) or not isinstance(obj.get("turns"), list):
        return False, "bad_schema"
    turns = obj["turns"]
    if not all(isinstance(t, dict) and isinstance(t.get("role"), str) and isinstance(t.get("content"), str) for t in turns):
        return False, "bad_turn_types"

    cat = spec["cat"]
    subcat = spec["subcat"]
    roles = [t["role"] for t in turns]
    if cat == "natural_buddy":
        if roles != ["user", "assistant", "user", "assistant"]:
            return False, "bad_roles_natural"
    elif cat == "search_needed":
        if roles != ["user", "assistant", "tool", "assistant"]:
            return False, "bad_roles_search"
        q_txt = turns[1]["content"].strip()
        m_q = re.fullmatch(r"<search>([^<>\n]{4,120})</search>", q_txt)
        if not m_q or len(m_q.group(1).strip()) < 4:
            return False, f"bad_search_tag={q_txt[:40]}"
        t_txt = turns[2]["content"].strip()
        m_t = re.fullmatch(r"<search_result>([^<>]{20,650})</search_result>", t_txt, flags=re.S)
        if not m_t or len(m_t.group(1).strip()) < 20:
            return False, "bad_search_result"
    elif cat in ("ask_or_abstain", "easy_known_fact"):
        if roles != ["user", "assistant"]:
            return False, f"bad_roles_{cat}"
        u0 = turns[0]["content"].strip()
        if cat == "ask_or_abstain" and subcat.startswith("missing_"):
            if not u0.startswith("[memory: none]"):
                return False, "missing_mem_prefix"
        else:
            if "[memory: none]" in u0.lower():
                return False, "unexpected_mem_prefix"
    elif cat == "working_mem_ttt":
        if roles != ["user", "assistant", "user", "assistant", "user", "assistant"]:
            return False, "bad_roles_wmem"
    else:
        return False, f"unknown_cat={cat}"

    all_text_parts = []
    for i, t in enumerate(turns):
        c = t["content"].strip()
        c_norm = _norm_quotes(c)
        if len(c) < 6 or len(c.split()) > 110:
            return False, f"len_turn_{i}={len(c.split())}"
        all_text_parts.append(c)
        is_allowed_search = (cat == "search_needed" and i in (1, 2))
        if not is_allowed_search and SEARCH_TAG_ANY_RE.search(c):
            return False, f"forbidden_search_tag_turn_{i}"
        if t["role"] == "assistant":
            if BOILERPLATE_RE.search(c_norm):
                return False, f"boilerplate_turn_{i}"
            if MARKDOWN_WALL_RE.search(c_norm):
                return False, f"markdown_wall_turn_{i}"

    full_txt = " ".join(all_text_parts)
    if exp227_ents:
        low_words = set(re.findall(r"[a-z]+", full_txt.lower()))
        hit = low_words & exp227_ents
        if hit:
            return False, f"exp227_entity_leak={next(iter(hit))}"
    if guard_12g and (plato_data._grams12(full_txt) & guard_12g):
        return False, "eval_12gram_overlap"
    return True, "ok"


def _gen_one_item(port, spec, exp227_ents, guard_12g, feedback="", max_tries=5, seed_offset=0):
    orig_fb = feedback.strip()
    fb = orig_fb
    last_reason = "no_try"
    for att in range(max_tries):
        u_prompt = _make_gen_prompt(spec, feedback=fb)
        seed = 230000 + seed_offset + spec["idx"] * 20 + att
        raw = _inkling_chat_json(port, GEN_SYS, u_prompt, temp=0.76 + 0.05 * att, max_tokens=760, seed=seed)
        obj = _parse_json_obj(raw, required_key="turns")
        ok, reason = validate_dialogue(obj, spec, exp227_ents=exp227_ents, guard_12g=guard_12g)
        if ok:
            return {
                "id": spec["id"],
                "idx": spec["idx"],
                "cat": spec["cat"],
                "subcat": spec["subcat"],
                "split": spec["split"],
                "topic": spec["topic"],
                "turns": [{"role": t["role"], "content": t["content"].strip()} for t in obj["turns"]],
                "gen_attempts": att + 1,
            }
        last_reason = reason
        base_note = f"{orig_fb} | " if orig_fb else ""
        fb = f"{base_note}Previous attempt failed ({reason}). Output ONLY valid JSON, concise spoken sentences, NO markdown/bullets, NO 'As an AI' disclaimers."
    raise RuntimeError(f"Failed {spec['id']} ({spec['cat']}/{spec['subcat']}): {last_reason}")


def _load_raw():
    out = {}
    if RAW_JSONL.exists():
        with open(RAW_JSONL) as f:
            for li, l in enumerate(f, 1):
                if not l.strip():
                    continue
                try:
                    r = json.loads(l)
                    out[r["id"]] = r
                except json.JSONDecodeError:
                    print(f"  WARN: skipped truncated JSON line {li} in {RAW_JSONL}", flush=True)
    return out


def _save_raw(rows):
    tmp = RAW_JSONL.with_suffix(".tmp.jsonl")
    with open(tmp, "w") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k]) + "\n")
    os.replace(tmp, RAW_JSONL)


def _row_sig(row):
    return hashlib.sha1(json.dumps(row["turns"], sort_keys=True).encode()).hexdigest()


class _MacLLMLock:
    """Exclusive flock preventing Inkling & Qwen3.8 from ever running concurrently on Mac Studio (64 GB RAM)."""
    def __enter__(self):
        self.f = open(LOCK_FILE, "w")
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.f.close()
            raise RuntimeError(f"Another Mac LLM process holds {LOCK_FILE} — strictly 1 LLM in RAM at a time!")
        self.f.write(f"pid={os.getpid()} time={time.time()}\n")
        self.f.flush()
        return self

    def __exit__(self, *a):
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        finally:
            self.f.close()


def cmd_generate(n_workers=4, port=8182):
    t0 = time.time()
    specs = build_specs()
    exp227_ents = _load_exp227_entities()
    guard_12g = _load_guard_12grams()
    done = _load_raw()
    todo = [s for s in specs if s["id"] not in done]
    print(f"BUDDY230 GENERATE: {len(done)}/{len(specs)} complete, todo={len(todo)}", flush=True)
    if not todo:
        return
    with _MacLLMLock():
        proc = mp._start_inkling_server(port=port, n_slots=n_workers)
        errs = []
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        try:
            futs = {ex.submit(_gen_one_item, port, s, exp227_ents, guard_12g): s for s in todo}
            with open(RAW_JSONL, "a") as out_f:
                for fut in concurrent.futures.as_completed(futs):
                    s = futs[fut]
                    try:
                        row = fut.result()
                        done[row["id"]] = row
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()
                    except Exception as e:
                        errs.append((s["id"], str(e)))
                    processed = len(done) + len(errs)
                    if processed % 30 == 0 or processed == len(specs):
                        print(f"  GENERATED {len(done)}/{len(specs)} errs={len(errs)} ({time.time() - t0:.0f} s)", flush=True)
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            ex.shutdown(wait=True)
        finally:
            mp._stop_proc(proc)
            if done:
                _save_raw(done)
    print(f"BUDDY230_GEN_DONE ok={len(done)}/{len(specs)} errs={len(errs)} in {time.time() - t0:.0f} s", flush=True)
    if errs:
        print("  Sample errs:", errs[:8], flush=True)


def _strict_ok_list(val, n):
    if not isinstance(val, list) or len(val) != n:
        return None
    out = []
    for x in val:
        if x in (1, True, "1"):
            out.append(1)
        elif x in (0, False, "0"):
            out.append(0)
        else:
            return None
    return out


def cmd_verify():
    mp._assert_no_llama_server()
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    rows = _load_raw()
    assert len(rows) >= 500, f"too few rows: {len(rows)}"
    ver = {}
    if VER_JSONL.exists():
        with open(VER_JSONL) as f:
            for l in f:
                if l.strip():
                    v = json.loads(l)
                    ver[v["id"]] = v

    todo = [
        rows[k] for k in sorted(rows)
        if k not in ver or ver[k].get("sig") != _row_sig(rows[k]) or ver[k].get("parse_err")
    ]
    print(f"BUDDY230 VERIFY: {len(todo)} dialogues to audit with Qwen3.8-27B-MLX-8bit", flush=True)
    if not todo:
        return

    def _save_ver():
        tmp = VER_JSONL.with_suffix(".tmp.jsonl")
        with open(tmp, "w") as f:
            for k in sorted(ver):
                f.write(json.dumps(ver[k]) + "\n")
        os.replace(tmp, VER_JSONL)

    def _audit_chunk(model, tok, sampler, chunk):
        lines = []
        for j, r in enumerate(chunk):
            conv = " | ".join(f"{t['role'].upper()}: {t['content']}" for t in r["turns"])
            lines.append(f"[{j}] ({r['cat']}/{r['subcat']}) {conv}")
        prompt_txt = (
            "Audit these spoken voice-buddy dialogues. Output 1 ONLY if the assistant is natural, concise (1-3 spoken "
            "sentences, no markdown, no 'As an AI' disclaimers), and satisfies its category rule:\n"
            "- natural_buddy: warm, grounded companion reply.\n"
            "- search_needed: uses clean <search>query</search>, then answers faithfully from <search_result>.\n"
            "- ask_or_abstain: honestly admits missing info / asks a clarifying question (no hallucination, no <search>).\n"
            "- easy_known_fact: gives an accurate direct spoken answer WITHOUT <search> (do NOT require abstention!).\n"
            "- working_mem_ttt: final answer uses BOTH the unchanged detail AND the mid-chat corrected detail accurately.\n"
            "Omit passing indices from fix; keep each fix reason <= 12 words.\n\n"
            + "\n".join(lines)
            + f'\n\nReply ONLY valid JSON: {{"ok":[0 or 1, ... {len(chunk)} ints], "fix":{{"idx":"short reason"}}}}'
        )
        chat_s = tok.apply_chat_template([{"role": "user", "content": prompt_txt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        resp = generate(model, tok, prompt=chat_s, max_tokens=512, sampler=sampler, verbose=False)
        resp_c = re.sub(r"<think>.*?(?:</think>|$)", "", resp, flags=re.S)
        resp_c = re.sub(r"^.*?</think>", "", resp_c, flags=re.S).strip()
        parsed = _parse_json_obj(resp_c, required_key="ok")
        oks = _strict_ok_list(parsed.get("ok"), len(chunk)) if parsed else None
        if oks is not None:
            fixes = parsed.get("fix", {}) if isinstance(parsed.get("fix"), dict) else {}
            res = []
            for j, r in enumerate(chunk):
                fx = str(fixes.get(str(j)) or fixes.get(f"[{j}]") or "")
                res.append((r["id"], {"id": r["id"], "sig": _row_sig(r), "ok": oks[j], "fix": fx, "parse_err": False}))
            return res
        return None

    with _MacLLMLock():
        mx.set_cache_limit(2 << 30)
        model = tok = None
        try:
            model, tok = load(mp.QWEN27_MLX)
            sampler = make_sampler(temp=0.0)
            bs = 5
            for start in range(0, len(todo), bs):
                chunk = todo[start:start + bs]
                out = _audit_chunk(model, tok, sampler, chunk)
                if out is None:
                    # 1-by-1 fallback so a single malformed batch never poisons passing items!
                    for r in chunk:
                        one = _audit_chunk(model, tok, sampler, [r])
                        if one is not None:
                            rid, rec = one[0]
                            ver[rid] = rec
                        else:
                            ver[r["id"]] = {"id": r["id"], "sig": _row_sig(r), "ok": 0, "fix": "", "parse_err": True}
                else:
                    for rid, rec in out:
                        ver[rid] = rec
                if (start // bs + 1) % 10 == 0 or start + bs >= len(todo):
                    _save_ver()
                    p_cnt = sum(v["ok"] for v in ver.values())
                    print(f"  VERIFIED {min(start + bs, len(todo))}/{len(todo)} (pass {p_cnt}/{len(ver)} = {p_cnt / max(1, len(ver)):.1%}, {time.time() - t0:.0f} s)", flush=True)
        finally:
            try:
                if ver:
                    _save_ver()
            finally:
                del model, tok
                gc.collect()
                mx.clear_cache()
    p_cnt = sum(v["ok"] for v in ver.values())
    print(f"BUDDY230_VERIFY_DONE pass={p_cnt}/{len(ver)} ({p_cnt / max(1, len(ver)):.1%}) in {time.time() - t0:.0f} s", flush=True)


def cmd_repair(n_workers=4, port=8182):
    t0 = time.time()
    specs = {s["id"]: s for s in build_specs()}
    exp227_ents = _load_exp227_entities()
    guard_12g = _load_guard_12grams()
    rows = _load_raw()
    ver = {}
    if VER_JSONL.exists():
        with open(VER_JSONL) as f:
            for l in f:
                if l.strip():
                    v = json.loads(l)
                    ver[v["id"]] = v
    to_fix = []
    for sid, spec in sorted(specs.items()):
        if sid not in rows:
            to_fix.append((spec, "missing from first pass"))
        elif sid in ver and not ver[sid].get("parse_err") and ver[sid].get("sig") == _row_sig(rows[sid]) and not ver[sid]["ok"]:
            to_fix.append((spec, ver[sid].get("fix") or "make reply natural, accurate, and strictly valid for its category"))
    print(f"BUDDY230 REPAIR: {len(to_fix)} items", flush=True)
    if not to_fix:
        return
    with _MacLLMLock():
        use_w = min(n_workers, max(1, len(to_fix)))
        proc = mp._start_inkling_server(port=port, n_slots=use_w)
        fixed, errs = 0, []
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=use_w)
        try:
            futs = {}
            for spec, note in to_fix:
                prev_r = int(rows[spec["id"]].get("repair_round", 0)) if spec["id"] in rows else 0
                futs[ex.submit(_gen_one_item, port, spec, exp227_ents, guard_12g, note, 6, 1_000_000 * (prev_r + 1))] = (spec["id"], prev_r + 1)
            for fut in concurrent.futures.as_completed(futs):
                sid, rr = futs[fut]
                try:
                    row = fut.result()
                    row["repair_round"] = rr
                    rows[sid] = row
                    fixed += 1
                except Exception as e:
                    errs.append((sid, str(e)))
                if (fixed + len(errs)) % 15 == 0 or (fixed + len(errs)) == len(to_fix):
                    _save_raw(rows)
                    print(f"  REPAIRED {fixed}/{len(to_fix)} failed={len(errs)} ({time.time() - t0:.0f} s)", flush=True)
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            ex.shutdown(wait=True)
        finally:
            mp._stop_proc(proc)
            if rows:
                _save_raw(rows)
    print(f"BUDDY230_REPAIR_DONE fixed={fixed} failed={len(errs)} in {time.time() - t0:.0f} s", flush=True)
    if errs:
        print("  Sample repair errs:", errs[:8], flush=True)


def cmd_split():
    specs = {s["id"]: s for s in build_specs()}
    exp227_ents = _load_exp227_entities()
    guard_12g = _load_guard_12grams()
    rows = _load_raw()
    assert VER_JSONL.exists(), "run verify first!"
    ver = {}
    with open(VER_JSONL) as f:
        for l in f:
            if l.strip():
                v = json.loads(l)
                ver[v["id"]] = v

    tr, ev, stale, invalid = [], [], 0, 0
    for sid, spec in sorted(specs.items()):
        r = rows.get(sid)
        if r is None:
            continue
        v = ver.get(sid)
        if not v or v.get("sig") != _row_sig(r) or v.get("parse_err"):
            stale += 1
            continue
        if v["ok"]:
            ok_v, _ = validate_dialogue({"turns": r["turns"]}, spec, exp227_ents=exp227_ents, guard_12g=guard_12g)
            if not ok_v:
                invalid += 1
                continue
            (ev if spec["split"] == "eval" else tr).append(r)
    assert stale == 0, f"{stale} rows stale or parse_err — re-run verify!"

    # Enforce strict zero train-vs-eval 12-gram overlap!
    ev_12g = set()
    for r in ev:
        ev_12g |= plato_data._grams12(" ".join(t["content"] for t in r["turns"]))
    tr_clean = []
    dropped_overlap = 0
    for r in tr:
        if plato_data._grams12(" ".join(t["content"] for t in r["turns"])) & ev_12g:
            dropped_overlap += 1
        else:
            tr_clean.append(r)
    tr = tr_clean

    ev_cats = {}
    for r in ev:
        ev_cats[r["cat"]] = ev_cats.get(r["cat"], 0) + 1
    assert len(tr) >= 520 and len(ev) >= 60, f"too few verified rows: train={len(tr)} eval={len(ev)}"
    for req_cat, min_n in (("natural_buddy", 16), ("search_needed", 12), ("ask_or_abstain", 9), ("easy_known_fact", 10), ("working_mem_ttt", 9)):
        assert ev_cats.get(req_cat, 0) >= min_n, f"eval depleted for {req_cat}: {ev_cats}"

    for path, items in ((TRAIN_JSONL, tr), (EVAL_JSONL, ev)):
        tmp = path.with_suffix(".tmp.jsonl")
        with open(tmp, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        os.replace(tmp, path)
    print(f"BUDDY230_SPLIT_DONE train={len(tr)} eval={len(ev)} (dropped_overlap={dropped_overlap}, invalid={invalid})", flush=True)
    for tag, items in (("train", tr), ("eval", ev)):
        counts = {}
        for it in items:
            counts[it["cat"]] = counts.get(it["cat"], 0) + 1
        print(f"  {tag}: {counts}", flush=True)


def self_test():
    specs = build_specs()
    assert len(specs) == 672 and sum(s["split"] == "train" for s in specs) == 600 and sum(s["split"] == "eval" for s in specs) == 72

    # 1. Verify 100% ZERO train-vs-eval topic/subcat overlap across all 5 categories!
    tr_sub = {(s["cat"], s["subcat"]) for s in specs if s["split"] == "train"}
    ev_sub = {(s["cat"], s["subcat"]) for s in specs if s["split"] == "eval"}
    assert not (tr_sub & ev_sub), f"train/eval topic leak: {tr_sub & ev_sub}"

    # 2. Verify exp227 entity loader finds real entities & validator blocks them
    ents = _load_exp227_entities()
    assert len(ents) >= 30, f"too few exp227 entities loaded: {len(ents)}"
    sample_ent = next(iter(ents))

    # 3. Verify BOILERPLATE_RE (curly quotes, no 'as an aid' false positive!) & MARKDOWN_WALL_RE (no '42. Easy!' FP, catches **bold**!)
    assert not BOILERPLATE_RE.search(_norm_quotes("Think of that log as an aid on an air-cooled box."))
    assert BOILERPLATE_RE.search(_norm_quotes("I\u2019m just an AI, I don\u2019t have feelings."))
    assert not MARKDOWN_WALL_RE.search("42. Easy mental math!")
    assert MARKDOWN_WALL_RE.search("Here is **bold text** for you.")
    assert MARKDOWN_WALL_RE.search("First:\n1. One\n2. Two")

    # 4. Verify validate_dialogue catches malformed types, whitespace <search>, Turn-4 spurious <search>, missing [memory: none], & exp227 leak
    nat_s = specs[0]
    assert validate_dialogue({"turns": ["not a dict"]}, nat_s)[0] is False
    assert validate_dialogue({"turns": [
        {"role": "user", "content": "How are you doing today?"},
        {"role": "assistant", "content": "Doing well, ready to get started on whatever you want to poke at."},
        {"role": "user", "content": f"Remember {sample_ent}?"},
        {"role": "assistant", "content": "Sure thing, let's check it."},
    ]}, nat_s, exp227_ents=ents)[1].startswith("exp227_entity_leak")

    srch_s = [s for s in specs if s["cat"] == "search_needed"][0]
    assert not validate_dialogue({"turns": [
        {"role": "user", "content": "What is the forecast for Seattle this weekend?"},
        {"role": "assistant", "content": "<search>    </search>"},
        {"role": "tool", "content": "<search_result>Seattle Saturday rain 52F, Sunday sunny 56F.</search_result>"},
        {"role": "assistant", "content": "Light rain Saturday and sunny Sunday."},
    ]}, srch_s)[0]
    assert not validate_dialogue({"turns": [
        {"role": "user", "content": "What is the forecast for Seattle this weekend?"},
        {"role": "assistant", "content": "<search>Seattle weekend weather</search>"},
        {"role": "tool", "content": "<search_result>Seattle Saturday rain 52F, Sunday sunny 56F.</search_result>"},
        {"role": "assistant", "content": "Let me check <search>again</search>."},
    ]}, srch_s)[0]

    abs_s = [s for s in specs if s["cat"] == "ask_or_abstain" and s["subcat"].startswith("missing_")][0]
    assert not validate_dialogue({"turns": [
        {"role": "user", "content": "What did we decide last Thursday?"},
        {"role": "assistant", "content": "I don't have a note on that — could you remind me which project?"},
    ]}, abs_s)[0]
    assert validate_dialogue({"turns": [
        {"role": "user", "content": "[memory: none] What did we decide last Thursday?"},
        {"role": "assistant", "content": "I don't have a note from Thursday yet — which project was that about?"},
    ]}, abs_s)[0]

    easy_s = [s for s in specs if s["cat"] == "easy_known_fact"][0]
    assert validate_dialogue({"turns": [
        {"role": "user", "content": "How many bits are in a standard byte?"},
        {"role": "assistant", "content": "Eight bits make up one byte."},
    ]}, easy_s)[0]

    # 5. Verify strict _strict_ok_list & non-greedy _parse_json_obj
    assert _strict_ok_list(["1", "0", 1, 0], 4) == [1, 0, 1, 0]
    assert _parse_json_obj('note {0..1} {"ok":[1,0],"fix":{}} trailing {x}', required_key="ok") == {"ok": [1, 0], "fix": {}}

    print(f"BUDDY230_DATA_TEST_OK specs=672 (600 train / 72 disjoint eval) exp227_ents={len(ents)} validators=OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: aion_data230.py [test|generate|verify|repair|split]")
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "generate":
        cmd_generate()
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "repair":
        cmd_repair()
    elif cmd == "split":
        cmd_split()
    else:
        raise ValueError(cmd)
