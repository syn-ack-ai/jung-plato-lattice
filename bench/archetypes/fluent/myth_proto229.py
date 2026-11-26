"""Addendum 229 Stage 2A (Amendment 229b): Modernized Public-Domain Myth/Mechanism Generator & Double-Denoised Prototype Builder.

Generates 200 strictly disjoint narrative/causal pattern families (100 classic world myths/fables/folktales + 100
classic causal/systems mechanisms — 0 exact or motif overlap with items3, items7, or any comp/mid eval set) x 5
completely disguised modern everyday retellings = 1,000 clean modern stories (4 aligned narrative beats each).

Strict sequential memory discipline on Mac Studio (64 GB unified RAM — NEVER load 2 LLMs concurrently!):
  1. `test`       unit tests (strict disjointness vs items3/items7, ban vs setting check, LORO math, nonzero-mean & collapse checks)
  2. `generate`   local Inkling-2-Mini (Metal llama-server + DSpark draft, low reasoning budget 128, -np 4) generates 1,000 stories
  3. `verify`     local Qwen3.8-27B-MLX-8bit verifies faithfulness, disguise & cross-setting distinctness (resumable)
  4. `repair`     local Inkling-2-Mini regenerates ONLY failed slots using Qwen3.8 feedback (repair, don't drop!)
  5. `extract`    extracts 30%-depth states across 5 Consensus families (+ optional blind Phi-4/OLMo3), 1 model at a time
  6. `project`    projects into FROZEN Addendum 228 C5 frame (anchored to targets_train.npz) & computes Leave-One-Retelling-Out
                  (LORO) Double-Denoised Prototypes + centered tie-safe cross-setting retrieval diagnostic!

Strictly public text only (zero Jung private text, zero eval 12-gram overlap)."""
import concurrent.futures
import gc
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO = HERE / "plato"
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from fluent import plato_data

RAW_JSONL = PLATO / "myth229_raw.jsonl"
VER_JSONL = PLATO / "myth229_verified.jsonl"
OUT_NPZ = PLATO / "targets_myth229.npz"
META_JSON = PLATO / "myth229_meta.json"

INKLING_GGUF = os.environ.get("INKLING_GGUF_PATH", "Inkling-2-Mini-IQ4_NL.gguf")
DSPARK_GGUF = os.environ.get("INKLING_DSPARK_GGUF_PATH", "dspark-Inkling-2-Mini-Q8_0.gguf")
QWEN27_MLX = os.environ.get("QWEN38_MLX_MODEL", "lmstudio-community/Qwen3.8-27B-MLX-8bit")
LLAMA_SRV = str(Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server")))

# 25 diverse everyday modern settings (5 distinct settings assigned per family)
MODERN_SETTINGS = [
    "a busy suburban emergency clinic",
    "a neighborhood sourdough bakery",
    "a small software engineering team",
    "a municipal bus transit depot",
    "a local bicycle repair shop",
    "a city public library branch",
    "a regional commuter airport",
    "a community theater backstage crew",
    "a coastal harbor freight dock",
    "a high school chemistry laboratory",
    "a district fire station",
    "a local public radio studio",
    "a city botanical conservatory",
    "a freight rail switching yard",
    "a residential architecture studio",
    "a family dental practice",
    "a coastal weather observatory",
    "a small town weekly newspaper",
    "a regional power grid control room",
    "a county mountain rescue team",
    "a cooperative pottery workshop",
    "a regional grocery distribution warehouse",
    "a civic youth orchestra",
    "a suburban veterinary clinic",
    "a municipal water treatment plant",
]

ARCHAIC_BAN = re.compile(
    r"\b(king|queen|prince|princess|emperor|empress|throne|kingdom|palace|castle|knight|sword|spear|shield|"
    r"god|gods|goddess|deity|zeus|apollo|athena|odin|thor|loki|magic|magical|spell|wizard|witch|sorcerer|"
    r"fairy|dragon|giant|monster|curse|prophecy|oracle|myth|fable|parable|folktale|legend|moral of the story|"
    r"once upon a time|aesop|homer|olympus|valhalla|underworld)\b",
    re.I,
)

PREFIX_BAN = re.compile(
    r"^\s*(?:[#\-*\"'\u201c]|(?:title|beat|sentence|story|vignette|scene)\s*\d*\s*:|\(?1[\).]\s)",
    re.I,
)
INLINE_BEAT_BAN = re.compile(r"\([1-4]\)|beat\s+[1-4]\s*:", re.I)

# 100 classic world myths / fables / folktales / parables — STRICTLY DISJOINT from items3, items7, and each other!
# Tuple format: (fam_id, pattern_summary, explicit_disguise_ban_terms)
MYTH_FAMILIES = [
    ("olive_tree_contest", "competing to win a community's trust, one rival offers a flashy dramatic display while the other offers a modest, endlessly useful gift that sustains everyday life and wins", ("athena", "poseidon", "olive tree")),
    ("midas", "someone gets a wish for everything they touch to turn into their prized objective, only to ruin what they actually need to live", ("midas", "golden touch")),
    ("pandora", "forbidden curiosity makes someone open a sealed container, unleashing lasting troubles while leaving one small sliver of hope inside", ("pandora", "pandora's")),
    ("prometheus", "a rebel steals guarded knowledge or tools from hoarding elites to share with ordinary people and pays a harsh personal penalty", ("prometheus",)),
    ("odysseus_sirens", "knowing in advance that an irresistible temptation lies ahead, a planner binds their own hands and blocks teammates' exposure before entering the danger zone so they pass through safely", ("odysseus", "ulysses", "sirens")),
    ("sisyphus", "someone is trapped repeating a grueling uphill task whose progress slips back to the bottom just before completion", ("sisyphus", "sisyphean")),
    ("narcissus", "someone becomes so absorbed in admiring their own image or metrics that they neglect real relationships and waste away", ("narcissus", "narcissist", "narcissism")),
    ("tantalus", "someone is placed inches away from the exact reward they crave, yet every time they reach for it, it pulls just out of grasp", ("tantalus", "tantalize")),
    ("pygmalion", "a meticulous creator pours devotion into building an idealized work until their dedication brings an unexpectedly living outcome", ("pygmalion", "galatea")),
    ("persephone", "a small careless bargain made in a strange place binds someone to split their time permanently between two worlds", ("persephone", "pomegranate", "hades")),
    ("cupid_psyche", "suspicion whispered by jealous onlookers tempts a partner to break a promise of trust, forcing an arduous quest to repair the bond", ("cupid", "psyche", "eros")),
    ("bellerophon", "after brilliant victories won with careful humility, hubris drives a champion to attempt an impossible ascent and suffer a steep fall", ("bellerophon", "pegasus", "chimera")),
    ("heracles_antaeus", "an opponent seems unbeatable as long as they stay connected to their hidden source of support; lifting them off that support ends the fight", ("heracles", "hercules", "antaeus")),
    ("daedalus_labyrinth", "an ingenious designer builds a complex maze so intricate that even they must invent an unconventional escape to get out", ("daedalus", "labyrinth", "minos", "minotaur")),
    ("echo_voice", "someone loses the capacity to initiate original statements and can only repeat fragments of whatever the last speaker said", ("nymph", "hera")),
    ("cadmus_teeth", "sowing a sharp provocation into a hostile rival force causes the rivals to turn on one another until only a tiny cooperative remnant remains", ("cadmus", "spartoi")),
    ("alcestis", "when no one else will accept a heavy cost on another's behalf, one devoted partner steps forward voluntarily and is ultimately rescued", ("alcestis", "admetus")),
    ("king_canute", "a sensible leader stages a demonstration of nature ignoring their orders to cure flatterers of believing authority overrides physical reality", ("canute", "cnut")),
    ("cincinnatus", "a capable outsider is summoned from quiet life to solve an acute crisis, resolves it swiftly, and immediately hands back power", ("cincinnatus", "dictator")),
    ("horatius_bridge", "one defender holds a narrow chokepoint just long enough for teammates behind them to cut off an advancing threat", ("horatius", "tiber")),
    ("pyrrhic_victory", "a side pushes so hard to win a single contest that the resources lost in winning guarantee defeat in the wider campaign", ("pyrrhic", "pyrrhus")),
    ("crossing_rubicon", "someone takes one small, unmistakable step past a boundary from which retreat is impossible, committing fully to the consequences", ("rubicon", "caesar")),
    ("prodigal_son", "a reckless junior squanders their share far from home, returns humbled expecting rejection, and is welcomed back while the steady sibling resents it", ("prodigal",)),
    ("jonah_duty", "someone runs in the opposite direction to avoid a difficult assignment, gets caught in a crisis their flight caused, and ends up doing the job anyway", ("jonah", "nineveh", "whale")),
    ("tower_of_babel", "an overconfident group builds an ever taller shared structure until fractured communication leaves them unable to coordinate and the project stalls", ("babel",)),
    ("noahs_ark", "someone prepares patiently for a coming disaster while neighbors mock them, saving a core remnant when the crisis hits", ("noah", "ark")),
    ("cain_abel", "bitter resentment over a sibling's better-received work festers into a destructive act that the perpetrator then tries to deny", ("cain", "abel")),
    ("gideon_trim", "a leader facing huge odds intentionally trims an oversized, unfocused group down to a tiny, alert core and succeeds through surprise", ("gideon", "midian")),
    ("balaam_warning", "a stubborn traveler angrily pushes forward while a humble helper keeps swerving away from an unseen hazard right ahead", ("balaam",)),
    ("crow_and_pitcher", "unable to reach a scarce resource directly, a patient problem-solver drops small increments in one by one until the level rises within reach", ("pitcher",)),
    ("fox_and_crow", "a cunning flatterer showers lavish praise on someone holding a prize until vanity makes them open up and drop it", ("fox", "crow")),
    ("belling_the_cat", "a committee enthusiastically approves a clever plan to stop a threat, then falls silent when asked who will actually take the risk to carry it out", ("bell the cat", "belling")),
    ("wolf_sheeps_clothing", "a predator disguises themselves in the uniform of a harmless group to slip past defenses until a small inconsistency gives them away", ("sheep's clothing", "wolf")),
    ("dog_in_manger", "someone guards a resource they cannot use themselves purely to keep anyone else from benefiting from it", ("manger",)),
    ("fox_and_stork", "a host serves a guest in a format only the host can use, and the guest returns the exact same treatment in a format only the guest can use", ("stork", "fox")),
    ("milkmaid_pail", "daydreaming through an elaborate chain of future windfalls from one fragile asset, someone loses focus and breaks the asset on the spot", ("milkmaid",)),
    ("bundle_of_sticks", "individuals who are easily broken one by one discover they are unbreakable when bound together as a unit", ("bundle of sticks",)),
    ("oak_and_reed", "a rigid giant snaps in a fierce storm while a flexible smaller neighbor bends with the blast and springs back intact", ("reed", "oak")),
    ("frog_and_ox", "trying to inflate themselves to match a naturally larger competitor, someone strains past their structural limit and bursts", ("frog", "ox")),
    ("donkey_and_salt", "a trick that lightened a load once backfires disastrously when repeated blindly under a different cargo that absorbs water", ("donkey", "sponges")),
    ("cats_paw", "a smooth persuader flatters a naive helper into doing a risky task so the persuader keeps the reward while the helper takes the burn", ("cat's paw", "chestnuts")),
    ("fisherman_wife", "repeatedly granted larger upgrades, an insatiable partner keeps demanding more until one excessive demand resets everything back to the start", ("flounder", "fisherman")),
    ("bremen_musicians", "four aging, cast-off misfits pool their modest remaining strengths, coordinate a clever bluff, and win themselves a secure home", ("bremen",)),
    ("brave_little_tailor", "a quick-witted ordinary worker turns an exaggerated boast and smart psychology into outmaneuvering much stronger rivals", ("tailor", "seven at one blow")),
    ("shoemaker_elves", "quiet anonymous helpers fix stalled work overnight; when the grateful owner leaves a direct reward, the helpers celebrate and move on", ("shoemaker", "elves", "cobbler")),
    ("bluebeard_room", "given free run of a grand place except one locked room, a newcomer opens it and discovers evidence of a dark hidden pattern", ("bluebeard",)),
    ("pied_piper", "a specialist clears a severe nuisance for an agreed fee; when leaders renege after the problem vanishes, the specialist extracts a far harsher cost", ("pied piper", "hamelin")),
    ("cinderella_slipper", "an overlooked worker hidden behind daily chores shines briefly on merit and is later identified by one exact, unforgeable fit", ("cinderella", "glass slipper")),
    ("jack_beanstalk", "trading a practical asset for a seemingly foolish gamble opens access to an unexpected high-stakes opportunity that requires nerve to survive", ("beanstalk", "golden harp")),
    ("pinocchio_nose", "every small fabricated excuse creates an increasingly visible tell that grows harder to conceal until honesty reverses it", ("pinocchio", "geppetto")),
    ("red_riding_hood", "straying off a clear path to chat with a plausible stranger gives the stranger the exact information needed to set a trap ahead", ("riding hood", "grandmother", "wolf")),
    ("snow_white_mirror", "obsessed with remaining number one, a powerful rival sabotages a rising newcomer in disguise, only for humble allies to save the newcomer", ("snow white", "poisoned apple", "dwarfs")),
    ("urashima_taro", "a kind helper visits an enchanting retreat where time passes differently; returning home, they discover everyone they knew is long gone", ("urashima", "taro", "dragon palace")),
    ("issun_boshi", "underestimated because of tiny stature, a resourceful junior slips inside a huge threat's defenses where nobody else can reach and neutralizes it from within", ("issun", "boshi", "oni")),
    ("bamboo_cutter", "humble caretakers raise an extraordinary foundling who sets impossible tests for shallow suitors before returning to her distant origin", ("kaguya", "bamboo cutter")),
    ("momotaro_party", "a determined young traveler shares simple provisions with unlikely companions along the road, forming a loyal team that overcomes a fortress", ("momotaro", "peach boy")),
    ("foolish_old_man", "mocked for clearing an obstacle far too big for one lifetime, a patient planner points out that steady multi-generation effort will finish it", ("yugong", "foolish old man")),
    ("jingwei_sea", "after a terrible loss, a tiny persistent worker keeps dropping pebbles one by one into a vast hazard, refusing to surrender to scale", ("jingwei",)),
    ("kuafu_sun", "chasing an endlessly receding goal at full sprint without pacing burns out the runner right before reaching refreshment", ("kuafu",)),
    ("white_snake", "a devoted partner hides an unconventional background to live peacefully until an intolerant outsider exposes them and splits the family", ("bai suzhen", "fahai")),
    ("monkey_crocodile", "lured into a trap by a false invitation, a sharp captive invents a reason why the prize was left back at safety and talks their way free", ("crocodile", "monkey")),
    ("blue_jackal", "an accidental makeover fools an entire community into treating an ordinary outsider as elite until an instinctive slip reveals the truth", ("jackal", "indigo")),
    ("talkative_turtle", "carried safely across danger by friends on one rule of keeping quiet, someone snaps back at jeering bystanders and lets go", ("turtle", "tortoise")),
    ("four_friends_trap", "four very different small allies combine speed, lookout, stealth, and cutting to free a captured friend no single one could rescue", ("panchatantra",)),
    ("maui_slows_sun", "when daily cycles rush past too fast for anyone to finish work, a daring team ropes and slows the pace down to a livable rhythm", ("maui",)),
    ("raven_steals_sun", "a sly trickster infiltrates a hoarder's house in harmless disguise to release a light meant for everyone", ("raven",)),
    ("coyote_steals_fire", "a relay of ordinary runners passes a guarded resource hand to hand across a gauntlet so no single interceptor can reclaim it", ("coyote",)),
    ("sedna_sea", "breaking faith with a vulnerable caretaker creates a storm of scarcity that only lifts when people return and make respectful amends", ("sedna",)),
    ("beowulf_grendel", "an outside specialist confronts a threat others could not contain, defeats it without fancy weapons, and then must face the deeper source behind it", ("beowulf", "grendel", "heorot")),
    ("sword_in_stone", "a humble assistant casually retrieves a needed tool nobody else could budge, revealing quiet legitimacy without having sought status", ("arthur", "excalibur", "merlin")),
    ("gawain_green_knight", "a challenger honors a frightening promise to return on time, stumbles on one tiny hidden compromise, and learns humility from the test", ("gawain", "green knight", "camelot")),
    ("william_tell", "forced by a petty tyrant to make an impossibly narrow shot under threat, a skilled marksman hits the mark while holding a backup plan", ("william tell", "gessler", "crossbow")),
    ("faust_bargain", "someone trades their long-term integrity for immediate mastery and convenience, only to face the bill coming due", ("faust", "faustian", "mephistopheles")),
    ("don_quixote_windmills", "swept up in grand romantic narratives, someone mistakes ordinary everyday machinery for heroic enemies and charges straight into embarrassment", ("quixote", "sancho", "windmills")),
    ("midas_barber_reeds", "unable to bear keeping an embarrassing secret about the boss, a confidant whispers it into a hidden spot that ends up broadcasting it across the whole town", ("barber", "donkey ears")),
    ("john_henry", "a proud veteran races a newly introduced automated machine to prove human craft can match it, wins the race, and collapses from the strain", ("john henry", "steam drill")),
    ("aladdin_lamp", "an ordinary street kid discovers a dusty overlooked tool of immense power, outsmarts the schemer who wanted it, and learns restraint", ("aladdin", "genie")),
    ("sinbad_roc", "stranded in an isolated valley, a resourceful traveler spots a predictable giant pattern and straps onto it to escape", ("sinbad", "roc")),
    ("gilgamesh_enkidu", "two fiercely matched rivals clash head-on, earn deep mutual respect, and become inseparable partners until loss teaches humility", ("gilgamesh", "enkidu", "uruk")),
    ("hephaestus_net", "instead of confronting a slick pair in an open fight, a quiet craftsperson rigs an invisible trap that catches them red-handed in front of witnesses", ("hephaestus", "vulcan", "aphrodite", "ares")),
    ("thor_stolen_hammer", "when a vital piece of gear is stolen for ransom, a blunt powerhouse reluctantly uses clever disguise instead of brute force to recover it", ("mjolnir", "thrym")),
    ("selkie_skin", "keeping a partner bound by hiding the key to their freedom works for years, until the moment they find it and leave immediately", ("selkie", "sealskin")),
    ("finn_salmon", "an apprentice tending a slow careful preparation accidentally tastes the very first drop meant for the master and gains the insight first", ("finn maccool", "fionn", "salmon of knowledge")),
    ("flying_dutchman", "a reckless captain swears to force passage against impossible conditions and ends up doomed to endless fruitless loops", ("flying dutchman",)),
    ("robin_hood_sheriff", "an official squeezes ordinary residents unfairly until a clever outlaw redirects the excess back to the people and mocks the official's traps", ("robin hood", "nottingham", "sherwood")),
    ("tristan_iseult", "an accidental mix-up creates an unbreakable loyalty that conflicts directly with duty to a trusted leader", ("tristan", "iseult", "isolde")),
    ("roland_horn", "out of stubborn pride, a commander refuses to call for backup until the crisis is far too late to save the squad", ("roland", "olifant", "roncevaux", "charlemagne")),
    ("jackdaw_borrowed_feathers", "someone decorates themselves with other people's credentials and work to join an elite group, until each rightful owner reclaims their piece in public and exposes them", ("jackdaw", "peacock")),
    ("three_wishes_sausage", "granted scarce high-value opportunities, a squabbling pair wastes the first on a petty whim, burns the second in anger, and spends the third merely undoing the mess", ("three wishes", "sausage")),
    ("ass_in_lions_skin", "an intimidating disguise makes everyone clear the way in fear until the impostor opens their mouth to boast and immediately gives themselves away", ("lion's skin", "donkey")),
    ("wolf_and_crane", "someone begs a specialist for risky emergency help with big promises, then afterward claims that merely walking away unharmed is reward enough", ("wolf", "crane")),
    ("honest_woodcutter", "when offered shiny upgraded replacements for a modest lost tool, someone insists truthfully on only their own plain one and is rewarded", ("woodcutter", "hermes")),
    ("man_and_satyr", "someone loses trust in a companion who uses the same simple action to warm cold hands and cool hot soup, mistaking one versatile tool for two-faced deceit", ("satyr", "blow hot and cold")),
    ("astronomer_in_well", "so absorbed in scanning distant abstract horizons that someone misses an obvious hazard right under their feet until a practical bystander pulls them out", ("astronomer", "thales")),
    ("crab_and_mother", "an advisor scolds a junior for walking crookedly while demonstrating the exact same crooked habit themselves", ("crab",)),
    ("hare_many_friends", "someone who relies on casual acquaintances for emergency help learns each one passes the buck until they must rely on themselves", ("hare", "hounds")),
    ("shepherd_and_sea", "drawn by a calm stretch into risking everything on one venture, a cautious worker gets burned by a sudden squall and never trusts calm appearances again", ("shepherd",)),
    ("miller_son_donkey", "trying to please every passing critic's contradictory advice leaves a team exhausted, ridiculous, and worse off than when they started", ("donkey",)),
    ("two_pots_river", "a fragile partner refuses to travel closely beside a heavy sturdy partner because even a friendly bump would shatter the fragile one", ("earthen pot", "brass pot")),
    ("belly_and_members", "visible workers strike against an invisible coordinating center they think does nothing, only to grow weak themselves until they restore cooperation", ("belly", "fable of the belly")),
]

# 100 classic causal / decision / systems mechanisms — STRICTLY DISJOINT from items3, items7, and each other!
MECH_FAMILIES = [
    ("tragedy_commons", "individual users each take a little extra from a shared resource because the gain is private while the depletion is split across everyone, until the shared resource collapses", ("tragedy of the commons",)),
    ("moral_hazard", "once someone is shielded from bearing the downside of risky choices, they quietly take bigger risks than they would if exposed", ("moral hazard",)),
    ("sunk_cost", "because a team has already poured heavy time and money into a failing effort, they keep funding it instead of cutting losses", ("sunk cost", "sunk-cost")),
    ("cobra_effect", "a reward paid per reported problem accidentally incentivizes people to manufacture more of the problem to collect the reward", ("cobra effect", "perverse incentive")),
    ("goodharts_law", "once a useful proxy metric becomes the official target for rewards, people game the number until it stops measuring the real goal", ("goodhart",)),
    ("survivorship_bias", "studying only the cases that made it through screening hides the silent failures and leads analysts to reinforce the wrong spots", ("survivorship bias", "abraham wald")),
    ("confirmation_bias", "already leaning toward one explanation, an investigator eagerly accepts supporting clues while explaining away clear contradicting evidence", ("confirmation bias",)),
    ("framing_effect", "presenting the exact same numerical odds as a loss rather than a gain flips decision-makers from cautious to risk-seeking", ("framing effect",)),
    ("dunning_kruger", "a beginner who knows only the basics feels brimming with confidence, while a seasoned specialist sees all the edge cases and speaks cautiously", ("dunning", "kruger")),
    ("availability_heuristic", "one vivid recent incident looms so large in people's minds that they vastly overestimate its frequency compared to quieter, larger risks", ("availability heuristic", "availability bias")),
    ("halo_effect", "one standout visible strength leads evaluators to assume excellence across completely unrelated skills without checking", ("halo effect",)),
    ("gamblers_fallacy", "after a string of identical independent outcomes, someone bets heavily that the opposite result is overdue even though the odds never changed", ("gambler's fallacy", "gamblers fallacy")),
    ("prisoners_dilemma", "two sides would both fare well by cooperating, yet each fears the other will defect first, so both defect and end up worse off", ("prisoner's dilemma", "prisoners dilemma")),
    ("stag_hunt", "a high payoff requires two partners to trust and act together, while a modest solo option is safe; any flicker of doubt pulls both into the modest solo choice", ("stag hunt",)),
    ("ultimatum_game", "offered a small share of a windfall that feels blatantly unfair, a recipient rejects the deal on principle so neither side gets anything", ("ultimatum game",)),
    ("dollar_auction", "two bidders keep escalating past what a prize is actually worth because the second-place bidder still has to pay their bid and tries to minimize losses", ("dollar auction",)),
    ("free_rider", "because everyone benefits from a shared maintenance task regardless of who pitches in, contributors shrink until the effort stalls", ("free rider", "free-rider")),
    ("lindy_effect", "a simple tool or practice that has already survived decades of testing keeps outlasting flashy newcomers that break within months", ("lindy",)),
    ("chestertons_fence", "a newcomer tears down an old rule whose purpose looks obsolete, only to immediately trigger the exact hazard the rule was quietly preventing", ("chesterton",)),
    ("jevons_paradox", "making a resource much more efficient lowers its effective cost so much that total consumption rises rather than falls", ("jevons",)),
    ("induced_demand", "widening a crowded route briefly eases congestion, which attracts enough extra traffic that delays soon return to where they started", ("induced demand",)),
    ("hawthorne_effect", "simply announcing that a workflow is being observed makes workers temporarily improve performance regardless of whether the new change actually helps", ("hawthorne",)),
    ("streetlight_effect", "instead of investigating where a problem actually occurs, searchers look only in the easy, well-lit area where data happens to be convenient", ("streetlight effect", "drunkard's search")),
    ("law_of_instrument", "someone deeply attached to one favorite tool tries to solve every unrelated problem with that same tool even when it fits poorly", ("law of the instrument", "maslow's hammer")),
    ("bikeshedding", "a review committee rubber-stamps a huge complex decision in two minutes, then argues for an hour over a trivial familiar detail", ("bikeshed", "parkinson's law of triviality")),
    ("hofstadters_law", "a project runs behind schedule even after the planners deliberately padded the timeline to account for overruns", ("hofstadter",)),
    ("brandolinis_law", "a quick careless rumor takes ten seconds to spread, yet disproving it thoroughly takes ten times the effort and still lags behind", ("brandolini", "bullshit asymmetry")),
    ("network_effect", "each additional user joining a shared platform makes it more valuable for every existing member, creating runaway pull toward the leader", ("network effect", "metcalfe")),
    ("path_dependence", "an arbitrary early standard becomes so embedded in equipment and habits that nobody switches even after a clearly superior design arrives", ("path dependence", "qwerty")),
    ("innovators_dilemma", "a dominant provider listens carefully to its best current customers and ignores a crude cheaper rival until that rival improves and overtakes the market", ("innovator's dilemma", "christensen")),
    ("tipping_point", "gradual pressure produces almost no visible change for a long stretch until one tiny additional increment flips the whole system abruptly", ("tipping point",)),
    ("resonance_amplification", "small rhythmic pushes timed to match a structure's natural swing build up into huge violent oscillations", ("resonance",)),
    ("theory_of_constraints", "speeding up fast steps upstream of a single slow bottleneck only piles up waiting backlog instead of raising total output", ("theory of constraints", "goldratt")),
    ("littles_law", "letting twice as many unfinished jobs sit in progress at the same time doubles how long each customer waits even though throughput stays flat", ("little's law", "littles law")),
    ("square_cube_scaling", "scaling up a proven small design uniformly multiplies its weight and heat faster than its supporting surface area, causing the oversized version to overheat or buckle", ("square-cube", "square cube")),
    ("berksons_paradox", "looking only at accepted cases makes two completely independent strengths look negatively correlated because weakness in one required strength in the other", ("berkson",)),
    ("illusion_of_control", "giving people meaningless cosmetic dials or buttons makes them believe they are steering an outcome that is actually pure chance", ("illusion of control",)),
    ("overfitting_rule", "a procedure tuned too tightly to last month's exact quirks performs flawlessly on old records and stumbles badly on fresh real cases", ("overfitting", "overfit")),
    ("explore_exploit", "sticking exclusively to today's best known option misses far better alternatives, while constantly trying new options wastes time that could harvest wins", ("explore-exploit", "multi-armed bandit")),
    ("secretary_problem", "sampling a first batch purely to calibrate standards before committing to the next candidate that beats the baseline yields a near-optimal pick", ("secretary problem", "optimal stopping")),
    ("mere_exposure_effect", "simply encountering an unfamiliar neutral option repeatedly without any change in quality makes people gradually prefer it over equally good novelties", ("mere exposure", "zajonc")),
    ("swiss_cheese_model", "a serious failure happens only when small harmless gaps in several independent safety layers happen to line up at the exact same moment", ("swiss cheese", "james reason")),
    ("cascading_failure", "one overloaded node trips offline and dumps its workload onto neighbors, pushing each neighbor past capacity in a rapid domino chain", ("cascading failure",)),
    ("single_point_failure", "an elaborate system with plenty of backups still halts completely because every path passes through one unbacked shared component", ("single point of failure", "spof")),
    ("graceful_degradation", "under extreme overload, a well-designed operation deliberately drops non-essential extras so the core vital service stays steady", ("graceful degradation",)),
    ("groupthink", "a tight-knit team prizes harmony so highly that doubters stay quiet and the group marches unanimously into an avoidable blunder", ("groupthink",)),
    ("abilene_paradox", "every member privately dislikes a proposed outing, yet each goes along assuming everyone else wants it, leaving the whole group miserable", ("abilene",)),
    ("asch_conformity", "even when an answer is visibly plain, hearing several peers state the same wrong answer pressures a lone holdout to doubt their own eyes", ("asch",)),
    ("foot_in_the_door", "getting someone to agree to one tiny harmless request makes them markedly more likely to say yes to a much larger follow-up request", ("foot-in-the-door", "foot in the door")),
    ("door_in_the_face", "opening with an extreme request that gets refused makes the moderate target request that follows seem reasonable and easy to accept", ("door-in-the-face", "door in the face")),
    ("decoy_effect", "introducing a slightly worse third option that nobody picks nudges buyers away from the basic choice toward the pricier premium option", ("decoy effect", "asymmetric dominance")),
    ("endowment_effect", "merely owning an ordinary item for a short while makes someone demand far more to give it up than they would ever pay to buy it", ("endowment effect",)),
    ("loss_aversion", "the sting of losing a modest amount motivates far stronger action than the prospect of gaining the exact same amount", ("loss aversion", "prospect theory")),
    ("hyperbolic_discounting", "someone happily chooses a larger later reward when both options are far away, yet flips to the smaller immediate reward as soon as it is right in front of them", ("hyperbolic discounting", "present bias")),
    ("ikea_effort_valuation", "people attach noticeably higher value and loyalty to an item or plan they helped assemble themselves than to an identical ready-made one", ("ikea effect", "effort justification")),
    ("zeigarnik_effect", "tasks left half-finished keep nagging at attention and stay vividly remembered, whereas completed tasks fade smoothly out of mind", ("zeigarnik",)),
    ("paradox_of_choice", "expanding a menu from a few clear options to dozens overwhelms customers so fewer of them pick anything and those who do feel less satisfied", ("paradox of choice", "choice overload")),
    ("peak_end_rule", "people judge an entire long experience almost entirely by its single sharpest moment and how it ended, rather than its average minute", ("peak-end", "peak end")),
    ("spotlight_effect", "someone agonizes over a minor visible blunder assuming everyone noticed, when in reality almost nobody paid attention", ("spotlight effect",)),
    ("fundamental_attribution", "when a teammate is late we blame their careless character, yet when we are late ourselves we point to unavoidable traffic and context", ("fundamental attribution",)),
    ("reactance_backfire", "issuing a blunt heavy-handed prohibition makes people feel their freedom is threatened and drives them to crave and pursue the forbidden option more strongly", ("reactance", "psychological reactance")),
    ("omission_bias", "people judge a harmful outcome caused by taking an action far more harshly than an equally bad outcome caused by doing nothing", ("omission bias",)),
    ("status_quo_bias", "keeping whichever option is pre-selected as the default wins overwhelmingly even when switching takes almost zero effort", ("status quo bias", "default effect")),
    ("recency_bias", "reviewers weight the most recent week far more heavily than eleven earlier months of steady performance", ("recency bias",)),
    ("echo_chamber", "filtering out dissenting voices leaves a group hearing only louder versions of its own assumptions until its views drift to extremes", ("echo chamber",)),
    ("diffusion_of_innovation", "a useful new method sits unnoticed with a few tinkerers, catches on with pragmatic trendsetters, and then sweeps the cautious majority", ("diffusion of innovation", "everett rogers")),
    ("choice_supportive_bias", "immediately after picking between two close options, someone begins exaggerating the flaws of the rejected option and the virtues of the one they picked", ("choice-supportive", "post-purchase rationalization")),
    ("first_mover_burden", "the pioneer spends heavily educating users and discovering pitfalls, allowing a fast follower to copy what works and skip the costly mistakes", ("first-mover", "fast follower")),
    ("long_tail", "a few hits dominate immediate attention, yet the huge spread of small niche requests collectively equals or exceeds the hits", ("long tail",)),
    ("percolation_threshold", "adding scattered links does little for overall reach until density crosses one critical mark where isolated pockets suddenly merge into one connected network", ("percolation",)),
    ("ecological_fallacy", "applying a broad regional average directly to one specific local case gets the individual prediction backward", ("ecological fallacy",)),
    ("selection_bias", "relying only on volunteers who chose to fill out a voluntary survey skews conclusions toward the most intense opinions", ("selection bias", "self-selection")),
    ("curse_of_extra_knobs", "adding too many extra criteria to a small sample makes every case look unique and breaks simple reliable patterns", ("curse of dimensionality",)),
    ("kelly_sizing", "betting the entire stake on a favorable edge eventually hits one bad streak and wipes out, whereas sizing stakes proportionally compounds steadily", ("kelly criterion",)),
    ("redundancy_failover", "keeping a hot spare ready on standby lets a service absorb a sudden hardware failure with barely a flicker", ("failover",)),
    ("observer_interference", "the very act of inserting a heavy probe to measure a delicate process disturbs the process and alters the reading", ("observer effect", "heisenberg")),
    ("curse_of_knowledge", "once a specialist masters a concept deeply, they can no longer imagine what it feels like not to know it, so their instructions skip essential basics for beginners", ("curse of knowledge",)),
    ("broken_windows_drift", "leaving small visible maintenance lapses unaddressed signals nobody is watching and invites larger neglect to spread", ("broken windows",)),
    ("pygmalion_expectation", "quietly giving a randomly selected group high expectations and patient coaching elicits genuinely stronger performance", ("pygmalion effect", "rosenthal")),
    ("dunbar_scaling", "as a friendly group grows past the size where everyone knows everyone directly, informal trust breaks down and explicit roles become necessary", ("dunbar",)),
    ("benfords_leading_digit", "naturally spanning measurements start far more often with small leading digits than large ones, so fabricated uniform numbers stand out immediately on audit", ("benford",)),
    ("normal_accident", "in a tightly coupled, complex system where parts interact unexpectedly, minor routine slips combine faster than operators can diagnose them", ("normal accident", "perrow")),
    ("contrast_effect", "judging an ordinary option right after seeing an extreme alternative makes it look dramatically better or worse than it would on its own", ("contrast effect",)),
    ("rebound_overload", "fixing an upstream clog suddenly sends a pent-up wave downstream that floods the next stage", ("thundering herd",)),
    ("calibration_drift", "tiny unnoticed shifts in a measuring instrument accumulate over weeks until every batch quietly misses spec", ("calibration drift",)),
    ("batch_starvation", "prioritizing only urgent rush tickets starves routine preventive maintenance until routine systems start turning into urgent failures", ("starvation",)),
    ("local_optimum_trap", "making only small improvements that immediately help keeps a workflow stuck on a mediocre hill because reaching a far better design requires a temporary dip", ("local optimum", "hill climbing")),
    ("ringelmann_loafing", "as more people are added to a shared pull where individual effort is unmeasured, each person quietly eases up so average effort per worker drops", ("ringelmann", "social loafing")),
    ("conjunction_fallacy", "adding vivid specific details makes a combined scenario feel more plausible than the broader single event, even though every extra condition lowers the true probability", ("conjunction fallacy", "linda problem")),
    ("incentive_misalignment", "paying intake clerks for speed and repair techs for rework creates a steady stream of sloppy intake tickets", ("incentive misalignment",)),
    ("queuing_nonlinear_knee", "pushing a service desk from eighty percent busy to ninety-eight percent busy does not nudge wait times slightly; waits explode nonlinearly toward gridlock", ("queuing theory", "kingman")),
    ("licensing_effect", "doing one conspicuous good deed makes someone feel they have earned slack and quietly lowers their guard on the very next choice", ("moral licensing", "licensing effect")),
    ("survivor_workload_penalty", "consistently handing tough extra work to the most reliable member burns them out while unreliable members keep light loads", ("performance punishment",)),
    ("hold_up_problem", "once one partner invests heavily in custom equipment useful only to a specific buyer, the buyer gains leverage to squeeze the deal terms afterward", ("hold-up problem", "holdup problem")),
    ("commitment_device", "deliberately adding an upfront penalty or locking away an easy exit binds a person's future self to stick to a difficult long-term plan", ("commitment device", "ulysses pact")),
    ("j_curve_dip", "introducing a genuinely better process initially drops productivity while workers unlearn old habits, before performance climbs far above the old ceiling", ("j-curve", "j curve")),
    ("stiction_slip", "friction holds a stuck part completely motionless as force climbs until it suddenly snaps loose and overshoots far past the target", ("stiction", "stick-slip")),
    ("second_system_bloat", "after succeeding with a lean, focused first design, the team crams every postponed wish-list feature into version two until it becomes bloated and unusable", ("second-system", "second system effect")),
    ("leaky_abstraction", "a convenient simplified wrapper hides messy low-level details during routine work, until an edge case breaks through and forces operators to understand the hidden machinery underneath", ("leaky abstraction", "spolsky")),
    ("cross_subsidy_unraveling", "when a competitor skims off only the easy profitable cases, the broad service left with only hard cases can no longer break even", ("cream skimming", "cross-subsidy")),
]


def all_families():
    out = []
    for idx, (fid, pat, bans) in enumerate(MYTH_FAMILIES):
        out.append({"fam_idx": idx, "fam_id": fid, "kind": "myth", "pattern": pat, "bans": tuple(bans)})
    off = len(MYTH_FAMILIES)
    for idx, (fid, pat, bans) in enumerate(MECH_FAMILIES):
        out.append({"fam_idx": off + idx, "fam_id": fid, "kind": "mech", "pattern": pat, "bans": tuple(bans)})
    return out


def settings_for_family(fam_idx):
    """Deterministically pick 5 distinct modern settings for family fam_idx."""
    rng = random.Random(22900 + fam_idx)
    return rng.sample(MODERN_SETTINGS, 5)


def segment_4beats(text):
    """Split a modern vignette into exact 4 narrative beats, preferring clean sentence endings (even when a 4-sentence
    vignette has exactly 3 interior sentence breaks!) before falling back to plato_data.segment_passage."""
    text = text.strip()
    if not text:
        return []
    sent_ends = [m.end() for m in re.finditer(r"[.!?][\"'\u201d\u2019)]*\s+", text)]
    L = len(text)
    if len(sent_ends) == 3:
        cuts = sent_ends + [L]
        segs, prev = [], 0
        for pos in cuts:
            if pos > prev:
                segs.append(text[prev:pos])
                prev = pos
        if len(segs) == 4 and all(len(s.strip()) >= 15 for s in segs):
            assert "".join(segs) == text
            return segs
    if len(sent_ends) > 3:
        chosen = []
        for k in range(1, 4):
            tgt = int(round(L * k / 4.0))
            best = min(sent_ends, key=lambda c: abs(c - tgt))
            if 15 <= best <= L - 15 and (not chosen or best - chosen[-1] >= 15):
                chosen.append(best)
        if len(chosen) == 3:
            chosen.append(L)
            segs, prev = [], 0
            for pos in chosen:
                segs.append(text[prev:pos])
                prev = pos
            assert "".join(segs) == text
            return segs
    return plato_data.segment_passage(text, n_pts=4)


def _content_words(txt):
    return {w for w in plato_data._words(txt) if len(w) >= 5}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def check_story_deterministic(txt, fam, other_texts, guard_12g, tk):
    """Return (ok, reason) checking length, 4-beat segmentation, archaic/prefix ban, explicit disguise bans,
    eval 12-gram guard, and cross-retelling lexical distinctness."""
    txt = " ".join(txt.strip().split())
    if not txt:
        return False, "empty"
    if PREFIX_BAN.search(txt) or INLINE_BEAT_BAN.search(txt):
        return False, "formatting_prefix"
    ws = txt.split()
    if not (70 <= len(ws) <= 195):
        return False, f"word_count={len(ws)}"
    if txt[-1] not in ".!?\"'\u201d":
        return False, "incomplete_end"
    m_arc = ARCHAIC_BAN.search(txt)
    if m_arc:
        return False, f"archaic={m_arc.group(0)}"
    low = txt.lower()
    for term in fam.get("bans", ()):
        if re.search(rf"\b{re.escape(term.lower())}\b", low):
            return False, f"ban_leak={term}"
    if plato_data._grams12(txt) & guard_12g:
        return False, "eval_12gram_overlap"
    segs = segment_4beats(txt)
    if len(segs) != 4:
        return False, f"n_segs={len(segs)}"
    ids, pts = plato_data.boundary_token_indices(lambda s: tk.encode(s).ids, segs)
    if len(ids) > 360 or len(pts) != 4:
        return False, f"bpe_len={len(ids)}_pts={len(pts)}"
    cw = _content_words(txt)
    for ot in other_texts:
        jac = _jaccard(cw, _content_words(ot))
        if jac > 0.22:
            return False, f"overlap_jac={jac:.2f}"
    return True, "ok"


# ---------------------------------------------------------------- Process & Memory Guards (64 GB Unified RAM Discipline)

def _assert_no_llama_server():
    r = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True)
    assert r.returncode != 0, "An existing llama-server is running on GPU/RAM! Stop it before loading another model."


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _stop_proc(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _start_inkling_server(port=8182, n_slots=4, ctx=8192):
    _assert_no_llama_server()
    assert not _port_open(port), f"port {port} already in use!"
    cmd = [LLAMA_SRV, "-m", INKLING_GGUF]
    if Path(DSPARK_GGUF).exists():
        cmd += ["-md", DSPARK_GGUF, "--spec-type", "draft-dspark", "-ngld", "999"]
    cmd += [
        "-ngl", "999", "-c", str(ctx), "-np", str(n_slots),
        "--host", "127.0.0.1", "--port", str(port),
        "--reasoning-format", "deepseek", "--reasoning-effort", "low", "--reasoning-budget", "128",
        "--no-webui",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                raise RuntimeError(f"Inkling llama-server exited early rc={proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return proc
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Inkling llama-server failed to become healthy within 60 s")
    except BaseException:
        _stop_proc(proc)
        raise


def _inkling_chat(port, sys_prompt, user_prompt, temp=0.80, max_tokens=560, seed=1):
    body = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temp,
        "top_p": 0.92,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    txt = data["choices"][0]["message"].get("content") or ""
    # Strip closed or unclosed <think> blocks and bare ...</think> prefixes cleanly
    txt = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.S)
    txt = re.sub(r"^.*?</think>", "", txt, flags=re.S).strip()
    # Strip ONLY surrounding markdown fences (never delete the story inside!)
    txt = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", txt).strip()
    txt = " ".join(l.strip() for l in txt.splitlines() if l.strip())
    return txt


GEN_SYS = (
    "You are a concise realist fiction writer. Write a grounded, contemporary 4-sentence vignette "
    "(100 to 145 words) in plain everyday modern English that enacts the requested causal arc in a specific modern workplace. "
    "NEVER name any myth, fable, psychological term, economic term, or moral. NEVER use titles, numbered labels, bullet points, "
    "or archaic fantasy words (no kings, queens, gods, magic, castles, swords, legends). Output ONLY the single plain 4-sentence paragraph."
)


def _gen_one_story(port, fam, r_idx, setting, other_texts, guard_12g, tk, feedback="", max_tries=6, seed_offset=0):
    fb_line = f"\nFix this issue from a previous draft: {feedback}" if feedback else ""
    last_reason = "no_try"
    for att in range(max_tries):
        u_prompt = (
            f"Modern setting: {setting}.\n"
            f"Underlying causal arc to enact naturally through ordinary actions and consequences: {fam['pattern']}.\n"
            f"Write exactly four smooth, natural sentences in one single paragraph (105-140 words): "
            f"sentence 1 sets the ordinary baseline in {setting}; sentence 2 introduces the catalyst or temptation; "
            f"sentence 3 delivers the decisive turn or escalation; sentence 4 shows the concrete outcome.\n"
            f"Use ordinary modern names and concrete details from {setting}. Do NOT number the sentences or state the abstract principle.{fb_line}"
        )
        seed = 229000 + seed_offset + fam["fam_idx"] * 100 + r_idx * 10 + att
        txt = _inkling_chat(port, GEN_SYS, u_prompt, temp=0.76 + 0.05 * att, max_tokens=560, seed=seed)
        ok, reason = check_story_deterministic(txt, fam, other_texts, guard_12g, tk)
        if ok:
            segs = segment_4beats(txt)
            return {
                "id": f"m229_{fam['fam_idx']:03d}_r{r_idx}",
                "fam_idx": fam["fam_idx"],
                "fam_id": fam["fam_id"],
                "kind": fam["kind"],
                "pattern": fam["pattern"],
                "retelling": r_idx,
                "setting": setting,
                "text": txt,
                "segs": segs,
                "gen_attempts": att + 1,
            }
        last_reason = reason
        fb_line = f"\nPrevious draft failed check ({reason}). Write exactly 4 natural sentences (110-135 words), plain paragraph, no labels, no myth/jargon terms."
    raise RuntimeError(f"Failed clean story for {fam['fam_id']} r{r_idx}: last={last_reason}")


def _load_raw_dedup():
    done = {}
    if RAW_JSONL.exists():
        with open(RAW_JSONL) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    done[r["id"]] = r
    return done


def _save_raw_atomic(rows_by_id):
    tmp = RAW_JSONL.with_suffix(".tmp.jsonl")
    with open(tmp, "w") as wf:
        for k in sorted(rows_by_id):
            wf.write(json.dumps(rows_by_id[k]) + "\n")
    os.replace(tmp, RAW_JSONL)


def _family_sig(rows_sorted):
    return hashlib.sha1("|".join(r["text"] for r in rows_sorted).encode("utf-8")).hexdigest()


def _load_full_guard_12grams():
    g = plato_data._load_eval_guard_12grams()
    from fluent import arch_rot229
    for p in arch_rot229.load_arch_pairs():
        g |= plato_data._grams12(p["done"])
        g |= plato_data._grams12(p["neglected"])
    return g


def cmd_generate(n_workers=4, port=8182):
    t0 = time.time()
    tk = plato_data.load_bpe8k()
    guard_12g = _load_full_guard_12grams()
    fams = all_families()
    done = _load_raw_dedup()
    print(f"MYTH229 GENERATE: {len(done)}/{len(fams) * 5} already complete", flush=True)
    if len(done) == len(fams) * 5:
        print("MYTH229 GENERATE: all 1,000 stories already exist!", flush=True)
        return

    todo_fams = [f for f in fams if any(f"m229_{f['fam_idx']:03d}_r{r}" not in done for r in range(5))]
    # Snapshot per-family existing stories on main thread (zero concurrent dict mutation!)
    snap = {}
    for f in todo_fams:
        fi = f["fam_idx"]
        snap[fi] = {r: done[f"m229_{fi:03d}_r{r}"]["text"] for r in range(5) if f"m229_{fi:03d}_r{r}" in done}

    proc = _start_inkling_server(port=port, n_slots=n_workers)
    errors = []
    try:
        def work_family(fam):
            fi = fam["fam_idx"]
            sets = settings_for_family(fi)
            existing_map = dict(snap[fi])
            new_rows, fam_errs = [], []
            for r_idx in range(5):
                if r_idx in existing_map:
                    continue
                try:
                    row = _gen_one_story(port, fam, r_idx, sets[r_idx], list(existing_map.values()), guard_12g, tk)
                    existing_map[r_idx] = row["text"]
                    new_rows.append(row)
                except Exception as e:
                    fam_errs.append((f"m229_{fi:03d}_r{r_idx}", str(e)))
            return new_rows, fam_errs

        completed_fams = len(fams) - len(todo_fams)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(work_family, fam): fam for fam in todo_fams}
            with open(RAW_JSONL, "a") as out_f:
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        rows, fam_errs = fut.result()
                    except Exception as e:
                        errors.append((futs[fut]["fam_id"], str(e)))
                        continue
                    for r in rows:
                        done[r["id"]] = r
                        out_f.write(json.dumps(r) + "\n")
                    out_f.flush()
                    errors.extend(fam_errs)
                    completed_fams += 1
                    if completed_fams % 10 == 0 or completed_fams == len(fams):
                        print(f"  GENERATED families {completed_fams}/{len(fams)} stories={len(done)} errs={len(errors)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        _stop_proc(proc)
        if done:
            _save_raw_atomic(done)
    print(f"MYTH229_GEN_DONE stories={len(done)}/{len(fams) * 5} errs={len(errors)} in {time.time() - t0:.0f} s", flush=True)
    if errors:
        print("  Sample errors:", errors[:10], flush=True)


# ---------------------------------------------------------------- Step 2: Local Qwen3.8-27B-MLX-8bit verification (STRICTLY SEQUENTIAL!)

def cmd_verify(max_fams=None):
    _assert_no_llama_server()
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    assert RAW_JSONL.exists(), f"missing {RAW_JSONL}; run generate first!"
    done = _load_raw_dedup()
    raw_by_fam = defaultdict(list)
    for r in done.values():
        raw_by_fam[r["fam_idx"]].append(r)

    ver_map = {}
    if VER_JSONL.exists():
        with open(VER_JSONL) as f:
            for l in f:
                if l.strip():
                    v = json.loads(l)
                    ver_map[v["fam_idx"]] = v

    todo = []
    for fi in sorted(raw_by_fam):
        rows = sorted(raw_by_fam[fi], key=lambda r: r["retelling"])
        if len(rows) != 5 or [r["retelling"] for r in rows] != [0, 1, 2, 3, 4]:
            print(f"  WARN: fam {fi} incomplete ({len(rows)} stories), skipping verify until generate completes", flush=True)
            continue
        sig = _family_sig(rows)
        if fi not in ver_map or ver_map[fi].get("sig") != sig:
            todo.append((fi, rows, sig))
    if max_fams is not None:
        todo = todo[:max_fams]
    print(f"MYTH229 VERIFY: {len(todo)} families to check with Qwen3.8-27B-MLX-8bit", flush=True)
    if not todo:
        print("MYTH229 VERIFY: all families already verified!", flush=True)
        return

    def _save_ver():
        tmp = VER_JSONL.with_suffix(".tmp.jsonl")
        with open(tmp, "w") as wf:
            for k in sorted(ver_map):
                wf.write(json.dumps(ver_map[k]) + "\n")
        os.replace(tmp, VER_JSONL)

    mx.set_cache_limit(2 << 30)
    model = tok = None
    try:
        model, tok = load(QWEN27_MLX)
        sampler = make_sampler(temp=0.0)
        for idx, (fi, rows, sig) in enumerate(todo):
            pat = rows[0]["pattern"]
            fid = rows[0]["fam_id"]
            block = "\n".join(f"[{r['retelling']}] ({r['setting']}): {r['text']}" for r in rows)
            prompt_txt = (
                f"Audit these 5 modern vignettes against the underlying pattern.\n"
                f"Pattern ({fid}): {pat}\n\n"
                f"{block}\n\n"
                f"For each vignette 0..4, output 1 ONLY if it: (a) faithfully enacts the core causal pattern, "
                f"(b) stays fully disguised in modern everyday prose without naming the myth/concept, and "
                f"(c) feels distinct in setting and plot details from the other vignettes. Otherwise output 0.\n"
                f'Reply with ONLY valid JSON: {{"ok":[0 or 1, ... 5 ints], "fix":{{"idx":"short reason"}}}}'
            )
            msgs = [{"role": "user", "content": prompt_txt}]
            chat_s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            resp = generate(model, tok, prompt=chat_s, max_tokens=320, sampler=sampler, verbose=False)
            resp_c = re.sub(r"<think>.*?(?:</think>|$)", "", resp, flags=re.S)
            resp_c = re.sub(r"^.*?</think>", "", resp_c, flags=re.S).strip()
            m_js = re.search(r"\{.*\}", resp_c, flags=re.S)
            parsed = None
            if m_js:
                try:
                    parsed = json.loads(m_js.group(0))
                except Exception:
                    parsed = None
            if parsed and isinstance(parsed.get("ok"), list) and len(parsed["ok"]) == 5:
                ok_vec = [int(bool(x)) for x in parsed["ok"]]
                fix_map = parsed.get("fix", {}) if isinstance(parsed.get("fix"), dict) else {}
            else:
                ok_vec = [0] * 5
                fix_map = {"parse": resp_c[:160]}
            ver_map[fi] = {"fam_idx": fi, "fam_id": fid, "sig": sig, "ok": ok_vec, "fix": fix_map}
            if (idx + 1) % 10 == 0 or idx == len(todo) - 1:
                tot_pass = sum(sum(v["ok"]) for v in ver_map.values())
                tot_all = sum(len(v["ok"]) for v in ver_map.values())
                print(f"  VERIFIED {idx + 1}/{len(todo)} (total pass {tot_pass}/{tot_all} = {tot_pass / max(1, tot_all):.1%}, {time.time() - t0:.0f} s)", flush=True)
                _save_ver()
    finally:
        if ver_map:
            _save_ver()
        del model, tok
        gc.collect()
        mx.clear_cache()

    tot_pass = sum(sum(v["ok"]) for v in ver_map.values())
    tot_all = sum(len(v["ok"]) for v in ver_map.values())
    print(f"MYTH229_VERIFY_DONE pass={tot_pass}/{tot_all} ({tot_pass / max(1, tot_all):.1%}) in {time.time() - t0:.0f} s", flush=True)


def cmd_repair(n_workers=4, port=8182):
    """Regenerate ONLY Qwen3.8-flagged slots with Inkling-2-Mini (repair, don't drop!), checkpointing atomically."""
    t0 = time.time()
    assert RAW_JSONL.exists() and VER_JSONL.exists()
    tk = plato_data.load_bpe8k()
    guard_12g = _load_full_guard_12grams()
    fams = {f["fam_idx"]: f for f in all_families()}
    rows_by_id = _load_raw_dedup()

    to_fix_by_fam = defaultdict(list)
    with open(VER_JSONL) as f:
        for l in f:
            if l.strip():
                v = json.loads(l)
                fi = v["fam_idx"]
                for r_idx, ok_flag in enumerate(v["ok"]):
                    if not ok_flag:
                        note = str(v.get("fix", {}).get(str(r_idx), "make the causal arc unmistakable and strictly disguised"))
                        to_fix_by_fam[fi].append((r_idx, note))
    n_slots_fix = sum(len(v) for v in to_fix_by_fam.values())
    print(f"MYTH229 REPAIR: {n_slots_fix} failed slots across {len(to_fix_by_fam)} families", flush=True)
    if not to_fix_by_fam:
        return

    snap = {
        fi: {r: rows_by_id[f"m229_{fi:03d}_r{r}"] for r in range(5) if f"m229_{fi:03d}_r{r}" in rows_by_id}
        for fi in to_fix_by_fam
    }
    use_workers = min(n_workers, max(1, len(to_fix_by_fam)))
    proc = _start_inkling_server(port=port, n_slots=use_workers)
    fixed_cnt, fail_cnt = 0, 0
    try:
        def repair_fam(fi):
            fam = fams[fi]
            sets = settings_for_family(fi)
            cur = dict(snap[fi])
            updated, errs = [], []
            for r_idx, note in to_fix_by_fam[fi]:
                others = [cur[j]["text"] for j in range(5) if j != r_idx and j in cur]
                prev_round = int(cur[r_idx].get("repair_round", 0)) if r_idx in cur else 0
                seed_off = 1_000_000 * (prev_round + 1)
                try:
                    new_row = _gen_one_story(port, fam, r_idx, sets[r_idx], others, guard_12g, tk, feedback=note, max_tries=6, seed_offset=seed_off)
                    new_row["repair_round"] = prev_round + 1
                    cur[r_idx] = new_row
                    updated.append(new_row)
                except Exception as e:
                    errs.append((f"m229_{fi:03d}_r{r_idx}", str(e)))
            return updated, errs

        with concurrent.futures.ThreadPoolExecutor(max_workers=use_workers) as ex:
            futs = {ex.submit(repair_fam, fi): fi for fi in sorted(to_fix_by_fam)}
            done_fams = 0
            for fut in concurrent.futures.as_completed(futs):
                updated, errs = fut.result()
                for row in updated:
                    rows_by_id[row["id"]] = row
                fixed_cnt += len(updated)
                fail_cnt += len(errs)
                done_fams += 1
                if done_fams % 10 == 0 or done_fams == len(to_fix_by_fam):
                    _save_raw_atomic(rows_by_id)
                    print(f"  REPAIRED fams={done_fams}/{len(to_fix_by_fam)} fixed={fixed_cnt} failed={fail_cnt} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        _stop_proc(proc)
        _save_raw_atomic(rows_by_id)
    print(f"MYTH229_REPAIR_DONE fixed={fixed_cnt} failed={fail_cnt} in {time.time() - t0:.0f} s (run `verify` next to confirm!)", flush=True)


# ---------------------------------------------------------------- Step 4: 5-Family Teacher Extraction (1 model in RAM at a time!)

def _load_clean_stories():
    """Return sorted list of Qwen3.8-verified stories (validating family sig against current RAW_JSONL!)."""
    assert RAW_JSONL.exists(), f"missing {RAW_JSONL}"
    assert VER_JSONL.exists(), f"missing {VER_JSONL}; run `verify` before `extract` or `project`!"
    rows = _load_raw_dedup()
    by_fam_raw = defaultdict(list)
    for r in rows.values():
        by_fam_raw[r["fam_idx"]].append(r)

    ver_map = {}
    with open(VER_JSONL) as f:
        for l in f:
            if l.strip():
                v = json.loads(l)
                ver_map[v["fam_idx"]] = v

    kept = []
    stale = []
    for fi in sorted(by_fam_raw):
        f_rows = sorted(by_fam_raw[fi], key=lambda r: r["retelling"])
        if len(f_rows) != 5 or fi not in ver_map:
            continue
        if ver_map[fi].get("sig") != _family_sig(f_rows):
            stale.append(fi)
            continue
        passed = [r for r, flag in zip(f_rows, ver_map[fi]["ok"]) if flag]
        if len(passed) >= 3:
            kept.extend(passed)
    assert not stale, f"{len(stale)} families were modified after verification (e.g. {stale[:5]}); re-run `verify` first!"
    return kept


def _story_sha1s(stories):
    return np.array([hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:16] for r in stories])


def _extract_model_stories(key, stories):
    out_npz = PLATO / f"myth229_raw_{key}.npz"
    want_ids = np.array([r["id"] for r in stories])
    want_sha = _story_sha1s(stories)
    if out_npz.exists():
        with np.load(out_npz) as d:
            if "ids" in d.files and "text_sha1" in d.files and np.array_equal(d["ids"], want_ids) and np.array_equal(d["text_sha1"], want_sha):
                print(f"MYTH229 EXTRACT {key}: up-to-date ({out_npz}), skipping", flush=True)
                return
        print(f"MYTH229 EXTRACT {key}: stories changed since last extraction, re-extracting ...", flush=True)

    _assert_no_llama_server()
    spec = plato_data.MODEL_SPECS[key]
    t0 = time.time()

    if spec["backend"] == "llamacpp":
        port = spec["port"]
        assert not _port_open(port), f"port {port} in use!"
        cmd = [
            LLAMA_SRV, "-m", spec["path"], "-ngl", "999", "-c", "2048", "-b", "1024", "-ub", "1024",
            "--embeddings", "--pooling", "none", "--embd-normalize", "-1",
            "--host", "127.0.0.1", "--port", str(port), "--no-webui",
        ]
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
                    raise RuntimeError(f"{key} llama-server exited early rc={proc.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            ok = True
                            break
                except Exception:
                    time.sleep(0.5)
            assert ok, f"{key} llama-server failed to start"
            enc_fn = lambda s: post("/tokenize", {"content": s, "add_special": True})["tokens"]
            pts_list, story_list, cnts, oks = [], [], [], []
            D_dim = None
            for idx, r in enumerate(stories):
                full_ids, pt_idxs = plato_data.boundary_token_indices(enc_fn, r["segs"])
                good = (2 <= len(full_ids) <= 512 and len(pt_idxs) == len(r["segs"]) == 4)
                if not good:
                    if D_dim is None:
                        emb = post("/embedding", {"content": "hello world"})
                        arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
                        D_dim = int(arr.shape[1])
                    story_list.append(np.zeros(D_dim, dtype=np.float32))
                    cnts.append(0)
                    oks.append(False)
                    continue
                emb = post("/embedding", {"content": r["text"]})
                arr = np.asarray(emb[0]["embedding"] if isinstance(emb, list) else emb["embedding"], dtype=np.float32)
                D_dim = int(arr.shape[1])
                if arr.shape[0] != len(full_ids):
                    story_list.append(np.zeros(D_dim, dtype=np.float32))
                    cnts.append(0)
                    oks.append(False)
                    continue
                pts_list.append(arr[pt_idxs])
                story_list.append(arr[1:].mean(axis=0))
                cnts.append(len(pt_idxs))
                oks.append(True)
                if (idx + 1) % 250 == 0 or idx == len(stories) - 1:
                    print(f"  EXTRACT {key} {idx + 1}/{len(stories)} ok={sum(oks)} ({time.time() - t0:.0f} s)", flush=True)
        finally:
            _stop_proc(proc)
    else:
        import mlx.core as mx
        from mlx_lm import load

        mx.set_cache_limit(2 << 30)
        print(f"MYTH229 EXTRACT {key}: loading {spec['path']} ...", flush=True)
        model = tok = layers = tgt_blk = None
        patched = {}
        rec_buf = []
        pts_list, story_list, cnts, oks = [], [], [], []
        D_dim = None
        try:
            model, tok = load(spec["path"])
            layers = plato_data._find_layers(model)
            nl = len(layers)
            tgt_idx = max(1, min(nl - 2, int(round(spec["depth_frac"] * (nl - 1)))))
            tgt_blk = layers[tgt_idx]
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
            enc_fn = lambda s: plato_data._text_ids_mlx(tok, s)
            for idx, r in enumerate(stories):
                full_ids, pt_idxs = plato_data.boundary_token_indices(enc_fn, r["segs"])
                good = (2 <= len(full_ids) <= 512 and len(pt_idxs) == len(r["segs"]) == 4)
                if not good:
                    if D_dim is None:
                        rec_buf.clear()
                        _ = model(mx.array([full_ids[:4] or [0, 1]]))
                        mx.eval(rec_buf[0])
                        D_dim = int(rec_buf[0].shape[-1])
                    story_list.append(np.zeros(D_dim, dtype=np.float32))
                    cnts.append(0)
                    oks.append(False)
                    continue
                rec_buf.clear()
                _ = model(mx.array([full_ids]))
                assert len(rec_buf) == 1
                h = rec_buf[0].astype(mx.float32)
                D_dim = int(h.shape[-1])
                hp = h[mx.array(pt_idxs)]
                hm = h[1:].mean(axis=0)
                mx.eval(hp, hm)
                pts_list.append(np.array(hp, dtype=np.float32))
                story_list.append(np.array(hm, dtype=np.float32))
                cnts.append(len(pt_idxs))
                oks.append(True)
                if (idx + 1) % 250 == 0 or idx == len(stories) - 1:
                    print(f"  EXTRACT {key} {idx + 1}/{len(stories)} ok={sum(oks)} ({time.time() - t0:.0f} s)", flush=True)
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            rec_buf.clear()
            del layers, tgt_blk, model, tok
            gc.collect()
            mx.clear_cache()

    assert pts_list, f"{key}: zero valid stories extracted!"
    pts_arr = np.concatenate(pts_list, axis=0)
    story_arr = np.stack(story_list, axis=0)
    assert np.isfinite(pts_arr).all() and np.isfinite(story_arr).all()
    pack = lambda a: a.astype(np.float16) if np.abs(a).max() < 6e4 else a.astype(np.float32)
    plato_data._atomic_savez(
        out_npz,
        pts=pack(pts_arr),
        story=pack(story_arr),
        cnt=np.array(cnts, dtype=np.int32),
        ok=np.array(oks, dtype=bool),
        ids=want_ids,
        text_sha1=want_sha,
    )
    print(f"MYTH229_EXTRACT_DONE {key}: ok={sum(oks)}/{len(oks)} pts={pts_arr.shape} ({time.time() - t0:.0f} s)", flush=True)


def cmd_extract(keys=None):
    stories = _load_clean_stories()
    assert len(stories) >= 500, f"too few verified stories: {len(stories)}"
    use_keys = keys if keys else list(plato_data.ALL_KEYS)
    print(f"MYTH229 EXTRACT: {len(stories)} verified stories across {use_keys}", flush=True)
    for k in use_keys:
        _extract_model_stories(k, stories)


# ---------------------------------------------------------------- Step 5: Frozen C5 Projection + LORO Double-Denoised Prototypes

def _fit_frozen_c5_frame(d_plato=256):
    """Refit exact Addendum 228 5-family GPA frame on passages.jsonl tr_pts and anchor explicitly to targets_train.npz['consensus_256']."""
    tk = plato_data.load_bpe8k()
    bpe_enc = lambda s: tk.encode(s).ids
    with open(PLATO / "passages.jsonl") as f:
        train_rows = [json.loads(l) for l in f]

    raw = {}
    for k in plato_data.ALL_KEYS:
        with np.load(PLATO / f"raw_{k}.npz") as d:
            raw[k] = {fld: d[fld] for fld in d.files}

    tr_keep_masks = {k: [] for k in plato_data.ALL_KEYS}
    for r_i, r in enumerate(train_rows):
        n_seg = len(r["segs"])
        b_ids, b_pts = plato_data.boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == n_seg
            and all(bool(raw[k]["tr_ok"][r_i]) and int(raw[k]["tr_cnt"][r_i]) == n_seg for k in plato_data.ALL_KEYS)
        )
        for k in plato_data.ALL_KEYS:
            tr_keep_masks[k].extend([ok] * int(raw[k]["tr_cnt"][r_i]))

    pca_params = {}
    Y_tr = {}
    for k in plato_data.ALL_KEYS:
        m_tr = np.array(tr_keep_masks[k], dtype=bool)
        X_tr = raw[k]["tr_pts"][m_tr].astype(np.float64)
        mu = X_tr.mean(axis=0, keepdims=True)
        Xc = X_tr - mu
        vals, vecs = np.linalg.eigh((Xc.T @ Xc) / float(len(Xc)))
        idx = np.argsort(vals)[::-1][:d_plato]
        V = vecs[:, idx]
        Z_tr = Xc @ V
        rms = max(1e-12, float(np.sqrt(np.mean(Z_tr ** 2))))
        pca_params[k] = (mu, V, rms)
        Y_tr[k] = Z_tr / rms

    C5_tr, _, _, _, rots = plato_data.generalized_procrustes_consensus(
        [Y_tr[k] for k in plato_data.CONSENSUS_KEYS],
        [Y_tr[k][:2] for k in plato_data.CONSENSUS_KEYS],
        n_iters=10,
    )
    with np.load(PLATO / "targets_train.npz") as d_tgt:
        saved_c5 = d_tgt["consensus_256"].astype(np.float64)
    assert C5_tr.shape == saved_c5.shape, f"C5 shape mismatch {C5_tr.shape} vs {saved_c5.shape}"
    anchor_map = plato_data.fit_procrustes_map(C5_tr, saved_c5)
    C5_anchored = plato_data.apply_procrustes_map(C5_tr, anchor_map)
    max_err = float(np.max(np.abs(C5_anchored - saved_c5)))
    rms_err = float(np.sqrt(np.mean((C5_anchored - saved_c5) ** 2)))
    cos_err = float(np.sum(C5_anchored * saved_c5) / max(1e-12, np.linalg.norm(C5_anchored) * np.linalg.norm(saved_c5)))
    assert rms_err < 1e-3 and cos_err > 0.99999 and max_err < 1e-2, f"Frozen C5 anchor mismatch: max={max_err:.5f} rms={rms_err:.5f} cos={cos_err:.6f}"

    raw_avg = np.mean([plato_data.apply_procrustes_map(Y_tr[k], rots[i]) for i, k in enumerate(plato_data.CONSENSUS_KEYS)], axis=0)
    tr_mu = raw_avg.mean(axis=0, keepdims=True)
    tr_rms = max(1e-12, float(np.sqrt(np.mean((raw_avg - tr_mu) ** 2))))

    return {
        "pca": pca_params,
        "rots": dict(zip(plato_data.CONSENSUS_KEYS, rots)),
        "tr_mu": tr_mu,
        "tr_rms": tr_rms,
        "anchor_map": anchor_map,
        "anchor_max_err": max_err,
        "anchor_rms_err": rms_err,
        "anchor_cos": cos_err,
    }


def compute_loro_prototypes(Z_inst, fam_ids, n_beats=4):
    """Compute Leave-One-Retelling-Out (LORO) Double-Denoised Myth-Pattern Prototypes per beat j=0..n_beats-1.
    Z_inst: (N_stories, n_beats, D) in frozen C5 space.
    For each story i in family f, prototype(i, j) = mean_{m in f, m != i} Z_inst[m, j], rescaled per beat to match
    the exact RMS norm of Z_inst[:, j, :] so downstream cosine+MSE loss sees identical geometry scale!"""
    N, B, D = Z_inst.shape
    assert B == n_beats
    by_fam = defaultdict(list)
    for i, f in enumerate(fam_ids):
        by_fam[f].append(i)

    Z_loro = np.zeros_like(Z_inst, dtype=np.float64)
    Z_full = np.zeros_like(Z_inst, dtype=np.float64)
    for f, idxs in by_fam.items():
        assert len(idxs) >= 2, f"family {f} has only {len(idxs)} stories; need >= 2 for LORO"
        sub = Z_inst[idxs]
        tot = sub.sum(axis=0)
        mean_full = tot / float(len(idxs))
        for local_pos, gi in enumerate(idxs):
            Z_full[gi] = mean_full
            Z_loro[gi] = (tot - sub[local_pos]) / float(len(idxs) - 1)

    for j in range(B):
        inst_rms = float(np.sqrt(np.mean(Z_inst[:, j, :] ** 2)))
        loro_rms = max(1e-12, float(np.sqrt(np.mean(Z_loro[:, j, :] ** 2))))
        full_rms = max(1e-12, float(np.sqrt(np.mean(Z_full[:, j, :] ** 2))))
        Z_loro[:, j, :] *= (inst_rms / loro_rms)
        Z_full[:, j, :] *= (inst_rms / full_rms)
    return Z_loro, Z_full


def _eval_cross_setting_retrieval(Z_inst, fam_ids, center=True):
    """Centered (primary), tie-safe pre-training diagnostic: does the LORO average of the OTHER retellings of the same
    pattern (in completely different modern settings) retrieve the held-out retelling over all F families?
    Mean-centering per beat removes nonzero-mean k-1 vs k norm bias; tie-safe fractional ranking prevents collapse inflation."""
    N, B, D = Z_inst.shape
    fam_arr = np.asarray(fam_ids)
    uniq_fams = sorted(set(fam_ids))
    fam_to_col = {f: c for c, f in enumerate(uniq_fams)}
    F = len(uniq_fams)
    out = {}
    for j in range(B):
        a = Z_inst[:, j, :].astype(np.float64)
        if center:
            a = a - a.mean(axis=0, keepdims=True)
        # Recompute LORO & full prototypes cleanly in this centered (or raw) frame
        b_loro = np.zeros_like(a)
        full_protos = np.zeros((F, D), dtype=np.float64)
        for c, f in enumerate(uniq_fams):
            idxs = np.flatnonzero(fam_arr == f)
            sub = a[idxs]
            tot = sub.sum(axis=0)
            p_full = tot / float(len(idxs))
            nrm = np.linalg.norm(p_full)
            full_protos[c] = p_full / nrm if nrm > 1e-12 else 0.0
            for lp, gi in enumerate(idxs):
                p_lo = (tot - sub[lp]) / float(max(1, len(idxs) - 1))
                b_loro[gi] = p_lo
        an_n = np.linalg.norm(a, axis=1, keepdims=True)
        bn_n = np.linalg.norm(b_loro, axis=1, keepdims=True)
        an = np.where(an_n > 1e-12, a / np.maximum(1e-12, an_n), 0.0)
        bn = np.where(bn_n > 1e-12, b_loro / np.maximum(1e-12, bn_n), 0.0)

        sims = an @ full_protos.T  # (N, F)
        same_cos = np.sum(an * bn, axis=1)
        top1_credits, top5_credits, diff_cos_list = [], [], []
        for i in range(N):
            c_own = fam_to_col[fam_ids[i]]
            row = sims[i].copy()
            row[c_own] = same_cos[i]
            strictly_better = int(np.sum(row > same_cos[i] + 1e-12))
            ties = int(np.sum(np.abs(row - same_cos[i]) <= 1e-12))  # includes c_own itself (>= 1)
            # Exact expected Top-k credit under random tie-breaking
            if strictly_better == 0:
                top1_credits.append(1.0 / ties)
            else:
                top1_credits.append(0.0)
            if strictly_better < 5:
                slots_left = 5 - strictly_better
                top5_credits.append(min(1.0, slots_left / float(ties)))
            else:
                top5_credits.append(0.0)
            diff_cos_list.append(float((row.sum() - same_cos[i]) / max(1, F - 1)))
        out[f"beat_{j}"] = {
            "same_fam_cos": round(float(same_cos.mean()), 4),
            "diff_fam_cos": round(float(np.mean(diff_cos_list)), 4),
            "gap": round(float(same_cos.mean() - np.mean(diff_cos_list)), 4),
            "top1_over_fams": round(float(np.mean(top1_credits)), 4),
            "top5_over_fams": round(float(np.mean(top5_credits)), 4),
            "n_fams": F,
        }
    return out


def cmd_project():
    t0 = time.time()
    tk = plato_data.load_bpe8k()
    bpe_enc = lambda s: tk.encode(s).ids
    stories = _load_clean_stories()
    want_ids = np.array([r["id"] for r in stories])
    want_sha = _story_sha1s(stories)
    frame = _fit_frozen_c5_frame(d_plato=256)
    print(f"MYTH229 PROJECT: anchored frozen C5 frame (cos={frame['anchor_cos']:.6f}, rms_err={frame['anchor_rms_err']:.2e})", flush=True)

    raw_m = {}
    avail_keys = [k for k in plato_data.ALL_KEYS if (PLATO / f"myth229_raw_{k}.npz").exists()]
    for k in plato_data.CONSENSUS_KEYS:
        assert k in avail_keys, f"missing consensus extraction for {k}!"
    for k in avail_keys:
        with np.load(PLATO / f"myth229_raw_{k}.npz") as d:
            raw_m[k] = {fld: d[fld] for fld in d.files}
            assert np.array_equal(raw_m[k]["ids"], want_ids), f"{k} story IDs mismatch — re-run extract!"
            assert "text_sha1" in raw_m[k] and np.array_equal(raw_m[k]["text_sha1"], want_sha), f"{k} story hashes stale — re-run extract!"

    keep_prelim = []
    bpe_ids_all, bpe_pts_all = [], []
    pt_slices = {k: [] for k in avail_keys}
    cursor = {k: 0 for k in avail_keys}
    for i, r in enumerate(stories):
        b_ids, b_pts = plato_data.boundary_token_indices(bpe_enc, r["segs"])
        ok = (
            2 <= len(b_ids) <= 512 and len(b_pts) == 4
            and all(bool(raw_m[k]["ok"][i]) and int(raw_m[k]["cnt"][i]) == 4 for k in plato_data.CONSENSUS_KEYS)
        )
        for k in avail_keys:
            c_i = int(raw_m[k]["cnt"][i])
            pt_slices[k].append((cursor[k], cursor[k] + c_i))
            cursor[k] += c_i
        keep_prelim.append(ok)
        bpe_ids_all.append(b_ids)
        bpe_pts_all.append(b_pts)
    for k in avail_keys:
        assert cursor[k] == len(raw_m[k]["pts"]), f"{k} cursor mismatch {cursor[k]} vs {len(raw_m[k]['pts'])}"

    by_fam_ok = defaultdict(list)
    for i, (r, ok) in enumerate(zip(stories, keep_prelim)):
        if ok:
            by_fam_ok[r["fam_idx"]].append(i)
    keep_indices = []
    for fi in sorted(by_fam_ok):
        if len(by_fam_ok[fi]) >= 3:
            keep_indices.extend(by_fam_ok[fi])

    N = len(keep_indices)
    n_fams_kept = len([f for f in by_fam_ok if len(by_fam_ok[f]) >= 3])
    print(f"MYTH229 PROJECT: strictly aligned stories={N}/{len(stories)} across {n_fams_kept} families", flush=True)

    per_teacher_c5 = []
    for k in plato_data.CONSENSUS_KEYS:
        X_k = np.concatenate([raw_m[k]["pts"][pt_slices[k][i][0]:pt_slices[k][i][1]] for i in keep_indices], axis=0).astype(np.float64)
        mu, V, rms = frame["pca"][k]
        Y_k = ((X_k - mu) @ V) / rms
        aligned_k = plato_data.apply_procrustes_map(Y_k, frame["rots"][k])
        per_teacher_c5.append(aligned_k)
    C5_raw = np.mean(per_teacher_c5, axis=0)
    C5_std = (C5_raw - frame["tr_mu"]) / frame["tr_rms"]
    C5_inst_flat = plato_data.apply_procrustes_map(C5_std, frame["anchor_map"])
    Z_inst = C5_inst_flat.reshape(N, 4, 256)

    kept_stories = [stories[i] for i in keep_indices]
    fam_ids = [r["fam_id"] for r in kept_stories]
    fam_idxs = np.array([r["fam_idx"] for r in kept_stories], dtype=np.int32)
    kinds = np.array([r["kind"] for r in kept_stories])
    story_ids_k = np.array([r["id"] for r in kept_stories])
    settings_k = np.array([r["setting"] for r in kept_stories])

    Z_loro, Z_full = compute_loro_prototypes(Z_inst, fam_ids, n_beats=4)
    diag_c5_c = _eval_cross_setting_retrieval(Z_inst, fam_ids, center=True)
    diag_c5_r = _eval_cross_setting_retrieval(Z_inst, fam_ids, center=False)
    print(f"  Cross-setting LORO prototype retrieval inside frozen C5 (centered, chance ~ {100.0 / max(1, n_fams_kept):.2f}%):", flush=True)
    for b_k, m in diag_c5_c.items():
        print(f"    {b_k}: same_cos={m['same_fam_cos']:.4f} diff_cos={m['diff_fam_cos']:.4f} (gap={m['gap']:+.4f}) Top1={m['top1_over_fams']:.1%} Top5={m['top5_over_fams']:.1%}", flush=True)

    # Optional blind diagnostics (filters gracefully to valid subset with >= 3 per family!)
    blind_diag = {}
    for bk in plato_data.BLIND_KEYS:
        if bk in raw_m:
            by_fb = defaultdict(list)
            for i in keep_indices:
                if bool(raw_m[bk]["ok"][i]) and int(raw_m[bk]["cnt"][i]) == 4:
                    by_fb[stories[i]["fam_idx"]].append(i)
            bk_idx = [i for fi in sorted(by_fb) if len(by_fb[fi]) >= 3 for i in by_fb[fi]]
            if len(bk_idx) >= 30:
                X_b = np.concatenate([raw_m[bk]["pts"][pt_slices[bk][i][0]:pt_slices[bk][i][1]] for i in bk_idx], axis=0).astype(np.float64)
                mu_b, V_b, rms_b = frame["pca"][bk]
                Z_b = (((X_b - mu_b) @ V_b) / rms_b).reshape(len(bk_idx), 4, 256)
                bk_fids = [stories[i]["fam_id"] for i in bk_idx]
                blind_diag[bk] = _eval_cross_setting_retrieval(Z_b, bk_fids, center=True)
                b3 = blind_diag[bk]["beat_3"]
                print(f"  Blind {bk:9s} ({len(bk_idx)} stories) beat_3 centered LORO Top1={b3['top1_over_fams']:.1%} Top5={b3['top5_over_fams']:.1%} gap={b3['gap']:+.4f}", flush=True)

    seq_list = [bpe_ids_all[i] for i in keep_indices]
    pt_list = [bpe_pts_all[i] for i in keep_indices]
    ids_flat = np.concatenate([np.array(s, dtype=np.uint16) for s in seq_list])
    ids_off = np.zeros(N + 1, dtype=np.int64)
    ids_off[1:] = np.cumsum([len(s) for s in seq_list])
    pts_flat = np.concatenate([np.array(p, dtype=np.int32) for p in pt_list])
    pts_off = np.zeros(N + 1, dtype=np.int64)
    pts_off[1:] = np.cumsum([len(p) for p in pt_list])

    C5_inst_16 = Z_inst.reshape(N * 4, 256).astype(np.float16)
    C5_loro_16 = Z_loro.reshape(N * 4, 256).astype(np.float16)
    C5_full_16 = Z_full.reshape(N * 4, 256).astype(np.float16)
    assert len(pts_flat) == len(C5_inst_16)
    assert np.isfinite(C5_inst_16).all() and np.isfinite(C5_loro_16).all() and np.isfinite(C5_full_16).all(), "fp16 overflow!"

    plato_data._atomic_savez(
        OUT_NPZ,
        ids_flat=ids_flat, ids_off=ids_off,
        pts_flat=pts_flat, pts_off=pts_off,
        story_id=story_ids_k, setting=settings_k,
        fam_idx=fam_idxs, fam_id=np.array(fam_ids), kinds=kinds,
        c5_instance_256=C5_inst_16,
        c5_proto_loro_256=C5_loro_16,
        c5_proto_full_256=C5_full_16,
    )
    meta = {
        "n_stories": N,
        "n_families": n_fams_kept,
        "n_myth_stories": int(np.sum(kinds == "myth")),
        "n_mech_stories": int(np.sum(kinds == "mech")),
        "c5_anchor_cos": frame["anchor_cos"],
        "c5_anchor_rms_err": frame["anchor_rms_err"],
        "cross_setting_loro_c5_centered": diag_c5_c,
        "cross_setting_loro_c5_raw": diag_c5_r,
        "cross_setting_loro_blind_centered": blind_diag,
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp_m = META_JSON.with_suffix(".tmp.json")
    tmp_m.write_text(json.dumps(meta, indent=1))
    os.replace(tmp_m, META_JSON)
    print(f"MYTH229_PROJECT_DONE stories={N} fams={n_fams_kept} saved={OUT_NPZ} ({meta['elapsed_s']} s)", flush=True)


# ---------------------------------------------------------------- Unit tests

def self_test():
    import importlib
    fams = all_families()
    assert len(fams) == 200, f"expected 200 families, got {len(fams)}"
    assert sum(f["kind"] == "myth" for f in fams) == 100
    assert sum(f["kind"] == "mech" for f in fams) == 100
    ids = [f["fam_id"] for f in fams]
    assert len(set(ids)) == 200, "duplicate fam_id!"

    # Strict disjointness check against items3 (100+100) & items7 (80+80) using ABSOLUTE paths!
    prev_fd = os.environ.get("FRESH_DIR")
    try:
        os.environ.pop("FRESH_DIR", None)
        fresh = importlib.import_module("fresh")
        fresh = importlib.reload(fresh)
        assert len(fresh.A) == 100 and len(fresh.B) == 100, f"items3 load failed: {len(fresh.A)}/{len(fresh.B)}"
        forb = {x[0] for x in fresh.A} | {x[0] for x in fresh.B}
        os.environ["FRESH_DIR"] = str(ARCH / "items7")
        fresh = importlib.reload(fresh)
        assert len(fresh.A) == 80 and len(fresh.B) == 80, f"items7 load failed: {len(fresh.A)}/{len(fresh.B)}"
        forb |= {x[0] for x in fresh.A} | {x[0] for x in fresh.B}
    finally:
        if prev_fd is None:
            os.environ.pop("FRESH_DIR", None)
        else:
            os.environ["FRESH_DIR"] = prev_fd
        importlib.reload(fresh)
    overlap = set(ids) & forb
    assert not overlap, f"OVERLAP with items3/items7: {overlap}"

    # Verify every family's explicit bans have ZERO collision with its assigned 5 modern settings!
    for f in fams:
        assert len(f["bans"]) >= 1, f"{f['fam_id']} has empty bans!"
        ss = settings_for_family(f["fam_idx"])
        assert len(ss) == 5 and len(set(ss)) == 5
        for s in ss:
            sl = s.lower()
            for term in f["bans"]:
                assert not re.search(rf"\b{re.escape(term.lower())}\b", sl), f"ban {term!r} collides with setting {s!r} on {f['fam_id']}"

    # Verify deterministic filter & 4-sentence beat splitter
    tk = plato_data.load_bpe8k()
    fam0 = fams[1]  # midas
    bad_arc = (
        "The king rode past the castle with a magic sword while the village watched in quiet fear across the valley. "
        "Every guard on the stone wall waited for the royal signal as mist drifted over the empty road at dawn. "
        "Before noon the young herald arrived carrying a sealed scroll from the northern tower with urgent news. "
        "By evening the entire court gathered beside the iron gate to hear what had happened beyond the hills."
    )
    ok1, r1 = check_story_deterministic(bad_arc, fam0, [], set(), tk)
    assert not ok1 and "archaic" in r1, f"failed to block archaic: {ok1} {r1}"
    good_mod = (
        "Elena ran the overnight dispatch board at the transit depot and prided herself on trimming every minute off route turnarounds. "
        "When a new routing tablet arrived on Tuesday, the senior supervisor warned her not to override the automatic spacing buffer during rain. "
        "Energized by how fast her first three buses cleared downtown, Elena disabled the buffer completely to chase a record schedule across the whole east corridor. "
        "Within twenty minutes six buses bunched behind a stalled van at Fifth Avenue, freezing the entire line until dawn."
    )
    ok2, r2 = check_story_deterministic(good_mod, fam0, [], set(), tk)
    assert ok2, f"good modern story rejected: {r2}"
    segs4 = segment_4beats(good_mod)
    assert len(segs4) == 4 and all(s.strip().endswith(".") for s in segs4), f"4-sentence split broke mid-sentence: {segs4}"

    # Verify LORO prototype math + centered tie-safe retrieval on (a) synthetic signal, (b) nonzero-mean noise, (c) constant collapse!
    rng = np.random.default_rng(229)
    n_f, k_r, B, D = 40, 5, 4, 64
    proto_true = rng.normal(0, 1.0, (n_f, B, D))
    Z = np.zeros((n_f * k_r, B, D))
    fids = []
    for fi in range(n_f):
        for ri in range(k_r):
            Z[fi * k_r + ri] = proto_true[fi] + rng.normal(0, 1.2, (B, D))
            fids.append(f"f{fi}")
    Z_loro, _ = compute_loro_prototypes(Z, fids, n_beats=B)
    for j in range(B):
        assert abs(float(np.sqrt(np.mean(Z[:, j, :] ** 2))) - float(np.sqrt(np.mean(Z_loro[:, j, :] ** 2)))) < 1e-10
    diag = _eval_cross_setting_retrieval(Z, fids, center=True)
    assert diag["beat_3"]["top1_over_fams"] > 0.90, f"synthetic LORO failed: {diag}"

    # Nonzero-mean pure noise with mixed k=3 and k=5 must stay calibrated near chance (1/40 = 0.025) and gap ~ 0!
    fids_mix = [f"f{fi}" for fi in range(n_f) for _ in range(3 if fi < 20 else 5)]
    mu_off = rng.normal(0, 0.8, (1, B, D))
    Z_noise = mu_off + rng.normal(0, 1.0, (len(fids_mix), B, D))
    diag_n = _eval_cross_setting_retrieval(Z_noise, fids_mix, center=True)
    assert abs(diag_n["beat_3"]["gap"]) < 0.03 and abs(diag_n["beat_3"]["top1_over_fams"] - 1.0 / n_f) < 0.03, f"nonzero-mean noise biased: {diag_n}"

    # Constant collapsed representation must score exact chance (1/40 = 0.025), NEVER 1.0!
    Z_const = np.ones((n_f * k_r, B, D))
    diag_c = _eval_cross_setting_retrieval(Z_const, fids, center=True)
    assert abs(diag_c["beat_3"]["top1_over_fams"] - 1.0 / n_f) < 1e-6, f"collapse tie bug: {diag_c}"

    print(
        f"MYTH229_TEST_OK fams=200 overlap_i3_i7=0 bans_clean=OK "
        f"synth_top1={diag['beat_3']['top1_over_fams']:.3f} noise_top1={diag_n['beat_3']['top1_over_fams']:.3f} "
        f"collapse_top1={diag_c['beat_3']['top1_over_fams']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "generate":
        cmd_generate()
    elif cmd == "verify":
        mf = int(sys.argv[2]) if len(sys.argv) > 2 else None
        cmd_verify(max_fams=mf)
    elif cmd == "repair":
        cmd_repair()
    elif cmd == "extract":
        cmd_extract(sys.argv[2:] if len(sys.argv) > 2 else None)
    elif cmd == "project":
        cmd_project()
    else:
        raise ValueError(f"unknown cmd {cmd}")
