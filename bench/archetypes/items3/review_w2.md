# Review of w2.py (writer 1, 2026-11-01)

Checked all 50 items (25 myths, 25 mechanisms) against BRIEF.md: true match keeps every causal step, foil is a
minimal edit that breaks the key step with the right brk label, lure is a different pattern on the anchor's surface,
surface is low-drama with no pattern, no giveaways, equal care across the two sets. The file already passed the
checker before the review. Ids and assigned patterns are unchanged. 8 items changed (2 myths, 6 mechanisms).

## Changes

| item | field | problem | change |
|---|---|---|---|
| A:damon_pythias | lure | Mirror image of the pattern: one friend betrays the other instead of standing by him. The brief rules out "just the pattern's mirror image". | New lure with a different pattern: the two friends mock the tyrant's poems, are sent to the quarries, and Pythias's blunt joke ("Take us back to the quarries") makes the tyrant laugh and free them. |
| A:gelert | true | Unclear wording: "fighting off a thief who grabs the cash, and locking the money in the back room" leaves open who has the money. The rescue is the key revelation, so it must be plain. | "fighting off a thief who had grabbed the cash, getting it back and locking it safely in the back room". |
| B:invasive_species | true, foil | A causal step was missing: the pattern needs a newcomer that was held in check *at home* and lacks those checks in the new place. The true match only said the new city had no rivals or rules. | Added "kept small at home by strict rules and strong rivals, launches in a big city that has neither". The same edit went into the foil, so the foil stays a minimal edit and still breaks the step where it multiplies. |
| B:critical_path | foil, brk | The foil still followed the pattern: rushing the biopsy did not help because the surgeon's booking was now the longest wait. That is the same longest-chain logic with a different chain, and rushing the blood tests still changed nothing. | The foil now puts the blood tests in the chain ("must be done before the scan can start"), so every speed-up moves surgery and no step lies off the chain. brk changed from link to cause, since the key condition (a parallel step off the chain) is now removed. |
| B:interference | lure | A generic accident (a speaker catches fire, crush at the gates) with no mechanism of its own, one of four fire lures in the B set. Myth lures are coherent alternative patterns, so this was a care gap between the sets. | New lure with a different mechanism on the same surface: a microphone picks up the speaker's sound and sends it back out louder, again and again, until it howls (a self-amplifying loop, not two waves adding). |
| B:streisand | lure | Generic accident (a drone crashes and burns the house), the same fire motif again. | New lure with a different mechanism: the singer posts aerial photos of her house herself while on tour, and burglars use them to plan a break-in (published information used by an adversary). |
| B:creeping_normality | lure | A sudden dam burst is close to the mirror image of gradual change, and is another generic disaster. | New lure with a different mechanism: high flood walls make people feel safe, so they build right up to them, and the record flood that tops the walls does more damage than all the old floods (protection raises exposure). |
| B:braess | lure | Small inconsistency: "A town opens a new road ... During the work a digger cuts a water main". | "opens" changed to "builds". |

## Checked and left unchanged

- A:penelope foil: it removes the key stalling step (undoing each day's work) but keeps the recognition by a shared
  secret, so half the pattern is still there. I kept it because the true match has both steps and the distinctive one
  is gone. Worth a second look if foils with partial patterns are a concern.
- A:erysichthon true: the gambling craving follows the destruction of the forest in time and in place (his own
  casino), not by a supernatural cause. The modern analog is acceptable.
- Mechanism lures that are accidents but still have structure (braess water main, hysteresis compass, externality
  bridge, schelling subway search) were kept. After the changes no generic fire lure remains in set B.
- brk split after the review: A cause 13 / link 12, B cause 12 / link 13.

Checker after the review: `items: A 25, B 25; FAIL 0, WARN 4` (the 4 WARNs are the anchor's own names Goliath and
Damocles in their lure and surface, which the brief requires).
