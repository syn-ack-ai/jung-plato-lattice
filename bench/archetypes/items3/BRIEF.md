# Brief for item writers (fresh item set, README addendum 9)

These items test whether language models see the same underlying pattern in two stories that share no names, setting
or objects. Each myth item has a mechanism counterpart, written in the same format and to the same standard. The test
compares how models from different companies represent myths with how they represent mechanisms. So the myth set and
the mechanism set must be equally well made. Do not make one set easier, more vivid or more stereotyped than the other.

Read these first, for format and style: `../items.py` (sets A and B: anchor, true, surface) and `../items_v2.py`
(lure, foil). Your items must be new: do not copy or paraphrase those, and use only the patterns you are assigned.

## Format

Write the Python file `items3/wK.py` (K = your writer number), defining two lists of 8-tuples of strings:

```python
A = [  # myths
    ("id", "pattern", "anchor", "true", "surface", "lure", "foil", "brk"),
    ...
]
B = [  # mechanisms, same format
    ...
]
```

Use the ids exactly as assigned. Use double-quoted strings. Inside a string, use single quotes or escape double quotes.

## The eight fields

1. **id**: as assigned.
2. **pattern**: one line, the standard reading of the story's underlying causal pattern (as in items.py), in general
   terms (no names). For a mechanism, its causal structure in plain words.
3. **anchor**: the source story told plainly in 2-3 sentences, 20-75 words, present tense.
   - Myths: with its own names (Joseph, Delilah, Anansi...).
   - Mechanisms: one concrete, standard example (as in items.py, e.g. a city widens a highway). Do NOT name the
     mechanism or use its technical term.
4. **true**: a different story with the same underlying pattern. 30-75 words, present tense.
   - Myths: an ordinary modern setting (work, family, school, sport, money, medicine, tech, neighbors...). Use none of
     the anchor's names, distinctive objects or setting. Nothing magical, no gods, no talking animals.
   - Mechanisms: a different domain from the anchor (if the anchor is traffic, the true might be email storage).
   - Every causal step of the pattern must be there, in order, with its cause and consequence.
5. **surface**: the anchor's own names, objects and setting, with an ordinary, low-drama course of events in which the
   pattern does NOT occur. 20-70 words.
6. **lure**: the anchor's own names, objects and setting, with an equally dramatic course of events following a
   DIFFERENT pattern. Not the same pattern with a twist, and not just the pattern's mirror image. 30-75 words.
7. **foil**: a minimal edit of **true**. Keep the same people, setting and most of the wording: at least about two
   thirds of true's words, in the same order. Change only the key causal step, so the pattern no longer holds.
   - The foil must stay eventful and plausible. Do not just append "but nothing happens".
   - The foil must not follow the anchor's pattern in some other way.
8. **brk**: which way the foil breaks the pattern:
   - `"cause"`: the key choice, act or condition is removed or replaced. Example: he never turns around, and loses
     her anyway for an unrelated reason.
   - `"link"`: the key act still happens, but it does not lead to the pattern's consequence, or the consequence comes
     from something else. Example: he turns around and she is fine.
   - Split your items about half and half between the two, within each set: 10-15 of each in 25.

## Rules that make the test fair (an automated checker enforces most of them)

- **Length**: in each item, true, lure and foil each have 30-75 words, and the longest of the three is at most 1.3x
  the shortest. Across your 25 items in each set, true must be the strictly longest of the three in at most 10 items
  and the strictly shortest in at least 5. Vary which option is longest. Keep myth and mechanism options about
  equally long on average (aim for 45-60 words each).
- **No giveaways**: true, foil, lure and surface must not name or allude to the source story or mechanism. No:
  - names from the anchor in true or foil;
  - idioms that name a story (Trojan horse, Achilles heel, sour grapes, cry wolf, Midas touch, Goldilocks,
    good Samaritan, prodigal, David and Goliath, writing on the wall, golden goose, Pandora's box...);
  - technical terms that name a mechanism (bottleneck, feedback, placebo, leverage, herd immunity, keystone, tipping
    point, snowball, vicious circle, "X's law", "X effect", "X paradox");
  - morals or explanations ("she learned that...", "the lesson is...", "ironically...").
  - Tell what happens and why, plainly, as the examples do.
- **Do not let the true match share rare words with the anchor.** If the anchor has a vineyard, the true match should
  not mention grapes or vines.
- **Each lure and surface reuses the anchor's names and setting**, so a model matching on surface picks them.
- **The true match must not be the only option with a happy ending, a sad ending or a twist.** Endings should not give
  the answer away.
- **Plain English**: short sentences, common words, as in the examples. Present tense throughout.

## Check your work

Run `python3 check_items3.py items3/wK.py` from the `bench/archetypes` directory and fix every FAIL. Then read
every WARN and fix it unless it is clearly fine. Run it again until there are no FAILs.

Then reread every item once more and ask:
- Would a careful reader, comparing what happens and why, pick the true match over the lure and the foil?
- Does the foil truly lack the pattern?
- Is the lure a different pattern, not the same one?

## Report

In your final message, give the checker's last summary line and any WARN you left in place, with the reason. Do not
paste the items.
