# Review of items3/w3.py (by writer 2)

I checked all 50 items (25 myths, 25 mechanisms) against BRIEF.md: does the true match keep every causal step, is the foil a minimal edit that really breaks the key step (and does brk say how), is the lure a different pattern, is the surface low-drama, is there a giveaway, would a careful reader pick the true match, and were both sets made with equal care.

4 items changed, all myths. The mechanism set needed no changes: every true match keeps its causal chain in a new domain, and every foil breaks the named step.

## Changes

| item | field | problem | change |
|---|---|---|---|
| belshazzar | true | The mayor's fall came from a rigged vote count, which has nothing to do with his contempt (the medals used as coasters). True and foil differed only in the coasters, and in both the fall had the same unrelated cause, so the pattern's key link (contempt brings the judgment) was missing from the true match. | The screen's news alert now reports that a guest's photo of the medal coasters is spreading across the country, and the council removes him before dawn. The foil is unchanged (medals honored; fall from the rigged vote), so it now breaks the cause as its brk label says. Word similarity 0.64. |
| jacob_esau | lure | Nearly a mirror image of the pattern: an elder sibling wrongs the younger, the younger goes into exile and comes back years later for revenge. The brief rules out "the pattern's mirror image". | New lure on the same surface with a different pattern: in a drought Esau finds a hidden spring, leads the family there, is made head of the clan, and Jacob gladly serves under him. |
| ekalavya | lure | It shared the pattern's first half (refused by Drona, masters archery alone, outshines the princes) and ended in reward. It was also the only option with a happy ending besides the foil, which left the true match as the only sad ending. | New lure: Ekalavya guides Drona's hunt, saves two princes from a flash flood and drowns trying to save the third; Drona carves his name on the palace gate. The pattern is different (self-sacrifice), and now the true match is no longer the only sad ending. A few words were added to keep the 1.3x length rule (lengths now 67/60/71). |
| town_country_mouse | true, foil | "A nurse is drawn to a fashion company" is unclear. | Changed to "drawn to a job at a fashion company" in both, so they still match. |
| town_country_mouse | lure | After the edit above, the true match stopped being the shortest option, and set A fell to the minimum of 5/25 items where true is shortest. | Added "in the city" (3 words). Set A is back to 6/25. |

## Checked and left as is

- **Foil labels:** a few could be argued either way (money_supply's foil adds candy as well as tokens; keystone's foil is rescued by a spam filter). Each one still clearly breaks the pattern, so I kept the writer's labels.
- **WARNs:** `goldilocks` in the lure and surface is the anchor's own name, which the brief requires there.
- **Length cue:** in both sets the foil is the longest option in 11 of 25 items. This does not point to the true match (a longest-option picker would choose the foil), so I did not rewrite it. It could be evened out in a later pass.
- **Endings and giveaways:** I checked every item for an ending only the true match has, for idioms, and for morals. Apart from ekalavya, none found.

## Checker after edits

`items: A 25, B 25; FAIL 0, WARN 2`

## Self-review (lures)

By writer 3, after cross-reviewing w0. Many mechanism lures in this file were plain accidents (a flood, a cut hand, a
broken card reader, an oil spill, a snapped bar, a power surge, a gas fire, a burst pipe) with no mechanism of their
own, while the myth lures follow real alternative patterns. Each one now follows a different, real mechanism on the
anchor's surface. None is the item's own mechanism or its mirror image, and the true match is never the only option
with its kind of ending. Three true matches were in the same domain as their anchor and were moved, with their
foils. Ids, pattern lines and brk labels are unchanged.

16 items changed (all in set B).

| item | field | what was wrong | what changed |
|---|---|---|---|
| principal_agent | lure | Water main floods the house; no permit, so no insurance: an accident. | Winner's curse: she takes the lowest of five bids, the builder is low because he underestimated the job, goes bankrupt halfway, and a second builder costs twice as much. |
| deadlock | lure | A cook cuts her hand: an accident. | Cross-contamination: raw chicken and salad on the one unwashed board, twenty diners ill, restaurant closed for a month. |
| head_of_line | lure | The card reader breaks: an accident. | Loss leader: milk sold below cost brings in customers who buy other things, so profits rise. |
| anticommons | lure | Toxic waste found under the lot: a discovery, no mechanism. | Negative externality: the heirs sell to a nightclub, the noise drives neighbors away, and the heirs' own homes next door lose value. |
| keystone | lure | Oil spill: an accident. | Biomagnification: a trace chemical in each clam builds up in otters that eat thousands, and they stop breeding. |
| antibiotic_resistance | true | Weeds and weedkiller are the same domain as bacteria and antibiotics (a chemical kills organisms, resistant ones survive). | Spam filter: a few spammers write in a style it misses, they get all the clicks, others copy them, and the filter stops almost nothing. |
| antibiotic_resistance | foil | Rewritten to match the new true. | Rules change monthly, so missed spammers are caught next month; junk still comes from new spammers (cause kept). Similarity 0.68. |
| antibiotic_resistance | lure | Allergic reaction: a plain bad event. | Drug interaction: the antibiotic slows the liver's breakdown of her blood thinner, which builds up until she bleeds. |
| lever | lure | The bar snaps and the rock rolls onto his foot: an accident. | Frost wedging: water poured into a crack freezes, expands and splits the rock. |
| doppler | true | Race-car engines heard from the trackside are the same setting as an ambulance siren heard at a crossing. | Ferry in a swell: heading into it, it meets a crest every second (a buoy, every three); heading home with it, every ten seconds. |
| doppler | foil | Rewritten to match the new true. | The ferry is tied to the buoy with its engines off; the crest rate changes only because a gale comes and goes (cause kept). Similarity 0.66. |
| doppler | lure | A driver with loud music does not hear the siren: an accident. | Echoes: among tall buildings the siren bounces off every wall, nobody can tell where it is coming from, and a driver pulls out in front of it. |
| ratchet | lure | Overtightened bolt snaps: an accident. | Galvanic corrosion: road salt between a steel bolt and an aluminum block eats the aluminum and seizes the bolt. |
| backoff | true | Two people phoning each other is the same domain (colliding messages on one line) as computers on one cable. | Two neighbors sharing one driveway back out at the same time; after each clash, each waits a die roll of minutes, then two dice, then three, and one gets out. |
| backoff | foil | Rewritten to match the new true. | Fixed five-minute waits: they clash every time, until one takes the bus (cause kept). |
| backoff | lure | Power surge burns out the network cards: an accident. | Switching loop: a spare cable plugged in at both ends makes every message circle and double until the network freezes; a loop-blocking switch fixes it. |
| parkinsons_law | lure | She slips on ice: an accident. | Reciprocity: her postcard draws a letter, then a parcel, then a visit, until the estranged niece moves in next door. |
| bystander | lure | Crowd pushes a woman onto the tracks: an accident, and it opened with people rushing to help (close to the mirror of the pattern). | Knock-on delays: the train is held ten minutes for the medics, and since trains run close together, the delay spreads down the whole line. Who helped is not mentioned. |
| batching | lure | Gas leak and fire: an accident. | Word of mouth: each happy customer tells friends, orders double monthly, and she must hire help and rent a second oven. |
| feedback_delay | lure | A pipe bursts: an accident. | Shared supply: a toilet flushed next door drops the cold-water pressure, the mix turns hot and he is scalded at once (no delay, no overcorrection). |
| money_multiplier | lure | A clerk empties the vault: a crime, no mechanism. | Inflation: her 2 percent interest is far below price rises, so her savings buy half as much. |
| tolerance_stacking | lure | Client refuses to pay; legal fees: no mechanism. | Wood movement: drawers fitted in a dry winter workshop swell in the damp seaside house and all stick. |

Kept, since each already follows a real mechanism: bank_run (loans concentrated in one factory), money_supply
(easily forged notes lose shops' trust), competitive_exclusion (predator-prey cycle), hedonic_treadmill (kin
demands drain a windfall), regulatory_capture (strict air rules push travel onto deadlier roads), large_numbers
(gambler's fallacy and doubling bets), metal_fatigue (one overload bends the clip), dutch_disease (a border dispute
drives investors away), bubble (insurers withdraw cover after a flood, so low-lying homes cannot be sold).

Checker after these edits: `items: A 25, B 25; FAIL 0, WARN 2` (both WARNs: 'goldilocks' in the lure and surface,
the anchor's own name). Set B: true strictly longest in 4/25, strictly shortest in 10/25; tf-idf picks the lure in
25/25 and never prefers true over surface.
