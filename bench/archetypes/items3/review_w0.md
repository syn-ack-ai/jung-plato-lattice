# Cross-review of items3/w0.py (by writer 3)

Checked all 50 items against BRIEF.md: true keeps every causal step, foil is a minimal edit that breaks the key step,
lure is a different pattern on the anchor's surface, surface is low-drama, no giveaways (names, idioms, morals,
rare words shared with the anchor, an ending only the true match has), a careful reader picks true.

Result: 19 items changed (3 myths, 16 mechanisms), 25 fields. Ids, assigned patterns, pattern lines and brk labels
kept. Checker after edits: `items: A 25, B 25; FAIL 0, WARN 2` (the two WARNs are achilles_wrath's lure and surface
using the anchor's own name Achilles, as the brief requires).

## Main problems found

1. Mechanism lures that were generic accidents with no mechanism of their own (fire, shock, flood, a stray sock, a
   scandal), while the myth lures are real alternative patterns. Replaced with a real, different mechanism on the
   anchor's surface: leverage_finance, venturi, fuse, pipelining, hormesis, race_condition.
2. Mechanism true matches in the same domain as the anchor, so the match was nearly literal (the myth trues are
   fully re-surfaced): trophic_cascade (wolves/elk -> otters/urchins, both ecology; the otter story is also the
   anchor of keystone in w3), eutrophication (lake algae -> fish-tank algae, the same biology). Moved to other
   domains, all steps kept.
3. Endings where the true match was the only option with a good outcome: mimicry, learning_curve,
   price_discrimination, backpressure, regression_to_mean, tit_for_tat. Lure endings changed (the lure's own
   pattern kept or replaced as noted).
4. One myth lure that was the same pattern with roles swapped (north_wind_sun), one foil that kept most of the
   pattern (gyges).
5. True matches sharing rare words with the anchor: amaterasu, base_rate, hormesis, peter_principle.

## Change log

| item | field | what was wrong | what changed |
|---|---|---|---|
| gyges | foil | Foil kept the crime enabled by acting unseen and only added getting caught, so most of the pattern (impunity leads to crime) was still there. | He reports the account and it is closed; the boss's later firing for fraud and his promotion come by another route (link kept). |
| north_wind_sun | lure | Sun's harsh heat loses and the Wind is honored: the anchor's pattern (harshness loses, the other wins) with roles swapped. | New pattern: while two rivals fight each other, a third (the Clouds) takes the prize. |
| amaterasu | true | Shared rare anchor words 'gather' and 'loudly'. | 'crowd the street under her window ... sing along at full voice'. |
| amaterasu | foil | Similarity to the edited true fell to 0.59. | Added 'her old songs' (minimal edit restored, 0.60+). |
| regression_to_mean | lure | Only the true had a good ending (true: pain eases; foil: worse; lure: runway crash). | Lure ends with the trainee unhurt and the school adding crosswind practice; its mechanism (praise makes trainees avoid hard practice) is unchanged. |
| race_condition | lure | Overwork, then a typo: a generic mistake rather than a clear mechanism. | Cheque kiting: the bank pays cash before cheques clear, and a customer exploits the clearing delay for a year. |
| backpressure | lure | Only the true had a good ending. | Strike spreads to suppliers that depend on the factory (same mechanism), then a deal raises pay across the region. |
| leverage_finance | lure | Flood not covered by insurance: a generic accident. | Hold-up: after buying, she finds the only road crosses a neighbor's land, and he keeps raising the fee because she cannot leave. |
| trophic_cascade | true | Otters/urchins/kelp is the same domain (ecology) as wolves/elk/willows; also duplicates w3's keystone anchor. | Town lays off parking wardens -> all-day parkers multiply and fill the short-stay spaces -> shops that lived on quick visits close. |
| trophic_cascade | foil | Rewritten to match the new true. | Wardens kept; a bypass takes the traffic away and the shops close anyway (cause kept). |
| eutrophication | true | Fish-tank algae bloom is the same biology as the lake anchor. | Cheap credit feeds a boom of startups; when they fail, write-offs use up the banks' spare money, lending stops, and old firms close. |
| eutrophication | foil | Rewritten to match the new true. | Startups fail, but the government covers the banks' losses, lending goes on, and the old firms carry on (link kept). |
| mimicry | lure | Only the true had a good ending. | Same mechanism (spraying kills the pollinators), then next spring he rents beehives and has his best crop. |
| venturi | lure | Fuel drips on the exhaust and the car burns: a generic fire. | Vapor lock: on a hot climb the fuel boils in the line, the pump cannot move vapor, and the engine stalls until it cools. |
| fuse | lure | Miswired switch gives a shock: generic human error. The true was also the only good ending. | A loose socket screw makes a tiny hot contact that heats day by day and chars the plastic; the smell wakes the family in time. |
| peter_principle | true | Shared 'badly' with the anchor. | 'chairs it poorly'. |
| learning_curve | lure | Only the true had a good ending. | Same mechanism (one design error copied into every plane), then the fixed type flies for decades safely. |
| tit_for_tat | lure | Ruinous lawsuit; only the true had a good ending. | Bargaining: the downstream farmer, whose orchard earns more per drop, buys the water weekly and both end richer. |
| pipelining | lure | Red sock dyes the next loads: a generic mishap. | Broken windows: one broken washer left for months leads to vandalism, litter and lost customers, until a new owner fixes everything. |
| base_rate | true | Shared rare anchor words 'catches' and 'wrongly'. | 'stops ... blocks only one honest purchase in a hundred by mistake'. |
| base_rate | foil | Same wording as true's first sentence. | Same change as true. |
| price_discrimination | lure | Only the true had a good ending. | Same mechanism (overbooking), but over the year the extra seats earn more than the bad days cost. |
| hormesis | true | Shared rare anchor words 'moderate', 'far', 'too'. | 'steady, gentle winds', 'a gale much stronger than anything', 'bends it over until its trunk splits'. |
| hormesis | foil | Same words as true. | Same change as true. |
| hormesis | lure | Banned drug in a powder, team suspended: a scandal with no mechanism. | Risk compensation: a support belt makes a lifter feel safe, he skips warm-ups and rushes, and pulls a hamstring the belt does not protect. |

## Checked and left as they are

- Myths: all other true matches keep every step on a new modern surface; foils break the key step; lures are real
  alternative patterns (e.g. joseph's Potiphar false accusation, augean's cheated worker's revenge, boy_wolf's lie
  that scapegoats a stranger).
- Mechanism lures kept: positive_feedback (single shared cable), arms_race (closed border makes smugglers rich),
  winners_curse (pumping too fast ruins the reservoir), trophic_cascade (disease spillover from dogs), eutrophication
  (biomagnification), memory_leak (shared certificate expiry), peter_principle (key-person dependency),
  information_cascade (gentrification), base_rate (crowded testing lines spread flu), evaporative_cooling and inertia
  (thermal expansion), crowding_out (printing money), greshams_law (forgery leads to barter).
- Note: in set B the true match is the strictly shortest option in 10 of 25 items (allowed, but a shortest-option
  picker would score 40% against 33% chance). Not changed.
