# Review of items7/w3.py (2026-11-01)

Blind pick: a script printed each anchor with true, lure and foil in a seeded random order, labeled X/Y/Z. The key went to a separate file, and I read it only after I had written down all 40 picks. I had not read the pattern lines. I had read the hints in assign7.py first, as the task says, so the pick was blind to the labels but not to the hints. I picked true in 40/40 items, so there were 0 blind-pick failures before the fixes. The fixes below come from the novelty, named-mechanism and ending checks.
Checker after the fixes: `python3 check_items3.py --dir items7 items7/w3.py` gives FAIL 0, WARN 0 (true strictly shortest in 8/20 for A and 7/20 for B; strictly longest in 3/20 and 4/20).
Counts: OK 32, FIXED 8, REWRITTEN 0. brk split is unchanged: A 9 cause / 11 link, B 10 cause / 10 link.

## A (myths)

- unroped_guide: OK. The trader will not let the juniors check her (the scorned slow ones), they check only each other, and no one catches her error. In the foil, she is checked and an exchange fault causes the loss (cause). The anchor keeps the arc-a dynamic: she would not be bound to them, so they cannot hold her.
- milestone_courier: OK. Road rage at slow cars leads to a sentence to stand still and count every passing car. In the foil, he is calm and a knee operation puts him there (cause). The lure (kindness repaid by the carter's son) is a generic gratitude motif, not the lion-and-mouse turn.
- lie_spring: OK. He reads the results against advice, knows the real odds, and kind words no longer comfort him. In the foil, the odds are good, so the comfort is true (link).
- taxed_laughter: OK. A beach fee to fill the budget empties the beach, and her own son never learns to swim. In the foil, the beach stays free and algae cause the loss (cause).
- lie_jar: FIXED (lure). The old lure (a child's warning is ignored, the disaster comes, then she is believed) was the Cassandra / unheeded-warning plot, and cassandra is in the famous set. The new lure is a different pattern on the same surface: Mett carries everyone's water, they stop walking to the spring, and when she leaves they are helpless.
- ring_in_bread: FIXED (foil). The true match (her voice stalls) was the only option that ended badly: in the foil, the cashier left "for a better-paid job", and the lure ends well. The foil now ends with the cashier's docked pay never returned. The link break is unchanged.
- map_omission: OK. Taking the lane off the maps keeps out tourists and the ambulance alike, and he must put it back himself. In the foil, he puts it on the maps and a flood blocks the ambulance (cause). The lure (the exaggerated map comes true) is only a distant echo of the Agloe "paper town" anecdote: the cause and the turn differ. Kept.
- weighed_words: FIXED (lure). The old lure (the deputy learns every speech from her and uses her own words to take her seat) was the ungrateful-student turn (Hou Yi and Feng Meng; All About Eve). New lure: a merchant rigs her lent scale with light weights, and the town blames her and votes her out. It keeps a sad ending, so the true match is still not the only sad option.
- fear_market: OK. The cured investor holds through the real crash that his old panic would have fled. In the foil, the market recovers (link).
- first_snowflake: OK. The frozen lineup stops the team's growth until love for his daughter makes him let it go. In the foil, the lineup changes and money troubles cause the slump (cause). The arc-c release by love holds.
- dying_forge: FIXED (lure). "Burns her own furniture, then her own door, to keep it lit" was Bernard Palissy's well-known act (he burned his furniture to feed his kiln). She now cuts down her own apple trees and the oak by her door while the town keeps its wood for itself. The rest of the lure is unchanged.
- alike_fields: OK. Making every score equal leaves the office no top nurse to take. In the foil, the office takes the longest-serving nurse (link).
- question_children: FIXED (lure). "takes his little sister's place" before a child-eating monster was the volunteer-tribute turn (Theseus; The Hunger Games). New lure: a frightened girl truthfully tells Ulla where the grain is hidden, the village starves, and she is shunned. It is a different pattern with a sad ending.
- shadow_rent: OK. The stall holders' larger shared network swallows his router's count. In the foil, a council ruling ends the fee (cause).
- confessed_fear: OK. The champion helps only those who admit a hidden weakness, and helps the girl with that very fear. In the foil, the rule is where you live (cause).
- bent_road: OK. The unpaved corner, left out of mercy, soaks up the flood. In the foil, she buys and paves the corner and new drains keep the lot dry (cause). The Sanssouci miller echo comes only from the hint.
- returned_coin: OK. His returned wallet shames the founder into paying for a shelter for the whole district. In the foil, she thanks him and drives on (link).
- delta_crossing: OK. She goes with the stream of runners to where the barriers end. In the foil, a marshal opens a gap (cause).
- millers_coat: FIXED (lure). The old lure (driven out, grows rich as a miller elsewhere, and in the famine the village must buy flour from him) was the Joseph plot, and joseph_brothers is in the famous set. New lure: Cade secretly keeps coins from the coat's lining, the widow learns of it, and he is driven out.
- low_barges: OK. Doubled rents sink the grand restaurants, and the rent-free trucks inherit the lunch crowd. In the foil, a chain takes over and has the trucks banned (link).

## B (mechanisms)

- lint_plug: OK. Two flows meet at the tight corner, lock each other in place, and the queue backs up into one of the sources. In the foil, the buses go another way and a broken gate causes the queue (cause).
- milk_truck_bridge: OK. A cost-saving move puts heavy nightly jobs on the free shared server, it wears out, and all large jobs are banned. In the foil, the manager upgrades it (link).
- fortnight_bins: OK. Monthly clean-up leaves junk posts up, they draw scam accounts to the forum, and a review pulls its app. In the foil, a password leak brings the scammers (cause).
- plastic_ladders: OK. The light vans cannot carry the saw, the deep holes are skipped, and they spread over winter. In the foil, a dry winter leaves them as they are (link).
- quiet_fan: FIXED (lure). The old lure (flour dust builds up and a spark lights it) is a combustible-dust fire, a named textbook hazard. The new lure is a different chain: less airflow, a hot and damp top floor, weevils in the flour, the stock dumped.
- pen_shade: FIXED (lure). The old lure (the farm's waste feed makes the neighbor's mussels grow faster) is the textbook principle behind integrated multi-trophic aquaculture. New lure: the pens block the mussel farmer's channel, and his storm-night shortcut through them wrecks his boat.
- laptop_exams: OK. Phone tickets close the kiosk, and the only cash machine on its wall goes with it. In the foil, the machine is moved to the library (link).
- four_day_week: OK. The freed Thursday fills the only MRI scanner and pushes the stroke clinic to after midnight. In the foil, the private scans go elsewhere (link).
- blue_filter: OK. A timed power-saving mode makes the scales read low, so overweight parcels pass. In the foil, only the screens dim and a unit mix-up causes the error (cause).
- thin_newsprint: OK. Thin walls let the spin music through, the yoga instructions are lost, and the class moves to the end studio. In the foil, there is no bleed and the move is for windows (cause).
- narrow_beacon: OK. Text alerts leave out the residents without mobiles, and they are trapped. In the foil, a warden knocks on their doors (link).
- cocoa_mulch: OK. Sweet-like beads are swallowed and swell, and the hospital extends its hours. In the foil, the beads dissolve and a flu outbreak causes the hours (link).
- blocked_breeze: OK. The hotel's shade forces grow lamps, and the power cost reaches the box price. In the foil, the hotel pays the bill (link).
- panel_shade_grass: OK. The sheltered ledge draws pigeons whose droppings eat the girders. In the foil, the ledge is sealed and road salt causes the corrosion (cause).
- aluminium_dust: OK. Salt passes filters built for mud and germs, and the water fails its test. In the foil, dilution keeps it under the limit (link).
- firework_smoke: OK. The festival jam holds the ice truck past the only deep tide of the week. In the foil, the road is clear and a breakdown causes the miss (cause).
- precut_pumpkin: OK. Phone sharing closes the lab, and the darkroom club must use phones. In the foil, the club orders supplies online (link).
- screw_caps: OK. Lost tolls stop the upkeep, and the weak banks give way in a storm. In the foil, the tolls continue and a record storm overtops strong banks (cause).
- free_refills: OK. Free energy drinks spoil the crew's sleep and raise sick days. In the foil, the drinks are caffeine-free (cause).
- cruise_days: OK. The regulars come early to avoid the lunch crowd and use up the towels first. In the foil, the pool doubles its towel order (link).
