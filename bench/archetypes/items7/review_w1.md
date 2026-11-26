# Review of items7/w1.py (2026-11-01)

Blind pick: a separate agent saw each anchor with true, lure and foil shuffled. It did not see the pattern line or the labels. It picked true in 40/40 items, all with high confidence, so there were 0 blind-pick failures before the fixes. The fixes below come from the faithfulness, novelty and ending checks.
Checker after the fixes: `python3 check_items3.py --dir items7 items7/w1.py` gives FAIL 0, WARN 0.
Counts: OK 30, FIXED 10, REWRITTEN 0.

## A (myths)

- woken_hill: OK. The neglectful owner is roused, turns to his own profit and leaves the team worse off. The arc-a dynamic (the power turns its face away) holds.
- traded_coats: OK. The survivor holds all the comfort and cannot use it. The foil enjoys it. The lure is a different pattern (greed, then a reversal).
- whistled_flock: OK. Imitation steals the followers, and in the crisis they all return to the old guide. The foil drops the imitation and keeps the crisis (cause).
- fine_net: OK. The capture tightens until it holds what must pass, and the load sinks the owner. The foil adds capacity (link).
- literal_guard: OK. The letter of the rule keeps out the help, and the protected boy is harmed. In the foil, a spare inhaler breaks the link.
- ungrateful_well: OK. The gift is withdrawn from the one who gives no thanks, and he must go through outsiders. In the foil, he thanks the neighbor and the loss has an unrelated cause.
- covered_mirrors: OK. Taking away everyone's view blinds the salesman to himself, and he is shamed in public. The Snow White and Havisham motif comes only from the hint.
- unfinished_song: OK. The vow outlives the man who never returns and becomes a custom. The reserved booth recalls real fallen-soldier customs, but not any story's plot.
- rain_flute: OK. The gift's plenty is drawn unseen from the neighbor. In the foil, the list is public and the neighbor's loss is unrelated (cause).
- eclipse_exile: OK. Every step is present (cast out, finds the error, returns, arrives as it comes true). Nothing echoes the eclipse-trick stories.
- buried_bell: OK. Removing the reminder makes the town forget, and the remover carries the memory alone. In the foil, the families keep their own memorial (link).
- taught_tongue: FIXED (lure). The true was the only option with a happy ending (foil and lure both ended in loss). The new lure has a different pattern with a good ending: the banned knowledge becomes the only cure in a fever, and the ruler must beg for it.
- oldest_ram: OK. The weak choose a tribute that weakens the taker and renews themselves. In the foil, giving the best players reverses the choice (cause).
- cracked_jar: OK. The made object keeps showing the ruler his flaw until he relents. In the foil, the clock is reset (link).
- honest_arrow: OK. Admitting luck costs the prize but wins the greater post. In the foil, there is no admission and she wins both (cause).
- truth_bell: FIXED (lure). "a small child points out who really spoke" was the signature turn of The Emperor's New Clothes. It is now an old net-mender who stood beside the rival.
- gentle_hives: OK. The rule rules out the bold, and the gentle one wins. In the foil, the mare is sedated and dragged up instead (cause).
- forgiven_carter: FIXED (lure). The old lure (refuses to forgive, drives him out, no hand helps her, the grain rots) was the pattern's mirror image. The new lure has a different pattern: the wall built against the fatal bend causes the next crash.
- birthday_birches: FIXED (true, foil, lure). In the true, the boy could not see painted stones "in a power cut", so the setting is now thick fog, and the foil was adjusted to match (he takes the far side of the road and never passes the wall). In the lure, a woodcutter leading the lost girl deeper into the forest and leaving her was a Hansel and Gretel turn. It is now a peddler who slips off with the ring.
- empty_lot: FIXED (foil). The true was the only option that ended well, because the foil ended with her selling the field cheaply. In the foil she now sells it and opens a cafe with the money. The link break is unchanged.

## B (mechanisms)

- tram_cheese: OK. Vibration cracks the skin, damp gets in and the collection moves. In the foil, the dry galleries stop the damp (link).
- snow_cannon_spring: OK. Shared diesel drawn below a higher feed pipe cuts off the far generator. In the foil, the level stays high and the battery is flat (cause).
- sealed_limestone: FIXED (lure). The old lure ran falcons leave, then pigeons boom unhunted, which is a trophic cascade, a named textbook effect. Now the slick ledges send the flock to the town hall roof, where its droppings block the gutters and a storm floods the chamber.
- oven_wall_fridge: OK. The cooler sits against a hidden heat source, runs flat out, fails over a holiday, and the stock is lost.
- fence_hedgehogs: OK. The detour onto the road gathers accidents in one place, and each accident draws helpers into the traffic. In the foil, a footpath stops the accidents (link).
- honeydew_parking: FIXED (lure). The old lure (shade, then cafes, rents rise, old shops priced out) was green gentrification, a named process. The new lure: roots lift the paving, a fall, a lawsuit and the trees are cut down.
- dog_path_rushes: OK. Trampling makes a wet dip, growth takes hold and the scrubber jams. In the foil, the dip is mopped dry (link).
- thatch_premium: OK. The price rise drives users off copper lines, the specialists go, and the lifts that must keep copper get no service.
- cup_tax: OK. The load shifts onto one old shared charging cart, which burns out. In the foil, a new power unit is fitted (link).
- dusty_markings: FIXED (true, foil). The true had stronger disinfectant fading the stickers. That copies etched_tanks' key step (a stronger cleaner damages the surface) and is not "outside grime covers the marks". Now plaster grit from the building work next door coats them, and a porter is paid to wipe them. The foil matches.
- etched_tanks: OK. A harsher abrasive grooves the enamel, bacteria hide where the brush cannot reach, and cavities follow. In the foil, there is no grooving and sodas cause the cavities (cause).
- brine_sewer: OK. Waste kills the compost's microbes, and half-rotten waste goes out. In the foil, the microbes thrive (link).
- warm_river_parasite: OK. Heat lets the germ breed, the spa closes and the dependent studio fails. In the foil, the tanks stay cool and a broken pump closes the spa (cause).
- steam_plume_ice: OK. Warm damp exhaust freezes on one cold flight of steps only. In the foil, heated steps keep it water (link). The false-alarm lure is not the cry-wolf pattern.
- matte_buses: OK. A change for looks soaks up heat, the cooling fails and the pensioners move to a worse option. In the foil, the building is shaded and the cooling fails from age (cause).
- marking_soles: FIXED (lure). In the old lure, a cheap sole leads to a slip, then a lawsuit, then schools are barred. That was the anchor's pattern with a twist (the supplier's cost cut reaches the schools through the museum). The new lure has a different pattern: the streaks expose rubber soles sold as leather, and the factory is fined for fraud.
- revolving_door: OK. The main route refuses heavy use, the heavy users move to the backup form, and it fails. In the foil, there is no diversion and a botched update breaks it (cause).
- night_pressure: OK. The quiet-hours cut leaves the late users without food, their own open stores breed mice. In the foil, the tins are sealed (link).
- beet_dust: OK. The upstream residue jams marking, the term overruns and the summer course is cancelled. In the foil, a strike causes the delay (cause).
- hard_road_studs: FIXED (foil). The foil said the skates "bite into" the strip and then "slide away unharmed", which contradicts itself. It now says they "strike" the strip.
