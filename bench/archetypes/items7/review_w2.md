# Review of items7/w2.py (2026-11-01)

Method. Blind pick: a fresh agent that had not seen the file got an unlabeled sheet: each anchor with true, lure and
foil shuffled, no pattern line. It picked the true match in 40/40 items, all at high confidence, with no ties, so there
were 0 blind-pick failures before fixes. (The reviewer had already read the labeled file, so the reviewer did not do
the pick.) Faithfulness, novelty (novelty_notes.md plus the reviewer's own knowledge) and the arc check
(arcs.json) were then done by hand on every option. Checker after fixes:
`items: A 20, B 20; FAIL 0, WARN 0`.

Counts: 36 OK, 4 FIXED, 0 REWRITTEN.

## A (myths)

- tall_chimney: OK. Showing off leads to excess power, which ruins what it serves. Lure: the chimney saves lost traders. Foil: no excess, and a drought kills the roses (cause). Keeps arc a.
- swapped_lenses: OK. The broken pledge harms his own child on the other side. Lure: a rescue. Foil: the old engine works (link).
- dream_rumor: OK. The malicious rumor draws people to the rival. Foil: sincere praise, so no backfire (cause). The lure's secret overnight baking is a faint image echo of The Elves and the Shoemaker, not its plot.
- name_thief: OK. He removes the way to a place, loses his own way back and stays within sight of it. Lure: the town names itself. Foil: fog alone (cause).
- sand_names: OK. Whatever she forgets is lost, and what is kept shrinks to her memory. Foil: the supplier covers for her (link). The Canute motif is noted in arcs.json.
- imitated_thrush: FIXED. The lure (Sadi frees the thrush from a net, and later its song leads her home) was the grateful-animal tale (Lion and the Mouse type, which novelty_notes lists as known). New lure: the woods go quiet from an outside cause, and Sadi trades a month of work to save the nest.
- wet_coins: OK. Hush money taints her pay until she confesses to the family. Foil: no taint and no confession (link).
- dimmed_light: OK. A safety light cut to save money causes a death, and he lives on the gain. Foil: the light is on and he still profits (cause). The lure's burning of furniture and boat is a distant image echo (Fogg burning the Henrietta), not a plot.
- unmissed_price: OK. A bargain for "what she won't miss" takes the power to grieve. Foil: grief intact (link). Keeps arc b.
- blind_mason: FIXED. The foil gave no reason why spices measured by weight came out uneven, which contradicts the true. Added "Without taste to guide her".
- nameless_river: OK. She gives her name to the threatening power, the town is saved and she is forgotten. Foil: she keeps her name (cause).
- stamp_ponds: OK. The mocking ads become the shop's draw. Foil: the success comes from a new school (link).
- bell_wall: FIXED. The lure (the smith sells blades to raiders, who turn them on his village) had the turn of Aesop's The Woodcutter and the Trees (the victim supplies the weapon) and of the Iron Man origin. New lure: the village blames the smith, drives him out and is left with no one to forge for it.
- hawthorn_wall: OK. Kept to the letter of the ban, a grown barrier is no fence. Foil: no ban, and the barrier fails (cause).
- opened_millpond: OK. Giving the well away drains it, and the draining clears the old source. Foil: the draining wrecks the shaft (link).
- wrong_key: OK. The broken login is fixed properly instead of forced, and she is rewarded. Foil: she is punished (link).
- shared_kiln: OK. The rival shares his own means, the results tie and both win. The true's link from the help to the tie is implicit ("on his machine next to his own") but present. Foil: a teacher helps, not the rival (cause).
- red_in_blood: FIXED. The lure (Wrenna serves the northern queen, her heart grows cold and her mother does not know her) was Kai's plot in Andersen's The Snow Queen. New lure: a small request for one thread, granted. Notes: the anchor's hoarded-element motif also echoes Raven and the box of daylight (motif, not plot). The true's memorizing to get past a ban faintly echoes the Samuel Slater anecdote; kept, because the return to the community is the invented part.
- current_daughter: OK. Fighting the slant for years, she becomes it. Foil: she never fought (cause). Keeps arc i.
- shunned_island: OK. The cut-off team is spared and becomes the center. Foil: a cloud restore makes them unneeded (link).

## B (mechanisms)

- led_jasmine: OK. Late readiness makes regulars give up, and the trade that depended on them fails. Foil: they keep coming (link).
- damp_paper: OK. A damp store makes the stock swell and jam in the wet season. Foil: dry storage and a worn saw (cause).
- nesting_crates: OK. Freed capacity is sold, and the buyer's costs fall. Foil: the space is used in-house (link).
- glossy_crossing: OK. A slick marked zone pushes people just beside it, where watchers see them late. Foil: no slick surface, and sun glare (cause).
- dry_manure: OK. The changed input makes the waste unfit for the process its buyer needs. Foil: the pulp is fine, and a supplier closes (link). The lure reaches the same grower by another chain (sacks and rats), and the judge kept them apart.
- mice_substation: OK. Intrusion, short, cutout, and a process that needs warmth fails. The true's first step is water in a socket rather than chewed insulation, but the chain is the same. Foil: the socket stays dry (cause).
- noon_council: OK. A time change silences a group, and the decision against them passes. Foil: they object online (link).
- car_free_sunday: OK. The ban pushes the activity into the hour before, onto new people. Foil: an unrelated launch (link).
- datacenter_switch: OK. He drops his own protection because of a neighbor's side effect, and fails when it stops. Foil: he keeps his own lamp (cause).
- fuel_cap_side: OK. A design change funnels users to one side, and the queue blocks the way. Foil: staff divert the queue (link).
- solar_snow_slab: OK. A slick surface sheds its load in one mass, which blocks a way. Foil: a rough hood, and rice clogs the drain (cause).
- swallow_bolts: OK. Corrosive drip on the fixings causes a failure on the busiest day. Foil: the drip does no harm, and a truck snaps a cable (link).
- ammonia_brass: OK. An unnoticed emission spoils the neighbor's goods. Foil: a clean flue and a leaking pipe (cause).
- copper_handles: OK. Polishing for looks wears off the protective coat. Foil: the coat survives (link).
- reservoir_drawdown: OK. A new attraction jams the one shared route, and an essential service is blocked. The true's first half is what the web calls a flash crowd. No term is named, and the invented chain is intact. Foil: a separate server, and a power cut (cause).
- heat_pump_bats: OK. A lost warm refuge drives the animals into a valuable building. Foil: the cats protect the books (link).
- wire_fences: OK. Losing free shelter exposes the young, and costly shelter must be built. Foil: the trees are planted, not cut (cause).
- sauna_archive: OK. A damp exhaust rises and ruins what is stored above. Foil: air conditioning dries it, and lightning (link).
- cold_organ: OK. Cold delays the output, the leader acts early and the followers jump. Foil: no delay, and nerves (cause).
- flood_wall_meadow: OK. The barrier also blocks the good inflow, the yield thins and the buyer leaves. Foil: a mild filter and a cheaper rival (cause).
