# Review of items7/w0.py (writer 0)

Reviewer: Claude, 2026-11-01. Blind pick: a fresh agent saw only the anchor and the shuffled, unlabeled true/lure/foil
of all 40 items (no pattern line, no file access). It chose the true match in 40/40, all with high confidence, so
there were 0 blind-pick failures before the fixes. A second fresh agent re-ran the blind pick on the 8 items whose
options changed (8/8 correct, high confidence) and checked them for known-story echoes. That found the Penelope
device in moon_debt's true, which was then replaced and re-checked blind. `python3 check_items3.py --dir items7
items7/w0.py`: FAIL 0, WARN 0 before and after. brk split after fixes: A 10 cause / 10 link, B 9 cause / 11 link.

Echoes found that sit in the assigned hints, not in this writer's options (left for the pattern inventor):
grandfather_oak = Chesterton's fence (structural); moon_debt = "ad Kalendas Graecas" (structural); ferry_army =
work-to-rule (already noted in novelty_notes.md); snail_stone = the Tortoise and the Hare (already noted);
hoarded_seed = predator satiation in reverse (structural).

## A (myths)

- pear_for_wind: OK. True carries every step (the set-aside share taken, help withheld from her alone, her business fails); link foil (he keeps helping, she closes for an unrelated reason); lure is a reconciliation story.
- fathers_stick: OK. Taking the car disables the mother; cause foil leaves the car behind. The lure's gold hidden in a cane is a motif only (Sancho Panza's cane judgment), not that plot.
- raft_of_doors: OK. Shared rescue, one man keeps his share back and is left behind; cause foil hands over the fuel.
- bled_mural: OK. The overwrite itself, in a medium that keeps versions, exposes both works and names; link foil uses a tool that keeps only the latest version.
- singing_gold: OK. The lure grows sweeter step by step until the family cannot reach him; cause foil: the wins never come closer, so he quits.
- clock_heartbeats: OK. Curiosity reveals the mentor's remaining life, carried in silence; link foil reveals a retirement instead.
- grandfather_oak: FIXED. Lure (a builder mocked for his oak barn sees it alone survive a fire, and the neighbors shelter their grain in it) echoed Noah and the Three Little Pigs. Replaced with a timber merchant who lies about the oak and cheats Aldo (fraud, a different pattern).
- hoarded_seed: FIXED. Lure (seed lost, neighbor accused and driven out, seed found in his own barn wall) was the plot of the Liezi fable of the man who lost his axe, and repeated the set's ring_in_bread pattern. Replaced with Ulf sowing the valley's shared seed early against a warning and losing it to frost.
- proud_bell: OK. The elite-only hall refuses shelter, people die, and it is turned into a soup kitchen; cause foil opens it.
- dyer_exile: OK. The mistake becomes a skill in exile, and she returns as the only one who can warn; link foil has her abroad. Hacking to change grades is a film image (WarGames), motif only.
- ring_lens: OK. The love token is sold for a shared protection that outlasts her, honored on the anniversary; link foil: the device fails.
- tide_bride: OK. The bond draws the bully ever closer until she leaves so the school is spared; link foil: his visits turn benign.
- moon_debt: FIXED. Lure (the creditor forgives the debt, and the debtor later cares for him when he is old and ruined) was mercy repaid, the Lion and the Mouse, which novelty_notes.md already rejected (spared_rival). Replaced with a secret debt that the street misreads as gambling, and the gossip ruins him. The re-check also found that the true (the builder leaves one job undone on every house, so none is ever "finished") used Penelope's web device. New true: the supplier is talked into taking whatever is in the builder's account on the last day of the month, and the crew is paid the day before, so the account, full at mid-month like the full moon, is always empty on the due day. New link foil: a court freezes the account first and he is paid.
- ferry_army: FIXED. Lure (she saves the warlord's son midstream, he swears to spare her village and burns it anyway) echoed the Scorpion and the Frog and the Farmer and the Viper (the latter rejected in novelty_notes.md). Replaced with Hethe made keeper of the crossing, who fleeces her own village and loses everything when the warlord falls.
- fed_bandit: FIXED. Lure (a bandit hired to drive off a worse gang becomes the valley's oppressor) was the plot of Aesop's The Horse and the Stag. Replaced with a failed poisoning that provokes the bandit's revenge.
- snail_stone: FIXED. Lure was the same pattern with another hero (the otter carries the stone to the end and is given a home). Replaced with Ommet refusing to let go of her stone, so the shared work fails and the river floods every meadow.
- silent_field: OK. The stated rule hides the real test, and the one who breaks it to help passes; link foil hires the rule-keeper. A hidden kindness test is the anchor's own motif.
- two_rowers: OK. Rivalry stalls the shared task until a child's count gives one rhythm; link foil: they ignore him and a crane does it. A piano carried up stairs is a slapstick image (Laurel and Hardy), motif only.
- answered_fire: OK. Persistent signaling is at last answered and the answer makes the rescue possible. The blind picker found the cause foil close (the rescue still comes from the lost man's call) but chose the true on the persistence step.
- varn_wheel: OK. The rule against being on top leads the rich to give everything away and ends poverty; cause foil hides the money abroad instead.

## B (mechanisms)

- salted_hedge: OK. Spray drift kills the reeds, and the mud they held blocks the water intake; cause foil sprays only on still days.
- match_shift: OK. Two timetables collide on one line, incoming staff are late and outgoing staff stay on; cause foil schedules the backup apart.
- bog_road: OK. Engine heat dries and shrinks the planks, and the mount loosens; cause foil puts a plate under the engine.
- white_roof_glare: OK. The energy saving pushes a nuisance onto neighbors, who then use more energy; link foil: the gym fixes it.
- dropped_flight: OK. A cut link makes the data miss the forecast window, so forecasters switch to a substitute; cause foil upgrades the link.
- early_grapes: OK. Readiness comes before the skilled markers are free, and the temps cannot tell good from bad; link foil gives them an answer key.
- clover_wheat: OK. A surplus causes fast, weak growth that an ordinary stress knocks over; link foil: the company survives.
- app_transfers: OK. The new requirement pushes users back to the one old outlet, which overflows on peak days; cause foil keeps the phone line.
- no_saturday_post: FIXED. Foil (copies cut, the guides run out by the tenth, the halls withdraw) still followed the pattern another way: a distribution cut keeps the listing from readers when it is useful. The blind picker found it close. New foil: the guide stays monthly, and the halls withdraw because visitors buy tickets on a new app.
- cat_router: OK. Heating off, mice nest in the one warm box, it overheats, and the alarms die; cause foil keeps the heating on.
- carbon_bikes: OK. The reader cannot sense the new passes, users break the rule, and enforcement gathers there; link foil updates the reader.
- spiked_ledges: FIXED. brk relabeled from cause to link: the blocking still happens but does not displace anyone, and the demo fails for an unrelated reason. Text unchanged.
- beaver_bunkers: FIXED. True (delayed discharge fills the wards, ED trolleys pile up, operations are cancelled) is the widely known, named "bed blocking" / exit-block chain. Replaced true and foil: a video site's archive job stops, used space rises on every server, the smallest disks fill first, and those servers are taken offline. The new cause foil keeps the job working and loses the servers to a cooling fault.
- rotting_sandbags: OK. Left-over soda sacks split and their dust packs the extractor mesh; link foil: the mesh is washed weekly.
- lined_canal: OK. Stopping the waste cuts the pigs' hidden feed, the herd goes, and the butcher leaves; link foil: the farmer switches feed.
- heated_seats: OK. The comfort upgrade leads to a crash after it ends, and the next stop sells the remedy. "Sugar crash" is a popular notion and is not named in the text.
- ring_road: OK. The emptied room draws new users who damage the table, which is then removed; link foil: a proper table is bought.
- spray_film: OK. A dust film delays the battery fill past the peak-price window; cause foil cleans the panels.
- radio_clocks: OK. Glare hides the shared cue and the halves drift apart; link foil: they follow the bass drum.
- sourdough_heat: OK. The shifted kitchen hours make the leftovers arrive after the shelter's hour of need; link foil: the shelter serves them at breakfast.

## Counts

OK 31 (A 14, B 17), FIXED 9 (A 6, B 3), REWRITTEN 0. Blind-pick failures before fixes: 0 of 40.
