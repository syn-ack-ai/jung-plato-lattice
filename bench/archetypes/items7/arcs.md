# items7 myth arcs (2026-11-01)

Arc type, Jungian and Chasidic reading, coherence (1-5: does the consequence follow from the act with symbolic
necessity) and echo flag for the final 80 invented myths in `assign7.py`. Machine-readable copy: `arcs.json`
(`famous_mix`, `invented`, plus `famous`: the arc given to each of the 100 famous myths).

## Arc mix

Target = famous count x 0.8; every type must land within +-2 of it.

| type | arc | famous (100) | target (80) | invented before | invented after |
|---|---|---|---|---|---|
| a | transgression/hubris, then fitting or ironic consequence | 41 | 32.8 | 45 | 33 |
| b | bargain or gift with a hidden price | 4 | 3.2 | 11 | 3 |
| c | descent, exile or loss, then return or ascent | 5 | 4.0 | 3 | 4 |
| d | sacrifice that brings renewal, rescue or liberation | 7 | 5.6 | 6 | 5 |
| e | trickster: cleverness or the weak beating power | 15 | 12.0 | 3 | 12 |
| f | trial or test of virtue rewarded | 14 | 11.2 | 4 | 11 |
| g | reconciliation or forgiveness | 2 | 1.6 | 5 | 2 |
| h | quest or rescue achieved | 5 | 4.0 | 0 | 4 |
| i | transformation or metamorphosis | 2 | 1.6 | 3 | 2 |
| j | reversal: lowly raised or mighty brought low | 5 | 4.0 | 0 | 4 |

Mean coherence: 4.51 before, 4.53 after. Echo flags before: none 23, motif 50, structural 7, plot 0; after: none 22, motif 58, structural 0, plot 0.

The famous set itself is cautionary-heavy (41 of 100 are type a). Tragic-irony myths with no clean arc
(Cassandra, Gelert, Aegeus, the blind men and the elephant) were counted as a; the same rule was applied to the
invented set (for example traded_coats, unfinished_song). Judgment calls on the famous set:

- samson: a over d: the hint centres on the betrayed secret and the fall; the temple is the last act
- esther: d over h: she risks her life to save her people
- good_samaritan: f over j: the passers-by are tested; the despised one passes
- cassandra: a: tragic irony of ignored warnings (no clean arc fits)
- achilles_heel: j: the mightiest brought low by one small point
- penelope: e over c: the hint centres on her weaving trick
- cyclops_nobody: a over e: the boast after the escape is the turn
- antigone: a: the hint ends on Creon's stubbornness destroying his family
- dido_aeneas: d: love given up for the founding duty
- gyges: j: a shepherd seizes the throne (no consequence in the hint)
- emperor_clothes: a over e: vanity publicly exposed
- ugly_duckling: j over i: the scorned one is revealed as the noble
- sleeping_beauty: a: the futile attempt to avoid a curse (ironic fate)
- tortoise_hare: a over j: the hint centres on the hare's overconfidence
- fox_grapes: a: cautionary fable, no arc fits cleanly
- blind_elephant: a: cautionary parable, no arc fits cleanly
- goldilocks: h: a search that ends in 'just right' (weak fit)
- nightingale: g over c: the wronged bird returns and saves the emperor
- savitri: e over h: she wins by a boon Death cannot grant without returning him
- frog_well: a: the narrow view humbled (Zhuangzi's boastful frog)
- amaterasu: c over e: withdrawal of the light and its return
- rostam_sohrab: a: tragic recognition too late
- gelert: a: a rash act and its irreversible consequence
- aegeus_sail: a: a forgotten sign and its ironic consequence

## Final 80

New patterns (added in the rebalance) are marked *.

| # | id | arc | coh | echo | nearest known story | Jung | Chasidic |
|---|---|---|---|---|---|---|---|
| 0 | pear_for_wind | a | 4 | motif | harvest customs that leave the last fruit or sheaf for the spirits | inflation (the ego takes the share owed to the transpersonal) | middah k'neged middah |
| 1 | taught_tongue* | e | 5 | motif | Horace, 'captive Greece took her rude conqueror captive'; Jochebed nursing Moses for Pharaoh's daughter | the conquered anima reshapes the conqueror | birur (the captor is raised by the captives' sparks) |
| 2 | woken_hill | a | 4 | none | - | inflation (commanding the unconscious) | hester panim (the hill turns its face away) |
| 3 | stamp_ponds* | e | 4 | motif | giant-footprint lakes (Paul Bunyan) | the shadow's force turned to use | birur (the oppressor's blow becomes the blessing) |
| 4 | tall_chimney | a | 5 | motif | Tower of Babel (built higher than the sacred) | inflation and enantiodromia | shevirat ha-kelim (more fire than the vessels can hold) |
| 5 | unroped_guide | a | 5 | none | - | inflation (the hero who needs no one) | middah k'neged middah |
| 6 | honest_arrow* | f | 4 | none | - | none | emet |
| 7 | opened_millpond* | f | 4 | motif | the widow of Zarephath (1 Kings 17): the giver's store never runs out | none | chesed returned (giving uncovers the giver's source) |
| 8 | confessed_fear* | f | 4 | motif | oracle tales (Delphi's 'know thyself') | the persona dropped, the shadow confessed, the Self answers | vidui (confession) and bittul |
| 9 | snail_stone* | f | 5 | motif | Aesop, Zeus and the Tortoise (a house carried forever, there as a punishment); the tortoise and the hare | individuation (the burden carried becomes one's house) | nisayon (a trial that raises) |
| 10 | rain_flute | b | 5 | none | - | shadow (one's gain is another's unseen loss) | none |
| 11 | red_in_blood* | h | 4 | motif | Andersen, The Snow Queen; colour-thief picture books | the journey to the cold queen (feeling brought back) | mesirut nefesh (the self as the vessel) |
| 12 | truth_bell* | f | 5 | motif | Bocca della Verita; the Bell of Atri | integrity without persona | emet |
| 13 | wrong_key | f | 4 | none | - | individuation (one's own answer to the riddle) | nisayon (a trial that raises) |
| 14 | bent_road | f | 3 | motif | the miller of Sanssouci (the king's plan bends around a poor man's mill) | the Self (the crooked path that protects) | middah k'neged middah (shelter for shelter) |
| 15 | silent_field | f | 5 | motif | The Kind and the Unkind Girls (ATU 480) hidden-test tales | persona vs the Self (letter of the rule vs its spirit) | pikuach nefesh overrides the rule |
| 16 | dyer_exile | c | 5 | motif | exile-returns-to-save tales (Joseph) | night-sea journey (the shadow's gift brought home) | yeridah l'tzorech aliyah |
| 17 | varn_wheel* | j | 4 | motif | the Jubilee year (Leviticus 25) | fortune's wheel (rota fortunae) | none |
| 18 | fathers_stick | a | 5 | motif | the prodigal son | puer aeternus (the son's flight disables the father) | teshuvah that comes too late |
| 19 | eclipse_exile | c | 4 | motif | eclipse predictions (A Connecticut Yankee; Columbus 1504) | night-sea journey | yeridah l'tzorech aliyah; teshuvah |
| 20 | traded_coats | a | 5 | none | - | loss of the other half (the twin) | none |
| 21 | swapped_lenses | a | 5 | none | - | shadow (the betrayed pledge comes home) | middah k'neged middah |
| 22 | blind_mason* | c | 4 | motif | the blind seer (Tiresias): loss of sight brings another sight | the wounded healer (the wound becomes the gift) | yeridah l'tzorech aliyah |
| 23 | milestone_courier | a | 5 | motif | Ovid's punishing metamorphoses | inflation and enantiodromia (speed becomes stillness) | middah k'neged middah |
| 24 | current_daughter | i | 4 | motif | Ovid's river metamorphoses (Arethusa) | enantiodromia (opposition becomes identity) | bittul (merging into the greater) |
| 25 | alike_fields* | e | 4 | none | - | trickster | none |
| 26 | delta_crossing* | h | 4 | none | - | night-sea journey (surrender to the current) | bittul |
| 27 | unmissed_price | b | 5 | motif | Jephthah's vow; 'what you have at home without knowing it' bargains | loss of the feeling function | none |
| 28 | ring_lens | d | 4 | none | - | coniunctio given to the community | mesirut nefesh |
| 29 | buried_bell | d | 5 | none | - | bearer of the collective's memory (scapegoat) | mesirut nefesh |
| 30 | raft_of_doors | a | 5 | motif | The Little Red Hen (no share for one who gives nothing) | none | middah k'neged middah |
| 31 | moon_debt | e | 4 | motif | fool tales about the moon; impossible-date promises | trickster | none |
| 32 | whistled_flock | a | 4 | none | - | none | middah k'neged middah |
| 33 | dream_rumor | a | 5 | motif | Balaam | shadow and enantiodromia | the curse turned into blessing (Balaam) |
| 34 | lie_spring | a | 5 | motif | magic truth-perception gifts | shadow-sight destroys the persona's comfort | emet without chesed |
| 35 | empty_lot* | j | 5 | motif | the division of the world by lot (Zeus, Poseidon, Hades); the youngest son's poor share | the void as the creative matrix | tzimtzum (only an emptied space can receive creation) |
| 36 | shunned_island* | j | 5 | motif | Poe, The Masque of the Red Death (inverted); the Decameron's flight from plague | the rejected (shadow) becomes the centre | the rejected stone becomes the cornerstone (Psalm 118:22) |
| 37 | gentle_hives* | f | 4 | motif | only the pure may reach the prize (Galahad and the Grail) | none | middah k'neged middah (gentleness answered with gentleness) |
| 38 | oldest_ram* | e | 4 | none | - | trickster | none |
| 39 | bell_wall | e | 4 | motif | Isaiah 2:4 (swords into ploughshares) | sublimation of aggression | birur (the forge's fire raised) |
| 40 | bled_mural | a | 5 | motif | pentimento; Salieri and Mozart | shadow made visible | middah k'neged middah |
| 41 | fine_net | a | 5 | none | - | inflation and enantiodromia | middah k'neged middah |
| 42 | question_children* | e | 4 | motif | the Sphinx's riddle | trickster against the devouring mother | none |
| 43 | ferry_army* | e | 4 | motif | work-to-rule and delaying tactics (no tale with this plot known) | trickster | none |
| 44 | nameless_river | d | 5 | motif | Moses: 'blot me out of your book' | ego dissolved into the Self | bittul; mesirut nefesh |
| 45 | first_snowflake | c | 4 | motif | Frozen (2013) endless winter; Persephone | puella clinging to one season, released by love | teshuvah |
| 46 | name_thief | a | 5 | none | - | loss of home (the Self out of reach) | middah k'neged middah; galut |
| 47 | taxed_laughter | a | 5 | motif | the never-laughing princess; Timm Thaler (a sold laugh) | repression of the feeling function | middah k'neged middah |
| 48 | dying_forge | d | 4 | motif | eternal-flame legends (the Vestal fire) | renewal of the collective's sacred fire | yeridah l'tzorech aliyah (the fire falls and is relit from a higher source) |
| 49 | singing_gold | a | 5 | motif | sirens; Tolkien's dwarves who delved too deep | seduction by the unconscious (siren anima) | yeridah without aliyah |
| 50 | low_barges* | j | 5 | motif | Aesop, The Oak and the Reed | enantiodromia (the high falls, the low rises) | none |
| 51 | literal_guard | a | 5 | none | - | senex rigidity (persona over Self) | pikuach nefesh ignored (letter over spirit) |
| 52 | sand_names | a | 4 | motif | King Canute and the tide | ego consciousness against the unconscious sea | birur (what is named is kept) |
| 53 | answered_fire* | h | 4 | motif | Hero and Leander's lamp | anima as the guiding light | it'aruta d'le-tata (an awakening from below draws an answer from above) |
| 54 | fear_market | b | 5 | motif | Grimm, The Youth Who Went Forth to Learn What Fear Was | the shadow's protective function | yirah as protection |
| 55 | shared_kiln* | f | 4 | none | - | reconciling the opposites (the rival as brother) | chesed to a rival |
| 56 | cracked_jar | e | 4 | motif | Nathan's parable to David | trickster/fool speaking truth to the king | tochachah leading to teshuvah |
| 57 | lie_jar | a | 5 | motif | Pandora's jar | return of the repressed | middah k'neged middah |
| 58 | hawthorn_wall* | e | 4 | motif | curse and prophecy loopholes (Birnam Wood in Macbeth); the stone cut without hands (Daniel 2:34); the thorn hedge of Sleeping Beauty | trickster (the loophole) | none |
| 59 | clock_heartbeats | a | 4 | motif | life-candle tales (Godfather Death) | confrontation with mortality | hester panim (what should stay hidden) |
| 60 | shadow_rent* | e | 4 | motif | land-measure tricks (Dido's oxhide) | trickster | none |
| 61 | ungrateful_well | a | 5 | none | - | none | hakarat ha-tov withheld; middah k'neged middah |
| 62 | imitated_thrush | a | 4 | motif | Andersen, The Nightingale; Marsyas | inflation (persona usurps the natural voice) | tikkun (she must fill what she emptied) |
| 63 | ring_in_bread | a | 5 | motif | ring found in a fish or loaf (Polycrates) | the shadow swallowed | teshuvah refused |
| 64 | birthday_birches* | h | 4 | motif | Hansel and Gretel (path markers); birth-tree customs | the mother's care as the path home | none |
| 65 | two_rowers | g | 5 | motif | Isaiah 11:6, 'a little child shall lead them' | the divine child uniting opposites | achdut (shalom) |
| 66 | millers_coat | i | 5 | motif | impostor tales (Martin Guerre) | identification with the persona | none |
| 67 | grandfather_oak | a | 5 | motif | Erysichthon | severing the ancestral root | bal tashchit; middah k'neged middah |
| 68 | covered_mirrors | a | 5 | motif | Snow White's queen; Miss Havisham | narcissism (puella aeterna) | middah k'neged middah |
| 69 | wet_coins | a | 5 | motif | blood money (Judas); Lady Macbeth's hands | shadow, integrated by confession | teshuvah |
| 70 | map_omission | a | 5 | none | - | shadow of concealment | hester panim (self-made hiddenness) |
| 71 | hoarded_seed | a | 5 | none | - | none | middah k'neged middah |
| 72 | unfinished_song | a | 4 | motif | Penelope's waiting | anima longing; the incomplete coniunctio | galut awaiting tikkun |
| 73 | tide_bride | d | 5 | motif | the Doge of Venice's marriage to the sea | coniunctio with the unconscious | tzimtzum (she withdraws so the island can exist) |
| 74 | returned_coin | f | 4 | motif | Mercury and the Woodman (honesty rewarded) | the Self in the beggar (the king's shadow) | the hidden tzaddik |
| 75 | forgiven_carter | g | 4 | none | - | shadow integration | teshuvah |
| 76 | dimmed_light | a | 5 | motif | wreckers' false-light legends | none | middah k'neged middah |
| 77 | weighed_words | a | 5 | none | - | none | none |
| 78 | fed_bandit* | e | 5 | motif | the bond of shared salt (the robber chief who refuses salt in Ali Baba) | trickster; the shadow tamed by being fed | itkafya through chesed (the other side is given its portion and bound by it) |
| 79 | proud_bell | a | 5 | motif | bells melted down for other uses | inflation and enantiodromia | birur (melted into everyday vessels) |

## Replaced originals (25)

| original | arc | coh | echo | replaced by | reason |
|---|---|---|---|---|---|
| counted_stars | a | 4 | structural | stamp_ponds (e) | structural echo of Clarke, 'The Nine Billion Names of God' (completing a count puts the stars out) |
| owl_secret | a | 5 | structural | shunned_island (j) | structural echo of Ascalaphus (Ovid, Met. 5): an informer becomes the owl |
| beautiful_boat | a | 5 | structural | answered_fire (h) | structural echo of the parable of the talents (a treasure kept unused fails when needed); the talents is in the famous set |
| told_bees | a | 5 | structural | birthday_birches (h) | structural echo of Midas's barber and the reeds: a secret told to nature is broadcast by it |
| frost_warmth | b | 5 | structural | truth_bell (f) | structural echo of Hauff, 'Das kalte Herz' (warmth traded to a spirit for prosperity; cold to the family); also surplus type b |
| bridge_bride | d | 5 | structural | low_barges (j) | structural echo of Lysistrata (marital union withheld as leverage until the men end an injustice) |
| river_bends | i | 4 | structural | gentle_hives (f) | structural echo of Alpheus and Arethusa (a river in love with a bathing girl follows her) |
| erased_law | a | 3 | none | taught_tongue (e) | weakest cautionary: coherence 3 (why everyone else forgets how to read is arbitrary); surplus type a |
| weathervane_judge | a | 3 | motif | confessed_fear (f) | weakest cautionary: coherence 3 (condemning his son does not follow from forgetting how to judge); motif echo of Rabelais's Judge Bridlegoose; surplus type a |
| canyon_echo | a | 4 | motif | honest_arrow (f) | weakest cautionary: coherence 4 with a motif echo (Echo and Narcissus); surplus type a |
| river_purchase | a | 4 | motif | opened_millpond (f) | weakest cautionary: coherence 4 with a motif echo (Midas); surplus type a |
| skin_map | a | 4 | motif | alike_fields (e) | weakest cautionary: coherence 4 with a motif echo (The Illustrated Man, Memento); surplus type a |
| beeless_hive | a | 4 | motif | delta_crossing (h) | weakest cautionary: coherence 4, cheating-the-devil motif, and a second frost bargain beside frost_warmth; surplus type a |
| tomorrow_wind | a | 4 | motif | empty_lot (j) | weakest cautionary: coherence 4 with a motif echo (Looking-Glass memory); surplus type a |
| early_grave | a | 4 | motif | fed_bandit (e) | weakest cautionary: coherence 4, rests on a folk belief about open graves; surplus type a |
| dream_loom | b | 5 | motif | snail_stone (f) | surplus type b (11 against a target of 3); motif echo of La Peau de chagrin |
| honest_scale | b | 4 | none | red_in_blood (h) | surplus type b; coherence 4 |
| grief_birch | b | 5 | motif | blind_mason (c) | surplus type b; motif echo of Midas's barber and the reeds |
| thrown_bout | b | 4 | none | question_children (e) | surplus type b; coherence 4 |
| sea_returns_boat | b | 4 | motif | ferry_army (e) | surplus type b; coherence 4, motif echo of The Monkey's Paw |
| sold_echoes | b | 4 | none | hawthorn_wall (e) | surplus type b; coherence 4 |
| fog_lover | b | 5 | motif | shared_kiln (f) | surplus type b; motif echo of demon-lover ballads and the Snow Maiden |
| empty_chair | g | 4 | motif | varn_wheel (j) | surplus type g (5 against a target of 2); motif echo of Jacob and Esau and the prodigal son |
| stolen_lullaby | g | 4 | none | oldest_ram (e) | surplus type g; coherence 4 |
| axe_memory | g | 4 | none | shadow_rent (e) | surplus type g; coherence 4 |
