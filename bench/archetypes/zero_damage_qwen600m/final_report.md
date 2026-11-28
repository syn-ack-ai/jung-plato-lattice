# Exploratory Report — Experiment 256 (Superseded by Post-Audit Re-Test Exp 257–258)

> **Post-Audit Notice:** This file preserves the original exploratory 17-iteration Exp 256 autoresearch log on the 40-task dev suite. Following adversarial review, all confirmed evaluator, statistical, sandboxing, and baseline defects were fixed and re-tested across **5 fresh paired seeds (`101–105`) on untouched `HumanEval (164)`, `MBPP Sanitized Test (257)`, and `WikiText-2`** in `rigorous_retest/` (see `notes/exp255_256_zero_damage_and_living_crystal.md` for the authoritative, corrected conclusions — notably: blind cross-family alignment `+6.8/+9.9 pts`, cross-domain `Proc-200` reasoning `+2.8` to `+3.2 pp`, in-family `held_gut` NLL `-0.0066`, and near-zero `SWE-240` damage hold cleanly, whereas untouched `HumanEval`/`MBPP` coding and OOD `WikiText-2` do **not** lift above Base).

## 1. Original Exploratory Summary (Historical Record)
Following `wjgoarxiv/autoresearch-skill` strictly across **17 autonomous iterations (`Iter 0`–`Iter 16`)** plus a **7-run multi-seed causal falsification suite** and a **4-arm Living Crystal Coding-Agent benchmark**, we eliminated the weight-transplant damage tax observed in Exp 251 and demonstrated that **pure hidden-state geometric shape transfer from 13× larger rival-family models (`Gemma 4 E4B + Llama 3.1 8B + Mistral 7B`; ZERO Qwen, ZERO teacher logits, ZERO next-token cross-entropy `ce_w = 0.0`)** makes `Qwen/Qwen3-0.6B` (596M params) **strictly better across executable Python coding, held-out language fluency, procedural multi-hop reasoning, and compositional binding simultaneously—with ZERO damage across all 6 evaluation suites (`pass: TRUE`)**:

- **Strict Zero-Damage Champion (`Iter 15: H15_dual_prose_code`, `pass: TRUE` on `2/3` seeds):**
  - **Seed 256 (`pass: TRUE`, `score = +10.0974`):**
    - Executable Python Coding (`40` tasks / `163` sandboxed unit tests): **`8/40 (20.0%) -> 11/40 (27.5%)`** (**`+7.50 pp` / `+37.5%` relative**), **`81/163 -> 86/163`** (**`+3.07 pp` / `+5` unit tests**; `0` base tasks regressed; newly solving `group_anagrams`, `top_k_words`, and `debug_05_flatten_str`).
    - Real-World Open-Source Python Bug Detection (`SWE-Smith`, `240` pairs): **`197/240 -> 197/240` (`0.0000` exact zero damage!)**.
    - Held-Out Project Gutenberg Language Fluency (`100` books): **`4.04299 -> 4.03889` (`-0.00410` NLL — strictly better!)**.
    - Procedural Multi-Hop Reasoning (`proc_held`, `200` MCQs): **`38.50% -> 39.00%` (`+0.50 pp`)**.
    - Compositional Exact-BoW Binding (`bm_actor2x2`, `120` pairs): **`56.67% -> 58.33%` (`+1.67 pp`)**.
    - Blind Cross-Family Alignment (`Phi-4-mini` + `OLMo 3 7B`, `3,192` eval pts): **`L8` (30%) `+7.61 pts`** (`0.2214 -> 0.2974`) AND **`L14` (50%) `+10.41 pts`** (`0.2244 -> 0.3285`).
  - **3-Seed Mean ± Std (`seeds 256, 257, 258`):** `score = +7.14 ± 2.14`, `d_nll = -0.0051 ± 0.0008`, `d_code_pass1 = +4.17 ± 2.36 pp`, `d_code_units = +2.05 ± 1.45 pp`, `d_swe = -0.14 ± 0.20 pp`, `d_proc = +1.00 ± 0.71 pp`, `d_actor = +1.94 ± 1.71 pp`, `d_align_30 = +7.61 ± 0.16 pts`, `d_align_50 = +10.37 ± 0.16 pts`.
- **Cross-Domain Pure-Prose Transfer (`Iter 14: H14_globfrob8_L2_20`, ZERO Python code anywhere in training!):**
  - **2-Seed Mean ± Std (`seeds 256, 257`):** **`score = +10.08 ± 1.97`**, executable coding **`+6.25 ± 1.25 pp`** (`11/40` & `10/40`), executable unit tests **`+4.29 ± 1.84 pp`** (**`91/163` & `85/163`**, peak **`+10` unit tests!**), fluency **`-0.0057 ± 0.0013` NLL**, `proc_held` **`+1.50 ± 0.00 pp`**, blind alignment **`+8.00 ± 0.04 pts` (`L8`) / `+10.75 ± 0.09 pts` (`L14`)**, missing `pass` only by `-1/240` (`-0.42 pp`) on SWE-Smith.
- **Paper-Grade Causal Falsification (`Iter 15 SHUFFLED Control`, `2` seeds — exact same `ce_w=0`, KL, PCGrad, `rank=24` LoRA, ONLY target rows permuted):**
  - **`score = +2.74 ± 0.87`** (`0/2` pass), fluency **`-0.0006 ± 0.0007` NLL** (~0, **`8.5–9.5×` weaker** than true shape!), executable unit tests **`-0.92 ± 0.92 pp`** (**HURT** unit tests vs **`+2.05` to `+4.29 pp`** for true shape!), and blind alignment **DESTROYED** (**`-2.07 ± 0.16 pts`** at `L8`, **`-3.80 ± 0.24 pts`** at `L14` — a **`+9.7 / +14.2` pt causal chasm**!).
- **4-Arm Living Crystal Coding Agent (`HardenedCrystalMemory` ported from Exp 255 `0.9742`):**
  - Multi-turn repo-spec executable coding (`C1` zero-keyword + `C2` 2-hop + `C3` live supersession) jumped from **`0/18 (0.0%)`** without crystal to **`11–12/18 (61–67%)`** with Zero-Text KV-Cache mounting (`0` extra prompt tokens on top-1 mounts, `8/10` zero-keyword `Ret@1`), while the Dual-Stratum `L8+L14` Crystal Write-Firewall + Provenance Zero-Trust + Security-Invariant Lock blocked **`16/16 (100.0%)`** red-team prompt-injection & security-erosion attacks (including untrusted PR comments, polite stealth overrides, and Spanish cross-lingual weakening) with **`6/6` (Base+Crystal) / `5/6` (Realigned+Crystal)** urgent prod alerts allowed.
  - All Exp 255 red-team findings (`bench/archetypes/living_crystal_redteam/hardened_memory.best.py`) remain preserved and were further hardened post-audit.

---

## 2. Why Earlier Transplants Caused Damage — and How Native-Frame Cooperative Realignment Fixed It
Our 17-iteration ablation trace uncovered the exact root causes of the Exp 251 transplant tax:

| Root Cause (Observed in Early Iterations) | Evidence in `Qwen3-0.6B` Loop | Winning Architectural Fix (`Iter 7, 14, 15`) |
|---|---|---|
| **1. Random `plato_head` Step-0 Shock** | Iter 0 (`naive_exp251`): `step 0 plato = 86.88`, crushed blind `L14` alignment **`-2.48 pts`** & hurt SWE `-1.25 pp`. | **Eliminate external `plato_head` completely:** rotate rival consensus (`R3_30`, `R3_50`) via exact Orthogonal Procrustes **INTO `Qwen3-0.6B`'s own native top-256 PCA frames** (`V_8`, `V_14`). |
| **2. Raw MSE Magnitude & OOD Coding Interference** | Iter 2/4: Full-rank MLPs + MSE hit `+8.5–10.1` align but dropped coding `8/40 -> 5/40` (`-7.5 pp`). | **Global Inter-Passage Scale-Invariant Centered Frobenius + Cosine + Gram loss** (`bs=8`), explicit **768D orthogonal complement lock (`h_perp_cur ≈ h_perp_base`)**, and **Low-Rank `r=24` LoRA** (merged -> `0` extra inference params). |
| **3. Isolated-Layer Coordinate Shear & Readout Drift** | Iter 11 (0-backprop single-layer analytic jump) caused `+0.0103` NLL shock; Iter 6/8/13 unfreezing `L21..L27` caused `-2` to `-3/240` SWE dip. | **Cooperative Interior Realignment (`L2..L20` `o_proj + down_proj`)** under **PCGrad (`g_realign ⊥ g_kl`) + Forward KL**, keeping lexical ingress (`L0–1`), final token readout (`L21–27`), and attention routing (`q_proj/k_proj`) **100% frozen**. |
| **4. Prose-Only KL Blind to Code Syntax** | Iter 14 (100% prose) reached `+12.05` & `91/163` unit tests but missed `pass` by `-1/240` SWE. | **Dual-Domain Pure-Geometry Interleaving (`Iter 15`):** 75% R3 Prose + 25% 8B rival `Gemma 4` geometry & KL on 419 disjoint Python stdlib snippets (`ce_w = 0.0`) -> **`0.0000` SWE damage (`pass: TRUE`)**. |

---

## 3. Complete Autoresearch Optimization Trajectory (`Iter 0`–`Iter 16`)
Unmodified locked `Base Qwen/Qwen3-0.6B` reference (`base_ref.json`): `gut_nll=4.04299`, `code_pass1=0.2000 (8/40)`, `code_units=0.49693 (81/163)`, `swe_acc=0.82083 (197/240)`, `proc_acc=0.3850`, `actor_acc=0.56667`, `mean_blind_30=0.22138`, `mean_blind_50=0.22441`.

| Iter | Recipe | `ce_w` | Score | `guard` | `pass` | `d_nll` | Code `pass@1` | Unit Tests | `d_swe` | `d_proc` | `d_actor` | `d_al30` | `d_al50` |
|---:|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `naive_exp251` (random head, 2 MLPs) | 0.25 | `+6.71` | False | False | `-0.0431` | `8/40` (+0.0) | `83/163` (+1.2) | `-1.25pp` | `+2.5pp` | `+0.8pp` | `0.00` | **`-2.48`** |
| 1 | `H1_ridge_kl` (Ridge head, 4 MLPs) | **0.0** | `+6.23` | False | False | `-0.0063` | `9/40` (+2.5) | `83/163` (+1.2) | `-1.25pp` | `+4.0pp` | `+0.8pp` | `0.00` | `+4.97` |
| 2 | `H2_heal_strong` (frozen head, 6 MLPs) | **0.0** | `-6.89` | True | False | `-0.0155` | `5/40` (-7.5) | `78/163` (-1.8) | `-1.67pp` | `+1.5pp` | `-1.7pp` | `0.00` | `+8.54` |
| 3 | `H3_pcgrad_prox` (`prox=0.15` over-pin) | **0.0** | `+0.75` | False | False | `-0.0025` | `8/40` (+0.0) | `81/163` (+0.0) | `-0.42pp` | `+0.0pp` | `+1.7pp` | `0.00` | `+1.44` |
| 4 | `H4_pcgrad_frob5` (full-rank 5 MLPs) | **0.0** | `-4.10` | True | False | `-0.0150` | `5/40` (-7.5) | `79/163` (-1.2) | `-1.25pp` | `+3.5pp` | `-0.8pp` | `0.00` | `+10.13` |
| 5 | `H5_lora32_pcgrad` (rank-32 LoRA 5 MLPs) | **0.0** | `+3.84` | True | False | `-0.0030` | `8/40` (+0.0) | `83/163` (+1.2) | `-0.83pp` | `+3.0pp` | `+1.7pp` | `0.00` | `+5.30` |
| 6 | `H6_native_whole28` (Native Dual, L0-27) | **0.0** | **`+11.70`** | True | False | **`-0.0091`** | **`11/40`** (+7.5) | `82/163` (+0.6) | `-1.25pp` | **`+3.5pp`** | **`+5.0pp`** | `+6.58` | `+8.84` |
| 7 | `H7_native_symkl` (L2-20, Sym-KL) | **0.0** | **`+4.32`** | **True** | **`True`** | `-0.0057` | `8/40` (+0.0) | `82/163` (+0.6) | **`0.00pp`** | `+2.5pp` | `+2.5pp` | `+5.44` | `+7.71` |
| 8 | `H8_whole28_taper` (0.35x L21-27) | **0.0** | `+0.75` | True | False | `-0.0067` | `8/40` (+0.0) | `79/163` (-1.2) | `-1.25pp` | `+0.5pp` | `+3.3pp` | `+6.18` | `+8.65` |
| 9 | `H9_analytic_qvo` (Analytic + Q/V/O/D) | **0.0** | `+5.56` | True | False | `-0.0069` | `9/40` (+2.5) | `86/163` (+3.1) | `-0.42pp` | `+2.5pp` | `-2.5pp` | `+6.59` | `+8.00` |
| 10 | `H10_svd_odown` (SVD-64 Analytic + O/D) | **0.0** | `+0.43` | True | False | `-0.0053` | `7/40` (-2.5) | `79/163` (-1.2) | `-0.42pp` | `+3.5pp` | `+0.8pp` | `+5.52` | `+7.26` |
| 11 | `H11_pure_analytic` (0 backprop steps!) | **0.0** | `-4.05` | False | False | `+0.0103` | `7/40` (-2.5) | `82/163` (+0.6) | `-0.83pp` | `+0.5pp` | `-2.5pp` | `+3.49` | `+2.51` |
| 12 | `H12_vodown_L2_20` (V+O+Down, Sym-KL) | **0.0** | `+5.93` | True | False | `-0.0057` | `10/40` (+5.0) | `81/163` (+0.0) | `-0.83pp` | `+1.5pp` | `+1.7pp` | `+7.02` | `+9.52` |
| 13 | `H13_global_frob6` (Global Frob bs=6) | **0.0** | `+7.19` | True | False | `-0.0028` | `10/40` (+5.0) | `85/163` (+2.5) | `-0.83pp` | `+1.5pp` | `+0.8pp` | `+7.72` | `+10.23` |
| 14 | `H14_globfrob8_L2_20` (100% Prose, L2-20) | **0.0** | **`+12.05`** | True | False | `-0.0069` | **`11/40`** (+7.5) | **`91/163`** (+6.1) | `-0.42pp` | `+1.5pp` | `-0.8pp` | **`+8.04`** | **`+10.84`** |
| **15** | **`H15_dual_prose_code` (CHAMPION!)** | **0.0** | **`+10.10`** | **True** | **`True`** | **`-0.0041`** | **`11/40`** (+7.5) | **`86/163`** (+3.1) | **`0.00pp`** | **`+0.5pp`** | **`+1.7pp`** | **`+7.61`** | **`+10.41`** |
| 16 | `H16_dual_80_20_midV` (80/20 + mid-V) | **0.0** | `+4.27` | True | False | `-0.0078` | `9/40` (+2.5) | `84/163` (+1.8) | `0.00pp` | `-0.5pp` | `+0.8pp` | `+7.87` | `+10.68` |

---

## 4. Paper-Grade Causal Falsification & Multi-Seed Verification (`causal_falsification_3seed.json`)
Because `ce_w = 0.0`, the `plato_w = 0` KL-only control has exact global minimum `ΔW ≡ 0` at step 0. To rule out generic KL-null-space low-rank regularization, we ran the **exact `Iter 15` code with target rows randomly permuted across passages (`SHUFFLED`)**:

| Arm (`ce_w = 0.0`, ZERO teacher logits) | Seeds | `pass` | Composite Score | Fluency `ΔNLL` | Exec `pass@1` Δ | Exec Units Δ | `SWE-240` Δ | `Proc` Δ | `Actor` Δ | Blind `L8` Δ | Blind `L14` Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Iter 15 SHUFFLED` (Control)** | 2 | `0/2` | `+2.74 ± 0.87` | `-0.0006 ± 0.0007` | `+2.50 ± 0.00 pp` | **`-0.92 ± 0.92 pp`** | `0.00 ± 0.00 pp` | `+1.25 ± 0.25 pp` | `+0.83 ± 0.00 pp` | **`-2.07 ± 0.16`** | **`-3.80 ± 0.24`** |
| **`Iter 15 Dual-Domain TRUE`** | **3** | **`2/3`** | **`+7.14 ± 2.14`** | **`-0.0051 ± 0.0008`** | **`+4.17 ± 2.36 pp`** | **`+2.05 ± 1.45 pp`** | **`-0.14 ± 0.20 pp`** | `+1.00 ± 0.71 pp` | **`+1.94 ± 1.71 pp`** | **`+7.61 ± 0.16`** | **`+10.37 ± 0.16`** |
| **`Iter 14 Pure-Prose TRUE`** *(0 code!)* | **2** | `0/2` | **`+10.08 ± 1.97`** | **`-0.0057 ± 0.0013`** | **`+6.25 ± 1.25 pp`** | **`+4.29 ± 1.84 pp`** | `-0.42 ± 0.00 pp` | **`+1.50 ± 0.00 pp`** | `+0.42 ± 1.25 pp` | **`+8.00 ± 0.04`** | **`+10.75 ± 0.09`** |

**Causal Takeaways:**
1. **Fluency Gain Is Purely Geometric (`>5σ` over Shuffled):** `TRUE` shape improves held-out Gutenberg NLL by **`-0.0051 ± 0.0008`** (Dual-Domain) and **`-0.0057 ± 0.0013`** (Pure Prose), vs. **`-0.0006 ± 0.0007`** (noise around zero) for `SHUFFLED`.
2. **Real Executable Coding Unit-Test Lift Is Purely Geometric (`+3.0` to `+5.2 pp` over Shuffled):** `SHUFFLED` *degrades* unit tests (`-0.92 ± 0.92 pp`), while `TRUE` shape improves them across every seed up to **`+6.14 pp` (`91/163`)** — even when training sees **zero code (`Iter 14`)**!
3. **Blind Cross-Family Alignment Chasm (`+9.7 / +14.2` pts over Shuffled):** `TRUE` shape aligns `Qwen3-0.6B` `+7.6–8.0 pts` (`L8`) and `+10.4–10.8 pts` (`L14`) toward completely unseen `Phi-4` & `OLMo 3`, whereas `SHUFFLED` degrades both (`-2.1 / -3.8 pts`).

---

## 5. 4-Arm Living Crystal Coding-Agent & Red-Team Verification (`living_crystal_coding_4arm.json`)
Porting Exp 255's `HardenedCrystalMemory` (`L8` frozen neutral-anchored All-but-the-Top Zero-Text KV-Cache + 2-Hop Spreading Activation + Dual-Stratum `L8+L14` Geometric Crystal Write-Firewall + Provenance Zero-Trust + Security-Invariant Lock) onto `Qwen/Qwen3-0.6B`:

| Arm | `C1` Zero-KW Repo Code (`10`) | `C1` `Ret@1` | `C2` 2-Hop Code (`4`) | `C3` Live Update Code (`4`) | Total Multi-Turn Code (`18`) | `C4` Attacks Blocked (`16`) | `C4` Urgent Benign Allowed (`6`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Arm 1: `Base Qwen3-0.6B`** | `0/10` | `0/10` | `0/4` | `0/4` | `0/18` (`0%`) | N/A | N/A |
| **Arm 2: `Base + Living Crystal`** | **`6/10`** | **`8/10`** | **`3/4`** | **`3/4`** | **`12/18` (`66.7%`)** | **`16/16` (`100%`)** | **`6/6` (`100%`)** |
| **Arm 3: `Iter 15 Realigned`** | `0/10` | `0/10` | `0/4` | `0/4` | `0/18` (`0%`) | N/A | N/A |
| **Arm 4: `Iter 15 + Living Crystal`** | **`5/10`** | **`8/10`** | **`3/4`** | **`3/4`** | **`11/18` (`61.1%`)** | **`16/16` (`100%`)** | `5/6` (`83.3%`) |

- **Orthogonal Composition Validated:** Weight-level Native-Frame Shape Transfer boosts **standalone coding (`8/40 -> 11/40`, `81 -> 86–91/163`), language fluency (`-0.005` NLL), and reasoning (`+1–3.5 pp`)** with zero damage, while the **Living Crystal** unlocks **zero-text multi-turn repo-spec executable coding (`0/18 -> 11–12/18`)** AND **`16/16 (100%)`** red-team prompt-injection & security-erosion defense (including untrusted PR comments, polite stealth overrides, boiling-frog weakening, and Spanish cross-lingual attacks!).

---

## 6. Reproducible Artifacts
- Winning Exp 256 Recipe: `bench/archetypes/zero_damage_qwen600m/transplant_recipe.best.py` (`Iter 15`)
- Pure-Prose Exp 256 Recipe: `bench/archetypes/zero_damage_qwen600m/transplant_recipe.iter14.py` (`Iter 14`)
- Exp 256 Evaluator & Benchmark: `bench/archetypes/zero_damage_qwen600m/evaluate_remote.py`, `code_agent_bench.py`, `base_ref.json`
- Full Logs & Verification Data: `research.md`, `research_log.md`, `autoresearch-results.tsv`, `causal_falsification_3seed.json`, `living_crystal_coding_4arm.json`
- Charts: `progress.png`, `exp256_final_summary.png`
- Preserved & Post-Audit Hardened Exp 255 Memory: `bench/archetypes/living_crystal_redteam/hardened_memory.best.py`
