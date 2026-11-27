# Third-Party Licenses, Attribution, and Data Provenance

This repository's original source code, authored benchmarks, analysis scripts, and experimental result logs are released under the **MIT License** (see [LICENSE](LICENSE)). Third-party datasets, corpora, LaTeX class files, and open-weight models included or referenced here remain subject to their respective upstream licenses and attribution notices documented below.

---

## 1. OpenAssistant OASST2 (`passages.jsonl` and Evaluation Prompts)

- **Included artifacts:**
  - `bench/archetypes/fluent/plato/passages.jsonl` lines 1800–2398 (599 dialogue excerpts, `domain: "chat"`, `id: "chat_0000"`–`"chat_0598"`).
  - Additional held-out conversational evaluation excerpts in `bench/archetypes/fluent/comp/passages.jsonl` (if sampled from OASST2 by `bench/archetypes/fluent/plato_data.py`).
- **Upstream source:** OpenAssistant Conversations Dataset Release 2 (`OpenAssistant/oasst2`), Hugging Face Hub: <https://huggingface.co/datasets/OpenAssistant/oasst2>.
- **License:** **Apache License, Version 2.0** (full text redistributed in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)).
- **Transformations:** English turns were filtered to 50–195 words by `bench/archetypes/fluent/plato_data.py` (observed range in the 599 released excerpts: 50–191 words), normalized for whitespace, and segmented at sentence/clause boundaries (`segs`) to form the neutral dialogue alignment pool. No upstream endorsement is implied.

---

## 2. Linguistic & Commonsense Evaluation Benchmarks (`bench/archetypes/fluent/comp/`)

- **BLiMP (`comp/blimp.jsonl`):** Sampled minimal pairs from the Benchmark of Linguistic Minimal Pairs (Warstadt et al., 2020, *TACL* 8, 377–392; <https://github.com/alexwarstadt/blimp>). Upstream license: **CC BY 4.0**.
- **COPA (`comp/copa.jsonl`):** Choice of Plausible Alternatives items (Roemmele, Bejan, & Gordon, 2011; <https://people.ict.usc.edu/~gordon/copa.html>). Distributed here solely for non-commercial academic evaluation parity alongside the 5M distillation harness.
- **Additional synthetic & Gutenberg fixtures (`comp/mcq*.jsonl`, `comp/lastword.jsonl`, `comp/passages.jsonl`):** Generated from public-domain Project Gutenberg prose and synthetic procedural templates in `bench/archetypes/fluent/`.

---

## 3. Project Gutenberg Public-Domain Works (`passages.jsonl`, `gut_*` and `myth_*`)

- **Included artifacts:** `bench/archetypes/fluent/plato/passages.jsonl` (`domain: "gut"` and `domain: "myth"`) and `eval_passages.jsonl` (`held_gut`).
- **Upstream source:** Public-domain 19th- and early-20th-century literary and folklore editions from Project Gutenberg (<https://www.gutenberg.org/>), sampled independently by `bench/archetypes/fluent/plato_data.py` (excluding 200 held-out Gutenberg evaluation books and Jung Gutenberg IDs `{48225, 65903, 66041}`).
- **Note on historical folktale overlap:** Zero text from `j-hagedorn/trilogy`'s `aft.csv` compilation is redistributed in this repository. Across all 3,598 passages in `passages.jsonl`, the neutral `gut+proc+chat` subset actually used for Claim C's `neutral_only=True` rotations has **0/1,518** $\ge 12$-word overlaps with all rows of `aft.csv` (maximum 7 contiguous words against the 500 scored excerpts). In the independent Project Gutenberg `myth` pool, scored excerpts share at most 10 contiguous words, while 7 `myth` passages share $\ge 12$ words with full-tale rows in `aft.csv` (6 longer 21–126-word overlaps from shared 19th-century editions plus 1 14-word proverbial match `myth_0780` / `aft_idx 435`; 3 of the 7 belong to the 480 set: `aft_idx` 109, 1464, 1494). Full overlap diagnostics and `n=477` / `n=473` exclusion sensitivities are recorded in `bench/archetypes/fluent/plato/v6_round8_overlap_audit.json`.

---

## 4. Annotated Folktales Corpus (`j-hagedorn/trilogy`, `data/aft.csv`) — **NOT Redistributed**

- **Upstream source:** Joshua Hagedorn and Sándor Darányi (2022), *"Bearing a Bag-of-Tales: An Open Corpus of Annotated Folktales for Reproducible Research,"* *Journal of Open Humanities Data*, 8:16, pp. 1–10 (<https://doi.org/10.5334/johd.78>); GitHub repository `j-hagedorn/trilogy` (<https://github.com/j-hagedorn/trilogy>), `data/aft.csv` (pinned commit `a6ad5f30aac63f996de56b74c1f42d01c834dd35`), initially compiled with permission from D. L. Ashliman's *Folklore and Mythology Electronic Texts* (<https://sites.pitt.edu/~dash/folktexts.html>).
- **Upstream license note:** The `j-hagedorn/trilogy` repository's `LICENSE.md` specifies **CC BY-NC-SA 4.0**, whereas its `README.md` and Hagedorn & Darányi (2022) cite **CC BY-SA 4.0**; additionally, D. L. Ashliman's anthology notes copyright on his own translations and edited texts (distinct from traditional folktale plots themselves).
- **Conservative fetch-only policy:** Out of respect for both upstream share-alike/non-commercial terms and translator copyrights, **zero tale text from `aft.csv` is committed here**. Only non-text adjudication metadata (`bench/archetypes/selfshape/aft_clean248_manifest.json`: row indices `aft_idx`, ATU type IDs, division labels, provenance flags, inclusion flags, and per-excerpt SHA-256 checksums/word counts — carved out from original prose claims as factual/structural metadata alongside upstream ATU annotations) and numeric outputs are committed. Use `scripts/v6/fetch_trilogy_aft.py` to fetch `data/aft.csv` directly from upstream locally under upstream terms.

---

## 5. LaTeX Class File (`paper/arxiv/IEEEtran.cls`)

- **Included artifact:** `paper/arxiv/IEEEtran.cls` (V1.8b, 2015/08/26, Michael Shell; bundled in `paper/arxiv/` and `arxiv_submission.tar.gz` for deterministic arXiv compilation).
- **Upstream source:** CTAN `IEEEtran` package (<https://ctan.org/pkg/IEEEtran>).
- **License:** **LaTeX Project Public License (LPPL), Version 1.3** or later (<https://www.latex-project.org/lppl/lppl-1-3/>).

---

## 6. Open-Weight Language Models

All evaluated open-weight checkpoints are downloaded directly from Hugging Face Hub under their respective model licenses; no model weights are redistributed in this repository:
- `google/gemma-4-E4B-it` (bf16 HF for 8B transplants; MLX 4-bit / bf16 for extraction)
- `Qwen/Qwen3.5-9B` (MLX 8-bit; Hybrid Gated DeltaNet + Attention, multimodal image-text-to-text)
- `meta-llama/Llama-3.1-8B-Instruct` (MLX 4-bit)
- `mistralai/Mistral-7B-Instruct-v0.3` (MLX 4-bit)
- `microsoft/Phi-4-mini-instruct` (MLX 4-bit)
- `allenai/Olmo-3-7B-Instruct` (MLX 4-bit)
- `thinkingmachines/Inkling-2-Mini` (<https://huggingface.co/thinkingmachines/Inkling-2-Mini>, `IQ4_NL` GGUF served via `llama-server`; looped architecture, multimodal image-text-to-text)
