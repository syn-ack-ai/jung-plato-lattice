# arXiv Submission Package — *The Jung–Plato Lattice*

This directory contains the self-contained LaTeX source bundle for submitting **The Jung–Plato Lattice: Multi-Family Representation Denoising, Axial Coherence, and Cross-Tokenizer Distillation** to [arXiv.org](https://arxiv.org).

## Included Files
- `main.tex` — Two-column `IEEEtran` manuscript (`\pdfoutput=1` in first 5 lines for arXiv `pdflatex`, dual-compatible with `tectonic`)
- `IEEEtran.cls` — Standard IEEEtran class file (`V1.8b`)
- `references.bib` — All 53 verified BibTeX references
- `main.bbl` — Pre-compiled BibTeX bibliography (ensures deterministic citation rendering on arXiv AutoTeX)
- `figures/fig1_platonic_consensus.pdf` … `figures/fig6_epistemic_house_correctness.pdf` — Vector PDF figures generated directly from verified JSON artifacts
- `main.pdf` — Local preview PDF compiled via `tectonic` (`xdvipdfmx` backend; 0 undefined citations, 0 `Overfull \hbox` warnings; verify arXiv's `pdflatex` AutoTeX preview before final submission)
- `arxiv_submission.tar.gz` — Clean upload tarball containing ONLY the source files required by arXiv (`main.tex`, `references.bib`, `main.bbl`, `IEEEtran.cls`, `figures/fig*.pdf`)

## Suggested arXiv Metadata
- **Primary Category:** `cs.CL` (Computation and Language)
- **Cross-List Categories:** `cs.LG` (Machine Learning), `cs.AI` (Artificial Intelligence)
- **License:** CC BY 4.0 or arXiv.org perpetual, non-exclusive license
- **Comments:** 13 pages including references, 6 figures, 8 tables. Preprint DOI: https://doi.org/10.5281/zenodo.24572077. Code, verified JSON artifacts, and reproduction scripts are publicly available at https://github.com/syn-ack-ai/jung-plato-lattice.

## Ready-to-Paste Plain-Text arXiv Abstract (1,894 chars <= 1,920-char arXiv limit)
> We gave five independently trained LLM families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, and looped Inkling 2 Mini) the same public passages, recorded intermediate activations at matching clause endings, aligned them via 10-iteration Generalized Procrustes Analysis, and averaged them into a 256-dim consensus (C5). Calibrated on half of a sixth (3.8B Phi-4-mini) or seventh (7B OLMo 3) model's held-out passages, C5 predicts remaining activations by +10.4 to +11.3 Frobenius R^2 points above the strongest single teacher (R^2_opt 0.557/0.564), unchanged without the myth pool. Supervising one middle layer via a discarded linear head transfers C5 across unrelated tokenizers with zero inference overhead: across 24 5M runs, held-out Frobenius R^2 rises +15 to +23 points (mixed/null downstream gains; weighting the 2D HOUSE/SELF2 subspace lifts HOUSE cosine 0.200 to 0.619 while mcq_gain drops 0.172 to 0.119), and on 8B Gemma 4 E4B (1 paired seed), adding C5+U_jung lifts Phi-4/OLMo 3 Frobenius R^2 by +0.0245/+0.0235 vs. +0.0029/+0.0033 for a matched CE+KL control. Neutral-text rotations transfer Gemma 4's HOUSE coherence direction zero-shot into all six rival families at 97.4%-98.9% (beating a 95.5% BoW floor on 120/120 splits); 7-company valence decomposition shows 46.8% of HOUSE in C5 is general evaluative valence while the orthogonal 53.2% still transfers at 94.7%-98.1% (95.0%-100.0% on 60 valence-matched stories, where BoW also hits 100%). Disguised mythic plots show tight nulls and 58 named labels show no bug-fix alignment (p=0.81). In native 40%/60%-depth activations of Gemma 4 and Qwen3.5, CORRECTNESS + HOUSE_PERP clears nominal p<0.05 in 11/12 cells (6/12 Holm, 10/12 BH-FDR) where Aion prompting drops first-token entropy AUROC to 0.521-0.596; pure sentiment matches HOUSE under Aion prompting, while valence-orthogonal HOUSE hits 1.000 at Gemma L25 Neutral.

## Uploading to arXiv
1. Go to https://arxiv.org/submit and start a new submission.
2. Upload **`arxiv_submission.tar.gz`** directly (arXiv automatically unpacks it and detects `main.tex` + `\pdfoutput=1` + `main.bbl`).
3. Paste the plain-text abstract above into arXiv's metadata form and verify arXiv's `pdflatex` preview PDF matches `main.pdf`.

## Local Rebuild
```bash
# Using tectonic (macOS/Linux):
tectonic --keep-intermediates main.tex

# Or using standard TeX Live pdflatex + bibtex:
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
