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
- **Comments:** 14 pages including references, 6 figures, 10 tables. Preprint DOI: https://doi.org/10.5281/zenodo.24572077. Code, verified JSON artifacts, and reproduction scripts are publicly available at https://github.com/syn-ack-ai/jung-plato-lattice.

## Ready-to-Paste Plain-Text arXiv Abstract (1,881 chars <= 1,920-char arXiv limit)
> We gave five independent LLM families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, looped Inkling 2 Mini) identical public passages, recorded mid-layer clause-ending states, aligned them via 10-iteration Generalized Procrustes Analysis, and averaged them into a 256-dim consensus (C5). Calibrated on half of a sixth (Phi-4-mini) or seventh (OLMo 3) model's held-out passages, C5 predicts remaining activations by +10.4 to +11.3 Frobenius R^2 points above the best single teacher (R^2_opt 0.557/0.564), unchanged without the myth pool. Supervising one middle layer via a discarded head transfers C5 across unrelated tokenizers with zero inference overhead: across 24 5M runs, held-out R^2_frob rises +15 to +23 points (mixed/null downstream gains; weighting U_jung lifts HOUSE cos 0.200->0.619 while mcq_gain drops 0.172->0.119), and on 8B Gemma 4 E4B (1 paired seed), C5+U_jung lifts Phi-4/OLMo 3 R^2_frob by +0.0245/+0.0235 vs. +0.0029/+0.0033 for a matched CE+KL control. Neutral-text rotations transfer Gemma 4's HOUSE direction into all 6 rivals at 97.4%-98.9% (120/120 vs. 95.5% BoW); 46.8% of HOUSE is valence, yet on 60 exact identical-BoW swaps (BoW 50.0%, bigram 49.3%), 2-fold transfer hits 69.6%-79.3% raw and 65.7%-73.7% after removing 2D valence (5/6 p<0.05, 120/120 wins; HOUSE_perp->bs60 is 58.3%-68.3%). Moreover, 8 canonical named Jungian archetypes (40 shared settings x 8 = 320 stories) form a ~5D valence-orthogonal subspace in C5 transferring at p=0.001 in all 48/48 cells (8-way Top-1 85.9%-92.7% vs. BoW 80.6% / TF-IDF Ridge 89.1%). Disguised mythic plots stay null. In native 40%/60% states of Gemma 4 and Qwen3.5, CORRECTNESS+HOUSE_PERP clears p<0.05 in 11/12 cells (6/12 Holm) where Aion prompting lowers entropy AUROC to 0.521-0.596; pure sentiment matches HOUSE under Aion prompting while valence-orthogonal HOUSE hits 1.000 at Gemma L25 Neutral.

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
