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
- **Comments:** 12 pages including references, 6 figures, 7 tables. Preprint DOI: https://doi.org/10.5281/zenodo.24572077. Code, verified JSON artifacts, and reproduction scripts are publicly available at https://github.com/syn-ack-ai/jung-plato-lattice.

## Ready-to-Paste Plain-Text arXiv Abstract (1,663 chars <= 1,920-char arXiv limit)
> We gave five independently trained language model families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, and looped Inkling 2 Mini) the same public passages, recorded intermediate hidden states at matching clause endings, rotated their coordinate frames into alignment via 10-iteration Generalized Procrustes Analysis, and averaged them into a 256-dimensional consensus (C5). After calibrating a linear rotation on half of a sixth (3.8B Phi-4-mini) or seventh (7B OLMo 3) model's held-out passages, C5 predicts that model's remaining activations by +10.4 to +11.3 Frobenius R^2 points above the strongest single teacher (R^2_opt 0.557/0.564), holding unchanged when the dedicated myth pool is excluded. Supervising one intermediate layer via a temporary linear head discarded after training transfers this consensus across unrelated tokenizers with zero extra inference parameters: across 24 runs at 5M scale, held-out Frobenius R^2 jumps +15 to +23 points with zero seed overlap, and weighting the 2D HOUSE/SELF2 subspace raises student-to-teacher HOUSE alignment from 0.200 to 0.619 while reading-comprehension gain drops from 0.172 to 0.119. On 8B multimodal Gemma 4 E4B (1 paired seed), adding C5+U_jung lifts Phi-4 and OLMo 3 Frobenius R^2 by +0.0245 and +0.0235 versus +0.0029 and +0.0033 for a matched CE+KL control (Gutenberg NLL +0.0027). Neutral-text rotations transfer Gemma 4's HOUSE narrative-coherence direction zero-shot into all six rival families at 97.4%-98.9% across 20 setting-disjoint splits (beating a 95.5% bag-of-words floor on 120/120 splits), whereas 58 named labels show no above-chance bug-fix alignment and disguised mythic plots collapse once edit position is matched. Finally, in native 40%/60%-depth activations of Gemma 4 and Qwen3.5, combining factual correctness with Gram-Schmidt orthogonalized HOUSE_PERP clears nominal p<0.05 in 11/12 matched cells (6/12 Holm, 10/12 BH-FDR) where conversational prompting lowers first-token entropy AUROC to 0.521-0.596.

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
