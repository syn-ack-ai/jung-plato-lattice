# arXiv Submission Package — *The Jung–Plato Lattice*

This directory contains the self-contained LaTeX source bundle for submitting **The Jung–Plato Lattice: Multi-Family Representation Denoising, Compositional Probing, and Cross-Tokenizer Distillation** to [arXiv.org](https://arxiv.org).

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
- **Comments:** 15 pages including references, 6 figures, 12 tables. Preprint DOI: https://doi.org/10.5281/zenodo.24572076. Code, verified JSON artifacts, and reproduction scripts are publicly available at https://github.com/syn-ack-ai/jung-plato-lattice.

## Ready-to-Paste Plain-Text arXiv Abstract (1,883 chars <= 1,920-char arXiv limit)
> We gave five independent LLM families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, looped Inkling 2 Mini) identical public passages, recorded mid-layer clause-ending states, aligned them via 10-iteration Generalized Procrustes Analysis, and averaged them into a 256-dim consensus (C5). Calibrated on half of a sixth (Phi-4-mini) or seventh (OLMo 3) model's held-out passages, C5 predicts remaining states by +10.4 to +11.3 Frobenius R^2 points above the best single teacher (R^2_opt 0.557/0.564), unchanged without the myth pool. Supervising one middle layer via a discarded head transfers C5 across tokenizers with zero inference overhead: across 24 5M runs, R^2_frob rises +15 to +23 points (mixed/null downstream; weighting U_jung lifts HOUSE cos 0.200->0.619 while mcq_gain drops 0.172->0.119), and on 8B Gemma 4 E4B (1 paired seed), C5+U_jung lifts Phi-4/OLMo 3 R^2_frob by +0.0245/+0.0235 vs. +0.0029/+0.0033 for CE+KL control. Neutral-text rotations transfer HOUSE across 6 rivals at 97.4%-98.9% (120/120 vs. 95.5% BoW); 46.8% of HOUSE is valence. On 120 order-mirrored identical-BoW swaps (BoW 50.0%, bigram 49.1%), setting-disjoint 2-fold transfer perp VAL2 hits 67.8%-77.0% mean (6/6 Holm; 42.2%-58.8% both-mirrors vs. 25% chance) and 60.5%-75.3% last-token (5/6 Holm), zero-shot HOUSE_perp hits 61.7%-67.5% (6/6 Holm), and OLS intercepts at equal output prob (Delta NLL=0) stay positive in 5/5 MLX targets. Eight Jungian roles (85.9%-92.7%) trail 8 workplace roles (94.3%-98.1%; PR n.s. vs. shuffled labels, p=0.116/0.108). Zero-PN register holds on OLMo 3 (+0.045/+0.037, reversing on Phi-4); disguised plots stay null. In native 40%/60% states of Gemma 4 and Qwen3.5, CORRECTNESS+HOUSE_PERP clears p<0.05 in 11/12 cells (6/12 Holm) where Aion drops entropy AUROC to 0.521-0.596; sentiment matches HOUSE under Aion while HOUSE_perp hits 1.000 at Gemma L25 Neutral.

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
