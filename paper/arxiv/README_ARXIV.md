# arXiv Submission Package — *The Jung–Plato Lattice* (`v6` / Zenodo v3.0)

This directory contains the self-contained LaTeX source bundle for submitting **The Jung–Plato Lattice: Multi-Family Representation Denoising, Compositional Probing, and Cross-Tokenizer Distillation** to [arXiv.org](https://arxiv.org), formatted strictly per the IEEEtran arXiv layout and rhythm guidelines.

## Included Files
- `main.tex` — Two-column `IEEEtran` manuscript (`\pdfoutput=1` in first 5 lines for arXiv `pdflatex`, dual-compatible with `tectonic`, `\AddToHook{env/thebibliography/begin}{\label{ReferencesStart}}`, zero `\resizebox`, 2 native vector TikZ diagrams, 0 undefined citations, 0 `Overfull \hbox` warnings)
- `IEEEtran.cls` — Standard IEEEtran class file (`V1.8b`)
- `references.bib` — All 64 verified BibTeX references
- `main.bbl` — Pre-compiled BibTeX bibliography (ensures deterministic citation rendering on arXiv AutoTeX)
- `figures/fig1_platonic_consensus.pdf` … `figures/fig9_historical_folklore_audit.pdf` — All 9 JSON-driven vector PDF figures
- `main.pdf` — Local preview PDF compiled via `tectonic` (`xdvipdfmx` backend; 0 undefined citations, 0 `Overfull \hbox` warnings; verify arXiv's `pdflatex` preview before final submission)
- `arxiv_submission.tar.gz` — Clean upload tarball containing ONLY the source files required by arXiv (`main.tex`, `references.bib`, `main.bbl`, `IEEEtran.cls`, `figures/fig*.pdf`)

## Suggested arXiv Metadata
- **Primary Category:** `cs.CL` (Computation and Language)
- **Cross-List Categories:** `cs.LG` (Machine Learning), `cs.AI` (Artificial Intelligence)
- **License:** CC BY 4.0 or arXiv.org perpetual, non-exclusive license
- **Comments:** 18 pages including references, 11 figures (9 JSON-driven vector PDFs + 2 vector TikZ diagrams), 15 tables. Preprint DOI: https://doi.org/10.5281/zenodo.24572076. Code, verified JSON artifacts, and reproduction scripts are publicly available at https://github.com/syn-ack-ai/jung-plato-lattice.

## Ready-to-Paste Plain-Text arXiv Abstract (1,781 chars <= 1,920-char arXiv limit)
> Independently trained language models use different tokenizers, corpora, and architectures, yet their intermediate hidden states converge toward a shared geometry. Recording activations at identical clause endings across five model families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, and looped Inkling 2 Mini) and aligning them via 10-iteration Generalized Procrustes Analysis yields a 256-dimensional consensus (C5) that predicts two unseen families (Phi-4-mini, OLMo 3 7B) by +10.4 to +11.3 Frobenius R^2 points above the strongest single teacher. Supervising one intermediate layer via a discarded linear head transfers this geometry across tokenizers with zero extra inference parameters at 5M scale (+15 to +23 R^2_frob points, mixed downstream scores) and on 8B multimodal Gemma 4 E4B. Crucially, cross-family compositional binding peaks at 50% layer depth on a 240-pair actor-intent swap benchmark where all lexical and polarity baselines equal 50.0% (73.0%/73.1% mean-pool transfer, clearing cross-pool Bonferroni-8). Transplanting a strictly rival-only 50%-depth consensus (C3_50, excluding Gemma 4) onto Gemma 4 E4B layers 19-20 lifts blind Phi-4 and OLMo 3 R^2_frob by +0.0974 and +0.0887 (11.5x and 10.1x a matched CE+KL control, zero layer-12 drift, multimodal fixtures intact) while leaving unlabeled downstream probing unchanged. On 480 cleaned historical Aarne-Thompson-Uther folktales, Jungian labels beat within-setting shuffles (p_holm = 0.0417) but do not beat unlabeled corpus PCA, and three follow-up probes fail strict audits and are dropped. Framed through Jung's crystal analogy, disembodied language models lack a psyche and instead capture a mid-depth statistical projection of culturally transmitted narrative imagery.

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
