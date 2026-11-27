# arXiv Submission Package — *The Jung–Plato Lattice* (`v6` / Zenodo v3.0)

This directory contains the self-contained LaTeX source bundle for submitting **The Jung–Plato Lattice: Multi-Family Representation Denoising, Compositional Probing, and Cross-Tokenizer Distillation** to [arXiv.org](https://arxiv.org), formatted in two-column `IEEEtran` layout.

## Included Files
- `main.tex` — Two-column `IEEEtran` manuscript (`\pdfoutput=1` in first 5 lines for arXiv `pdflatex`, pure ASCII, dual-compatible with `tectonic` via `\ifx\XeTeXversion\undefined` `fontspec` branch, `\AddToHook{env/thebibliography/begin}{\label{ReferencesStart}}`, zero `\resizebox`, 2 native vector TikZ diagrams, 0 undefined citations, 0 `Overfull \hbox` warnings under local `tectonic`)
- `IEEEtran.cls` — Pure-ASCII `IEEEtran` class file (`V1.8b`, LPPL 1.3; verified ASCII-clean)
- `references.bib` — All 82 verified BibTeX references
- `main.bbl` — Pre-compiled BibTeX bibliography (ensures deterministic citation rendering on arXiv AutoTeX)
- `figures/fig1_platonic_consensus.pdf` … `figures/fig9_historical_folklore_audit.pdf` — All 9 JSON-driven vector PDF figures
- `main.pdf` — Local preview PDF compiled via `tectonic` (22 pages; TeX Live bundled `TeX Gyre Termes`, `TeX Gyre Heros`, and `TeX Gyre Cursor` via `fontspec` so bold, italic, bold-italic, small caps, and monospace embed cleanly with 0 undefined citations, 0 `Overfull \hbox` warnings, and 0 document font warnings; **always verify arXiv AutoTeX's `pdflatex` preview before final submission** because metric/font wrapping can differ slightly between XeTeX and `pdflatex`)
- `arxiv_submission.tar.gz` — Clean 13-file upload tarball containing ONLY the source files required by arXiv (`main.tex`, `references.bib`, `main.bbl`, `IEEEtran.cls`, `figures/fig*.pdf`; zero macOS `._*` AppleDouble sidecars, `uid=0/gid=0`, empty owner/group, verified SHA-256 parity in `scripts/verify_repro.py`)

## Suggested arXiv Metadata
- **Primary Category:** `cs.CL` (Computation and Language)
- **Cross-List Categories:** `cs.LG` (Machine Learning), `cs.AI` (Artificial Intelligence)
- **License:** CC BY 4.0 or arXiv.org perpetual, non-exclusive license
- **Comments:** 22 pages including references (local tectonic preview), 11 figures (9 JSON-driven vector PDFs + 2 vector TikZ diagrams), 17 tables (15 numeric tables + Rosetta Stone + model table). Preprint DOI: https://doi.org/10.5281/zenodo.24572076. Code, verified JSON artifacts, sanitized v6 runners, dual-audience Word review copy, and reproduction scripts are publicly available at https://github.com/syn-ack-ai/jung-plato-lattice.

## Ready-to-Paste Plain-Text arXiv Abstract (1,834 characters, 263 words; under arXiv's 1,920-character limit)
> Independently trained language models use different tokenizers, datasets, and architectures, yet their middle-layer activations align across families. Recording clause-ending states across five open-weight families (Gemma 4 E4B, Qwen3.5 9B, Llama 3.1 8B, Mistral 7B, and looped Inkling 2 Mini) and aligning them via 10-iteration Generalized Procrustes Analysis yields a 256-dimensional consensus (C5). On two excluded target families (Phi-4-mini, OLMo 3 7B), C5 predicts held-out activations by +8.3 and +7.8 strict R^2_opt points (+11.3 and +10.4 Frobenius R^2_frob points) above every teacher. Supervising one middle layer via a discarded linear head transfers this geometry across tokenizers with zero added inference parameters at 5M scale (+15 to +23 R^2_frob points, mixed downstream scores) and on 8B multimodal Gemma 4 E4B. On 240 actor-intent swaps where seven fitted linear lexical and polarity baselines sit at 50.0% (though an actor-lexicon times polarity rule hits 95.5%-96.3%), mean-pooled cross-family readout rises from 30% to its observed peak at 50% depth (73.0% Gemma / 73.1% Qwen overall, 64.4% / 58.3% adverse, 26.0% / 28.2% all-four-right vs. 6.25% chance), while single-layer steering gives a null effect. Transplanting rival-only 50%-depth C3_50 (zero Gemma 4) into Gemma 4 E4B layers 19-20 across three paired seeds lifts held-out-family alignment over matched control (+0.0869 Phi-4, +0.0780 OLMo 3 R^2_frob) without improving probing or MCQ. On 480 cleaned historical folktales, predicted archetype-to-division pairings fail (cos = -0.106, p = 0.952) and unlabeled PCA matches or beats study-selected Jungian labels, though labels slightly exceed shuffles (p_holm = 0.0417). Making no claim about machine consciousness or psyche, we study aligned subspaces as statistical projections of human cultural data.

## Local Rebuild
```bash
# Using tectonic (macOS/Linux):
tectonic --keep-intermediates --keep-logs main.tex

# Or using standard TeX Live pdflatex + bibtex (matches arXiv AutoTeX):
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
