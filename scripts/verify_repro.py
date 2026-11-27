"""One-command zero-GPU reproducibility & integrity verifier for The Jung–Plato Lattice.

Checks:
  1. Privacy & secret firewall across all tracked repository files (NUL-delimited `git ls-files -z`).
  2. Syntax check (py_compile) across all Python modules.
  3. Exact presence of all 6 publication figures (PNG + PDF in paper/figures/ and PDF in paper/arxiv/figures/),
     committed spoken-audio fixture, 53 verified references across paper.md, references.bib, and main.bbl,
     and arXiv LaTeX hygiene (\\pdfoutput=1 header and zero unescaped ~digits).
  4. Strict ordered, typed cell-level cross-verification of committed JSON artifacts against EVERY numeric
     table in BOTH `paper/paper.md` (Tables 1, pairwise matrix, 2, 2a, 2b, 2c, 2d, 2e, 3, 3b, 3c, 4, 5) AND
     `paper/arxiv/main.tex` (Tables tab:gpa, tab:teleport, tab:val, tab:bswap, tab:named, tab:loro, tab:5m, tab:8b, tab:epi).
  5. Comprehensive zero-GPU scientific unit tests with fully isolated temporary `JPL_CKPT_DIR` (explicitly
     reporting any optional tests skipped when optional packages are not installed).
"""
import glob
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "bench" / "archetypes"

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".aion", "local_data", "models",
    "local_ckpt", "local_eval",
}

BINARY_EXTS = {".png", ".pdf", ".npz", ".npy", ".bin", ".pt", ".wav", ".cls", ".gz", ".tar"}


def _is_skipped(p: Path) -> bool:
    # Match directory components ONLY (never skip a tracked root/nested file such as `.env`!)
    parent_parts = p.relative_to(ROOT).parts[:-1]
    return bool(SKIP_DIRS.intersection(parent_parts))


def check_privacy():
    # Generic secret & local-machine path firewall (zero personal identifiers embedded in source!)
    rules = [
        ("private_data_dir", re.compile(r"private_data", re.I)),
        ("local_home_path", re.compile(r"(?:/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+)")),
        ("api_token", re.compile(r"(?:gh[po]_[A-Za-z0-9]{10}|github_pat_[A-Za-z0-9_]{10}|hf_[A-Za-z0-9]{10}|sk-[A-Za-z0-9]{20})")),
        ("rfc1918_ip", re.compile(r"(?:\b192\.168\.\d+\.\d+|\b10\.\d+\.\d+\.\d+|\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)")),
        ("private_key", re.compile(r"BEGIN\s+(?:[A-Z0-9]+\s+)?PRIVATE\s+KEY", re.I)),
    ]
    # Use NUL-delimited `git ls-files -z` so unusual filenames are never mangled and NO tracked file is skipped!
    git_res = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT), capture_output=True)
    if git_res.returncode == 0 and git_res.stdout:
        candidates = [ROOT / os.fsdecode(part) for part in git_res.stdout.split(b"\0") if part]
        using_git = True
    else:
        candidates = [p for p in ROOT.rglob("*") if p.is_file() and not _is_skipped(p)]
        using_git = False

    hits = []
    self_path = Path(__file__).resolve()
    for p in candidates:
        if not p.is_file() or p.resolve() == self_path:
            continue
        if not using_git and _is_skipped(p):
            continue
        if p.suffix.lower() in BINARY_EXTS:
            continue
        txt = p.read_text(errors="ignore")
        matched_rules = [rname for rname, pat in rules if pat.search(txt)]
        if matched_rules:
            hits.append((str(p.relative_to(ROOT)), matched_rules))
    # Never echo matched credential/path fragments in failure output — report filename + rule names only!
    assert not hits, f"PRIVACY/SECRET FIREWALL FAILED (files & triggered rules): {hits}"
    print(f"[PASS] Privacy & secret firewall clean across all {len(candidates)} {'tracked' if using_git else 'scanned'} files.")


def check_py_compile():
    py_files = sorted(p for p in ROOT.rglob("*.py") if not _is_skipped(p))
    for pf in py_files:
        py_compile.compile(str(pf), doraise=True)
    print(f"[PASS] py_compile succeeded on all {len(py_files)} Python files.")


def check_paper_and_figures():
    paper = ROOT / "paper" / "paper.md"
    arxiv_dir = ROOT / "paper" / "arxiv"
    tex_path = arxiv_dir / "main.tex"
    bib_path = arxiv_dir / "references.bib"
    bbl_path = arxiv_dir / "main.bbl"

    txt = paper.read_text()
    tex = tex_path.read_text()
    bib = bib_path.read_text()
    bbl = bbl_path.read_text()

    figs = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", txt)
    assert len(figs) == 6, f"Expected 6 embedded figures, found {len(figs)}"
    for f in figs:
        png = paper.parent / f
        pdf = png.with_suffix(".pdf")
        arxiv_pdf = arxiv_dir / "figures" / pdf.name
        assert png.exists() and png.stat().st_size > 10_000, f"Missing PNG: {png}"
        assert pdf.exists() and pdf.stat().st_size > 5_000, f"Missing PDF: {pdf}"
        assert arxiv_pdf.exists() and arxiv_pdf.stat().st_size > 5_000, f"Missing arXiv PDF: {arxiv_pdf}"

    refs_md = [l for l in txt.split("## References", 1)[1].splitlines() if l.startswith("- ")]
    refs_bib = re.findall(r"^@[a-zA-Z]+\{", bib, flags=re.M)
    refs_bbl = re.findall(r"^\\bibitem\{", bbl, flags=re.M)
    assert len(refs_md) == 53, f"Expected 53 references in paper.md, found {len(refs_md)}"
    assert len(refs_bib) == 53, f"Expected 53 BibTeX entries in references.bib, found {len(refs_bib)}"
    assert len(refs_bbl) == 53, f"Expected 53 \\bibitem entries in main.bbl, found {len(refs_bbl)}"

    # arXiv LaTeX header & formatting hygiene checks
    top5 = "\n".join(tex.splitlines()[:5])
    assert r"\pdfoutput=1" in top5, "paper/arxiv/main.tex missing \\pdfoutput=1 in first 5 lines!"
    unescaped_tilde = re.findall(r"(?<!\$)(?<!\\sim)(?<!\\textasciitilde)~\d", tex)
    assert not unescaped_tilde, f"Unescaped ~digits found in main.tex (renders as blank space): {unescaped_tilde}"

    wav = ARCH / "aion_8b" / "assets" / "test_speech_16k.wav"
    assert wav.exists() and wav.stat().st_size > 10_000, f"Missing committed spoken-audio fixture: {wav}"

    # Guard against previously caught stale prose & param counts!
    for label, doc in [("paper.md", txt), ("main.tex", tex)]:
        assert "+0.0569" not in doc, f"Stale pre-audit number +0.0569 found in {label}!"
        assert "8,916,480" not in doc, f"Stale student total param count 8,916,480 found in {label}!"
        assert "8,916,608" in doc, f"Exact student total param count 8,916,608 missing from {label}!"

    print(f"[PASS] Paper & arXiv verified: 6 figures (PNG+PDF+arXiv PDF), {len(refs_md)}/{len(refs_bib)}/{len(refs_bbl)} refs (md/bib/bbl), speech WAV fixture, clean LaTeX.")


def _parse_md_tables(paper_txt: str):
    """Return list of markdown table rows as lists of stripped cell strings."""
    rows = []
    for line in paper_txt.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and not re.match(r"^\|[\s:\-|]+\|$", s):
            cells = [c.strip() for c in s.strip("|").split("|")]
            rows.append(cells)
    return rows


def _parse_md_table_after(paper_txt: str, header_needle: str):
    """Return rows of the single Markdown table whose header line contains `header_needle`."""
    lines = paper_txt.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and header_needle in s:
            rows = []
            for j in range(i + 1, len(lines)):
                sj = lines[j].strip()
                if not (sj.startswith("|") and sj.endswith("|")):
                    break
                if re.match(r"^\|[\s:\-|]+\|$", sj):
                    continue
                rows.append([c.strip() for c in sj.strip("|").split("|")])
            return rows
    raise AssertionError(f"Markdown table with header '{header_needle}' not found!")


def _clean_md(s: str) -> str:
    s = s.replace("**", "").replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip()


def _clean_tex(s: str) -> str:
    # Normalize LaTeX formatting macros into plain ordered tokens while preserving values/punctuation
    s = s.replace(r"\%", "%").replace(r"\_", "_").replace(r"\to", "->").replace(r"\times", "x")
    s = s.replace("---", "-").replace(r"$-$", "-").replace("--", "-").replace(r"\,", " ").replace("~", " ")
    # Strip formatting wrappers repeatedly (handles nested \textbf{\texttt{...}})
    for _ in range(4):
        s = re.sub(r"\\(?:textbf|texttt|textit|emph|text)\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _parse_tex_table(tex_txt: str, label: str):
    """Extract cleaned data rows (between \\midrule and \\bottomrule) from a LaTeX table with \\label{label}."""
    blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex_txt, flags=re.S)
    matches = [b for b in blocks if f"\\label{{{label}}}" in b]
    assert len(matches) == 1, f"LaTeX table \\label{{{label}}} expected 1 match, found {len(matches)}"
    block = matches[0]
    bm = re.search(r"\\midrule(.*?)\\bottomrule", block, flags=re.S)
    assert bm, f"LaTeX table \\label{{{label}}} missing \\midrule...\\bottomrule"
    rows = []
    for raw_line in bm.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith(r"\midrule") or line.startswith(r"\addlinespace"):
            continue
        line = re.sub(r"\\\\\s*(?:\[[^\]]*\])?\s*$", "", line)
        cells = [_clean_tex(c) for c in line.split("&")]
        rows.append(cells)
    return rows


def _fmt_pair_pct(v: float) -> str:
    pct = v * 100.0
    return "100%" if abs(pct - 100.0) < 0.05 else (f"{pct:.0f}%" if abs(pct - round(pct)) < 0.05 else f"{pct:.1f}%")


def _assert_row_regex(rows, row_key, col_regexes, label, exact_cols=None, min_cols=2, cleaner=_clean_md):
    matches = [
        r for r in rows
        if (len(r) == exact_cols if exact_cols is not None else len(r) >= min_cols)
        and row_key in cleaner(r[0])
    ]
    assert len(matches) == 1, f"Table check '{label}': expected 1 row matching '{row_key}', found {len(matches)}"
    row = [cleaner(c) for c in matches[0]]
    for col_idx, pat in col_regexes:
        assert col_idx < len(row), f"Table check '{label}': col {col_idx} out of range (len={len(row)})"
        assert re.match(pat, row[col_idx]), (
            f"Table check '{label}' col[{col_idx}] ordered mismatch:\n  got:      '{row[col_idx]}'\n  expected: /{pat}/"
        )


def check_headline_numbers_and_paper_sync():
    paper_txt = (ROOT / "paper" / "paper.md").read_text()
    tex_txt = (ROOT / "paper" / "arxiv" / "main.tex").read_text()
    md_rows = _parse_md_tables(paper_txt)

    def assert_in_doc(doc, substr, label):
        assert substr in doc, f"Document missing verified value for {label}: expected '{substr}'"

    # =========================================================================
    # 1. Unified 7-company GPA consensus + neutral-only audit (Table 1, Pairwise, Table 2, Table 2a, tab:gpa)
    # =========================================================================
    meta = json.load(open(ARCH / "fluent" / "plato" / "meta.json"))
    aud = json.load(open(ARCH / "fluent" / "plato" / "audit233_mac.json"))
    pf = aud["plato_full_tiesafe"]
    pn = aud["plato_neutral_only_no_myth"]
    assert pf["n_train_pts"] == 20339 and pn["n_train_pts"] == 13388
    assert abs(meta["to_blind_pts"]["phi4_mini"]["CONSENSUS_5FAM_INK"]["r2_frob"] - pf["blind"]["phi4_mini"]["C5"]["r2_frob"]) < 1e-4
    assert abs(meta["to_blind_pts"]["olmo3_7b"]["CONSENSUS_5FAM_INK"]["r2_frob"] - pf["blind"]["olmo3_7b"]["C5"]["r2_frob"]) < 1e-4

    t1_rows = [
        ("gemma4_e4b", "Gemma 4 E4B (Google"),
        ("llama31_8b", "Llama 3.1 8B (Meta"),
        ("mistral7b", "Mistral 7B (Mistral"),
        ("qwen35_9b", "Qwen3.5 9B (Alibaba"),
        ("inkling2_mini", "Inkling 2 Mini"),
        ("C4", "4-Family Consensus (C4"),
        ("C5", "5-Family Consensus (C5"),
    ]
    md_t1_rows = _parse_md_table_after(paper_txt, "Source Representation (256-dim PCA / GPA)")
    tex_gpa = _parse_tex_table(tex_txt, "tab:gpa")
    assert len(tex_gpa) == 7, f"Expected 7 rows in tab:gpa, got {len(tex_gpa)}"
    for k, row_key in t1_rows:
        p4 = pf["blind"]["phi4_mini"][k]
        ol = pf["blind"]["olmo3_7b"][k]
        p4_cos, p4_r2o, p4_r2f = f"{p4['cos']:.4f}", f"{p4['r2_opt_train_scalar']:.3f}", f"{p4['r2_frob']:.3f}"
        p4_t1, p4_t5 = f"{p4['top1_tiesafe']*100:.1f}%", f"{p4['top5_tiesafe']*100:.1f}%"
        ol_cos, ol_r2o, ol_r2f = f"{ol['cos']:.4f}", f"{ol['r2_opt_train_scalar']:.3f}", f"{ol['r2_frob']:.3f}"
        ol_t1, ol_t5 = f"{ol['top1_tiesafe']*100:.1f}%", f"{ol['top5_tiesafe']*100:.1f}%"
        checks = [
            (1, rf"^{re.escape(p4_cos)}$"),
            (2, rf"^{re.escape(p4_r2o)}\s*\({re.escape(p4_r2f)}\)$"),
            (3, rf"^{re.escape(p4_t1)}\s*/\s*{re.escape(p4_t5)}$"),
            (4, rf"^{re.escape(ol_cos)}$"),
            (5, rf"^{re.escape(ol_r2o)}\s*\({re.escape(ol_r2f)}\)$"),
            (6, rf"^{re.escape(ol_t1)}\s*/\s*{re.escape(ol_t5)}$"),
        ]
        _assert_row_regex(md_t1_rows, row_key, checks, f"MD Table 1 {k}", exact_cols=7)
        _assert_row_regex(tex_gpa, row_key, checks, f"TeX tab:gpa {k}", exact_cols=7, cleaner=lambda s: s)

    # Post-GPA Pairwise Cosine Matrix (paper.md)
    pw = meta["post_gpa_eval_pairwise_cos"]
    pw_models = [
        ("gemma4_e4b", "Gemma 4 E4B"),
        ("qwen35_9b", "Qwen3.5 9B"),
        ("llama31_8b", "Llama 3.1 8B"),
        ("mistral7b", "Mistral 7B"),
        ("inkling2_mini", "Inkling 2 Mini"),
    ]
    for i, (ki, row_key) in enumerate(pw_models):
        matches = [r for r in md_rows if len(r) == 6 and _clean_md(r[0]) == row_key]
        assert len(matches) == 1, f"Pairwise cosine matrix: expected 1 row for {row_key}, found {len(matches)}"
        for j, (kj, _) in enumerate(pw_models):
            val = 1.0 if i == j else pw.get(f"{ki}_vs_{kj}", pw.get(f"{kj}_vs_{ki}"))
            assert _clean_md(matches[0][j + 1]) == f"{val:.3f}", f"Pairwise {ki} vs {kj}: got {matches[0][j+1]}"

    # Table 2: Per-domain cosine breakdown (paper.md)
    singles = ["gemma4_e4b", "llama31_8b", "mistral7b", "qwen35_9b", "inkling2_mini"]
    dom_rows = [
        ("held_gut", "Held-out Gutenberg prose"),
        ("items3_mech", "items3 Mechanisms"),
        ("items3_myth", "items3 Myths"),
        ("proc_held", "Procedural reasoning"),
    ]
    for dk, rkey in dom_rows:
        p4_best = max(pf["blind"]["phi4_mini"][s]["by_domain_cos"][dk] for s in singles)
        p4_c4 = pf["blind"]["phi4_mini"]["C4"]["by_domain_cos"][dk]
        p4_c5 = pf["blind"]["phi4_mini"]["C5"]["by_domain_cos"][dk]
        ol_best = max(pf["blind"]["olmo3_7b"][s]["by_domain_cos"][dk] for s in singles)
        ol_c4 = pf["blind"]["olmo3_7b"]["C4"]["by_domain_cos"][dk]
        ol_c5 = pf["blind"]["olmo3_7b"]["C5"]["by_domain_cos"][dk]
        _assert_row_regex(md_rows, rkey, [
            (1, rf"^{p4_best:.4f}$"), (2, rf"^{p4_c4:.4f}$"), (3, rf"^{p4_c5:.4f}$"),
            (4, rf"^{ol_best:.4f}$"), (5, rf"^{ol_c4:.4f}$"), (6, rf"^{ol_c5:.4f}$"),
        ], f"MD Table 2 {dk}", exact_cols=7)

    # Table 2a: items3 Quartet Role Split (Full vs Myth-Pool-Excluded GPA, paper.md)
    for role in ("anchor", "lure", "true", "foil"):
        ol_f = pf["blind"]["olmo3_7b"]["items3_roles_C5_centered"][role]
        ol_n = pn["blind"]["olmo3_7b"]["items3_roles_C5_centered"][role]
        ph_f = pf["blind"]["phi4_mini"]["items3_roles_C5_centered"][role]
        ph_n = pn["blind"]["phi4_mini"]["items3_roles_C5_centered"][role]
        def _ci_pat(obj):
            d = f"{obj['diff']:+.4f}"
            lo = f"{obj['ci95'][0]:+.4f}"
            hi = f"{obj['ci95'][1]:+.4f}"
            return rf"^{re.escape(d)}\s*\[{re.escape(lo)},\s*{re.escape(hi)}\]"
        _assert_row_regex(md_rows, f"{role} (", [
            (1, _ci_pat(ol_f)), (2, _ci_pat(ol_n)), (3, _ci_pat(ph_f)), (4, _ci_pat(ph_n)),
        ], f"MD Table 2a {role}", exact_cols=5)

    for bk in ("phi4_mini", "olmo3_7b"):
        bs = pf["blind"][bk]["bootstrap_C5_minus_best_single"]
        assert_in_doc(paper_txt, f"+{bs['diff_clause_weighted']:.4f}", f"MD Sec 4.1 {bk} C5-best diff")
        assert_in_doc(tex_txt, f"+{bs['diff_clause_weighted']:.4f}", f"TeX Sec 4.1 {bk} C5-best diff")

    # =========================================================================
    # 2. 7-Company Zero-Shot Neutral-Text HOUSE Teleportation (Table 2b & tab:teleport)
    # =========================================================================
    arch = json.load(open(ARCH / "fluent" / "plato" / "arch229_report.json"))
    st = arch["modes"]["story_neutral_only"]["gemma_to_others"]
    ar_s = aud["arch_rot_strict_and_bow"]
    bow = ar_s["bow_lexical_baseline"]
    assert abs(bow["house_2f_setting_disjoint_20splits_mean"] - 0.9552) < 1e-4
    assert all(v["wins_over_bow_20splits"] == 20 for v in ar_s["neural_gemma_to_6_strict"].values())
    assert all(v["house_2f_lenres_fold_local"] >= 0.970 for v in ar_s["neural_gemma_to_6_strict"].values())
    assert all(v["house_noself_to_self2"] >= 0.974 for v in ar_s["neural_gemma_to_6_strict"].values())

    t2b_models = [
        ("qwen35_9b", "Qwen3.5 9B"),
        ("llama31_8b", "Llama 3.1 8B"),
        ("mistral7b", "Mistral 7B"),
        ("inkling2_mini", "Inkling 2 Mini"),
        ("phi4_mini", "Phi-4-mini"),
        ("olmo3_7b", "OLMo 3 7B"),
    ]
    tex_tele = _parse_tex_table(tex_txt, "tab:teleport")
    assert len(tex_tele) == 6, f"Expected 6 rows in tab:teleport, got {len(tex_tele)}"
    for mk, rkey in t2b_models:
        r = st[mk]
        rs = ar_s["neural_gemma_to_6_strict"][mk]
        i3_acc = f"{r['house_to_unseen_items3']['acc']*100:.1f}%"
        i3_p = r["house_to_unseen_items3"]["p_pos"]
        i3_tail = (
            rf"\((?:cos\s+)?p\s*=\s*{i3_p:.3f}\)"
            if i3_p < 0.05
            else r"\((?:n\.s\.|below 50%)\)"
        )
        h_cos = f"{r['house_disjoint']['cos']:+.3f}"
        h_p = f"{r['house_disjoint']['p_pos_max']:.3f}"
        h_acc = f"{r['house_disjoint']['acc']*100:.1f}%"
        h_lr = f"{rs['house_2f_lenres_fold_local']*100:.1f}%"
        h_pf = f"{r['perfection_acc_len_reversed']*100:.1f}%"
        s2_acc = f"{r['self2_disjoint']['acc']*100:.1f}%"
        h2s_acc = f"{r['house_to_unseen_self2']['acc']*100:.1f}%"
        checks = [
            (1, rf"^{re.escape(h_cos)}\s*\({re.escape(h_p)}\)$"),
            (2, rf"^{re.escape(h_acc)}$"),
            (3, rf"^{re.escape(h_lr)}$"),
            (4, rf"^{re.escape(h_pf)}$"),
            (5, rf"^{re.escape(s2_acc)}$"),
            (6, rf"^{re.escape(h2s_acc)}$"),
            (7, rf"^{re.escape(i3_acc)}\s*{i3_tail}$"),
        ]
        _assert_row_regex(md_rows, f"Gemma 4 → {rkey}", checks, f"MD Table 2b {mk}", exact_cols=8)
        _assert_row_regex(tex_tele, f"Gemma 4 -> {rkey}", checks, f"TeX tab:teleport {mk}", exact_cols=8, cleaner=lambda s: s)

    # Collinearity & variance-share transparency check in BOTH docs
    for label, doc in [("paper.md", paper_txt), ("main.tex", tex_txt)]:
        assert "+0.9539" in doc, f"Collinearity cos=+0.9539 missing from {label}!"
        assert "1.72" in doc, f"U_jung C5 variance share 1.72% missing from {label}!"

    # =========================================================================
    # 2b. Experiment 234 7-Company Valence Decomposition & Epistemic Sentiment Control (Table 2c & tab:val)
    # =========================================================================
    val_data = json.load(open(ARCH / "selfshape" / "valence_ctrl234_data.json"))
    assert len(val_data["sentiment_gen"]) == 60 and len(val_data["outcome_valence"]) == 40 and len(val_data["valence_matched"]) == 60
    v234 = json.load(open(ARCH / "fluent" / "plato" / "valence_ctrl234.json"))
    c5g = v234["c5_geometry"]
    assert abs(c5g["house_sqnorm_in_val2_span"] - 0.4679) < 1e-4
    assert abs(c5g["cos_house_sent"] - 0.6806) < 1e-4
    assert abs(v234["bow_valence_matched"]["all_60_20split_mean"] - 1.0) < 1e-6

    tex_val = _parse_tex_table(tex_txt, "tab:val")
    assert len(tex_val) == 6, f"Expected 6 rows in tab:val, got {len(tex_val)}"
    thc = v234["transfer_house_valence_controlled"]
    tvm = v234["transfer_valence_matched_60"]
    for mk, rkey in t2b_models:
        rh = thc[mk]
        rv = tvm[mk]
        s2h = f"{rh['sent_to_house_zs_acc']*100:.1f}%"
        hp_acc = f"{rh['house_perp_val2_dbl_20split_mean']*100:.1f}%"
        hp_cos = f"{rh['house_perp_val2_dbl_2fold_cos']:+.3f}"
        hp_p = f"{rh['house_perp_val2_dbl_2fold_p_max']:.3f}"
        s2v_acc = f"{rv['zs_sent_to_vm_acc']*100:.1f}%"
        s2v_cos = f"{rv['zs_sent_to_vm_cos']:+.3f}"
        h2v_acc = f"{rv['zs_house_perp_val2_dbl_to_vm_acc']*100:.1f}%"
        h2v_neg = _fmt_pair_pct(rv["zs_house_perp_val2_dbl_to_vm_both_neg_acc"])
        h2v_pos = _fmt_pair_pct(rv["zs_house_perp_val2_dbl_to_vm_both_pos_acc"])
        h2v_cos = f"{rv['zs_house_perp_val2_dbl_to_vm_cos']:+.3f}"
        h2v_p = f"{rv['zs_house_perp_val2_dbl_to_vm_p']:.3f}"
        checks = [
            (1, rf"^{re.escape(s2h)}$"),
            (2, rf"^{re.escape(hp_acc)}\s*\({re.escape(hp_cos)},\s*{re.escape(hp_p)}\)$"),
            (3, rf"^{re.escape(s2v_acc)}\s*\({re.escape(s2v_cos)}\)$"),
            (4, rf"^{re.escape(h2v_acc)}\s*\({re.escape(h2v_neg)}\s*/\s*{re.escape(h2v_pos)},\s*{re.escape(h2v_cos)},\s*{re.escape(h2v_p)}\)$"),
        ]
        _assert_row_regex(md_rows, f"Gemma 4 → {rkey}", checks, f"MD Table 2c {mk}", exact_cols=5)
        _assert_row_regex(tex_val, f"Gemma 4 -> {rkey}", checks, f"TeX tab:val {mk}", exact_cols=5, cleaner=lambda s: s)

    # Verify key epistemic sentiment-control numbers in BOTH paper.md and main.tex
    epi_sc = v234["epistemic_sentiment_control"]
    assert abs(epi_sc["gemma4_e4b"]["neutral_prompt"]["L25"]["neg_house_perp_corr_and_sent"]["auroc"] - 1.0) < 1e-6
    assert abs(epi_sc["gemma4_e4b"]["neutral_prompt"]["L25"]["neg_sent_perp_corr"]["auroc"] - 0.6024) < 5e-4
    assert abs(epi_sc["qwen35_9b"]["aion_prompt"]["L19"]["neg_house_perp_corr_and_sent"]["auroc"] - 0.5191) < 5e-4
    assert abs(epi_sc["qwen35_9b"]["aion_prompt"]["L19"]["neg_sent_perp_corr"]["auroc"] - 0.9774) < 5e-4
    for label, doc in [("paper.md", paper_txt), ("main.tex", tex_txt)]:
        for needle in ("46.8", "53.2", "0.602", "0.519", "0.977"):
            assert needle in doc, f"Experiment 234 headline '{needle}' missing from {label}!"

    # =========================================================================
    # 2c. Experiment 236 Exact Identical-BoW Swap Audit (Table 2d & tab:bswap)
    # =========================================================================
    bs_data = json.load(open(ARCH / "selfshape" / "bow_swap236_data.json"))
    assert len(bs_data["bow_swap_60"]) == 60
    b236 = json.load(open(ARCH / "fluent" / "plato" / "bow_swap236.json"))
    assert abs(b236["lexical_baselines"]["bow_bs60_unigram_20split_mean"] - 0.5) < 1e-12
    assert abs(b236["lexical_baselines"]["bow_house_to_vm60_unigram"] - 0.9083) < 5e-4
    md_bs_rows = _parse_md_table_after(paper_txt, "bow_swap236.json")
    assert len(md_bs_rows) == 6, f"Expected 6 rows in MD Table 2d, got {len(md_bs_rows)}"
    tex_bs = _parse_tex_table(tex_txt, "tab:bswap")
    assert len(tex_bs) == 6, f"Expected 6 rows in tab:bswap, got {len(tex_bs)}"
    for mk, rkey in t2b_models:
        v = b236["transfer_bow_swap_60"][mk]
        s_acc, s_p = f"{v['zs_sent_to_bs_acc']*100:.1f}%", f"{v['zs_sent_to_bs_p']:.3f}"
        h_acc, h_p = f"{v['zs_house_to_bs_acc']*100:.1f}%", f"{v['zs_house_to_bs_p']:.3f}"
        hp_acc, hp_c, hp_p = f"{v['zs_house_perp_val2_dbl_to_bs_acc']*100:.1f}%", f"{v['zs_house_perp_val2_dbl_to_bs_cos']:+.3f}", f"{v['zs_house_perp_val2_dbl_to_bs_p']:.3f}"
        hp_ns = r"\s+n\.s\." if v["zs_house_perp_val2_dbl_to_bs_p"] >= 0.05 else ""
        r2_acc, r2_p = f"{v['bs_2fold_raw_20split_mean']*100:.1f}%", f"{v['bs_2fold_raw_p_max']:.3f}"
        p2_acc, p2_c, p2_p = f"{v['bs_2fold_perp_val2_20split_mean']*100:.1f}%", f"{v['bs_2fold_perp_val2_cos']:+.3f}", f"{v['bs_2fold_perp_val2_p_max']:.3f}"
        p2_ns = r"\s+n\.s\." if v["bs_2fold_perp_val2_p_max"] >= 0.05 else ""
        p5_acc = f"{v['bs_2fold_perp_val5_20split_mean']*100:.1f}%"
        checks = [
            (1, rf"^{re.escape(s_acc)}\s*\({re.escape(s_p)}\)$"),
            (2, rf"^{re.escape(h_acc)}\s*\({re.escape(h_p)}\)$"),
            (3, rf"^{re.escape(hp_acc)}\s*\({re.escape(hp_c)},\s*{re.escape(hp_p)}{hp_ns}\)$"),
            (4, rf"^{re.escape(r2_acc)}\s*\({re.escape(r2_p)}\)$"),
            (5, rf"^{re.escape(p2_acc)}\s*\({re.escape(p2_c)},\s*{re.escape(p2_p)}{p2_ns}\)$"),
            (6, rf"^{re.escape(p5_acc)}$"),
        ]
        _assert_row_regex(md_bs_rows, f"Gemma 4 → {rkey}", checks, f"MD Table 2d {mk}", exact_cols=7)
        _assert_row_regex(tex_bs, f"Gemma 4 -> {rkey}", checks, f"TeX tab:bswap {mk}", exact_cols=7, cleaner=lambda s: s)

    # =========================================================================
    # 2d. Experiment 235 Fair Trial of 8 Canonical Named Archetypes (Table 2e & tab:named)
    # =========================================================================
    na_data = json.load(open(ARCH / "selfshape" / "named_arch235_data.json"))
    assert len(na_data) == 320
    n235 = json.load(open(ARCH / "fluent" / "plato" / "named_arch235.json"))
    assert abs(n235["c5_geometry"]["effective_rank_pr_raw"] - 5.009) < 1e-3
    assert abs(n235["c5_geometry"]["effective_rank_pr_perp_val3"] - 5.06) < 1e-2
    md_na_rows = _parse_md_table_after(paper_txt, "named_arch235.json")
    assert len(md_na_rows) == 6, f"Expected 6 rows in MD Table 2e, got {len(md_na_rows)}"
    tex_na = _parse_tex_table(tex_txt, "tab:named")
    assert len(tex_na) == 6, f"Expected 6 rows in tab:named, got {len(tex_na)}"
    for mk, rkey in t2b_models:
        r = n235["zero_shot_gemma_to_6"][mk]["raw"]
        p3 = n235["zero_shot_gemma_to_6"][mk]["perp_val3_dbl"]
        lr = n235["zero_shot_gemma_to_6"][mk]["lenres_fold_local"]
        assert all(v["p_perm_max"] <= 0.001 for v in p3["per_archetype"].values())
        checks = [
            (1, rf"^{r['top1_8way_mean']*100:.1f}%\s*\({r['top1_wins_over_tfidf_ridge_20']}/20\)$"),
            (2, rf"^{p3['top1_8way_mean']*100:.1f}%\s*\({p3['top1_wins_over_tfidf_ridge_20']}/20\)$"),
            (3, rf"^{r['macro_auroc_mean']:.3f}$"),
            (4, rf"^{p3['macro_auroc_mean']:.3f}$"),
            (5, rf"^{lr['top1_8way_mean']*100:.1f}%$"),
            (6, rf"^{r['within_setting_top1_mean']*100:.1f}%$"),
        ]
        _assert_row_regex(md_na_rows, f"Gemma 4 → {rkey}", checks, f"MD Table 2e {mk}", exact_cols=7)
        _assert_row_regex(tex_na, f"Gemma 4 -> {rkey}", checks, f"TeX tab:named {mk}", exact_cols=7, cleaner=lambda s: s)

    # =========================================================================
    # 3. 199-Family Strict LORO Retrieval vs. Strict-LQO TF-IDF (Table 3b & tab:loro)
    # =========================================================================
    myth = json.load(open(ARCH / "fluent" / "plato" / "myth229_meta.json"))
    assert abs(myth["cross_setting_loro_c5_centered"]["beat_1"]["top1_over_fams"] - 0.4062) < 1e-4
    ml = aud["myth_loro_strict_and_tfidf"]
    assert ml["family_size_dist"] == {"5": 182, "4": 14, "3": 3}
    n_stories = sum(int(k) * v for k, v in ml["family_size_dist"].items())
    assert n_stories == 975
    exact_1d = lambda v: f"{(round(v * n_stories) / n_stories) * 100.0:.1f}"

    tex_loro = _parse_tex_table(tex_txt, "tab:loro")
    assert len(tex_loro) == 4, f"Expected 4 rows in tab:loro, got {len(tex_loro)}"
    for bname in ("beat_0", "beat_1", "beat_2", "beat_3"):
        bt = ml["beats"][bname]
        assert bt["tfidf_unigram_baseline"]["idf_mode"] == "strict_leave_query_out_vocab_and_idf"
        c5_t1, c5_t5 = exact_1d(bt["c5_strict_loro"]["top1_story_avg"]), exact_1d(bt["c5_strict_loro"]["top5_story_avg"])
        c5_eq = f"{bt['c5_strict_loro']['top1_equal_fam']*100:.1f}"
        p4_t1, p4_t5 = exact_1d(bt["phi4_strict_loro"]["top1_story_avg"]), exact_1d(bt["phi4_strict_loro"]["top5_story_avg"])
        ol_t1, ol_t5 = exact_1d(bt["olmo3_strict_loro"]["top1_story_avg"]), exact_1d(bt["olmo3_strict_loro"]["top5_story_avg"])
        tf_t1, tf_t5 = exact_1d(bt["tfidf_unigram_baseline"]["top1_story_avg"]), exact_1d(bt["tfidf_unigram_baseline"]["top5_story_avg"])
        wk_m = f"{bt['c5_strict_loro']['within_kind_myth_top1']*100:.1f}"
        wk_c = f"{bt['c5_strict_loro']['within_kind_mech_top1']*100:.1f}"

        _assert_row_regex(md_rows, bname, [
            (1, rf"^{re.escape(c5_t1)}%\s*/\s*{re.escape(c5_t5)}%\s*\({re.escape(c5_eq)}%\)$"),
            (2, rf"^{re.escape(p4_t1)}%\s*/\s*{re.escape(p4_t5)}%$"),
            (3, rf"^{re.escape(ol_t1)}%\s*/\s*{re.escape(ol_t5)}%$"),
            (4, rf"^{re.escape(tf_t1)}%\s*/\s*{re.escape(tf_t5)}%$"),
            (5, rf"^{re.escape(wk_m)}%\s*/\s*{re.escape(wk_c)}%$"),
        ], f"MD Table 3b {bname}", exact_cols=6)
        _assert_row_regex(tex_loro, bname, [
            (1, rf"^{re.escape(c5_t1)}\s*/\s*{re.escape(c5_t5)}$"),
            (2, rf"^{re.escape(p4_t1)}\s*/\s*{re.escape(p4_t5)}$"),
            (3, rf"^{re.escape(ol_t1)}\s*/\s*{re.escape(ol_t5)}$"),
            (4, rf"^{re.escape(tf_t1)}\s*/\s*{re.escape(tf_t5)}$"),
        ], f"TeX tab:loro {bname}", exact_cols=5, cleaner=lambda s: s)

    # =========================================================================
    # 4. ALL 24 5M Student Runs (Table 3, Table 3c, & tab:5m — STRICT ORDERED!)
    # =========================================================================
    expected_5m = {
        "P0": (-0.0290, -0.0726, 1.4714, 1.2235, 0.6546, 0.1553),
        "P1": (+0.0147, +0.0512, 1.4788, 1.2316, 0.6478, 0.1474),
        "P2": (+0.1223, +0.1472, 1.4780, 1.2310, 0.6405, 0.1508),
        "P3": (+0.1276, +0.1538, 1.4830, 1.2367, 0.6537, 0.1519),
        "E229_M0": (+0.1318, +0.1393, 1.4796, 1.2325, 0.6493, 0.1548),
        "E229_M1": (+0.1265, +0.1296, 1.4792, 1.2303, 0.6500, 0.1406),
        "E229_M2": (+0.1316, +0.1351, 1.4801, 1.2321, 0.6572, 0.1190),
        "E229_M3": (+0.1310, +0.1355, 1.4778, 1.2297, 0.6447, 0.1723),
    }
    t3_keys = {
        "P0": "P0 Plain control",
        "P1": "P1 Single-teacher Gemma 4",
        "P2": "P2 5-Family Platonic Consensus",
        "P3": "P3 Consensus + Myth/Folklore",
    }
    t3c_keys = {
        "E229_M0": "M0 Annealed C5",
        "E229_M1": "M1 + LORO Myth Prototypes",
        "E229_M2": "M2 + Boosted U_jung",
        "E229_M3": "M3 − Ablated U_jung",
    }
    tex5m_keys = {
        "P0": "P0 Plain Control",
        "P1": "P1 Gemma 4 Only",
        "P2": "P2 5-Fam C5",
        "P3": "P3 C5 + Folklore",
        "E229_M0": "M0 Annealed C5",
        "E229_M1": "M1 + LORO Proto",
        "E229_M2": "M2 +",
        "E229_M3": "M3 -",
    }
    tex_5m = _parse_tex_table(tex_txt, "tab:5m")
    assert len(tex_5m) == 8, f"Expected 8 rows in tab:5m, got {len(tex_5m)}"

    for arm, (ep4, eol, ebk, ech, ebl, emq) in expected_5m.items():
        cfs = sorted(glob.glob(str(ARCH / "fluent" / "runs" / "comp" / f"{arm}_s*.json")))
        tfs = sorted(glob.glob(str(ARCH / "fluent" / "runs" / f"{arm}_s*.json")))
        assert len(cfs) == 3 and len(tfs) == 3, f"Missing 3 seeds for {arm}: comp={len(cfs)} train={len(tfs)}"
        cs = [json.load(open(f)) for f in cfs]
        ts = [json.load(open(f)) for f in tfs]

        p4_cos = float(np.mean([c["platonic"]["blind_phi4_pts"]["cos"] for c in cs]))
        p4_r2f = float(np.mean([c["platonic"]["blind_phi4_pts"]["r2_frob"] for c in cs]))
        p4_r2o = float(np.mean([c["platonic"]["blind_phi4_pts"]["r2_opt"] for c in cs]))
        p4_t1 = float(np.mean([c["platonic"]["blind_phi4_pts"]["top1"] for c in cs])) * 100.0
        p4_st1 = float(np.mean([c["platonic"]["blind_phi4_story"]["top1"] for c in cs])) * 100.0

        ol_cos = float(np.mean([c["platonic"]["blind_olmo3_pts"]["cos"] for c in cs]))
        ol_r2f = float(np.mean([c["platonic"]["blind_olmo3_pts"]["r2_frob"] for c in cs]))
        ol_r2o = float(np.mean([c["platonic"]["blind_olmo3_pts"]["r2_opt"] for c in cs]))
        ol_t1 = float(np.mean([c["platonic"]["blind_olmo3_pts"]["top1"] for c in cs])) * 100.0
        ol_st1 = float(np.mean([c["platonic"]["blind_olmo3_story"]["top1"] for c in cs])) * 100.0

        myth_tf = float(np.mean([c["platonic"]["items3_layer2_centered"]["myth"]["TF"] for c in cs]))
        myth_tl = float(np.mean([c["platonic"]["items3_layer2_centered"]["myth"]["TL"] for c in cs]))
        mech_tf = float(np.mean([c["platonic"]["items3_layer2_centered"]["mech"]["TF"] for c in cs]))
        mech_tl = float(np.mean([c["platonic"]["items3_layer2_centered"]["mech"]["TL"] for c in cs]))
        proc_g = float(np.mean([c["proc"]["overall"]["gain"] for c in cs]))
        v2_g = float(np.mean([c["v2"]["overall"]["gain"] for c in cs]))
        bk_bpb = float(np.mean([t["books"]["bpb"] for t in ts]))
        ch_bpb = float(np.mean([t["chat_held"]["bpb"] for t in ts]))
        blimp_v = float(np.mean([c["blimp"] for c in cs]))
        copa_v = float(np.mean([c["copa"] for c in cs]))
        mcq_v = float(np.mean([c["mcq_gain"] for c in cs]))

        assert abs(p4_r2f - ep4) < 5e-4 and abs(ol_r2f - eol) < 5e-4
        assert abs(bk_bpb - ebk) < 5e-4 and abs(ch_bpb - ech) < 5e-4
        assert abs(blimp_v - ebl) < 5e-4 and abs(mcq_v - emq) < 5e-4

        if arm in t3_keys:
            _assert_row_regex(md_rows, t3_keys[arm], [
                (1, rf"^{p4_cos:.4f}\s*/\s*{re.escape(f'{p4_r2f:+.3f}')}\s*\({p4_t1:.1f}%\s*/\s*{p4_st1:.1f}%\)$"),
                (2, rf"^{ol_cos:.4f}\s*/\s*{re.escape(f'{ol_r2f:+.3f}')}\s*\({ol_t1:.1f}%\s*/\s*{ol_st1:.1f}%\)$"),
                (3, rf"^{myth_tf:.2f}\s*/\s*{myth_tl:.2f}$"),
                (4, rf"^{mech_tf:.2f}\s*/\s*{mech_tl:.2f}$"),
                (5, rf"^{re.escape(f'{proc_g:+.3f}')}$"),
                (6, rf"^{re.escape(f'{mcq_v:+.3f}')}\s*/\s*{re.escape(f'{v2_g:+.3f}')}$"),
                (7, rf"^{bk_bpb:.4f}\s*/\s*{ch_bpb:.4f}$"),
                (8, rf"^{blimp_v:.3f}\s*/\s*{copa_v:.3f}$"),
            ], f"MD Table 3 {arm}", exact_cols=9)
            _assert_row_regex(tex_5m, tex5m_keys[arm], [
                (1, rf"^{p4_cos:.4f}\s*/\s*{re.escape(f'{p4_r2f:+.3f}')}$"),
                (2, rf"^{ol_cos:.4f}\s*/\s*{re.escape(f'{ol_r2f:+.3f}')}$"),
                (3, rf"^{bk_bpb:.4f}\s*/\s*{ch_bpb:.4f}$"),
                (4, rf"^{blimp_v:.3f}$"),
                (5, rf"^{copa_v:.3f}$"),
                (6, rf"^{re.escape(f'{mcq_v:+.3f}')}$"),
                (7, r"^-$"),
            ], f"TeX tab:5m {arm}", exact_cols=8, cleaner=lambda s: s)

        if arm in t3c_keys:
            head_cos = float(np.mean([c["jung_coherence"]["plato_head_to_u_jung_c5"]["cos_house"] for c in cs]))
            bb_cos = float(np.mean([c["jung_coherence"]["backbone_neutral_procrustes_to_c5"]["cos_house"] for c in cs]))
            zs_h = float(np.mean([c["jung_coherence"]["backbone_neutral_procrustes_to_c5"]["zeroshot_house_acc"] for c in cs])) * 100.0
            zs_s2 = float(np.mean([c["jung_coherence"]["backbone_neutral_procrustes_to_c5"]["zeroshot_self2_acc"] for c in cs])) * 100.0
            perf_l = float(np.mean([c["jung_coherence"]["last"]["house_perf_acc"] for c in cs])) * 100.0
            perf_m = float(np.mean([c["jung_coherence"]["mean"]["house_perf_acc"] for c in cs])) * 100.0

            _assert_row_regex(md_rows, t3c_keys[arm], [
                (1, rf"^{p4_cos:.4f}\s*/\s*{re.escape(f'{p4_r2f:+.3f}')}\s*\({p4_r2o:.3f}\)$"),
                (2, rf"^{ol_cos:.4f}\s*/\s*{re.escape(f'{ol_r2f:+.3f}')}\s*\({ol_r2o:.3f}\)$"),
                (3, rf"^{blimp_v:.3f}$"),
                (4, rf"^{copa_v:.3f}$"),
                (5, rf"^{mcq_v:.3f}$"),
                (6, rf"^{re.escape(f'{proc_g:+.3f}')}\s*/\s*{re.escape(f'{v2_g:+.3f}')}$"),
                (7, rf"^{re.escape(f'{head_cos:+.3f}')}$"),
                (8, rf"^{re.escape(f'{bb_cos:+.3f}')}\s*/\s*{zs_h:.1f}%\s*/\s*{zs_s2:.1f}%$"),
                (9, rf"^{perf_l:.1f}%\s*/\s*{perf_m:.1f}%$"),
            ], f"MD Table 3c {arm}", exact_cols=10)
            _assert_row_regex(tex_5m, tex5m_keys[arm], [
                (1, rf"^{p4_cos:.4f}\s*/\s*{re.escape(f'{p4_r2f:+.3f}')}$"),
                (2, rf"^{ol_cos:.4f}\s*/\s*{re.escape(f'{ol_r2f:+.3f}')}$"),
                (3, rf"^{bk_bpb:.4f}\s*/\s*{ch_bpb:.4f}$"),
                (4, rf"^{blimp_v:.3f}$"),
                (5, rf"^{copa_v:.3f}$"),
                (6, rf"^{re.escape(f'{mcq_v:+.3f}')}$"),
                (7, rf"^{re.escape(f'{bb_cos:+.3f}')}\s*/\s*{zs_h:.1f}%\s*/\s*{zs_s2:.1f}%$"),
            ], f"TeX tab:5m {arm}", exact_cols=8, cleaner=lambda s: s)

    # =========================================================================
    # 5. 8B Gemma 4 E4B Paired Causal Control (ALL 4 rows of Table 4 & tab:8b!)
    # =========================================================================
    base = json.load(open(ARCH / "aion_8b" / "runs" / "e4b_base_eval.json"))["platonic_jung"]
    ctrl = json.load(open(ARCH / "aion_8b" / "runs" / "e4b_stage_a_ctrl_eval.json"))
    stga = json.load(open(ARCH / "aion_8b" / "runs" / "e4b_stage_a_eval.json"))
    tex_8b = _parse_tex_table(tex_txt, "tab:8b")
    assert len(tex_8b) == 4, f"Expected 4 rows in tab:8b, got {len(tex_8b)}"

    for md_key, tex_key, obj, is_base, md_c6, tex_c5, lbl in [
        ("Unmodified Base E4B", "Unmodified Base E4B", base, True,
         r'^Verified\s*\("Red"\s*/\s*"blue pelican"\)$', r"^Pass\s*\(``Red''\s*/\s*``blue pelican''\)$", "Table 4 Base"),
        ("Matched CE+KL Control", "Matched CE+KL Ctrl", ctrl, False,
         r"^Frozen towers$", r"^Frozen towers$", "Table 4 Ctrl"),
        ("Stage A Platonic", "Stage A (C5", stga, False,
         r'^Directly verified\s*\("Red"\s*/\s*"blue pelican"\)$', r"^Directly verified$", "Table 4 Stage A"),
    ]:
        p4, ol, c5 = obj["blind_phi4_pts"], obj["blind_olmo3_pts"], obj["c5_eval_pts"]
        nll = obj["held_nll"]["held_gut_nll"]
        jg = obj["jung_L12"]
        if is_base:
            p4_md = rf"^{p4['cos']:.4f}\s*/\s*\+{p4['r2_frob']:.4f}\s*\({p4['top1']*100:.2f}%\)$"
            ol_md = rf"^{ol['cos']:.4f}\s*/\s*\+{ol['r2_frob']:.4f}\s*\({ol['top1']*100:.2f}%\)$"
            nll_md = rf"^{nll:.4f}$"
        else:
            dp4 = p4["r2_frob"] - base["blind_phi4_pts"]["r2_frob"]
            dol = ol["r2_frob"] - base["blind_olmo3_pts"]["r2_frob"]
            dnll = nll - base["held_nll"]["held_gut_nll"]
            p4_md = rf"^{p4['cos']:.4f}\s*/\s*\+{p4['r2_frob']:.4f}\s*\({re.escape(f'{dp4:+.4f}')}\)\s*\({p4['top1']*100:.2f}%\)$"
            ol_md = rf"^{ol['cos']:.4f}\s*/\s*\+{ol['r2_frob']:.4f}\s*\({re.escape(f'{dol:+.4f}')}\)\s*\({ol['top1']*100:.2f}%\)$"
            nll_md = rf"^{nll:.4f}\s*\({re.escape(f'{dnll:+.4f}')}\)$"
        h2s = "100%" if abs(jg["house_to_self2_acc"] - 1.0) < 1e-4 else f"{jg['house_to_self2_acc']*100:.1f}%"
        _assert_row_regex(md_rows, md_key, [
            (1, p4_md),
            (2, ol_md),
            (3, rf"^{c5['cos']:.4f}\s*/\s*\+{c5['r2_frob']:.4f}$"),
            (4, nll_md),
            (5, rf"^{jg['house_2f_acc']*100:.2f}%\s*\({jg['perf_acc']*100:.1f}%\s*/\s*{re.escape(h2s)}\)$"),
            (6, md_c6),
        ], f"MD {lbl}", exact_cols=7)
        _assert_row_regex(tex_8b, tex_key, [
            (1, rf"^{p4['cos']:.4f}\s*/\s*\+{p4['r2_frob']:.4f}\s*\({p4['top1']*100:.2f}%\)$"),
            (2, rf"^{ol['cos']:.4f}\s*/\s*\+{ol['r2_frob']:.4f}\s*\({ol['top1']*100:.2f}%\)$"),
            (3, rf"^{c5['cos']:.4f}\s*/\s*\+{c5['r2_frob']:.4f}$"),
            (4, rf"^{nll:.4f}$"),
            (5, tex_c5),
        ], f"TeX {lbl}", exact_cols=6, cleaner=lambda s: s)

    # Computed Difference (Stage A - CE+KL Ctrl) row in BOTH Table 4 and tab:8b!
    dp4_cos = stga["blind_phi4_pts"]["cos"] - ctrl["blind_phi4_pts"]["cos"]
    dp4_r2 = stga["blind_phi4_pts"]["r2_frob"] - ctrl["blind_phi4_pts"]["r2_frob"]
    dp4_a = stga["blind_phi4_pts"]["r2_frob"] - base["blind_phi4_pts"]["r2_frob"]
    dp4_c = ctrl["blind_phi4_pts"]["r2_frob"] - base["blind_phi4_pts"]["r2_frob"]
    dp4_pct = 100.0 * dp4_r2 / dp4_a
    dp4_rat = dp4_a / dp4_c

    dol_cos = stga["blind_olmo3_pts"]["cos"] - ctrl["blind_olmo3_pts"]["cos"]
    dol_r2 = stga["blind_olmo3_pts"]["r2_frob"] - ctrl["blind_olmo3_pts"]["r2_frob"]
    dol_a = stga["blind_olmo3_pts"]["r2_frob"] - base["blind_olmo3_pts"]["r2_frob"]
    dol_c = ctrl["blind_olmo3_pts"]["r2_frob"] - base["blind_olmo3_pts"]["r2_frob"]
    dol_pct = 100.0 * dol_r2 / dol_a
    dol_rat = dol_a / dol_c

    dc5_cos = stga["c5_eval_pts"]["cos"] - ctrl["c5_eval_pts"]["cos"]
    dc5_r2 = stga["c5_eval_pts"]["r2_frob"] - ctrl["c5_eval_pts"]["r2_frob"]
    dnll_ac = stga["held_nll"]["held_gut_nll"] - ctrl["held_nll"]["held_gut_nll"]
    dh_pts = (stga["jung_L12"]["house_2f_acc"] - ctrl["jung_L12"]["house_2f_acc"]) * 100.0
    dperf = (stga["jung_L12"]["perf_acc"] - ctrl["jung_L12"]["perf_acc"]) * 100.0

    _assert_row_regex(md_rows, "Difference (Stage A − CE+KL Ctrl)", [
        (1, rf"^{re.escape(f'{dp4_cos:+.4f}')}\s*/\s*{re.escape(f'{dp4_r2:+.4f}')}\s*\({dp4_pct:.1f}%,\s*{dp4_rat:.2f}×\)$"),
        (2, rf"^{re.escape(f'{dol_cos:+.4f}')}\s*/\s*{re.escape(f'{dol_r2:+.4f}')}\s*\({dol_pct:.1f}%,\s*{dol_rat:.2f}×\)$"),
        (3, rf"^{re.escape(f'{dc5_cos:+.4f}')}\s*/\s*{re.escape(f'{dc5_r2:+.4f}')}$"),
        (4, rf"^{re.escape(f'{dnll_ac:+.4f}')}$"),
        (5, rf"^{re.escape(f'{dh_pts:+.2f}')}\s*pts\s*\({re.escape(f'{dperf:+.1f}')}\s*perf\)$"),
        (6, r"^0 extra params$"),
    ], "MD Table 4 Diff", exact_cols=7)
    _assert_row_regex(tex_8b, "Difference (Stage A - Ctrl)", [
        (1, rf"^{re.escape(f'{dp4_cos:+.4f}')}\s*/\s*{re.escape(f'{dp4_r2:+.4f}')}\s*\({dp4_pct:.1f}%\)$"),
        (2, rf"^{re.escape(f'{dol_cos:+.4f}')}\s*/\s*{re.escape(f'{dol_r2:+.4f}')}\s*\({dol_pct:.1f}%\)$"),
        (3, rf"^{re.escape(f'{dc5_cos:+.4f}')}\s*/\s*{re.escape(f'{dc5_r2:+.4f}')}$"),
        (4, rf"^{re.escape(f'{dnll_ac:+.4f}')}$"),
        (5, r"^0 extra params$"),
    ], "TeX tab:8b Diff", exact_cols=6, cleaner=lambda s: s)

    # =========================================================================
    # 6. Native-Layer Epistemic Probing (ALL 14 rows of Table 5 & tab:epi — STRICT n.s.!)
    # =========================================================================
    gm2 = json.load(open(ARCH / "aion_8b" / "runs" / "audit233_epistemic_gemma_2x2.json"))
    qw2 = json.load(open(ARCH / "aion_8b" / "runs" / "audit233_epistemic_qwen_2x2.json"))
    assert gm2.get("stage_b_eval_mode") == "titans_enabled_cold_reset"
    for mm_key in ("base_multimodal", "stage_a_multimodal"):
        mm = gm2[mm_key]
        assert mm["vision_ok"] is True and mm["audio_ok"] is True
        assert mm.get("audio_fixture") == "speech_blue_pelican"
        assert "blue" in mm["audio_speech_reply"].lower() and "pelican" in mm["audio_speech_reply"].lower()

    tex_epi = _parse_tex_table(tex_txt, "tab:epi")
    assert len(tex_epi) == 14, f"Expected 14 rows in tab:epi, got {len(tex_epi)}"

    t5_spec = [
        ("Gemma 4 Base × Neutral (L17", "Gemma 4 Base x Neutral (L17", gm2["base_e4b"]["neutral_prompt"]["L17"], "clean_tense_clean_24", False),
        ("Gemma 4 Base × Aion Prompt (L17", "Gemma 4 Base x Aion (L17", gm2["base_e4b"]["buddy_prompt"]["L17"], "clean_tense_clean_24", False),
        ("Gemma 4 Stage-B (Titans ON) × Neutral (L17", "Gemma 4 Stage-B (ON) x Neutral (L17", gm2["stage_b_e4b"]["neutral_prompt"]["L17"], "clean_tense_clean_24", False),
        ("Gemma 4 Stage-B (Titans ON) × Aion (L17", "Gemma 4 Stage-B (ON) x Aion (L17", gm2["stage_b_e4b"]["buddy_prompt"]["L17"], "clean_tense_clean_24", False),
        ("Gemma 4 Base × Neutral (L25", "Gemma 4 Base x Neutral (L25", gm2["base_e4b"]["neutral_prompt"]["L25"], "clean_tense_clean_24", False),
        ("Gemma 4 Base × Aion Prompt (L25", "Gemma 4 Base x Aion (L25", gm2["base_e4b"]["buddy_prompt"]["L25"], "clean_tense_clean_24", False),
        ("Gemma 4 Stage-B (Titans ON) × Neutral (L25", "Gemma 4 Stage-B (ON) x Neutral (L25", gm2["stage_b_e4b"]["neutral_prompt"]["L25"], "clean_tense_clean_24", False),
        ("Gemma 4 Stage-B (Titans ON) × Aion (L25", "Gemma 4 Stage-B (ON) x Aion (L25", gm2["stage_b_e4b"]["buddy_prompt"]["L25"], "clean_tense_clean_24", False),
        ("Qwen3.5-9B Base × Neutral (L12", "Qwen3.5-9B Base x Neutral (L12", qw2["neutral_prompt"]["L12"], "clean_tense_clean_24", False),
        ("Qwen3.5-9B Base × Aion Prompt (L12", "Qwen3.5-9B Base x Aion (L12", qw2["buddy_prompt"]["L12"], "clean_tense_clean_24", False),
        ("Qwen3.5-9B Base × Neutral (L19", "Qwen3.5-9B Base x Neutral (L19", qw2["neutral_prompt"]["L19"], "clean_tense_clean_24", False),
        ("Qwen3.5-9B Base × Aion Prompt (L19", "Qwen3.5-9B Base x Aion (L19", qw2["buddy_prompt"]["L19"], "clean_tense_clean_24", False),
        ("Dedup-10 Nonce OOS: Gemma 4 Base", "Dedup-10 Nonce: Gemma 4 Base", gm2["base_e4b"]["buddy_prompt_dedup10"]["L25"], "dedup_10_oos", True),
        ("Dedup-10 Nonce OOS: Gemma 4 Stage-B", "Dedup-10 Nonce: Gemma 4 Stage-B", gm2["stage_b_e4b"]["buddy_prompt_dedup10"]["L25"], "dedup_10_oos", True),
    ]
    for md_key, tex_key, cell, sub_key, check_ent_pair in t5_spec:
        assert abs(cell["perp_cos_check"]) < 1e-5
        sub = cell["subsets"][sub_key]
        e_sym = re.escape(f"{sub['entropy_best_sym_auroc']:.3f}")
        e_pos = re.escape(f"+{sub['entropy_pos_auroc']:.3f}")
        if check_ent_pair:
            ent_md = rf"^{e_sym}\s*\({e_pos},\s*{re.escape(_fmt_pair_pct(sub['entropy_pos_paired']))}\s*pair\)$"
        else:
            ent_md = rf"^{e_sym}\s*\({e_pos}(?:,\s*near\s+chance)?\)$"

        nc_a, nc_p = re.escape(f"{sub['neg_corr_auroc']:.3f}"), re.escape(f"{sub['neg_corr_p_rand500']:.3f}")
        nc_ns = r"\s+n\.s\." if sub["neg_corr_p_rand500"] >= 0.05 else ""
        hp_a, hp_p = re.escape(f"{sub['neg_house_perp_auroc']:.3f}"), re.escape(f"{sub['neg_house_perp_p_rand500']:.3f}")
        hp_pair_opt = rf"(?:{re.escape(_fmt_pair_pct(sub['neg_house_perp_paired']))},\s*)?"
        hp_ns = r"\s+n\.s\." if sub["neg_house_perp_p_rand500"] >= 0.05 else ""
        cb_a, cb_p = re.escape(f"{sub['combined_perp_auroc']:.3f}"), re.escape(f"{sub['combined_perp_p_mirrored500']:.3f}")
        # Strict: ONLY `n.s.` allowed when p >= 0.05 (no `null` alias!)
        cb_ns = r"\s+n\.s\." if sub["combined_perp_p_mirrored500"] >= 0.05 else ""
        cb_pair = re.escape(_fmt_pair_pct(sub["combined_perp_paired"]))
        n95 = re.escape(f"{sub['rand500_mirrored2_p95']:.3f}")

        _assert_row_regex(md_rows, md_key, [
            (1, rf"^{sub['entropy_mean_known_nats']:.3f}\s*→\s*{sub['entropy_mean_unk_nats']:.3f}$"),
            (2, ent_md),
            (3, rf"^{nc_a}\s*\({nc_p}{nc_ns}\)$"),
            (4, rf"^{hp_a}\s*\({hp_pair_opt}{hp_p}{hp_ns}\)$"),
            (5, rf"^{cb_a}\s*\({cb_pair},\s*{cb_p}{cb_ns}\)$"),
            (6, rf"^{n95}$"),
        ], f"MD Table 5 {md_key}", exact_cols=7)

        _assert_row_regex(tex_epi, tex_key, [
            (1, rf"^{sub['entropy_mean_known_nats']:.3f}\s*->\s*{sub['entropy_mean_unk_nats']:.3f}$"),
            (2, rf"^{e_sym}$"),
            (3, rf"^{nc_a}\s*\({nc_p}{nc_ns}\)$"),
            (4, rf"^{hp_a}\s*\({hp_p}{hp_ns}\)$"),
            (5, rf"^{cb_a}\s*\({cb_pair},\s*{cb_p}{cb_ns}\)$"),
            (6, rf"^{n95}$"),
        ], f"TeX tab:epi {tex_key}", exact_cols=7, cleaner=lambda s: s)

    print("[PASS] All 6 pillars, Experiments 234-236 (valence, exact-BoW swaps, 8 named archetypes), 24 student JSONs, ALL paper.md tables (1..5), AND ALL 9 main.tex tables (tab:gpa..tab:epi) match cleanly.")


def run_unit_tests():
    tests = [
        ("sceptic_ctrl230", [sys.executable, str(ARCH / "aion_8b" / "sceptic_ctrl230.py"), "test"]),
        ("arch_rot229", [sys.executable, str(ARCH / "fluent" / "arch_rot229.py"), "test"]),
        ("valence_ctrl234", [sys.executable, str(ARCH / "fluent" / "valence_ctrl234.py"), "test"]),
        ("named_arch235", [sys.executable, str(ARCH / "fluent" / "named_arch235.py"), "test"]),
        ("bow_swap236", [sys.executable, str(ARCH / "fluent" / "bow_swap236.py"), "test"]),
        ("audit233_epistemic_overlap", [sys.executable, "-c", "from aion_8b import audit233_epistemic as a; a._verify_zero_entity_overlap()"]),
    ]
    skipped = []
    # Include tokenizer/torch tests automatically whenever optional packages are present; log skips explicitly!
    try:
        import tokenizers  # noqa: F401
        tests.append(("plato_data", [sys.executable, str(ARCH / "fluent" / "plato_data.py"), "test"]))
        tests.append(("myth_proto229", [sys.executable, str(ARCH / "fluent" / "myth_proto229.py"), "test"]))
    except ImportError:
        skipped.extend(["plato_data (missing tokenizers)", "myth_proto229 (missing tokenizers)"])
    try:
        import torch  # noqa: F401
        tests.append(("exp228", [sys.executable, str(ARCH / "fluent" / "exp228.py"), "test"]))
        tests.append(("exp229", [sys.executable, str(ARCH / "fluent" / "exp229.py"), "test"]))
        tests.append(("e4b_living", [sys.executable, str(ARCH / "aion_8b" / "e4b_living.py"), "test"]))
    except ImportError:
        skipped.extend(["exp228/exp229/e4b_living (missing torch)"])

    # Strictly isolate JPL_CKPT_DIR in a temp sandbox so tests NEVER touch user checkpoints!
    with tempfile.TemporaryDirectory(prefix="jpl_repro_test_") as td:
        env = dict(os.environ)
        env["JPL_CKPT_DIR"] = str(Path(td) / "ckpt")
        for name, cmd in tests:
            res = subprocess.run(cmd, cwd=str(ARCH), env=env, capture_output=True, text=True)
            assert res.returncode == 0, f"Unit test '{name}' failed:\n{res.stdout}\n{res.stderr}"
    print(f"[PASS] Zero-GPU unit tests passed ({len(tests)}/{len(tests)}: {', '.join(n for n, _ in tests)}).")
    if skipped:
        print(f"[SKIP] Optional tests skipped due to missing packages: {', '.join(skipped)}")


if __name__ == "__main__":
    check_privacy()
    check_py_compile()
    check_paper_and_figures()
    check_headline_numbers_and_paper_sync()
    run_unit_tests()
    print("\nALL_REPRO_CHECKS_PASSED_OK")
