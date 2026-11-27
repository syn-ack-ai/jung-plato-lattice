"""One-command zero-GPU reproducibility & integrity verifier for The Jung–Plato Lattice.

Checks:
  1. Privacy, license & secret firewall across all tracked and untracked non-ignored files (`git ls-files`),
     rejecting redistributed `aft.csv`/`aft_real_myths_500.json`/`aft_excerpts.json` (case-insensitive + raw SHA-256),
     personal home paths, and `./systemone/`.
  2. Read-only-safe Python 3.10+ syntax check (in-memory `ast.parse(..., feature_version=(3, 10))` + `compile`)
     across all Python modules (zero `.pyc` writes).
  3. Exact presence of all 9 publication figures (PNG + PDF in paper/figures/ and PDF in paper/arxiv/figures/),
     hero PNG, committed spoken-audio fixture, dual-audience Word review copy (`paper/The_Jung_Plato_Lattice_v6.docx`),
     87 verified references across paper.md, references.bib, and main.bbl, third-party license files
     (`THIRD_PARTY_LICENSES.md`, `LICENSES/Apache-2.0.txt`), pure-ASCII `IEEEtran.cls`, mandatory AI-use disclosure,
     and strict arXiv LaTeX hygiene (\\pdfoutput=1 header, pure ASCII main.tex, zero \\resizebox, zero mid-sentence ---,
     zero unescaped ~digits).
  4. Strict ordered, typed cell-level cross-verification of committed JSON artifacts against EVERY numeric
     table (including all CI endpoints, all Exp 251 delta columns, all 10/10 multimodal fixtures, Round-8
     `n=477`/`n=473` overlap audits, Cross-Lingual causal story audit, and byte-for-byte `script_sha256` parity across
     all 5 active v6 JSONs & runners) in BOTH `paper/paper.md` AND `paper/arxiv/main.tex` (all 16 tabular tables +
     Rosetta Stone), plus target- and order-bound Exp 251 canonical OOF bootstrap checks and `README.md` headline checks,
     PLUS a 12/12 in-memory corruption regression self-test.
  5. Strict `paper/arxiv/arxiv_submission.tar.gz` hygiene & SHA-256 byte-parity inspection (exact 13-file allowlist,
     zero AppleDouble `._*` sidecars, uid=0/gid=0, empty uname/gname, zero xattr PAX headers).
  6. Comprehensive zero-GPU scientific unit tests with fully isolated temporary `JPL_CKPT_DIR` (explicitly
     reporting any optional tests skipped when optional packages are not installed).
"""
import ast
import copy
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "bench" / "archetypes"

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".aion", "local_data", "models",
    "local_ckpt", "local_eval", "folklore_local_data", "folklore_test", "local_scratch",
}

BINARY_EXTS = {".png", ".pdf", ".npz", ".npy", ".bin", ".pt", ".wav", ".cls", ".gz", ".tar", ".docx"}


def _is_skipped(p: Path) -> bool:
    # Match directory components ONLY (never skip a tracked root/nested file such as `.env`!)
    parent_parts = p.relative_to(ROOT).parts[:-1]
    return bool(SKIP_DIRS.intersection(parent_parts))


def check_privacy():
    # Generic secret & local-machine path firewall (zero personal identifiers embedded in source!)
    rules = [
        ("private_data_dir", re.compile(r"private_data", re.I)),
        ("local_home_path", re.compile(r"(?:/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+)")),
        ("local_systemone_rel", re.compile(r"\./systemone/")),
        ("api_token", re.compile(r"(?:gh[po]_[A-Za-z0-9]{10}|github_pat_[A-Za-z0-9_]{10}|hf_[A-Za-z0-9]{10}|sk-[A-Za-z0-9]{20})")),
        ("rfc1918_ip", re.compile(r"(?:\b192\.168\.\d+\.\d+|\b10\.\d+\.\d+\.\d+|\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)")),
        ("private_key", re.compile(r"BEGIN\s+(?:[A-Z0-9]+\s+)?PRIVATE\s+KEY", re.I)),
    ]
    forbidden_filenames = {"aft.csv", "aft_real_myths_500.json", "aft_excerpts.json"}
    forbidden_sha256s = {
        "5214f213b2ef99dded16fcd4f5fca9b1e2c0675aff99ea475b1839c21f441ba1",  # raw third-party aft.csv
    }
    # Use NUL-delimited `git ls-files` for tracked PLUS untracked non-ignored files so new runners are always scanned!
    git_res = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT), capture_output=True)
    git_oth = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=str(ROOT), capture_output=True)
    if git_res.returncode == 0 and git_res.stdout:
        raw_parts = [p for p in git_res.stdout.split(b"\0") if p]
        if git_oth.returncode == 0 and git_oth.stdout:
            raw_parts.extend(p for p in git_oth.stdout.split(b"\0") if p)
        seen = set()
        candidates = []
        for part in raw_parts:
            fp = ROOT / os.fsdecode(part)
            if fp not in seen:
                seen.add(fp)
                candidates.append(fp)
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
        if p.name.lower() in forbidden_filenames:
            hits.append((str(p.relative_to(ROOT)), ["forbidden_third_party_folklore_file"]))
            continue
        raw_bytes = p.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() in forbidden_sha256s:
            hits.append((str(p.relative_to(ROOT)), ["forbidden_third_party_folklore_sha256"]))
            continue
        if p.suffix.lower() in BINARY_EXTS:
            continue
        txt = raw_bytes.decode("utf-8", errors="ignore")
        matched_rules = [rname for rname, pat in rules if pat.search(txt)]
        if matched_rules:
            hits.append((str(p.relative_to(ROOT)), matched_rules))
    # Never echo matched credential/path fragments in failure output — report filename + rule names only!
    assert not hits, f"PRIVACY/SECRET FIREWALL FAILED (files & triggered rules): {hits}"
    print(f"[PASS] Privacy, license & secret firewall clean across all {len(candidates)} {'tracked' if using_git else 'scanned'} files.")


def check_py_compile():
    py_files = sorted(p for p in ROOT.rglob("*.py") if not _is_skipped(p))
    for pf in py_files:
        src = pf.read_text(encoding="utf-8")
        # In-memory Python 3.10+ parse & compile: works cleanly on read-only FS (no .pyc writes) and enforces >=3.10 syntax!
        ast.parse(src, filename=str(pf), feature_version=(3, 10))
        compile(src, str(pf), "exec")
    print(f"[PASS] In-memory Python 3.10+ syntax & bytecode compile succeeded on all {len(py_files)} Python files (zero .pyc writes).")


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
    assert len(figs) == 9, f"Expected 9 embedded figures, found {len(figs)}"
    for f in figs:
        png = paper.parent / f
        pdf = png.with_suffix(".pdf")
        arxiv_pdf = arxiv_dir / "figures" / pdf.name
        assert png.exists() and png.stat().st_size > 10_000, f"Missing PNG: {png}"
        assert pdf.exists() and pdf.stat().st_size > 5_000, f"Missing PDF: {pdf}"
        assert arxiv_pdf.exists() and arxiv_pdf.stat().st_size > 5_000, f"Missing arXiv PDF: {arxiv_pdf}"

    hero = paper.parent / "figures" / "hero_jung_plato_lattice.png"
    assert hero.exists() and hero.stat().st_size > 50_000, f"Missing hero PNG: {hero}"

    refs_md = [l for l in txt.split("## References", 1)[1].splitlines() if l.startswith("- ")]
    refs_bib = re.findall(r"^@[a-zA-Z]+\{", bib, flags=re.M)
    refs_bbl = re.findall(r"^\\bibitem\{", bbl, flags=re.M)
    assert len(refs_md) == 87, f"Expected 87 references in paper.md, found {len(refs_md)}"
    assert len(refs_bib) == 87, f"Expected 87 BibTeX entries in references.bib, found {len(refs_bib)}"
    assert len(refs_bbl) == 87, f"Expected 87 \\bibitem entries in main.bbl, found {len(refs_bbl)}"

    # Dual-audience Microsoft Word (.docx) review copy presence & archive validity
    docx_path = ROOT / "paper" / "The_Jung_Plato_Lattice_v6.docx"
    assert docx_path.exists() and docx_path.stat().st_size > 1_000_000 and zipfile.is_zipfile(docx_path), (
        f"Missing or invalid Word review copy: {docx_path}"
    )

    # Pure-ASCII check on IEEEtran.cls (arXiv pdflatex portability)
    cls_bytes = (arxiv_dir / "IEEEtran.cls").read_bytes()
    cls_non_ascii = [b for b in cls_bytes if b > 127]
    assert not cls_non_ascii, f"Non-ASCII bytes found in paper/arxiv/IEEEtran.cls: count={len(cls_non_ascii)}"

    # Third-party license attribution & v6 runner suite presence
    tp_lic = ROOT / "THIRD_PARTY_LICENSES.md"
    apache_lic = ROOT / "LICENSES" / "Apache-2.0.txt"
    assert tp_lic.exists() and tp_lic.stat().st_size > 500, f"Missing {tp_lic}"
    assert apache_lic.exists() and apache_lic.stat().st_size > 10_000, f"Missing {apache_lic}"
    v6_runners = sorted((ROOT / "scripts" / "v6").glob("*.py"))
    assert len(v6_runners) >= 30, f"Expected >=30 sanitized v6 scripts in scripts/v6/, found {len(v6_runners)}"
    for req_runner in (
        "fetch_trilogy_aft.py", "round8_overlap_audit.py", "e4b_transplant50.py",
        "e4b_transplant50_cuda.py", "analyze_3seed_251.py", "exp247_actor_binding.py",
        "round7_claim_c_seal.py", "crosslang_audit.py",
    ):
        assert (ROOT / "scripts" / "v6" / req_runner).exists(), f"Missing required v6 runner: {req_runner}"

    # arXiv LaTeX header & latex-arxiv-SKILL formatting hygiene checks
    top5 = "\n".join(tex.splitlines()[:5])
    assert r"\pdfoutput=1" in top5, "paper/arxiv/main.tex missing \\pdfoutput=1 in first 5 lines!"
    assert r"\AddToHook{env/thebibliography/begin}{\label{ReferencesStart}}" in tex, "Missing ReferencesStart hook!"
    assert r"\resizebox" not in tex, "latex-arxiv-SKILL forbids \\resizebox in main.tex!"
    assert "---" not in tex, "latex-rhythm-refiner forbids mid-sentence em-dashes (---) in main.tex!"
    non_ascii = sorted({(hex(ord(c)), c) for c in tex if ord(c) > 127})
    assert not non_ascii, f"Non-ASCII characters found in main.tex (breaks pdflatex portability): {non_ascii}"
    unescaped_tilde = re.findall(r"(?<!\$)(?<!\\sim)(?<!\\textasciitilde)~\d", tex)
    assert not unescaped_tilde, f"Unescaped ~digits found in main.tex (renders as blank space): {unescaped_tilde}"
    tex_models = _parse_tex_table(tex, "tab:models")
    assert len(tex_models) == 7, f"Expected 7 model rows in tab:models, got {len(tex_models)}"

    wav = ARCH / "aion_8b" / "assets" / "test_speech_16k.wav"
    assert wav.exists() and wav.stat().st_size > 10_000, f"Missing committed spoken-audio fixture: {wav}"

    # Guard against previously caught stale prose, erroneous CIs, citations & param counts!
    readme_txt = (ROOT / "README.md").read_text()
    for label, doc in [("paper.md", txt), ("main.tex", tex), ("README.md", readme_txt)]:
        for forbidden in (
            "+0.0569", "8,916,480", "+0.78,+10.22", "+0.78, +10.22", "0.0945",
            "lack a psyche", "four major breakthroughs", "Four major breakthroughs",
            "Xi et al.", "1468-5922.00282", "canonical Jungian", "0.04225", "0.03631",
            "Audit-Verified Preprint",
        ):
            assert forbidden not in doc, f"Forbidden/stale string '{forbidden}' found in {label}!"
    for label, doc in [("paper.md", txt), ("main.tex", tex)]:
        assert "8,916,608" in doc, f"Exact student total param count 8,916,608 missing from {label}!"
        assert "Statement on AI Assistance" in doc, f"Mandatory arXiv AI-use disclosure missing from {label}!"
    assert "AI Assistance" in readme_txt, "AI-use disclosure missing from README.md!"
    for needle in (
        "+0.0869", "+0.0780", "0.13803", "0.15729", "0.0417", "0.0405", "0.0414",
        "64.4%", "58.3%", "-0.106", "0.952", "+0.0789", "+0.0225", "+0.1204",
    ):
        assert needle in readme_txt, f"Headline value '{needle}' missing from README.md!"

    print(f"[PASS] Paper, Word .docx, licenses, v6 runners ({len(v6_runners)}), & arXiv hygiene verified: 9 figures + hero PNG, {len(refs_md)}/{len(refs_bib)}/{len(refs_bbl)} refs (md/bib/bbl), pure-ASCII LaTeX & IEEEtran.cls.")


def check_arxiv_tarball():
    arxiv_dir = ROOT / "paper" / "arxiv"
    tar_path = arxiv_dir / "arxiv_submission.tar.gz"
    assert tar_path.exists(), f"Missing arXiv tarball: {tar_path}"
    expected = {
        "main.tex",
        "references.bib",
        "main.bbl",
        "IEEEtran.cls",
        "figures/fig1_platonic_consensus.pdf",
        "figures/fig2_archetype_audit_and_teleport.pdf",
        "figures/fig3_loro_myth_amplification.pdf",
        "figures/fig4_5m_distillation_ujung.pdf",
        "figures/fig5_8b_stage_a_causal.pdf",
        "figures/fig6_epistemic_house_correctness.pdf",
        "figures/fig7_depth_and_actor_binding.pdf",
        "figures/fig8_e4b_50pct_transplant.pdf",
        "figures/fig9_historical_folklore_audit.pdf",
    }
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        names = [m.name.lstrip("./") for m in members]
        appledouble = [n for n in names if "/._" in n or n.startswith("._")]
        assert not appledouble, f"AppleDouble sidecars found in tarball: {appledouble}"
        assert set(names) == expected and len(names) == len(expected), (
            f"Tarball member mismatch! missing={expected - set(names)} extra={set(names) - expected}"
        )
        for m in members:
            norm = m.name.lstrip("./")
            assert m.uid == 0 and m.gid == 0, f"Non-zero uid/gid in tarball member {norm}: uid={m.uid}, gid={m.gid}"
            assert m.uname == "" and m.gname == "", f"Non-empty uname/gname in tarball member {norm}: {m.uname}/{m.gname}"
            xattr_keys = [k for k in m.pax_headers if "xattr" in k.lower() or "schily" in k.lower() or "apple" in k.lower()]
            assert not xattr_keys, f"Extended attribute PAX headers in {norm}: {xattr_keys}"
            disk_bytes = (arxiv_dir / norm).read_bytes()
            tar_bytes = tf.extractfile(m).read()
            assert hashlib.sha256(tar_bytes).hexdigest() == hashlib.sha256(disk_bytes).hexdigest(), (
                f"SHA-256 mismatch between tarball member '{norm}' and on-disk paper/arxiv/{norm}!"
            )
    print(f"[PASS] arXiv tarball clean: exact {len(expected)} allowlisted files, 0 AppleDouble/xattr headers, uid=0/gid=0, 100% SHA-256 parity with paper/arxiv/.")


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
    s = s.replace(r"\%", "%").replace(r"\_", "_").replace(r"\to", "->").replace(r"\times", "x").replace(r"\le", "<=").replace(r"\pm", "+/-")
    s = s.replace(r"\ ", " ")
    s = s.replace(r"^\dagger", "").replace(r"^*", "")
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
    # 2e. Experiment 237 Referee Controls (8 Non-Jungian Roles Table 2f & tab:nonjung, Order-Balanced bs60, Zero-PN items3)
    # =========================================================================
    nj_data = json.load(open(ARCH / "selfshape" / "nonjung_roles237_data.json"))
    assert len(nj_data) == 320
    r237 = json.load(open(ARCH / "fluent" / "plato" / "referee_ctrl237.json"))
    assert abs(r237["nonjung_roles_c5_geometry"]["effective_rank_pr_raw"] - 4.379) < 1e-3
    assert abs(r237["nonjung_roles_c5_geometry"]["effective_rank_pr_perp_val3"] - 4.74) < 1e-2
    assert abs(r237["items3_zero_proper_name_control"]["heldout_myth_pool_excluded"]["olmo3_7b"]["anchor"]["diff_no_pn"] - 0.0454) < 5e-4
    assert abs(r237["items3_zero_proper_name_control"]["heldout_myth_pool_excluded"]["phi4_mini"]["anchor"]["diff_no_pn"] + 0.0167) < 5e-4
    assert abs(r237["bow_swap_word_order_and_last_token"]["last_token"]["qwen35_9b"]["balanced_2fold_perp_v2"] - 0.757) < 1e-3
    md_nj_rows = _parse_md_table_after(paper_txt, "referee_ctrl237.json")
    assert len(md_nj_rows) == 6, f"Expected 6 rows in MD Table 2f, got {len(md_nj_rows)}"
    tex_nj = _parse_tex_table(tex_txt, "tab:nonjung")
    assert len(tex_nj) == 6, f"Expected 6 rows in tab:nonjung, got {len(tex_nj)}"
    for mk, rkey in t2b_models:
        r = r237["nonjung_roles_gemma_to_6"][mk]["raw"]
        p3 = r237["nonjung_roles_gemma_to_6"][mk]["perp_val3_dbl"]
        lr = r237["nonjung_roles_gemma_to_6"][mk]["lenres_top1_mean"]
        assert all(v["p_perm_max"] <= 0.001 for v in r["per_archetype"].values())
        assert all(v["p_perm_max"] <= 0.001 for v in p3["per_archetype"].values())
        checks = [
            (1, rf"^{r['top1_8way_mean']*100:.1f}%\s*\({r['top1_wins_over_bow_20']}/20\s*/\s*{r['top1_wins_over_tfidf_ridge_20']}/20\)$"),
            (2, rf"^{p3['top1_8way_mean']*100:.1f}%\s*\({p3['top1_wins_over_tfidf_ridge_20']}/20\)$"),
            (3, rf"^{r['macro_auroc_mean']:.3f}$"),
            (4, rf"^{p3['macro_auroc_mean']:.3f}$"),
            (5, rf"^{lr*100:.1f}%$"),
            (6, rf"^{r['within_setting_top1_mean']*100:.1f}%$"),
        ]
        _assert_row_regex(md_nj_rows, f"Gemma 4 → {rkey}", checks, f"MD Table 2f {mk}", exact_cols=7)
        _assert_row_regex(tex_nj, f"Gemma 4 -> {rkey}", checks, f"TeX tab:nonjung {mk}", exact_cols=7, cleaner=lambda s: s)
    for label, doc in [("paper.md", paper_txt), ("main.tex", tex_txt)]:
        for needle in ("4.38", "4.74", "+0.0454", "-0.0167", "75.7"):
            assert needle in doc, f"Experiment 237 headline '{needle}' missing from {label}!"

    # =========================================================================
    # 2f. Experiment 238 Mirrored 120-Pair Order-Balanced Swaps, Output Prob ΔNLL, & Shuffled PR Null (Table 2g & tab:bmirror)
    # =========================================================================
    bm_data = json.load(open(ARCH / "selfshape" / "bow_mirror238_data.json"))
    assert len(bm_data["bow_mirror_120"]) == 120
    b238 = json.load(open(ARCH / "fluent" / "plato" / "bow_mirror238.json"))
    assert abs(b238["lexical_baselines"]["bow_bm120_unigram_20split_mean"] - 0.5) < 1e-12
    assert abs(b238["role_pr_shuffled_null_c5"]["jung_8_c5"]["p_null_le_obs"] - 0.1158) < 5e-4
    assert abs(b238["role_pr_shuffled_null_c5"]["nonjung_8_c5"]["p_null_le_obs"] - 0.1078) < 5e-4
    assert all(p < 0.05 for p in b238["holm_adjusted_p"]["mean_pool"]["cv2_perp_v2_p_holm"].values())
    assert all(p < 0.05 for p in b238["holm_adjusted_p"]["mean_pool"]["zs_house_perp_v2_p_holm"].values())
    md_bm_rows = _parse_md_table_after(paper_txt, "bow_mirror238.json")
    assert len(md_bm_rows) == 6, f"Expected 6 rows in MD Table 2g, got {len(md_bm_rows)}"
    tex_bm = _parse_tex_table(tex_txt, "tab:bmirror")
    assert len(tex_bm) == 6, f"Expected 6 rows in tab:bmirror, got {len(tex_bm)}"
    for mk, rkey in t2b_models:
        vm = b238["transfer_bm120"]["mean_pool"][mk]
        vl = b238["transfer_bm120"]["last_token"][mk]
        hm_h = b238["holm_adjusted_p"]["mean_pool"]["zs_house_perp_v2_p_holm"][mk]
        hm_2 = b238["holm_adjusted_p"]["mean_pool"]["cv2_perp_v2_p_holm"][mk]
        hl_2 = b238["holm_adjusted_p"]["last_token"]["cv2_perp_v2_p_holm"][mk]
        l_ns = r"\s+n\.s\." if hl_2 >= 0.05 else ""
        if mk in b238["output_prob_nll_baseline"]:
            ob = b238["output_prob_nll_baseline"][mk]
            ols = vm["plausibility_control"]["ols_p2_vs_target_dnll"]
            c1_pat = rf"^{ob['mirror120_mean_nll_acc']*100:.1f}%\s*\({ob['both_mirrors_right_mean_nll']*100:.1f}%\)$"
            r_s = f"{ols['r_score_vs_dnll']:+.3f}"
            i_s = f"{ols['ols_intercept_at_dnll0']:+.3f}"
            lo_s = f"{ols['ols_intercept_ci95'][0]:+.3f}"
            hi_s = f"{ols['ols_intercept_ci95'][1]:+.3f}"
            sh_s = f"{ols['intercept_share_of_mean']*100:.0f}%"
            c5_md = rf"^r={re.escape(r_s)},\s*{re.escape(i_s)}\s*\[{re.escape(lo_s)},\s*{re.escape(hi_s)}\]\s*\({re.escape(sh_s)}\)$"
            c5_tx = rf"^{re.escape(r_s)},\s*{re.escape(i_s)}\s*\[{re.escape(lo_s)},\s*{re.escape(hi_s)}\]$"
        else:
            c1_pat = r"^(?:\(GGUF\)|-)$"
            c5_md = r"^\(GGUF — no logits\)$"
            c5_tx = r"^-$"
        vm_cos = f"{vm['cv2_perp_v2_cos']:+.3f}"
        vl_cos = f"{vl['cv2_perp_v2_cos']:+.3f}"
        common = [
            (1, c1_pat),
            (2, rf"^{vm['zs_house_perp_v2_120']*100:.1f}%\s*\({vm['zs_house_perp_v2_both_mirrors']*100:.1f}%,\s*{hm_h:.3f}\)$"),
            (3, rf"^{vm['cv2_perp_v2_120']*100:.1f}%\s*\({vm['cv2_perp_v2_both_mirrors']*100:.1f}%,\s*{re.escape(vm_cos)},\s*{hm_2:.3f}\)$"),
            (4, rf"^{vl['cv2_perp_v2_120']*100:.1f}%\s*\({vl['cv2_perp_v2_both_mirrors']*100:.1f}%,\s*{re.escape(vl_cos)},\s*{hl_2:.3f}{l_ns}\)$"),
        ]
        _assert_row_regex(md_bm_rows, f"Gemma 4 → {rkey}", common + [(5, c5_md)], f"MD Table 2g {mk}", exact_cols=6)
        _assert_row_regex(tex_bm, f"Gemma 4 -> {rkey}", common + [(5, c5_tx)], f"TeX tab:bmirror {mk}", exact_cols=6, cleaner=lambda s: s)
    for label, doc in [("paper.md", paper_txt), ("main.tex", tex_txt)]:
        for needle in ("5.533", "5.051", "0.116", "0.108", "25.0"):
            assert needle in doc, f"Experiment 238 headline '{needle}' missing from {label}!"

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

    # =========================================================================
    # 7. v6 / Zenodo v3.0 Audits: FULL cell + CI verification in BOTH main.tex AND paper.md!
    # =========================================================================
    _verify_v6_tables(paper_txt, tex_txt)

    print("[PASS] All 6 v5 pillars + ALL v6/Zenodo v3.0 internally specified audits (Claims A/B/C, Overlap R8, Exp 251, Dropped Exps 241/243/244), 24 student JSONs, ALL paper.md tables, AND ALL 15 numeric main.tex tables (including ALL CI endpoints) match cleanly.")


def _fmt_ci_pp(pair):
    return f"[{pair[0]*100:+.2f}, {pair[1]*100:+.2f}]"


def _ci_pp_regex(pair):
    lo = re.escape(f"{pair[0]*100:+.2f}")
    hi = re.escape(f"{pair[1]*100:+.2f}")
    return rf"^\[{lo},\s*{hi}\]$"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_v6_tables(paper_txt: str, tex_txt: str, e251_override=None):
    e247 = json.load(open(ARCH / "fluent" / "plato" / "v6_exp247_binding_results.json"))
    nat5 = json.load(open(ARCH / "fluent" / "plato" / "v6_within_model_native_5depth.json"))
    r4f = json.load(open(ARCH / "fluent" / "plato" / "v6_round4_folklore_results.json"))
    r5b = json.load(open(ARCH / "fluent" / "plato" / "v6_round5_binding_depth_results.json"))
    r6p = json.load(open(ARCH / "fluent" / "plato" / "v6_round6_final_polish.json"))
    r7c = json.load(open(ARCH / "fluent" / "plato" / "v6_round7_claim_c_seal.json"))
    r8o = json.load(open(ARCH / "fluent" / "plato" / "v6_round8_overlap_audit.json"))
    d5m = json.load(open(ARCH / "fluent" / "plato" / "v6_distill50_5m.json"))
    drop = json.load(open(ARCH / "fluent" / "plato" / "v6_dropped_and_depth_audits.json"))
    xling = json.load(open(ARCH / "crosslang" / "v6_crosslang_audit.json"))
    e251 = e251_override if e251_override is not None else json.load(open(ARCH / "aion_8b" / "runs" / "e4b_transplant50_exp251.json"))

    # Byte-for-byte script & data SHA-256 parity across all 5 active regenerated v6 runners
    v6_dir = ROOT / "scripts" / "v6"
    assert e251["script_sha256"] == _sha256_file(v6_dir / "analyze_3seed_251.py"), "script_sha256 mismatch: analyze_3seed_251.py!"
    assert r8o["script_sha256"] == _sha256_file(v6_dir / "round8_overlap_audit.py"), "script_sha256 mismatch: round8_overlap_audit.py!"
    assert r7c["script_sha256"] == _sha256_file(v6_dir / "round7_claim_c_seal.py"), "script_sha256 mismatch: round7_claim_c_seal.py!"
    assert e247["manifest"]["script_sha256"] == _sha256_file(v6_dir / "exp247_actor_binding.py"), "script_sha256 mismatch: exp247_actor_binding.py!"
    assert e247["manifest"]["data_sha256"] == _sha256_file(ARCH / "selfshape" / "bm_actor2x2_247_data.json"), "data_sha256 mismatch: bm_actor2x2_247_data.json!"
    assert xling["script_sha256"] == _sha256_file(v6_dir / "crosslang_audit.py"), "script_sha256 mismatch: crosslang_audit.py!"
    assert xling["items_sha256"] == _sha256_file(ARCH / "crosslang" / "items.jsonl"), "items_sha256 mismatch: crosslang/items.jsonl!"
    assert xling["checked_sha256"] == _sha256_file(ARCH / "crosslang" / "checked.jsonl"), "checked_sha256 mismatch: crosslang/checked.jsonl!"
    assert xling["sign_sha256"] == _sha256_file(ARCH / "crosslang" / "sign.jsonl"), "sign_sha256 mismatch: crosslang/sign.jsonl!"

    # Cross-Lingual Causal Story Form audit invariants (v6_crosslang_audit.json)
    assert len(xling["models"]) == 5 and len(xling["languages"]) == 4 and xling["c1_primary"]["n_patterns"] == 78
    assert abs(xling["c1_primary"]["mean"] - 0.0789) < 5e-4
    assert abs(xling["c1_edited_clauses_only"]["mean"] - 0.0564) < 5e-4
    assert abs(xling["c1_story_beyond_clauses_paired"]["mean"] - 0.0225) < 5e-4
    assert abs(xling["clear_sign_matched_subset"]["mean"] - 0.1204) < 5e-4
    assert abs(xling["fixed_within_outcome_shuffle_floor"]["mean"] - 0.0046) < 5e-4
    for label, doc in [("paper.md", paper_txt), ("main.tex", tex_txt)]:
        for ndl in ("+0.0789", "+0.0630", "+0.0944", "+0.0564", "+0.0225", "+0.0139", "+0.0305", "+0.0178", "-0.0133", "+0.0484", "+0.1204", "+0.0046"):
            assert ndl in doc, f"Cross-lingual audit figure '{ndl}' missing from {label}!"

    expected_l12_keys = {
        "base", "ctrl50", "ctrl50_s252", "ctrl50_s253",
        "c4_50", "c4_50_s252", "c4_50_s253",
        "c3_50", "c3_50_s252", "c3_50_s253",
    }
    assert set(e251["l12_invariance_max_abs_diff"].keys()) == expected_l12_keys, (
        f"Unexpected l12_invariance_max_abs_diff keys: {set(e251['l12_invariance_max_abs_diff'].keys())}"
    )
    assert all(v == 0.0 for v in e251["l12_invariance_max_abs_diff"].values())
    assert "quick_blind" not in json.dumps(e251), "Stale quick_blind key found anywhere inside e251 JSON!"
    assert abs(e247["actor2x2_5depth_curves"]["d50"]["mean"]["gemma4_e4b"]["cv2_val5"] - 0.7301) < 5e-4
    assert abs(e247["actor2x2_5depth_curves"]["d50"]["mean"]["qwen35_9b"]["cv2_val5"] - 0.7309) < 5e-4
    assert abs(nat5["within_model_native_5depth"]["mean"]["bm_actor2x2_247"]["d50"]["mean_6models"] - 0.7715) < 5e-4
    assert abs(r4f["bases_480_v_dir"]["Q_jung7_labeled"] - 0.13803) < 1e-5
    assert abs(r7c["jung_shuffle_10k"]["n480"]["holm3_with_200base_fam_jobs"] - 0.0417) < 5e-4
    assert abs(drop["exp241_centered_ablation_bm_rel242_log"]["interaction_fc_minus_topic"] + 0.007) < 5e-4
    assert drop["exp244_epistemic_matched120"]["per_model"]["gemma4_e4b"]["buddy"]["admissibility_pass_le_0p70"] is False
    # Round-8 overlap audit invariants (all 1,518 rows, 3,598 passages, 7 >=12w myth hits, n=477 and n=473)
    assert r8o["aft_csv_rows_scanned"] == 1518
    assert r8o["passages_scanned_all"] == 3598
    assert r8o["passages_scanned_neutral"] == 2398
    assert r8o["neutral_passages_ge12_word_overlaps_vs_all_1518_aft_csv"] == 0
    assert len(r8o["all_myth_pool_ge12_word_overlaps_vs_all_1518_aft_csv"]) == 7
    assert all(h["in_scored_excerpt"] is False for h in r8o["all_myth_pool_ge12_word_overlaps_vs_all_1518_aft_csv"])
    assert len(r8o["full_tale_shared_public_domain_editions_in_gutenberg_myth_pool"]) == 6
    assert r8o["max_contiguous_word_overlap_neutral_passages_vs_scored_500"]["max_words"] == 7
    assert r8o["max_contiguous_word_overlap_all_passages_vs_scored_500"]["max_words"] == 10
    n477 = r8o["sensitivity_excluding_3_full_tale_edition_matches"]["n477_excl_3_fulltale"]
    n473 = r8o["sensitivity_excluding_3_full_tale_edition_matches"]["n473_excl_3_and_4framing"]
    assert abs(n477["v_dir"]["Q_jung7_labeled"] - 0.13772) < 1e-5
    assert abs(n477["jung_10k_shuffle"]["holm3_upper"] - 0.0405) < 1e-5
    assert abs(n477["v_dir"]["PCA7_jung_setcent"] - 0.15685) < 1e-5
    assert abs(n473["v_dir"]["Q_jung7_labeled"] - 0.13687) < 1e-5
    assert abs(n473["jung_10k_shuffle"]["holm3_upper"] - 0.0414) < 1e-5
    assert abs(n473["v_dir"]["PCA7_jung_setcent"] - 0.15498) < 1e-5
    for label, doc, pct in [("paper.md", paper_txt, "%"), ("main.tex", tex_txt, r"\%")]:
        for needle in (
            "0.13772", "0.0405", "0.15685", "0.13687", "0.0414", "0.15498",
            "134/10000", "0.0135", "0.07064", "137/10000", "0.0138", "0.07195",
            "0.058", "0.118", "0.054", "0.114",
            "95.5", "96.3", "82.97", f"64.4{pct}", f"58.3{pct}", f"26.0{pct}", f"28.2{pct}",
            "+0.0888, +0.0864, +0.0855", "+0.0799, +0.0777, +0.0765",
            f"18.8{pct}", f"19.4{pct}", "0.0042",
            "[+0.00071, +0.00713]", "[+0.00062, +0.00704]",
            "+0.16", "-0.35", "10.2", "9.0", "0.5733", "0.5104", "10/10",
        ):
            assert needle in doc, f"Verified v6 audit figure '{needle}' missing from {label}!"
    # 5M d50 screen sanity
    assert abs(d5m["consensus50_diag"]["d50"]["olmo3_7b"]["C4"]["r2_opt"] - 0.5564) < 5e-4

    # -------------------------------------------------------------------------
    # 7a. tab:actor_depth (TeX) & Table 2h (MD) — ALL 6 rows × ALL 10 cols incl. ALL CI endpoints!
    # -------------------------------------------------------------------------
    tex_ad = _parse_tex_table(tex_txt, "tab:actor_depth")
    md_ad = _parse_md_table_after(paper_txt, "Protocol (6 Standard Models, `bm_actor2x2_247`)")
    assert len(tex_ad) == 6, f"Expected 6 rows in tab:actor_depth, got {len(tex_ad)}"
    assert len(md_ad) == 6, f"Expected 6 rows in MD Table 2h, got {len(md_ad)}"

    bonf8 = r6p["claim_b_cross_family_bonf8_across_both_pools"]
    nll_d50 = r5b["d50_refit_in_boot_nll_and_strata"]
    for tex_key, md_key, pool, src in [
        ("G4->5 (mean", "Cross-Fam G4 → 5 (mean", "mean", "gemma4_e4b"),
        ("Qw->5 (mean", "Cross-Fam Qw → 5 (mean", "mean", "qwen35_9b"),
        ("G4->5 (last", "Cross-Fam G4 → 5 (last", "last", "gemma4_e4b"),
        ("Qw->5 (last", "Cross-Fam Qw → 5 (last", "last", "qwen35_9b"),
    ]:
        vals = [f"{e247['actor2x2_5depth_curves'][d][pool][src]['cv2_val5']*100:.2f}%" for d in ("d15", "d30", "d50", "d70", "d85")]
        ci50_30 = bonf8[pool][src]["d50_minus_d30_bonf8_across_pools"]["basic_reflected"]
        ci50_70 = bonf8[pool][src]["d50_minus_d70_bonf8_across_pools"]["basic_reflected"]
        vw = f"{nll_d50[pool][src]['cv2_minus_whole_nll_refit_boot']['obs']*100:+.2f}"
        vc = f"{nll_d50[pool][src]['cv2_minus_changed_nll_refit_boot']['obs']*100:+.2f}"
        checks = (
            [(i + 1, rf"^{re.escape(v)}$") for i, v in enumerate(vals)]
            + [
                (6, _ci_pp_regex(ci50_30)),
                (7, _ci_pp_regex(ci50_70)),
                (8, rf"^{re.escape(vw)}$"),
                (9, rf"^{re.escape(vc)}$"),
            ]
        )
        _assert_row_regex(tex_ad, tex_key, checks, f"TeX tab:actor_depth {tex_key}", exact_cols=10, cleaner=lambda s: s)
        _assert_row_regex(md_ad, md_key, checks, f"MD Table 2h {md_key}", exact_cols=10)

    # Native 6-model rows (mean & last) in BOTH TeX tab:actor_depth & MD Table 2h
    for tex_key, md_key, pool in [
        ("Native Mean (6-Mod)", "Native Within-Model Mean (6-Mod)", "mean"),
        ("Native Last (6-Mod)", "Native Within-Model Last (6-Mod)", "last"),
    ]:
        vals = [f"{nat5['within_model_native_5depth'][pool]['bm_actor2x2_247'][d]['mean_6models']*100:.2f}%" for d in ("d15", "d30", "d50", "d70", "d85")]
        nat_ci = r5b["actor2x2_depth_percentile_and_basic_bonf"][pool]["within_model_native_bonf2"]
        ci50_30 = nat_ci["d50_minus_d30_bonf2"]["basic_reflected"]
        ci50_70 = nat_ci["d50_minus_d70_bonf2"]["basic_reflected"]
        checks_tex = (
            [(i + 1, rf"^{re.escape(v)}$") for i, v in enumerate(vals)]
            + [(6, _ci_pp_regex(ci50_30)), (7, _ci_pp_regex(ci50_70)), (8, r"^-$"), (9, r"^-$")]
        )
        checks_md = (
            [(i + 1, rf"^{re.escape(v)}$") for i, v in enumerate(vals)]
            + [(6, _ci_pp_regex(ci50_30)), (7, _ci_pp_regex(ci50_70)), (8, r"^—$"), (9, r"^—$")]
        )
        _assert_row_regex(tex_ad, tex_key, checks_tex, f"TeX tab:actor_depth {tex_key}", exact_cols=10, cleaner=lambda s: s)
        _assert_row_regex(md_ad, md_key, checks_md, f"MD Table 2h {md_key}", exact_cols=10)

    # -------------------------------------------------------------------------
    # 7b. tab:folklore (TeX) & Table 2i (MD) — ALL 7 rows × ALL 6 cols!
    # -------------------------------------------------------------------------
    tex_fk = _parse_tex_table(tex_txt, "tab:folklore")
    md_fk = _parse_md_table_after(paper_txt, "Subspace Basis (7D in Neutral `C5`)")
    assert len(tex_fk) == 7, f"Expected 7 rows in tab:folklore, got {len(tex_fk)}"
    assert len(md_fk) == 7, f"Expected 7 rows in MD Table 2i, got {len(md_fk)}"

    b480 = r4f["bases_480_v_dir"]
    b476 = r6p["claim_c_476_complete_structural_suite"]["bases_v_dir"]
    shuf480 = r7c["shuffle_200base_with_holm3"]["n480"]
    shuf476 = r7c["shuffle_200base_with_holm3"]["n476"]
    h7_480 = r7c["fixed_source_div_stratified_holm7_pct_and_basic"]["n480"]
    h7_476 = r7c["fixed_source_div_stratified_holm7_pct_and_basic"]["n476"]
    j10k = r7c["jung_shuffle_10k"]

    def _h7_pair(h7, key):
        return f"{h7['holm7_percentile'][key]:.3f}", f"{h7['holm7_basic'][key]:.3f}"

    fk_spec = [
        ("PCA7_jung_unlabeled", None, None,
         rf"^{_h7_pair(h7_480, 'Q_jung7_minus_PCA7_jung_unlabeled')[0]}\s*/\s*{_h7_pair(h7_480, 'Q_jung7_minus_PCA7_jung_unlabeled')[1]}\s*\(476:\s*{_h7_pair(h7_476, 'Q_jung7_minus_PCA7_jung_unlabeled')[0]}\s*/\s*{_h7_pair(h7_476, 'Q_jung7_minus_PCA7_jung_unlabeled')[1]}\)$"),
        ("PCA7_jung_setcent", None, None,
         rf"^{_h7_pair(h7_480, 'Q_jung7_minus_PCA7_jung_setcent')[0]}\s*/\s*{_h7_pair(h7_480, 'Q_jung7_minus_PCA7_jung_setcent')[1]}\s*\(476:\s*{_h7_pair(h7_476, 'Q_jung7_minus_PCA7_jung_setcent')[0]}\s*/\s*{_h7_pair(h7_476, 'Q_jung7_minus_PCA7_jung_setcent')[1]}\)$"),
        ("PCA7_fam_unlabeled", None, None, r"^(?:—|-)$"),
        ("Q_jung7_labeled",
         rf"^{shuf480['per_corpus']['jung']['null_200_mean']:.5f}\s*/\s*{shuf476['per_corpus']['jung']['null_200_mean']:.5f}$",
         rf"^{j10k['n480']['holm3_with_200base_fam_jobs']:.4f}\s*/\s*{j10k['n476']['holm3_with_200base_fam_jobs']:.4f}(?:\s*\(10k\))?$",
         r"^Reference$"),
        ("Q_fam7_labeled",
         rf"^{shuf480['per_corpus']['fam']['null_200_mean']:.5f}\s*/\s*{shuf476['per_corpus']['fam']['null_200_mean']:.5f}$",
         rf"^{shuf480['holm3']['fam']:.3f}\s*/\s*{shuf476['holm3']['fam']:.3f}\s+n\.s\.(?:\s*\(200\))?$",
         rf"^{_h7_pair(h7_480, 'Q_jung7_minus_Q_fam7')[0]}\s*/\s*{_h7_pair(h7_480, 'Q_jung7_minus_Q_fam7')[1]}\s*\(476:\s*{_h7_pair(h7_476, 'Q_jung7_minus_Q_fam7')[0]}\s*/\s*{_h7_pair(h7_476, 'Q_jung7_minus_Q_fam7')[1]}\s+n\.s\.\)$"),
        ("PCA7_jobs_unlabeled", None, None,
         rf"^vs\.\s*PCA7_jung:\s*{_h7_pair(h7_480, 'PCA7_jung_minus_PCA7_jobs')[0]}\s*/\s*{_h7_pair(h7_480, 'PCA7_jung_minus_PCA7_jobs')[1]}$"),
        ("Q_jobs7_labeled",
         rf"^{shuf480['per_corpus']['jobs']['null_200_mean']:.5f}\s*/\s*{shuf476['per_corpus']['jobs']['null_200_mean']:.5f}$",
         rf"^{shuf480['holm3']['jobs']:.3f}\s*/\s*{shuf476['holm3']['jobs']:.3f}\s+n\.s\.(?:\s*\(200\))?$",
         rf"^{_h7_pair(h7_480, 'Q_jung7_minus_Q_jobs7')[0]}\s*/\s*{_h7_pair(h7_480, 'Q_jung7_minus_Q_jobs7')[1]}\s*\(both splits\)$"),
    ]
    for bk, c3_pat, c4_pat, c5_pat in fk_spec:
        dash = r"^(?:—|-)$"
        checks = [
            (1, rf"^{b480[bk]:.5f}$"),
            (2, rf"^{b476[bk]:.5f}$"),
            (3, c3_pat or dash),
            (4, c4_pat or dash),
            (5, c5_pat),
        ]
        _assert_row_regex(tex_fk, bk, checks, f"TeX tab:folklore {bk}", exact_cols=6, cleaner=lambda s: s)
        _assert_row_regex(md_fk, bk, checks, f"MD Table 2i {bk}", exact_cols=6)

    # -------------------------------------------------------------------------
    # 7c. tab:folk_transfer (TeX) & Table 2j (MD) — ALL 4 splits × ALL 7 cols!
    # -------------------------------------------------------------------------
    tex_ft = _parse_tex_table(tex_txt, "tab:folk_transfer")
    md_ft = _parse_md_table_after(paper_txt, "True-9 Western→Non-Western Refit Split")
    assert len(tex_ft) == 4, f"Expected 4 rows in tab:folk_transfer, got {len(tex_ft)}"
    assert len(md_ft) == 4, f"Expected 4 rows in MD Table 2j, got {len(md_ft)}"

    for rkey, obj, neu_key, bal_key, ch_key in [
        ("480 Main", r6p["claim_c_true_9class_refit_480"], "neural", "balanced_acc", "holm8_chance"),
        ("476 Main", r7c["transfer_476_true9_20k"], "neural_point_estimates", "bal_acc", "holm8_chance_2k"),
        ("480 Excl-15", r6p["claim_c_true_9class_refit_480_excl15"], "neural", "balanced_acc", "holm8_chance"),
        ("476 Excl-15", r7c["transfer_476_excl15_true9_20k"], "neural_point_estimates", "bal_acc", "holm8_chance_2k"),
    ]:
        lex_lo = min(v["macro_auroc"] for v in obj["lexical_baselines"].values())
        lex_hi = max(v["macro_auroc"] for v in obj["lexical_baselines"].values())
        neu_lo = min(v["macro_auroc"] for v in obj[neu_key].values())
        neu_hi = max(v["macro_auroc"] for v in obj[neu_key].values())
        bal_lo = min(v[bal_key] for v in obj[neu_key].values()) * 100.0
        bal_hi = max(v[bal_key] for v in obj[neu_key].values()) * 100.0
        h8_lex = sum(1 for v in obj["holm8_all3_20k"].values() if v < 0.05)
        h8_ch = sum(1 for v in obj[ch_key].values() if v < 0.05)
        ch_ceil = 0.006 if max(obj[ch_key].values()) <= 0.006 else 0.008
        checks = [
            (1, rf"^{obj['n_train']}\s*/\s*{obj['n_test']}$"),
            (2, rf"^{lex_lo:.4f}(?:–|-){lex_hi:.4f}$"),
            (3, rf"^{neu_lo:.4f}(?:–|-){neu_hi:.4f}$"),
            (4, rf"^{bal_lo:.2f}%(?:–|-){bal_hi:.2f}%$"),
            (5, rf"^{h8_lex}/8$"),
            (6, rf"^{h8_ch}/8\s*\((?:≤|<=)\s*{ch_ceil:.3f}\)$"),
        ]
        _assert_row_regex(tex_ft, rkey, checks, f"TeX tab:folk_transfer {rkey}", exact_cols=7, cleaner=lambda s: s)
        _assert_row_regex(md_ft, rkey, checks, f"MD Table 2j {rkey}", exact_cols=7)

    # -------------------------------------------------------------------------
    # 7d. tab:exp251 (TeX) & Table 4b (MD) — 3-seed (251-253) ALL 5 rows × ALL 7 cols + 10/10 multimodal + canonical OOF CIs!
    # -------------------------------------------------------------------------
    assert e251["n_seeds"] == 3 and e251["seeds"] == [251, 252, 253]
    assert len(e251["l12_invariance_max_abs_diff"]) == 10 and all(v == 0.0 for v in e251["l12_invariance_max_abs_diff"].values())
    assert "weight_drift_mlp_l19_l20" in e251["train_config"]
    # Verify ALL 10/10 multimodal checkpoints directly (base + 3 arms × 3 seeds)
    mm_base = e251["lm_guardrails"]["base"]["multimodal"]
    assert mm_base["vision_ok"] is True and mm_base["audio_ok"] is True, "Base multimodal check failed!"
    for ak in ("ctrl50", "c4_50", "c3_50"):
        assert e251["lm_guardrails"][ak]["multimodal_all_seeds_pass"] is True, f"{ak} multimodal_all_seeds_pass != True!"
        for sd in ("251", "252", "253"):
            mm = e251["lm_guardrails"][ak]["per_seed"][sd]["multimodal"]
            assert mm["vision_ok"] is True and mm["audio_ok"] is True, f"Multimodal check failed for {ak} seed {sd}!"
            assert "quick_blind_phi4" not in e251["lm_guardrails"][ak]["per_seed"][sd], f"Stale quick_blind key leaked in {ak} s{sd}!"

    tex_251 = _parse_tex_table(tex_txt, "tab:exp251")
    md_251 = _parse_md_table_after(paper_txt, "8B Gemma 4 E4B Arm (50%, L20, 3 Seeds, `e4b_transplant50_exp251.json`)")
    assert len(tex_251) == 5, f"Expected 5 rows in tab:exp251, got {len(tex_251)}"
    assert len(md_251) == 5, f"Expected 5 rows in MD Table 4b, got {len(md_251)}"
    for ak, tkey, mkey in [
        ("base", "Base E4B", "Base E4B (Unmodified, 50%)"),
        ("ctrl50", "ctrl50 (w=0)", "Matched CE+KL ctrl50"),
        ("c4_50", "c4_50 (+G4)", "4-Fam c4_50"),
        ("c3_50", "c3_50 (0 G4)", "Rival-Only c3_50"),
    ]:
        p4 = e251["blind_50pct_alignment"][ak]["blind_phi4_50"]
        ol = e251["blind_50pct_alignment"][ak]["blind_olmo3_50"]
        p4_r2o = p4.get("r2_opt_fp", p4["r2_opt"])
        p4_t1 = p4.get("top1_fp", p4["top1"])
        ol_r2o = ol.get("r2_opt_fp", ol["r2_opt"])
        ol_t1 = ol.get("top1_fp", ol["top1"])
        nll = e251["lm_guardrails"][ak]["held_gut_nll_mean"]
        mcq = e251["lm_guardrails"][ak]["proc_held"]["overall"]
        nat_a = e251["probing_50pct"][ak]["within_native"]["mean"]["bm_actor2x2_247"]["acc"] * 100.0
        mcq_w1 = f"{mcq['with']*100 + 1e-9:.1f}"
        mcq_g1 = f"{mcq['gain']*100 + 1e-9:.1f}"
        if ak == "base":
            p4_tex = rf"^{p4['cos']:.4f}\s*/\s*\+{p4['r2_frob']:.4f}\s*\({p4_r2o:.3f},\s*{p4_t1*100:.1f}%\)$"
            ol_tex = rf"^{ol['cos']:.4f}\s*/\s*\+{ol['r2_frob']:.4f}\s*\({ol_r2o:.3f},\s*{ol_t1*100:.1f}%\)$"
            p4_md, ol_md = p4_tex, ol_tex
            va_tex, va_md = r"^0\.0\s*/\s*Pass$", r"^0\.0\s*/\s*Pass$"
        else:
            p4_sd = p4["sd"]["r2_frob"]
            ol_sd = ol["sd"]["r2_frob"]
            p4_tex = rf"^{p4['cos']:.4f}\s*/\s*\+{p4['r2_frob']:.4f}\+/-{p4_sd:.4f}\s*\({p4_r2o:.3f},\s*{p4_t1*100:.1f}%\)$"
            ol_tex = rf"^{ol['cos']:.4f}\s*/\s*\+{ol['r2_frob']:.4f}\+/-{ol_sd:.4f}\s*\({ol_r2o:.3f},\s*{ol_t1*100:.1f}%\)$"
            p4_md = rf"^{p4['cos']:.4f}\s*/\s*\+{p4['r2_frob']:.4f}\s*±\s*{p4_sd:.4f}\s*\({p4_r2o:.3f},\s*{p4_t1*100:.1f}%\)$"
            ol_md = rf"^{ol['cos']:.4f}\s*/\s*\+{ol['r2_frob']:.4f}\s*±\s*{ol_sd:.4f}\s*\({ol_r2o:.3f},\s*{ol_t1*100:.1f}%\)$"
            va_tex, va_md = r"^0\.0\s*/\s*3/3$", r"^0\.0\s*/\s*3/3 Pass$"
        _assert_row_regex(tex_251, tkey, [
            (1, p4_tex),
            (2, ol_tex),
            (3, rf"^{nll:.4f}$"),
            (4, rf"^{re.escape(mcq_w1)}%\s*\(\+{re.escape(mcq_g1)}\)$"),
            (5, rf"^{nat_a:.2f}%$"),
            (6, va_tex),
        ], f"TeX tab:exp251 {ak}", exact_cols=7, cleaner=lambda s: s)
        _assert_row_regex(md_251, mkey, [
            (1, p4_md),
            (2, ol_md),
            (3, rf"^{nll:.4f}$"),
            (4, rf"^{mcq['with']*100:.2f}%\s*\(\+{re.escape(mcq_g1)}\)$"),
            (5, rf"^{nat_a:.2f}%$"),
            (6, va_md),
        ], f"MD Table 4b {ak}", exact_cols=7)
    # Paired 3-seed Delta row (c3_50 - ctrl50): verify ALL 6 data columns (including MCQ col 4 & params col 6!)
    pd3 = e251["blind_50pct_paired_deltas_3seed"]["c3_50_minus_ctrl50"]
    dp4_c = pd3["phi4_mini"]["delta_cos_pt"]["mean"]
    dp4_r = pd3["phi4_mini"]["delta_r2_frob"]["mean"]
    dp4_rsd = pd3["phi4_mini"]["delta_r2_frob"]["sd"]
    dp4_t = pd3["phi4_mini"]["delta_top1"]["mean"] * 100.0
    dol_c = pd3["olmo3_7b"]["delta_cos_pt"]["mean"]
    dol_r = pd3["olmo3_7b"]["delta_r2_frob"]["mean"]
    dol_rsd = pd3["olmo3_7b"]["delta_r2_frob"]["sd"]
    dol_t = pd3["olmo3_7b"]["delta_top1"]["mean"] * 100.0
    dnll = pd3["lm_and_probing"]["held_gut_nll_delta"]["mean"]
    dnll_sd = pd3["lm_and_probing"]["held_gut_nll_delta"]["sd"]
    dmcq = pd3["lm_and_probing"]["proc_held_with_delta"]["mean"] * 100.0
    dmcq_sd = pd3["lm_and_probing"]["proc_held_with_delta"]["sd"] * 100.0
    dnat = pd3["lm_and_probing"]["within_native_actor_mean_delta"]["mean"] * 100.0
    dnat_sd = pd3["lm_and_probing"]["within_native_actor_mean_delta"]["sd"] * 100.0
    _assert_row_regex(tex_251, "c3_50-ctrl50", [
        (1, rf"^{re.escape(f'{dp4_c:+.4f}')}\s*/\s*{re.escape(f'{dp4_r:+.4f}')}\+/-{dp4_rsd:.4f}\s*\({re.escape(f'{dp4_t:+.1f}')}\s*pp\)$"),
        (2, rf"^{re.escape(f'{dol_c:+.4f}')}\s*/\s*{re.escape(f'{dol_r:+.4f}')}\+/-{dol_rsd:.4f}\s*\({re.escape(f'{dol_t:+.1f}')}\s*pp\)$"),
        (3, rf"^{re.escape(f'{dnll:+.4f}')}$"),
        (4, rf"^{abs(dmcq):.1f}\s*pp\s*\(n\.s\.\)$"),
        (5, rf"^{re.escape(f'{dnat:+.2f}')}$"),
        (6, r"^0\s*extra$"),
    ], "TeX tab:exp251 Delta", exact_cols=7, cleaner=lambda s: s)
    _assert_row_regex(md_251, "Paired Delta (c3_50 − ctrl50", [
        (1, rf"^{re.escape(f'{dp4_c:+.4f}')}\s*/\s*{re.escape(f'{dp4_r:+.4f}')}\s*±\s*{dp4_rsd:.4f}\s*\({re.escape(f'{dp4_t:+.1f}')}\s*pp\)$"),
        (2, rf"^{re.escape(f'{dol_c:+.4f}')}\s*/\s*{re.escape(f'{dol_r:+.4f}')}\s*±\s*{dol_rsd:.4f}\s*\({re.escape(f'{dol_t:+.1f}')}\s*pp\)$"),
        (3, rf"^{re.escape(f'{dnll:+.4f}')}\s*±\s*{dnll_sd:.4f}$"),
        (4, rf"^{abs(dmcq):.2f}\s*±\s*{dmcq_sd:.2f}\s*pp\s*\(n\.s\.\)$"),
        (5, rf"^{re.escape(f'{dnat:+.2f}')}\s*±\s*{dnat_sd:.2f}\s*pp$"),
        (6, r"^0\s*extra\s*params$"),
    ], "MD Table 4b Delta", exact_cols=7)

    # Verify Exp 251 canonical OOF passage bootstrap CI endpoints (target- AND order-bound regex!) in BOTH TeX & MD!
    cb_p = pd3["phi4_mini"]["clause_cos_3seed_mean_paired_boot"]
    cb_o = pd3["olmo3_7b"]["clause_cos_3seed_mean_paired_boot"]
    gb_p = pd3["phi4_mini"]["group_cos_3seed_mean_paired_boot"]
    gb_o = pd3["olmo3_7b"]["group_cos_3seed_mean_paired_boot"]

    def _oof_clause_pat(cb, target_lbl, wrap_ch):
        w = re.escape(wrap_ch)
        obs = re.escape(f"{cb['obs']:+.5f}")
        plo, phi = re.escape(f"{cb['ci95_pct'][0]:+.5f}"), re.escape(f"{cb['ci95_pct'][1]:+.5f}")
        blo, bhi = re.escape(f"{cb['ci95_basic'][0]:+.5f}"), re.escape(f"{cb['ci95_basic'][1]:+.5f}")
        return (
            rf"{w}{obs}{w}\s*\("
            rf"(?:{w})?pct\s+(?:{w})?\[{plo},\s*{phi}\]{w},\s*"
            rf"(?:{w})?basic\s+(?:{w})?\[{blo},\s*{bhi}\]{w}\)\s*on\s*{re.escape(target_lbl)}"
        )

    def _oof_group_pat(gp, go, wrap_ch):
        w = re.escape(wrap_ch)
        po, plo, phi = re.escape(f"{gp['obs']:+.5f}"), re.escape(f"{gp['ci95_pct'][0]:+.5f}"), re.escape(f"{gp['ci95_pct'][1]:+.5f}")
        oo, olo, ohi = re.escape(f"{go['obs']:+.5f}"), re.escape(f"{go['ci95_pct'][0]:+.5f}"), re.escape(f"{go['ci95_pct'][1]:+.5f}")
        return (
            rf"{w}{po}{w}\s*\({w}\[{plo},\s*{phi}\]{w}\)\s*and\s*"
            rf"{w}{oo}{w}\s*\({w}\[{olo},\s*{ohi}\]{w}\)"
        )

    for label, doc, wch in [("paper.md", paper_txt, "`"), ("main.tex", tex_txt, "$")]:
        assert re.search(_oof_clause_pat(cb_p, "Phi-4", wch), doc), f"Target/order-bound Phi-4 OOF clause CI mismatch in {label}!"
        assert re.search(_oof_clause_pat(cb_o, "OLMo 3", wch), doc), f"Target/order-bound OLMo 3 OOF clause CI mismatch in {label}!"
        assert re.search(_oof_group_pat(gb_p, gb_o, wch), doc), f"Target/order-bound equal-group OOF CI mismatch in {label}!"

    # Verify Abstract & Conclusion carry exact JSON-backed Exp 251 & Claim C headlines in BOTH docs
    m_tex_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_txt, re.S)
    m_tex_con = re.search(r"\\section\{Conclusion\}(.*?)\\FloatBarrier", tex_txt, re.S)
    m_md_abs = re.search(r"## Abstract\n+(.*?)\n+>", paper_txt, re.S)
    m_md_con = re.search(r"## 8\. Conclusion\n+(.*?)\n+---", paper_txt, re.S)
    assert m_tex_abs and m_tex_con and m_md_abs and m_md_con, "Missing Abstract or Conclusion block!"
    for blk_name, blk_txt, pct, extra in [
        ("main.tex Abstract", m_tex_abs.group(1), r"\%", ("-0.106", "0.952")),
        ("main.tex Conclusion", m_tex_con.group(1), r"\%", ("0.13803", "0.07300")),
        ("paper.md Abstract", m_md_abs.group(1), "%", ("-0.106", "0.952")),
        ("paper.md Conclusion", m_md_con.group(1), "%", ("0.13803", "0.07300")),
    ]:
        for ndl in (f"{dp4_r:+.4f}", f"{dol_r:+.4f}", f"64.4{pct}", f"58.3{pct}", "0.0417") + extra:
            assert ndl in blk_txt, f"Headline '{ndl}' missing from {blk_name}!"


def check_verifier_corruption_regression():
    """In-memory self-test proving that 12 injected corruptions across TeX, MD, and JSON raise AssertionError."""
    paper_txt = (ROOT / "paper" / "paper.md").read_text()
    tex_txt = (ROOT / "paper" / "arxiv" / "main.tex").read_text()
    e251_base = json.load(open(ARCH / "aion_8b" / "runs" / "e4b_transplant50_exp251.json"))

    def _expect_fail(desc, fn):
        try:
            fn()
        except AssertionError:
            return
        raise AssertionError(f"REGRESSION SELF-TEST FAILED: {desc} passed _verify_v6_tables!")

    # 1. Corrupt Qwen last-token d50-d30 CI in main.tex
    assert "[-1.58,+9.50]" in tex_txt
    _expect_fail("corrupted main.tex Qwen CI", lambda: _verify_v6_tables(paper_txt, tex_txt.replace("[-1.58,+9.50]", "[+0.78,+10.22]")))

    # 2. Corrupt Qwen last-token d50-d30 CI in paper.md
    assert "[-1.58, +9.50]" in paper_txt
    _expect_fail("corrupted paper.md Qwen CI", lambda: _verify_v6_tables(paper_txt.replace("[-1.58, +9.50]", "[+0.78, +10.22]"), tex_txt))

    # 3. Corrupt a tab:folk_transfer cell in main.tex
    assert "0.5957--0.6271" in tex_txt
    _expect_fail("corrupted tab:folk_transfer cell", lambda: _verify_v6_tables(paper_txt, tex_txt.replace("0.5957--0.6271", "0.5924--0.6254")))

    # 4-5. Corrupt Exp 251 MCQ delta col in BOTH main.tex AND paper.md
    assert "0.0 pp (n.s.)" in tex_txt and "0.00 ± 0.50 pp` (n.s.)" in paper_txt
    _expect_fail("corrupted tab:exp251 MCQ delta col", lambda: _verify_v6_tables(paper_txt, tex_txt.replace("0.0 pp (n.s.)", "99.9 pp (n.s.)")))
    _expect_fail("corrupted Table 4b MCQ delta col", lambda: _verify_v6_tables(paper_txt.replace("0.00 ± 0.50 pp` (n.s.)", "99.90 ± 0.50 pp` (n.s.)"), tex_txt))

    # 6-7. Corrupt Exp 251 canonical OOF bootstrap CI endpoint in BOTH main.tex AND paper.md
    assert "+0.03533" in tex_txt and "+0.03533" in paper_txt
    _expect_fail("corrupted main.tex Exp 251 OOF CI", lambda: _verify_v6_tables(paper_txt, tex_txt.replace("+0.03533", "+0.99000")))
    _expect_fail("corrupted paper.md Exp 251 OOF CI", lambda: _verify_v6_tables(paper_txt.replace("+0.03533", "+0.99000"), tex_txt))

    # 8. Corrupt seed-253 audio multimodal check in e251 JSON
    bad_e251 = copy.deepcopy(e251_base)
    bad_e251["lm_guardrails"]["c3_50"]["per_seed"]["253"]["multimodal"]["audio_ok"] = False
    _expect_fail("corrupted e251 seed-253 audio_ok=False", lambda: _verify_v6_tables(paper_txt, tex_txt, e251_override=bad_e251))

    # 9. Reverse Phi-4 OOF CI endpoints in main.tex (must fail order-bound check!)
    assert "[+0.03533, +0.03733]" in tex_txt
    _expect_fail("reversed main.tex Phi-4 OOF CI", lambda: _verify_v6_tables(paper_txt, tex_txt.replace("[+0.03533, +0.03733]", "[+0.03733, +0.03533]")))

    # 10. Swap Phi-4 and OLMo 3 OOF point estimates in paper.md (must fail target-bound check!)
    assert "+0.03632" in paper_txt and "+0.03163" in paper_txt
    swapped_md = paper_txt.replace("+0.03632", "__TMP__").replace("+0.03163", "+0.03632").replace("__TMP__", "+0.03163")
    _expect_fail("swapped paper.md Phi-4/OLMo 3 OOF point estimates", lambda: _verify_v6_tables(swapped_md, tex_txt))

    # 11. Inject nested quick_blind key inside e251 JSON (must fail recursive guard!)
    bad_qb = copy.deepcopy(e251_base)
    bad_qb["lm_guardrails"]["base"]["quick_blind_phi4_50"] = 0.99
    _expect_fail("injected nested quick_blind key in e251", lambda: _verify_v6_tables(paper_txt, tex_txt, e251_override=bad_qb))

    # 12. Replace canonical L12 invariance checkpoint key `base` with fictitious key (must fail exact key-set guard!)
    bad_l12 = copy.deepcopy(e251_base)
    del bad_l12["l12_invariance_max_abs_diff"]["base"]
    bad_l12["l12_invariance_max_abs_diff"]["fake_ckpt"] = 0.0
    _expect_fail("fictitious L12 checkpoint key in e251", lambda: _verify_v6_tables(paper_txt, tex_txt, e251_override=bad_l12))

    print("[PASS] Verifier corruption regression self-test passed (12/12 injected CI, order/target-swap, table, delta-col, quick_blind, L12-key, & multimodal corruptions caught).")


def run_unit_tests():
    tests = [
        ("sceptic_ctrl230", [sys.executable, str(ARCH / "aion_8b" / "sceptic_ctrl230.py"), "test"]),
        ("arch_rot229", [sys.executable, str(ARCH / "fluent" / "arch_rot229.py"), "test"]),
        ("valence_ctrl234", [sys.executable, str(ARCH / "fluent" / "valence_ctrl234.py"), "test"]),
        ("named_arch235", [sys.executable, str(ARCH / "fluent" / "named_arch235.py"), "test"]),
        ("bow_swap236", [sys.executable, str(ARCH / "fluent" / "bow_swap236.py"), "test"]),
        ("nonjung_roles237", [sys.executable, str(ARCH / "fluent" / "nonjung_roles237.py"), "test"]),
        ("bow_mirror238", [sys.executable, str(ARCH / "fluent" / "bow_mirror238.py"), "test"]),
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

    # Strictly isolate JPL_CKPT_DIR in a temp sandbox and disable .pyc writes so tests NEVER pollute the workspace!
    with tempfile.TemporaryDirectory(prefix="jpl_repro_test_") as td:
        env = dict(os.environ)
        env["JPL_CKPT_DIR"] = str(Path(td) / "ckpt")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
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
    check_arxiv_tarball()
    check_headline_numbers_and_paper_sync()
    check_verifier_corruption_regression()
    run_unit_tests()
    print("\nALL_REPRO_CHECKS_PASSED_OK")
