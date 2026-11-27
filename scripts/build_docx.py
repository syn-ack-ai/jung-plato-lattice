"""Generate a clean, dual-audience Microsoft Word (.docx) review manuscript from paper/paper.md.

Designed for both clinical/academic psychologists and ML/AI engineers reviewing with Word
Track Changes and Comments:
  - Clean Georgia/Calibri typography and native Word Heading 1-4 styles (Navigation Pane ready)
  - Color-coded shaded callout boxes for "For ML/AI Engineers", "For Psychologists", Reader Guides,
    and historical quotations
  - Styled Word tables with repeating navy headers, subtle row shading, and cell padding
  - High-resolution embedded PNG figures (fig1..fig9) scaled cleanly to 6.4 inches wide with captions
  - Full Unicode rendering for inline and display LaTeX math (zero raw TeX leaks in Word)
  - Running header and dynamic page numbers in footer
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "paper" / "paper.md"
DEFAULT_DOCX = ROOT / "paper" / "The_Jung_Plato_Lattice_v6.docx"

NAVY = RGBColor(0x1B, 0x36, 0x5D)
SLATE = RGBColor(0x33, 0x4E, 0x68)
DARK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x52, 0x60, 0x6D)
TEAL = RGBColor(0x0E, 0x62, 0x51)


def _replace_frac(s: str) -> str:
    """Replace LaTeX \\frac{num}{den} (including nested braces) with (num) / (den)."""
    while "\\frac{" in s:
        idx = s.index("\\frac{")
        pos = idx + len("\\frac")
        groups = []
        ok = True
        for _ in range(2):
            if pos >= len(s) or s[pos] != "{":
                ok = False
                break
            depth = 0
            start = pos + 1
            end = None
            for k in range(pos, len(s)):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
            if end is None:
                ok = False
                break
            groups.append(s[start:end])
            pos = end + 1
        if not ok:
            break
        s = s[:idx] + f"({groups[0]}) / ({groups[1]})" + s[pos:]
    return s


def _convert_math_expr(inner: str) -> str:
    """Convert a LaTeX math body into crisp, readable Unicode text for Word."""
    s = inner.strip()
    subs = [
        (r"\mathbb{E}", "E"),
        (r"\hat{z}", "ẑ"),
        (r"\hat{Z}", "Ẑ"),
        (r"\tilde{Z}", "Z̃"),
        (r"R^2_{\text{frob}}", "R²_frob"),
        (r"R^2_{\text{opt}}", "R²_opt"),
        (r"R^2_{\mathrm{frob}}", "R²_frob"),
        (r"R^2_{\mathrm{opt}}", "R²_opt"),
        (r"R^\top", "Rᵀ"),
        (r"V^\top", "Vᵀ"),
        (r"^\top", "ᵀ"),
        (r"v_{\text{dir}}", "v_dir"),
        (r"v_{\mathrm{dir}}", "v_dir"),
        (r"n_{\text{tr}}", "n_tr"),
        (r"n_{\text{te}}", "n_te"),
        (r"p_{\text{holm}}", "p_holm"),
        (r"\Delta\text{NLL}", "ΔNLL"),
        (r"\Delta", "Δ"),
        (r"\Sigma", "Σ"),
        (r"\sum", "Σ"),
        (r"\lambda", "λ"),
        (r"\rho", "ρ"),
        (r"\mu", "μ"),
        (r"\sigma", "σ"),
        (r"\alpha", "α"),
        (r"\pm", "±"),
        (r"\times", "×"),
        (r"\approx", "≈"),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\to", "→"),
        (r"\perp", "⊥"),
        (r"\cos", "cos"),
        (r"\sim", "~"),
        (r"\cdot", "·"),
        (r"\qquad", "   "),
        (r"\quad", "  "),
        (r"\|", "‖"),
        (r"\_", "_"),
        (r"\%", "%"),
        (r"\sqrt{N}", "√N"),
        (r"\sqrt{R²_opt}", "√R²_opt"),
        (r"\sqrt", "√"),
    ]
    for a, b in subs:
        s = s.replace(a, b)
    s = _replace_frac(s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", s)
    s = s.replace("10^{-9}", "10⁻⁹").replace("10^{-7}", "10⁻⁷").replace("10^{-11}", "10⁻¹¹")
    s = s.replace("^2", "²")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tex_math_to_unicode(s: str) -> str:
    """Convert both display $$...$$ and inline $...$ LaTeX expressions to clean Unicode."""
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: _convert_math_expr(m.group(1)), s)
    s = re.sub(r"\$([^$]+)\$", lambda m: _convert_math_expr(m.group(1)), s)
    return s


def _add_formatted_runs(
    p,
    text: str,
    base_size: float = 10.5,
    base_color: RGBColor = DARK,
    bold_default: bool = False,
    italic_default: bool = False,
    font_name: str = "Georgia",
):
    """Recursively parse `code`, ***bold+italic***, **bold**, *italic*, and [label](url) cleanly into Word runs."""
    text = _tex_math_to_unicode(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)

    def _emit_plain(chunk: str, bold: bool, italic: bool):
        if not chunk:
            return
        r = p.add_run(chunk)
        r.font.name = font_name
        r.font.size = Pt(base_size)
        r.font.color.rgb = base_color
        if bold:
            r.bold = True
        if italic:
            r.italic = True

    def _emit_code(chunk: str, bold: bool, italic: bool):
        if not chunk:
            return
        rc = p.add_run(chunk)
        rc.font.name = "Menlo"
        rc.font.size = Pt(max(8.0, base_size - 1.2))
        rc.font.color.rgb = NAVY if base_color != RGBColor(0xFF, 0xFF, 0xFF) else base_color
        if bold:
            rc.bold = True
        if italic:
            rc.italic = True

    # Priority order: backtick code first (never split math/code), then ***, then **, then safe *italic* (skip math 0.5*(...))
    token_re = re.compile(
        r"(`[^`\n]+`|\*\*\*(?!\s)(.+?)(?<!\s)\*\*\*|\*\*(?!\s)(.+?)(?<!\s)\*\*|(?<![0-9A-Za-z_])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![0-9A-Za-z_]))"
    )

    def _walk(segment: str, bold: bool, italic: bool):
        pos = 0
        for m in token_re.finditer(segment):
            if m.start() > pos:
                _emit_plain(segment[pos : m.start()], bold, italic)
            tok = m.group(0)
            if tok.startswith("`") and tok.endswith("`"):
                _emit_code(tok[1:-1], bold, italic)
            elif tok.startswith("***") and tok.endswith("***"):
                _walk(m.group(2), True, True)
            elif tok.startswith("**") and tok.endswith("**"):
                _walk(m.group(3), True, italic)
            elif tok.startswith("*") and tok.endswith("*"):
                _walk(m.group(4), bold, True)
            pos = m.end()
        if pos < len(segment):
            _emit_plain(segment[pos:], bold, italic)

    _walk(text, bold_default, italic_default)


def _set_cell_shading(cell, hex_fill: str):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_fill}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def _set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="1B365D"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9E2EC"/>'
        '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)


def _add_callout(doc: Document, lines: list[str], fill="F0F4F8", border_col="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    cell = tbl.cell(0, 0)
    _set_cell_shading(cell, fill)
    _set_cell_margins(cell, top=110, bottom=110, left=160, right=140)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="20" w:space="0" w:color="{border_col}"/>'
        '<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        "</w:tcBorders>"
    )
    tcPr.append(tcBorders)
    for idx, ln in enumerate(lines):
        if not ln.strip():
            continue
        p = cell.paragraphs[0] if idx == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        _add_formatted_runs(p, ln, base_size=9.8, base_color=DARK)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def _add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run("The Jung–Plato Lattice (v6 Dual-Audience Review Copy)   |   Page ")
    r1.font.name = "Calibri"
    r1.font.size = Pt(8.5)
    r1.font.color.rgb = MUTED
    fld = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"/>')
    p._p.append(fld)


def build_docx(md_path: Path = DEFAULT_MD, out_path: Path = DEFAULT_DOCX):
    doc = Document()

    for sec in doc.sections:
        sec.top_margin = Inches(0.85)
        sec.bottom_margin = Inches(0.85)
        sec.left_margin = Inches(0.85)
        sec.right_margin = Inches(0.85)
        sec.different_first_page_header_footer = True
        hdr = sec.header.paragraphs[0]
        hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hr = hdr.add_run("The Jung–Plato Lattice | Preprint v6 (Dual-Audience Review Copy)")
        hr.font.name = "Calibri"
        hr.font.size = Pt(8.5)
        hr.font.color.rgb = MUTED
        _add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Georgia"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = DARK
    normal.paragraph_format.line_spacing = 1.18
    normal.paragraph_format.space_after = Pt(6)

    for lvl, sz, col in [(1, 18, NAVY), (2, 13.5, NAVY), (3, 11.5, SLATE), (4, 10.5, TEAL)]:
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = "Calibri"
        st.font.bold = True
        st.font.size = Pt(sz)
        st.font.color.rgb = col
        st.paragraph_format.space_before = Pt(12 if lvl <= 2 else 8)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    in_refs = False
    first_title_done = False

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if s == "---" or not s:
            i += 1
            continue

        # Standalone display math line $$...$$
        if s.startswith("$$") and s.endswith("$$"):
            eq = _convert_math_expr(s[2:-2])
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(eq)
            r.font.name = "Cambria Math"
            r.font.size = Pt(10.0)
            r.font.color.rgb = NAVY
            i += 1
            continue

        # Headings
        if s.startswith("# "):
            title = s[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(title)
            r.font.name = "Calibri"
            r.font.bold = True
            r.font.size = Pt(19)
            r.font.color.rgb = NAVY
            pa = doc.add_paragraph()
            pa.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pa.paragraph_format.space_after = Pt(4)
            ra = pa.add_run("Benjamin Babeshkin  |  Independent Research (syn-ack-ai)  |  DOI: 10.5281/zenodo.24572076")
            ra.font.name = "Calibri"
            ra.font.size = Pt(10)
            ra.font.color.rgb = SLATE
            first_title_done = True
            i += 1
            continue

        if s.startswith("## "):
            h_txt = s[3:].strip()
            if h_txt.lower() == "references":
                in_refs = True
            hp = doc.add_heading("", level=2)
            _add_formatted_runs(hp, h_txt, base_size=13.5, base_color=NAVY, bold_default=True, font_name="Calibri")
            i += 1
            continue

        if s.startswith("### "):
            hp = doc.add_heading("", level=3)
            _add_formatted_runs(hp, s[4:].strip(), base_size=11.5, base_color=SLATE, bold_default=True, font_name="Calibri")
            i += 1
            continue

        if s.startswith("#### "):
            hp = doc.add_heading("", level=4)
            _add_formatted_runs(hp, s[5:].strip(), base_size=10.5, base_color=TEAL, bold_default=True, font_name="Calibri")
            i += 1
            continue

        # Blockquote / dual-audience callout block
        if s.startswith(">"):
            q_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            joined = " ".join(q_lines)
            if "For Psychologists" in joined or "Analytical Psychology" in joined:
                _add_callout(doc, q_lines, fill="EBF7F4", border_col="0E6251")
            elif "For ML" in joined or "AI Engineers" in joined:
                _add_callout(doc, q_lines, fill="ECF2FA", border_col="1B365D")
            elif "Reader Guide" in joined or "Plain-English" in joined:
                _add_callout(doc, q_lines, fill="FDF8EC", border_col="B7791F")
            else:
                _add_callout(doc, q_lines, fill="F4F6F8", border_col="486581")
            continue

        # Image line ![caption](figures/...)
        im = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if im:
            _, rel_path = im.group(1), im.group(2)
            img_path = (md_path.parent / rel_path).resolve()
            if img_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(6)
                p_img.paragraph_format.space_after = Pt(2)
                p_img.paragraph_format.keep_with_next = True
                p_img.add_run().add_picture(str(img_path), width=Inches(6.4))
            i += 1
            if i < len(lines) and lines[i].strip().startswith("*Figure"):
                cap = lines[i].strip().strip("*")
                pc = doc.add_paragraph()
                pc.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pc.paragraph_format.space_before = Pt(1)
                pc.paragraph_format.space_after = Pt(8)
                pc.paragraph_format.line_spacing = 1.1
                _add_formatted_runs(pc, cap, base_size=8.8, base_color=MUTED, italic_default=True)
                i += 1
            continue

        # Markdown table
        if s.startswith("|") and s.endswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            parsed = []
            for tl in tbl_lines:
                if re.match(r"^\|[\s:\-|]+\|$", tl):
                    continue
                parsed.append([c.strip() for c in tl.strip("|").split("|")])
            if parsed:
                n_cols = len(parsed[0])
                tbl = doc.add_table(rows=len(parsed), cols=n_cols)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                tbl.autofit = True
                _set_table_borders(tbl)
                f_sz = 7.2 if n_cols >= 8 else (7.8 if n_cols >= 6 else 8.4)
                for r_idx, row_vals in enumerate(parsed):
                    trPr = tbl.rows[r_idx]._tr.get_or_add_trPr()
                    if r_idx == 0:
                        trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
                    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
                    for c_idx in range(n_cols):
                        val = row_vals[c_idx] if c_idx < len(row_vals) else ""
                        cell = tbl.cell(r_idx, c_idx)
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        _set_cell_margins(cell, top=42, bottom=42, left=55, right=55)
                        if r_idx == 0:
                            _set_cell_shading(cell, "1B365D")
                        elif r_idx % 2 == 0:
                            _set_cell_shading(cell, "F7F9FC")
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_before = Pt(0)
                        p.paragraph_format.space_after = Pt(0)
                        p.paragraph_format.line_spacing = 1.05
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                        if r_idx == 0:
                            clean_hdr = _tex_math_to_unicode(val.replace("**", "").replace("`", ""))
                            run = p.add_run(clean_hdr)
                            run.font.name = "Calibri"
                            run.font.bold = True
                            run.font.size = Pt(f_sz)
                            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        else:
                            _add_formatted_runs(p, val, base_size=f_sz, base_color=DARK)
                sp = doc.add_paragraph()
                sp.paragraph_format.space_before = Pt(0)
                sp.paragraph_format.space_after = Pt(3)
            if i < len(lines) and lines[i].strip() == "":
                if i + 1 < len(lines) and (lines[i + 1].strip().startswith("*Table") or lines[i + 1].strip().startswith("*Conceptual")):
                    i += 1
            if i < len(lines) and (lines[i].strip().startswith("*Table") or lines[i].strip().startswith("*Conceptual")):
                cap = lines[i].strip().strip("*")
                pc = doc.add_paragraph()
                pc.paragraph_format.space_before = Pt(1)
                pc.paragraph_format.space_after = Pt(8)
                pc.paragraph_format.line_spacing = 1.1
                _add_formatted_runs(pc, cap, base_size=8.6, base_color=MUTED, italic_default=True)
                i += 1
            continue

        # Bullet / numbered list items
        if re.match(r"^(\s*[-*]|\s*\d+\.)\s+", raw):
            indent_lvl = len(raw) - len(raw.lstrip(" "))
            content = re.sub(r"^(\s*[-*]|\s*\d+\.)\s+", "", raw).strip()
            if in_refs:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.35)
                p.paragraph_format.first_line_indent = Inches(-0.35)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.1
                _add_formatted_runs(p, content, base_size=8.8, base_color=DARK)
            else:
                p = doc.add_paragraph(style="List Bullet" if raw.lstrip().startswith(("-", "*")) else "List Number")
                if indent_lvl >= 2:
                    p.paragraph_format.left_indent = Inches(0.55)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(3.5)
                p.paragraph_format.line_spacing = 1.15
                _add_formatted_runs(p, content, base_size=10.0, base_color=DARK)
            i += 1
            continue

        # Subtitle line right after title
        if first_title_done and s.startswith("*") and s.endswith("*") and i <= 5:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            _add_formatted_runs(p, s[1:-1], base_size=9.0, base_color=MUTED, italic_default=True)
            i += 1
            continue

        # Standard paragraph
        para_lines = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt == "---"
                or nxt.startswith(("#", ">", "|", "![", "- ", "* ", "$$"))
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.18
        _add_formatted_runs(p, " ".join(para_lines), base_size=10.5, base_color=DARK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))

    # Validate zero raw TeX math leaks and zero stray Markdown tokens in output docx
    check_doc = Document(str(out_path))
    all_txt = "\n".join(
        [p.text for p in check_doc.paragraphs]
        + [c.text for t in check_doc.tables for r in t.rows for c in r.cells]
    )
    leaks = re.findall(r"(\$[^$\n]{1,40}\$|\\frac|\\mathbb|\\hat\{|\\tilde\{|\\text\{|`[^`\n]+`|\*\*[^*\n]+\*\*)", all_txt)
    assert not leaks, f"Raw LaTeX/Markdown leaks found in {out_path}: {leaks[:10]}"
    print(
        f"SAVED Word manuscript: {out_path} ({out_path.stat().st_size:,} bytes; "
        f"paragraphs={len(check_doc.paragraphs)}, tables={len(check_doc.tables)}, shapes={len(check_doc.inline_shapes)}, tex_leaks=0)"
    )


def main():
    ap = argparse.ArgumentParser(description="Build Microsoft Word (.docx) review copy from paper/paper.md.")
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--out", type=Path, default=DEFAULT_DOCX)
    args = ap.parse_args()
    build_docx(args.md, args.out)


if __name__ == "__main__":
    main()
