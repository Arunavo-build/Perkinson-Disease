import os
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ARTIFACT_MD = r"C:\Users\Arunavo\.gemini\antigravity\brain\95ef30d1-213b-49db-a37e-a40ddf14f571\Q1_Journal_Manuscript.md"
WORKSPACE_DIR = r"d:\Perkinson disease"
WORKSPACE_MD = os.path.join(WORKSPACE_DIR, "Q1_Journal_Manuscript.md")
WORKSPACE_DOCX = os.path.join(WORKSPACE_DIR, "Q1_Journal_Manuscript.docx")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)


def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_formatted_runs(p, text, base_size=10.5, base_italic=False):
    # Clean LaTeX inline math markers for Word readability
    text = text.replace(r"\mathcal{D}_{\text{TR}}", "D_TR").replace(r"\mathcal{D}_{\text{BN}}", "D_BN").replace(r"\mathcal{D}_{\text{OX}}", "D_OX")
    text = text.replace(r"\rightarrow", "→").replace(r"\leftrightarrow", "↔").replace(r"\pm", "±")
    text = text.replace(r"\le", "≤").replace(r"\ge", "≥").replace(r"\ne", "≠").replace(r"\approx", "≈")
    text = text.replace(r"\alpha", "α").replace(r"\tau", "τ").replace(r"\eta", "η").replace(r"\Delta", "Δ")
    text = text.replace(r"\sigma", "σ").replace(r"\mu", "μ").replace(r"\lambda", "λ").replace(r"\gamma", "γ")
    text = text.replace(r"\phi", "φ").replace(r"\Phi", "Φ").replace(r"\chi^2", "χ²").replace(r"\times", "×")
    text = re.sub(r"\\mathbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = text.replace("$", "")

    # Split by bold (**...**) and code (`...`)
    tokens = re.split(r"(\*\*.*?\*\*|`.*?`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            run = p.add_run(tok[2:-2])
            run.bold = True
            run.italic = base_italic
            run.font.size = Pt(base_size)
            run.font.name = "Times New Roman"
        elif tok.startswith("`") and tok.endswith("`"):
            run = p.add_run(tok[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(base_size - 1)
            run.font.color.rgb = RGBColor(20, 50, 110)
        else:
            run = p.add_run(tok)
            run.italic = base_italic
            run.font.size = Pt(base_size)
            run.font.name = "Times New Roman"


def build_files():
    with open(ARTIFACT_MD, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 1. Save Workspace Markdown with relative figure paths
    ws_md_content = md_content.replace(
        "/C:/Users/Arunavo/.gemini/antigravity/brain/95ef30d1-213b-49db-a37e-a40ddf14f571/",
        "figures/"
    )
    with open(WORKSPACE_MD, "w", encoding="utf-8") as f:
        f.write(ws_md_content)
    print(f"Saved workspace Markdown to: {WORKSPACE_MD}")

    # 2. Build Word Document (.docx)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)

    lines = ws_md_content.splitlines()
    i = 0
    in_code_block = False
    code_lines = []
    in_carousel = False

    while i < len(lines):
        line = lines[i].rstrip()

        # Handle carousel markers
        if line.startswith("````carousel"):
            in_carousel = True
            i += 1
            continue
        if in_carousel and line.startswith("````"):
            in_carousel = False
            i += 1
            continue
        if in_carousel and line.strip() == "<!-- slide -->":
            i += 1
            continue

        # Handle code blocks (Algorithms)
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                in_code_block = False
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.left_indent = Inches(0.2)
                run = p.add_run("\n".join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(15, 35, 65)
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Skip horizontal rules or empty lines
        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        # Handle Title (# ...)
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(line[2:].strip())
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(16)
            run.font.color.rgb = RGBColor(10, 35, 80)
            i += 1
            continue

        # Handle Section Headings (## ...)
        if line.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(line[3:].strip())
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(15, 45, 95)
            i += 1
            continue

        # Handle Subsections (### ...)
        if line.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(line[4:].strip())
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(11.5)
            run.font.color.rgb = RGBColor(25, 60, 115)
            i += 1
            continue

        # Handle Sub-subsections (#### ...)
        if line.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(line[5:].strip())
            run.bold = True
            run.italic = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)
            i += 1
            continue

        # Handle Blockquotes / Highlights (> ...)
        if line.startswith(">"):
            clean_q = line.lstrip("> ").strip()
            if clean_q.startswith("[!"):
                i += 1
                continue
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_after = Pt(3)
            add_formatted_runs(p, clean_q, base_size=10)
            i += 1
            continue

        # Handle Display Equations ($$ ... $$)
        if line.startswith("$$") and line.endswith("$$"):
            eq_txt = line[2:-2].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            add_formatted_runs(p, eq_txt, base_size=10, base_italic=True)
            i += 1
            continue

        # Handle Images (![caption](path))
        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if img_match:
            alt_text, rel_path = img_match.groups()
            abs_img = os.path.join(WORKSPACE_DIR, rel_path.replace("/", os.sep))
            if os.path.exists(abs_img):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                run = p.add_run()
                run.add_picture(abs_img, width=Inches(6.2))
            i += 1
            continue

        # Handle Markdown Tables (| ... |)
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                t_line = lines[i].strip()
                # Skip separator row (| :--- | :---: |)
                if not re.match(r"^\|[\s:\-\|]+\|$", t_line):
                    cells = [c.strip() for c in t_line.strip("|").split("|")]
                    table_lines.append(cells)
                i += 1

            if table_lines:
                num_rows = len(table_lines)
                num_cols = max(len(r) for r in table_lines)
                tbl = doc.add_table(rows=num_rows, cols=num_cols)
                tbl.style = 'Table Grid'
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

                for r_idx, row_data in enumerate(table_lines):
                    row_cells = tbl.rows[r_idx].cells
                    for c_idx in range(min(len(row_data), num_cols)):
                        cell = row_cells[c_idx]
                        set_cell_margins(cell, top=45, bottom=45, left=65, right=65)
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_before = Pt(1)
                        p.paragraph_format.space_after = Pt(1)
                        if r_idx == 0:
                            set_cell_background(cell, "1B365D")
                            add_formatted_runs(p, row_data[c_idx], base_size=8.5)
                            for run in p.runs:
                                run.bold = True
                                run.font.color.rgb = RGBColor(255, 255, 255)
                        else:
                            if r_idx % 2 == 1:
                                set_cell_background(cell, "F4F7FA")
                            add_formatted_runs(p, row_data[c_idx], base_size=8.5)
                doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # Handle Bullet Points (- ... or 1. ...)
        if line.lstrip().startswith("- "):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(3)
            add_formatted_runs(p, line.lstrip()[2:].strip(), base_size=10.5)
            i += 1
            continue

        num_match = re.match(r"^(\d+\.)\s+(.*)", line.lstrip())
        if num_match:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(3)
            r_num = p.add_run(num_match.group(1) + " ")
            r_num.bold = True
            r_num.font.name = "Times New Roman"
            r_num.font.size = Pt(10.5)
            add_formatted_runs(p, num_match.group(2), base_size=10.5)
            i += 1
            continue

        # Handle Figure Captions (*Figure X: ...*)
        if line.startswith("*Figure ") and line.endswith("*"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            add_formatted_runs(p, line.strip("*"), base_size=9.5, base_italic=True)
            i += 1
            continue

        # Regular Paragraph
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6)
        add_formatted_runs(p, line, base_size=10.5)
        i += 1

    doc.save(WORKSPACE_DOCX)
    print(f"Saved publication-ready Word manuscript to: {WORKSPACE_DOCX}")


if __name__ == "__main__":
    build_files()
