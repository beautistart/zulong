"""Generate PDF from the competitive analysis markdown report.

Uses fpdf2 with Microsoft YaHei font for CJK support.
"""
import re
import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

MD_PATH = r"d:\AI\project\zulong_beta4\docs\祖龙系统竞品深度分析报告.md"
PDF_PATH = r"d:\AI\project\zulong_beta4\docs\祖龙系统深度技术分析报告.pdf"

# Find a CJK-capable font on Windows
FONT_DIR = r"C:\Windows\Fonts"
FONT_REGULAR = os.path.join(FONT_DIR, "msyh.ttc")
FONT_BOLD = os.path.join(FONT_DIR, "msyhbd.ttc")

if not os.path.exists(FONT_REGULAR):
    FONT_REGULAR = os.path.join(FONT_DIR, "simsun.ttc")
    FONT_BOLD = FONT_REGULAR


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("zh", "", FONT_REGULAR)
        self.add_font("zh", "B", FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("zh", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8, "ZULONG System - Deep Technical Analysis Report",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("zh", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")

    def write_title(self, text):
        self.set_font("zh", "B", 20)
        self.set_text_color(30, 30, 30)
        self.cell(0, 14, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)

    def write_h2(self, text):
        self.ln(6)
        self.set_font("zh", "B", 14)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(60, 120, 200)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def write_h3(self, text):
        self.ln(3)
        self.set_font("zh", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def write_h4(self, text):
        self.ln(2)
        self.set_font("zh", "B", 10)
        self.set_text_color(50, 50, 80)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def write_paragraph(self, text):
        self.set_font("zh", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def write_bullet(self, text):
        self.set_font("zh", "", 9.5)
        self.set_text_color(30, 30, 30)
        bullet_w = 5
        self.cell(bullet_w, 5.5, chr(8226))
        x = self.get_x()
        w = self.w - self.r_margin - x
        self.multi_cell(w, 5.5, text)
        self.ln(0.5)

    def write_code_block(self, text):
        self.set_font("zh", "", 8.5)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(50, 50, 50)
        for line in text.split("\n"):
            self.cell(0, 5, "  " + line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(3)

    def _measure_text_width(self, text, font_style="", font_size=8.5):
        """Measure width of text string in current units."""
        self.set_font("zh", font_style, font_size)
        return self.get_string_width(text)

    def write_table(self, headers, rows):
        """Render table with auto-calculated column widths."""
        usable_w = self.w - self.l_margin - self.r_margin
        n_cols = len(headers)

        # Calculate column widths based on content
        col_widths = []
        for ci in range(n_cols):
            max_w = self._measure_text_width(headers[ci].strip(), "B", 8.5) + 4
            for row in rows:
                if ci < len(row):
                    cw = self._measure_text_width(row[ci].strip(), "", 8.5) + 4
                    max_w = max(max_w, cw)
            col_widths.append(max_w)

        # Scale widths to fit usable width
        total_w = sum(col_widths)
        if total_w > usable_w:
            scale = usable_w / total_w
            col_widths = [w * scale for w in col_widths]
        elif total_w < usable_w:
            # Distribute extra space proportionally
            extra = usable_w - total_w
            col_widths = [w + extra / n_cols for w in col_widths]

        row_h = 6.5

        # Check if table fits on current page, if not add page
        needed_h = (len(rows) + 1) * row_h + 10
        if self.get_y() + needed_h > self.h - self.b_margin:
            self.add_page()

        # Header row
        self.set_font("zh", "B", 8.5)
        self.set_fill_color(50, 100, 180)
        self.set_text_color(255, 255, 255)
        for ci, h in enumerate(headers):
            self.cell(col_widths[ci], row_h, h.strip(), border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("zh", "", 8.5)
        fill_flag = False
        for row in rows:
            if fill_flag:
                self.set_fill_color(238, 243, 255)
            else:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 30, 30)

            for ci in range(n_cols):
                cell_text = row[ci].strip() if ci < len(row) else ""
                w = col_widths[ci]
                # Truncate only if really necessary
                while self.get_string_width(cell_text) > w - 2 and len(cell_text) > 3:
                    cell_text = cell_text[:-2] + ".."
                self.cell(w, row_h, cell_text, border=1, fill=True, align="C")
            self.ln()
            fill_flag = not fill_flag

        self.ln(3)


def parse_table(lines, start_idx):
    """Parse markdown table starting at start_idx."""
    headers = [c.strip() for c in lines[start_idx].split("|")[1:-1]]
    data_start = start_idx + 2
    rows = []
    idx = data_start
    while idx < len(lines) and "|" in lines[idx] and lines[idx].strip().startswith("|"):
        cells = [c.strip() for c in lines[idx].split("|")[1:-1]]
        cells = [re.sub(r'\*\*(.*?)\*\*', r'\1', c) for c in cells]
        rows.append(cells)
        idx += 1
    return headers, rows, idx


def generate_pdf():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    pdf = PDFReport()
    pdf.add_page()

    in_code_block = False
    code_buffer = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code block toggle
        if line.strip().startswith("```"):
            if in_code_block:
                pdf.write_code_block("\n".join(code_buffer))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            i += 1
            continue

        # Title (# )
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            pdf.write_title(title)
            i += 1
            continue

        # H2 (## )
        if stripped.startswith("## "):
            text = stripped[3:].strip()
            pdf.write_h2(text)
            i += 1
            continue

        # H3 (### )
        if stripped.startswith("### ") and not stripped.startswith("#### "):
            text = stripped[4:].strip()
            pdf.write_h3(text)
            i += 1
            continue

        # H4 (#### )
        if stripped.startswith("#### "):
            text = stripped[5:].strip()
            pdf.write_h4(text)
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            text = stripped.lstrip("> ").strip()
            pdf.set_font("zh", "", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, text)
            pdf.ln(1)
            i += 1
            continue

        # Table
        if "|" in stripped and stripped.startswith("|") and i + 1 < len(lines) and "---" in lines[i + 1]:
            headers, rows, end_idx = parse_table(lines, i)
            if headers and rows:
                pdf.write_table(headers, rows)
            i = end_idx
            continue

        # Bullet list
        if stripped.startswith("- ") or stripped.startswith("* "):
            text = stripped[2:].strip()
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            pdf.write_bullet(text)
            i += 1
            continue

        # Numbered list
        if re.match(r'^\d+\.', stripped):
            text = re.sub(r'^\d+\.\s*', '', stripped)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            pdf.write_bullet(text)
            i += 1
            continue

        # Normal paragraph - clean markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
        pdf.write_paragraph(text)
        i += 1

    pdf.output(PDF_PATH)
    print(f"PDF saved to: {PDF_PATH}")
    print(f"File size: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    generate_pdf()
