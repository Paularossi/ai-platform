"""
core/pdf_export.py

Renders the Agent 0 final policy brief to a PDF. Handles the small subset of
markdown the brief actually uses (##/### headings, paragraphs, - / * / numbered
list items, **bold**, *italic*) - this is not a general markdown parser.
"""

from __future__ import annotations

import io
import re

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate


def _inline_markdown_to_html(text: str) -> str:
    """Convert **bold** / *italic* to the inline tags reportlab's Paragraph understands."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def brief_to_pdf_bytes(brief_text: str, topic: str) -> bytes:
    """Render the final policy brief text to a PDF and return its bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BriefTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=17, spaceAfter=6,
    )
    topic_style = ParagraphStyle(
        "BriefTopic", parent=styles["Normal"], fontName="Helvetica-Oblique",
        fontSize=11, textColor="#444444", spaceAfter=20,
    )
    h2_style = ParagraphStyle(
        "BriefH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, spaceBefore=14, spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        "BriefH3", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=11.5, spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BriefBody", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=8,
    )
    bullet_style = ParagraphStyle("BriefBullet", parent=body_style, spaceAfter=4)

    story = [
        Paragraph("Policy Brief", title_style),
        Paragraph(_inline_markdown_to_html(topic), topic_style),
    ]

    paragraph_buffer: list[str] = []
    list_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            story.append(Paragraph(_inline_markdown_to_html(" ".join(paragraph_buffer)), body_style))
            paragraph_buffer.clear()

    def flush_list() -> None:
        if list_buffer:
            story.append(ListFlowable(
                [ListItem(Paragraph(_inline_markdown_to_html(item), bullet_style)) for item in list_buffer],
                bulletType="bullet", leftIndent=18,
            ))
            list_buffer.clear()

    for raw_line in brief_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue
        if line.startswith("### "):
            flush_paragraph(); flush_list()
            story.append(Paragraph(_inline_markdown_to_html(line[4:]), h3_style))
        elif line.startswith("## "):
            flush_paragraph(); flush_list()
            story.append(Paragraph(_inline_markdown_to_html(line[3:]), h2_style))
        elif line.startswith("# "):
            flush_paragraph(); flush_list()
            story.append(Paragraph(_inline_markdown_to_html(line[2:]), h2_style))
        elif re.match(r"^[-*]\s+", line):
            flush_paragraph()
            list_buffer.append(re.sub(r"^[-*]\s+", "", line))
        elif re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            list_buffer.append(re.sub(r"^\d+\.\s+", "", line))
        else:
            flush_list()
            paragraph_buffer.append(line)

    flush_paragraph()
    flush_list()

    doc.build(story)
    return buf.getvalue()
