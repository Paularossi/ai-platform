"""
core/pdf_export.py

Renders the Agent 0 final policy brief to a PDF. Handles the small subset of
markdown the brief actually uses (##/### headings, paragraphs, - / * / numbered
list items, **bold**, *italic*) - this is not a general markdown parser.

No fixed-language label or page title is baked in here: the brief's own
leading heading (in whatever language Agent 0 wrote it) becomes the title,
falling back to the human-authored topic text if the brief has none.

Font: prefers a bundled/system Unicode TrueType font over ReportLab's base14
Helvetica so scripts beyond Latin-1 (Cyrillic, Greek, Vietnamese, etc.) render
instead of dropping to boxes/blanks. This still doesn't give full "any
language" support - ReportLab's Paragraph flowable has no bidi reordering or
complex text shaping, so right-to-left scripts (Arabic, Hebrew) and CJK/Indic
scripts won't lay out correctly even with a font that has the glyphs. Getting
those right would need a different rendering engine, not just a font swap.
"""

from __future__ import annotations

import io
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate

# Matches a markdown thematic break on its own line: ---, ***, ___ (3+ chars,
# optionally spaced, e.g. "- - -"). Rendered as an actual rule rather than literal dashes.
_THEMATIC_BREAK_RE = re.compile(r"^(?:-\s*){3,}$|^(?:\*\s*){3,}$|^(?:_\s*){3,}$")

# Setext heading underlines: a line of only === or only --- directly under a
# single line of text makes that text a heading (CommonMark). Models emit this
# often; without it the underline shows up as a stray rule (or literal "===")
# under an unstyled paragraph line.
_SETEXT_EQ_RE = re.compile(r"^={3,}$")
_SETEXT_DASH_RE = re.compile(r"^-{3,}$")

# ATX heading, tolerantly: 1-6 hashes, the space after them optional (models
# frequently emit "####Heading"), optional closing hashes ("## Title ##").
_ATX_HEADING_RE = re.compile(r"^(#{1,6})(?!#)\s*(.+?)\s*#*\s*$")

# (regular, bold, italic, bold-italic) paths for the first broad-coverage
# Unicode font found on the host. Checked in order; Segoe UI covers Windows
# dev/deploy targets, DejaVu Sans covers the Debian/Ubuntu-based Linux images
# most cloud deployments (e.g. Streamlit Community Cloud) use.
_UNICODE_FONT_CANDIDATES: list[tuple[str, str, str, str]] = [
    (
        r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\segoeuii.ttf", r"C:\Windows\Fonts\segoeuiz.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    ),
]

_FONT_NAMES: dict[str, str] | None = None  # cached after first resolution


def _resolve_font_names() -> dict[str, str]:
    """
    Register the first Unicode font found on the host (once per process) and
    return its {"regular", "bold", "italic", "bold_italic"} names. Falls back
    to base14 Helvetica (Latin-1 only) if none of the candidates exist -
    export still works everywhere, just with narrower script coverage on
    hosts without one of these fonts installed.
    """
    global _FONT_NAMES
    if _FONT_NAMES is not None:
        return _FONT_NAMES

    for regular, bold, italic, bold_italic in _UNICODE_FONT_CANDIDATES:
        if all(os.path.isfile(p) for p in (regular, bold, italic, bold_italic)):
            pdfmetrics.registerFont(TTFont("UnicodeSans", regular))
            pdfmetrics.registerFont(TTFont("UnicodeSans-Bold", bold))
            pdfmetrics.registerFont(TTFont("UnicodeSans-Italic", italic))
            pdfmetrics.registerFont(TTFont("UnicodeSans-BoldItalic", bold_italic))
            pdfmetrics.registerFontFamily(
                "UnicodeSans", normal="UnicodeSans", bold="UnicodeSans-Bold",
                italic="UnicodeSans-Italic", boldItalic="UnicodeSans-BoldItalic",
            )
            _FONT_NAMES = {
                "regular": "UnicodeSans", "bold": "UnicodeSans-Bold",
                "italic": "UnicodeSans-Italic", "bold_italic": "UnicodeSans-BoldItalic",
            }
            return _FONT_NAMES

    _FONT_NAMES = {
        "regular": "Helvetica", "bold": "Helvetica-Bold",
        "italic": "Helvetica-Oblique", "bold_italic": "Helvetica-BoldOblique",
    }
    return _FONT_NAMES


def _inline_markdown_to_html(text: str) -> str:
    """
    Convert **bold** / *italic* to the inline tags reportlab's Paragraph
    understands. XML-escapes the text first: Paragraph() parses its input as
    markup, so an unescaped '<' in the brief (e.g. "x<y") is a hard
    ValueError that kills the whole export, and '&'/'>' can misrender.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _split_title(brief_text: str, topic: str) -> tuple[str, str, str]:
    """
    Returns (title, subtitle, remaining_body). Uses the brief's own leading
    '#'/'##'/'###' heading as the title when present - in whatever language
    or script Agent 0 wrote it in - with the topic shown as a subtitle below
    it. Falls back to the topic itself as the title (with no subtitle) when
    the brief has no leading heading, so no fixed-language label is ever
    hardcoded into the document.
    """
    lines = brief_text.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines):
        first = lines[idx].strip()
        match = _ATX_HEADING_RE.match(first)
        if match and len(match.group(1)) <= 3 and match.group(2):
            return match.group(2), topic, "\n".join(lines[idx + 1:])
        # Setext title: first line of text underlined with === or ---
        if idx + 1 < len(lines):
            underline = lines[idx + 1].strip()
            if first and not first.startswith("#") and (
                _SETEXT_EQ_RE.match(underline) or _SETEXT_DASH_RE.match(underline)
            ):
                return first, topic, "\n".join(lines[idx + 2:])
    return topic, "", brief_text


def brief_to_pdf_bytes(brief_text: str, topic: str) -> bytes:
    """Render the final policy brief text to a PDF and return its bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
    )
    font = _resolve_font_names()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BriefTitle", parent=styles["Title"], fontName=font["bold"],
        fontSize=17, spaceAfter=6,
    )
    topic_style = ParagraphStyle(
        "BriefTopic", parent=styles["Normal"], fontName=font["italic"],
        fontSize=11, textColor="#444444", spaceAfter=20,
    )
    h2_style = ParagraphStyle(
        "BriefH2", parent=styles["Heading2"], fontName=font["bold"],
        fontSize=13, spaceBefore=14, spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        "BriefH3", parent=styles["Heading3"], fontName=font["bold"],
        fontSize=11.5, spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BriefBody", parent=styles["Normal"], fontName=font["regular"],
        fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=8,
    )
    bullet_style = ParagraphStyle("BriefBullet", parent=body_style, spaceAfter=4)

    title, subtitle, body_text = _split_title(brief_text, topic)
    story = [Paragraph(_inline_markdown_to_html(title), title_style)]
    if subtitle:
        story.append(Paragraph(_inline_markdown_to_html(subtitle), topic_style))

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

    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue
        is_setext_underline = bool(_SETEXT_EQ_RE.match(line) or _SETEXT_DASH_RE.match(line))
        if is_setext_underline and len(paragraph_buffer) == 1:
            # Setext heading: the single buffered text line above this
            # underline is the heading, not a paragraph followed by a rule.
            heading_text = paragraph_buffer.pop()
            style = h2_style if _SETEXT_EQ_RE.match(line) else h3_style
            story.append(Paragraph(_inline_markdown_to_html(heading_text), style))
        elif _SETEXT_EQ_RE.match(line):
            # A stray === with nothing (or a multi-line paragraph) above it:
            # render a rule rather than leaking literal equals signs into text.
            flush_paragraph(); flush_list()
            story.append(HRFlowable(
                width="100%", thickness=0.6, color=colors.HexColor("#cccccc"),
                spaceBefore=10, spaceAfter=10,
            ))
        elif _THEMATIC_BREAK_RE.match(line):
            flush_paragraph(); flush_list()
            story.append(HRFlowable(
                width="100%", thickness=0.6, color=colors.HexColor("#cccccc"),
                spaceBefore=10, spaceAfter=10,
            ))
        elif (heading_match := _ATX_HEADING_RE.match(line)) and heading_match.group(2):
            # Any heading depth (#-######), so level 1-2 share h2 and everything level 3+ shares h3
            flush_paragraph(); flush_list()
            level = len(heading_match.group(1))
            style = h2_style if level <= 2 else h3_style
            story.append(Paragraph(_inline_markdown_to_html(heading_match.group(2)), style))
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
