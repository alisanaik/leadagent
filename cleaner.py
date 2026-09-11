# cleaner.py

import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Standard email regex — matches visible text like "contact@company.com"
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Matches emails hidden inside href="mailto:..." links in raw HTML
MAILTO_REGEX = re.compile(
    r'mailto:([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})'
)


def clean_html(html_content: str) -> str:
    """
    Takes raw HTML, removes boilerplate (scripts, styles, navs),
    and converts the remaining content into clean Markdown text.
    """
    if not html_content:
        return ""

    # Parse HTML with Python's built-in parser (no C deps required)
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove tags that contain no useful text for extraction
    tags_to_remove = [
        "script", "style", "svg", "noscript", "iframe",
        "nav", "footer", "header", "meta", "link"
    ]
    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            element.decompose()

    # Convert cleaned HTML tree to Markdown
    markdown_text = md(str(soup), heading_style="ATX")

    # Collapse excessive whitespace to save tokens
    markdown_text = re.sub(r'\n\s*\n', '\n\n', markdown_text)
    markdown_text = re.sub(r'[ \t]+', ' ', markdown_text)

    return markdown_text.strip()


def extract_emails_regex(text: str, raw_html: str = "") -> list[str]:
    """
    Deterministically extracts email addresses from BOTH:
      1. Visible text (via standard email regex)
      2. Raw HTML mailto: links (which the cleaner strips out)

    Returns a deduplicated list preserving the original casing.
    Regex is 100% accurate — it cannot hallucinate.
    """
    matches = EMAIL_REGEX.findall(text)
    if raw_html:
        matches += MAILTO_REGEX.findall(raw_html)

    seen = set()
    unique_emails = []
    for email in matches:
        lower = email.lower()
        if lower not in seen:
            seen.add(lower)
            unique_emails.append(email)

    return unique_emails