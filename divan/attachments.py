"""Document and URL ingestion for Divan deliberations.

Handles loading files (PDF, text, markdown, etc.) and URLs, extracting
text content, and formatting it for injection into advisor prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ~4000 tokens per attachment
MAX_ATTACHMENT_CHARS = 12000


@dataclass
class Attachment:
    """A loaded document or URL with extracted text content."""

    name: str  # filename or URL
    content: str  # extracted text
    source: str  # "file" or "url"


def load_file_attachment(filepath: str) -> Attachment:
    """Load a file and extract its text content.

    Supports PDF (via pymupdf), and plain text formats (.md, .txt, .csv, etc.).
    Content is truncated to MAX_ATTACHMENT_CHARS.

    Args:
        filepath: Path to the file to load.

    Returns:
        An Attachment with extracted text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    name = path.name

    if path.suffix.lower() == ".pdf":
        content = _extract_pdf_text(path)
    else:
        # Treat as plain text
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

    content = _truncate(content)
    return Attachment(name=name, content=content, source="file")


def load_url_attachment(url: str) -> Attachment:
    """Fetch a URL and extract its text content.

    Uses httpx to fetch the page and strips HTML tags to extract readable text.
    Content is truncated to MAX_ATTACHMENT_CHARS.

    Args:
        url: The URL to fetch.

    Returns:
        An Attachment with extracted text.

    Raises:
        RuntimeError: If the URL cannot be fetched.
    """
    import httpx

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch URL: {e}") from e

    content_type = response.headers.get("content-type", "")
    raw = response.text

    if "html" in content_type:
        content = _strip_html(raw)
    else:
        content = raw

    content = _truncate(content)
    return Attachment(name=url, content=content, source="url")


def format_attachments(attachments: list[Attachment]) -> str:
    """Format a list of attachments into a block for advisor prompts.

    Returns a string like:

        Attached documents:

        --- filename.pdf ---
        [content]
        ---

        --- https://example.com ---
        [content]
        ---
    """
    if not attachments:
        return ""

    parts = ["Attached documents:"]
    for att in attachments:
        parts.append("")
        parts.append(f"--- {att.name} ---")
        parts.append(att.content)
        parts.append("---")

    return "\n".join(parts)


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using pymupdf."""
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF support. Install it with: uv add pymupdf"
        )

    doc = pymupdf.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()

    return "\n\n".join(pages)


def _strip_html(html: str) -> str:
    """Strip HTML tags and extract readable text.

    Removes script/style blocks, then strips remaining tags,
    and collapses excessive whitespace.
    """
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Collapse whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _truncate(text: str) -> str:
    """Truncate text to MAX_ATTACHMENT_CHARS with a marker."""
    if len(text) <= MAX_ATTACHMENT_CHARS:
        return text
    return text[:MAX_ATTACHMENT_CHARS] + "\n\n[... content truncated ...]"
