from __future__ import annotations

from typing import List

from schemas import ParsedManuscript


def get_title(manuscript: ParsedManuscript) -> str:
    return manuscript.metadata.title or "Untitled manuscript"


def get_abstract(manuscript: ParsedManuscript) -> str:
    return manuscript.metadata.abstractText or ""


def get_sections_text(
    manuscript: ParsedManuscript, include_headings: bool = True
) -> str:
    blocks: List[str] = []
    for section in manuscript.metadata.sections:
        heading = (section.heading or "").strip()
        text = (section.text or "").strip()
        if not text:
            continue
        if include_headings and heading:
            blocks.append(f"## {heading}\n{text}")
        else:
            blocks.append(text)
    return "\n\n".join(blocks)


def get_references_text(manuscript: ParsedManuscript, max_refs: int = 50) -> str:
    refs = manuscript.metadata.references[:max_refs]
    lines: List[str] = []
    for idx, ref in enumerate(refs):
        authors = ", ".join(ref.author) if ref.author else "Unknown authors"
        title = ref.title or "Untitled"
        venue = ref.venue or "Unknown venue"
        year = ref.year or "Unknown year"
        lines.append(f"[{idx}] {authors}. {title}. {venue}. {year}.")
    return "\n".join(lines)


def get_reference_mentions_text(
    manuscript: ParsedManuscript, max_mentions: int = 50
) -> str:
    mentions = manuscript.metadata.referenceMentions[:max_mentions]
    lines: List[str] = []
    for mention in mentions:
        lines.append(f"[ref {mention.referenceID}] context={mention.context}")
    return "\n".join(lines)


def build_manuscript_context(
    manuscript: ParsedManuscript,
    max_refs: int = 40,
    max_mentions: int = 40,
) -> str:
    title = get_title(manuscript)
    abstract = get_abstract(manuscript)
    sections = get_sections_text(manuscript)
    references = get_references_text(manuscript, max_refs=max_refs)
    mentions = get_reference_mentions_text(manuscript, max_mentions=max_mentions)

    return f"""
TITLE:
{title}

ABSTRACT:
{abstract}

SECTIONS:
{sections}

REFERENCES:
{references}

REFERENCE MENTIONS:
{mentions}
""".strip()
