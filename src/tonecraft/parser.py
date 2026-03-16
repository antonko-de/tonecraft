"""Parse a domain context .md file into a ContextDocument."""

import logging
import re
from pathlib import Path

from tonecraft.schemas import ContextDocument

logger = logging.getLogger(__name__)

# Required headings in the context file
_REQUIRED = ("Domain", "Roles", "Topics")


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown into a dict of {heading: body} using top-level # headings."""
    sections: dict[str, str] = {}
    current_heading: str | None = None
    lines: list[str] = []

    for line in text.splitlines():
        match = re.match(r"^#\s+(.+)$", line)
        if match:
            if current_heading is not None:
                sections[current_heading] = "\n".join(lines).strip()
            current_heading = match.group(1).strip()
            lines = []
        else:
            if current_heading is not None:
                lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(lines).strip()

    return sections


def _extract_roles(roles_text: str) -> tuple[str, str]:
    """Extract questioner and responder roles from the Roles section body."""
    questioner = ""
    responder = ""
    for line in roles_text.splitlines():
        if re.search(r"\*\*Questioner\*\*", line, re.IGNORECASE):
            questioner = re.sub(r".*\*\*Questioner\*\*\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        elif re.search(r"\*\*Responder\*\*", line, re.IGNORECASE):
            responder = re.sub(r".*\*\*Responder\*\*\s*:\s*", "", line, flags=re.IGNORECASE).strip()
    return questioner, responder


def _extract_list_items(text: str) -> list[str]:
    """Extract bullet list items (- item or * item) from a section body."""
    items = []
    for line in text.splitlines():
        match = re.match(r"^\s*[-*]\s+(.+)$", line)
        if match:
            items.append(match.group(1).strip())
    return items


def parse_context(path: str | Path) -> ContextDocument:
    path = Path(path)
    logger.debug("Parsing context file: %s", path)
    text = path.read_text(encoding="utf-8")
    sections = _split_sections(text)

    for required in _REQUIRED:
        if required not in sections:
            raise ValueError(
                f"Missing required section '# {required}' in {path.name}. "
                f"Required sections: {', '.join(_REQUIRED)}."
            )

    domain = sections["Domain"]
    if not domain:
        raise ValueError(f"'# Domain' section is empty in {path.name}.")

    questioner_role, responder_role = _extract_roles(sections["Roles"])

    tone_guidelines = sections.get("Tone Guidelines", "").strip()

    topics = _extract_list_items(sections["Topics"])
    if not topics:
        raise ValueError(f"'# Topics' section has no list items in {path.name}.")

    constraints = _extract_list_items(sections.get("Constraints", ""))

    logger.info("Parsed context: domain=%r, topics=%d", domain[:40], len(topics))
    return ContextDocument(
        domain=domain,
        questioner_role=questioner_role,
        responder_role=responder_role,
        tone_guidelines=tone_guidelines,
        topics=topics,
        constraints=constraints,
    )
