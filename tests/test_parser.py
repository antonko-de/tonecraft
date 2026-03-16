"""Tests for the markdown context file parser."""

from pathlib import Path

import pytest

from tonecraft.parser import parse_context
from tonecraft.schemas import ContextDocument


def test_parse_restaurant_example():
    example = Path("examples/restaurant.md")
    doc = parse_context(example)
    assert isinstance(doc, ContextDocument)
    assert "restaurant" in doc.domain.lower()
    assert doc.questioner_role
    assert doc.responder_role
    assert len(doc.topics) > 0


def test_parse_sample_md_file(sample_md_file: Path):
    doc = parse_context(sample_md_file)
    assert isinstance(doc, ContextDocument)
    assert doc.domain
    assert doc.questioner_role
    assert doc.responder_role
    assert len(doc.topics) >= 1


def test_missing_domain_heading_raises(tmp_path: Path):
    md = tmp_path / "bad.md"
    md.write_text("""\
# Roles

**Questioner**: Guest
**Responder**: Waiter

# Tone Guidelines

Neutral tone.

# Topics

- Food
""")
    with pytest.raises(ValueError, match="Domain"):
        parse_context(md)


def test_missing_roles_heading_raises(tmp_path: Path):
    md = tmp_path / "bad.md"
    md.write_text("""\
# Domain

A restaurant.

# Tone Guidelines

Neutral.

# Topics

- Food
""")
    with pytest.raises(ValueError, match="Roles"):
        parse_context(md)


def test_missing_topics_heading_raises(tmp_path: Path):
    md = tmp_path / "bad.md"
    md.write_text("""\
# Domain

A restaurant.

# Roles

**Questioner**: Guest
**Responder**: Waiter

# Tone Guidelines

Neutral.
""")
    with pytest.raises(ValueError, match="Topics"):
        parse_context(md)


def test_unknown_headings_ignored(tmp_path: Path):
    md = tmp_path / "extra.md"
    md.write_text("""\
# Domain

A coffee shop.

# Roles

**Questioner**: Customer
**Responder**: Barista

# Tone Guidelines

Casual and friendly.

# Topics

- Coffee drinks
- Pastries

# Some Unknown Section

This should be ignored silently.
""")
    doc = parse_context(md)
    assert doc.domain
    assert len(doc.topics) == 2


def test_empty_topics_section_raises(tmp_path: Path):
    md = tmp_path / "empty_topics.md"
    md.write_text("""\
# Domain

A restaurant.

# Roles

**Questioner**: Guest
**Responder**: Waiter

# Tone Guidelines

Neutral.

# Topics

""")
    with pytest.raises(ValueError):
        parse_context(md)


def test_constraints_are_optional(tmp_path: Path):
    md = tmp_path / "no_constraints.md"
    md.write_text("""\
# Domain

A bookstore.

# Roles

**Questioner**: Customer
**Responder**: Bookseller

# Tone Guidelines

Helpful and knowledgeable.

# Topics

- Fiction recommendations
- Author information
""")
    doc = parse_context(md)
    assert doc.constraints == []
