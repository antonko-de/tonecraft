"""Input validation and prompt injection detection for domain context files."""

import logging
import re
import unicodedata

from tonecraft.schemas import AgentBrief, ContextDocument, GuardrailConfig, GuardrailResult, QAPair

logger = logging.getLogger(__name__)

# Zero-width and invisible characters commonly used to bypass regex matching
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u00ad\ufeff\u2060\u180e]")

# Prompt injection patterns (case-insensitive, applied after normalization)
_INJECTION_PATTERNS = [
    # Classic "ignore instructions" family
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"do\s+not\s+follow\s+(your|the|any)\s+(guidelines|rules|instructions)",
    r"override\s+(your\s+)?(previous\s+)?instructions?",
    # System prompt / role injection
    r"\[SYSTEM\]",
    r"<\s*/?system\s*>",
    r"<\s*/?instructions?\s*>",
    r"new\s+system\s+prompt",
    r"(system|assistant|user)\s*:\s",   # role-tag injection
    # Identity / restriction bypass
    r"you\s+are\s+now\s+(a\s+)?different",
    r"act\s+as\s+(DAN|an?\s+unrestricted|an?\s+unfiltered)",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"jailbreak",
    r"bypass\s+(all\s+)?(safety|security|content)\s+(rules|filters|checks)",
    # Exfiltration / task hijacking
    r"reveal\s+(your|the)\s+(system\s+)?prompt",
    r"instead\s+of\s+.{0,40}(output|reveal|show|print)",
    r"output\s+(the\s+)?(system\s+)?prompt",
    # Safety/restriction bypass
    r"(no|without)\s+restrictions?",
    r"bypass(es)?\s+(all\s+)?(safety|security|content|any)",
    r"ignor(e|ing)\s+(all\s+)?(safety|rules|guidelines)",
]

# Disallowed content keywords (applied to lowercased normalized text)
_BLOCKLIST = [
    "how to harm",
    "how to kill",
    "how to hurt",
    "illegal drugs",
    "contraband",
    "child exploitation",
    "csam",
    "how to make a bomb",
    "how to make explosives",
    "how to synthesize",
    "self-harm instructions",
    "suicide method",
]


# Minimal Cyrillic/Greek confusables → Latin equivalents.
# NFKC alone does not handle script-level lookalikes.
# Each entry maps a non-Latin codepoint to its visually identical Latin equivalent.
_CONFUSABLES = str.maketrans({
    # Cyrillic lookalikes
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "і": "i", "А": "A", "В": "B", "С": "C",
    "Е": "E", "І": "I", "К": "K", "М": "M", "О": "O",
    "Р": "P", "Т": "T", "Х": "X",
    # Greek lookalikes
    "α": "a", "β": "b", "ε": "e", "κ": "k", "ν": "v",
    "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
})


def _normalize(text: str) -> str:
    """Normalize text to defeat unicode/homoglyph/zero-width bypass attempts.

    Steps:
    1. NFKC normalization — collapses fullwidth, compatibility, and composed forms
       (e.g. ［SYSTEM］ → [SYSTEM], ａ → a)
    2. Strip zero-width and invisible characters commonly used to split keywords
    3. Map common Cyrillic/Greek script lookalikes to their Latin equivalents
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.translate(_CONFUSABLES)
    return text


def _scan(text: str, config: GuardrailConfig) -> list[str]:
    """Run all pattern checks on an already-normalized text string.

    Returns a list of violation messages. Empty list = clean.
    Collects ALL violations rather than stopping at the first match,
    so callers get full visibility into what was detected.
    """
    violations: list[str] = []
    text_lower = text.lower()

    # Injection patterns
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"Prompt injection pattern detected: '{pattern}'.")

    # Built-in blocklist
    for phrase in _BLOCKLIST:
        if phrase in text_lower:
            violations.append(f"Disallowed content detected: '{phrase}'.")

    # Custom blocklist from config
    for phrase in config.custom_blocklist:
        if phrase.lower() in text_lower:
            violations.append(f"Custom blocklist match: '{phrase}'.")

    return violations


def validate_context(doc: ContextDocument, config: GuardrailConfig) -> GuardrailResult:
    """Validate a parsed ContextDocument against guardrail rules.

    Checks (in order, all violations collected — no early exit):
    1. Total character length
    2. Topic count bounds
    3. Injection patterns and blocklist — scanned per-field AND on full text
       to catch both individual-field attacks and cross-field split injections

    Returns a GuardrailResult. Does not raise — callers decide what to do
    with violations (the parser raises GuardrailViolationError if needed).
    """
    violations: list[str] = []

    # Normalize all fields up front
    domain = _normalize(doc.domain)
    q_role = _normalize(doc.questioner_role)
    r_role = _normalize(doc.responder_role)
    tone = _normalize(doc.tone_guidelines)
    topics = [_normalize(t) for t in doc.topics]
    constraints = [_normalize(c) for c in doc.constraints]

    # 1. Length check (on normalized concatenated text)
    full_text = " ||| ".join([domain, q_role, r_role, tone, " ".join(topics), " ".join(constraints)])
    if len(full_text) > config.max_length:
        violations.append(
            f"Context length ({len(full_text)} chars) exceeds maximum ({config.max_length} chars)."
        )

    # 2. Topic count
    n_topics = len(doc.topics)
    if n_topics < config.min_topics:
        violations.append(f"Too few topics ({n_topics}); minimum is {config.min_topics}.")
    if n_topics > config.max_topics:
        violations.append(f"Too many topics ({n_topics}); maximum is {config.max_topics}.")

    # 3. Scan each field individually (catches single-field attacks)
    #    and the full joined text (catches cross-field split injections)
    fields_to_scan = {
        "domain": domain,
        "questioner_role": q_role,
        "responder_role": r_role,
        "tone_guidelines": tone,
        "topics": " ".join(topics),
        "constraints": " ".join(constraints),
        "_full": full_text,
    }
    seen: set[str] = set()
    for field_name, field_text in fields_to_scan.items():
        for violation in _scan(field_text, config):
            if violation not in seen:
                seen.add(violation)
                violations.append(violation)

    passed = len(violations) == 0
    if not passed:
        logger.warning("Guardrail violations (%d): %s", len(violations), violations)
    else:
        logger.debug("Guardrail passed for domain: %r", doc.domain[:40])

    return GuardrailResult(passed=passed, violations=violations)


def validate_brief(brief: AgentBrief, config: GuardrailConfig) -> GuardrailResult:
    """Validate domain expert AgentBrief output before passing to downstream agents.

    Prevents indirect prompt injection: a crafted context that passes input
    guardrails but manipulates the domain expert into producing malicious briefs.
    """
    violations: list[str] = []
    seen: set[str] = set()

    fields = {
        "persona": _normalize(brief.persona),
        "topic_focus": _normalize(brief.topic_focus),
        "directives": " ||| ".join(_normalize(d) for d in brief.directives),
    }
    for field_name, text in fields.items():
        for violation in _scan(text, config):
            if violation not in seen:
                seen.add(violation)
                violations.append(violation)

    passed = len(violations) == 0
    if not passed:
        logger.warning("AgentBrief guardrail violations: %s", violations)
    return GuardrailResult(passed=passed, violations=violations)


def validate_qa_pair(pair: QAPair, config: GuardrailConfig) -> GuardrailResult:
    """Validate a generated QAPair before writing to output.

    Prevents training data poisoning: a compromised LLM producing output
    containing injection payloads or harmful content that would be baked
    into the fine-tuning dataset.
    """
    violations: list[str] = []
    seen: set[str] = set()

    fields = {
        "question": _normalize(pair.question),
        "answer": _normalize(pair.answer),
    }
    for field_name, text in fields.items():
        for violation in _scan(text, config):
            if violation not in seen:
                seen.add(violation)
                violations.append(violation)

    passed = len(violations) == 0
    if not passed:
        logger.warning("QAPair guardrail violations: %s", violations)
    return GuardrailResult(passed=passed, violations=violations)
