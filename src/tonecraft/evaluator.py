"""Filters QA pairs by confidence threshold."""

import logging

from tonecraft.schemas import QAPair

logger = logging.getLogger(__name__)


def evaluate(pairs: list[QAPair], threshold: float) -> tuple[list[QAPair], list[QAPair]]:
    """Split pairs into accepted (confidence >= threshold) and rejected."""
    accepted = [p for p in pairs if p.confidence >= threshold]
    rejected = [p for p in pairs if p.confidence < threshold]
    logger.info(
        "Evaluated %d pairs (threshold=%.2f): %d accepted, %d rejected",
        len(pairs), threshold, len(accepted), len(rejected),
    )
    if rejected:
        for p in rejected:
            logger.debug("Rejected [confidence=%.2f, topic=%r]: %r", p.confidence, p.topic, p.question[:60])
    return accepted, rejected
