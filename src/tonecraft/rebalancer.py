"""Compares accepted QA pairs against target distribution and reports gaps."""

import logging
from collections import Counter

from tonecraft.schemas import DistributionSchema, QAPair

logger = logging.getLogger(__name__)


def compute_gaps(pairs: list[QAPair], distribution: DistributionSchema) -> dict[str, int]:
    """Return topics that are under their target count and by how much.

    Returns a dict of {topic: deficit} for topics where accepted pair count
    is below the target. Topics at or above target are not included.
    """
    counts = Counter(p.topic for p in pairs)
    gaps = {}
    for td in distribution.topics:
        have = counts.get(td.topic, 0)
        deficit = td.target_count - have
        if deficit > 0:
            gaps[td.topic] = deficit
            logger.info(
                "Gap detected — topic=%r: have %d / target %d (need %d more)",
                td.topic, have, td.target_count, deficit,
            )
        else:
            logger.debug("Topic %r satisfied: have %d / target %d", td.topic, have, td.target_count)
    if not gaps:
        logger.info("Distribution balanced — all topics met their targets.")
    return gaps
