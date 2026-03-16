"""Tests for the distribution rebalancer."""

import pytest

from tonecraft.rebalancer import compute_gaps
from tonecraft.schemas import DistributionSchema, QAPair, TopicDistribution


def _distribution(*topics: tuple[str, int]) -> DistributionSchema:
    total = sum(count for _, count in topics)
    items = [
        TopicDistribution(topic=t, weight=count / total, target_count=count)
        for t, count in topics
    ]
    return DistributionSchema(
        topics=items,
        recommended_pair_count=total,
        data_sizing_rationale="test",
    )


def _pairs(topic: str, n: int) -> list[QAPair]:
    return [QAPair(question="Q", answer="A", topic=topic, confidence=0.9) for _ in range(n)]


def test_balanced_distribution_returns_no_gaps():
    dist = _distribution(("Menu", 10), ("Wine", 10))
    pairs = _pairs("Menu", 10) + _pairs("Wine", 10)
    gaps = compute_gaps(pairs, dist)
    assert gaps == {}


def test_skewed_distribution_identifies_underrepresented_topic():
    dist = _distribution(("Menu", 10), ("Wine", 10))
    pairs = _pairs("Menu", 10) + _pairs("Wine", 3)
    gaps = compute_gaps(pairs, dist)
    assert "Wine" in gaps
    assert gaps["Wine"] == 7


def test_completely_missing_topic_has_full_gap():
    dist = _distribution(("Menu", 5), ("Wine", 5), ("Dessert", 5))
    pairs = _pairs("Menu", 5) + _pairs("Wine", 5)
    gaps = compute_gaps(pairs, dist)
    assert gaps.get("Dessert") == 5


def test_gap_sizes_proportional_to_deficit():
    dist = _distribution(("Menu", 10), ("Wine", 10), ("Dessert", 10))
    pairs = _pairs("Menu", 10) + _pairs("Wine", 6) + _pairs("Dessert", 2)
    gaps = compute_gaps(pairs, dist)
    assert gaps["Wine"] == 4
    assert gaps["Dessert"] == 8


def test_over_represented_topic_produces_no_gap():
    dist = _distribution(("Menu", 5), ("Wine", 5))
    pairs = _pairs("Menu", 10) + _pairs("Wine", 5)
    gaps = compute_gaps(pairs, dist)
    assert "Menu" not in gaps
    assert "Wine" not in gaps


def test_returns_dict():
    dist = _distribution(("Menu", 5))
    gaps = compute_gaps([], dist)
    assert isinstance(gaps, dict)


def test_empty_pairs_all_topics_are_gaps():
    dist = _distribution(("Menu", 4), ("Wine", 6))
    gaps = compute_gaps([], dist)
    assert gaps["Menu"] == 4
    assert gaps["Wine"] == 6
