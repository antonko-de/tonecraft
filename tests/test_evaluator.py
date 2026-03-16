"""Tests for the QA pair evaluator."""

import pytest

from tonecraft.evaluator import evaluate
from tonecraft.schemas import QAPair


def _pair(confidence: float, topic: str = "Menu") -> QAPair:
    return QAPair(question="Q", answer="A", topic=topic, confidence=confidence)


def test_pairs_above_threshold_are_accepted():
    pairs = [_pair(0.9), _pair(0.8)]
    accepted, rejected = evaluate(pairs, threshold=0.7)
    assert len(accepted) == 2
    assert len(rejected) == 0


def test_pairs_below_threshold_are_rejected():
    pairs = [_pair(0.5), _pair(0.3)]
    accepted, rejected = evaluate(pairs, threshold=0.7)
    assert len(accepted) == 0
    assert len(rejected) == 2


def test_pair_exactly_at_threshold_is_accepted():
    pairs = [_pair(0.7)]
    accepted, rejected = evaluate(pairs, threshold=0.7)
    assert len(accepted) == 1
    assert len(rejected) == 0


def test_empty_input_returns_empty_lists():
    accepted, rejected = evaluate([], threshold=0.7)
    assert accepted == []
    assert rejected == []


def test_mixed_confidence_split_correctly():
    pairs = [_pair(0.9), _pair(0.6), _pair(0.75), _pair(0.4), _pair(0.7)]
    accepted, rejected = evaluate(pairs, threshold=0.7)
    assert len(accepted) == 3
    assert len(rejected) == 2


def test_returns_tuple_of_two_lists():
    result = evaluate([_pair(0.8)], threshold=0.7)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


def test_original_pairs_preserved_in_output():
    pair = _pair(0.9, topic="Wine")
    accepted, _ = evaluate([pair], threshold=0.7)
    assert accepted[0] is pair
