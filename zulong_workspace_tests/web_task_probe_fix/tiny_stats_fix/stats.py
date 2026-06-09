"""tiny_stats_fix.stats — minimal statistical functions using only the standard library."""

from typing import List, Union

Number = Union[int, float]


def add(a: Number, b: Number) -> Number:
    """Return the sum of two numbers."""
    return a + b


def mean(values: List[Number]) -> float:
    """Return the arithmetic mean of a non-empty list of numbers.

    Raises:
        ValueError: if *values* is empty.
    """
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)


def variance(values: List[Number]) -> float:
    """Return the population variance of a non-empty list of numbers.

    Raises:
        ValueError: if *values* is empty.
    """
    if not values:
        raise ValueError("variance() requires at least one value")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)
