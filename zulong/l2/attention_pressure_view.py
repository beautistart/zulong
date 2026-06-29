"""Context pressure presentation helpers.

This module keeps three concepts separate:

1. **LLM original context**: the full context-window size reported by the LLM.
2. **Threshold budget**: the full context-window size configured in Web.
   ``threshold_budget_ratio`` is kept as compatibility telemetry and is 1.0
   in normal runtime.
3. **Actual occupied context**: the context tokens currently used by messages.
4. **Context pressure**: ``occupied_context_tokens / threshold_budget_tokens``.

Yellow/red thresholds are trigger lines on the computed pressure ratio.  They
are not budgets and must not become the denominator of the LLM-facing pressure.
The LLM-facing prompt receives only the computed pressure percentage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ContextPressureView:
    """Pressure values grounded in the LLM original context budget."""

    occupied_context_tokens: float
    llm_context_budget_tokens: float
    threshold_budget_ratio: float
    threshold_budget_tokens: float
    context_pressure_ratio: float
    context_pressure_percent: float
    yellow_ratio: float
    red_ratio: float
    tier: str

    @property
    def raw_ratio(self) -> float:
        """Backward-compatible alias for context_pressure_ratio."""
        return self.context_pressure_ratio

    @property
    def threshold_relative_ratio(self) -> float:
        """Backward-compatible alias; no second normalization is performed."""
        return self.context_pressure_ratio

    @property
    def threshold_relative_percent(self) -> float:
        """Backward-compatible alias; no second normalization is performed."""
        return self.context_pressure_percent

    @property
    def active_threshold_ratio(self) -> float:
        if self.tier == "red":
            return self.red_ratio
        if self.tier == "yellow":
            return self.yellow_ratio
        return self.yellow_ratio

    @property
    def active_threshold_budget_tokens(self) -> float:
        return self.threshold_budget_tokens * self.active_threshold_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "occupied_context_tokens": round(self.occupied_context_tokens, 1),
            "llm_context_budget_tokens": round(self.llm_context_budget_tokens, 1),
            "threshold_budget_ratio": round(self.threshold_budget_ratio, 4),
            "threshold_budget_tokens": round(self.threshold_budget_tokens, 1),
            "context_pressure_ratio": round(self.context_pressure_ratio, 4),
            "context_pressure_percent": round(self.context_pressure_percent, 1),
            "yellow_ratio": round(self.yellow_ratio, 4),
            "red_ratio": round(self.red_ratio, 4),
            "tier": self.tier,
            "active_threshold_ratio": round(self.active_threshold_ratio, 4),
            "active_threshold_budget_tokens": round(self.active_threshold_budget_tokens, 1),
            # Backward-compatible telemetry names.
            "raw_ratio": round(self.context_pressure_ratio, 4),
            "threshold_relative_ratio": round(self.context_pressure_ratio, 4),
            "threshold_relative_percent": round(self.context_pressure_percent, 1),
            "budget_reference": "threshold_budget_is_100_percent",
        }


def _safe_non_negative(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return default
    return parsed


def _safe_positive(value: Any, default: float) -> float:
    parsed = _safe_non_negative(value, default)
    return parsed if parsed > 0 else default


def _derive_tier(context_pressure_ratio: float, yellow_ratio: float, red_ratio: float, tier: str = "") -> str:
    normalized = str(tier or "").strip().lower()
    if normalized in {"red", "yellow", "green"}:
        return normalized
    if context_pressure_ratio > red_ratio:
        return "red"
    if context_pressure_ratio > yellow_ratio:
        return "yellow"
    return "green"


def build_context_pressure_view(
    occupied_context_tokens: Any,
    llm_context_budget_tokens: Any,
    threshold_budget_ratio: Any,
    yellow_ratio: Any,
    red_ratio: Any,
    *,
    tier: str = "",
) -> ContextPressureView:
    """Build a context-pressure view from token counts."""
    occupied = _safe_non_negative(occupied_context_tokens, 0.0)
    llm_budget = _safe_positive(llm_context_budget_tokens, 1.0)
    # v2.9.28: Web-configured full context window is the threshold budget.
    # Keep the ratio field for backward-compatible telemetry, but ignore legacy
    # temporary values such as 0.50/0.15 so they cannot become system policy.
    threshold_ratio = 1.0
    threshold_budget = max(1.0, llm_budget * threshold_ratio)
    pressure_ratio = occupied / threshold_budget
    red = _safe_positive(red_ratio, 1.0)
    yellow = _safe_positive(yellow_ratio, min(0.90, red))
    if yellow > red:
        yellow = red
    pressure_tier = _derive_tier(pressure_ratio, yellow, red, tier=tier)
    return ContextPressureView(
        occupied_context_tokens=occupied,
        llm_context_budget_tokens=llm_budget,
        threshold_budget_ratio=threshold_ratio,
        threshold_budget_tokens=threshold_budget,
        context_pressure_ratio=pressure_ratio,
        context_pressure_percent=pressure_ratio * 100.0,
        yellow_ratio=yellow,
        red_ratio=red,
        tier=pressure_tier,
    )


def build_threshold_pressure_view(
    raw_ratio: Any,
    yellow_ratio: Any,
    red_ratio: Any,
    *,
    tier: str = "",
) -> ContextPressureView:
    """Backward-compatible ratio entry point.

    ``raw_ratio`` is already the context pressure ratio
    (actual occupied context divided by the threshold budget).  It is therefore
    returned directly as LLM-facing pressure, not divided again by yellow/red
    trigger lines.
    """
    pressure_ratio = _safe_non_negative(raw_ratio, 0.0)
    return build_context_pressure_view(
        pressure_ratio,
        1.0,
        1.0,
        yellow_ratio,
        red_ratio,
        tier=tier,
    )
