from __future__ import annotations

from loguru import logger


class EdgeDetector:
    def calculate_edge(
        self, model_probability: float, market_probability: float
    ) -> float:
        """
        Edge = model probability minus market implied probability.
        Positive edge means model says YES is more likely than priced.
        Negative edge means model says NO is more likely than priced.
        """
        return float(model_probability - market_probability)

    def has_edge(self, edge: float, min_edge: float = 0.08) -> bool:
        """Return True if the absolute edge meets the minimum threshold."""
        return abs(edge) >= min_edge

    def get_direction(self, edge: float) -> str:
        """
        Determine trade direction based on edge sign.
        Positive edge → buy YES (market underpricing YES outcome).
        Negative edge → buy NO (market underpricing NO outcome).
        """
        if edge > 0:
            return "BUY_YES"
        elif edge < 0:
            return "BUY_NO"
        return "NO_TRADE"

    def adjust_edge_for_spread(
        self, edge: float, spread_pct: float, direction: str
    ) -> float:
        """
        Reduce effective edge by half the spread (entry cost).
        BUY_YES pays the ask; BUY_NO also pays the ask (which is 1 - bid).
        """
        effective_edge = abs(edge) - (spread_pct / 2.0)
        return max(0.0, effective_edge) * (1 if edge >= 0 else -1)

    def probability_to_decimal_odds(self, probability: float) -> float:
        """Convert a probability to decimal odds (e.g. 0.6 → 1.667)."""
        if probability <= 0 or probability >= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
        return 1.0 / probability

    def implied_edge_ratio(self, edge: float, market_probability: float) -> float:
        """
        Edge as a fraction of the market probability — a relative measure.
        E.g. 0.08 edge on a 0.40 market is a 20% relative edge.
        """
        if market_probability <= 0:
            return 0.0
        return edge / market_probability
