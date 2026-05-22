from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from loguru import logger

from ..models import Trade, TradeSignal

# City coordinates for distance calculation
CITY_COORDS: dict[str, tuple[float, float]] = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Miami": (25.7617, -80.1918),
    "Chicago": (41.8781, -87.6298),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8688, 151.2093),
    "Dubai": (25.2048, 55.2708),
    "Berlin": (52.5200, 13.4050),
    "Toronto": (43.6532, -79.3832),
    "Singapore": (1.3521, 103.8198),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


class CorrelationDetector:
    """
    Detect correlated exposure across open positions.
    Trades that are geographically or meteorologically correlated
    should count against position limits.
    """

    CORRELATION_LIMIT_DEFAULT = 0.7

    def calculate_portfolio_correlation(self, open_trades: list[Trade]) -> float:
        """
        Return a portfolio-level correlation score (0-1).
        Higher means open positions are more correlated.
        Uses pairwise average correlation.
        """
        if len(open_trades) < 2:
            return 0.0

        pairs = []
        for i in range(len(open_trades)):
            for j in range(i + 1, len(open_trades)):
                score = self._pairwise_correlation(open_trades[i], open_trades[j])
                pairs.append(score)

        return float(sum(pairs) / len(pairs)) if pairs else 0.0

    def get_correlated_exposure(
        self, new_signal: TradeSignal, open_trades: list[Trade]
    ) -> float:
        """
        Return a correlation score (0-1) between the proposed new trade and
        the existing portfolio.  1.0 = perfectly correlated.
        """
        if not open_trades:
            return 0.0

        scores = []
        for trade in open_trades:
            # Build a pseudo-Trade wrapper for the new signal using the open trade's market
            score = self._signal_vs_trade_correlation(new_signal, trade)
            scores.append(score)

        return float(max(scores)) if scores else 0.0

    def exceeds_correlation_limit(
        self, score: float, limit: float = CORRELATION_LIMIT_DEFAULT
    ) -> bool:
        return score >= limit

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _pairwise_correlation(self, t1: Trade, t2: Trade) -> float:
        """Return a 0-1 correlation score between two open trades."""
        # Same market
        if t1.market.id == t2.market.id:
            return 1.0

        # Same location
        if (
            t1.market.location
            and t2.market.location
            and t1.market.location.lower() == t2.market.location.lower()
        ):
            return 1.0

        # Geographic distance
        dist_score = self._location_distance_score(
            t1.market.location, t2.market.location
        )

        # Same date window (same week)
        date_score = self._date_proximity_score(
            t1.market.resolution_date, t2.market.resolution_date
        )

        return float(dist_score * 0.7 + date_score * 0.3)

    def _signal_vs_trade_correlation(
        self, signal: TradeSignal, trade: Trade
    ) -> float:
        """Correlation score between a new signal and an existing open trade."""
        if signal.market_id == trade.market.id:
            return 1.0
        # Without full market data for the new signal we use market_id prefix heuristic
        # (e.g. same slug prefix indicates same region)
        prefix_match = (
            signal.market_id[:8] == trade.market.id[:8]
            if len(signal.market_id) >= 8 and len(trade.market.id) >= 8
            else False
        )
        return 0.6 if prefix_match else 0.0

    def _location_distance_score(
        self, loc1: Optional[str], loc2: Optional[str]
    ) -> float:
        """
        Convert geographic location info to a 0-1 correlation score.
        Same city → 1.0. Within 200 km → 0.8. Same country (rough) → 0.4.
        """
        if not loc1 or not loc2:
            return 0.0
        if loc1.lower() == loc2.lower():
            return 1.0

        coords1 = CITY_COORDS.get(loc1)
        coords2 = CITY_COORDS.get(loc2)

        if coords1 and coords2:
            km = haversine_km(*coords1, *coords2)
            if km <= 200:
                return 0.8
            elif km <= 500:
                return 0.5
            elif km <= 1500:
                return 0.2
            return 0.0

        return 0.0

    def _date_proximity_score(
        self, date1: Optional[str], date2: Optional[str]
    ) -> float:
        """Return 0.4 if both dates fall in the same calendar week, else 0."""
        if not date1 or not date2:
            return 0.0
        try:
            d1 = datetime.fromisoformat(date1.replace("Z", "+00:00"))
            d2 = datetime.fromisoformat(date2.replace("Z", "+00:00"))
            same_week = abs((d1 - d2).days) <= 7
            return 0.4 if same_week else 0.0
        except Exception:
            return 0.0
