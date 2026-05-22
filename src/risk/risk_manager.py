from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from loguru import logger

from ..models import Trade, TradeSignal


class RiskManager:
    """
    Kelly-based position sizing with hard risk caps.
    All sizing is expressed as a fraction of total capital.
    """

    def __init__(
        self,
        max_trade_exposure: float = 0.05,
        max_daily_exposure: float = 0.15,
        max_concurrent_positions: int = 5,
        kelly_fraction: float = 0.25,
        starting_capital: float = 10_000.0,
        min_kelly_bet: float = 10.0,
        max_single_bet: float = 500.0,
    ):
        self.max_trade_exposure = max_trade_exposure
        self.max_daily_exposure = max_daily_exposure
        self.max_concurrent_positions = max_concurrent_positions
        self.kelly_fraction = kelly_fraction
        self.starting_capital = starting_capital
        self.min_kelly_bet = min_kelly_bet
        self.max_single_bet = max_single_bet

    def calculate_kelly_size(
        self,
        probability: float,
        market_price: float,
        kelly_fraction: Optional[float] = None,
        capital: Optional[float] = None,
    ) -> float:
        """
        Fractional Kelly formula.

        f* = (p * b - (1-p)) / b

        where b = net odds = (1/price - 1)
        Returns dollar amount to bet.
        """
        kf = kelly_fraction if kelly_fraction is not None else self.kelly_fraction
        cap = capital if capital is not None else self.starting_capital

        try:
            # Clamp inputs to valid ranges
            probability = max(0.001, min(0.999, probability))
            market_price = max(0.001, min(0.999, market_price))

            b = (1.0 / market_price) - 1.0   # net odds
            if b <= 0:
                return 0.0

            f_star = (probability * b - (1.0 - probability)) / b

            if f_star <= 0:
                return 0.0

            # Apply fractional Kelly
            fraction = f_star * kf

            # Cap at max_trade_exposure
            fraction = min(fraction, self.max_trade_exposure)

            dollar_size = fraction * cap
            dollar_size = max(self.min_kelly_bet, min(self.max_single_bet, dollar_size))
            return round(dollar_size, 2)

        except Exception as exc:
            logger.warning(f"Kelly calculation error: {exc}")
            return 0.0

    def check_daily_exposure(
        self,
        open_trades: list[Trade],
        proposed_size: float,
        capital: float,
    ) -> bool:
        """
        Return True if adding the proposed trade keeps daily exposure within limit.
        """
        today_str = date.today().isoformat()
        daily_committed = 0.0

        for trade in open_trades:
            if trade.timestamp and trade.timestamp.startswith(today_str):
                daily_committed += trade.paper_size

        new_total = daily_committed + proposed_size
        limit = capital * self.max_daily_exposure
        ok = new_total <= limit

        if not ok:
            logger.info(
                f"Daily exposure check FAILED: committed={daily_committed:.2f}, "
                f"proposed={proposed_size:.2f}, limit={limit:.2f}"
            )
        return ok

    def check_position_count(self, open_trades: list[Trade]) -> bool:
        """Return True if we have fewer open positions than the maximum."""
        open_count = sum(1 for t in open_trades if t.status == "OPEN")
        ok = open_count < self.max_concurrent_positions
        if not ok:
            logger.info(
                f"Position count check FAILED: {open_count} >= {self.max_concurrent_positions}"
            )
        return ok

    def get_approved_size(
        self,
        signal: TradeSignal,
        capital: float,
        open_trades: list[Trade],
    ) -> float:
        """
        Run all risk checks and return the approved dollar size.
        Returns 0.0 if any check fails.
        """
        # Check position count
        if not self.check_position_count(open_trades):
            logger.info(
                f"Trade {signal.market_id} rejected: max positions reached"
            )
            return 0.0

        # Determine preliminary size from Kelly
        raw_size = signal.kelly_size if signal.kelly_size > 0 else self.calculate_kelly_size(
            signal.model_probability, signal.market_probability, capital=capital
        )

        if raw_size <= 0:
            logger.info(f"Trade {signal.market_id} rejected: zero Kelly size")
            return 0.0

        # Check daily exposure
        if not self.check_daily_exposure(open_trades, raw_size, capital):
            # Try a reduced size that fits
            today_str = date.today().isoformat()
            daily_committed = sum(
                t.paper_size
                for t in open_trades
                if t.timestamp and t.timestamp.startswith(today_str)
            )
            remaining = capital * self.max_daily_exposure - daily_committed
            if remaining < self.min_kelly_bet:
                logger.info(
                    f"Trade {signal.market_id} rejected: no daily budget remaining"
                )
                return 0.0
            raw_size = min(raw_size, remaining)

        return round(raw_size, 2)

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskManager":
        risk = cfg.get("risk", {})
        return cls(
            max_trade_exposure=risk.get("max_trade_exposure", 0.05),
            max_daily_exposure=risk.get("max_daily_exposure", 0.15),
            max_concurrent_positions=risk.get("max_concurrent_positions", 5),
            kelly_fraction=risk.get("kelly_fraction", 0.25),
            starting_capital=risk.get("starting_capital", 10_000.0),
            min_kelly_bet=risk.get("min_kelly_bet", 10.0),
            max_single_bet=risk.get("max_single_bet", 500.0),
        )
