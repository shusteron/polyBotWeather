from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from loguru import logger

from ..models import MarketData, Trade, TradeSignal


class PaperTrader:
    """
    Simulated paper trading engine.
    All positions are virtual — no real money is at risk.
    State is persisted to data/paper_trades.json.
    """

    def __init__(
        self,
        capital: float = 10_000.0,
        data_dir: str = "data",
        simulated_slippage: float = 0.002,
        simulated_fee: float = 0.002,
    ):
        self.initial_capital = capital
        self.data_dir = data_dir
        self.simulated_slippage = simulated_slippage
        self.simulated_fee = simulated_fee
        self._trades_file = os.path.join(data_dir, "paper_trades.json")
        self._state: dict = {"capital": capital, "trades": []}
        self._load_state()

    @property
    def capital(self) -> float:
        return float(self._state.get("capital", self.initial_capital))

    def execute_trade(self, signal: TradeSignal, market: MarketData) -> Optional[Trade]:
        """
        Simulate a fill at the current market price (plus slippage/fees).
        Returns the created Trade object or None if insufficient capital.
        """
        size = signal.recommended_size
        if size <= 0:
            logger.warning(f"execute_trade called with zero size for {signal.market_id}")
            return None

        if size > self.capital:
            logger.warning(
                f"Insufficient paper capital: need ${size:.2f}, have ${self.capital:.2f}"
            )
            size = self.capital * 0.9  # use 90% of remaining capital as fallback

        # Determine fill price with slippage
        if signal.action == "BUY_YES":
            base_price = market.yes_price
            fill_price = min(0.999, base_price + self.simulated_slippage)
        elif signal.action == "BUY_NO":
            base_price = market.no_price
            fill_price = min(0.999, base_price + self.simulated_slippage)
        else:
            logger.warning(f"Cannot execute trade with action {signal.action}")
            return None

        # Apply fee
        effective_cost = size * (1.0 + self.simulated_fee)
        if effective_cost > self.capital:
            effective_cost = self.capital
            size = effective_cost / (1.0 + self.simulated_fee)

        trade = Trade(
            signal=signal,
            market=market,
            timestamp=datetime.utcnow().isoformat(),
            paper_fill_price=round(fill_price, 6),
            paper_size=round(size, 2),
            status="OPEN",
            outcome=None,
            pnl=None,
        )

        # Deduct from paper capital
        self._state["capital"] = round(self.capital - effective_cost, 4)
        self._state["trades"].append(trade.model_dump())
        self._save_state()

        logger.info(
            f"Paper trade OPENED: {signal.action} {market.title} "
            f"| fill={fill_price:.4f} | size=${size:.2f}"
        )
        return trade

    def resolve_trade(self, market_id: str, outcome: bool) -> Optional[Trade]:
        """
        Close an open paper trade.
        outcome=True → YES resolves at $1.00
        outcome=False → NO resolves at $1.00 (YES is worthless at $0)
        """
        for i, t_dict in enumerate(self._state["trades"]):
            if t_dict.get("market", {}).get("id") == market_id and t_dict.get("status") == "OPEN":
                trade = Trade(**t_dict)

                # Calculate PnL
                fill_price = trade.paper_fill_price
                size = trade.paper_size
                action = trade.signal.action

                if action == "BUY_YES":
                    payout = size / fill_price if outcome else 0.0
                elif action == "BUY_NO":
                    payout = size / fill_price if not outcome else 0.0
                else:
                    payout = 0.0

                pnl = round(payout - size, 4)

                # Update state
                self._state["trades"][i]["status"] = "CLOSED"
                self._state["trades"][i]["outcome"] = outcome
                self._state["trades"][i]["pnl"] = pnl
                self._state["capital"] = round(self.capital + payout, 4)
                self._save_state()

                resolved = Trade(**self._state["trades"][i])
                logger.info(
                    f"Paper trade RESOLVED: {market_id} "
                    f"| outcome={outcome} | PnL=${pnl:.2f}"
                )
                return resolved

        logger.warning(f"No open trade found for market_id={market_id}")
        return None

    def cancel_trade(self, market_id: str) -> Optional[Trade]:
        """Cancel an open trade and return the staked capital."""
        for i, t_dict in enumerate(self._state["trades"]):
            if t_dict.get("market", {}).get("id") == market_id and t_dict.get("status") == "OPEN":
                self._state["trades"][i]["status"] = "CANCELLED"
                refund = t_dict.get("paper_size", 0.0)
                self._state["capital"] = round(self.capital + refund, 4)
                self._save_state()
                return Trade(**self._state["trades"][i])
        return None

    def get_open_trades(self) -> list[Trade]:
        """Return all open (unresolved) trades."""
        result = []
        for t_dict in self._state["trades"]:
            if t_dict.get("status") == "OPEN":
                try:
                    result.append(Trade(**t_dict))
                except Exception as exc:
                    logger.warning(f"Failed to deserialize trade: {exc}")
        return result

    def get_all_trades(self) -> list[Trade]:
        """Return all trades regardless of status."""
        result = []
        for t_dict in self._state["trades"]:
            try:
                result.append(Trade(**t_dict))
            except Exception as exc:
                logger.warning(f"Failed to deserialize trade: {exc}")
        return result

    def get_pnl_summary(self) -> dict:
        """Return total PnL, win rate, and average edge captured."""
        closed = [
            t for t in self._state["trades"]
            if t.get("status") == "CLOSED" and t.get("pnl") is not None
        ]
        if not closed:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_edge_captured": 0.0,
                "n_trades": 0,
                "n_wins": 0,
                "n_losses": 0,
            }

        total_pnl = sum(t["pnl"] for t in closed)
        wins = [t for t in closed if t["pnl"] > 0]
        win_rate = len(wins) / len(closed)
        avg_edge = sum(
            abs(t.get("signal", {}).get("edge", 0.0)) for t in closed
        ) / len(closed)

        return {
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "avg_edge_captured": round(avg_edge, 4),
            "n_trades": len(closed),
            "n_wins": len(wins),
            "n_losses": len(closed) - len(wins),
        }

    def get_portfolio_value(self) -> float:
        """Available capital + mark-to-market value of open positions."""
        open_value = sum(t.paper_size for t in self.get_open_trades())
        return round(self.capital + open_value, 2)

    def _save_state(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            with open(self._trades_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as exc:
            logger.error(f"Failed to save paper trades state: {exc}")

    def _load_state(self) -> None:
        if os.path.exists(self._trades_file):
            try:
                with open(self._trades_file, "r") as f:
                    loaded = json.load(f)
                self._state = loaded
            except Exception as exc:
                logger.warning(f"Failed to load paper trades state: {exc}")
                self._state = {"capital": self.initial_capital, "trades": []}
        else:
            self._state = {"capital": self.initial_capital, "trades": []}
