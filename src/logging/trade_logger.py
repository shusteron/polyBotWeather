from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from ..models import (
    CalibrationRecord,
    EnsembleAnalysis,
    MarketData,
    Trade,
)


class TradeLogger:
    """
    Full audit trail for all bot events.
    Logs go to console + file. Structured events are stored as JSONL.
    """

    def __init__(
        self,
        data_dir: str = "data",
        log_file: str = "data/bot.log",
        events_file: str = "data/events.jsonl",
        level: str = "DEBUG",
        rotation: str = "10 MB",
        retention: str = "30 days",
    ):
        self.data_dir = data_dir
        self.log_file = log_file
        self.events_file = events_file
        os.makedirs(data_dir, exist_ok=True)

        # Remove default loguru sink
        logger.remove()

        # Console sink
        logger.add(sys.stderr, level=level, colorize=True,
                   format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                          "<level>{level: <8}</level> | "
                          "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                          "<level>{message}</level>")

        # File sink with rotation
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    def log_scan_start(self, n_markets: int) -> None:
        msg = f"Market scan started — evaluating {n_markets} weather markets"
        logger.info(msg)
        self._write_event("SCAN_START", {"n_markets": n_markets})

    def log_market_analysis(
        self,
        market: MarketData,
        ensemble: EnsembleAnalysis,
        confidence: Any,
        edge: float,
    ) -> None:
        msg = (
            f"Analysing [{market.id}] '{market.title[:60]}' | "
            f"ensemble_mean={ensemble.weighted_mean:.2f} | "
            f"edge={edge:+.4f} | confidence={confidence.total:.1f}"
        )
        logger.debug(msg)
        self._write_event("MARKET_ANALYSIS", {
            "market_id": market.id,
            "title": market.title,
            "location": market.location,
            "threshold": market.threshold,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
            "ensemble_mean": ensemble.weighted_mean,
            "ensemble_std": ensemble.weighted_std,
            "provider_agreement": ensemble.provider_agreement,
            "edge": edge,
            "confidence": confidence.total,
            "confidence_breakdown": confidence.breakdown,
        })

    def log_trade_rejected(self, market: MarketData, reasons: list[str]) -> None:
        msg = (
            f"Trade REJECTED [{market.id}] '{market.title[:60]}' | "
            f"reasons: {'; '.join(reasons)}"
        )
        logger.info(msg)
        self._write_event("TRADE_REJECTED", {
            "market_id": market.id,
            "title": market.title,
            "rejection_reasons": reasons,
        })

    def log_trade_entered(self, trade: Trade) -> None:
        msg = (
            f"Trade ENTERED [{trade.market.id}] | "
            f"action={trade.signal.action} | "
            f"fill=${trade.paper_fill_price:.4f} | "
            f"size=${trade.paper_size:.2f} | "
            f"edge={trade.signal.edge:+.4f}"
        )
        logger.success(msg)
        self._write_event("TRADE_ENTERED", {
            "market_id": trade.market.id,
            "title": trade.market.title,
            "action": trade.signal.action,
            "fill_price": trade.paper_fill_price,
            "size": trade.paper_size,
            "edge": trade.signal.edge,
            "model_probability": trade.signal.model_probability,
            "market_probability": trade.signal.market_probability,
            "confidence": trade.signal.confidence_score.total,
            "kelly_size": trade.signal.kelly_size,
            "timestamp": trade.timestamp,
        })

    def log_trade_resolved(self, trade: Trade) -> None:
        pnl = trade.pnl or 0.0
        emoji = "+" if pnl >= 0 else ""
        msg = (
            f"Trade RESOLVED [{trade.market.id}] | "
            f"outcome={trade.outcome} | "
            f"PnL=${emoji}{pnl:.2f}"
        )
        if pnl >= 0:
            logger.success(msg)
        else:
            logger.warning(msg)

        self._write_event("TRADE_RESOLVED", {
            "market_id": trade.market.id,
            "title": trade.market.title,
            "action": trade.signal.action,
            "outcome": trade.outcome,
            "pnl": pnl,
            "fill_price": trade.paper_fill_price,
            "size": trade.paper_size,
            "edge": trade.signal.edge,
        })

    def log_calibration_update(self, record: CalibrationRecord) -> None:
        msg = (
            f"Calibration updated [{record.market_id}] | "
            f"predicted={record.predicted_probability:.3f} | "
            f"outcome={record.outcome}"
        )
        logger.debug(msg)
        self._write_event("CALIBRATION_UPDATE", record.model_dump())

    def log_error(self, context: str, error: Exception) -> None:
        logger.error(f"ERROR in {context}: {error}")
        self._write_event("ERROR", {"context": context, "error": str(error)})

    def log_status(self, portfolio_value: float, n_open: int, pnl_summary: dict) -> None:
        msg = (
            f"STATUS | portfolio=${portfolio_value:.2f} | "
            f"open_positions={n_open} | "
            f"total_pnl=${pnl_summary.get('total_pnl', 0):.2f} | "
            f"win_rate={pnl_summary.get('win_rate', 0):.1%}"
        )
        logger.info(msg)
        self._write_event("STATUS", {
            "portfolio_value": portfolio_value,
            "n_open_positions": n_open,
            **pnl_summary,
        })

    def _write_event(self, event_type: str, payload: dict) -> None:
        """Append a structured JSON event to the events JSONL file."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **payload,
        }
        try:
            with open(self.events_file, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:
            logger.error(f"Failed to write event to JSONL: {exc}")
