from __future__ import annotations

import os
from datetime import datetime, date
from typing import Optional

import yaml
from loguru import logger

from .analysis.calibration import CalibrationEngine
from .analysis.confidence import ConfidenceScorer
from .analysis.correlation import CorrelationDetector
from .analysis.edge_detection import EdgeDetector
from .analysis.threshold import ThresholdDistanceEngine
from .execution.paper_trader import PaperTrader
from .execution.trade_filter import TradeFilter
from .export.excel_exporter import ExcelExporter
from .logging.trade_logger import TradeLogger
from .market_scanner import MarketScanner
from .models import (
    CalibrationRecord,
    ConfidenceScore,
    ForecastStabilityRecord,
    MarketData,
    Trade,
    TradeSignal,
)
from .risk.risk_manager import RiskManager
from .weather.ensemble import EnsembleAnalyzer
from .weather.open_meteo import OpenMeteoClient
from .weather.noaa import NOAAClient
from .weather.stability import ForecastStabilityEngine
from .weather.stations import get_icao, apply_bias_correction
from .market_scanner import get_city_coords

# City → (lat, lon) for coordinate lookup
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "New York": (40.7128, -74.0060),
    "NYC": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "LA": (34.0522, -118.2437),
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
    "Hong Kong": (22.3193, 114.1694),
    "Mumbai": (19.0760, 72.8777),
    "São Paulo": (-23.5505, -46.6333),
    "Sao Paulo": (-23.5505, -46.6333),
    "Mexico City": (19.4326, -99.1332),
    "Seoul": (37.5665, 126.9780),
    "Amsterdam": (52.3676, 4.9041),
    "Madrid": (40.4168, -3.7038),
    "Rome": (41.9028, 12.4964),
    "Bangkok": (13.7563, 100.5018),
    "Cairo": (30.0444, 31.2357),
    "Lagos": (6.5244, 3.3792),
    "Nairobi": (-1.2921, 36.8219),
    "Phoenix": (33.4484, -112.0740),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "San Francisco": (37.7749, -122.4194),
    "Las Vegas": (36.1699, -115.1398),
    "Orlando": (28.5383, -81.3792),
    "New Orleans": (29.9511, -90.0715),
    "Minneapolis": (44.9778, -93.2650),
    "Portland": (45.5051, -122.6750),
    "Nashville": (36.1627, -86.7816),
    "Austin": (30.2672, -97.7431),
    "Washington": (38.9072, -77.0369),
    "Philadelphia": (39.9526, -75.1652),
    "Charlotte": (35.2271, -80.8431),
    "Tampa": (27.9506, -82.4572),
    "San Diego": (32.7157, -117.1611),
    "Detroit": (42.3314, -83.0458),
    "Lisbon": (38.7169, -9.1399),
    "Vienna": (48.2082, 16.3738),
    "Warsaw": (52.2297, 21.0122),
    "Prague": (50.0755, 14.4378),
    "Copenhagen": (55.6761, 12.5683),
    "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686),
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config(path: str = CONFIG_PATH) -> dict:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        logger.warning(f"Config file not found at {abs_path}, using defaults")
        return {}
    with open(abs_path, "r") as f:
        return yaml.safe_load(f) or {}


class EliteWeatherBot:
    """
    Main orchestrator for the weather prediction market bot.
    Coordinates market scanning, forecast retrieval, analysis, and paper trading.
    """

    def __init__(self, config_path: str = CONFIG_PATH, data_dir: str = "data"):
        self.cfg = _load_config(config_path)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        log_cfg = self.cfg.get("logging", {})

        # Initialize logger first so all subsequent components can log
        self.trade_logger = TradeLogger(
            data_dir=data_dir,
            log_file=log_cfg.get("log_file", os.path.join(data_dir, "bot.log")),
            events_file=log_cfg.get("events_file", os.path.join(data_dir, "events.jsonl")),
            level=log_cfg.get("level", "DEBUG"),
            rotation=log_cfg.get("rotation", "10 MB"),
            retention=log_cfg.get("retention", "30 days"),
        )

        # Core components
        self.scanner = MarketScanner(
            base_url=self.cfg.get("polymarket", {}).get("base_urls", {}).get("gamma",
                                                                               "https://gamma-api.polymarket.com")
        )
        self.open_meteo = OpenMeteoClient(
            timeout=self.cfg.get("weather", {}).get("request_timeout", 30)
        )
        self.noaa = NOAAClient(
            timeout=self.cfg.get("weather", {}).get("request_timeout", 30)
        )
        model_weights = self.cfg.get("weather", {}).get("model_weights", None)
        self.ensemble_analyzer = EnsembleAnalyzer(model_weights=model_weights)
        self.stability_engine = ForecastStabilityEngine(data_dir=data_dir)

        # Analysis components
        self.edge_detector = EdgeDetector()
        self.calibration_engine = CalibrationEngine(data_dir=data_dir)
        self.confidence_scorer = ConfidenceScorer()
        self.threshold_engine = ThresholdDistanceEngine()
        self.correlation_detector = CorrelationDetector()

        # Risk and execution
        self.risk_manager = RiskManager.from_config(self.cfg)
        self.trade_filter = TradeFilter.from_config(self.cfg)

        capital = self.cfg.get("risk", {}).get("starting_capital", 10_000.0)
        exec_cfg = self.cfg.get("execution", {})
        self.paper_trader = PaperTrader(
            capital=capital,
            data_dir=data_dir,
            simulated_slippage=exec_cfg.get("simulated_slippage", 0.002),
            simulated_fee=exec_cfg.get("simulated_fee", 0.002),
        )

        # Export
        self.exporter = ExcelExporter()

        # Track skipped trades for reporting
        self._skipped_trades: list[dict] = []

        logger.info("EliteWeatherBot initialized successfully")

    def run_scan_cycle(self) -> None:
        """
        Full scan cycle:
        1. Scan Polymarket for weather markets
        2. Fetch forecasts for each market
        3. Analyze edge, confidence, risk
        4. Paper trade qualified signals
        """
        logger.info("=== Starting scan cycle ===")

        # 1. Scan markets
        markets = self.scanner.scan_weather_markets(limit=100)
        self.trade_logger.log_scan_start(len(markets))

        if not markets:
            logger.warning("No weather markets found — scan cycle complete")
            return

        open_trades = self.paper_trader.get_open_trades()

        for market in markets:
            try:
                self._process_market(market, open_trades)
            except Exception as exc:
                self.trade_logger.log_error(f"process_market({market.id})", exc)
                logger.exception(f"Unhandled error processing market {market.id}")

        # Final status
        self.trade_logger.log_status(
            self.paper_trader.get_portfolio_value(),
            len(self.paper_trader.get_open_trades()),
            self.paper_trader.get_pnl_summary(),
        )
        logger.info("=== Scan cycle complete ===")

    def _process_market(self, market: MarketData, open_trades: list[Trade]) -> None:
        """Process a single market through the full analysis pipeline."""
        # 2. Resolve coordinates
        coords = self._get_coords(market)
        if not coords:
            logger.debug(f"Skipping {market.id} — no coordinates for location '{market.location}'")
            return

        lat, lon = coords

        # 3. Determine target date from resolution date
        target_date = self._get_target_date(market)
        if not target_date:
            logger.debug(f"Skipping {market.id} — cannot determine target date")
            return

        # 4. Determine measurement type from market title
        title_lower = market.title.lower()
        if "highest" in title_lower or "high temp" in title_lower or "maximum" in title_lower:
            measurement = "max"
        elif "lowest" in title_lower or "low temp" in title_lower or "minimum" in title_lower:
            measurement = "min"
        elif market.event_type in ("precipitation", "snow"):
            measurement = "sum"
        else:
            measurement = "max"

        # 5. Fetch ensemble forecasts from all 5 free models
        forecasts = self.open_meteo.get_ensemble_forecast(
            lat, lon, target_date,
            location=market.location or market.id,
            measurement=measurement,
        )

        # 5b. METAR station cross-check — validates model direction against real observations
        icao = get_icao(market.location)
        station_check = None
        if icao:
            market.resolution_station = market.resolution_station or icao
            # Pull recent METAR to detect if model is systematically off for this station
            recent = self.noaa.get_metar(icao, hours=3)
            if recent:
                station_temp = recent.get("temp")
                if station_temp is not None:
                    station_check = float(station_temp)
                    logger.debug(
                        f"Station {icao} current temp: {station_check:.1f}°C "
                        f"(model ensemble mean: {forecasts[0].mean:.1f}°C if available)"
                    )
        if not forecasts:
            logger.debug(f"Skipping {market.id} — no forecast data available")
            return

        # 5. Track stability
        for forecast in forecasts:
            self.stability_engine.record_forecast(
                market.location or market.id, target_date, forecast
            )

        stability = self.stability_engine.get_stability_record(
            market.location or market.id,
            target_date,
            threshold=market.threshold,
        )

        # 6. Ensemble analysis
        ensemble = self.ensemble_analyzer.analyze(forecasts)

        # 7. Calculate model probability using the correct direction
        if market.threshold is None:
            logger.debug(f"Skipping {market.id} — no threshold defined")
            return

        direction = market.threshold_direction  # "exact", "above", "below"

        # Open-Meteo returns Celsius. Convert threshold to Celsius if needed.
        threshold_c = market.threshold
        if market.threshold_unit == "fahrenheit":
            threshold_c = (market.threshold - 32) * 5 / 9

        if direction == "above":
            model_prob = self.ensemble_analyzer.calculate_probability_above_threshold(
                ensemble, threshold_c
            )
        elif direction == "below":
            model_prob = self.ensemble_analyzer.calculate_probability_below_threshold(
                ensemble, threshold_c
            )
        else:
            # "exact" bracket: ±0.5°C window around threshold
            model_prob = self.ensemble_analyzer.calculate_probability_in_range(
                ensemble,
                low=threshold_c - 0.5,
                high=threshold_c + 0.5,
            )

        # 8. Market implied probability
        market_prob = market.yes_price

        # 9. Edge detection
        edge = self.edge_detector.calculate_edge(model_prob, market_prob)
        direction = self.edge_detector.get_direction(edge)

        # 10. Calibration score
        calibration_score = self.calibration_engine.get_calibration_score()

        # 11. Confidence scoring
        confidence = self.confidence_scorer.score(
            market=market,
            ensemble=ensemble,
            stability=stability,
            calibration_score=calibration_score,
            edge=edge,
        )

        self.trade_logger.log_market_analysis(market, ensemble, confidence, edge)

        # 12. Trade filter (NO_TRADE bias)
        should_trade, rejection_reasons = self.trade_filter.evaluate(
            market=market,
            ensemble=ensemble,
            stability=stability,
            confidence=confidence,
            edge=edge,
        )

        if not should_trade:
            self.trade_logger.log_trade_rejected(market, rejection_reasons)
            self._skipped_trades.append({
                "market_id": market.id,
                "title": market.title,
                "location": market.location,
                "model_probability": model_prob,
                "market_probability": market_prob,
                "edge": edge,
                "confidence": confidence.total,
                "rejection_reasons": rejection_reasons,
            })
            return

        # 13. Correlation check
        corr_score = self.correlation_detector.get_correlated_exposure(
            TradeSignal(
                market_id=market.id,
                action=direction,
                model_probability=model_prob,
                market_probability=market_prob,
                edge=edge,
                confidence_score=confidence,
                rejection_reasons=[],
            ),
            open_trades,
        )
        if self.correlation_detector.exceeds_correlation_limit(corr_score):
            reason = f"Portfolio correlation too high: {corr_score:.2f}"
            self.trade_logger.log_trade_rejected(market, [reason])
            self._skipped_trades.append({
                "market_id": market.id,
                "title": market.title,
                "location": market.location,
                "model_probability": model_prob,
                "market_probability": market_prob,
                "edge": edge,
                "confidence": confidence.total,
                "rejection_reasons": [reason],
            })
            return

        # 14. Kelly sizing
        entry_price = market.yes_price if direction == "BUY_YES" else market.no_price
        kelly_size = self.risk_manager.calculate_kelly_size(
            model_prob, entry_price, capital=self.paper_trader.capital
        )

        signal = TradeSignal(
            market_id=market.id,
            action=direction,
            model_probability=model_prob,
            market_probability=market_prob,
            edge=edge,
            confidence_score=confidence,
            rejection_reasons=[],
            kelly_size=kelly_size,
            recommended_size=kelly_size,
        )

        # 15. Final risk approval
        approved_size = self.risk_manager.get_approved_size(
            signal, self.paper_trader.capital, open_trades
        )
        if approved_size <= 0:
            reason = "Risk manager rejected trade (size=0)"
            self.trade_logger.log_trade_rejected(market, [reason])
            return

        signal.recommended_size = approved_size

        # 16. Execute paper trade
        trade = self.paper_trader.execute_trade(signal, market)
        if trade:
            open_trades.append(trade)
            self.trade_logger.log_trade_entered(trade)

            # Record for calibration
            calib_record = CalibrationRecord(
                market_id=market.id,
                predicted_probability=model_prob,
                market_probability=market_prob,
                edge=edge,
                confidence=confidence.total,
                resolution_date=market.resolution_date,
            )
            self.calibration_engine.record_trade(calib_record)
            self.trade_logger.log_calibration_update(calib_record)

    def resolve_expired_markets(self) -> None:
        """
        Check all open trades whose resolution_date has passed.
        Attempts NOAA METAR lookup for stations with known IDs.
        Marks others as needing manual resolution.
        """
        open_trades = self.paper_trader.get_open_trades()
        now = datetime.utcnow()

        for trade in open_trades:
            market = trade.market
            if not market.resolution_date:
                continue
            try:
                resolution_dt = datetime.fromisoformat(
                    market.resolution_date.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                continue

            if now < resolution_dt:
                continue

            logger.info(f"Market {market.id} is past resolution date — attempting auto-resolve")

            # Try METAR for weather station resolution
            resolved = False
            if market.resolution_station:
                metar = self.noaa.get_metar(market.resolution_station)
                if metar:
                    # Parse temperature from METAR
                    temp_c = metar.get("temp", None)
                    if temp_c is not None and market.threshold is not None:
                        outcome = float(temp_c) > market.threshold
                        resolved_trade = self.paper_trader.resolve_trade(market.id, outcome)
                        if resolved_trade:
                            self.trade_logger.log_trade_resolved(resolved_trade)
                            self.calibration_engine.update_outcome(
                                market.id, outcome, resolved_trade.pnl or 0.0
                            )
                            resolved = True

            if not resolved:
                logger.info(
                    f"Cannot auto-resolve {market.id} '{market.title[:50]}' — manual resolution required"
                )

    def export_report(self, output_dir: str = "exports/") -> str:
        """Generate a full Excel report and return the file path."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"weather_bot_report_{timestamp}.xlsx")

        all_trades = self.paper_trader.get_all_trades()
        calibration_records = self.calibration_engine._records

        path = self.exporter.export_full_report(
            trades=all_trades,
            skipped_trades=self._skipped_trades,
            calibration_records=calibration_records,
            output_path=output_path,
        )
        logger.info(f"Report exported: {path}")
        return path

    def print_status(self) -> None:
        """Print current portfolio status to console."""
        open_trades = self.paper_trader.get_open_trades()
        pnl = self.paper_trader.get_pnl_summary()
        portfolio_val = self.paper_trader.get_portfolio_value()

        print("\n" + "=" * 60)
        print("  ELITE WEATHER BOT — Portfolio Status")
        print("=" * 60)
        print(f"  Paper Capital:       ${self.paper_trader.capital:,.2f}")
        print(f"  Portfolio Value:     ${portfolio_val:,.2f}")
        print(f"  Open Positions:      {len(open_trades)}")
        print(f"  Total Trades (closed): {pnl['n_trades']}")
        print(f"  Win Rate:            {pnl['win_rate']:.1%}")
        print(f"  Total PnL:           ${pnl['total_pnl']:+,.2f}")
        print(f"  Avg Edge Captured:   {pnl['avg_edge_captured']:.4f}")
        print("=" * 60)

        if open_trades:
            print("\n  Open Positions:")
            for t in open_trades:
                print(
                    f"    [{t.market.id[:12]}] {t.signal.action} | "
                    f"fill={t.paper_fill_price:.4f} | "
                    f"size=${t.paper_size:.2f} | "
                    f"edge={t.signal.edge:+.4f}"
                )
        print()

    def _get_coords(self, market: MarketData) -> Optional[tuple[float, float]]:
        """Look up coordinates for the market location."""
        return get_city_coords(market.location)

    def _get_target_date(self, market: MarketData) -> Optional[str]:
        """Extract the target date string (YYYY-MM-DD) from the market."""
        if not market.resolution_date:
            return None
        try:
            dt = datetime.fromisoformat(market.resolution_date.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
