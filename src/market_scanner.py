from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from loguru import logger

from .models import MarketData

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# Extracts temperature like "23°C", "32°F", "23 degrees"
TEMP_FROM_QUESTION_RE = re.compile(
    r"be\s+([\d]+(?:\.\d+)?)\s*°?\s*([CF]|degrees?)(?:\s+or\s+(below|above|higher|lower|above))?",
    re.IGNORECASE,
)

# Precipitation / other thresholds
THRESHOLD_PATTERN = re.compile(
    r"(above|below|over|under|exceed|reach|hit)\s*([\-\d\.]+)\s*[°]?\s*([CF]|celsius|fahrenheit|degrees|mm|inches|mph|km/h)?",
    re.IGNORECASE,
)

# Non-weather events to exclude even if tagged weather
EXCLUDE_PATTERNS = re.compile(
    r"(earthquake|quake|volcano|eruption|meteor|pandemic|measles|flu|hantavirus|ebola|"
    r"sea ice|arctic|space weather|megaquake|natural disaster|tornado count|"
    r"named storm count|hurricane count)",
    re.IGNORECASE,
)

# City → (lat, lon) — comprehensive list matching Polymarket weather cities
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "New York": (40.7128, -74.0060),
    "NYC": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Miami": (25.7617, -80.1918),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Dallas": (32.7767, -96.7970),
    "Austin": (30.2672, -97.7431),
    "Atlanta": (33.7490, -84.3880),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "San Francisco": (37.7749, -122.4194),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Madrid": (40.4168, -3.7038),
    "Milan": (45.4642, 9.1900),
    "Amsterdam": (52.3676, 4.9041),
    "Warsaw": (52.2297, 21.0122),
    "Vienna": (48.2082, 16.3738),
    "Prague": (50.0755, 14.4378),
    "Helsinki": (60.1699, 24.9384),
    "Stockholm": (59.3293, 18.0686),
    "Oslo": (59.9139, 10.7522),
    "Copenhagen": (55.6761, 12.5683),
    "Munich": (48.1351, 11.5820),
    "Istanbul": (41.0082, 28.9784),
    "Ankara": (39.9334, 32.8597),
    "Moscow": (55.7558, 37.6173),
    "Tokyo": (35.6762, 139.6503),
    "Seoul": (37.5665, 126.9780),
    "Busan": (35.1796, 129.0756),
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Shenzhen": (22.5431, 114.0579),
    "Guangzhou": (23.1291, 113.2644),
    "Chengdu": (30.5728, 104.0668),
    "Chongqing": (29.4316, 106.9123),
    "Wuhan": (30.5928, 114.3055),
    "Qingdao": (36.0671, 120.3826),
    "Hong Kong": (22.3193, 114.1694),
    "Taipei": (25.0330, 121.5654),
    "Singapore": (1.3521, 103.8198),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Manila": (14.5995, 120.9842),
    "Bangkok": (13.7563, 100.5018),
    "Mumbai": (19.0760, 72.8777),
    "Lucknow": (26.8467, 80.9462),
    "Karachi": (24.8607, 67.0011),
    "Jeddah": (21.4858, 39.1925),
    "Dubai": (25.2048, 55.2708),
    "Cairo": (30.0444, 31.2357),
    "Nairobi": (-1.2921, 36.8219),
    "Lagos": (6.5244, 3.3792),
    "Cape Town": (-33.9249, 18.4241),
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Wellington": (-41.2866, 174.7756),
    "Toronto": (43.6532, -79.3832),
    "Mexico City": (19.4326, -99.1332),
    "Sao Paulo": (-23.5505, -46.6333),
    "São Paulo": (-23.5505, -46.6333),
    "Buenos Aires": (-34.6037, -58.3816),
    "Tel Aviv": (32.0853, 34.7818),
    "Panama City": (8.9936, -79.5197),
}


def get_city_coords(location: Optional[str]) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for a city name, case-insensitive."""
    if not location:
        return None
    for city, coords in CITY_COORDINATES.items():
        if city.lower() == location.lower() or city.lower() in location.lower():
            return coords
    return None


class MarketScanner:
    def __init__(self, base_url: str = POLYMARKET_GAMMA_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WeatherPredictionBot/1.0"})

    def scan_weather_markets(self, limit: int = 500) -> list[MarketData]:
        """Scan the Polymarket weather tab for active meteorological markets."""
        events: list[dict] = []
        offset = 0
        now = datetime.now(timezone.utc).isoformat()[:10]

        while True:
            try:
                resp = self.session.get(
                    f"{self.base_url}/events",
                    params={"limit": 100, "tag_slug": "weather", "closed": "false", "offset": offset},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                batch = resp.json()
            except Exception as exc:
                logger.warning(f"Error fetching weather events at offset {offset}: {exc}")
                break

            if not batch:
                break

            for event in batch:
                end_date = (event.get("endDate") or "")[:10]
                title = event.get("title", "")
                if end_date > now and not EXCLUDE_PATTERNS.search(title):
                    events.append(event)

            if len(batch) < 100:
                break
            offset += 100
            time.sleep(0.25)

        markets: list[MarketData] = []
        for event in events:
            markets.extend(self._parse_event(event))

        logger.info(f"Found {len(markets)} weather markets from Polymarket weather tab")
        return markets

    def _parse_event(self, event: dict) -> list[MarketData]:
        """
        Parse a Polymarket weather event into one MarketData per sub-market outcome.
        Each sub-market is a temperature bracket like "Will it be exactly 25°C?" or "25°C or higher?"
        """
        title = event.get("title", "")
        location = self._extract_location(title)
        event_type = self._classify_event_type(title)
        resolution_date = event.get("endDate") or None
        resolution_source = event.get("resolutionSource") or None
        description = event.get("description") or ""
        resolution_station = self._extract_station(description)
        event_volume = float(event.get("volume") or 0)

        sub_markets = event.get("markets") or []
        if not sub_markets:
            return []

        results: list[MarketData] = []
        for m in sub_markets:
            try:
                market_id = str(m.get("id") or m.get("conditionId") or "")
                if not market_id:
                    continue

                # The question contains the actual temperature bracket
                question = m.get("question") or m.get("groupItemTitle") or title

                # Extract the real numeric temperature threshold from the question text
                threshold, threshold_unit, direction = self._extract_temp_from_question(question)
                if threshold is None:
                    # Fallback: try generic threshold pattern
                    threshold, threshold_unit = self._extract_threshold_generic(question)
                    direction = "above"

                yes_price, no_price = self._parse_prices(m)
                spread = abs(yes_price + no_price - 1.0)
                liquidity = float(m.get("liquidityNum") or m.get("liquidity") or 0)
                volume = float(m.get("volumeNum") or m.get("volume") or event_volume)

                results.append(MarketData(
                    id=market_id,
                    title=question,
                    location=location,
                    event_type=event_type,
                    threshold=threshold,
                    threshold_unit=threshold_unit,
                    threshold_direction=direction,
                    resolution_date=resolution_date,
                    yes_price=yes_price,
                    no_price=no_price,
                    spread=spread,
                    liquidity=liquidity,
                    volume=volume,
                    resolution_source=resolution_source,
                    resolution_station=resolution_station,
                ))
            except Exception as exc:
                logger.debug(f"Skipping sub-market in '{title}': {exc}")

        return results

    def _extract_temp_from_question(self, question: str) -> tuple[Optional[float], str, str]:
        """
        Parse temperature and direction from a question like:
          'Will the highest temperature in London be 25°C on May 23?'
          'Will the lowest temperature in London be 20°C or higher on May 24?'
          'Will the highest temperature in London be 22°C or below on May 23?'
        Returns (temperature_value, unit, direction) where direction ∈ {above, below, exact}
        """
        # Try "be X°C or below/above/higher/lower"
        match = re.search(
            r"be\s+([\d]+(?:\.\d+)?)\s*°?\s*([CF])\s*(?:or\s+(below|above|higher|lower))?",
            question,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            raw_unit = match.group(2).upper()
            unit = "celsius" if raw_unit == "C" else "fahrenheit"
            qualifier = (match.group(3) or "").lower()
            if qualifier in ("below", "lower"):
                direction = "below"
            elif qualifier in ("above", "higher"):
                direction = "above"
            else:
                direction = "exact"
            return value, unit, direction

        # Try without °C suffix — e.g. "be 25 degrees"
        match2 = re.search(
            r"be\s+([\d]+(?:\.\d+)?)\s+degrees?(?:\s+celsius)?(?:\s+or\s+(below|above|higher|lower))?",
            question,
            re.IGNORECASE,
        )
        if match2:
            value = float(match2.group(1))
            qualifier = (match2.group(2) or "").lower()
            direction = "below" if qualifier in ("below", "lower") else "above" if qualifier in ("above", "higher") else "exact"
            return value, "celsius", direction

        return None, "celsius", "exact"

    def _extract_threshold_generic(self, text: str) -> tuple[Optional[float], Optional[str]]:
        match = THRESHOLD_PATTERN.search(text)
        if match:
            try:
                value = float(match.group(2))
                unit_raw = (match.group(3) or "").strip().upper()
                unit_map = {"C": "celsius", "F": "fahrenheit", "CELSIUS": "celsius",
                            "FAHRENHEIT": "fahrenheit", "MM": "mm", "INCHES": "inches"}
                return value, unit_map.get(unit_raw, "celsius")
            except ValueError:
                pass
        return None, None

    def _parse_prices(self, m: dict) -> tuple[float, float]:
        yes_price, no_price = 0.5, 0.5
        outcomes = m.get("outcomes")
        if isinstance(outcomes, list):
            for o in outcomes:
                name = str(o.get("name", "")).lower()
                price = float(o.get("price", 0.5))
                if name in ("yes", "true"):
                    yes_price = price
                elif name in ("no", "false"):
                    no_price = price
            if yes_price != 0.5 or no_price != 0.5:
                return yes_price, no_price

        prices_raw = m.get("outcomePrices")
        if isinstance(prices_raw, str):
            try:
                prices_list = json.loads(prices_raw)
                if len(prices_list) >= 2:
                    return float(prices_list[0]), float(prices_list[1])
            except Exception:
                pass
        elif isinstance(prices_raw, list) and len(prices_raw) >= 2:
            return float(prices_raw[0]), float(prices_raw[1])

        return yes_price, no_price

    def _extract_location(self, title: str) -> Optional[str]:
        for city in CITY_COORDINATES:
            if city.lower() in title.lower():
                return city
        match = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", title)
        if match:
            return match.group(1)
        return None

    def _classify_event_type(self, title: str) -> str:
        t = title.lower()
        if any(k in t for k in ("temperature", "highest temp", "lowest temp", "°c", "°f", "celsius", "fahrenheit")):
            return "temperature"
        if any(k in t for k in ("rain", "precipitation", "rainfall", "flood")):
            return "precipitation"
        if any(k in t for k in ("snow", "blizzard", "snowfall")):
            return "snow"
        if any(k in t for k in ("wind", "hurricane", "tornado", "storm", "cyclone", "named storm")):
            return "wind"
        if any(k in t for k in ("wildfire", "fire")):
            return "fire"
        return "weather"

    def _extract_station(self, description: str) -> Optional[str]:
        # ICAO station IDs are 4 uppercase letters
        match = re.search(r"\b([A-Z]{4})\b", description)
        if match:
            return match.group(1)
        return None
