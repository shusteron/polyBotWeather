from __future__ import annotations
from typing import Optional

# City → ICAO station ID that Polymarket uses for resolution
# Source: Polymarket market descriptions ("London City Airport", "JFK", etc.)
# These map to METAR stations on aviationweather.gov
CITY_TO_ICAO: dict[str, str] = {
    "London":        "EGLC",   # London City Airport (explicitly named in Polymarket desc)
    "Paris":         "LFPG",   # Charles de Gaulle
    "New York":      "KJFK",   # JFK
    "NYC":           "KJFK",
    "Los Angeles":   "KLAX",
    "Miami":         "KMIA",
    "Chicago":       "KORD",
    "Houston":       "KIAH",
    "Dallas":        "KDFW",
    "Austin":        "KAUS",
    "Atlanta":       "KATL",
    "Seattle":       "KSEA",
    "Denver":        "KDEN",
    "San Francisco": "KSFO",
    "Tokyo":         "RJTT",   # Haneda
    "Seoul":         "RKSI",   # Incheon
    "Busan":         "RKPK",
    "Beijing":       "ZBAA",
    "Shanghai":      "ZSPD",
    "Shenzhen":      "ZGSZ",
    "Guangzhou":     "ZGGG",
    "Chengdu":       "ZUUU",
    "Chongqing":     "ZUCK",
    "Wuhan":         "ZHHH",
    "Qingdao":       "ZSQD",
    "Hong Kong":     "VHHH",
    "Taipei":        "RCTP",
    "Singapore":     "WSSS",
    "Kuala Lumpur":  "WMKK",
    "Manila":        "RPLL",
    "Bangkok":       "VTBS",
    "Mumbai":        "VABB",
    "Lucknow":       "VILK",
    "Karachi":       "OPKC",
    "Jeddah":        "OEJN",
    "Dubai":         "OMDB",
    "Cairo":         "HECA",
    "Nairobi":       "HKJK",
    "Lagos":         "DNMM",
    "Cape Town":     "FACT",
    "Sydney":        "YSSY",
    "Melbourne":     "YMML",
    "Wellington":    "NZWN",
    "Toronto":       "CYYZ",
    "Mexico City":   "MMMX",
    "Sao Paulo":     "SBGR",
    "São Paulo":     "SBGR",
    "Buenos Aires":  "SAEZ",
    "Tel Aviv":      "LLBG",   # Ben Gurion
    "Istanbul":      "LTFM",   # Istanbul Airport
    "Ankara":        "LTAC",
    "Moscow":        "UUEE",   # Sheremetyevo
    "Berlin":        "EDDB",
    "Munich":        "EDDM",
    "Amsterdam":     "EHAM",
    "Madrid":        "LEMD",
    "Milan":         "LIMC",
    "Warsaw":        "EPWA",
    "Vienna":        "LOWW",
    "Helsinki":      "EFHK",
    "Stockholm":     "ESSA",
    "Oslo":          "ENGM",
    "Copenhagen":    "EKCH",
    "Prague":        "LKPR",
    "Panama City":   "MPTO",
}

# Known GFS bias corrections per station (°C) — positive means GFS runs warm
# Populated from paper trading results over time; starts empty
# Format: { "EGLC": {"max": +0.5, "min": -0.3}, ... }
GFS_BIAS: dict[str, dict[str, float]] = {}


def get_icao(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    for k, v in CITY_TO_ICAO.items():
        if k.lower() == city.lower() or k.lower() in city.lower():
            return v
    return None


def apply_bias_correction(value: float, station: str, measurement: str) -> float:
    """Apply known GFS bias for a station/measurement type if available."""
    bias_entry = GFS_BIAS.get(station, {})
    bias = bias_entry.get(measurement, 0.0)
    return value - bias  # subtract bias to get corrected value
