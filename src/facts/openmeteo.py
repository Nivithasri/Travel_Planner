"""Open-Meteo ERA5 archive (CC-BY 4.0, free tier non-commercial as resolved by the owner, D37).

One place per request, sequential. Not run until approved. Monthly figures are derived by us from
daily reanalysis values and are labelled as such (docs/facts-layer.md 2.2).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.collect.types import check_block
from src.common.http import FetchError, HttpClient
from src.common.schemas import PlaceRegistryEntry
from src.facts.envelope import envelope_path, make_envelope, place_dir, write_envelope
from src.facts.hosts import ApiHost, require_host

SOURCE = "open_meteo"
URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2024-12-31"
EXPECTED_DAYS = 3653  # 2015-01-01 to 2024-12-31 inclusive
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum"
# a day with at least this much precipitation is rainy (IMD convention, UNVERIFIED)
RAINY_DAY_MM = 2.5
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0); ERA5 reanalysis"


def request_params(lat: float, lon: float, elevation_m: float | None = None) -> dict:
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": START,
              "end_date": END, "daily": DAILY, "timezone": "Asia/Kolkata"}
    if elevation_m is not None:
        params["elevation"] = round(elevation_m, 1)
    return params


def validate_response(text: str) -> bool:
    try:
        doc = json.loads(text)
    except ValueError:
        return False
    daily = doc.get("daily") if isinstance(doc, dict) else None
    return (isinstance(daily, dict) and isinstance(daily.get("time"), list)
            and "error" not in doc)


def is_rainy(precip_mm: float | None) -> bool:
    return precip_mm is not None and precip_mm >= RAINY_DAY_MM


def monthly_normals(daily: dict) -> dict[int, dict]:
    """month (1-12) -> derived normals over the window: mean monthly rainfall (mm), mean rainy days,
    mean temperature (max+min)/2. Missing values are skipped, never filled."""
    days = daily["time"]
    tmax, tmin, rain = (daily.get(k) or [None] * len(days) for k in (
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum"))
    per_year: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"mm": 0.0, "rainy": 0, "n_rain": 0, "t": [], })
    for d, hi, lo, p in zip(days, tmax, tmin, rain, strict=False):
        year, month = int(d[:4]), int(d[5:7])
        cell = per_year[(year, month)]
        if p is not None:
            cell["mm"] += p
            cell["n_rain"] += 1
            cell["rainy"] += is_rainy(p)
        if hi is not None and lo is not None:
            cell["t"].append((hi + lo) / 2)
    out: dict[int, dict] = {}
    for month in range(1, 13):
        cells = [c for (y, m), c in per_year.items() if m == month and c["n_rain"]]
        temps = [t for (y, m), c in per_year.items() if m == month for t in c["t"]]
        if not cells:
            continue
        out[month] = {
            "rainfall_mm": sum(c["mm"] for c in cells) / len(cells),
            "rainy_days": sum(c["rainy"] for c in cells) / len(cells),
            "temp_c": (sum(temps) / len(temps)) if temps else None,
            "years": len(cells), "derived": True}
    return out


def fetch_place(http: HttpClient, host: ApiHost, entry: PlaceRegistryEntry, raw_root: Path,
                elevation_m: float | None = None, deadline: float | None = None,
                manifest: Path | None = None, filename: str = "archive_2015_2024.json") -> dict:
    params = request_params(entry.lat, entry.lon, elevation_m)
    require_host({host.host: host}, URL)
    try:
        res = http.fetch(URL, params, check_robots=not host.robots_exempt,
                         validate=validate_response, deadline=deadline)
    except FetchError as exc:
        check_block(exc, "open_meteo")
        return {"place": entry.name, "error": str(exc)[:200]}
    env = make_envelope(SOURCE, entry.name, URL, params, res, host.licence, ATTRIBUTION)
    path = envelope_path(raw_root, SOURCE, place_dir(entry.name), filename)
    wrote = write_envelope(path, env, manifest)
    daily = res.json()["daily"]
    days = len(daily["time"])
    nulls = sum(v is None for k in ("temperature_2m_max", "temperature_2m_min",
                                    "precipitation_sum") for v in daily.get(k, []))
    return {"place": entry.name, "status": res.status, "from_cache": res.from_cache,
            "wrote": wrote, "days": days,
            "nulls": nulls, "check_ok": days == EXPECTED_DAYS and nulls == 0,
            "bytes": len(res.text.encode("utf-8")), "elevation_param": elevation_m}
