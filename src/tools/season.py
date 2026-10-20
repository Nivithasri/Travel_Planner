"""Season data builder (Phase 5; D80): ERA5 monthly normals, verdict rules, evidence from chunks.

Offline. Inputs: the saved Open-Meteo files (the no-elevation file is primary, D42), the enriched
chunks and provenance.jsonl, config/routing_params.json. Output rows hold chunk ids, never text.
"""
from __future__ import annotations

import calendar
import json
from collections import defaultdict
from pathlib import Path

from src.common.schemas import REPO_ROOT
from src.facts.facts import place_dir

OPEN_METEO_RAW = REPO_ROOT / "data" / "raw" / "open_meteo"
ENRICHED_PATH = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
PROVENANCE_PATH = REPO_ROOT / "data" / "enriched" / "provenance.jsonl"
SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer",
                   6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
                   10: "post_monsoon", 11: "post_monsoon"}
PROVENANCE_CLASS = {"llm": "model_written", "fallback": "fallback", "rule_only": "rule_only"}
MAX_WARNINGS = 5


def climate_file(place: str, raw_root: Path = OPEN_METEO_RAW) -> Path | None:
    """The no-elevation file when it exists (primary, D42), else the plain archive file."""
    d = Path(raw_root) / place_dir(place)
    for name in ("archive_2015_2024_noelev.json", "archive_2015_2024.json"):
        if (d / name).exists():
            return d / name
    return None


def monthly_normals(daily: dict, rainy_day_mm: float) -> dict[int, dict]:
    """Calendar-month normals from daily ERA5 values. Rain and rainy days are averaged over the
    COMPLETE years of that month only (every calendar day has a precipitation value, D83); a
    rainy day has precipitation_sum >= rainy_day_mm. Tmax and Tmin average the available days."""
    tmax, tmin = defaultdict(list), defaultdict(list)
    rain, rainy, days = defaultdict(float), defaultdict(int), defaultdict(int)
    for day, hi, lo, pr in zip(daily["time"], daily["temperature_2m_max"],
                               daily["temperature_2m_min"], daily["precipitation_sum"],
                               strict=True):
        year, month = int(day[:4]), int(day[5:7])
        if hi is not None:
            tmax[month].append(hi)
        if lo is not None:
            tmin[month].append(lo)
        if pr is not None:
            rain[(year, month)] += pr
            days[(year, month)] += 1
            rainy[(year, month)] += 1 if pr >= rainy_day_mm else 0
    out = {}
    for m in range(1, 13):
        years = sorted(y for (y, mm), n in days.items()
                       if mm == m and n == calendar.monthrange(y, mm)[1])
        if not years or not tmax[m] or not tmin[m]:
            continue
        out[m] = {"tmax_c": round(sum(tmax[m]) / len(tmax[m]), 1),
                  "tmin_c": round(sum(tmin[m]) / len(tmin[m]), 1),
                  "rain_mm": round(sum(rain[(y, m)] for y in years) / len(years), 1),
                  "rainy_days": round(sum(rainy[(y, m)] for y in years) / len(years), 1),
                  "years": len(years)}
    return out


def verdict(clim: dict, hill: bool, th: dict) -> tuple[str, list[str], bool]:
    """(good | fair | poor, reasons, access_risk). Starting thresholds, D80."""
    verdict_, reasons = "good", []
    if clim["rain_mm"] >= th["poor_rain_mm"]:
        verdict_, reasons = "poor", [f"rain {clim['rain_mm']:g} mm >= {th['poor_rain_mm']:g} mm"]
    elif clim["rain_mm"] >= th["fair_rain_mm"]:
        verdict_, reasons = "fair", [f"rain {clim['rain_mm']:g} mm >= {th['fair_rain_mm']:g} mm"]
    if clim["tmax_c"] >= th["poor_tmax_c"]:
        verdict_ = "poor"
        reasons.append(f"Tmax {clim['tmax_c']:g} C >= {th['poor_tmax_c']:g} C")
    elif clim["tmax_c"] >= th["fair_tmax_c"]:
        if verdict_ == "good":
            verdict_ = "fair"
        reasons.append(f"Tmax {clim['tmax_c']:g} C >= {th['fair_tmax_c']:g} C")
    if not reasons:
        reasons = ["rain and heat below the thresholds"]
    return verdict_, reasons, bool(hill and clim["rain_mm"] >= th["poor_rain_mm"])


def load_evidence(chunks_path: Path = ENRICHED_PATH, prov_path: Path = PROVENANCE_PATH
                  ) -> dict[str, list[dict]]:
    """place -> [{chunk_id, source (model_written | fallback | rule_only), tags, is_warning}].
    Only ids and labels are kept, not text."""
    prov = {}
    for line in Path(prov_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prov[row["chunk_id"]] = PROVENANCE_CLASS[row["source"]]
    out: dict[str, list[dict]] = defaultdict(list)
    for line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c["chunk_id"] not in prov:
            raise ValueError(f"no provenance row for chunk {c['chunk_id']}")
        out[c["place"]].append({"chunk_id": c["chunk_id"], "source": prov[c["chunk_id"]],
                                "tags": list(c["season_tags"]),
                                "is_warning": bool(c["is_warning"])})
    return out


def month_evidence(rows: list[dict], month: int) -> tuple[dict, list[str], dict]:
    """Counts split by provenance for the month's season, warning ids and their provenance. Only
    MODEL-WRITTEN warnings are listed (at most MAX_WARNINGS, by chunk_id); rule-based is_warning
    values are not trusted as signals (D69, D83). A chunk is season-relevant when it carries the
    season tag or all_year."""
    season = SEASON_OF_MONTH[month]
    counts = {k: {"chunks": 0, "season_tagged": 0, "all_year": 0}
              for k in ("model_written", "fallback", "rule_only")}
    warn = []
    for r in rows:
        c = counts[r["source"]]
        c["chunks"] += 1
        c["season_tagged"] += season in r["tags"]
        c["all_year"] += "all_year" in r["tags"]
        if (r["is_warning"] and r["source"] == "model_written"
                and (season in r["tags"] or "all_year" in r["tags"])):
            warn.append(r)
    warn.sort(key=lambda r: r["chunk_id"])
    warn = warn[:MAX_WARNINGS]
    return counts, [w["chunk_id"] for w in warn], {w["chunk_id"]: w["source"] for w in warn}


def build_season(places: list[dict], params: dict, evidence: dict[str, list[dict]],
                 raw_root: Path = OPEN_METEO_RAW, airports: dict | None = None) -> dict:
    """places: destination node dicts (name, lat, lon, hill). Returns the season_data places map."""
    from src.tools.routing import nearest_airport
    out = {}
    for node in places:
        name = node["name"]
        path = climate_file(name, raw_root)
        if path is None:
            continue
        env = json.loads(path.read_text(encoding="utf-8"))
        normals = monthly_normals(env["response"]["daily"], params["rainy_day_mm"])
        months = {}
        for m, clim in normals.items():
            v, reasons, risk = verdict(clim, node["hill"], params["season_thresholds"])
            counts, wids, wsrc = month_evidence(evidence.get(name, []), m)
            months[str(m)] = {"climate": clim, "verdict": v, "reasons": reasons,
                              "access_risk": risk, "evidence": counts,
                              "warning_chunk_ids": wids, "warning_sources": wsrc}
        rows = evidence.get(name, [])
        warning_counts = {k: sum(r["is_warning"] and r["source"] == k for r in rows)
                          for k in ("model_written", "fallback", "rule_only")}
        entry = {"confidence": "low" if name in params["low_confidence_places"] else "medium",
                 "climate_file": path.name, "hill": node["hill"], "months": months,
                 "warning_counts": warning_counts}
        if airports:
            entry["nearest_airport"] = nearest_airport(node["lat"], node["lon"], airports)
        out[name] = entry
    return out
