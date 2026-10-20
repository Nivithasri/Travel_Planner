"""Nodes, the travel-time model, the calibration fit and the matrix (Phase 5; D76 to D83).

Offline and deterministic: the registry, the saved Overpass airports, the saved Wikidata entities
and config/routing_params.json are the only inputs.

Model (D77, D78 amended after the review, D82):
    road km  = great-circle km x road_factor
    plain h  = km / speed_short                           for km <= band_km
             = band_km / speed_short + (km - band_km) / speed_long   beyond (continuous)
    hours    = max(min_hours, plain h + ghat_surcharge_h x number of hill endpoints)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.schemas import REPO_ROOT
from src.facts.facts import extract_basic, haversine_km, place_dir

PARAMS_PATH = REPO_ROOT / "config" / "routing_params.json"
ROUTES_PATH = REPO_ROOT / "config" / "calibration_routes.json"
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "core_places.json"
WIKIDATA_RAW = REPO_ROOT / "data" / "raw" / "wikidata"
_METRE = "http://www.wikidata.org/entity/Q11573"
_HUB_SUFFIX = ("international airport", "airport", "airstrip", "air force station")
ALIAS_MAX_CHARS = 40
ALIAS_MAX_WORDS = 5


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wikidata_elevation_m(name: str, raw_root: Path = WIKIDATA_RAW) -> float | None:
    """Saved Wikidata elevation (P2044) in metres, None when absent or not in metres."""
    path = Path(raw_root) / place_dir(name) / "entity.json"
    if not path.exists():
        return None
    env = json.loads(path.read_text(encoding="utf-8"))
    entities = (env.get("response") or {}).get("entities") or {}
    if not entities:
        return None
    got = extract_basic(next(iter(entities.values()))).get("elevation")
    value = (got or {}).get("value")
    if not isinstance(value, dict) or value.get("unit") != _METRE:
        return None
    return float(value["amount"])


def hill_flag(name: str, elevation_m: float | None, params: dict) -> tuple[bool, str]:
    """D78: elevation >= hill_elevation_m, or an explicit override. Returns (flag, reason)."""
    if name in params["hill_overrides"]:
        shown = "n/a" if elevation_m is None else f"{elevation_m:g} m"
        return True, f"override (elevation {shown})"
    if elevation_m is not None and elevation_m >= params["hill_elevation_m"]:
        return True, f"elevation {elevation_m:g} m >= {params['hill_elevation_m']} m"
    return False, "no elevation" if elevation_m is None else f"elevation {elevation_m:g} m"


# ==== names and aliases ===========================================================

def clean_name(name: str) -> str:
    """Collapse whitespace and tighten ' ,' to ','."""
    return re.sub(r"\s+,", ",", " ".join(str(name).split()))


def _strip_suffix(text: str) -> str:
    out = text.strip()
    for suffix in _HUB_SUFFIX:
        if out.lower().endswith(" " + suffix):
            return out[: -len(suffix)].strip()
    return out


def hub_alias(name: str) -> str:
    """The part of an airport name before the first comma, without the airport words (B4)."""
    return _strip_suffix(clean_name(name).partition(",")[0])


def hub_aliases(name: str, iata: str) -> list[str]:
    """IATA code, the stripped head of the name, and the part after a comma (a town name)."""
    head, _, tail = clean_name(name).partition(",")
    out = {iata}
    for part in (head, tail):
        part = _strip_suffix(part)
        if part:
            out.add(part)
    return sorted(out)


def clean_aliases(aliases: list[str]) -> list[str]:
    """Drop junk aliases (sentences copied from Wikidata): over ALIAS_MAX_CHARS characters or
    ALIAS_MAX_WORDS words. The frozen registry is not changed (B4)."""
    keep = {clean_name(a) for a in aliases}
    return sorted(a for a in keep if a and len(a) <= ALIAS_MAX_CHARS
                  and len(a.split()) <= ALIAS_MAX_WORDS)


# ==== nodes =======================================================================

def select_hubs(airports: dict[str, dict], destinations: list[tuple[float, float]],
                holdouts: list[tuple[float, float]], min_km: float) -> list[str]:
    """D79: civil allowlist airports more than `min_km` from every destination and hold-out."""
    points = [*destinations, *holdouts]
    out = []
    for iata, a in sorted(airports.items()):
        if a.get("military"):
            continue
        if all(haversine_km(a["lat"], a["lon"], lat, lon) > min_km for lat, lon in points):
            out.append(iata)
    return out


def build_nodes(registry, airports: dict[str, dict], params: dict,
                raw_root: Path = WIKIDATA_RAW) -> tuple[list[dict], dict]:
    """Nodes (destinations, then hubs) and a report. Hold-outs are never nodes (D17)."""
    dest = [e for e in registry.entries if not e.holdout]
    held = [e for e in registry.entries if e.holdout]
    nodes, flagged = [], []
    for e in dest:
        elev = wikidata_elevation_m(e.name, raw_root)
        hill, why = hill_flag(e.name, elev, params)
        if hill:
            flagged.append((e.name, why))
        nodes.append({"id": place_dir(e.name), "name": e.name, "kind": "destination",
                      "state": e.state, "lat": e.lat, "lon": e.lon, "hill": hill,
                      "aliases": clean_aliases(e.aliases)})
    hubs = select_hubs(airports, [(e.lat, e.lon) for e in dest], [(e.lat, e.lon) for e in held],
                       params["hub_min_km"])
    for iata in hubs:
        a = airports[iata]
        nodes.append({"id": f"hub-{iata.lower()}", "name": clean_name(a["name"]),
                      "kind": "hub", "state": None, "lat": a["lat"], "lon": a["lon"],
                      "hill": False, "iata": iata, "aliases": hub_aliases(a["name"], iata)})
    report = {"destinations": len(dest), "hubs": hubs, "hill_nodes": flagged,
              "excluded_military": sorted(k for k, v in airports.items() if v.get("military"))}
    return nodes, report


def airport_aliases(nodes: list[dict], airports: dict[str, dict], max_km: float) -> dict[str, str]:
    """B3, D83: civil allowlist airports that are not hub nodes -> the nearest destination node id
    when it is within `max_km`. By the hub rule (D79) every such airport is within 25 km of one,
    so nearest_airport codes such as MAA or COK resolve in travel_time. The airport stands for
    that destination (an error of at most `max_km`)."""
    dest = [n for n in nodes if n["kind"] == "destination"]
    hubs = {n["iata"] for n in nodes if n["kind"] == "hub"}
    out = {}
    for iata, a in sorted(airports.items()):
        if a.get("military") or iata in hubs:
            continue
        dist, node_id = min((haversine_km(a["lat"], a["lon"], n["lat"], n["lon"]), n["id"])
                            for n in dest)
        if dist <= max_km:
            out[iata] = node_id
    return out


# ==== travel-time model ===========================================================

def hill_endpoints(a: dict, b: dict) -> int:
    return int(bool(a["hill"])) + int(bool(b["hill"]))


def plain_hours(km: float, fitted: dict) -> float:
    """Two-band plain driving time, continuous at band_km (D82)."""
    band = fitted["band_km"]
    if km <= band:
        return km / fitted["speed_short_kmh"]
    return band / fitted["speed_short_kmh"] + (km - band) / fitted["speed_long_kmh"]


def estimate(lat1: float, lon1: float, lat2: float, lon2: float, n_hill: int,
             fitted: dict, min_hours: float = 0.25) -> tuple[float, float]:
    """(road km, hours): great-circle km x factor, two-band plain time, plus the ghat surcharge
    once per hill endpoint (D78, D82)."""
    km = haversine_km(lat1, lon1, lat2, lon2) * fitted["road_factor"]
    hours = plain_hours(km, fitted) + fitted["ghat_surcharge_h"] * n_hill
    return km, max(min_hours, hours)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def route_rows(routes: list[dict], nodes: dict[str, dict]) -> list[dict]:
    rows = []
    for r in routes:
        a, b = nodes[r["a"]], nodes[r["b"]]
        gc = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        n_hill = hill_endpoints(a, b)
        rows.append({**r, "gc_km": gc, "n_hill": n_hill, "hill": n_hill > 0})
    return rows


def fit(rows: list[dict], band_km: float = 150.0) -> dict:
    """Medians over the 'fit' routes (D77, D82, D78 amended). Plain routes (no hill endpoint):
    road_factor = ref_km / great-circle km; speed_short = ref_km / ref_h for routes up to band_km;
    speed_long = marginal speed beyond the band, (ref_km - band) / (ref_h - band / speed_short).
    Hill routes: ghat_surcharge_h = median over routes of (ref_h - plain model hours) per hill
    endpoint, never below 0. Robust to a few wrong reference values."""
    plain = [r for r in rows if r["split"] == "fit" and r["n_hill"] == 0]
    hill = [r for r in rows if r["split"] == "fit" and r["n_hill"] > 0]
    short = [r for r in plain if r["ref_km"] <= band_km]
    long_ = [r for r in plain if r["ref_km"] > band_km]
    for name, group in (("short plain", short), ("long plain", long_), ("hill", hill)):
        if not group:
            raise ValueError(f"no fit routes for {name} edges")
    out = {"band_km": band_km,
           "road_factor": round(_median([r["ref_km"] / r["gc_km"] for r in plain]), 4)}
    out["speed_short_kmh"] = round(_median([r["ref_km"] / r["ref_h"] for r in short]), 2)
    marginal = []
    for r in long_:
        extra_h = r["ref_h"] - band_km / out["speed_short_kmh"]
        if extra_h > 0:
            marginal.append((r["ref_km"] - band_km) / extra_h)
    if not marginal:
        raise ValueError("no usable long plain routes")
    out["speed_long_kmh"] = round(_median(marginal), 2)
    base = {**out, "ghat_surcharge_h": 0.0}
    resid = [(r["ref_h"] - plain_hours(r["gc_km"] * out["road_factor"], base)) / r["n_hill"]
             for r in hill]
    out["ghat_surcharge_h"] = round(max(0.0, _median(resid)), 3)
    out.update({"n_fit_short": len(short), "n_fit_long": len(long_), "n_fit_hill": len(hill)})
    return out


def route_errors(rows: list[dict], fitted: dict, min_hours: float = 0.25) -> list[dict]:
    out = []
    for r in rows:
        km = r["gc_km"] * fitted["road_factor"]
        h = max(min_hours, plain_hours(km, fitted) + fitted["ghat_surcharge_h"] * r["n_hill"])
        out.append({"a": r["a"], "b": r["b"], "split": r["split"], "hill": r["hill"],
                    "ref_km": r["ref_km"], "km": round(km, 1), "ref_h": r["ref_h"],
                    "hours": round(h, 2),
                    "km_err_pct": round(100 * (km - r["ref_km"]) / r["ref_km"], 1),
                    "h_err_pct": round(100 * (h - r["ref_h"]) / r["ref_h"], 1)})
    return out


def error_summary(errors: list[dict]) -> dict:
    if not errors:
        return {"n": 0}
    n = len(errors)
    return {"n": n,
            "mae_h": round(sum(abs(e["hours"] - e["ref_h"]) for e in errors) / n, 2),
            "mape_h_pct": round(sum(abs(e["h_err_pct"]) for e in errors) / n, 1),
            "mape_km_pct": round(sum(abs(e["km_err_pct"]) for e in errors) / n, 1)}


def flag_large_errors(errors: list[dict], split: str = "validate",
                      limit_pct: float = 25.0) -> list[str]:
    """Routes of `split` whose hours are off by more than `limit_pct` (B2)."""
    return [f"{e['a']}-{e['b']} {e['h_err_pct']:+.1f}%" for e in errors
            if e["split"] == split and abs(e["h_err_pct"]) > limit_pct]


def leave_one_out(rows: list[dict], band_km: float = 150.0) -> dict:
    """Refit without each fit route, predict it, summarise the errors (fit routes only).
    A route whose removal empties a group is skipped."""
    errs = []
    for i, r in enumerate(rows):
        if r["split"] != "fit":
            continue
        try:
            f = fit([x for j, x in enumerate(rows) if j != i], band_km)
        except ValueError:
            continue
        errs.extend(route_errors([r], f))
    return error_summary(errs)


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def matrix_pairs(nodes: list[dict], fitted: dict, min_hours: float = 0.25) -> tuple[dict, dict]:
    """hours and km for every unordered pair of distinct nodes, keyed by pair_key."""
    hours, km = {}, {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            k_, h_ = estimate(a["lat"], a["lon"], b["lat"], b["lon"], hill_endpoints(a, b),
                              fitted, min_hours)
            key = pair_key(a["id"], b["id"])
            hours[key], km[key] = round(h_, 3), round(k_, 1)
    return dict(sorted(hours.items())), dict(sorted(km.items()))


def triangle_violations(nodes: list[dict], hours: dict[str, float], tol: float = 1e-9) -> int:
    """Count node pairs (a, c) for which some b gives a->b->c shorter than the direct a->c."""
    ids = [n["id"] for n in nodes]

    def t(x, y):
        return hours[pair_key(x, y)]

    bad = 0
    for i, a in enumerate(ids):
        for c in ids[i + 1:]:
            direct = t(a, c)
            if any(t(a, b) + t(b, c) < direct - tol for b in ids if b not in (a, c)):
                bad += 1
    return bad
