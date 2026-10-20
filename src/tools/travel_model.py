"""Nodes, the travel-time model, the calibration fit and the matrix (Phase 5; D76 to D79).

Offline and deterministic: the registry, the saved Overpass airports, the saved Wikidata entities
and config/routing_params.json are the only inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.schemas import REPO_ROOT
from src.facts.facts import extract_basic, haversine_km, place_dir

PARAMS_PATH = REPO_ROOT / "config" / "routing_params.json"
ROUTES_PATH = REPO_ROOT / "config" / "calibration_routes.json"
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "core_places.json"
WIKIDATA_RAW = REPO_ROOT / "data" / "raw" / "wikidata"
_METRE = "http://www.wikidata.org/entity/Q11573"
_HUB_SUFFIX = ("international airport", "airport", "airstrip", "air force station")


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


def hub_alias(name: str) -> str:
    out = name.strip()
    for suffix in _HUB_SUFFIX:
        if out.lower().endswith(" " + suffix):
            return out[: -len(suffix)].strip()
    return out


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
                      "aliases": sorted(e.aliases)})
    hubs = select_hubs(airports, [(e.lat, e.lon) for e in dest], [(e.lat, e.lon) for e in held],
                       params["hub_min_km"])
    for iata in hubs:
        a = airports[iata]
        nodes.append({"id": f"hub-{iata.lower()}", "name": a["name"], "kind": "hub",
                      "state": None, "lat": a["lat"], "lon": a["lon"], "hill": False,
                      "iata": iata, "aliases": sorted({hub_alias(a["name"]), iata})})
    report = {"destinations": len(dest), "hubs": hubs, "hill_nodes": flagged,
              "excluded_military": sorted(k for k, v in airports.items() if v.get("military"))}
    return nodes, report


# ==== travel-time model ===========================================================

def edge_hill(a: dict, b: dict) -> bool:
    return bool(a["hill"] or b["hill"])


def estimate(lat1: float, lon1: float, lat2: float, lon2: float, hill: bool,
             fitted: dict, min_hours: float = 0.25) -> tuple[float, float]:
    """(road km, hours) = great-circle km x factor, then / speed (D77, D78)."""
    prefix = "hill_" if hill else ""
    km = haversine_km(lat1, lon1, lat2, lon2) * fitted[prefix + "road_factor"]
    return km, max(min_hours, km / fitted[prefix + "speed_kmh"])


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def route_rows(routes: list[dict], nodes: dict[str, dict]) -> list[dict]:
    rows = []
    for r in routes:
        a, b = nodes[r["a"]], nodes[r["b"]]
        gc = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        rows.append({**r, "gc_km": gc, "hill": edge_hill(a, b)})
    return rows


def fit(rows: list[dict]) -> dict:
    """Medians over the 'fit' routes: factor = ref_km / great-circle km, speed = ref_km / ref_h,
    separately for plain and hill edges (D77, D78). Robust to a few wrong reference values."""
    out = {}
    for prefix, hill in (("", False), ("hill_", True)):
        use = [r for r in rows if r["split"] == "fit" and r["hill"] == hill]
        if not use:
            raise ValueError(f"no fit routes for {'hill' if hill else 'plain'} edges")
        out[prefix + "road_factor"] = round(_median([r["ref_km"] / r["gc_km"] for r in use]), 4)
        out[prefix + "speed_kmh"] = round(_median([r["ref_km"] / r["ref_h"] for r in use]), 2)
        out[prefix + "n_fit"] = len(use)
    return out


def route_errors(rows: list[dict], fitted: dict, min_hours: float = 0.25) -> list[dict]:
    out = []
    for r in rows:
        prefix = "hill_" if r["hill"] else ""
        km = r["gc_km"] * fitted[prefix + "road_factor"]
        h = max(min_hours, km / fitted[prefix + "speed_kmh"])
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


def leave_one_out(rows: list[dict]) -> dict:
    """Refit without each fit route, predict it, summarise the errors (fit routes only)."""
    errs = []
    for i, r in enumerate(rows):
        if r["split"] != "fit":
            continue
        try:
            f = fit([x for j, x in enumerate(rows) if j != i])
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
            k_, h_ = estimate(a["lat"], a["lon"], b["lat"], b["lon"], edge_hill(a, b), fitted,
                              min_hours)
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
