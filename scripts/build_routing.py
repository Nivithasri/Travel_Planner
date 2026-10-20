"""Build and check the Phase 5 routing data (D76 to D81). Offline, deterministic, no network.

    python scripts/build_routing.py calibrate   # fit, write config/routing_params.json
    python scripts/build_routing.py matrix      # write index/travel_matrix.json
    python scripts/build_routing.py season      # write index/season_data.json
    python scripts/build_routing.py sanity      # run the sanity cases, report every miss
    python scripts/build_routing.py all         # calibrate, matrix, season, manifest
    python scripts/build_routing.py report      # nodes, hill flags, hubs, calibration errors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.places.places import Registry  # noqa: E402
from src.tools import season as season_mod  # noqa: E402
from src.tools import travel_model as tm  # noqa: E402
from src.tools.routing import load_airports  # noqa: E402

MATRIX_OUT = ROOT / "index" / "travel_matrix.json"
SEASON_OUT = ROOT / "index" / "season_data.json"
MANIFEST = ROOT / "data" / "manifests" / "routing_manifest.json"
SANITY = ROOT / "config" / "routing_sanity_cases.json"
ATTRIBUTION = ("Climate: Weather data by Open-Meteo.com (CC BY 4.0), ERA5 reanalysis. Airports: "
               "OpenStreetMap contributors (ODbL). Elevations: Wikidata (CC0). Distances and "
               "times are model estimates fitted on unverified reference routes (D77).")


def dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context():
    registry = Registry.load(tm.REGISTRY_PATH)
    params = tm.load_params()
    airports = load_airports()
    nodes, report = tm.build_nodes(registry, airports, params)
    if len(report["hubs"]) != params["hub_expected_count"]:
        raise SystemExit(f"STOP: the hub rule gave {len(report['hubs'])} hubs, expected "
                         f"{params['hub_expected_count']} (D79). Report to the owner.")
    return registry, params, airports, nodes, report


def calibration(nodes: list[dict], params: dict):
    routes = json.loads(tm.ROUTES_PATH.read_text(encoding="utf-8"))["routes"]
    by_id = {n["id"]: n for n in nodes}
    rows = tm.route_rows(routes, by_id)
    return rows, tm.fit(rows)


def print_calibration(rows, fitted) -> dict:
    errors = tm.route_errors(rows, fitted)
    print(f"fitted (PROVISIONAL, agreement with UNVERIFIED reference values): {fitted}")
    print("route | split | edge | ref km -> km | ref h -> h | err% km / h")
    for e in errors:
        print(f"{e['a']}-{e['b']} | {e['split']} | {'hill' if e['hill'] else 'plain'} | "
              f"{e['ref_km']} -> {e['km']} | {e['ref_h']} -> {e['hours']} | "
              f"{e['km_err_pct']:+.1f} / {e['h_err_pct']:+.1f}")
    summary = {"fit": tm.error_summary([e for e in errors if e["split"] == "fit"]),
               "validate": tm.error_summary([e for e in errors if e["split"] == "validate"]),
               "leave_one_out": tm.leave_one_out(rows)}
    for k, v in summary.items():
        print(f"{k}: {v}")
    return summary


def cmd_calibrate() -> None:
    _, params, _, nodes, _ = context()
    rows, fitted = calibration(nodes, params)
    summary = print_calibration(rows, fitted)
    params["fitted"] = {**fitted, "summary": summary, "status": "PROVISIONAL"}
    tm.PARAMS_PATH.write_text(dump(params), encoding="utf-8", newline="\n")
    print(f"wrote {tm.PARAMS_PATH.relative_to(ROOT)}")


def cmd_matrix() -> dict:
    _, params, airports, nodes, report = context()
    if not params.get("fitted"):
        raise SystemExit("run `calibrate` first")
    hours, km = tm.matrix_pairs(nodes, params["fitted"], params["min_hours"])
    tri = tm.triangle_violations(nodes, hours)
    meta = {"status": "PROVISIONAL", "decision": "D76-D79", "attribution": ATTRIBUTION,
            "params": {k: v for k, v in params["fitted"].items() if k != "summary"},
            "nodes": len(nodes), "hubs": len(report["hubs"]), "pairs": len(hours),
            "triangle_violations": tri}
    MATRIX_OUT.write_text(dump({"meta": meta, "nodes": nodes, "hours": hours, "km": km}),
                          encoding="utf-8", newline="\n")
    print(f"wrote {MATRIX_OUT.relative_to(ROOT)}: {len(nodes)} nodes, {len(hours)} pairs, "
          f"triangle violations {tri}")
    return meta


def cmd_season() -> None:
    registry, params, airports, nodes, _ = context()
    dest = [n for n in nodes if n["kind"] == "destination"]
    evidence = season_mod.load_evidence()
    places = season_mod.build_season(dest, params, evidence, airports=airports)
    aliases = {n["name"]: n["aliases"] for n in dest}
    meta = {"status": "PROVISIONAL", "decision": "D80", "attribution": ATTRIBUTION,
            "thresholds": params["season_thresholds"], "rainy_day_mm": params["rainy_day_mm"],
            "aliases": aliases, "places": len(places)}
    SEASON_OUT.write_text(dump({"meta": meta, "places": places}), encoding="utf-8", newline="\n")
    print(f"wrote {SEASON_OUT.relative_to(ROOT)}: {len(places)} places x 12 months")


def cmd_sanity() -> None:
    from src.tools.check_season import SeasonTable
    from src.tools.travel_time import Matrix
    cases = json.loads(SANITY.read_text(encoding="utf-8"))
    matrix = Matrix(json.loads(MATRIX_OUT.read_text(encoding="utf-8")))
    table = SeasonTable(json.loads(SEASON_OUT.read_text(encoding="utf-8")))
    fitted = json.loads(tm.PARAMS_PATH.read_text(encoding="utf-8"))["fitted"]
    misses = 0
    for c in cases["travel"]:
        h = matrix.travel_time(c["a"], c["b"])
        ok = c["min_h"] <= h <= c["max_h"]
        extra = ""
        if c.get("hill_slower_than_plain"):
            ia, ib = matrix.resolve(c["a"]), matrix.resolve(c["b"])
            na, nb = matrix.nodes[ia], matrix.nodes[ib]
            _, plain_h = tm.estimate(na["lat"], na["lon"], nb["lat"], nb["lon"], False, fitted)
            ok = ok and h > plain_h
            extra = f" (plain model {plain_h:.2f} h)"
        misses += not ok
        print(f"{'PASS' if ok else 'MISS'} travel {c['a']}-{c['b']}: {h:.2f} h, "
              f"expected {c['min_h']} to {c['max_h']}{extra}")
    for c in cases["season"]:
        r = table.check(c["place"], c["month"])
        ok = r["verdict"] in c["expect"] and (not c.get("access_risk") or r["access_risk"])
        misses += not ok
        cl = r["climate"]
        print(f"{'PASS' if ok else 'MISS'} season {c['place']} month {c['month']}: "
              f"{r['verdict']} (access_risk {r['access_risk']}), expected {c['expect']}"
              f"{' + access_risk' if c.get('access_risk') else ''}; rain {cl['rain_mm']} mm, "
              f"Tmax {cl['tmax_c']} C, rainy days {cl['rainy_days']}")
    total = len(cases["travel"]) + len(cases["season"])
    print(f"sanity: {total - misses} of {total} as expected, {misses} misses (report all misses "
          f"together; a threshold change is a separate decision, D80)")


def cmd_manifest() -> None:
    inputs = [tm.REGISTRY_PATH, tm.ROUTES_PATH, tm.PARAMS_PATH,
              ROOT / "config" / "airports_allowlist.json",
              ROOT / "config" / "routing_sanity_cases.json",
              ROOT / "data" / "raw" / "overpass" / "_south" / "airports.json",
              season_mod.ENRICHED_PATH, season_mod.PROVENANCE_PATH]
    inputs += sorted((ROOT / "data" / "raw" / "open_meteo").glob("*/archive_2015_2024*.json"))
    inputs += sorted((ROOT / "data" / "raw" / "wikidata").glob("*/entity.json"))
    doc = {"inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs},
           "outputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                       for p in (MATRIX_OUT, SEASON_OUT)}}
    MANIFEST.write_text(dump(doc), encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(doc['inputs'])} inputs, 2 outputs")


def cmd_report() -> None:
    _, params, airports, nodes, report = context()
    print(f"nodes {len(nodes)}: {report['destinations']} destinations + {len(report['hubs'])} hubs")
    print(f"hubs: {report['hubs']}")
    print(f"military airports excluded (any distance): {report['excluded_military']}")
    print("hill-flagged nodes:")
    for name, why in report["hill_nodes"]:
        print(f"  {name}: {why}")
    rows, fitted = calibration(nodes, params)
    print_calibration(rows, fitted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["calibrate", "matrix", "season", "sanity", "all", "report"])
    args = ap.parse_args()
    if args.command == "calibrate":
        cmd_calibrate()
    elif args.command == "matrix":
        cmd_matrix()
    elif args.command == "season":
        cmd_season()
    elif args.command == "sanity":
        cmd_sanity()
    elif args.command == "report":
        cmd_report()
    else:
        cmd_calibrate()
        cmd_matrix()
        cmd_season()
        cmd_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
