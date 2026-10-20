"""Season normals, verdict rules, provenance-aware evidence and check_season (D80, D81)."""
from __future__ import annotations

import json
import time

import pytest

from src.common.schemas import REPO_ROOT
from src.places.places import Registry
from src.tools import check_season as cs
from src.tools import season as se
from src.tools.routing import UnknownPlaceError
from src.tools.travel_model import REGISTRY_PATH

TH = {"poor_rain_mm": 300, "fair_rain_mm": 150, "poor_tmax_c": 38, "fair_tmax_c": 35}


def daily(rows):
    return {"time": [r[0] for r in rows], "temperature_2m_max": [r[1] for r in rows],
            "temperature_2m_min": [r[2] for r in rows], "precipitation_sum": [r[3] for r in rows]}


def test_monthly_normals_average_over_years_and_count_rainy_days():
    d = daily([("2020-01-01", 30, 20, 2.5), ("2020-01-02", 32, 22, 2.49),
               ("2020-01-03", 28, 18, 0.0), ("2021-01-01", 31, 21, 10.0),
               ("2021-01-02", 29, 19, 0.0)])
    j = se.monthly_normals(d, 2.5)[1]
    assert j["years"] == 2 and j["tmax_c"] == 30.0 and j["tmin_c"] == 20.0
    assert j["rain_mm"] == round((2.5 + 2.49 + 10.0) / 2, 1)
    assert j["rainy_days"] == 1.0  # one rainy day in each year: 2.5 counts (>=), 2.49 does not


def test_rainy_day_boundary_is_inclusive():
    on = se.monthly_normals(daily([("2020-03-01", 30, 20, 2.5)]), 2.5)[3]
    below = se.monthly_normals(daily([("2020-03-01", 30, 20, 2.4999)]), 2.5)[3]
    assert on["rainy_days"] == 1.0 and below["rainy_days"] == 0.0


def test_null_values_are_skipped():
    d = daily([("2020-02-01", None, 20, None), ("2020-02-02", 30, None, 5.0)])
    n = se.monthly_normals(d, 2.5)[2]
    assert n["tmax_c"] == 30.0 and n["tmin_c"] == 20.0 and n["rain_mm"] == 5.0


def test_verdict_thresholds_and_access_risk():
    def v(rain, tmax, hill=False):
        return se.verdict({"rain_mm": rain, "tmax_c": tmax}, hill, TH)

    assert v(10, 25)[0] == "good"
    assert v(150, 25)[0] == "fair" and v(149.9, 25)[0] == "good"
    assert v(300, 25)[0] == "poor" and v(299.9, 25)[0] == "fair"
    assert v(10, 35)[0] == "fair" and v(10, 38)[0] == "poor"
    assert v(400, 25, hill=True)[2] is True and v(400, 25, hill=False)[2] is False
    assert v(200, 25, hill=True)[2] is False
    assert all(v(r, t)[1] for r in (0, 200, 400) for t in (20, 36, 40))


def test_climate_file_prefers_noelev(tmp_path):
    d = tmp_path / "gokarna"
    d.mkdir()
    (d / "archive_2015_2024.json").write_text("{}")
    assert se.climate_file("Gokarna", tmp_path).name == "archive_2015_2024.json"
    (d / "archive_2015_2024_noelev.json").write_text("{}")
    assert se.climate_file("Gokarna", tmp_path).name == "archive_2015_2024_noelev.json"
    assert se.climate_file("Nowhere", tmp_path) is None


def row(cid, source, tags, warn=False):
    return {"chunk_id": cid, "source": source, "tags": tags, "is_warning": warn}


def test_evidence_counts_are_split_by_provenance_and_rule_only_is_never_model_written():
    rows = [row("a", "model_written", ["monsoon"], True), row("b", "rule_only", ["monsoon"], True),
            row("c", "fallback", ["all_year"], True), row("d", "model_written", ["winter"], True),
            row("e", "rule_only", ["all_year"])]
    counts, ids, src = se.month_evidence(rows, 7)  # July is monsoon
    assert counts["model_written"] == {"chunks": 2, "season_tagged": 1, "all_year": 0}
    assert counts["rule_only"] == {"chunks": 2, "season_tagged": 1, "all_year": 1}
    assert counts["fallback"]["chunks"] == 1
    assert ids == ["a", "c", "b"]  # model-written first, then fallback, then rule-only
    assert src == {"a": "model_written", "b": "rule_only", "c": "fallback"}  # d is winter-only


def test_warning_list_is_capped_and_deterministic():
    rows = [row(f"{i:02d}", "model_written", ["all_year"], True) for i in range(9, -1, -1)]
    _, ids, _ = se.month_evidence(rows, 1)
    assert ids == ["00", "01", "02", "03", "04"]


def test_provenance_classes_come_from_the_sidecar(tmp_path):
    chunks = tmp_path / "c.jsonl"
    prov = tmp_path / "p.jsonl"
    base = {"place": "Munnar", "season_tags": ["all_year"], "is_warning": False}
    chunks.write_text("\n".join(json.dumps({**base, "chunk_id": c}) for c in "xyz"))
    prov.write_text("\n".join(json.dumps({"chunk_id": c, "source": s})
                              for c, s in (("x", "llm"), ("y", "fallback"), ("z", "rule_only"))))
    got = se.load_evidence(chunks, prov)["Munnar"]
    assert [r["source"] for r in got] == ["model_written", "fallback", "rule_only"]


def test_real_evidence_marks_hill_stations_as_mostly_rule_only():
    ev = se.load_evidence()
    munnar = {k: sum(r["source"] == k for r in ev["Munnar"])
              for k in ("model_written", "fallback", "rule_only")}
    assert munnar["rule_only"] > munnar["model_written"] and munnar["model_written"] == 2


def test_parse_month():
    assert cs.parse_month(7) == 7 and cs.parse_month("Jul") == 7 == cs.parse_month("JULY")
    for bad in (0, 13, -1, True, "13", "Julember", None, 7.0, ""):
        with pytest.raises(ValueError):
            cs.parse_month(bad)


def test_check_season_returns_the_contract_fields():
    r = cs.check_season("Munnar", 7)
    assert {"place", "month", "verdict", "reasons", "access_risk", "climate", "evidence",
            "warning_chunk_ids", "warning_sources", "confidence", "caveats"} <= set(r)
    assert r["verdict"] in ("good", "fair", "poor") and r["month"] == 7
    assert set(r["evidence"]) == {"model_written", "fallback", "rule_only"}
    no_model = r["evidence"]["model_written"]["chunks"] == 0
    assert any("rule-based" in c for c in r["caveats"]) is no_model


def test_confidence_is_low_for_the_five_places_and_medium_otherwise():
    low = {"Munnar", "Wayanad", "Kodaikanal", "Kodagu", "Ooty"}
    for e in Registry.load(REGISTRY_PATH).entries:
        if e.holdout:
            continue
        assert cs.check_season(e.name, 1)["confidence"] == ("low" if e.name in low else "medium")


def test_every_destination_every_month_is_answered_instantly():
    names = [e.name for e in Registry.load(REGISTRY_PATH).entries if not e.holdout]
    cs.check_season("Munnar", 1)
    t0 = time.perf_counter()
    for n in names:
        for m in range(1, 13):
            assert cs.check_season(n, m)["verdict"] in ("good", "fair", "poor")
    assert (time.perf_counter() - t0) / (26 * 12) < 0.001


def test_aliases_resolve_and_hubs_and_holdouts_are_unknown():
    assert cs.check_season("Coorg", "Jul")["place"] == "Kodagu"
    assert cs.check_season("bangalore", 1)["place"] == "Bengaluru"
    for name in ("Coimbatore", "CJB", "Thanjavur", "Thekkady", "Chikmagalur", "Araku Valley", "x"):
        with pytest.raises(UnknownPlaceError):
            cs.check_season(name, 1)


def test_monsoon_hill_stations_are_flagged_with_access_risk():
    for place in ("Munnar", "Kodagu", "Wayanad"):
        r = cs.check_season(place, 7)
        assert r["access_risk"] and r["verdict"] == "poor"


def test_noelev_files_are_primary_for_the_three_places():
    table = json.loads((REPO_ROOT / "index" / "season_data.json").read_text(encoding="utf-8"))
    for p in ("Gokarna", "Kanniyakumari", "Varkala"):
        assert table["places"][p]["climate_file"] == "archive_2015_2024_noelev.json"
    assert table["places"]["Munnar"]["climate_file"] == "archive_2015_2024.json"
