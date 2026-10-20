"""Season normals, verdict rules, provenance-aware evidence and check_season (D80, D81, D83)."""
from __future__ import annotations

import calendar
import importlib.util
import json
import time

import pytest

from src.common.schemas import REPO_ROOT
from src.places.places import Registry
from src.tools import check_season as cs
from src.tools import season as se
from src.tools import travel_model as tm
from src.tools.routing import AIRPORTS_RAW_PATH, UnknownPlaceError
from src.tools.travel_model import REGISTRY_PATH

TH = {"poor_rain_mm": 300, "fair_rain_mm": 150, "poor_tmax_c": 38, "fair_tmax_c": 35}
needs_enriched = pytest.mark.skipif(
    not (se.ENRICHED_PATH.exists() and se.PROVENANCE_PATH.exists()),
    reason="data/enriched is gitignored and absent on a fresh clone")
needs_raw = pytest.mark.skipif(
    not (se.ENRICHED_PATH.exists() and se.PROVENANCE_PATH.exists() and AIRPORTS_RAW_PATH.exists()
         and tm.WIKIDATA_RAW.exists() and se.OPEN_METEO_RAW.exists()),
    reason="data/raw and data/enriched are gitignored and absent on a fresh clone")


def month_days(year, month, tmax=30.0, tmin=20.0, rain=None):
    """Daily rows for every calendar day of the month; `rain` maps day number -> mm (default 0)."""
    rain = rain or {}
    return [(f"{year}-{month:02d}-{d:02d}", tmax, tmin, rain.get(d, 0.0))
            for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def daily(rows):
    return {"time": [r[0] for r in rows], "temperature_2m_max": [r[1] for r in rows],
            "temperature_2m_min": [r[2] for r in rows], "precipitation_sum": [r[3] for r in rows]}


def test_monthly_normals_average_over_years_and_count_rainy_days():
    rows = (month_days(2020, 1, tmax=30, tmin=20, rain={1: 2.5, 2: 2.49})
            + month_days(2021, 1, tmax=32, tmin=22, rain={1: 10.0}))
    j = se.monthly_normals(daily(rows), 2.5)[1]
    assert j["years"] == 2 and j["tmax_c"] == 31.0 and j["tmin_c"] == 21.0
    assert j["rain_mm"] == round((2.5 + 2.49 + 10.0) / 2, 1)
    assert j["rainy_days"] == 1.0  # one rainy day in each year: 2.5 counts (>=), 2.49 does not


def test_rainy_day_boundary_is_inclusive():
    on = se.monthly_normals(daily(month_days(2020, 3, rain={1: 2.5})), 2.5)[3]
    below = se.monthly_normals(daily(month_days(2020, 3, rain={1: 2.4999})), 2.5)[3]
    assert on["rainy_days"] == 1.0 and below["rainy_days"] == 0.0


def test_a_month_with_a_missing_precipitation_day_does_not_count_as_a_year():
    """B5: a partial month must not be averaged in as a full year (it would under-report rain)."""
    full = month_days(2020, 4, rain={1: 100.0})
    partial = month_days(2021, 4, rain={1: 100.0})
    partial[5] = (partial[5][0], 30.0, 20.0, None)  # one day without precipitation
    n = se.monthly_normals(daily(full + partial), 2.5)[4]
    assert n["years"] == 1 and n["rain_mm"] == 100.0  # the partial year is left out
    # a month whose only year is partial has no rain figure at all, so it is absent
    assert 4 not in se.monthly_normals(daily(partial), 2.5)


def test_a_month_with_absent_days_is_incomplete_too():
    rows = month_days(2020, 5, rain={1: 50.0})[:-3]  # last three days missing from the series
    assert 5 not in se.monthly_normals(daily(rows), 2.5)


def test_february_in_a_leap_year_needs_29_days():
    assert se.monthly_normals(daily(month_days(2020, 2, rain={1: 5.0})), 2.5)[2]["years"] == 1
    short = month_days(2020, 2, rain={1: 5.0})[:-1]
    assert 2 not in se.monthly_normals(daily(short), 2.5)


def test_null_temperatures_are_skipped_but_do_not_break_the_rain_year():
    rows = month_days(2020, 2, rain={1: 5.0})
    rows[0] = (rows[0][0], None, 20.0, 5.0)
    rows[1] = (rows[1][0], 30.0, None, 0.0)
    n = se.monthly_normals(daily(rows), 2.5)[2]
    assert n["tmax_c"] == 30.0 and n["tmin_c"] == 20.0 and n["rain_mm"] == 5.0 and n["years"] == 1


def test_mismatched_series_lengths_are_refused():
    bad = {"time": ["2020-01-01"], "temperature_2m_max": [1, 2], "temperature_2m_min": [1],
           "precipitation_sum": [1]}
    with pytest.raises(ValueError):
        se.monthly_normals(bad, 2.5)


def test_every_month_maps_to_a_season_and_december_is_winter():
    assert set(se.SEASON_OF_MONTH) == set(range(1, 13))
    assert se.SEASON_OF_MONTH[12] == "winter" and se.SEASON_OF_MONTH[1] == "winter"
    assert se.SEASON_OF_MONTH[2] == "winter" and se.SEASON_OF_MONTH[3] == "summer"
    assert se.SEASON_OF_MONTH[6] == "monsoon" and se.SEASON_OF_MONTH[9] == "monsoon"
    assert se.SEASON_OF_MONTH[10] == "post_monsoon" and se.SEASON_OF_MONTH[11] == "post_monsoon"


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


def test_evidence_counts_are_split_by_provenance_and_only_model_warnings_are_listed():
    rows = [row("a", "model_written", ["monsoon"], True), row("b", "rule_only", ["monsoon"], True),
            row("c", "fallback", ["all_year"], True), row("d", "model_written", ["winter"], True),
            row("e", "rule_only", ["all_year"])]
    counts, ids, src = se.month_evidence(rows, 7)  # July is monsoon
    assert counts["model_written"] == {"chunks": 2, "season_tagged": 1, "all_year": 0}
    assert counts["rule_only"] == {"chunks": 2, "season_tagged": 1, "all_year": 1}
    assert counts["fallback"]["chunks"] == 1
    assert ids == ["a"] and src == {"a": "model_written"}  # rule-based and fallback warnings drop


def test_a_place_without_model_warnings_lists_none():
    rows = [row("b", "rule_only", ["all_year"], True), row("c", "fallback", ["all_year"], True)]
    _, ids, src = se.month_evidence(rows, 1)
    assert ids == [] and src == {}


def test_warning_list_is_capped_and_ordered_by_chunk_id():
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


def test_a_chunk_without_a_provenance_row_is_an_error(tmp_path):
    chunks = tmp_path / "c.jsonl"
    prov = tmp_path / "p.jsonl"
    base = {"place": "Munnar", "season_tags": ["all_year"], "is_warning": False}
    chunks.write_text("\n".join(json.dumps({**base, "chunk_id": c}) for c in ("x", "orphan")))
    prov.write_text(json.dumps({"chunk_id": "x", "source": "llm"}))
    with pytest.raises(ValueError, match="orphan"):
        se.load_evidence(chunks, prov)


def test_build_season_on_synthetic_data(tmp_path):
    d = tmp_path / "munnar"
    d.mkdir()
    rows = []
    for year in (2020, 2021):  # a wet July, dry January
        rows += month_days(year, 1, tmax=22, tmin=10)
        rows += month_days(year, 7, tmax=19, tmin=14, rain={day: 20.0 for day in range(1, 21)})
    (d / "archive_2015_2024.json").write_text(
        json.dumps({"response": {"daily": daily(rows)}}), encoding="utf-8")
    params = {"rainy_day_mm": 2.5, "season_thresholds": TH, "low_confidence_places": ["Munnar"]}
    nodes = [{"name": "Munnar", "hill": True, "lat": 10.09, "lon": 77.06},
             {"name": "Nowhere", "hill": False, "lat": 0.0, "lon": 0.0}]
    ev = {"Munnar": [row("w1", "model_written", ["monsoon"], True),
                     row("r1", "rule_only", ["all_year"], True)]}
    airports = {"COK": {"lat": 10.15, "lon": 76.40, "name": "Cochin", "military": False}}
    out = se.build_season(nodes, params, ev, tmp_path, airports)
    assert list(out) == ["Munnar"]  # a place without a climate file is left out
    m = out["Munnar"]
    assert m["confidence"] == "low" and m["hill"] is True and set(m["months"]) == {"1", "7"}
    assert m["warning_counts"] == {"model_written": 1, "fallback": 0, "rule_only": 1}
    assert m["nearest_airport"]["iata"] == "COK"
    july, jan = m["months"]["7"], m["months"]["1"]
    assert july["verdict"] == "poor" and july["access_risk"] is True
    assert jan["verdict"] == "good" and jan["access_risk"] is False
    assert july["warning_chunk_ids"] == ["w1"] and jan["warning_chunk_ids"] == []
    assert july["evidence"]["model_written"]["season_tagged"] == 1


@needs_enriched
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
    assert any("tags are rule-based" in c for c in r["caveats"]) is no_model


def test_warning_caveats_follow_the_model_written_warning_count():
    munnar = cs.check_season("Munnar", 7)  # no model-written warnings, 5 rule-based ones
    assert munnar["warning_chunk_ids"] == []
    assert any("no model-written warnings" in c and "5 rule-based" in c for c in munnar["caveats"])
    chennai = cs.check_season("Chennai", 7)  # 7 model-written warnings
    assert not any("no model-written warnings" in c for c in chennai["caveats"])
    assert chennai["warning_chunk_ids"] and set(chennai["warning_sources"].values()) == {
        "model_written"}


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


def test_junk_alias_is_not_in_the_season_meta():
    table = json.loads((REPO_ROOT / "index" / "season_data.json").read_text(encoding="utf-8"))
    assert table["meta"]["aliases"]["Munnar"] == ["Kashmir of South India"]
    assert all(len(a) <= 40 for al in table["meta"]["aliases"].values() for a in al)


@needs_raw
def test_a_rebuild_equals_the_committed_index_files():
    """The committed files are exactly what the code builds from the saved inputs."""
    spec = importlib.util.spec_from_file_location(
        "build_routing", REPO_ROOT / "scripts" / "build_routing.py")
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    for builder, path in ((br.build_matrix_doc, br.MATRIX_OUT), (br.build_season_doc,
                                                                br.SEASON_OUT)):
        assert br.dump(builder()) == path.read_text(encoding="utf-8").replace("\r\n", "\n")
