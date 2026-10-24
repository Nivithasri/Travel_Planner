"""Phase 9 tooling (D123 to D128): run the three systems, build the blind scoring sheets, score,
check citations, run the lean judge, the live day, latency and the report.

    python scripts/phase9.py check-requests
    python scripts/phase9.py run --arm llm_only|plain_rag|full_corpus|full_live [--ids P01,P02]
    python scripts/phase9.py sheet --set main|live
    python scripts/phase9.py score --set main|live --filled <csv>
    python scripts/phase9.py crosscheck sample [--n 10]
    python scripts/phase9.py crosscheck score --a <main filled csv> --b <partner filled csv>
    python scripts/phase9.py citations sample [--n 50]
    python scripts/phase9.py citations score --filled <csv>
    python scripts/phase9.py judge [--budget-tokens 120000]
    python scripts/phase9.py latency --yes [--with-live]
    python scripts/phase9.py report

A run needs eval/phase9/requests.json with meta.status APPROVED (the owner's step). Records hold
answer text and source excerpts and live in .cache/phase9 (gitignored); sheets live in
eval/sheets/phase9 (gitignored); only scores, counts and agreement numbers go to
eval/phase9/results (committed). Use the .venv interpreter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from eval.phase9 import agreement, claims, report, request_set, sheets, store, systems  # noqa: E402

ARMS = systems.ARMS
MAIN_ARMS = ("llm_only", "plain_rag", "full_corpus")
SYSTEM_OF = {"llm_only": "llm_only", "plain_rag": "plain_rag", "full_corpus": "full",
             "full_live": "full"}
D33 = ROOT / "data" / "manifests" / "d33_excluded.json"


def out(msg: str = "") -> None:
    print(msg, flush=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def requests_by_id(require_approved: bool = True) -> dict[str, dict]:
    return {r["id"]: r for r in request_set.load_requests(require_approved=require_approved)}


def pick(reqs: dict[str, dict], ids: str | None) -> list[dict]:
    if not ids:
        return [reqs[k] for k in sorted(reqs)]
    wanted = [x.strip() for x in ids.split(",") if x.strip()]
    unknown = [x for x in wanted if x not in reqs]
    if unknown:
        raise SystemExit(f"unknown request id(s): {unknown}")
    return [reqs[k] for k in wanted]


# ---- check-requests ---------------------------------------------------------------------------

def city_ok(city: str) -> bool:
    from src.tools.travel_time import travel_time

    try:
        travel_time(city, "Bengaluru" if city.casefold() != "bengaluru" else "Chennai")
        return True
    except ValueError:
        return False


def cmd_check_requests(args) -> int:
    from src.retrieval.query_understanding import load_core_registry

    doc = request_set.load_document()
    problems = request_set.validate_requests(
        doc, registry_entries=load_core_registry().entries,
        excluded=read_json(D33)["excluded"], city_ok=city_ok)
    status = doc.get("meta", {}).get("status")
    if problems:
        out("PROBLEMS:")
        for p in problems:
            out(f"  - {p}")
        return 1
    n_ts = sum(1 for r in doc["requests"] if r["time_sensitive"])
    out(f"OK: {len(doc['requests'])} requests, {n_ts} time-sensitive, status {status}")
    return 0


# ---- run --------------------------------------------------------------------------------------

def cmd_run(args) -> int:
    from app.state import default_runner, safe_error
    from src.agents.deps import build_default_deps

    reqs = pick(requests_by_id(), args.ids)
    arm = args.arm
    git = systems.git_state()
    if args.expect_sha and git["git_sha"] != args.expect_sha:
        raise SystemExit(f"HEAD is {git['git_sha']}, expected {args.expect_sha}: the measured "
                         "code is not the frozen sha")
    if git["dirty"]:
        out("warning: the working tree has uncommitted changes; the record says dirty")
    live = arm == "full_live"
    if live and not guard_live(args, reqs):
        return 2
    deps = build_default_deps(live=live, parse=arm in ("full_corpus", "full_live"))
    runner = default_runner()
    log_before = store.line_count(store.REQUEST_LOG)
    searches = 0
    for req in reqs:
        path = store.record_path(arm, req["id"])
        if path.exists() and not args.force:
            out(f"{req['id']} {arm}: record exists, skipped")
            continue
        try:
            if arm == "llm_only":
                rec = systems.run_llm_only(req, deps.client, deps.model, deps.pipeline_cfg,
                                           git=git)
            elif arm == "plain_rag":
                rec = systems.run_plain_rag(req, deps.retrieve, deps.client, deps.model,
                                            deps.pipeline_cfg, git=git)
            else:
                rec = systems.run_full(req, deps, runner, live=live, git=git)
        except Exception as exc:  # a quota stop or an outage: keep what exists, resume later
            out(f"{req['id']} {arm}: stopped: {safe_error(exc)}")
            return 5
        store.save_record(rec, force=args.force)
        searches += int((rec.get("live") or {}).get("searches", 0))
        out(f"{req['id']} {arm}: tokens {rec['tokens'].get('total')}, critic {rec['critic']}, "
            f"citations {len(rec['citations'])}")
    if live:
        grew = store.line_count(store.REQUEST_LOG) - log_before
        flag = "OK" if grew == searches else "MISMATCH"
        out(f"request log grew by {grew} line(s); searches made {searches}: {flag}")
    return 0


def guard_live(args, reqs: list[dict]) -> bool:
    from src.common.schemas import load_live_settings
    from src.live.config import load_live_config

    cfg = load_live_config()
    ledger = None
    if store.LEDGER.exists():
        ledger = read_json(store.LEDGER)
    dates = {r["meta"]["utc_date"] for r in store.load_arm("full_live")}
    problems = store.live_guard(
        terms_status=cfg.real_calls_status(), key_present=bool(load_live_settings().api_key),
        ledger=ledger, planned_searches=len(reqs) * cfg.default_searches_per_request,
        credits_per_search=cfg.credits_per_search, credit_cap=args.credit_cap,
        monthly_stop_at=cfg.monthly_stop_at, existing_dates=dates,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    for p in problems:
        out(f"live guard: {p}")
    return not problems


# ---- sheets and scores ------------------------------------------------------------------------

def sheet_paths(name: str) -> tuple[Path, Path]:
    return store.SHEETS / f"{name}_sheet.csv", store.SHEETS / f"{name}_key.json"


def records_for(set_name: str, reqs: dict[str, dict]) -> list[dict]:
    arms = MAIN_ARMS if set_name == "main" else ("full_corpus", "full_live")
    recs = [r for a in arms for r in store.load_arm(a, reqs)]
    need = len(reqs) * len(arms)
    if len(recs) != need:
        raise SystemExit(f"the {set_name} sheet needs {need} records ({len(arms)} arms x "
                         f"{len(reqs)} requests), found {len(recs)}: run the missing arms first")
    return recs


def cmd_sheet(args) -> int:
    reqs = requests_by_id()
    recs = records_for(args.set, reqs)
    rows, key = sheets.build_sheet(recs, reqs, seed=args.seed)
    csv_path, key_path = sheet_paths(args.set)
    sheets.write_csv(rows, csv_path)
    write_json(key_path, key)
    out(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} rows) and the hidden key "
        f"{key_path.relative_to(ROOT)}; score the four dimensions 1 to 5 by eval/phase9/rubric.md")
    return 0


def scored(set_name: str, filled: Path):
    key = read_json(sheet_paths(set_name)[1])
    scores = sheets.read_scores(sheets.read_csv(filled), key)
    table = sheets.by_request_system(scores, key, arm=set_name == "live")
    return key, scores, table


def cmd_score(args) -> int:
    key, _scores, table = scored(args.set, Path(args.filled))
    by_system = report.per_system(table)
    pairs = report.SYSTEM_PAIRS if args.set == "main" else (("full_live", "full_corpus"),)
    comps = report.compare_pairs(by_system, pairs)
    result = {"set": args.set, "means": report.means(by_system), "comparisons": comps,
              "per_request": {f"{rid}|{system}": s for (rid, system), s in sorted(table.items())},
              "scorer": "owner (human)", "n_requests": len({k[0] for k in table})}
    write_json(store.RESULTS / f"{args.set}_scores.json", result)
    out(report.md_table(result["means"], ["system", "n", *sheets.DIMENSIONS]))
    out(report.md_table(comps, ["first", "second", "dimension", "n", "mean_diff", "ci", "p",
                                "p_holm"]))
    return 0


def cmd_crosscheck(args) -> int:
    csv_path, key_path = sheet_paths("main")
    key = read_json(key_path)
    if args.what == "sample":
        ids = sheets.cross_check_sample(key, args.n, seed=args.seed)
        sub = [r for r in sheets.read_csv(csv_path) if r["row_id"] in ids]
        for r in sub:
            for d in (*sheets.DIMENSIONS, "note"):
                r[d] = ""
        sub_path = store.SHEETS / "crosscheck_sheet.csv"
        sheets.write_csv(sub, sub_path)
        write_json(store.SHEETS / "crosscheck_key.json", {i: key[i] for i in ids})
        out(f"wrote {sub_path.relative_to(ROOT)} with {len(ids)} corpus-only rows for the second "
            "scorer (D124)")
        return 0
    sub_key = read_json(store.SHEETS / "crosscheck_key.json")
    a = sheets.read_scores([r for r in sheets.read_csv(Path(args.a)) if r["row_id"] in sub_key],
                           sub_key)
    b = sheets.read_scores(sheets.read_csv(Path(args.b)), sub_key)
    rep = agreement.agreement_by_dimension(a, b, sheets.DIMENSIONS)
    write_json(store.RESULTS / "crosscheck_agreement.json",
               {"n_rows": len(sub_key), "agreement": rep})
    for k, v in rep.items():
        out(f"{k}: {v}")
    return 0


# ---- citations --------------------------------------------------------------------------------

def cmd_citations(args) -> int:
    reqs = requests_by_id()
    paths = (store.SHEETS / "citations_sheet.csv", store.SHEETS / "citations_key.json")
    if args.what == "sample":
        recs = {(r["request_id"], r["arm"]): r for a in ("full_corpus", "full_live")
                for r in store.load_arm(a, reqs)}
        full = [r for r in recs.values() if r["arm"] == "full_corpus"]
        chosen = claims.sample_claims(full, args.n, seed=args.seed)
        chosen += claims.live_claims(recs.values())
        rows, key = claims.build_claim_sheet(chosen, recs, reqs, seed=args.seed)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        with paths[0].open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(claims.COLUMNS))
            w.writeheader()
            w.writerows(rows)
        write_json(paths[1], key)
        out(f"wrote {paths[0].relative_to(ROOT)}: {len(rows)} claims (mark supported 1 or 0)")
        return 0
    key = read_json(paths[1])
    marks = claims.read_marks(sheets.read_csv(Path(args.filled)), key)
    res = claims.precision(marks, key)
    write_json(store.RESULTS / "citation_precision.json", {**res, "scorer": "owner (human)"})
    out(json.dumps(res))
    return 0


# ---- judge ------------------------------------------------------------------------------------

def cmd_judge(args) -> int:
    from app.state import safe_error
    from src.common.llm import OpenAICompatClient
    from src.common.schemas import load_embed_local_dir, load_llm_settings
    from src.retrieval.build_index import load_encoder

    reqs = requests_by_id()
    settings = load_llm_settings()
    client = OpenAICompatClient.from_settings(max_tokens_cap=1000)
    encoder = load_encoder(load_embed_local_dir())
    jdir = store.CACHE / "judge"
    jdir.mkdir(parents=True, exist_ok=True)
    start = int(client.usage.get("total", 0))
    for arm in ("full_corpus", "plain_rag"):
        for rec in store.load_arm(arm, reqs):
            path = jdir / f"{arm}_{rec['request_id']}.json"
            if path.exists():
                continue
            if int(client.usage.get("total", 0)) - start > args.budget_tokens:
                out("token budget reached: stopping; rerun tomorrow to resume")
                return 5
            try:
                res = judge_one(rec, reqs[rec["request_id"]], client, settings.judge_model,
                                encoder)
            except Exception as exc:
                out(f"{arm} {rec['request_id']}: stopped: {safe_error(exc)}")
                return 5
            write_json(path, res)
            out(f"{arm} {rec['request_id']}: {res}")
    summarise_judge(jdir)
    return 0


def judge_one(rec: dict, req: dict, client, model: str, encoder) -> dict:
    from eval.phase9 import judge

    return judge.judge_record(rec, systems.request_line(req), client, model,
                              lambda texts: [list(v) for v in encoder(texts)])


def summarise_judge(jdir: Path) -> None:
    import statistics

    res: dict = {}
    for p in sorted(jdir.glob("*.json")):
        r = read_json(p)
        for k in ("faithfulness", "answer_relevancy", "context_precision"):
            if r.get(k) is not None:
                res.setdefault(r["arm"], {}).setdefault(k, []).append(r[k])
    summary = {arm: {k: {"n": len(v), "mean": round(statistics.fmean(v), 3)}
                     for k, v in d.items()} for arm, d in res.items()}
    write_json(store.RESULTS / "judge.json", {"judge": "hosted JUDGE_MODEL (not human)",
                                              "context_recall": "not computed (no references)",
                                              "arms": summary})


# ---- latency ----------------------------------------------------------------------------------

def cmd_latency(args) -> int:
    if not args.yes:
        out("latency runs make real LLM calls with every cache bypassed (about one full run of "
            "tokens per request); rerun with --yes to accept")
        return 2
    tmp = tempfile.mkdtemp(prefix="phase9_latency_")
    os.environ["CACHE_DIR"] = tmp  # D127: every cache (LLM, query vectors, live) misses
    from app.state import default_runner, safe_error
    from src.agents.deps import build_default_deps

    reqs = pick(requests_by_id(), args.ids)
    git = systems.git_state()
    runner = default_runner()
    for live in (False, True) if args.with_live else (False,):
        if live and not guard_live(args, reqs):
            return 2
        deps = build_default_deps(live=live, parse=True)
        waits = wrap_limiter(deps.client)
        deps.retrieve("warm up the embedding model", k=1)  # the model load is not latency
        for req in reqs:
            waits[0] = 0.0
            try:
                rec = systems.run_full(req, deps, runner, live=live, git=git)
            except Exception as exc:
                out(f"{req['id']}: stopped: {safe_error(exc)}")
                return 5
            rec["meta"]["limiter_wait_s"] = round(waits[0], 3)
            rec["arm"] = "latency_live" if live else "latency_corpus"
            store.save_record(rec, force=True)
            out(f"{req['id']} {rec['arm']}: wall {rec['meta']['wall_s']} s, limiter wait "
                f"{rec['meta']['limiter_wait_s']} s")
    return 0


def wrap_limiter(client) -> list[float]:
    """Total seconds the client's rate limiter slept, in a one-element list (D127)."""
    waited = [0.0]
    lim = getattr(client, "limiter", None)
    if lim is not None:
        real = lim._sleep

        def timed(seconds):
            waited[0] += float(seconds)
            real(seconds)

        lim._sleep = timed
    return waited


# ---- report -----------------------------------------------------------------------------------

def cmd_report(args) -> int:
    reqs = requests_by_id(require_approved=False)
    summary: dict = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                     "n_requests": len(reqs)}
    for name in ("main_scores", "live_scores", "citation_precision", "judge",
                 "crosscheck_agreement"):
        p = store.RESULTS / f"{name}.json"
        if p.exists():
            summary[name] = read_json(p)
    live = store.load_arm("full_live")
    summary["live"] = report.live_summary(live)
    lat = [r for a in ("latency_corpus", "latency_live") for r in store.load_arm(a)]
    summary["latency"] = report.latency_table(lat)
    if "main_scores" in summary:
        main = summary["main_scores"]["per_request"]
        table = {tuple(k.split("|")): v for k, v in main.items()}
        recs = store.load_arm("full_corpus") + live
        summary["case_candidates"] = report.case_candidates(recs, table)
    write_json(store.RESULTS / "summary.json", summary)
    lines = ["# Phase 9 summary (generated; scores and counts only, no answer text)", ""]
    for name in ("main_scores", "live_scores"):
        if name in summary:
            lines += [f"## {name}", report.md_table(summary[name]["means"],
                                                      ["system", "n", *sheets.DIMENSIONS]),
                      report.md_table(summary[name]["comparisons"],
                                      ["first", "second", "dimension", "n", "mean_diff", "ci",
                                       "p", "p_holm"]), ""]
    lines += ["## live arm (counts)", "```", json.dumps(summary["live"], indent=1), "```", "",
              "## latency", report.md_table(summary["latency"],
                                            ["arm", "node", "n", "median_s", "p90_s"])]
    (store.RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out(f"wrote {store.RESULTS.relative_to(ROOT)}/summary.json and summary.md")
    return 0


# ---- argument parsing -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-requests").set_defaults(fn=cmd_check_requests)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--ids")
    r.add_argument("--force", action="store_true")
    r.add_argument("--expect-sha")
    r.add_argument("--credit-cap", type=int, default=40)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("sheet")
    s.add_argument("--set", choices=("main", "live"), required=True)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_sheet)
    sc = sub.add_parser("score")
    sc.add_argument("--set", choices=("main", "live"), required=True)
    sc.add_argument("--filled", required=True)
    sc.set_defaults(fn=cmd_score)
    cc = sub.add_parser("crosscheck")
    cc.add_argument("what", choices=("sample", "score"))
    cc.add_argument("--n", type=int, default=10)
    cc.add_argument("--seed", type=int, default=42)
    cc.add_argument("--a")
    cc.add_argument("--b")
    cc.set_defaults(fn=cmd_crosscheck)
    ci = sub.add_parser("citations")
    ci.add_argument("what", choices=("sample", "score"))
    ci.add_argument("--n", type=int, default=50)
    ci.add_argument("--seed", type=int, default=42)
    ci.add_argument("--filled")
    ci.set_defaults(fn=cmd_citations)
    j = sub.add_parser("judge")
    j.add_argument("--budget-tokens", type=int, default=120_000)
    j.set_defaults(fn=cmd_judge)
    la = sub.add_parser("latency")
    la.add_argument("--yes", action="store_true")
    la.add_argument("--with-live", action="store_true")
    la.add_argument("--ids")
    la.add_argument("--credit-cap", type=int, default=40)
    la.set_defaults(fn=cmd_latency)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.monotonic()
    code = args.fn(args)
    out(f"done in {time.monotonic() - t0:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
