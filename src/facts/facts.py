"""Facts layer (D36, D37, D40-D42): RFC 9309 matcher, hold-out filter, api_hosts allowlist, raw JSON
envelopes, and the Wikidata, Overpass and Open-Meteo fetchers."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from src.collect.run import BlockDetected, check_block, iso_utc
from src.common.cache import append_jsonl, atomic_write_text, read_jsonl
from src.common.http import FetchError, FetchResult, HttpClient, full_url
from src.common.schemas import REPO_ROOT, PlaceRegistryEntry
from src.places.places import slugify

# ==== robots =======================================================================
# An RFC 9309 robots.txt matcher (D41): groups by user-agent, `*` wildcard, `$` end anchor,
# the longest matching rule wins, Allow wins a tie, no matching rule means allowed.
#
# Python's urllib.robotparser treats `*` literally, so it cannot read patterns such as
# `Allow: /wiki/Special:EntityData/*.`. This module is used only to INFORM decisions (the HttpClient
# robots check is unchanged). Percent-encoding is compared as written (no normalisation).

@dataclass(frozen=True)
class Rule:
    directive: str  # "allow" | "disallow"
    pattern: str
    line: int


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    rule: Rule | None  # the deciding rule, None when nothing matched
    group: str  # the user-agent token of the group that was used
    matches: tuple[Rule, ...] = ()  # every rule that matched, for display


def parse(text: str) -> list[tuple[list[str], list[Rule]]]:
    """Groups of (user-agent tokens in lower case, rules); adjacent user-agent lines share one."""
    groups: list[tuple[list[str], list[Rule]]] = []
    agents: list[str] = []
    rules: list[Rule] = []
    in_rules = False
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if in_rules:
                groups.append((agents, rules))
                agents, rules, in_rules = [], [], False
            agents.append(value.lower())
        elif key in ("allow", "disallow"):
            in_rules = True
            if value:  # an empty Disallow means "allow everything": no rule
                rules.append(Rule(key, value, n))
    if agents:
        groups.append((agents, rules))
    return groups


def _regex(pattern: str) -> re.Pattern:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    rx = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(rx + ("$" if anchored else ""))


def _agent_group(groups, user_agent: str) -> tuple[str, list[Rule]]:
    """Rules of the most specific group naming our product token, else the `*` group (merged)."""
    token = re.split(r"[/\s]", user_agent.strip(), maxsplit=1)[0].lower()
    named = [r for agents, rs in groups for a in agents if a != "*" and a in token for r in rs]
    if any(a != "*" and a in token for agents, _ in groups for a in agents):
        return token, named
    star = [r for agents, rs in groups if "*" in agents for r in rs]
    return "*", star


def check(robots_text: str, user_agent: str, url: str) -> Verdict:
    parts = urlsplit(url)
    target = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
    group, rules = _agent_group(parse(robots_text), user_agent)
    matched = [r for r in rules if _regex(r.pattern).match(target)]
    if not matched:
        return Verdict(True, None, group)
    best = max(matched, key=lambda r: (len(r.pattern), r.directive == "allow"))
    return Verdict(best.directive == "allow", best, group, tuple(matched))


# ==== holdout ======================================================================
# Hold-out filter for facts (D17, D37, docs/facts-layer.md 3.2).
#
# A feature is dropped when any of its name tags matches a hold-out name or alias (word match),
# or when "Periyar" stands next to Thekkady, Kumily, Lake, Dam, Tiger Reserve, National Park or
# Wildlife Sanctuary (O8), or when it lies within `radius_km` of a hold-out centre.

DROP_NAMES = (
    "Thanjavur", "Tanjore", "Thanjai", "Chikmagalur", "Chikkamagaluru", "Araku Valley", "Araku",
    "Thekkady", "Kumily", "Brihadisvara", "Brihadeeswarar",
    "Thekkadi", "Brihadeeswara",  # spelling variants found in the raw text (D48 amendment)
    "Mullaperiyar",  # the dam that forms Periyar Lake at Thekkady (O8 as amended in D48)
    "Borra Caves",  # Araku hills landmark (owner decision, D48 amendment)
)
# term -> canonical hold-out (registry name), so drops can be counted per hold-out (D48)
HOLDOUT_OF = {
    "thanjavur": "Thanjavur", "tanjore": "Thanjavur", "thanjai": "Thanjavur",
    "brihadisvara": "Thanjavur", "brihadeeswarar": "Thanjavur",
    "chikmagalur": "Chikmagalur", "chikkamagaluru": "Chikmagalur",
    "araku valley": "Araku Valley", "araku": "Araku Valley",
    "thekkady": "Thekkady", "kumily": "Thekkady",
    "thekkadi": "Thekkady", "brihadeeswara": "Thanjavur", "mullaperiyar": "Thekkady",
    "borra caves": "Araku Valley",
}
NAME_TAGS = ("name", "name:en", "alt_name", "old_name", "official_name", "short_name")
_PERIYAR = re.compile(
    r"Periyar\s+(?:Lake|Dam|Tiger\s+Reserve|National\s+Park|Wildlife\s+Sanctuary)"
    r"|(?:National\s+Park|Wildlife\s+Sanctuary)\W+Periyar", re.IGNORECASE)


def _word_re(terms: Iterable[str]) -> re.Pattern:
    alts = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in alts) + r")(?!\w)",
                      re.IGNORECASE)


class HoldoutFilter:
    def __init__(self, centres: Iterable[tuple[float, float]] = (),
                 extra_terms: Iterable[str] | dict[str, str] = (), radius_km: float = 10.0):
        self.centres = list(centres)
        self.radius_km = radius_km
        extra = extra_terms if isinstance(extra_terms, dict) else {t: t for t in extra_terms}
        self._canon = {**HOLDOUT_OF, **{t.strip().lower(): h for t, h in extra.items()}}
        self._names = _word_re(list(DROP_NAMES) + list(extra))

    @classmethod
    def from_registry(cls, registry, radius_km: float = 10.0) -> HoldoutFilter:
        held = [e for e in registry.entries if e.holdout]
        terms = {t: e.name for e in held for t in [e.name, *e.aliases] if len(t) > 3}
        return cls([(e.lat, e.lon) for e in held], terms, radius_km)

    def text_match(self, text: str) -> tuple[str, str] | None:
        """(canonical hold-out, matched term) for the first hold-out mention in `text`, else None.
        Word boundaries apply: "Arakuvalley" does not match "Araku" (D48)."""
        m = self._names.search(text)
        if m:
            term = m.group(0)
            return self._canon.get(term.lower(), term), term
        p = _PERIYAR.search(text)
        return ("Thekkady", p.group(0)) if p else None

    def name_matches(self, text: str) -> bool:
        return self.text_match(text) is not None

    def near_holdout(self, lat: float, lon: float) -> bool:
        return any(haversine_km(lat, lon, c[0], c[1]) < self.radius_km for c in self.centres)

    def reason(self, element: dict) -> str | None:
        tags = element.get("tags") or {}
        for key in NAME_TAGS:
            value = tags.get(key)
            if value and self.name_matches(value):
                return f"name ({key})"
        point = element.get("center") or element
        lat, lon = point.get("lat"), point.get("lon")
        if lat is not None and lon is not None and self.near_holdout(lat, lon):
            return f"within {self.radius_km:g} km of a hold-out centre"
        return None

    def filter(self, elements: Iterable[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
        kept, dropped = [], []
        for el in elements:
            why = self.reason(el)
            (dropped.append((el, why)) if why else kept.append(el))
        return kept, dropped



def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


# ==== hosts ========================================================================
# The API host allowlist (config/sources.json -> api_hosts, D37). Any other host is refused.

CONFIG_PATH = REPO_ROOT / "config" / "sources.json"
_PLAIN_HOST = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?")


@dataclass(frozen=True)
class ApiHost:
    host: str
    path_prefix: str
    min_delay_s: float
    max_requests_per_run: int
    licence: str
    terms_url: str
    robots_exempt: bool
    read_timeout_s: float | None = None  # optional per-host read timeout
    path_prefixes: tuple[str, ...] = ()  # every prefix configured for this host (D43)


def load_api_hosts(path: Path = CONFIG_PATH) -> dict[str, ApiHost]:
    doc = json.loads(path.read_text(encoding="utf-8")).get("api_hosts", {})
    out = {}
    for h in doc.get("hosts", []):
        name = str(h["host"]).lower()
        if not _PLAIN_HOST.fullmatch(name):  # no markdown, scheme, path or port
            raise ValueError(f"api_hosts host must be a plain hostname, got {h['host']!r}")
        if h["host"].lower() in out:  # a further path prefix for a host already listed
            old = out[h["host"].lower()]
            out[old.host] = ApiHost(old.host, old.path_prefix, old.min_delay_s,
                                    old.max_requests_per_run, old.licence, old.terms_url,
                                    old.robots_exempt, old.read_timeout_s,
                                    old.path_prefixes + (h["path_prefix"],))
            continue
        out[h["host"].lower()] = ApiHost(
            h["host"].lower(), h["path_prefix"], float(h["min_delay_s"]),
            int(h["max_requests_per_run"]), h["licence"], h["terms_url"],
            bool(h.get("robots_exempt", False)),
            float(h["read_timeout_s"]) if h.get("read_timeout_s") else None,
            (h["path_prefix"],))
    return out


def require_host(hosts: dict[str, ApiHost], url: str) -> ApiHost:
    parts = urlsplit(url)
    host = hosts.get(parts.netloc.lower())
    if host is None:
        raise ValueError(f"host {parts.netloc!r} is not in api_hosts (D37); refused")
    if not any(parts.path.startswith(p) for p in host.path_prefixes):
        raise ValueError(
            f"path {parts.path!r} is outside {list(host.path_prefixes)} on {host.host}")
    return host


# ==== envelope =====================================================================
# Raw JSON envelopes for facts (D37): one file per request under data/raw/<source>/<place>/.

def envelope_path(raw_root: Path, source: str, place_dir: str, filename: str) -> Path:
    return Path(raw_root) / source / place_dir / filename


def make_envelope(source: str, place: str, url: str, params: dict | None, res: FetchResult,
                  licence: str, attribution: str, source_revision: dict | None = None) -> dict:
    return {
        "source": source, "place": place, "request_url": full_url(url, params),
        "params": params or {}, "retrieved_at": iso_utc(res.retrieved_at), "status": res.status,
        "from_cache": res.from_cache, "licence": licence, "attribution": attribution,
        "source_revision": source_revision, "response": res.json(),
    }


def manifest_line(env: dict, path: Path) -> dict:
    """One raw-manifest line for an envelope: url, source, place, sha256 of the file, size.
    No content."""
    data = Path(path).read_bytes()
    return {"url": env["request_url"], "source": env["source"], "place": env["place"],
            "retrieved_at": env["retrieved_at"], "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "kind": "facts_envelope"}


def write_envelope(path: Path, env: dict, manifest: Path | None = None) -> bool:
    """Write the envelope; an existing file is kept when this answer came from the cache.
    A new write adds a line to the raw manifest when one is given."""
    if path.exists() and env["from_cache"]:
        return False
    atomic_write_text(path, json.dumps(env, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    if manifest is not None:
        append_jsonl(manifest, [manifest_line(env, path)])
    return True


def backfill_manifest(raw_root: Path, source: str, manifest: Path) -> int:
    """Add manifest lines for envelopes already on disk (no network). Idempotent: a (url, sha256)
    pair already in the manifest is not added twice. Returns the number of lines added."""
    seen = set()
    if Path(manifest).exists():
        seen = {(r.get("url"), r.get("sha256")) for r in read_jsonl(manifest)}
    added = 0
    for path in sorted((Path(raw_root) / source).glob("*/*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        line = manifest_line(env, path)
        if (line["url"], line["sha256"]) in seen:
            continue
        append_jsonl(manifest, [line])
        seen.add((line["url"], line["sha256"]))
        added += 1
    return added


def place_dir(name: str) -> str:
    return slugify(name)


# ==== wikidata =====================================================================
# Wikidata entity data (CC0) through Special:EntityData, one request per place (D37).
#
# The raw envelope holds the whole entity. `select_statement` and `extract_basic` are pure helpers
# for the Phase 5 place_facts build; the fetcher does not build a table.

WIKIDATA_SOURCE = "wikidata"
WIKIDATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
PROPS = {"P625": "coordinates", "P2044": "elevation", "P131": "district", "P1435": "heritage",
         "P1082": "population", "P31": "instance_of"}


def entity_url(qid: str) -> str:
    return WIKIDATA_URL.format(qid=qid)


def _is_entity(qid: str):
    def check(text: str) -> bool:
        try:
            doc = json.loads(text)
        except ValueError:
            return False
        return isinstance(doc, dict) and qid in (doc.get("entities") or {})
    return check


def point_in_time(statement: dict) -> str | None:
    """The P585 qualifier as 'YYYY-MM-DD' (month or day may be 00), or None."""
    for q in (statement.get("qualifiers") or {}).get("P585", []):
        value = (q.get("datavalue") or {}).get("value")
        if isinstance(value, dict) and value.get("time"):
            return value["time"].lstrip("+").split("T")[0]
    return None


def select_statement(statements: list[dict]) -> tuple[dict | None, bool]:
    """Pick the statement to use for a multi-valued property. Returns (statement, ambiguous).

    Deprecated rank is ignored; preferred rank wins over normal; among the candidates the latest
    P585 wins; if none has a P585 the first listed is taken and the choice is marked ambiguous.
    """
    live = [s for s in statements if s.get("rank", "normal") != "deprecated"]
    preferred = [s for s in live if s.get("rank") == "preferred"]
    pool = preferred or live
    if not pool:
        return None, False
    dated = [(point_in_time(s), i, s) for i, s in enumerate(pool) if point_in_time(s)]
    if dated:
        best = max(dated, key=lambda t: (t[0], -t[1]))
        return best[2], False
    return pool[0], len(pool) > 1


def _value(prop: str, statement: dict):
    dv = (statement.get("mainsnak") or {}).get("datavalue") or {}
    v = dv.get("value")
    if prop == "P625" and isinstance(v, dict):
        return {"lat": v.get("latitude"), "lon": v.get("longitude")}
    if prop in ("P2044", "P1082") and isinstance(v, dict):
        try:
            return {"amount": float(v.get("amount")), "unit": v.get("unit")}
        except (TypeError, ValueError):
            return None
    if isinstance(v, dict) and "id" in v:
        return v["id"]
    return v


def source_revision(entity: dict) -> dict:
    return {"lastrevid": entity.get("lastrevid"), "modified": entity.get("modified")}


def extract_basic(entity: dict) -> dict[str, dict]:
    """property name -> {value, as_of, ambiguous}. as_of is the P585 qualifier or None."""
    out: dict[str, dict] = {}
    claims = entity.get("claims") or {}
    for prop, name in PROPS.items():
        chosen, ambiguous = select_statement(claims.get(prop, []))
        if chosen is None:
            continue
        out[name] = {"value": _value(prop, chosen), "as_of": point_in_time(chosen),
                     "ambiguous": ambiguous}
    return out


def wikidata_fetch_place(http: HttpClient, host: ApiHost, entry: PlaceRegistryEntry, raw_root: Path,
                deadline: float | None = None, manifest: Path | None = None) -> dict:
    """Fetch and store one entity. Returns a small report dict."""
    qid = entry.wikidata_id
    if not qid:
        return {"place": entry.name, "error": "no wikidata_id in the registry"}
    url = entity_url(qid)
    require_host({host.host: host}, url)
    try:
        res = http.fetch(url, check_robots=not host.robots_exempt, validate=_is_entity(qid),
                         deadline=deadline)
    except FetchError as exc:
        check_block(exc, "wikidata")
        return {"place": entry.name, "error": str(exc)[:200]}
    if not res.ok:
        return {"place": entry.name, "error": f"HTTP {res.status}"}
    entity = res.json()["entities"][qid]
    env = make_envelope(WIKIDATA_SOURCE, entry.name, url, None, res, host.licence,
                        "none required (CC0)", source_revision(entity))
    path = envelope_path(raw_root, WIKIDATA_SOURCE, place_dir(entry.name), "entity.json")
    wrote = write_envelope(path, env, manifest)
    return {"place": entry.name, "qid": qid, "wrote": wrote, "from_cache": res.from_cache,
            "properties": sorted(extract_basic(entity))}


# ==== overpass =====================================================================
# OSM features through the public Overpass instance (ODbL, D37). Not run until approved.
#
# Airports only: ONE query over the south bounding box. The per-place station queries were removed
# (D40: not in the plan, ODbL share-alike cost, O12). Hold-outs are dropped when derived lists are
# made (HoldoutFilter), not from the raw file.

OVERPASS_SOURCE = "overpass"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SOUTH_BBOX = (8.0, 74.0, 20.0, 85.0)  # south, west, north, east
OVERPASS_ATTRIBUTION = "Contains information from OpenStreetMap, made available under the ODbL"
BLOCK_STATUS = (403, 406, 429)


def airports_query() -> str:
    s, w, n, e = SOUTH_BBOX
    return (f'[out:json][timeout:60];nwr["aeroway"="aerodrome"]["iata"]({s},{w},{n},{e});'
            "out center tags;")


def overpass_validate_response(text: str) -> bool:
    try:
        doc = json.loads(text)
    except ValueError:
        return False
    if not isinstance(doc, dict) or not isinstance(doc.get("elements"), list):
        return False
    remark = str(doc.get("remark", "")).lower()
    return not ("runtime error" in remark or "timed out" in remark)


def _fetch(http: HttpClient, host: ApiHost, query: str, deadline: float | None):
    url = OVERPASS_URL
    require_host({host.host: host}, url)
    try:
        res = http.fetch(url, {"data": query}, check_robots=not host.robots_exempt,
                         validate=overpass_validate_response, deadline=deadline)
    except FetchError as exc:
        check_block(exc, "overpass")
        if exc.status in BLOCK_STATUS:  # 406 is treated like 429 (usage policy)
            raise BlockDetected(f"overpass: HTTP {exc.status}") from exc
        raise
    return url, res


def fetch_airports(http: HttpClient, host: ApiHost, raw_root: Path,
                   deadline: float | None = None, manifest: Path | None = None) -> dict:
    query = airports_query()
    try:
        url, res = _fetch(http, host, query, deadline)
    except FetchError as exc:  # a block already raised BlockDetected; this is a plain failure
        return {"place": "_south", "error": str(exc)[:200]}
    env = make_envelope(OVERPASS_SOURCE, "_south", url, {"data": query}, res, host.licence,
                        OVERPASS_ATTRIBUTION)
    path = envelope_path(raw_root, OVERPASS_SOURCE, "_south", "airports.json")
    wrote = write_envelope(path, env, manifest)
    return {"place": "_south", "wrote": wrote, "elements": len(res.json()["elements"])}


# ==== openmeteo ====================================================================
# Open-Meteo ERA5 archive (CC-BY 4.0, free tier non-commercial as resolved by the owner, D37).
#
# One place per request, sequential. Not run until approved. Monthly figures are derived by us from
# daily reanalysis values and are labelled as such (docs/facts-layer.md 2.2).

OPEN_METEO_SOURCE = "open_meteo"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2024-12-31"
EXPECTED_DAYS = 3653  # 2015-01-01 to 2024-12-31 inclusive
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum"
OPEN_METEO_ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0); ERA5 reanalysis"


def request_params(lat: float, lon: float, elevation_m: float | None = None) -> dict:
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": START,
              "end_date": END, "daily": DAILY, "timezone": "Asia/Kolkata"}
    if elevation_m is not None:
        params["elevation"] = round(elevation_m, 1)
    return params


def open_meteo_validate_response(text: str) -> bool:
    try:
        doc = json.loads(text)
    except ValueError:
        return False
    daily = doc.get("daily") if isinstance(doc, dict) else None
    return (isinstance(daily, dict) and isinstance(daily.get("time"), list)
            and "error" not in doc)


def open_meteo_fetch_place(http: HttpClient, host: ApiHost, entry: PlaceRegistryEntry,
                           raw_root: Path, elevation_m: float | None = None,
                           deadline: float | None = None, manifest: Path | None = None,
                           filename: str = "archive_2015_2024.json") -> dict:
    params = request_params(entry.lat, entry.lon, elevation_m)
    require_host({host.host: host}, OPEN_METEO_URL)
    try:
        res = http.fetch(OPEN_METEO_URL, params, check_robots=not host.robots_exempt,
                         validate=open_meteo_validate_response, deadline=deadline)
    except FetchError as exc:
        check_block(exc, "open_meteo")
        return {"place": entry.name, "error": str(exc)[:200]}
    env = make_envelope(OPEN_METEO_SOURCE, entry.name, OPEN_METEO_URL, params, res, host.licence,
                        OPEN_METEO_ATTRIBUTION)
    path = envelope_path(raw_root, OPEN_METEO_SOURCE, place_dir(entry.name), filename)
    wrote = write_envelope(path, env, manifest)
    daily = res.json()["daily"]
    days = len(daily["time"])
    nulls = sum(v is None for k in ("temperature_2m_max", "temperature_2m_min",
                                    "precipitation_sum") for v in daily.get(k, []))
    return {"place": entry.name, "status": res.status, "from_cache": res.from_cache,
            "wrote": wrote, "days": days,
            "nulls": nulls, "check_ok": days == EXPECTED_DAYS and nulls == 0,
            "bytes": len(res.text.encode("utf-8")), "elevation_param": elevation_m}
