"""Phase 2: clean, chunk and tag raw records into BaseChunks (D48). No network, no file layout.

`process_place()` takes one place's raw records and returns its chunks, so the live path (data/live,
D17/D20) uses the same code as the batch run. Cleaning is a pure per-record function: the learned
boilerplate lines come from the committed rule file (config/boilerplate.json), never from the
records being processed. Batch-only steps (exact dedupe across places, near-duplicate removal,
attribution, statistics, reconciliation) are separate functions; scripts/process.py drives them.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from calendar import monthrange
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from src.collect.run import SOURCE_TYPE, passes_rule
from src.common.schemas import BaseChunk, make_chunk_id

TokenCounter = Callable[[str], int]

# ==== tokens ======================================================================
# The default counter is a dependency-free proxy for subword tokens. With TOKENIZER_DIR set, the
# local bge-m3 tokenizer.json is used (special tokens excluded). The freeze needs the real one.

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
PROXY_COUNTER = "regex-proxy"


def proxy_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def load_counter(tokenizer_dir: Path | None) -> tuple[TokenCounter, str]:
    """(counter, name). Never downloads: a missing tokenizer.json is an error."""
    if tokenizer_dir is None:
        return proxy_count, PROXY_COUNTER
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"TOKENIZER_DIR is set but {path} does not exist (nothing is downloaded, D48)")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return count, f"tokenizer.json sha256:{digest}"


# ==== clean =======================================================================
# Fixed rules (D48 4) plus the learned per-host lines from config/boilerplate.json.

SHARE_LINES = frozenset({"Facebook", "X", "Pinterest", "WhatsApp", "Linkedin", "Telegram"})
_FOOTER = re.compile(r"^(?:Recommend(?:ed)?\b|Read more\b)", re.IGNORECASE)
FOOTER_TAIL_CHARS = 400  # a "Recommend you read ..." line with less text after it ends the post
LINK_TITLE_CHARS = 80  # mid-post "Read more:" drops itself and the linked title after it
SECTION_BLOCKLIST = frozenset({
    "references", "external links", "see also", "further reading", "notes", "footnotes",
    "bibliography", "citations", "sources", "gallery"})
TAIL_LINES = frozenset({"Notes", "Bibliography", "References", "Endnotes", "Footnotes"})
_HEADER = re.compile(r"^(=+)\s*(.*?)\s*=+$")
_CITE = re.compile(r"\[\d+\]")


@dataclass
class Cleaned:
    text: str
    header_offsets: frozenset[int]
    removed: Counter = field(default_factory=Counter)  # rule -> characters removed


def host_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _learned(rules: dict | None, url: str) -> frozenset[str]:
    hosts = (rules or {}).get("hosts", {})
    return frozenset(x["line"] for x in hosts.get(host_of(url), {}).get("lines", []))


def _lines(body: str) -> list[str]:
    body = unicodedata.normalize("NFC", body).replace(" ", " ").replace("\r\n", "\n")
    out = []
    for line in body.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            out.append(line)
    return out


def _cut(lines: list[str], start: int, end: int, rule: str, removed: Counter) -> list[str]:
    removed[rule] += sum(len(x) + 1 for x in lines[start:end])
    return lines[:start] + lines[end:]


def _blog_rules(lines: list[str], removed: Counter) -> list[str]:
    i = 0
    while i < len(lines) and lines[i] in SHARE_LINES:
        i += 1
    lines = _cut(lines, 0, i, "share_header", removed)
    j = 0
    while j < len(lines):
        if _FOOTER.match(lines[j]):
            tail = sum(len(x) for x in lines[j + 1:])
            if tail <= FOOTER_TAIL_CHARS:
                lines = _cut(lines, j, len(lines), "related_footer", removed)
                break
            k = j + 1
            if k < len(lines) and len(lines[k]) < LINK_TITLE_CHARS:
                k += 1
            lines = _cut(lines, j, k, "read_more_link", removed)
            continue
        j += 1
    kept = []
    for line in lines:
        if _CREDIT.match(line):
            removed["image_credit"] += len(line) + 1
        else:
            kept.append(line)
    return _join_inline_links(kept, removed)


_CREDIT = re.compile(
    r"^(?:/|(?:image|photo)\s+(?:courtesy|source)"
    r"|(?:(?:image|photo)\s+(?:courtesy|source)\s*[-–:]?\s*)?shutterstock(?:\.com)?\.?)$",
    re.IGNORECASE)


_CONTINUES = re.compile(r"^[a-z,.;:!?)%]")
INLINE_LINK_CHARS = 40


def _join_inline_links(lines: list[str], removed: Counter) -> list[str]:
    """The page-to-text step put inline link text on its own line ("The whole / Bangalore /
    international airport"). A line is joined to the previous one when it starts with a lowercase
    letter or punctuation, or when it is short and the line after it does (a link fragment)."""
    out: list[str] = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        fragment = len(line) <= INLINE_LINK_CHARS and bool(_CONTINUES.match(nxt))
        if out and (_CONTINUES.match(line) or fragment):
            glue = "" if line[0] in ",.;:!?)%" else " "
            out[-1] = out[-1] + glue + line
            removed["inline_link_joins"] += 1  # a count of joins, not characters
        else:
            out.append(line)
    return out


def _wiki_sections(lines: list[str], removed: Counter) -> tuple[list[str], set[int]]:
    """Drop blocklisted sections, turn `== Title ==` into a plain `Title` line. Returns the lines
    and the indices of header lines."""
    out: list[str] = []
    levels: list[int | None] = []
    skip_level = None
    for line in lines:
        m = _HEADER.match(line)
        if m:
            level = len(m.group(1))
            if skip_level is not None and level > skip_level:
                removed["blocked_section"] += len(line) + 1
                continue
            skip_level = None
            if m.group(2).lower() in SECTION_BLOCKLIST:
                skip_level = level
                removed["blocked_section"] += len(line) + 1
                continue
            out.append(m.group(2))
            levels.append(level)
        elif skip_level is not None:
            removed["blocked_section"] += len(line) + 1
        else:
            out.append(line)
            levels.append(None)
    # a header with no body (next line is a header at the same or a higher level, or the end)
    keep = []
    for i, lvl in enumerate(levels):
        if lvl is not None:
            nxt = next((levels[k] for k in range(i + 1, len(levels))), "end")
            if nxt == "end" or (nxt is not None and nxt <= lvl):
                removed["empty_header"] += len(out[i]) + 1
                continue
        keep.append(i)
    return [out[i] for i in keep], {n for n, i in enumerate(keep) if levels[i] is not None}


def clean_record(rec: dict, rules: dict | None) -> Cleaned:
    """Pure: the same record and rule file always give the same text."""
    removed: Counter = Counter()
    lines = _lines(rec["body"])
    headers: set[int] = set()
    source = rec["source"]
    if source in ("wikipedia", "wikivoyage"):
        lines, headers = _wiki_sections(lines, removed)
    if source == "sahapedia":
        cut = next((i for i, x in enumerate(lines) if x in TAIL_LINES), None)
        if cut is not None:
            lines = _cut(lines, cut, len(lines), "notes_bibliography", removed)
    if source == "blog":
        lines = _blog_rules(lines, removed)
    learned = _learned(rules, rec["url"])
    if learned:
        kept, kept_headers = [], set()
        for i, x in enumerate(lines):
            if x in learned and i not in headers:
                removed["learned_line"] += len(x) + 1
                continue
            if i in headers:
                kept_headers.add(len(kept))
            kept.append(x)
        lines, headers = kept, kept_headers
    final, final_headers = [], set()
    for i, x in enumerate(lines):
        y = _CITE.sub("", x).strip()
        removed["citation_marker"] += len(x) - len(y)
        if not y:
            continue
        if i in headers:
            final_headers.add(len(final))
        final.append(y)
    offsets, pos = [], 0
    for x in final:
        offsets.append(pos)
        pos += len(x) + 1
    return Cleaned("\n".join(final), frozenset(offsets[i] for i in final_headers), removed)


def learn_boilerplate(records: Iterable[dict], hosts: Iterable[str], *, min_records: int = 3,
                      max_chars: int = 120, rejected: dict[str, set[str]] | None = None) -> dict:
    """Lines under `max_chars` that occur in >= `min_records` records (distinct urls) of a listed
    host, after the fixed rules; parsed section headers are exempt. A candidate listed in
    `rejected` (owner review, kept from the existing file) is reported but never applied."""
    hosts = sorted(set(hosts))
    rejected = rejected or {}
    seen: dict[str, Counter] = {h: Counter() for h in hosts}
    urls: set[str] = set()
    for rec in sorted(records, key=lambda r: r["url"]):
        h = host_of(rec["url"])
        if h not in seen or rec["url"] in urls:  # a post stored under two places counts once
            continue
        urls.add(rec["url"])
        c = clean_record(rec, None)
        starts, pos = [], 0
        for line in c.text.split("\n"):
            starts.append((pos, line))
            pos += len(line) + 1
        seen[h].update({line for s, line in starts
                        if s not in c.header_offsets and len(line) < max_chars})
    hosts_doc = {}
    for h in hosts:
        found = [(line, n) for line, n in sorted(seen[h].items(), key=lambda x: (-x[1], x[0]))
                 if n >= min_records]
        no = rejected.get(h, set())
        hosts_doc[h] = {
            "lines": [{"line": x, "records": n} for x, n in found if x not in no],
            "rejected": [{"line": x, "records": n} for x, n in found if x in no]}
    return {
        "_doc": "Learned boilerplate lines (D48). Written only by "
                "`python scripts/process.py learn`; cleaning applies `lines` and never learns. "
                "`rejected` = candidates the owner review keeps as content; never applied.",
        "min_records": min_records, "max_chars": max_chars, "hosts": hosts_doc,
    }


# ==== chunk =======================================================================
# Sentences packed greedily (D48 3): close at >= TARGET or when the next unit would pass MAX; a
# section header closes a chunk of >= MIN and starts a new one without overlap; overlap = whole
# trailing sentences up to OVERLAP tokens, counted inside MAX.

TARGET, MIN_TOKENS, MAX_TOKENS, OVERLAP, TAIL = 250, 200, 300, 40, 100
ABBREVIATIONS = frozenset({
    "dr.", "st.", "mr.", "mrs.", "ms.", "mt.", "rs.", "no.", "nos.", "jr.", "sr.", "vs.", "etc.",
    "e.g.", "i.e.", "a.m.", "p.m.", "approx.", "sq.", "ca.", "prof.", "govt.",
    "dept.", "rd.", "fig.", "op.", "cf.", "viz.", "sri.", "smt."})
_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+")
_INITIAL = re.compile(r"^[A-Z]\.$")


@dataclass(frozen=True)
class Unit:
    start: int
    end: int
    header: bool = False


def _sentences(line: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in _BOUNDARY.finditer(line):
        nxt = line[m.end():m.end() + 1]
        token = line[:m.start() + 1].rsplit(" ", 1)[-1].lstrip("(\"'[")
        if not nxt or nxt.islower() or token.lower() in ABBREVIATIONS or _INITIAL.match(token):
            continue
        end = m.end()
        while end > start and line[end - 1].isspace():
            end -= 1
        out.append((start, end))
        start = m.end()
    if start < len(line):
        out.append((start, len(line)))
    return out


def split_units(text: str, header_offsets: frozenset[int]) -> list[Unit]:
    units, pos = [], 0
    for line in text.split("\n"):
        start = pos
        pos += len(line) + 1
        if start in header_offsets:
            units.append(Unit(start, start + len(line), True))
            continue
        units.extend(Unit(start + s, start + e) for s, e in _sentences(line))
    return units


def _split_oversize(text: str, u: Unit, count: TokenCounter, max_tokens: int) -> list[Unit]:
    words = [(m.start() + u.start, m.end() + u.start)
             for m in re.finditer(r"\S+", text[u.start:u.end])]
    out, s, last_end = [], words[0][0], words[0][1]
    for ws, we in words[1:]:
        if count(text[s:we]) > max_tokens:
            out.append(Unit(s, last_end))
            s = ws
        last_end = we
    out.append(Unit(s, last_end))
    return out


@dataclass
class ChunkStats:
    oversize_splits: int = 0  # units over MAX split on words
    merged_tails: int = 0
    short_tails: int = 0  # tail under TAIL kept alone because merging would pass MAX


def chunk_text(text: str, header_offsets: frozenset[int], count: TokenCounter
               ) -> tuple[list[tuple[int, int]], ChunkStats]:
    """Character spans (start, end) of the chunks of one cleaned text."""
    st = ChunkStats()
    units: list[Unit] = []
    for u in split_units(text, header_offsets):
        if not u.header and count(text[u.start:u.end]) > MAX_TOKENS:
            st.oversize_splits += 1
            units.extend(_split_oversize(text, u, count, MAX_TOKENS))
        else:
            units.append(u)
    spans: list[tuple[int, int]] = []
    cur: list[Unit] = []
    fresh = 0  # units in `cur` that are not overlap
    for u in units:
        if cur:
            now = count(text[cur[0].start:cur[-1].end])
            if u.header and now >= MIN_TOKENS:
                spans.append((cur[0].start, cur[-1].end))
                cur, fresh = [], 0
            elif now >= TARGET or count(text[cur[0].start:u.end]) > MAX_TOKENS:
                spans.append((cur[0].start, cur[-1].end))
                ov: list[Unit] = []
                if not u.header:
                    for prev in reversed(cur[1:]):  # a proper suffix only
                        if (count(text[prev.start:cur[-1].end]) > OVERLAP
                                or count(text[prev.start:u.end]) > MAX_TOKENS):
                            break
                        ov.insert(0, prev)
                cur, fresh = ov, 0
        cur.append(u)
        fresh += 1
    if cur and fresh:
        spans.append((cur[0].start, cur[-1].end))
    if len(spans) >= 2 and count(text[spans[-1][0]:spans[-1][1]]) < TAIL:
        if count(text[spans[-2][0]:spans[-1][1]]) <= MAX_TOKENS:
            spans[-2:] = [(spans[-2][0], spans[-1][1])]
            st.merged_tails += 1
        else:
            st.short_tails += 1
    return spans, st


# ==== holdout & tag ===============================================================

WIKI_REVISION_SOURCES = frozenset({"wikipedia", "wikivoyage"})


def date_fields(rec: dict) -> dict:
    """D48 1: publish date -> published; wiki API revision -> revised; sitemap -> label."""
    if rec.get("publish_date"):
        return {"date": rec["publish_date"], "date_kind": "published", "revised_date": None}
    if rec["source"] in WIKI_REVISION_SOURCES and rec.get("lastmod"):
        return {"date": None, "date_kind": "revised", "revised_date": rec["lastmod"]}
    if rec.get("lastmod"):
        return {"date": None, "date_kind": "sitemap_lastmod", "revised_date": None}
    return {"date": None, "date_kind": "none", "revised_date": None}


def record_holdout(rec: dict, holdout) -> dict | None:
    """Hold-out named in the URL (non-alphanumerics read as spaces) or the title."""
    for where, text in (("url", re.sub(r"[^A-Za-z0-9]+", " ", rec["url"])),
                        ("title", rec.get("title") or "")):
        hit = holdout.text_match(text)
        if hit:
            return {"where": where, "holdout": hit[0], "term": hit[1]}
    return None


@dataclass
class PlaceReport:
    place: str
    records_in: int = 0
    record_drops: list[dict] = field(default_factory=list)
    chunks_created: int = 0
    chunk_drops: list[dict] = field(default_factory=list)
    removed: Counter = field(default_factory=Counter)
    chunking: ChunkStats = field(default_factory=ChunkStats)


def process_place(records: list[dict], entry, holdout, rules: dict | None,
                  count: TokenCounter) -> tuple[list[tuple[int, dict]], PlaceReport]:
    """One place's raw records -> [(offset, chunk dict)], hold-out drops applied (D48 6)."""
    if getattr(entry, "holdout", False):
        raise ValueError(f"{entry.name} is a hold-out and is never chunked")
    rep = PlaceReport(entry.name, records_in=len(records))
    out: list[tuple[int, dict]] = []
    for rec in sorted(records, key=lambda r: (r["source"], r["url"])):
        if rec["place"] != entry.name:
            raise ValueError(f"record place {rec['place']!r} != {entry.name!r}: {rec['url']}")
        hit = record_holdout(rec, holdout)
        if hit:
            rep.record_drops.append({"level": "record", "url": rec["url"], "place": entry.name,
                                     "source": rec["source"], **hit})
            continue
        c = clean_record(rec, rules)
        rep.removed.update(c.removed)
        spans, st = chunk_text(c.text, c.header_offsets, count)
        for k in ("oversize_splits", "merged_tails", "short_tails"):
            setattr(rep.chunking, k, getattr(rep.chunking, k) + getattr(st, k))
        for start, end in spans:
            rep.chunks_created += 1
            text = c.text[start:end]
            cid = make_chunk_id(rec["url"], start)
            hit = holdout.text_match(text)
            if hit:
                rep.chunk_drops.append({"level": "chunk", "chunk_id": cid, "url": rec["url"],
                                        "place": entry.name, "source": rec["source"],
                                        "holdout": hit[0], "term": hit[1]})
                continue
            chunk = BaseChunk(chunk_id=cid, text=text, source_type=SOURCE_TYPE[rec["source"]],
                              url=rec["url"], state=entry.state, place=entry.name,
                              **date_fields(rec))
            out.append((start, chunk.model_dump(mode="json")))
    return out, rep


# ==== dedupe ======================================================================

def dedupe_records(records: list[dict], order: dict[str, int], rules: dict | None
                   ) -> tuple[list[dict], list[dict]]:
    """Exact duplicates by cleaned-text sha256 across all places. The copy under the earlier
    registry place is kept (D48 5); every dropped copy is reported with both places."""
    kept, dups, first = [], [], {}
    for rec in sorted(records, key=lambda r: (order[r["place"]], r["source"], r["url"])):
        key = hashlib.sha256(clean_record(rec, rules).text.encode("utf-8")).hexdigest()
        if key in first:
            k = first[key]
            dups.append({"level": "exact_duplicate", "url": rec["url"], "place": rec["place"],
                         "source": rec["source"], "kept_url": k["url"], "kept_place": k["place"]})
            continue
        first[key] = rec
        kept.append(rec)
    return kept, dups


SHINGLE = 5
NEAR_DUP_THRESHOLD = 0.8


def _shingles(text: str) -> frozenset:
    w = re.findall(r"\w+", text.lower())
    if len(w) < SHINGLE:
        return frozenset({tuple(w)})
    return frozenset(tuple(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1))


def near_dups(chunks: list[dict], threshold: float = NEAR_DUP_THRESHOLD
              ) -> tuple[list[dict], list[dict]]:
    """Drop a chunk whose 5-word-shingle Jaccard with an earlier KEPT chunk is >= threshold.
    `chunks` must already be in output order (place, url, offset)."""
    index: dict[tuple, list[int]] = defaultdict(list)
    sets: list[frozenset] = []
    kept, drops = [], []
    for c in chunks:
        sh = _shingles(c["text"])
        shared: Counter = Counter()
        for s in sh:
            for i in index[s]:
                shared[i] += 1
        best, best_i = 0.0, None
        for i, n in sorted(shared.items()):
            j = n / (len(sh) + len(sets[i]) - n)
            if j > best:
                best, best_i = j, i
        if best_i is not None and best >= threshold:
            drops.append({"level": "near_duplicate", "chunk_id": c["chunk_id"], "url": c["url"],
                          "place": c["place"], "source_type": c["source_type"],
                          "kept_chunk_id": kept[best_i]["chunk_id"], "jaccard": round(best, 4)})
            continue
        for s in sh:
            index[s].append(len(kept))
        sets.append(sh)
        kept.append(c)
    return kept, drops


# ==== attribution =================================================================

def attribution(records: list[dict], chunks: list[dict]) -> dict:
    """url -> licence and provenance for every url that has at least one chunk (D48 8)."""
    used = {c["url"] for c in chunks}
    out = {}
    for r in records:
        if r["url"] in used:
            out[r["url"]] = {"title": r.get("title") or "", "source": r["source"],
                             "license": r.get("license") or "", "place": r["place"],
                             "revid": (r.get("extra") or {}).get("revid"),
                             "lastmod": r.get("lastmod"), "publish_date": r.get("publish_date")}
    return dict(sorted(out.items()))


# ==== stats =======================================================================

ESTIMATE = {"chunks": 1943, "low": 1619, "high": 2429, "source": "D39 (ii), LOG.md 2026-10-19"}
SCAN_STEMS = ("brihad", "tanj", "thanj", "araku", "thekk", "kumil", "chikm", "chikk", "periyar")


def months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return date(y, m + 1, min(d.day, monthrange(y, m + 1)[1]))


class ReconcileError(RuntimeError):
    pass


def reconcile(n_raw: int, n_kept: int, n_exact: int, n_record_holdout: int,
              created: int, out: int, n_chunk_holdout: int, n_near: int) -> list[str]:
    rows = [
        f"records: {n_raw} in = {n_kept} kept + {n_exact} exact duplicates "
        f"+ {n_record_holdout} hold-out (url/title)",
        f"chunks: {created} created = {out} out + {n_chunk_holdout} hold-out (text) "
        f"+ {n_near} near duplicates",
    ]
    if n_raw != n_kept + n_exact + n_record_holdout or created != out + n_chunk_holdout + n_near:
        raise ReconcileError("reconciliation failed:\n" + "\n".join(rows))
    return rows


def _pct(values: list[int], q: float) -> int:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else 0


def corpus_stats(chunks: list[dict], reports: list[PlaceReport], drops: list[dict],
                 count: TokenCounter, counter_name: str, ref_date: date,
                 d23_raw: dict[str, bool], places: list[str]) -> dict:
    tokens = [count(c["text"]) for c in chunks]
    by = lambda key: dict(sorted(Counter(key(c) for c in chunks).items()))  # noqa: E731
    window = months_before(ref_date, 18)
    published = [c for c in chunks if c["date_kind"] == "published"]
    fresh = [c for c in published if c["date"] >= window.isoformat()]
    holdout = Counter((d["holdout"], d["source"], d["level"]) for d in drops
                      if d["level"] in ("record", "chunk"))
    removed: Counter = Counter()
    chunking = Counter()
    for r in reports:
        removed.update(r.removed)
        for k in ("oversize_splits", "merged_tails", "short_tails"):
            chunking[k] += getattr(r.chunking, k)
    scan = {s: sum(s in c["text"].lower() for c in chunks) for s in SCAN_STEMS}
    d23 = {}
    for p in places:
        types = sorted({c["source_type"] for c in chunks if c["place"] == p})
        d23[p] = {"types_in_chunks": types, "passes_chunks": passes_rule(set(types)),
                  "passes_raw": d23_raw.get(p, False)}
    per_place = defaultdict(Counter)
    for c in chunks:
        per_place[c["place"]][c["source_type"]] += 1
    return {
        "counter": counter_name,
        "params": {"target": TARGET, "min": MIN_TOKENS, "max_incl_overlap": MAX_TOKENS,
                   "overlap_max": OVERLAP, "tail_merge_below": TAIL, "shingle": SHINGLE,
                   "near_dup_threshold": NEAR_DUP_THRESHOLD},
        "chunks": len(chunks),
        "estimate": {**ESTIMATE, "difference": len(chunks) - ESTIMATE["chunks"]},
        "tokens": {"min": min(tokens, default=0), "p50": _pct(tokens, 0.5),
                   "p95": _pct(tokens, 0.95), "max": max(tokens, default=0),
                   "under_min": sum(t < MIN_TOKENS for t in tokens),
                   "over_max": sum(t > MAX_TOKENS for t in tokens)},
        "per_source_type": by(lambda c: c["source_type"]),
        "per_state": by(lambda c: c["state"]),
        "per_place": {p: dict(sorted(per_place[p].items())) for p in sorted(per_place)},
        "per_published_year": by(lambda c: c["date"][:4] if c["date"] else "none"),
        "per_revised_year": by(lambda c: c["revised_date"][:4] if c["revised_date"] else "none"),
        "date_kind": by(lambda c: c["date_kind"]),
        "null_date_share": round(sum(c["date"] is None for c in chunks) / max(1, len(chunks)), 4),
        "freshness": {"reference_date": ref_date.isoformat(), "window_start": window.isoformat(),
                      "published_in_window": len(fresh),
                      "share": round(len(fresh) / max(1, len(chunks)), 4)},
        "removed_chars": dict(sorted(removed.items())),
        "chunking": dict(sorted(chunking.items())),
        "drops": dict(sorted(Counter(d["level"] for d in drops).items())),
        "holdout_drops": [{"holdout": h, "source": s, "level": lv, "n": n}
                          for (h, s, lv), n in sorted(holdout.items())],
        "substring_scan": scan,
        "d23_dry_run": d23,
    }
