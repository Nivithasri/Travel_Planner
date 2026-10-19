"""Phase 2 evidence (read-only, offline): every number the Phase 2 review asks for, from the
files on disk. Writes nothing. Run after `python scripts/process.py --write`.

    python scripts/phase2_evidence.py <section> [...]
    sections: manifest holdout blogdates rejoin credits splitter short neardup d23 attribution all
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.process.process as P  # noqa: E402
from src.collect.run import CORE_REGISTRY, RAW_ROOT, place_coverage  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402
from src.places.places import Registry  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
MANIFEST = REPO_ROOT / "data" / "manifests" / "raw_manifest.jsonl"


def records() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RAW_ROOT / "*" / "*" / "records.jsonl"))):
        out += [json.loads(line) for line in open(f, encoding="utf-8")]
    return out


def chunks() -> list[dict]:
    return [json.loads(line) for line in open(PROCESSED / "chunks.jsonl", encoding="utf-8")]


def rules() -> dict:
    return json.loads((REPO_ROOT / "config" / "boilerplate.json").read_text(encoding="utf-8"))


# ---- manifest


def manifest() -> None:
    lines = [json.loads(x) for x in open(MANIFEST, encoding="utf-8")]
    recs = records()
    by_url = collections.defaultdict(list)
    for r in recs:
        by_url[(r["source"], r["url"])].append(r)
    env_files = {}
    for src in ("wikidata", "open_meteo", "overpass"):
        for f in (RAW_ROOT / src).rglob("*.json"):
            env_files[hashlib.sha256(f.read_bytes()).hexdigest()] = f
    verified, mismatch, youtube, withdrawn, matched_recs, matched_envs = 0, [], 0, 0, set(), set()
    for m in lines:
        if m.get("event") == "withdrawn":
            withdrawn += 1
        elif m["source"] == "youtube":
            youtube += 1
        elif m.get("kind") == "facts_envelope":
            f = env_files.get(m["sha256"])
            if f is not None and f.stat().st_size == m["bytes"]:
                verified += 1
                matched_envs.add(m["sha256"])
            else:
                mismatch.append(m["url"])
        else:
            hit = [r for r in by_url[(m["source"], m["url"])] if r["content_sha256"] == m["sha256"]
                   and len(r["body"].encode("utf-8")) == m["bytes"]
                   and hashlib.sha256(r["body"].encode("utf-8")).hexdigest() == m["sha256"]]
            if hit:
                verified += 1
                matched_recs.update((r["source"], r["url"], r["place"]) for r in hit)
            else:
                mismatch.append(m["url"])
    unlisted_recs = [r["url"] for r in recs
                     if (r["source"], r["url"], r["place"]) not in matched_recs]
    unlisted_envs = [str(f) for h, f in env_files.items() if h not in matched_envs]
    print(f"manifest lines {len(lines)}: verified {verified} (sha256 + bytes against data/raw), "
          f"mismatch {len(mismatch)}, youtube {youtube}, withdrawn {withdrawn}")
    print(f"records on disk {len(recs)}, envelopes on disk {len(env_files)}; unlisted records "
          f"{len(unlisted_recs)}, unlisted envelopes {len(unlisted_envs)}")
    for u in mismatch + unlisted_recs + unlisted_envs:
        print("  ", u)


# ---- hold-out scans (independent of HoldoutFilter)

TERMS = ["Thekkady", "Thekkadi", "Kumily", "Periyar", "Mullaperiyar", "Araku", "Thanjavur",
         "Tanjavur", "Tanjore", "Thanjai", "Brihadisvara", "Brihadeeswara", "Brihadeshwara",
         "Chikmagalur", "Chikkamagaluru", "Chikmagaluru"]
LANDMARKS = ["Mullayanagiri", "Baba Budan", "Kemmangundi", "Kudremukh", "Borra Caves",
             "Saraswathi Mahal", "Big Temple"]


def lev(a: str, b: str, cap: int = 2) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def holdout() -> None:
    cs = chunks()
    att = json.loads((PROCESSED / "attribution.json").read_text(encoding="utf-8"))
    print(f"SCOPE: text of {len(cs)} chunks in data/processed/chunks.jsonl; url and title of the "
          f"{len(att)} urls in data/processed/attribution.json (= every url with a chunk). "
          "Case-insensitive substring regex, no word boundaries, no HoldoutFilter.")
    print("TERMS:", ", ".join(TERMS))
    for t in TERMS:
        rx = re.compile(re.escape(t), re.IGNORECASE)
        text_hits = [(c, m) for c in cs for m in rx.finditer(c["text"])]
        url_hits = [u for u in att if rx.search(u)]
        title_hits = [u for u, a in att.items() if rx.search(a["title"])]
        print(f"{t}: text {len(text_hits)} (chunks {len({c['chunk_id'] for c, _ in text_hits})}), "
              f"url {len(url_hits)}, title {len(title_hits)}")
        for c, m in text_hits:
            ctx = c["text"][max(0, m.start() - 45):m.end() + 35].replace("\n", " / ")
            print(f"    {c['place']} {c['source_type']}: ...{ctx}...")
    names = sorted({t.lower() for t in TERMS} | {"araku valley", "thanjai"}
                   | {a.lower() for e in Registry.load(CORE_REGISTRY).entries if e.holdout
                      for a in [e.name, *e.aliases]})
    single = sorted({w for n in names for w in re.split(r"[\s,]+", n) if len(w) > 3
                     and w not in {"valley"}})
    vocab = collections.Counter(w for c in cs for w in re.findall(r"\w+", c["text"].lower()))
    print(f"\nFUZZY: every distinct lower-cased token of the chunk text ({len(vocab)} tokens) "
          f"within Levenshtein distance <= 2 of a hold-out name/alias word ({', '.join(single)}):")
    for n in single:
        near = sorted((w, k) for w, k in vocab.items() if lev(w, n) <= 2)
        print(f"  {n}: " + (", ".join(f"{w} x{k}" for w, k in near) or "none"))
    print("\nRAW Chikmagal*/Chikkamagal* (case-insensitive) per record over body+title+url of all "
          "97 records in data/raw/*/*/records.jsonl:")
    rx = re.compile(r"chik+a?magal\w*", re.IGNORECASE)
    total = 0
    for r in records():
        n = len(rx.findall(r["body"] + " " + (r.get("title") or "") + " " + r["url"]))
        if n:
            total += n
            print(f"    {r['source']} {r['place']} {r['url']}: {n}")
    drops = [json.loads(x) for x in open(PROCESSED / "drops.jsonl", encoding="utf-8")]
    chik = sum(d.get("holdout") == "Chikmagalur" for d in drops)
    print(f"  raw occurrences {total}; Chikmagalur drops in drops.jsonl {chik}: "
          f"{'consistent' if (total == 0) == (chik == 0) else 'CHECK'}")
    print("\nLANDMARKS (owner list, UNVERIFIED; report only, nothing dropped), same scope:")
    for t in LANDMARKS:
        rx = re.compile(re.escape(t), re.IGNORECASE)
        hits = [(c, m) for c in cs for m in rx.finditer(c["text"])]
        print(f"  {t}: text {len(hits)}, url {sum(bool(rx.search(u)) for u in att)}, "
              f"title {sum(bool(rx.search(a['title'])) for a in att.values())}")
        for c, m in hits:
            ctx = c["text"][max(0, m.start() - 60):m.end() + 40].replace("\n", " / ")
            print(f"    {c['place']} {c['source_type']} {c['chunk_id']}: ...{ctx}...")


# ---- blog dates


def blogdates() -> None:
    from bs4 import BeautifulSoup

    blogs = [r for r in records() if r["source"] == "blog"]
    urls = {}
    for r in blogs:
        urls.setdefault(r["url"], r)
    print(f"SCOPE: {len(blogs)} blog records, {len(urls)} distinct urls; publish_date missing: "
          f"{sum(not r['publish_date'] for r in blogs)}")
    html = {}
    for p in glob.glob(str(REPO_ROOT / ".cache" / "http" / "*" / "*.json")):
        d = json.load(open(p, encoding="utf-8"))
        u = (d.get("inputs") or {}).get("url") if isinstance(d.get("inputs"), dict) else None
        if u and "inditales.com" in u and isinstance(d.get("value"), dict):
            html[u.replace("://www.", "://")] = d["value"].get("text") or ""
    print("5 earliest (distinct urls), with every date field in the cached HTML:")
    for r in sorted(urls.values(), key=lambda r: r["publish_date"])[:5]:
        s = BeautifulSoup(html.get(r["url"], ""), "lxml")
        meta = s.find("meta", property="article:published_time")
        t = s.find("time", attrs={"datetime": True})
        ld = re.findall(r'"datePublished":"([^"]+)"', html.get(r["url"], ""))
        gen = s.find("meta", attrs={"name": "generator"})
        print(f"  {r['publish_date']} {r['url']}\n     used: meta article:published_time="
              f"{meta.get('content') if meta else None}; JSON-LD datePublished={sorted(set(ld))}; "
              f"<time datetime>={t.get('datetime') if t else None}; generator="
              f"{gen.get('content') if gen else None}; yoast="
              f"{'yoast' in html.get(r['url'], '').lower()}")
    print("blog records per publish year:",
          dict(sorted(collections.Counter(r["publish_date"][:4] for r in blogs).items())))
    print("distinct urls per publish year:",
          dict(sorted(collections.Counter(r["publish_date"][:4] for r in urls.values()).items())))
    print("5 most frequent publish_date values (distinct urls):",
          collections.Counter(r["publish_date"] for r in urls.values()).most_common(5))


# ---- cleaning rules


def _pre_join(rec: dict) -> list[str]:
    orig = P._join_inline_links
    P._join_inline_links = lambda lines, removed: lines
    try:
        return P._blog_rules(P._lines(rec["body"]), collections.Counter())
    finally:
        P._join_inline_links = orig


def rejoin() -> None:
    blogs = {r["url"]: r for r in records() if r["source"] == "blog"}
    print(f"SCOPE: rule applies to source 'blog' only (hosts: "
          f"{sorted({P.host_of(u) for u in blogs})}); {len(blogs)} distinct blog urls")
    affected, samples, negatives = 0, [], []
    for url, r in sorted(blogs.items()):
        before = _pre_join(r)
        joined = 0
        for i, line in enumerate(before):
            nxt = before[i + 1] if i + 1 < len(before) else ""
            frag = len(line) <= P.INLINE_LINK_CHARS and bool(P._CONTINUES.match(nxt))
            if i and (P._CONTINUES.match(line) or frag):
                joined += 1
                if len(samples) < 5 and i >= 1 and len(before[i - 1]) < 120:
                    samples.append((url, before[i - 1:i + 2]))
            elif (i and len(line) <= P.INLINE_LINK_CHARS and len(negatives) < 6
                  and not P._CONTINUES.match(nxt)):
                negatives.append((url, before[i - 1][-50:], line, nxt[:50]))
        affected += joined
    print(f"lines joined to the previous line: {affected}")
    for url, ctx in samples:
        after = P._join_inline_links(list(ctx), collections.Counter())
        print(f"  {url}\n     before: {ctx}\n     after:  {after}")
    print("short lines kept separate (real data; previous tail | line | next head):")
    for url, prev, line, nxt in negatives:
        print(f"  {url}: ...{prev!r} | {line!r} | {nxt!r}...")


def credits() -> None:
    blogs = {r["url"]: r for r in records() if r["source"] == "blog"}
    hits = []
    for url, r in sorted(blogs.items()):
        lines = P._lines(r["body"])
        hits += [(url, x) for x in lines if P._CREDIT.match(x)]
    print(f"SCOPE: source 'blog' only, {len(blogs)} distinct urls. Image-credit lines dropped: "
          f"{len(hits)}")
    for url, x in hits[:10]:
        print(f"  {x!r}   ({url})")


def splitter() -> None:
    text = ("Dr. Rao met St. Mary at No. 5 gate. Entry is Rs. 500 for 3.5 km. "
            "Open 9 a.m. to 5 p.m. daily. (e.g. Sundays) Closed on Mondays.\n"
            "Hotel One, MG Road. ☏ +91 1234\nHotel Two, Beach Rd.")
    for u in P.split_units(text, frozenset()):
        print(f"  [{text[u.start:u.end]}]")
    s = json.loads((REPO_ROOT / "data" / "manifests" / "corpus_stats.json").read_text("utf-8"))
    print("sentences over 300 tokens split on words (corpus):", s["chunking"]["oversize_splits"])


# ---- short chunks


def short() -> None:
    reg = Registry.load(CORE_REGISTRY)
    kept_ids = {c["chunk_id"]: c for c in chunks()}
    rl = rules()
    cause = collections.Counter()
    by_src = collections.Counter()
    seen, done_urls = set(), set()
    for r in sorted(records(), key=lambda r: r["url"]):
        if reg.get(r["place"]).holdout or r["url"] in done_urls:  # a duplicate url once
            continue
        done_urls.add(r["url"])
        c = P.clean_record(r, rl)
        spans, _ = P.chunk_text(c.text, c.header_offsets, P.proxy_count)
        for k, (s, e) in enumerate(spans):
            cid = P.make_chunk_id(r["url"], s)
            if cid not in kept_ids or P.proxy_count(c.text[s:e]) >= P.MIN_TOKENS:
                continue
            if len(spans) == 1:
                why = "short document (one chunk)"
            elif k == len(spans) - 1:
                why = ("end of document: tail < 100 that could not merge" if
                       P.proxy_count(c.text[s:e]) < P.TAIL else "end of document: tail 100-199")
            elif spans[k + 1][0] in c.header_offsets:
                why = "section boundary"
            else:
                why = "next sentence would pass 300 (mid-document)"
            cause[why] += 1
            by_src[(why, kept_ids[cid]["source_type"])] += 1
            seen.add(cid)
    print(f"chunks under {P.MIN_TOKENS} tokens in chunks.jsonl: {len(seen)}")
    for k, v in cause.most_common():
        print(f"  {k}: {v}  " + str({s: n for (w, s), n in sorted(by_src.items()) if w == k}))


# ---- near duplicates


def neardup() -> None:
    cs = chunks()
    sets = [P._shingles(c["text"]) for c in cs]
    pairs = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = sets[i], sets[j]
            n = len(a & b)
            if n:
                pairs.append((n / (len(a) + len(b) - n), i, j))
    pairs.sort(reverse=True)
    print(f"SCOPE: all {len(cs)} kept chunks, 5-word shingles, {len(pairs)} pairs share >= 1 "
          "shingle")
    for label, sel in (("all pairs", pairs),
                       ("pairs from different urls", [p for p in pairs
                                                      if cs[p[1]]["url"] != cs[p[2]]["url"]])):
        print(f"top 5 Jaccard, {label}:")
        for jac, i, j in sel[:5]:
            print(f"  {jac:.4f}  {cs[i]['place']} {cs[i]['url']} | {cs[j]['place']} "
                  f"{cs[j]['url']}")


# ---- D23


def d23() -> None:
    reg = Registry.load(CORE_REGISTRY)
    cs = chunks()
    per = collections.defaultdict(collections.Counter)
    for c in cs:
        per[c["place"]][c["source_type"]] += 1
    print("| place | popularity | chunks | types in chunks | D23 raw | D23 chunks |")
    print("|---|---|---|---|---|---|")
    keep_chunks, keep_hidden, fail = 0, 0, []
    for e in reg.entries:
        if e.holdout:
            continue
        raw = place_coverage(e, RAW_ROOT).passes
        types = set(per[e.name])
        ok = P.passes_rule(types)
        n = sum(per[e.name].values())
        print(f"| {e.name} | {e.popularity} | {n} | {', '.join(sorted(types))} | {raw} | {ok} |")
        if ok:
            keep_chunks += n
            keep_hidden += e.popularity == "hidden_gem"
        else:
            fail.append((e.name, n, e.popularity))
    print(f"\nfailing at chunk level ({len(fail)}): "
          + "; ".join(f"{n} {k} chunks {p}" for n, k, p in fail))
    print(f"IF D33 were applied (NOT applied): {26 - len(fail)} places, {keep_chunks} chunks, "
          f"hidden_gem places {keep_hidden}")


def attribution() -> None:
    att = json.loads((PROCESSED / "attribution.json").read_text(encoding="utf-8"))
    print(f"{len(att)} urls; per source: "
          f"{dict(sorted(collections.Counter(a['source'] for a in att.values()).items()))}")
    for src in ("wikipedia", "wikivoyage", "sahapedia", "blog"):
        u, a = next((u, a) for u, a in att.items() if a["source"] == src)
        print(f"  {u}: {json.dumps(a, ensure_ascii=False, sort_keys=True)}")
    print("licence strings:", sorted({a["license"] for a in att.values()}))
    print("revid present:", collections.Counter(a["source"] for a in att.values()
                                                 if a["revid"] is not None))


SECTIONS = {"manifest": manifest, "holdout": holdout, "blogdates": blogdates, "rejoin": rejoin,
            "credits": credits, "splitter": splitter, "short": short, "neardup": neardup,
            "d23": d23, "attribution": attribution}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    names = sys.argv[1:] or ["all"]
    for n in (list(SECTIONS) if names == ["all"] else names):
        print(f"\n===== {n}")
        SECTIONS[n]()
