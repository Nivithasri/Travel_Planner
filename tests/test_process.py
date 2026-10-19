"""Phase 2 cleaning, chunking, dedupe and tagging (D48). Offline; synthetic records only."""
from __future__ import annotations

import json
import random

import pytest
from pydantic import ValidationError

from src.common.schemas import BaseChunk, PlaceRegistryEntry, make_chunk_id
from src.facts.facts import HoldoutFilter
from src.process.process import (
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP,
    ReconcileError,
    attribution,
    check_corpus,
    chunk_text,
    clean_record,
    dedupe_records,
    learn_boilerplate,
    load_counter,
    months_before,
    near_dups,
    process_place,
    proxy_count,
    reconcile,
    split_units,
)

SHARE = "Facebook\nX\nPinterest\nWhatsApp\nLinkedin\nTelegram\n"


def entry(name="Hampi", state="KA", holdout=False):
    return PlaceRegistryEntry(name=name, aliases=[], state=state, region_cluster="r",
                              popularity="popular", lat=15.3, lon=76.4, holdout=holdout)


def sentence(i: int, words: int = 12) -> str:
    return " ".join(f"W{i}x{k}" for k in range(words - 1)) + f" end{i}."


def para(n: int, start: int = 0, words: int = 12) -> str:
    return " ".join(sentence(i, words) for i in range(start, start + n))


def rec(body, source="blog", url="https://inditales.com/a/", place="Hampi", **kw):
    r = {"source": source, "url": url, "place": place, "body": body, "title": kw.get("title", "T"),
         "publish_date": kw.get("publish_date"), "lastmod": kw.get("lastmod"),
         "license": kw.get("license", "L"), "extra": kw.get("extra", {})}
    if source == "blog" and "publish_date" not in kw:
        r["publish_date"] = "2011-05-01"
    return r


def run(records, e=None, rules=None):
    return process_place(records, e or entry(), HoldoutFilter(), rules, proxy_count)


# ==== tokens


def test_proxy_counter_is_the_default_and_counts_words_and_punctuation():
    count, name = load_counter(None)
    assert name == "regex-proxy" and count("Rs. 500, open 9-5!") == 9


def test_tokenizer_dir_without_tokenizer_json_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        load_counter(tmp_path)
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        load_counter(tmp_path / "missing_dir")


def test_tokenizer_dir_path_counts_without_special_tokens_truncation_or_padding(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "the": 4, "fort": 5, "opens": 6}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tok.enable_truncation(max_length=3)
    tok.enable_padding(length=20, pad_id=3, pad_token="[PAD]")
    tok.save(str(tmp_path / "tokenizer.json"))
    count, name = load_counter(tmp_path)
    text = "the fort opens the fort opens at nine"  # 8 words, 2 unknown
    assert count(text) == 8  # no [CLS]/[SEP], not truncated to 3, not padded to 20
    assert name.startswith("tokenizer.json sha256:")
    assert proxy_count(text) == 8


# ==== clean


def test_share_header_footer_and_credits_are_removed():
    body = (SHARE + "Real first line about Hampi.\nFrom\nBangalore\n, we drove on.\n"
            "Image courtesy\nShutterstock.com\n/\nLast real line.\n"
            "Recommend you read the following travel blog on places.\nGol Gumbaz\nDandeli")
    c = clean_record(rec(body), None)
    assert c.text == "Real first line about Hampi.\nFrom Bangalore, we drove on.\nLast real line."
    assert c.removed["share_header"] and c.removed["related_footer"] and c.removed["image_credit"]


def test_mid_post_read_more_drops_only_the_link_line_and_title():
    body = "Intro text.\nRead more –\nLinked Post Title\n" + "Long paragraph. " * 40
    c = clean_record(rec(body), None)
    assert c.text.startswith("Intro text.\nLong paragraph.") and "Linked" not in c.text


def test_wiki_blocklisted_sections_are_dropped_and_headers_kept_as_lines():
    body = ("Lead text here.\n== History ==\nOld fort.[1]\n=== Later ===\nMore.[12]\n"
            "== See also ==\nList A\n=== Sub ===\nList B\n== Get in ==\nBy bus.\n"
            "== References ==\nRef 1\n== Empty ==\n")
    c = clean_record(rec(body, source="wikipedia", url="https://en.wikipedia.org/wiki/H"), None)
    assert c.text == "Lead text here.\nHistory\nOld fort.\nLater\nMore.\nGet in\nBy bus."
    lines = c.text.split("\n")
    heads = {c.text.index(x) for x in ("History", "Later", "Get in")}
    assert c.header_offsets == frozenset(heads) and "Ref 1" not in lines and "List B" not in lines


def test_sahapedia_notes_and_bibliography_are_cut():
    body = "Prose one.\nProse two.\nNotes\n1. A note\nBibliography\nBook"
    c = clean_record(rec(body, source="sahapedia", url="https://www.sahapedia.org/x"), None)
    assert c.text == "Prose one.\nProse two."


def test_learned_lines_are_applied_only_from_the_rule_file_and_never_to_headers():
    rules = {"hosts": {"inditales.com": {"lines": [{"line": "Junk", "records": 3}],
                                         "rejected": [{"line": "Travel Tips", "records": 4}]}}}
    c = clean_record(rec("Para one.\nJunk\nTravel Tips\nPara two."), rules)
    assert c.text == "Para one.\nTravel Tips\nPara two."
    wiki = rec("Lead.\n== Junk ==\nBody.", source="wikivoyage",
               url="https://inditales.com/wiki")  # same host, header exempt
    assert "Junk" in clean_record(wiki, rules).text


def test_learning_exempts_headers_counts_urls_once_and_keeps_rejections():
    recs = [rec(f"Common line\nUnique {i}.", url=f"https://inditales.com/{i}/") for i in range(3)]
    recs.append(rec("Common line\nOther.", url="https://inditales.com/0/", place="Ooty"))  # dup url
    doc = learn_boilerplate(recs, ["inditales.com"], min_records=3)
    assert doc["hosts"]["inditales.com"]["lines"] == [{"line": "Common line", "records": 3}]
    doc = learn_boilerplate(recs[:2], ["inditales.com"], min_records=3)
    assert doc["hosts"]["inditales.com"]["lines"] == []
    doc = learn_boilerplate(recs, ["inditales.com"], rejected={"inditales.com": {"Common line"}})
    assert doc["hosts"]["inditales.com"]["rejected"][0]["line"] == "Common line"
    assert doc["hosts"]["inditales.com"]["lines"] == []


def test_wiki_header_lines_survive_cleaning_and_are_never_learned():
    recs = [rec(f"Lead {i}.\n== Eat ==\nFood {i}.", source="wikivoyage",
                url=f"https://inditales.com/w{i}") for i in range(4)]
    assert learn_boilerplate(recs, ["inditales.com"])["hosts"]["inditales.com"]["lines"] == []
    assert "Eat" in clean_record(recs[0], None).text.split("\n")


def test_references_do_not_leak_into_chunks():
    body = para(30) + "\n== References ==\nREFLEAK one.\nREFLEAK two."
    out, _ = run([rec(body, source="wikipedia", url="https://en.wikipedia.org/wiki/H",
                      lastmod="2026-01-01")])
    assert out and not any("REFLEAK" in c["text"] for _, c in out)


# ==== sentence splitter


def test_splitter_keeps_abbreviations_numbers_and_times_inside_sentences():
    text = ("Dr. Rao met St. Mary at No. 5 gate. Entry is Rs. 500 for 3.5 km. "
            "Open 9 a.m. to 5 p.m. daily. (e.g. Sundays) Closed on Mondays.\n"
            "Hotel One, MG Road. ☏ +91 1234\nHotel Two, Beach Rd.")
    units = [text[u.start:u.end] for u in split_units(text, frozenset())]
    assert units == [
        "Dr. Rao met St. Mary at No. 5 gate.", "Entry is Rs. 500 for 3.5 km.",
        "Open 9 a.m. to 5 p.m. daily.", "(e.g. Sundays) Closed on Mondays.",
        "Hotel One, MG Road.", "☏ +91 1234", "Hotel Two, Beach Rd."]


def test_sentence_over_max_is_split_on_words_and_counted():
    text = " ".join(f"w{i}" for i in range(700)) + "."
    spans, st = chunk_text(text, frozenset(), proxy_count)
    assert st.oversize_splits == 1
    assert all(proxy_count(text[s:e]) <= MAX_TOKENS for s, e in spans)


# ==== chunk


def test_chunks_respect_hard_max_including_overlap_and_min_except_tail():
    text = para(120)
    spans, _ = chunk_text(text, frozenset(), proxy_count)
    sizes = [proxy_count(text[s:e]) for s, e in spans]
    assert max(sizes) <= MAX_TOKENS and len(spans) > 5
    assert all(x >= MIN_TOKENS for x in sizes[:-1])


def test_consecutive_chunks_overlap_by_whole_sentences_up_to_the_limit():
    text = para(120)
    spans, _ = chunk_text(text, frozenset(), proxy_count)
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert s2 < e1  # overlap exists
        ov = text[s2:e1]
        assert proxy_count(ov) <= OVERLAP and ov.endswith(".")
        assert s2 == 0 or text[s2 - 1] == " "  # starts at a sentence start


def test_no_overlap_into_a_chunk_that_starts_at_a_header():
    text = para(17) + "\nGet in\n" + para(25, start=100)  # 221 tokens before the header
    heads = frozenset({text.index("\nGet in") + 1})
    spans, _ = chunk_text(text, heads, proxy_count)
    starts = [s for s, _ in spans]
    hi = starts.index(text.index("\nGet in") + 1)
    assert spans[hi - 1][1] <= starts[hi]  # previous chunk ends before the header chunk


def test_tail_merges_only_if_result_fits_else_stays_short():
    # tail of 1 sentence (12 tokens + '.') after a chunk that has room -> merged
    text = para(19)  # ~247 tokens + tail
    spans, st = chunk_text(text + " " + sentence(99), frozenset(), proxy_count)
    assert len(spans) == 1 or st.merged_tails == 1
    # big previous chunk: tail cannot merge without passing 300 -> kept short, counted
    text = para(23) + " " + sentence(98, words=30)
    spans, st = chunk_text(text, frozenset(), proxy_count)
    sizes = [proxy_count(text[s:e]) for s, e in spans]
    assert max(sizes) <= MAX_TOKENS
    assert st.merged_tails + st.short_tails <= 1


def test_offsets_map_back_and_ids_are_url_offset_hashes():
    r = rec(SHARE + para(60))
    out, _ = run([r])
    text = clean_record(r, None).text
    for off, c in out:
        assert text[off:off + len(c["text"])] == c["text"]
        assert c["chunk_id"] == make_chunk_id(r["url"], off)


# ==== dates and tags


@pytest.mark.parametrize("kw,expect", [
    ({"source": "blog", "publish_date": "2011-05-01", "lastmod": "2024-01-01"},
     ("2011-05-01", "published", None)),
    ({"source": "wikipedia", "url": "https://en.wikipedia.org/wiki/H", "lastmod": "2026-09-01"},
     (None, "revised", "2026-09-01")),
    ({"source": "wikivoyage", "url": "https://en.wikivoyage.org/wiki/H", "lastmod": "2026-08-01"},
     (None, "revised", "2026-08-01")),
    ({"source": "sahapedia", "url": "https://www.sahapedia.org/x", "lastmod": "2019-06-28"},
     (None, "sitemap_lastmod", None)),
    ({"source": "blog", "publish_date": None}, (None, "none", None)),
])
def test_date_kind_per_source(kw, expect):
    source = kw.pop("source")
    out, _ = run([rec(para(30), source=source, **kw)])
    c = out[0][1]
    assert (c["date"], c["date_kind"], c["revised_date"]) == expect


def test_chunks_carry_contract_fields_and_map_sahapedia_to_reference():
    out, _ = run([rec(para(30), source="sahapedia", url="https://www.sahapedia.org/x")])
    c = out[0][1]
    assert c["source_type"] == "reference" and c["state"] == "KA" and c["place"] == "Hampi"
    assert c["context_note"] == "" and c["aspect"] is None
    BaseChunk(**c)
    assert all(ch["source_type"] != "live" for _, ch in out)


def test_holdout_entry_and_foreign_record_are_refused():
    with pytest.raises(ValueError):
        run([rec(para(5))], e=entry("Thekkady", "KL", holdout=True))
    with pytest.raises(ValueError):
        run([rec(para(5), place="Ooty")])


# ==== hold-out drop in the pipeline


def test_chunk_naming_a_holdout_is_dropped_and_logged():
    body = para(25) + " We went on to Thekkady for spices. " + para(25, start=200)
    out, rep = run([rec(body)])
    assert rep.chunks_created == len(out) + len(rep.chunk_drops)
    assert rep.chunk_drops and all(d["holdout"] == "Thekkady" for d in rep.chunk_drops)
    assert not any("Thekkady" in c["text"] for _, c in out)


@pytest.mark.parametrize("kw", [{"url": "https://inditales.com/train-ride-araku-valley/"},
                                {"title": "Weekend in Chikmagalur"}])
def test_record_with_holdout_in_url_or_title_is_dropped_whole(kw):
    out, rep = run([rec(para(30), **kw)])
    assert out == [] and len(rep.record_drops) == 1 and rep.chunks_created == 0


def test_river_periyar_does_not_drop():
    out, rep = run([rec(para(10) + " The Periyar river floods. " + para(10, start=50))])
    assert rep.chunk_drops == [] and out


# ==== dedupe


def test_exact_duplicate_is_kept_under_the_earlier_registry_place_without_id_collision():
    body = SHARE + para(40)
    a = rec(body, place="Kodaikanal", url="https://inditales.com/ooty-kodai/")
    b = rec(body, place="Ooty", url="https://inditales.com/ooty-kodai/")
    order = {"Ooty": 0, "Kodaikanal": 1}
    kept, dups = dedupe_records([a, b], order, None)
    assert [r["place"] for r in kept] == ["Ooty"]
    assert dups == [{"level": "exact_duplicate", "url": a["url"], "place": "Kodaikanal",
                     "source": "blog", "kept_url": b["url"], "kept_place": "Ooty"}]
    ids = [c["chunk_id"] for r in kept for _, c in run([r], e=entry("Ooty", "TN"))[0]]
    assert len(ids) == len(set(ids))


def test_near_duplicates_are_dropped_and_distinct_text_kept():
    base = {"url": "u", "place": "P", "source_type": "blog"}
    t = para(20)
    chunks = [{**base, "chunk_id": "a", "text": t},
              {**base, "chunk_id": "b", "text": t.replace("end3.", "finish3.")},
              {**base, "chunk_id": "c", "text": para(20, start=500)}]
    kept, drops = near_dups(chunks)
    assert [c["chunk_id"] for c in kept] == ["a", "c"]
    assert drops[0]["kept_chunk_id"] == "a" and drops[0]["jaccard"] >= 0.8


# ==== determinism and live/batch parity


def test_one_record_alone_gives_the_same_chunks_as_in_a_batch():
    r1 = rec(SHARE + para(50), url="https://inditales.com/a/")
    r2 = rec(para(40, start=900), source="wikivoyage", url="https://en.wikivoyage.org/wiki/H",
             lastmod="2026-01-01")
    alone, _ = run([r1])
    batch, _ = run([r2, r1])
    assert alone == [x for x in batch if x[1]["url"] == r1["url"]]


def test_shuffled_input_gives_identical_ids_and_bytes():
    recs = [rec(para(30, start=i * 100), url=f"https://inditales.com/{i}/") for i in range(6)]

    def dump(rs):
        out, _ = run(rs)
        rows = sorted(out, key=lambda x: (x[1]["url"], x[0]))
        return "".join(json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n" for _, c in rows)

    first = dump(recs)
    shuffled = recs[:]
    random.Random(7).shuffle(shuffled)
    assert dump(shuffled) == first and "\r" not in first


# ==== reconciliation, attribution, dates


def test_reconcile_raises_on_mismatch():
    assert len(reconcile(97, 96, 1, 0, 10, 8, 2, 0)) == 2
    with pytest.raises(ReconcileError):
        reconcile(97, 95, 1, 0, 10, 8, 2, 0)
    with pytest.raises(ReconcileError):
        reconcile(97, 96, 1, 0, 10, 8, 1, 0)


def test_attribution_is_keyed_by_url_for_urls_with_chunks_only():
    r1 = rec(para(5), url="https://en.wikipedia.org/wiki/A", source="wikipedia",
             license="CC BY-SA", extra={"revid": 42}, lastmod="2026-01-01")
    r2 = rec(para(5), url="https://inditales.com/b/")
    a = attribution([r1, r2], [{"url": r1["url"]}])
    assert list(a) == [r1["url"]] and a[r1["url"]]["revid"] == 42
    assert a[r1["url"]]["license"] == "CC BY-SA"


def test_eighteen_month_window():
    from datetime import date
    assert months_before(date(2026, 10, 18), 18) == date(2025, 4, 18)
    assert months_before(date(2026, 8, 31), 18) == date(2025, 2, 28)


def test_basechunk_rejects_a_live_chunk():
    out, _ = run([rec(para(30))])
    with pytest.raises(ValidationError):
        BaseChunk(**{**out[0][1], "source_type": "live"})


def test_periyar_dam_and_mullaperiyar_drop_in_the_pipeline_but_bus_stand_is_kept():
    for phrase, dropped in (("fostered by the Periyar Dam.", True), ("the Mullaperiyar Dam.", True),
                            ("on Periyar Lake.", True), ("from the Periyar Bus Stand.", False),
                            ("along the Periyar river.", False)):
        out, rep = run([rec(para(5) + " Water is " + phrase + " " + para(5, start=60))])
        assert bool(rep.chunk_drops) is dropped, phrase


def test_check_corpus_rejects_live_holdout_and_duplicate_ids():
    out, _ = run([rec(para(40))])
    chunks = [c for _, c in out]
    check_corpus(chunks, HoldoutFilter())
    for bad in ([{**chunks[0], "source_type": "live"}],
                [{**chunks[0], "text": chunks[0]["text"] + " Kumily"}],
                [chunks[0], chunks[0]]):
        with pytest.raises(ReconcileError):
            check_corpus(bad, HoldoutFilter())


def test_reference_date_is_not_an_input_to_chunks():
    import inspect

    assert "date" not in inspect.signature(process_place).parameters
    out, _ = run([rec(para(30))])
    assert set(out[0][1]) == set(BaseChunk.model_fields)
