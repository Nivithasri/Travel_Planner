from __future__ import annotations

import src.collect.size_estimate as se
from src.collect.size_estimate import build_matrix, tag_aspects

WIKI = """Intro text about the place.
== History ==
Old stories.
=== Early days ===
More stories.
== See also ==
Related page one.
=== Sub of see also ===
Related page two.
== Notes ==
A footnote.
== Transport ==
Buses run daily.
== References ==
Some citation.
"""


def test_strict_drops_back_matter_sections_with_their_subsections_and_lists_them():
    text, dropped = se.estimate_strict(WIKI, "wikipedia")
    assert dropped == ["See also", "Notes", "References"]
    assert text == "Intro text about the place. Old stories. More stories. Buses run daily."
    assert "Related" not in text and "citation" not in text


def test_basic_level_keeps_back_matter_text_and_strict_is_never_larger():
    basic = se.estimate_clean(WIKI, "wikipedia")
    strict, _ = se.estimate_strict(WIKI, "wikipedia")
    assert "Related page one." in basic and len(strict) < len(basic)


def test_strict_leaves_non_wiki_sources_unchanged():
    text, dropped = se.estimate_strict("Facebook\nX\nHello there", "blog")
    assert text == "Hello there" and dropped == []


def test_module_is_labelled_estimate_only_and_lives_outside_src_process():
    assert "ESTIMATE ONLY" in se.__doc__ and "process" not in se.__file__.replace("\\", "/")


# ---- merged from tests/test_matrix.py: 

def test_tag_aspects_uses_whole_word_keywords_only():
    assert tag_aspects("The temple is closed on Mondays; entry fee Rs. 50.") == {"closure", "price"}
    assert tag_aspects("A business review.") == set()  # "bus" and "view" inside words do not count


def rec(source, body, publish=None, place="Examplepura"):
    return {"source": source, "place": place, "body": body, "publish_date": publish}


def test_build_matrix_counts_types_dates_and_ignores_missing_publish_dates():
    recs = [
        rec("blog", "Best time is the monsoon. Bus from the airport.", "2022-05-01"),
        rec("blog", "Opening hours are posted.", "2024-11-21"),
        rec("wikipedia", "== History ==\nBuilt in the 12th century by a king."),
    ]
    m = build_matrix(recs)["Examplepura"]
    assert m["records"] == 3 and m["dated"] == 2 and m["newest"] == "2024-11-21"
    assert m["by_type"]["blog"]["records"] == 2 and m["by_type"]["reference"]["records"] == 1
    asp = m["aspects"]
    assert asp["season"] == 1 and asp["transport"] == 1 and asp["history"] == 1


def test_size_estimate_cleaning_is_conservative_and_labelled_estimate_only():
    import src.collect.size_estimate as se

    assert "ESTIMATE ONLY" in se.__doc__
    text = "Facebook\nX\nPinterest\nBody text here\n== See ==\nMore text [Music] done"
    assert se.estimate_clean(text, "blog") == "Body text here == See == More text done"
    wiki = se.estimate_clean(text, "wikivoyage")
    assert wiki == "Body text here More text done"  # headings dropped too
    assert round(se.chunks_for(8500, 250)) == 10  # 850 chars per 250-token chunk
