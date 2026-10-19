from __future__ import annotations

from src.facts import robots

UA = "TravelPlannerResearch/0.1 (academic IR project; x@example.org)"
BASE = "https://www.example.org"


def verdict(text, path):
    return robots.check(text, UA, BASE + path)


def test_no_rules_or_no_match_means_allowed():
    assert verdict("", "/a").allowed
    assert verdict("User-agent: *\nDisallow: /private/\n", "/public").allowed
    assert verdict("User-agent: *\nDisallow:\n", "/anything").allowed  # empty Disallow


def test_longest_match_wins_over_a_shorter_opposite_rule():
    text = "User-agent: *\nDisallow: /a/\nAllow: /a/b/\n"
    assert not verdict(text, "/a/x").allowed
    v = verdict(text, "/a/b/x")
    assert v.allowed and v.rule.pattern == "/a/b/"


def test_allow_wins_a_tie_in_length():
    text = "User-agent: *\nDisallow: /same\nAllow: /same\n"
    v = verdict(text, "/same/page")
    assert v.allowed and v.rule.directive == "allow"


def test_star_wildcard_and_end_anchor():
    text = "User-agent: *\nDisallow: /*.pdf$\nAllow: /files/*.pdf$\n"
    assert not verdict(text, "/doc/report.pdf").allowed
    assert verdict(text, "/files/report.pdf").allowed
    assert verdict(text, "/doc/report.pdf?x=1").allowed  # `$` anchors at the end of path+query
    assert verdict("User-agent: *\nDisallow: /a*b\n", "/a-anything-b").allowed is False


def test_the_wikidata_style_pair_allows_a_json_file_but_not_other_special_pages():
    text = ("User-agent: *\nDisallow: /wiki/Special:\n"
            "Disallow: /wiki/Special:EntityData/\nAllow: /wiki/Special:EntityData/*.\n")
    v = verdict(text, "/wiki/Special:EntityData/Q1361.json")
    assert v.allowed and v.rule.pattern == "/wiki/Special:EntityData/*."
    assert len(v.matches) == 3
    assert not verdict(text, "/wiki/Special:EntityData/Q1361").allowed  # no dot: Allow can't match
    assert not verdict(text, "/wiki/Special:Search").allowed


def test_a_group_naming_our_product_token_beats_the_star_group():
    text = ("User-agent: *\nDisallow: /\n\n"
            "User-agent: TravelPlannerResearch\nAllow: /\nDisallow: /secret/\n")
    v = verdict(text, "/page")
    assert v.allowed and v.group == "travelplannerresearch"
    assert not verdict(text, "/secret/x").allowed


def test_consecutive_user_agent_lines_share_one_group_and_comments_are_ignored():
    text = "User-agent: Foo\nUser-agent: *  # all\nDisallow: /x # no\n"
    assert not verdict(text, "/x/y").allowed and verdict(text, "/z").allowed


def test_query_string_is_part_of_the_matched_target():
    text = "User-agent: *\nDisallow: /search?q=\n"
    assert not verdict(text, "/search?q=abc").allowed and verdict(text, "/search").allowed
