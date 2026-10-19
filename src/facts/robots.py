"""An RFC 9309 robots.txt matcher (D41): groups by user-agent, `*` wildcard, `$` end anchor,
the longest matching rule wins, Allow wins a tie, no matching rule means allowed.

Python's urllib.robotparser treats `*` literally, so it cannot read patterns such as
`Allow: /wiki/Special:EntityData/*.`. This module is used only to INFORM decisions (the HttpClient
robots check is unchanged). Percent-encoding is compared as written (no normalisation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


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
