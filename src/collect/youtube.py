"""YouTube refusal stub (D27, D30): the transcript collector was removed.

Transcripts and the Data API stay OFF for stored data (robots.txt disallows the transcript paths;
API terms cap storage at 30 days). The name stays in SOURCES so an explicit request is reported
as skipped, with zero requests, never as a failure.
"""
from __future__ import annotations

from src.collect.types import SourceResult

SKIPPED_D27 = "skipped (D27): YouTube transcripts are dropped; robots.txt disallows the paths"


class YouTubeCollector:
    source = "youtube"

    def __init__(self, *args, **kwargs):  # ignores everything: no key, no client, no cache
        pass

    def collect(self, *args, **kwargs) -> SourceResult:
        return SourceResult(skipped=SKIPPED_D27)
