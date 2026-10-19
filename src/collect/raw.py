"""Raw record store: data/raw/<source>/<place_slug>/records.jsonl (docs/contracts.md "Raw record").

One JSON object per line. A record is identified by (url, content_sha256): writing the same page
with the same content twice is a no-op, so a refresh that finds nothing new adds nothing. The
optional manifest (url, source, place, retrieved_at, sha256, bytes; no content) is the committed
trace of what was collected.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.cache import append_jsonl, read_jsonl
from src.places.registry import slugify

RAW_SOURCES = ("wikivoyage", "wikipedia", "sahapedia", "reddit", "youtube", "blog", "official")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RawRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    place: str
    url: str = Field(min_length=1)
    retrieved_at: str
    publish_date: str | None = None
    lastmod: str | None = None  # never used as a publish date
    content_sha256: str
    license: str = ""
    title: str = ""
    body: str = Field(min_length=1)
    extra: dict = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if v not in RAW_SOURCES:
            raise ValueError(f"unknown raw source {v!r}")
        return v

    @field_validator("publish_date", "lastmod")
    @classmethod
    def _date(cls, v: str | None) -> str | None:
        if v is not None and not _DATE.match(v):
            raise ValueError("date must be YYYY-MM-DD or null")
        return v


def make_record(source: str, place: str, url: str, body: str, retrieved_at: float,
                **kw) -> RawRecord:
    return RawRecord(source=source, place=place, url=url, body=body,
                     retrieved_at=iso_utc(retrieved_at), content_sha256=sha256_text(body), **kw)


class RawStore:
    def __init__(self, raw_root: str | Path, manifest: str | Path | None = None):
        self.root = Path(raw_root)
        self.manifest = Path(manifest) if manifest else None

    def path_for(self, source: str, place: str) -> Path:
        return self.root / source / slugify(place) / "records.jsonl"

    def read(self, source: str, place: str) -> list[dict]:
        p = self.path_for(source, place)
        return list(read_jsonl(p)) if p.exists() else []

    def write(self, rec: RawRecord) -> bool:
        """Append the record; False when the same url with the same content is already stored."""
        path = self.path_for(rec.source, rec.place)
        if any(r["url"] == rec.url and r["content_sha256"] == rec.content_sha256
               for r in self.read(rec.source, rec.place)):
            return False
        append_jsonl(path, [rec.model_dump(mode="json")])
        if self.manifest is not None:
            append_jsonl(self.manifest, [{
                "url": rec.url, "source": rec.source, "place": rec.place,
                "retrieved_at": rec.retrieved_at, "sha256": rec.content_sha256,
                "bytes": len(rec.body.encode("utf-8"))}])
        return True
