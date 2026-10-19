"""ESTIMATE ONLY: raw bytes, cleaned characters and projected chunk counts from data/raw.

    python scripts/estimate_corpus_size.py

Every number is an upper bound on the Phase 2 corpus (see src/collect/size_estimate.py). Two
levels are reported: the basic estimate and a stricter one that also drops back-matter sections.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.size_estimate import (  # noqa: E402
    BACK_MATTER,
    CHARS_PER_TOKEN,
    OVERLAP,
    chunks_for,
    estimate_clean,
    estimate_strict,
)
from src.common.config import REPO_ROOT  # noqa: E402

N_PLACES = 26  # collectable core places (30 minus 4 hold-outs)


def main() -> int:
    recs = [json.loads(line) for f in sorted((REPO_ROOT / "data" / "raw").glob("*/*/records.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    C = collections.Counter
    raw_bytes, raw_chars, basic, strict = C(), C(), C(), C()
    s_raw, s_basic, s_strict = C(), C(), C()
    dropped: dict = collections.defaultdict(C)
    for r in recs:
        body, src, pl = r["body"], r["source"], r["place"]
        b = len(estimate_clean(body, src))
        t, heads = estimate_strict(body, src)
        raw_bytes[pl] += len(body.encode("utf-8"))
        raw_chars[pl] += len(body)
        basic[pl] += b
        strict[pl] += len(t)
        s_raw[src] += len(body)
        s_basic[src] += b
        s_strict[src] += len(t)
        for h in heads:
            dropped[src][h] += 1
    print("ESTIMATE ONLY (upper bound: dedup, off-topic filtering and O8 drops come later)")
    print(f"assumptions: {CHARS_PER_TOKEN:g} chars/token, {OVERLAP:.0%} chunk overlap")
    print(f"strict level drops these section headings (and their sub-sections): "
          f"{', '.join(BACK_MATTER)}\n")
    print(f"{'place':10} {'raw bytes':>10} {'raw chars':>10} {'basic':>9} {'strict':>9}")
    for p in sorted(basic):
        print(f"{p:10} {raw_bytes[p]:>10} {raw_chars[p]:>10} {basic[p]:>9} {strict[p]:>9}")
    tb, tr, tc, ts = (sum(raw_bytes.values()), sum(raw_chars.values()), sum(basic.values()),
                      sum(strict.values()))
    print(f"{'TOTAL':10} {tb:>10} {tr:>10} {tc:>9} {ts:>9}   "
          f"(basic {tc / tr:.1%}, strict {ts / tr:.1%} of raw)")
    print("\nper source: raw chars -> basic (ratio) -> strict (ratio)")
    for s in sorted(s_raw):
        print(f"  {s:11} {s_raw[s]:>8} -> {s_basic[s]:>8} ({s_basic[s] / s_raw[s]:.1%})"
              f" -> {s_strict[s]:>8} ({s_strict[s] / s_raw[s]:.1%})")
    print("headings actually dropped by the strict level:",
          {s: dict(h) for s, h in dropped.items()} or "none")
    n = len(basic)
    for label, c in (("BASIC", basic), ("STRICT", strict)):
        print(f"\n{label}: chunks per place and projection for {N_PLACES} places "
              f"(mean of the {n} measured places; min/max place in brackets):")
        for t in (200, 250, 300):
            per = [chunks_for(c[p], t) for p in c]
            mean = sum(per) / n
            print(f"  {t} tokens: {mean:6.0f} per place [{min(per):.0f}..{max(per):.0f}]"
                  f"  -> {N_PLACES * mean:7,.0f} chunks  (range {N_PLACES * min(per):,.0f}"
                  f" to {N_PLACES * max(per):,.0f})")
    m250 = N_PLACES * sum(chunks_for(strict[p], 250) for p in strict) / n
    print(f"\nplan target: 10,000 to 20,000 chunks. Strict projection (250 tokens) = {m250:,.0f}: "
          f"{'BELOW' if m250 < 10000 else 'within'} the 10,000 floor ({m250 / 10000:.0%} of it).")
    print(f"Caveat: {n} places measured (the collectable core list); no Reddit and no official",
          "pages yet, so the traveler-voice and freshness share is small; Phase 2 dedup and",
          "O8 drops will lower the count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
