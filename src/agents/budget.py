"""Rough trip budget (D101). The tier figures in config/budget_tiers.json are ASSUMED, written
from memory, and are NOT derived from the corpus; every output says so until the owner supplies
figures."""
from __future__ import annotations

from src.agents.state import load_budget_tiers

LABEL = ("rough estimate from ASSUMED tier figures, not derived from the corpus and not a quote "
         "(config/budget_tiers.json, D101)")


def estimate(days: int, tier: str, intercity_hours: float, *, cap_inr: int | None = None,
             cfg: dict | None = None) -> dict:
    """{'days','tier','stay','food','local_transport','activities','intercity','total',
    'per_day_total','within_cap','status','label'} in INR. `within_cap` is None without a cap and
    otherwise compares the total with the cap plus the configured tolerance."""
    cfg = cfg or load_budget_tiers()
    if tier not in cfg["per_day"]:
        raise ValueError(f"unknown budget tier {tier!r}")
    if days < 1 or intercity_hours < 0:
        raise ValueError("days must be at least 1 and travel hours cannot be negative")
    per = cfg["per_day"][tier]
    lines = {k: int(v) * days for k, v in per.items()}
    intercity = round(intercity_hours * cfg["intercity_per_hour"][tier])
    total = sum(lines.values()) + intercity
    within = None if cap_inr is None else total <= cap_inr * (1 + cfg["tolerance"])
    return {"days": days, "tier": tier, **lines, "intercity": intercity, "total": total,
            "per_day_total": sum(int(v) for v in per.values()), "within_cap": within,
            "cap_inr": cap_inr, "currency": cfg["currency"], "status": cfg["status"],
            "label": LABEL}


def budget_line(b: dict) -> str:
    """The computed budget line for the itinerary (tool provenance, not a chunk citation)."""
    cap = ""
    if b["within_cap"] is not None:
        cap = (f"; your cap INR {b['cap_inr']:,} is "
               + ("covered" if b["within_cap"] else "NOT covered") + " by this estimate")
    return (f"Rough budget ({b['tier']} tier, {b['days']} days): about INR {b['total']:,} in total "
            f"(stay {b['stay']:,}, food {b['food']:,}, local transport {b['local_transport']:,}, "
            f"activities {b['activities']:,}, intercity {b['intercity']:,}){cap}. "
            f"{b['label'][0].upper()}{b['label'][1:]}. (computed: budget table)")
