from __future__ import annotations

from pathlib import Path
from typing import Any

from stats.io import DayDataset, load_day


def load_day_context(
    day_dir: Path,
    *,
    include_book: bool = True,
    include_trades: bool = True,
    include_events: bool = True,
    include_gaps: bool = False,
) -> dict[str, Any]:
    dataset: DayDataset = load_day(day_dir)
    context: dict[str, Any] = {
        "dataset": dataset,
        "day_dir": dataset.day_dir,
        "exchange": dataset.exchange,
        "symbol": dataset.symbol,
        "day": dataset.day,
    }
    if include_book:
        context["book"] = dataset.load_book()
    if include_trades:
        context["trades"] = dataset.load_trades()
    if include_events:
        context["events"] = dataset.load_events()
    if include_gaps:
        context["gaps"] = dataset.load_gaps()
    return context
