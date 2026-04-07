from __future__ import annotations

from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd

from stats.io import DayDataset
from stats.replay import iter_market_events, replay_ranges
from stats.utils.common import ensure_dataset


def replay_summary(
    day_dir_or_dataset: Path | DayDataset,
    *,
    replay_on_gap: str = "strict",
) -> dict[str, Any]:
    dataset = ensure_dataset(day_dir_or_dataset)
    segments_total = len(dataset.build_segments())
    kept_ranges = replay_ranges(dataset, on_gap=replay_on_gap)
    kept_segment_ids = {(item.segment_index, item.epoch_id, item.segment_tag) for item in kept_ranges}
    return {
        "replay_on_gap": replay_on_gap,
        "segments_total": segments_total,
        "segments_kept": len(kept_segment_ids),
        "segments_skipped": max(0, segments_total - len(kept_segment_ids)),
    }


def load_market_preview(
    day_dir_or_dataset: Path | DayDataset,
    *,
    limit: int = 20,
    replay_on_gap: str = "skip-segment",
) -> pd.DataFrame:
    dataset = ensure_dataset(day_dir_or_dataset)
    return pd.DataFrame(
        asdict(event)
        for event in islice(iter_market_events(dataset, on_gap=replay_on_gap), int(limit))
    )


def load_day_context(
    day_dir: Path,
    *,
    include_book: bool = True,
    include_trades: bool = True,
    include_events: bool = True,
    include_gaps: bool = False,
    replay_on_gap: str | None = None,
    include_market_preview: bool = False,
    market_preview_limit: int = 20,
) -> dict[str, Any]:
    dataset: DayDataset = ensure_dataset(day_dir)
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
    if replay_on_gap is not None:
        context["replay_summary"] = replay_summary(dataset, replay_on_gap=replay_on_gap)
    if include_market_preview:
        policy = replay_on_gap if replay_on_gap is not None else "skip-segment"
        context["market_preview"] = load_market_preview(
            dataset,
            limit=market_preview_limit,
            replay_on_gap=policy,
        )
    return context
