from __future__ import annotations

from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd

from stats.io import DayDataset, load_day
from stats.replay import iter_market_events, replay_ranges
from stats.tables import get_or_build_top_of_book_table, get_or_build_trades_table
from stats.utils.common import ensure_dataset


def find_backtester_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "stats").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the exchange-data-backtester project root")


def resolve_day_dir(project_root: Path, *, symbol: str, day: str, exchange: str = "binance") -> Path:
    candidates = [
        project_root.parent / "exchange-data-recorder" / "data" / symbol / day,
        project_root.parent / "exchange-data-recorder" / "data" / exchange / symbol / day,
        project_root.parent / "ExchangeDataRecorder" / "data" / symbol / day,
        project_root.parent / "ExchangeDataRecorder" / "data" / exchange / symbol / day,
        project_root / "data" / symbol / day,
        project_root / "data" / exchange / symbol / day,
    ]
    for candidate in candidates:
        if (candidate / "schema.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find data directory for {exchange} {symbol} {day}")


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


def load_orderflow_day(
    *,
    day: str,
    symbol: str = "BTCUSDC",
    exchange: str = "binance",
    replay_on_gap: str = "skip-segment",
    project_root: Path | None = None,
) -> tuple[DayDataset, pd.DataFrame, pd.DataFrame, pd.Series]:
    root = find_backtester_root() if project_root is None else Path(project_root).resolve()
    day_dir = resolve_day_dir(root, symbol=symbol, day=day, exchange=exchange)
    dataset = load_day(day_dir)
    trades = get_or_build_trades_table(dataset)
    top = get_or_build_top_of_book_table(dataset, on_gap=replay_on_gap)
    replay_info = replay_summary(dataset, replay_on_gap=replay_on_gap)
    summary = pd.Series(
        {
            "exchange": dataset.exchange,
            "symbol": dataset.symbol,
            "day": dataset.day,
            "day_dir": str(dataset.day_dir),
            "trades_rows": len(trades),
            "top_rows": len(top),
            "trade_start_utc": trades["ts"].min() if not trades.empty else pd.NaT,
            "trade_end_utc": trades["ts"].max() if not trades.empty else pd.NaT,
            "top_start_utc": top["ts"].min() if not top.empty else pd.NaT,
            "top_end_utc": top["ts"].max() if not top.empty else pd.NaT,
            "segments_total": replay_info["segments_total"],
            "segments_kept": replay_info["segments_kept"],
            "segments_skipped": replay_info["segments_skipped"],
        }
    )
    return dataset, trades, top, summary
