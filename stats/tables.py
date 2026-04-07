from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stats.features import add_aggressor_sign, trades_to_bars
from stats.io import DayDataset
from stats.replay import replay_book_frames, replay_top_of_book
from stats.utils.cache import cache_path, load_or_build_parquet
from stats.utils.common import ensure_dataset, to_utc_datetime_ms


def _add_microprice(book: pd.DataFrame) -> pd.DataFrame:
    out = book.copy()
    if not {"bid1_price", "bid1_qty", "ask1_price", "ask1_qty"} <= set(out.columns):
        out["microprice"] = np.nan
        return out
    denom = out["bid1_qty"] + out["ask1_qty"]
    out["microprice"] = np.where(
        denom > 0,
        (out["ask1_price"] * out["bid1_qty"] + out["bid1_price"] * out["ask1_qty"]) / denom,
        np.nan,
    )
    return out


def _expected_book_level_columns(top_n: int) -> list[str]:
    cols = ["event_type", "recv_seq", "recv_time_ms", "event_time_ms", "epoch_id", "segment_index", "segment_tag", "ts"]
    for level in range(1, int(top_n) + 1):
        cols.extend(
            [
                f"bid{level}_price",
                f"bid{level}_qty",
                f"ask{level}_price",
                f"ask{level}_qty",
            ]
        )
    return cols


def _build_top_of_book_table(dataset: DayDataset, *, on_gap: str) -> pd.DataFrame:
    out = replay_top_of_book(dataset, on_gap=on_gap)
    if out.empty:
        out = out.copy()
        out["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        out["microprice"] = pd.Series(dtype="float64")
    else:
        out = _add_microprice(out)
        out["ts"] = to_utc_datetime_ms(out["recv_time_ms"])
    return out


def _build_book_levels_table(dataset: DayDataset, *, top_n: int, on_gap: str, show_progress: bool) -> pd.DataFrame:
    out = replay_book_frames(dataset, top_n=top_n, on_gap=on_gap, show_progress=show_progress)
    if out.empty:
        out = out.copy()
        out["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        out = out.copy()
        out["ts"] = to_utc_datetime_ms(out["recv_time_ms"])
    return out


def _build_trades_table(dataset: DayDataset) -> pd.DataFrame:
    trades = dataset.load_trades()
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=[
                "event_time_ms",
                "recv_time_ms",
                "recv_seq",
                "run_id",
                "trade_id",
                "trade_time_ms",
                "price",
                "qty",
                "is_buyer_maker",
                "side",
                "ord_type",
                "exchange",
                "symbol",
                "aggr_sign",
                "signed_qty",
                "notional",
                "signed_notional",
                "ts",
                "trade_ts",
            ]
        )

    out = add_aggressor_sign(trades)
    out["ts"] = to_utc_datetime_ms(out["recv_time_ms"])
    out["trade_ts"] = to_utc_datetime_ms(out["trade_time_ms"])
    return out


def _build_market_grid(dataset: DayDataset, *, grid_freq: str, on_gap: str, force: bool) -> pd.DataFrame:
    top = get_or_build_top_of_book_table(dataset, on_gap=on_gap, force=force)
    trades = get_or_build_trades_table(dataset, force=force)

    if top.empty:
        book_grid = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    else:
        book_grid = (
            top.sort_values("ts")
            .set_index("ts")
            .resample(grid_freq)
            .last()
            .ffill()
        )
        book_grid["book_updates"] = (
            top.assign(one=1)
            .set_index("ts")["one"]
            .resample(grid_freq)
            .sum()
            .fillna(0)
            .astype("int64")
        )
        book_grid["valid_book"] = True

    if trades.empty:
        trades_grid = pd.DataFrame(index=book_grid.index.copy())
    else:
        trades_grid = trades_to_bars(trades, grid_freq=grid_freq, time_col_ms="recv_time_ms")

    grid = book_grid.join(trades_grid, how="outer").sort_index()
    if "valid_book" in grid.columns:
        grid["valid_book"] = grid["valid_book"].fillna(False)
    if "book_updates" in grid.columns:
        grid["book_updates"] = grid["book_updates"].fillna(0).astype("int64")
    return grid


def get_or_build_top_of_book_table(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: str = "skip-segment",
    force: bool = False,
) -> pd.DataFrame:
    dataset = ensure_dataset(dataset_or_day_dir)
    params = {"on_gap": on_gap}
    path = cache_path(dataset.day_dir, "top_of_book", params, ext="parquet")
    required_columns = _expected_book_level_columns(1) + ["mid", "spread", "spread_bps", "microprice"]
    return load_or_build_parquet(
        path,
        build=lambda: _build_top_of_book_table(dataset, on_gap=on_gap),
        force=force,
        required_columns=required_columns,
        index=False,
    )


def get_or_build_book_levels_table(
    dataset_or_day_dir: DayDataset | Path,
    *,
    top_n: int = 5,
    on_gap: str = "skip-segment",
    show_progress: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    dataset = ensure_dataset(dataset_or_day_dir)
    params = {"top_n": int(top_n), "on_gap": on_gap}
    path = cache_path(dataset.day_dir, "book_levels", params, ext="parquet")
    return load_or_build_parquet(
        path,
        build=lambda: _build_book_levels_table(dataset, top_n=top_n, on_gap=on_gap, show_progress=show_progress),
        force=force,
        required_columns=_expected_book_level_columns(top_n),
        index=False,
    )


def get_or_build_trades_table(
    dataset_or_day_dir: DayDataset | Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    dataset = ensure_dataset(dataset_or_day_dir)
    params: dict[str, object] = {}
    path = cache_path(dataset.day_dir, "trades_enriched", params, ext="parquet")
    return load_or_build_parquet(
        path,
        build=lambda: _build_trades_table(dataset),
        force=force,
        required_columns=[
            "event_time_ms",
            "recv_time_ms",
            "recv_seq",
            "run_id",
            "trade_id",
            "trade_time_ms",
            "price",
            "qty",
            "is_buyer_maker",
            "side",
            "ord_type",
            "exchange",
            "symbol",
            "aggr_sign",
            "signed_qty",
            "notional",
            "signed_notional",
            "ts",
            "trade_ts",
        ],
        index=False,
    )


def get_or_build_market_grid(
    dataset_or_day_dir: DayDataset | Path,
    *,
    grid_freq: str = "100ms",
    on_gap: str = "skip-segment",
    force: bool = False,
) -> pd.DataFrame:
    dataset = ensure_dataset(dataset_or_day_dir)
    params = {"grid_freq": grid_freq, "on_gap": on_gap}
    path = cache_path(dataset.day_dir, "market_grid", params, ext="parquet")
    return load_or_build_parquet(
        path,
        build=lambda: _build_market_grid(dataset, grid_freq=grid_freq, on_gap=on_gap, force=force),
        force=force,
        required_columns=["book_updates", "valid_book"],
        validator=lambda frame: isinstance(frame.index, pd.DatetimeIndex),
        index=True,
    )
