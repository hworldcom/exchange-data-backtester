from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stats.features import add_aggressor_sign, trades_to_bars
from stats.io import DayDataset, load_day
from stats.replay import replay_top_of_book
from stats.utils.cache import cache_path


def _ensure_dataset(dataset_or_day_dir: DayDataset | Path) -> DayDataset:
    if isinstance(dataset_or_day_dir, DayDataset):
        return dataset_or_day_dir
    return load_day(dataset_or_day_dir)


def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("int64"), unit="ms", utc=True)


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


def get_or_build_top_of_book_table(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: str = "skip-segment",
    force: bool = False,
) -> pd.DataFrame:
    dataset = _ensure_dataset(dataset_or_day_dir)
    params = {"on_gap": on_gap}
    path = cache_path(dataset.day_dir, "top_of_book", params, ext="parquet")
    if path.exists() and not force:
        return pd.read_parquet(path)

    out = replay_top_of_book(dataset, on_gap=on_gap)
    if out.empty:
        out = out.copy()
        out["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        out["microprice"] = pd.Series(dtype="float64")
    else:
        out = _add_microprice(out)
        out["ts"] = _to_ts(out["recv_time_ms"])
    out.to_parquet(path, index=False)
    return out


def get_or_build_trades_table(
    dataset_or_day_dir: DayDataset | Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    dataset = _ensure_dataset(dataset_or_day_dir)
    params: dict[str, object] = {}
    path = cache_path(dataset.day_dir, "trades_enriched", params, ext="parquet")
    if path.exists() and not force:
        return pd.read_parquet(path)

    trades = dataset.load_trades()
    if trades is None or trades.empty:
        out = pd.DataFrame(
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
    else:
        out = add_aggressor_sign(trades)
        out["ts"] = _to_ts(out["recv_time_ms"])
        out["trade_ts"] = _to_ts(out["trade_time_ms"])
    out.to_parquet(path, index=False)
    return out


def get_or_build_market_grid(
    dataset_or_day_dir: DayDataset | Path,
    *,
    grid_freq: str = "100ms",
    on_gap: str = "skip-segment",
    force: bool = False,
) -> pd.DataFrame:
    dataset = _ensure_dataset(dataset_or_day_dir)
    params = {"grid_freq": grid_freq, "on_gap": on_gap}
    path = cache_path(dataset.day_dir, "market_grid", params, ext="parquet")
    if path.exists() and not force:
        return pd.read_parquet(path)

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
    grid.to_parquet(path)
    return grid
