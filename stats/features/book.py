from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def to_datetime_index_ms(df: pd.DataFrame, time_col_ms: str) -> pd.DatetimeIndex:
    if time_col_ms not in df.columns:
        raise KeyError(f"{time_col_ms} not found in dataframe")
    return pd.to_datetime(df[time_col_ms].astype("int64"), unit="ms", utc=True)


def resample_book(book: pd.DataFrame, *, time_col_ms: str = "recv_time_ms", grid_freq: str = "100ms") -> pd.DataFrame:
    ts = to_datetime_index_ms(book, time_col_ms)
    return book.assign(ts=ts).sort_values("ts").set_index("ts").resample(grid_freq).last().ffill()


def compute_mid_spread(book: pd.DataFrame) -> pd.DataFrame:
    required = {"bid1_price", "ask1_price"}
    missing = required - set(book.columns)
    if missing:
        raise KeyError(f"Missing book columns: {sorted(missing)}")
    out = pd.DataFrame(index=book.index)
    out["mid"] = (book["bid1_price"] + book["ask1_price"]) / 2.0
    out["spread"] = book["ask1_price"] - book["bid1_price"]
    out["spread_bps"] = np.where(out["mid"] > 0, 1e4 * out["spread"] / out["mid"], np.nan)
    return out


def compute_depth_imbalance(book: pd.DataFrame, *, levels: Sequence[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=book.index)
    for level in levels:
        bid_cols = [f"bid{k}_qty" for k in range(1, level + 1)]
        ask_cols = [f"ask{k}_qty" for k in range(1, level + 1)]
        missing = [name for name in bid_cols + ask_cols if name not in book.columns]
        if missing:
            raise KeyError(f"Missing depth columns for level={level}: {missing[:8]}")
        bid_sum = book[bid_cols].sum(axis=1)
        ask_sum = book[ask_cols].sum(axis=1)
        denom = (bid_sum + ask_sum).replace(0, np.nan)
        out[f"imbalance_{level}"] = (bid_sum - ask_sum) / denom
        out[f"share_bid_{level}"] = bid_sum / denom
    return out
