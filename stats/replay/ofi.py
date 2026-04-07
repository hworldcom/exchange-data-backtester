from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stats.features.returns import _timedelta_to_exact_steps
from stats.io.dataset import DayDataset
from stats.replay.binance import replay_top_of_book
from stats.utils.cache import cache_path, load_or_build_parquet
from stats.utils.common import ensure_dataset, to_utc_datetime_ms


def _cont_ofi(
    prev_bid_price: float,
    prev_bid_qty: float,
    prev_ask_price: float,
    prev_ask_qty: float,
    cur_bid_price: float,
    cur_bid_qty: float,
    cur_ask_price: float,
    cur_ask_qty: float,
) -> float:
    if np.isnan(prev_bid_price) or np.isnan(cur_bid_price):
        bid_term = 0.0
    elif cur_bid_price == prev_bid_price:
        bid_term = cur_bid_qty - prev_bid_qty
    elif cur_bid_price > prev_bid_price:
        bid_term = cur_bid_qty
    else:
        bid_term = -prev_bid_qty

    if np.isnan(prev_ask_price) or np.isnan(cur_ask_price):
        ask_term = 0.0
    elif cur_ask_price == prev_ask_price:
        ask_term = cur_ask_qty - prev_ask_qty
    elif cur_ask_price < prev_ask_price:
        ask_term = -cur_ask_qty
    else:
        ask_term = prev_ask_qty

    return float(bid_term - ask_term)


def compute_ofi_events(dataset_or_day_dir: DayDataset | Path, *, on_gap: str = "strict") -> pd.DataFrame:
    """Compute top-of-book OFI event-by-event from replayed book states."""
    top = replay_top_of_book(dataset_or_day_dir, on_gap=on_gap)
    if top.empty:
        return pd.DataFrame(columns=["recv_time_ms", "recv_seq", "epoch_id", "segment_index", "segment_tag", "ofi"])

    prev = top[
        ["bid1_price", "bid1_qty", "ask1_price", "ask1_qty", "epoch_id", "segment_index"]
    ].shift(1)

    same_segment = (
        (top["epoch_id"] == prev["epoch_id"])
        & (top["segment_index"] == prev["segment_index"])
    )

    ofi = np.zeros(len(top), dtype=float)
    valid_idx = np.flatnonzero(same_segment.to_numpy())
    for idx in valid_idx:
        ofi[idx] = _cont_ofi(
            float(prev.iloc[idx]["bid1_price"]),
            float(prev.iloc[idx]["bid1_qty"]),
            float(prev.iloc[idx]["ask1_price"]),
            float(prev.iloc[idx]["ask1_qty"]),
            float(top.iloc[idx]["bid1_price"]),
            float(top.iloc[idx]["bid1_qty"]),
            float(top.iloc[idx]["ask1_price"]),
            float(top.iloc[idx]["ask1_qty"]),
        )

    out = top[["recv_time_ms", "recv_seq", "epoch_id", "segment_index", "segment_tag"]].copy()
    out["ofi"] = ofi
    return out


def ofi_to_grid(ofi_events: pd.DataFrame, *, grid_freq: str = "100ms") -> pd.DataFrame:
    """Aggregate OFI events onto a fixed time grid."""
    if ofi_events.empty:
        idx = pd.DatetimeIndex([], tz="UTC", name="ts")
        return pd.DataFrame(columns=["ofi_sum", "ofi_abs_sum", "ofi_count"], index=idx)
    ts = to_utc_datetime_ms(ofi_events["recv_time_ms"])
    df = ofi_events.assign(ts=ts).sort_values("ts").set_index("ts")
    return df.resample(grid_freq).agg(
        ofi_sum=("ofi", "sum"),
        ofi_abs_sum=("ofi", lambda values: float(np.abs(values).sum())),
        ofi_count=("ofi", "size"),
    )


def rolling_sum_on_grid(series: pd.Series, *, window_ms: int, grid_freq: str) -> pd.Series:
    """Compute rolling sums on a fixed grid with exact window-to-grid alignment.

    A window such as 150ms on a 100ms grid is rejected instead of silently
    becoming a one-bar 100ms rolling sum.
    """
    if window_ms <= 0:
        return series
    step = pd.Timedelta(grid_freq)
    window = pd.Timedelta(milliseconds=int(window_ms))
    bars = _timedelta_to_exact_steps(window, step, label="window")
    return series.rolling(window=bars, min_periods=1).sum()


def get_or_build_ofi_grid(
    dataset_or_day_dir: DayDataset | Path,
    *,
    grid_freq: str = "100ms",
    windows_ms: Iterable[int] = (100, 500, 1000),
    on_gap: str = "strict",
    force: bool = False,
) -> pd.DataFrame:
    """Build or load OFI grid features and rolling OFI windows.

    Every `window_ms` entry must be an exact multiple of `grid_freq`.
    """
    dataset = ensure_dataset(dataset_or_day_dir)
    params = {
        "grid_freq": grid_freq,
        "windows_ms": list(int(value) for value in windows_ms),
        "on_gap": on_gap,
    }
    path = cache_path(dataset.day_dir, "ofi_grid", params, ext="parquet")

    required_columns = ["ofi_sum", "ofi_abs_sum", "ofi_count"] + [
        name
        for window_ms in params["windows_ms"]
        for name in (f"ofi_sum_{window_ms}ms", f"ofi_abs_sum_{window_ms}ms")
    ]

    def build() -> pd.DataFrame:
        events = compute_ofi_events(dataset, on_gap=on_gap)
        out = ofi_to_grid(events, grid_freq=grid_freq)
        for window_ms in params["windows_ms"]:
            out[f"ofi_sum_{window_ms}ms"] = rolling_sum_on_grid(
                out["ofi_sum"], window_ms=window_ms, grid_freq=grid_freq
            )
            out[f"ofi_abs_sum_{window_ms}ms"] = rolling_sum_on_grid(
                out["ofi_abs_sum"], window_ms=window_ms, grid_freq=grid_freq
            )
        return out

    return load_or_build_parquet(
        path,
        build=build,
        force=force,
        required_columns=required_columns,
        validator=lambda frame: isinstance(frame.index, pd.DatetimeIndex),
        index=True,
    )
