"""Microprice state features derived from replayed top-of-book states."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats.io import DayDataset
from stats.utils.common import to_utc_datetime_ms


def resolve_tick_size(dataset: DayDataset, book_states: pd.DataFrame) -> float:
    """Return the instrument tick size from metadata or infer it from observed spreads."""
    instrument = getattr(dataset, "instrument", None)
    if instrument is not None and instrument.tick_size not in (None, ""):
        return float(instrument.tick_size)
    positive_spreads = book_states.loc[book_states["spread"] > 0, "spread"]
    if positive_spreads.empty:
        raise RuntimeError("Could not infer tick size: no positive spreads observed")
    return float(np.round(float(positive_spreads.min()), 10))


def bucket_imbalance(
    series: pd.Series,
    *,
    bucket_count: int | None,
    bucket_edges: list[float] | None,
) -> pd.Series:
    """Bucket level-1 imbalance either by explicit edges or quantiles."""
    clean = series.clip(lower=0.0, upper=1.0)
    if bucket_edges is not None:
        if not bucket_edges:
            raise ValueError("bucket_edges must be non-empty when provided")
        edges = [float(v) for v in bucket_edges]
        if edges[0] > 0.0:
            edges = [0.0] + edges
        if edges[-1] < 1.0:
            edges = edges + [1.0]
        bucket = pd.cut(clean, bins=edges, labels=False, include_lowest=True, duplicates="drop")
        return (bucket.astype("Int64") + 1).astype("Int64")
    if bucket_count is None:
        raise ValueError("Either bucket_count or bucket_edges must be provided for imbalance bucketing")
    if int(bucket_count) < 1:
        raise ValueError("bucket_count must be >= 1")
    q = max(1, min(int(bucket_count), int(clean.nunique())))
    try:
        bucket = pd.qcut(clean, q=q, labels=False, duplicates="drop")
    except ValueError:
        bucket = pd.cut(clean, bins=q, labels=False, include_lowest=True)
    return (bucket.astype("Int64") + 1).astype("Int64")


def bucket_spread_ticks(series: pd.Series, bucket_values: list[int]) -> pd.Series:
    """Map spread-in-ticks to labeled buckets, with the final bucket acting as a tail floor."""
    if not bucket_values:
        raise ValueError("bucket_values must be non-empty")
    bucket_values = sorted(int(v) for v in bucket_values)
    tail_floor = bucket_values[-1]
    labels = []
    for value in series.astype("int64"):
        if value >= tail_floor:
            labels.append(f"{tail_floor}+")
        else:
            labels.append(str(value))
    return pd.Series(labels, index=series.index, dtype="string")


def add_microprice_features(
    dataset: DayDataset,
    book_states: pd.DataFrame,
    *,
    event_time_or_recv_time: str = "recv",
    imbalance_bucket_count: int | None = 10,
    imbalance_bucket_edges: list[float] | None = None,
    spread_bucket_values: list[int] | tuple[int, ...] = (1, 2, 3),
) -> tuple[pd.DataFrame, float, str]:
    """Add standard microprice state features to replayed top-of-book rows."""
    time_col_ms = "recv_time_ms" if event_time_or_recv_time == "recv" else "event_time_ms"
    tick_size = resolve_tick_size(dataset, book_states)

    features = book_states.copy()
    features = features.loc[
        (features["bid1_price"] > 0)
        & (features["ask1_price"] > 0)
        & (features["bid1_qty"] > 0)
        & (features["ask1_qty"] > 0)
        & (features["ask1_price"] >= features["bid1_price"])
    ].copy()
    features["mid_price"] = (features["bid1_price"] + features["ask1_price"]) / 2.0
    features["spread_abs"] = features["ask1_price"] - features["bid1_price"]
    features["spread_ticks"] = np.rint(features["spread_abs"] / tick_size).astype("int64")
    denom = features["bid1_qty"] + features["ask1_qty"]
    features["level1_imbalance"] = features["bid1_qty"] / denom
    features["microprice_weighted_mid"] = np.where(
        denom > 0,
        (features["bid1_qty"] * features["ask1_price"] + features["ask1_qty"] * features["bid1_price"]) / denom,
        np.nan,
    )
    features["weighted_mid_adjustment"] = features["microprice_weighted_mid"] - features["mid_price"]
    features["queue_sum"] = denom
    features["analysis_ts"] = to_utc_datetime_ms(features[time_col_ms])
    features["tick_size"] = tick_size
    features["imbalance_bucket"] = bucket_imbalance(
        features["level1_imbalance"],
        bucket_count=imbalance_bucket_count,
        bucket_edges=imbalance_bucket_edges,
    )
    features["spread_bucket"] = bucket_spread_ticks(features["spread_ticks"], list(spread_bucket_values))
    features["state_id"] = (
        features["spread_bucket"].astype("string") + "|I" + features["imbalance_bucket"].astype("string")
    )
    features["exchange"] = dataset.day_dir.parent.parent.name
    features["symbol"] = dataset.day_dir.parent.name
    features["day"] = dataset.day_dir.name
    features["day_dir"] = str(dataset.day_dir)
    return features, tick_size, time_col_ms
