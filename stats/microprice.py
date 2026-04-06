from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from stats.io import DayDataset, load_day
from stats.tables import get_or_build_top_of_book_table
from stats.utils.cache import analysis_cache_path, cache_path


def _ensure_dataset(dataset_or_day_dir: DayDataset | Path) -> DayDataset:
    if isinstance(dataset_or_day_dir, DayDataset):
        return dataset_or_day_dir
    return load_day(dataset_or_day_dir)


def _required_labeled_columns(resolved_future_move_k: int) -> set[str]:
    required = {
        "future_move_k",
        "future_move_definition",
        "target_mid",
        "tau_k_recv_seq",
        "tau_k_time_ms",
        "tau_k_analysis_ts",
        "time_to_target_ms",
        "delta_mid_target",
        "direction_target",
    }
    if int(resolved_future_move_k) == 1:
        required |= {
            "next_mid",
            "tau1_recv_seq",
            "tau1_time_ms",
            "tau1_analysis_ts",
            "time_to_tau1_ms",
            "delta_mid_tau1",
            "direction_tau1",
        }
    return required


def _load_valid_cached_labeled_table(path: Path, *, resolved_future_move_k: int) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    missing = _required_labeled_columns(resolved_future_move_k) - set(cached.columns)
    if missing:
        return None
    return cached


def resolve_tick_size(dataset: DayDataset, book_states: pd.DataFrame) -> float:
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
    clean = series.clip(lower=0.0, upper=1.0)
    if bucket_edges is not None:
        edges = [float(v) for v in bucket_edges]
        if edges[0] > 0.0:
            edges = [0.0] + edges
        if edges[-1] < 1.0:
            edges = edges + [1.0]
        bucket = pd.cut(clean, bins=edges, labels=False, include_lowest=True, duplicates="drop")
        return (bucket.astype("Int64") + 1).astype("Int64")
    if bucket_count is None:
        raise ValueError("Either bucket_count or bucket_edges must be provided for imbalance bucketing")
    q = max(1, min(int(bucket_count), int(clean.nunique())))
    try:
        bucket = pd.qcut(clean, q=q, labels=False, duplicates="drop")
    except ValueError:
        bucket = pd.cut(clean, bins=q, labels=False, include_lowest=True)
    return (bucket.astype("Int64") + 1).astype("Int64")


def bucket_spread_ticks(series: pd.Series, bucket_values: list[int]) -> pd.Series:
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
    features["analysis_ts"] = pd.to_datetime(features[time_col_ms], unit="ms", utc=True)
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


def _resolve_future_move_k(*, future_move_definition: str, future_move_k: int) -> int:
    if future_move_definition not in {"next_mid_change", "kth_mid_change"}:
        raise ValueError(
            "future_move_definition must be one of {'next_mid_change', 'kth_mid_change'}"
        )
    if future_move_definition == "next_mid_change":
        return 1
    if int(future_move_k) < 1:
        raise ValueError("future_move_k must be >= 1")
    return int(future_move_k)


def label_kth_mid_change(
    frame: pd.DataFrame,
    *,
    k: int = 1,
    time_col_ms: str = "recv_time_ms",
) -> pd.DataFrame:
    if int(k) < 1:
        raise ValueError("k must be >= 1")

    labeled_parts = []
    for _, segment_frame in frame.groupby(["segment_index", "epoch_id"], sort=False):
        part = segment_frame.sort_values("recv_seq").copy()
        part["mid_run_id"] = part["mid_price"].ne(part["mid_price"].shift()).cumsum()
        run_summary = (
            part.groupby("mid_run_id", sort=False)
            .agg(
                run_mid=("mid_price", "first"),
                tau_k_recv_seq=("recv_seq", lambda s: s.iloc[0]),
                tau_k_time_ms=(time_col_ms, lambda s: s.iloc[0]),
            )
            .reset_index()
        )
        run_summary["target_mid"] = run_summary["run_mid"].shift(-k)
        run_summary["tau_k_recv_seq"] = run_summary["tau_k_recv_seq"].shift(-k)
        run_summary["tau_k_time_ms"] = run_summary["tau_k_time_ms"].shift(-k)
        part = part.merge(
            run_summary[["mid_run_id", "target_mid", "tau_k_recv_seq", "tau_k_time_ms"]],
            on="mid_run_id",
            how="left",
        )
        part["future_move_k"] = int(k)
        part["time_to_target_ms"] = part["tau_k_time_ms"] - part[time_col_ms]
        part["delta_mid_target"] = part["target_mid"] - part["mid_price"]
        part["direction_target"] = np.sign(part["delta_mid_target"])
        labeled_parts.append(part)

    if not labeled_parts:
        return pd.DataFrame()

    labeled = pd.concat(labeled_parts, ignore_index=True)
    labeled = labeled.loc[labeled["target_mid"].notna()].copy()
    labeled["tau_k_analysis_ts"] = pd.to_datetime(labeled["tau_k_time_ms"], unit="ms", utc=True)

    if int(k) == 1:
        # Backward-compatible aliases for older notebook/test code.
        labeled["next_mid"] = labeled["target_mid"]
        labeled["tau1_recv_seq"] = labeled["tau_k_recv_seq"]
        labeled["tau1_time_ms"] = labeled["tau_k_time_ms"]
        labeled["time_to_tau1_ms"] = labeled["time_to_target_ms"]
        labeled["delta_mid_tau1"] = labeled["delta_mid_target"]
        labeled["direction_tau1"] = labeled["direction_target"]
        labeled["tau1_analysis_ts"] = labeled["tau_k_analysis_ts"]

    return labeled


def label_next_mid_change(frame: pd.DataFrame, *, time_col_ms: str = "recv_time_ms") -> pd.DataFrame:
    return label_kth_mid_change(frame, k=1, time_col_ms=time_col_ms)


def get_or_build_microprice_labeled_table(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: str = "strict",
    event_time_or_recv_time: str = "recv",
    imbalance_bucket_count: int | None = 10,
    imbalance_bucket_edges: list[float] | None = None,
    spread_bucket_values: list[int] | tuple[int, ...] = (1, 2, 3),
    future_move_definition: str = "next_mid_change",
    future_move_k: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    dataset = _ensure_dataset(dataset_or_day_dir)
    resolved_future_move_k = _resolve_future_move_k(
        future_move_definition=future_move_definition,
        future_move_k=future_move_k,
    )
    params = {
        "on_gap": on_gap,
        "event_time_or_recv_time": event_time_or_recv_time,
        "imbalance_bucket_count": imbalance_bucket_count,
        "imbalance_bucket_edges": imbalance_bucket_edges,
        "spread_bucket_values": list(spread_bucket_values),
        "future_move_definition": future_move_definition,
        "future_move_k": resolved_future_move_k,
    }
    path = cache_path(dataset.day_dir, "microprice_labeled", params, ext="parquet")
    if not force:
        cached = _load_valid_cached_labeled_table(path, resolved_future_move_k=resolved_future_move_k)
        if cached is not None:
            return cached

    book_states = get_or_build_top_of_book_table(dataset, on_gap=on_gap, force=force)
    features, _, time_col_ms = add_microprice_features(
        dataset,
        book_states,
        event_time_or_recv_time=event_time_or_recv_time,
        imbalance_bucket_count=imbalance_bucket_count,
        imbalance_bucket_edges=imbalance_bucket_edges,
        spread_bucket_values=spread_bucket_values,
    )
    labeled = label_kth_mid_change(features, k=resolved_future_move_k, time_col_ms=time_col_ms)
    labeled["future_move_definition"] = future_move_definition
    labeled.to_parquet(path, index=False)
    return labeled


def get_or_build_pooled_microprice_labeled_table(
    datasets_or_day_dirs: Iterable[DayDataset | Path],
    *,
    cache_root: Path,
    on_gap: str = "strict",
    event_time_or_recv_time: str = "recv",
    imbalance_bucket_count: int | None = 10,
    imbalance_bucket_edges: list[float] | None = None,
    spread_bucket_values: list[int] | tuple[int, ...] = (1, 2, 3),
    future_move_definition: str = "next_mid_change",
    future_move_k: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    datasets = [_ensure_dataset(item) for item in datasets_or_day_dirs]
    if not datasets:
        raise ValueError("Need at least one dataset or day_dir to build a pooled microprice table")

    exchanges = {dataset.day_dir.parent.parent.name for dataset in datasets}
    symbols = {dataset.day_dir.parent.name for dataset in datasets}
    exchange_key = next(iter(exchanges)) if len(exchanges) == 1 else "mixed"
    symbol_key = next(iter(symbols)) if len(symbols) == 1 else "mixed"
    resolved_future_move_k = _resolve_future_move_k(
        future_move_definition=future_move_definition,
        future_move_k=future_move_k,
    )
    params = {
        "day_dirs": sorted(str(dataset.day_dir.resolve()) for dataset in datasets),
        "on_gap": on_gap,
        "event_time_or_recv_time": event_time_or_recv_time,
        "imbalance_bucket_count": imbalance_bucket_count,
        "imbalance_bucket_edges": imbalance_bucket_edges,
        "spread_bucket_values": list(spread_bucket_values),
        "future_move_definition": future_move_definition,
        "future_move_k": resolved_future_move_k,
    }
    path = analysis_cache_path(
        cache_root,
        "microprice_labeled_pooled",
        params,
        namespace=f"microprice_g1/{exchange_key}/{symbol_key}",
        ext="parquet",
    )
    if not force:
        cached = _load_valid_cached_labeled_table(path, resolved_future_move_k=resolved_future_move_k)
        if cached is not None:
            return cached

    parts = [
        get_or_build_microprice_labeled_table(
            dataset,
            on_gap=on_gap,
            event_time_or_recv_time=event_time_or_recv_time,
            imbalance_bucket_count=imbalance_bucket_count,
            imbalance_bucket_edges=imbalance_bucket_edges,
            spread_bucket_values=spread_bucket_values,
            future_move_definition=future_move_definition,
            future_move_k=resolved_future_move_k,
            force=force,
        )
        for dataset in datasets
    ]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_parquet(path, index=False)
    return out


def estimate_g1_tables(
    labeled: pd.DataFrame,
    *,
    min_obs_per_bucket: int = 200,
    delta_col: str = "delta_mid_target",
    direction_col: str = "direction_target",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g1_table_long = (
        labeled.groupby(["spread_bucket", "imbalance_bucket"], dropna=False)
        .agg(
            g1=(delta_col, "mean"),
            p_up=(direction_col, lambda s: float((s > 0).mean())),
            p_down=(direction_col, lambda s: float((s < 0).mean())),
            obs_count=(delta_col, "size"),
            std_delta=(delta_col, "std"),
            weighted_mid_baseline=("weighted_mid_adjustment", "mean"),
        )
        .reset_index()
    )
    g1_table_long["std_error"] = g1_table_long["std_delta"] / np.sqrt(g1_table_long["obs_count"].clip(lower=1))
    g1_table_long = g1_table_long.loc[g1_table_long["obs_count"] >= min_obs_per_bucket].copy()
    if g1_table_long.empty:
        raise RuntimeError("No state buckets survived min_obs_per_bucket filtering. Lower min_obs_per_bucket or use a busier pooled sample.")
    g1_table_long = g1_table_long.sort_values(["spread_bucket", "imbalance_bucket"])
    g1_table_pivot = g1_table_long.pivot(index="imbalance_bucket", columns="spread_bucket", values="g1")
    diagnostics_table = g1_table_long[["spread_bucket", "imbalance_bucket", "obs_count", "std_error"]].copy()
    return g1_table_long, g1_table_pivot, diagnostics_table
