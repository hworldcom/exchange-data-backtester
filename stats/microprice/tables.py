"""Cached microprice tables and g1 estimation built on replayed book states."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from stats.io import DayDataset
from stats.tables import get_or_build_top_of_book_table
from stats.utils.cache import analysis_cache_path, cache_path, load_or_build_parquet
from stats.utils.common import ensure_dataset

from .features import add_microprice_features
from .labeling import label_kth_mid_change, resolve_future_move_k


def required_labeled_columns(resolved_future_move_k: int) -> set[str]:
    """Return the required schema for cached labeled microprice tables."""
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


def has_required_labeled_columns(frame: pd.DataFrame, *, resolved_future_move_k: int) -> bool:
    """Return whether a labeled microprice cache has the required schema."""
    return required_labeled_columns(resolved_future_move_k) <= set(frame.columns)


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
    """Load or build a single-day labeled microprice table."""
    dataset = ensure_dataset(dataset_or_day_dir)
    resolved_future_move_k = resolve_future_move_k(
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

    def build() -> pd.DataFrame:
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
        return labeled

    return load_or_build_parquet(
        path,
        build=build,
        force=force,
        required_columns=required_labeled_columns(resolved_future_move_k),
        validator=lambda frame: has_required_labeled_columns(frame, resolved_future_move_k=resolved_future_move_k),
        index=False,
    )


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
    """Load or build a pooled labeled microprice table across multiple days."""
    datasets = [ensure_dataset(item) for item in datasets_or_day_dirs]
    if not datasets:
        raise ValueError("Need at least one dataset or day_dir to build a pooled microprice table")

    exchanges = {dataset.day_dir.parent.parent.name for dataset in datasets}
    symbols = {dataset.day_dir.parent.name for dataset in datasets}
    exchange_key = next(iter(exchanges)) if len(exchanges) == 1 else "mixed"
    symbol_key = next(iter(symbols)) if len(symbols) == 1 else "mixed"
    resolved_future_move_k = resolve_future_move_k(
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

    def build() -> pd.DataFrame:
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
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return load_or_build_parquet(
        path,
        build=build,
        force=force,
        required_columns=required_labeled_columns(resolved_future_move_k),
        validator=lambda frame: has_required_labeled_columns(frame, resolved_future_move_k=resolved_future_move_k),
        index=False,
    )


def estimate_g1_tables(
    labeled: pd.DataFrame,
    *,
    min_obs_per_bucket: int = 200,
    delta_col: str = "delta_mid_target",
    direction_col: str = "direction_target",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate g1 and related diagnostics by spread and imbalance state."""
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
        raise RuntimeError(
            "No state buckets survived min_obs_per_bucket filtering. Lower min_obs_per_bucket or use a busier pooled sample."
        )
    g1_table_long = g1_table_long.sort_values(["spread_bucket", "imbalance_bucket"])
    g1_table_pivot = g1_table_long.pivot(index="imbalance_bucket", columns="spread_bucket", values="g1")
    diagnostics_table = g1_table_long[["spread_bucket", "imbalance_bucket", "obs_count", "std_error"]].copy()
    return g1_table_long, g1_table_pivot, diagnostics_table
