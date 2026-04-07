from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from stats.features.book import compute_mid_spread, resample_book
from stats.features.returns import forward_returns, _timedelta_to_exact_steps


def make_book_grid(book_raw: pd.DataFrame, *, time_col_ms: str, obs_grid: str) -> pd.DataFrame:
    """Resample raw book states onto a fixed observation grid."""
    return resample_book(book_raw, time_col_ms=time_col_ms, grid_freq=obs_grid)


def depth_sums(book_g: pd.DataFrame, *, levels: int) -> tuple[pd.Series, pd.Series]:
    bid_cols = [f"bid{k}_qty" for k in range(1, levels + 1)]
    ask_cols = [f"ask{k}_qty" for k in range(1, levels + 1)]
    missing = [name for name in bid_cols + ask_cols if name not in book_g.columns]
    if missing:
        raise KeyError(f"Missing depth columns for levels={levels}: {missing[:10]}")
    return book_g[bid_cols].sum(axis=1), book_g[ask_cols].sum(axis=1)


def compute_book_core(book_g: pd.DataFrame, *, levels: int) -> pd.DataFrame:
    bid_sum, ask_sum = depth_sums(book_g, levels=levels)
    core = compute_mid_spread(book_g)
    core["B"] = bid_sum
    core["A"] = ask_sum
    core["ratio"] = bid_sum / ask_sum.replace(0, np.nan)
    return core


def compute_obi_features(core: pd.DataFrame) -> pd.DataFrame:
    denom = (core["B"] + core["A"]).replace(0, np.nan)
    safe_b = core["B"].replace(0, np.nan)
    safe_a = core["A"].replace(0, np.nan)
    out = pd.DataFrame(index=core.index)
    out["obi"] = (core["B"] - core["A"]) / denom
    out["share_bid"] = core["B"] / denom
    out["signed_ratio"] = np.where(core["B"] >= core["A"], core["B"] / safe_a, -(core["A"] / safe_b))
    out["log_ratio"] = np.log(core["B"] / safe_a)
    out["snd"] = out["obi"]
    return out


def forward_log_returns(mid: pd.Series, *, obs_grid: str, horizons_ms: Sequence[int]) -> pd.DataFrame:
    """Convenience wrapper for log forward returns on a fixed grid.

    `horizons_ms` must align exactly with `obs_grid`; non-divisible horizons are
    rejected instead of being truncated.
    """
    return forward_returns(mid, horizons_ms=horizons_ms, grid_freq=obs_grid, log=True)


def anchors_by_horizon(df: pd.DataFrame, *, obs_grid: str, h_ms: int) -> pd.DataFrame:
    """Downsample anchors every `h_ms`, requiring exact grid alignment."""
    steps = _timedelta_to_exact_steps(
        pd.Timedelta(milliseconds=int(h_ms)),
        pd.Timedelta(obs_grid),
        label="horizon",
    )
    return df.iloc[::steps].copy()


def anchors_by_decision(df: pd.DataFrame, *, obs_grid: str, decision_interval_ms: int) -> pd.DataFrame:
    """Downsample decision times every fixed interval, requiring exact grid alignment."""
    steps = _timedelta_to_exact_steps(
        pd.Timedelta(milliseconds=int(decision_interval_ms)),
        pd.Timedelta(obs_grid),
        label="decision interval",
    )
    return df.iloc[::steps].copy()


def mask_extreme_ratio(ratio: pd.Series, *, K: float) -> pd.Series:
    return (ratio >= K) | (ratio <= 1.0 / K)


def run_length_same_sign(values: pd.Series) -> pd.Series:
    signs = np.sign(values.astype(float))
    out = np.zeros(len(signs), dtype=np.int32)
    prev = 0.0
    streak = 0
    for idx, sign in enumerate(signs):
        if not np.isfinite(sign) or sign == 0:
            prev = 0.0
            streak = 0
            continue
        if float(sign) == prev and streak > 0:
            streak += 1
        else:
            streak = 1
        prev = float(sign)
        out[idx] = streak
    return pd.Series(out, index=values.index)


def run_length_extreme(values: pd.Series, *, K: float) -> pd.Series:
    threshold = float(np.log(K))
    flags = np.abs(values.astype(float)) >= threshold
    out = np.zeros(len(flags), dtype=np.int32)
    streak = 0
    for idx, flag in enumerate(flags):
        if bool(flag):
            streak += 1
            out[idx] = streak
        else:
            streak = 0
    return pd.Series(out, index=values.index)


def run_length_extreme_direction(values: pd.Series, *, K: float) -> pd.Series:
    threshold = float(np.log(K))
    vals = values.astype(float)
    out = np.zeros(len(vals), dtype=np.int32)
    prev_sign = 0.0
    streak = 0
    for idx, value in enumerate(vals):
        if not np.isfinite(value) or abs(value) < threshold:
            prev_sign = 0.0
            streak = 0
            continue
        sign = float(np.sign(value))
        if sign == 0.0:
            prev_sign = 0.0
            streak = 0
            continue
        if sign == prev_sign and streak > 0:
            streak += 1
        else:
            streak = 1
        prev_sign = sign
        out[idx] = streak
    return pd.Series(out, index=values.index)
