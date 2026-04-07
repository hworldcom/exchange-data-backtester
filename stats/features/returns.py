from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _infer_step_timedelta(index: pd.Index) -> pd.Timedelta:
    """Infer the grid step from a strictly increasing datetime index."""
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer the grid step")
    step = index[1] - index[0]
    if not isinstance(step, pd.Timedelta):
        step = pd.Timedelta(step)
    if step <= pd.Timedelta(0):
        raise ValueError("Time index must be strictly increasing")
    return step


def _timedelta_to_exact_steps(duration: pd.Timedelta, step: pd.Timedelta, *, label: str) -> int:
    """Convert a duration to integer grid steps, rejecting non-aligned horizons.

    This helper intentionally fails fast instead of flooring to the nearest
    smaller bar count. A 150ms horizon on a 100ms grid must not silently become
    a 100ms return.
    """
    if duration <= pd.Timedelta(0):
        raise ValueError(f"{label} must be positive")
    if step <= pd.Timedelta(0):
        raise ValueError("grid step must be positive")

    steps_float = duration / step
    steps_int = int(round(float(steps_float)))
    if steps_int < 1 or not np.isclose(steps_float, steps_int):
        raise ValueError(f"{label}={duration} must be an integer multiple of the grid step={step}")
    return steps_int


def forward_returns(
    mid: pd.Series,
    *,
    horizons_ms: Sequence[int],
    grid_freq: str | None = None,
    log: bool = True,
) -> pd.DataFrame:
    """Compute forward returns on a regular observation grid.

    Every requested horizon must be an exact multiple of the grid step. This is
    a hard requirement so the feature label matches the actual lookahead used in
    the shift.
    """
    if not isinstance(mid.index, pd.DatetimeIndex):
        raise TypeError("mid must be indexed by a DatetimeIndex")
    step = pd.Timedelta(grid_freq) if grid_freq is not None else _infer_step_timedelta(mid.index)

    out = pd.DataFrame(index=mid.index)
    for horizon_ms in horizons_ms:
        horizon = pd.Timedelta(milliseconds=int(horizon_ms))
        steps = _timedelta_to_exact_steps(horizon, step, label="horizon")
        future = mid.shift(-steps)
        if log:
            out[f"fwd_ret_{horizon_ms}ms"] = np.log(future) - np.log(mid)
        else:
            out[f"fwd_ret_{horizon_ms}ms"] = (future / mid) - 1.0
    return out
