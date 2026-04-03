from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _infer_step_timedelta(index: pd.Index) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer the grid step")
    step = index[1] - index[0]
    if not isinstance(step, pd.Timedelta):
        step = pd.Timedelta(step)
    if step <= pd.Timedelta(0):
        raise ValueError("Time index must be strictly increasing")
    return step


def forward_returns(
    mid: pd.Series,
    *,
    horizons_ms: Sequence[int],
    grid_freq: str | None = None,
    log: bool = True,
) -> pd.DataFrame:
    if not isinstance(mid.index, pd.DatetimeIndex):
        raise TypeError("mid must be indexed by a DatetimeIndex")
    step = pd.Timedelta(grid_freq) if grid_freq is not None else _infer_step_timedelta(mid.index)

    out = pd.DataFrame(index=mid.index)
    for horizon_ms in horizons_ms:
        horizon = pd.Timedelta(milliseconds=int(horizon_ms))
        steps = int(horizon / step)
        if steps < 1:
            out[f"fwd_ret_{horizon_ms}ms"] = np.nan
            continue
        future = mid.shift(-steps)
        if log:
            out[f"fwd_ret_{horizon_ms}ms"] = np.log(future) - np.log(mid)
        else:
            out[f"fwd_ret_{horizon_ms}ms"] = (future / mid) - 1.0
    return out
