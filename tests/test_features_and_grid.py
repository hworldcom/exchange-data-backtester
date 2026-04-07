from __future__ import annotations

import pandas as pd
import pytest

from stats.analysis.grid_framework import anchors_by_decision, anchors_by_horizon
from stats.features.returns import forward_returns
from stats.features.trades import add_aggressor_sign
from stats.replay.ofi import rolling_sum_on_grid


def test_add_aggressor_sign_preserves_unknown_direction() -> None:
    """Verify that unknown trade direction stays unknown instead of being forced into a side."""
    trades = pd.DataFrame(
        {
            "side": ["buy", None, ""],
            "is_buyer_maker": [0, 1, pd.NA],
            "price": [100.0, 101.0, 102.0],
            "qty": [1.0, 2.0, 3.0],
        }
    )

    out = add_aggressor_sign(trades)

    assert out["aggr_sign"].iloc[0] == pytest.approx(1.0)
    assert out["aggr_sign"].iloc[1] == pytest.approx(-1.0)
    assert pd.isna(out["aggr_sign"].iloc[2])
    assert list(out["signed_qty"].iloc[:2]) == [1.0, -2.0]
    assert pd.isna(out["signed_qty"].iloc[2])


def test_forward_returns_requires_exact_grid_multiples() -> None:
    """Verify that forward returns only accept horizons aligned to the grid."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T00:00:00.100Z"),
            pd.Timestamp("2026-01-01T00:00:00.200Z"),
        ]
    )
    mid = pd.Series([100.0, 101.0, 102.0], index=idx)

    exact = forward_returns(mid, horizons_ms=[100, 200], grid_freq="100ms", log=False)
    assert exact["fwd_ret_100ms"].iloc[0] == pytest.approx(0.01)
    assert exact["fwd_ret_200ms"].iloc[0] == pytest.approx(0.02)

    with pytest.raises(ValueError, match="must be an integer multiple"):
        forward_returns(mid, horizons_ms=[150], grid_freq="100ms", log=False)


def test_anchor_helpers_require_exact_grid_multiples() -> None:
    """Verify that anchor helpers reject horizons that do not align with the observation grid."""
    frame = pd.DataFrame({"value": [1, 2, 3, 4]})

    anchored = anchors_by_horizon(frame, obs_grid="100ms", h_ms=200)
    assert list(anchored.index) == [0, 2]

    decision = anchors_by_decision(frame, obs_grid="100ms", decision_interval_ms=200)
    assert list(decision.index) == [0, 2]

    with pytest.raises(ValueError, match="must be an integer multiple"):
        anchors_by_horizon(frame, obs_grid="100ms", h_ms=150)

    with pytest.raises(ValueError, match="must be an integer multiple"):
        anchors_by_decision(frame, obs_grid="100ms", decision_interval_ms=150)


def test_rolling_sum_on_grid_requires_exact_grid_multiples() -> None:
    """Verify that rolling OFI windows must align exactly to the grid."""
    idx = pd.date_range("2026-01-01", periods=4, freq="100ms", tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    exact = rolling_sum_on_grid(series, window_ms=200, grid_freq="100ms")
    assert exact.tolist() == [1.0, 3.0, 5.0, 7.0]

    with pytest.raises(ValueError, match="must be an integer multiple"):
        rolling_sum_on_grid(series, window_ms=150, grid_freq="100ms")
