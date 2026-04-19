from __future__ import annotations

import pandas as pd
import pytest

from stats.analysis.grid_framework import anchors_by_decision, anchors_by_horizon
from stats.features.returns import forward_returns
from stats.features.trades import add_aggressor_sign, make_trade_frame, normalize_trade_side
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


def test_normalize_trade_side_uses_consistent_precedence() -> None:
    """Verify that all trade-side sources normalize through one precedence order."""
    trades = pd.DataFrame(
        {
            "side": ["buy", None, None, None, None],
            "trade_side": ["sell", "sell", None, None, None],
            "is_buyer_maker": [1, 0, 0, pd.NA, pd.NA],
            "aggr_sign": [-1.0, 1.0, -1.0, -1.0, 0.0],
        }
    )

    out = normalize_trade_side(trades)

    assert out["aggr_sign"].tolist()[:4] == [1.0, -1.0, 1.0, -1.0]
    assert pd.isna(out["aggr_sign"].iloc[4])
    assert out["trade_side"].tolist()[:4] == ["buy", "sell", "buy", "sell"]
    assert out["trade_side"].iloc[4] is None


def test_make_trade_frame_attaches_default_diagnostics() -> None:
    """Verify that trade alignment reports dropped rows without changing the return type."""
    trades = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:00.000Z",
                    "2026-01-01T00:00:00.100Z",
                    "2026-01-01T00:00:00.200Z",
                    "2026-01-01T00:00:00.300Z",
                ],
                utc=True,
            ),
            "price": [100.0, 101.0, 102.0, 103.0],
            "qty": [1.0, 2.0, float("nan"), 4.0],
            "aggr_sign": [1.0, float("nan"), -1.0, 1.0],
        }
    )
    top = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00:00.050Z", "2026-01-01T00:00:00.250Z"], utc=True),
            "mid": [100.5, 102.5],
        }
    )

    out = make_trade_frame(trades, top)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert out["price"].iloc[0] == pytest.approx(103.0)
    assert out.attrs["diagnostics"] == {
        "input_trades": 4,
        "input_book_rows": 2,
        "usable_book_rows": 2,
        "matched_book_mid": 3,
        "missing_book_mid": 1,
        "missing_aggr_sign": 1,
        "missing_qty": 1,
        "dropped_rows": 3,
        "output_rows": 1,
        "max_book_staleness": None,
    }


def test_make_trade_frame_respects_max_book_staleness() -> None:
    """Verify that stale book matches are treated as missing when a tolerance is provided."""
    trades = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00:00.100Z", "2026-01-01T00:00:00.300Z"], utc=True),
            "price": [100.0, 101.0],
            "qty": [1.0, 2.0],
            "aggr_sign": [1.0, -1.0],
        }
    )
    top = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00:00.000Z"], utc=True),
            "mid": [100.5],
        }
    )

    out = make_trade_frame(trades, top, max_book_staleness="150ms")

    assert len(out) == 1
    assert out["price"].iloc[0] == pytest.approx(100.0)
    assert out.attrs["diagnostics"]["missing_book_mid"] == 1
    assert out.attrs["diagnostics"]["dropped_rows"] == 1
    assert out.attrs["diagnostics"]["max_book_staleness"] == str(pd.Timedelta("150ms"))


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
