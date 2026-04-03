from __future__ import annotations

import numpy as np
import pandas as pd


def add_aggressor_sign(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    if "side" in out.columns:
        side = out["side"].astype(str).str.lower()
        out["aggr_sign"] = np.where(side == "sell", -1.0, np.where(side == "buy", 1.0, np.nan))
    elif "is_buyer_maker" in out.columns:
        is_buyer_maker = out["is_buyer_maker"].astype(bool)
        out["aggr_sign"] = np.where(is_buyer_maker, -1.0, 1.0)
    else:
        out["aggr_sign"] = np.nan
    out["signed_qty"] = out["aggr_sign"] * out["qty"]
    out["notional"] = out["price"] * out["qty"]
    out["signed_notional"] = out["aggr_sign"] * out["notional"]
    return out


def trades_to_bars(
    trades: pd.DataFrame,
    *,
    grid_freq: str = "100ms",
    time_col_ms: str = "recv_time_ms",
) -> pd.DataFrame:
    if "signed_qty" not in trades.columns or "signed_notional" not in trades.columns:
        trades = add_aggressor_sign(trades)
    ts = pd.to_datetime(trades[time_col_ms].astype("int64"), unit="ms", utc=True)
    frame = trades.assign(ts=ts).sort_values("ts").set_index("ts")
    bars = frame.resample(grid_freq).agg(
        trade_count=("qty", "size"),
        total_qty=("qty", "sum"),
        signed_qty=("signed_qty", "sum"),
        total_notional=("notional", "sum"),
        signed_notional=("signed_notional", "sum"),
    )
    bars["tfi_qty"] = bars["signed_qty"] / bars["total_qty"].replace(0, np.nan)
    bars["tfi_notional"] = bars["signed_notional"] / bars["total_notional"].replace(0, np.nan)
    return bars
