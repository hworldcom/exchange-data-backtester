from __future__ import annotations

import numpy as np
import pandas as pd
from stats.utils.common import to_utc_datetime_ms


def add_aggressor_sign(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach aggressor direction and signed trade measures.

    Preference order:
    1. Use the explicit `side` column when present.
    2. Fall back to `is_buyer_maker` only for rows where `side` is missing.
    3. Leave direction unknown as `NaN` if neither field is usable.
    """
    out = trades.copy()
    out["aggr_sign"] = np.nan

    if "side" in out.columns:
        side = out["side"].astype("string").str.lower()
        buy_mask = side.eq("buy").fillna(False)
        sell_mask = side.eq("sell").fillna(False)
        side_sign = pd.Series(
            np.where(sell_mask, -1.0, np.where(buy_mask, 1.0, np.nan)),
            index=out.index,
            dtype="float64",
        )
        out.loc[side_sign.notna(), "aggr_sign"] = side_sign[side_sign.notna()]

    if "is_buyer_maker" in out.columns:
        maker = pd.to_numeric(out["is_buyer_maker"], errors="coerce")
        maker_buy_mask = maker.eq(0).fillna(False)
        maker_sell_mask = maker.eq(1).fillna(False)
        maker_sign = pd.Series(
            np.where(maker_sell_mask, -1.0, np.where(maker_buy_mask, 1.0, np.nan)),
            index=out.index,
            dtype="float64",
        )
        fill_mask = out["aggr_sign"].isna() & maker_sign.notna()
        out.loc[fill_mask, "aggr_sign"] = maker_sign[fill_mask]

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
    ts = to_utc_datetime_ms(trades[time_col_ms])
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
