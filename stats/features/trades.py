from __future__ import annotations

import numpy as np
import pandas as pd
from stats.utils.common import to_utc_datetime_ms


def _side_to_sign(values: pd.Series) -> pd.Series:
    side = values.astype("string").str.lower()
    buy_mask = side.eq("buy").fillna(False)
    sell_mask = side.eq("sell").fillna(False)
    return pd.Series(
        np.where(sell_mask, -1.0, np.where(buy_mask, 1.0, np.nan)),
        index=values.index,
        dtype="float64",
    )


def normalize_trade_side(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical aggressor sign and side columns.

    Preference order:
    1. Use the explicit `side` column when present.
    2. Use an existing `trade_side` column for rows where `side` is missing.
    3. Fall back to `is_buyer_maker` for rows where side fields are missing.
    4. Use an existing `aggr_sign` only when no side source is usable.
    5. Leave direction unknown as `NaN` if no field is usable.
    """
    out = trades.copy()
    aggr_sign = pd.Series(np.nan, index=out.index, dtype="float64")

    if "side" in out.columns:
        side_sign = _side_to_sign(out["side"])
        aggr_sign.loc[side_sign.notna()] = side_sign[side_sign.notna()]

    if "trade_side" in out.columns:
        trade_side_sign = _side_to_sign(out["trade_side"])
        fill_mask = aggr_sign.isna() & trade_side_sign.notna()
        aggr_sign.loc[fill_mask] = trade_side_sign[fill_mask]

    if "is_buyer_maker" in out.columns:
        maker = pd.to_numeric(out["is_buyer_maker"], errors="coerce")
        maker_buy_mask = maker.eq(0).fillna(False)
        maker_sell_mask = maker.eq(1).fillna(False)
        maker_sign = pd.Series(
            np.where(maker_sell_mask, -1.0, np.where(maker_buy_mask, 1.0, np.nan)),
            index=out.index,
            dtype="float64",
        )
        fill_mask = aggr_sign.isna() & maker_sign.notna()
        aggr_sign.loc[fill_mask] = maker_sign[fill_mask]

    if "aggr_sign" in out.columns:
        existing_sign = pd.to_numeric(out["aggr_sign"], errors="coerce")
        existing_sign = pd.Series(
            np.where(existing_sign > 0, 1.0, np.where(existing_sign < 0, -1.0, np.nan)),
            index=out.index,
            dtype="float64",
        )
        fill_mask = aggr_sign.isna() & existing_sign.notna()
        aggr_sign.loc[fill_mask] = existing_sign[fill_mask]

    out["aggr_sign"] = aggr_sign
    out["trade_side"] = np.where(aggr_sign > 0, "buy", np.where(aggr_sign < 0, "sell", None))
    return out


def add_aggressor_sign(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach aggressor direction and signed trade measures."""
    out = normalize_trade_side(trades)

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


def make_trade_frame(
    trades_df: pd.DataFrame,
    top_df: pd.DataFrame,
    *,
    include_log_mid: bool = False,
    include_trade_idx: bool = False,
    diagnostics: bool = True,
    max_book_staleness: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    trade_cols = ["ts", "price", "qty", "aggr_sign"]
    missing = [c for c in trade_cols if c not in trades_df.columns]
    if missing:
        raise KeyError(f"missing trade columns: {missing}")

    if "ts" not in top_df.columns or "mid" not in top_df.columns:
        raise KeyError("missing top-of-book columns: ['ts', 'mid']")

    book = top_df[["ts", "mid"]].dropna().sort_values("ts").rename(columns={"mid": "mid_at_book"})
    trade_frame = trades_df[trade_cols].copy().sort_values("ts")
    trade_frame["ts"] = pd.to_datetime(trade_frame["ts"], utc=True)
    book["ts"] = pd.to_datetime(book["ts"], utc=True)

    tolerance = None if max_book_staleness is None else pd.Timedelta(max_book_staleness)
    aligned = pd.merge_asof(trade_frame, book, on="ts", direction="backward", tolerance=tolerance)
    missing_book_mid = aligned["mid_at_book"].isna()
    missing_aggr_sign = aligned["aggr_sign"].isna()
    missing_qty = aligned["qty"].isna()
    keep_mask = ~(missing_book_mid | missing_aggr_sign | missing_qty)

    diagnostic_summary = {
        "input_trades": int(len(trades_df)),
        "input_book_rows": int(len(top_df)),
        "usable_book_rows": int(len(book)),
        "matched_book_mid": int((~missing_book_mid).sum()),
        "missing_book_mid": int(missing_book_mid.sum()),
        "missing_aggr_sign": int(missing_aggr_sign.sum()),
        "missing_qty": int(missing_qty.sum()),
        "dropped_rows": int((~keep_mask).sum()),
        "output_rows": int(keep_mask.sum()),
        "max_book_staleness": None if tolerance is None else str(tolerance),
    }

    aligned = aligned.loc[keep_mask].reset_index(drop=True)
    aligned["aggr_sign"] = aligned["aggr_sign"].astype(float)
    aligned["qty"] = aligned["qty"].astype(float)
    if include_log_mid:
        aligned["log_mid"] = np.log(aligned["mid_at_book"])
    if include_trade_idx:
        aligned["trade_idx"] = np.arange(len(aligned))
    if diagnostics:
        aligned.attrs["diagnostics"] = diagnostic_summary
    return aligned
