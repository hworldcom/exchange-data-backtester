"""Shared helpers for replayed book queues, price tracking, and trade alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stats.features.trades import normalize_trade_side


BOOK_SIDE_TO_AGGRESSOR = {"bid": "sell", "ask": "buy"}
AGGRESSOR_TO_BOOK_SIDE = {"buy": "ask", "sell": "bid"}


@dataclass(frozen=True)
class _BookWindowSideState:
    visible_count: np.ndarray
    worst_price: np.ndarray


@dataclass(frozen=True)
class _BookWindowState:
    bid: _BookWindowSideState
    ask: _BookWindowSideState


def _book_level_columns(side: str, max_level: int) -> list[str]:
    """Return the expected `price`/`qty` column names for one book side."""
    return [f"{side}{level}_{suffix}" for level in range(1, max_level + 1) for suffix in ("price", "qty")]


def cumulative_depth(book_row: pd.Series, *, side: str, level: int) -> float:
    """Sum visible quantity from level 1 through `level` for one side of one book row."""
    if side not in {"bid", "ask"}:
        raise ValueError("side must be 'bid' or 'ask'")
    if level <= 0:
        raise ValueError("level must be positive")

    cols = [f"{side}{idx}_qty" for idx in range(1, level + 1)]
    missing = [name for name in cols if name not in book_row.index]
    if missing:
        raise KeyError(f"Missing depth columns for side={side}, level={level}: {missing[:8]}")
    return float(pd.to_numeric(book_row[cols], errors="coerce").fillna(0.0).clip(lower=0.0).sum())


def build_layer_event_stream(book_levels: pd.DataFrame, trades: pd.DataFrame, *, time_col_ms: str = "recv_time_ms") -> pd.DataFrame:
    """Merge replayed book rows and trades into one recv-sequence ordered event stream.

    The output keeps book states as `event_type="book"` rows and trades as
    `event_type="trade"` rows. It also adds cumulative aggressive buy/sell
    volume so later functions can ask when a given visible queue could have been
    consumed by future trade flow.
    """
    if not book_levels.empty and time_col_ms not in book_levels.columns:
        raise KeyError(f"book_levels missing time column: {time_col_ms!r}")
    if not trades.empty and time_col_ms not in trades.columns:
        raise KeyError(f"trades missing time column: {time_col_ms!r}")

    book = book_levels.copy()
    book["event_type"] = "book"
    book["event_order"] = 0
    book["price"] = np.nan
    book["qty"] = np.nan
    book["trade_side"] = None
    book["aggr_sign"] = np.nan
    book["trade_id"] = np.nan
    book["trade_time_ms"] = np.nan
    book["trade_count"] = 0

    trade = normalize_trade_side(trades)
    trade["event_type"] = "trade"
    trade["event_order"] = 1
    trade["trade_count"] = 1

    keep_book = [
        "event_type",
        "event_order",
        "event_time_ms",
        "recv_time_ms",
        "recv_seq",
        "epoch_id",
        "segment_index",
        "segment_tag",
    ] + [col for col in book.columns if col.startswith(("bid", "ask"))]
    if time_col_ms not in keep_book:
        keep_book.append(time_col_ms)

    keep_trade = [
        "event_type",
        "event_order",
        "event_time_ms",
        "recv_time_ms",
        "recv_seq",
        "trade_time_ms",
        "trade_id",
        "price",
        "qty",
        "trade_side",
        "aggr_sign",
        "trade_count",
        "run_id",
        "exchange",
        "symbol",
    ]
    if time_col_ms not in keep_trade:
        keep_trade.append(time_col_ms)
    if "ord_type" in trade.columns:
        keep_trade.append("ord_type")

    book_out = book.loc[:, [col for col in keep_book if col in book.columns or col.startswith(("bid", "ask"))]].copy()
    trade_out = trade.loc[:, [col for col in keep_trade if col in trade.columns]].copy()

    stream = pd.concat([book_out, trade_out], ignore_index=True, sort=False)
    sort_cols = ["recv_seq", "event_order"]
    if time_col_ms in stream.columns and time_col_ms not in sort_cols:
        sort_cols.append(time_col_ms)
    if "trade_time_ms" in stream.columns:
        sort_cols.append("trade_time_ms")
    if "trade_id" in stream.columns:
        sort_cols.append("trade_id")
    stream = stream.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    stream["stream_pos"] = np.arange(len(stream), dtype=np.int64)

    trade_mask = stream["event_type"] == "trade"
    stream["buy_qty"] = np.where(trade_mask & (stream["trade_side"] == "buy"), stream["qty"].astype(float), 0.0)
    stream["sell_qty"] = np.where(trade_mask & (stream["trade_side"] == "sell"), stream["qty"].astype(float), 0.0)
    stream["cum_buy_qty"] = stream["buy_qty"].cumsum()
    stream["cum_sell_qty"] = stream["sell_qty"].cumsum()
    stream["trade_event_count"] = trade_mask.astype(np.int64)
    stream["cum_trade_event_count"] = stream["trade_event_count"].cumsum()
    return stream


def _infer_tracked_top_n(book_levels: pd.DataFrame) -> int:
    """Infer the deepest tracked level from `bid*`/`ask*` price columns."""
    levels: list[int] = []
    for col in book_levels.columns:
        if col.startswith(("bid", "ask")) and col.endswith("_price"):
            prefix = col[:-6]
            digits = "".join(ch for ch in prefix if ch.isdigit())
            if digits:
                levels.append(int(digits))
    if not levels:
        raise KeyError("book_levels must include bid*/ask* price columns")
    return max(levels)


def _book_window_side_state(book_levels: pd.DataFrame, *, side: str, tracked_top_n: int) -> _BookWindowSideState:
    """Summarize visible depth-window coverage for one side of the book."""
    n_rows = len(book_levels)
    price_cols = [
        f"{side}{level}_price"
        for level in range(1, tracked_top_n + 1)
        if f"{side}{level}_price" in book_levels.columns
    ]
    if not price_cols:
        return _BookWindowSideState(
            visible_count=np.zeros(n_rows, dtype=np.int64),
            worst_price=np.full(n_rows, np.nan),
        )

    prices = book_levels[price_cols].apply(pd.to_numeric, errors="coerce")
    visible_count = prices.notna().sum(axis=1).to_numpy(dtype=np.int64)
    if side == "bid":
        worst_price = prices.min(axis=1, skipna=True).to_numpy(dtype=float)
    elif side == "ask":
        worst_price = prices.max(axis=1, skipna=True).to_numpy(dtype=float)
    else:
        raise ValueError("side must be 'bid' or 'ask'")
    return _BookWindowSideState(visible_count=visible_count, worst_price=worst_price)


def _build_book_window_state(book_levels: pd.DataFrame, *, tracked_top_n: int) -> _BookWindowState:
    """Build per-row visibility state for both sides of a tracked book window."""
    return _BookWindowState(
        bid=_book_window_side_state(book_levels, side="bid", tracked_top_n=tracked_top_n),
        ask=_book_window_side_state(book_levels, side="ask", tracked_top_n=tracked_top_n),
    )


def _build_initial_price_observations(book_levels: pd.DataFrame, *, max_initial_level: int) -> pd.DataFrame:
    """Create one starting observation per book row, side, and initial level.

    Each output row represents the exact price and quantity seen at `t0` for one
    initial level. Later logic tracks that same price forward through time even
    if it reranks to another level.
    """
    n_rows = len(book_levels)
    book_row = np.arange(n_rows, dtype=np.int64)
    rows: list[pd.DataFrame] = []
    for side, aggressor_side in BOOK_SIDE_TO_AGGRESSOR.items():
        for level in range(1, max_initial_level + 1):
            price_col = f"{side}{level}_price"
            qty_col = f"{side}{level}_qty"
            if price_col not in book_levels.columns or qty_col not in book_levels.columns:
                continue
            part = pd.DataFrame(
                {
                    "book_row": book_row,
                    "book_recv_seq": pd.to_numeric(book_levels["recv_seq"], errors="coerce").astype("Int64"),
                    "book_time_ms": pd.to_numeric(book_levels["recv_time_ms"], errors="coerce").astype("Int64"),
                    "book_side": side,
                    "aggressor_side": aggressor_side,
                    "level": level,
                    "initial_price": pd.to_numeric(book_levels[price_col], errors="coerce"),
                    "initial_qty": pd.to_numeric(book_levels[qty_col], errors="coerce"),
                }
            )
            part = part[np.isfinite(part["initial_price"]) & (part["initial_qty"] > 0)].copy()
            rows.append(part)
    if not rows:
        return pd.DataFrame(
            columns=[
                "book_row",
                "book_recv_seq",
                "book_time_ms",
                "book_side",
                "aggressor_side",
                "level",
                "initial_price",
                "initial_qty",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["book_recv_seq"] = out["book_recv_seq"].astype(np.int64)
    out["book_time_ms"] = out["book_time_ms"].astype(np.int64)
    out["level"] = out["level"].astype(np.int64)
    return out


def _build_price_presence_runs(book_levels: pd.DataFrame, *, tracked_top_n: int) -> pd.DataFrame:
    """Record contiguous runs where a given side/price stays visible in the tracked book."""
    n_rows = len(book_levels)
    book_row = np.arange(n_rows, dtype=np.int64)
    frames: list[pd.DataFrame] = []
    for side in ("bid", "ask"):
        for level in range(1, tracked_top_n + 1):
            price_col = f"{side}{level}_price"
            if price_col not in book_levels.columns:
                continue
            part = pd.DataFrame(
                {
                    "book_row": book_row,
                    "book_side": side,
                    "price": pd.to_numeric(book_levels[price_col], errors="coerce"),
                }
            )
            part = part[np.isfinite(part["price"])].copy()
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["book_row", "book_side", "price", "run_end_row"])

    presence = pd.concat(frames, ignore_index=True).drop_duplicates(["book_row", "book_side", "price"])
    presence = presence.sort_values(["book_side", "price", "book_row"], kind="mergesort").reset_index(drop=True)
    gap = presence.groupby(["book_side", "price"], sort=False)["book_row"].diff().fillna(1).ne(1)
    presence["run_id"] = gap.groupby([presence["book_side"], presence["price"]]).cumsum()
    presence["run_end_row"] = presence.groupby(["book_side", "price", "run_id"], sort=False)["book_row"].transform("max")
    return presence.loc[:, ["book_row", "book_side", "price", "run_end_row"]]


def _build_cumulative_trades_by_interval(
    book_levels: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate trade quantity by price into book-to-book intervals.

    Trades are assigned to the latest replayed book row at or before their
    `recv_seq`, then cumulatively summed by aggressor side and price. This lets
    later functions ask how much aggressive flow hit a specific price between
    two book observations.
    """
    if trades is None or trades.empty or book_levels.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "cum_qty"])

    recv_seq = pd.to_numeric(book_levels["recv_seq"], errors="coerce").to_numpy(dtype=np.int64)
    trade = normalize_trade_side(trades)

    trade["price"] = pd.to_numeric(trade["price"], errors="coerce")
    trade["qty"] = pd.to_numeric(trade["qty"], errors="coerce")
    trade["recv_seq"] = pd.to_numeric(trade["recv_seq"], errors="coerce")
    trade = trade[np.isfinite(trade["price"]) & (trade["qty"] > 0) & np.isfinite(trade["recv_seq"])].copy()
    if trade.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "cum_qty"])

    trade["interval_row"] = np.searchsorted(recv_seq, trade["recv_seq"].to_numpy(dtype=np.int64), side="right") - 1
    trade = trade[trade["interval_row"] >= 0].copy()
    if trade.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "cum_qty"])

    grouped = (
        trade.groupby(["trade_side", "price", "interval_row"], dropna=True, sort=True)["qty"]
        .sum()
        .reset_index()
        .rename(columns={"trade_side": "aggressor_side"})
    )
    grouped["cum_qty"] = grouped.groupby(["aggressor_side", "price"], sort=False)["qty"].cumsum()
    return grouped.loc[:, ["aggressor_side", "price", "interval_row", "cum_qty"]]


def _normalize_trade_rows(trades: pd.DataFrame, *, time_col_ms: str = "recv_time_ms") -> pd.DataFrame:
    """Normalize trades into a common side/price/qty/time schema."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "qty", "recv_seq", "event_time_ms", "trade_id"])

    trade = normalize_trade_side(trades)

    event_time_col = time_col_ms if time_col_ms in trade.columns else "recv_time_ms"
    trade["aggressor_side"] = trade["trade_side"]
    trade["price"] = pd.to_numeric(trade["price"], errors="coerce")
    trade["qty"] = pd.to_numeric(trade["qty"], errors="coerce")
    trade["recv_seq"] = pd.to_numeric(trade["recv_seq"], errors="coerce")
    trade["event_time_ms"] = pd.to_numeric(trade[event_time_col], errors="coerce")
    if "trade_id" not in trade.columns:
        trade["trade_id"] = np.arange(len(trade), dtype=np.int64)
    trade = trade[
        trade["aggressor_side"].isin(["buy", "sell"])
        & np.isfinite(trade["price"])
        & (trade["qty"] > 0)
        & np.isfinite(trade["recv_seq"])
        & np.isfinite(trade["event_time_ms"])
    ].copy()
    return trade.loc[:, ["aggressor_side", "price", "qty", "recv_seq", "event_time_ms", "trade_id"]]


def _build_trade_events(
    book_levels: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    time_col_ms: str = "recv_time_ms",
) -> pd.DataFrame:
    """Build per-trade consumption events keyed by aggressor side and exact price."""
    if book_levels.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"])

    trade = _normalize_trade_rows(trades, time_col_ms=time_col_ms)
    if trade.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"])

    recv_seq = pd.to_numeric(book_levels["recv_seq"], errors="coerce").to_numpy(dtype=np.int64)
    trade["interval_row"] = np.searchsorted(recv_seq, trade["recv_seq"].to_numpy(dtype=np.int64), side="right") - 1
    trade = trade[trade["interval_row"] >= 0].copy()
    if trade.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"])

    trade["event_kind"] = "trade"
    trade = trade.sort_values(["aggressor_side", "price", "interval_row", "event_time_ms", "trade_id"], kind="mergesort")
    return trade.rename(columns={"qty": "event_qty"}).loc[
        :,
        ["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"],
    ]


def _build_price_quantity_rows(book_levels: pd.DataFrame, *, tracked_top_n: int) -> pd.DataFrame:
    """Expand the tracked book into one row per visible side/price/qty observation."""
    n_rows = len(book_levels)
    book_row = np.arange(n_rows, dtype=np.int64)
    frames: list[pd.DataFrame] = []
    for side in ("bid", "ask"):
        for level in range(1, tracked_top_n + 1):
            price_col = f"{side}{level}_price"
            qty_col = f"{side}{level}_qty"
            if price_col not in book_levels.columns or qty_col not in book_levels.columns:
                continue
            part = pd.DataFrame(
                {
                    "book_row": book_row,
                    "book_side": side,
                    "price": pd.to_numeric(book_levels[price_col], errors="coerce"),
                    "qty": pd.to_numeric(book_levels[qty_col], errors="coerce"),
                }
            )
            part = part[np.isfinite(part["price"]) & (part["qty"] > 0)].copy()
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["book_row", "book_side", "price", "qty"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["book_row", "book_side", "price"], keep="first")
