"""Price-level survival tracking and queue-consumption timing proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .book_queue_common import (
    BOOK_SIDE_TO_AGGRESSOR,
    _build_cumulative_trades_by_interval,
    _build_initial_price_observations,
    _build_price_presence_runs,
    _build_price_quantity_rows,
    _build_trade_events,
    _infer_tracked_top_n,
)


def _build_nontrade_queue_events(
    book_levels: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    tracked_top_n: int,
    time_col_ms: str = "recv_time_ms",
) -> pd.DataFrame:
    """Infer interval-level non-trade removals at exact prices.

    For each price visible in one book row, this compares its quantity to the
    next book row. If the next quantity is known, the reduction that is not
    explained by aggressive trades at that price is emitted as a `nontrade`
    queue-consumption event at the next book timestamp.
    """
    if len(book_levels) < 2:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"])

    qty_rows = _build_price_quantity_rows(book_levels, tracked_top_n=tracked_top_n)
    if qty_rows.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"])

    trade_events = _build_trade_events(book_levels, trades, time_col_ms=time_col_ms)
    if trade_events.empty:
        trade_interval = pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "trade_qty"])
    else:
        trade_interval = (
            trade_events.groupby(["aggressor_side", "price", "interval_row"], sort=False)["event_qty"]
            .sum()
            .reset_index()
            .rename(columns={"event_qty": "trade_qty"})
        )

    n_rows = len(book_levels)
    time_values = pd.to_numeric(book_levels[time_col_ms], errors="coerce").to_numpy(dtype=np.int64)
    bid_price_cols = [f"bid{level}_price" for level in range(1, tracked_top_n + 1) if f"bid{level}_price" in book_levels.columns]
    ask_price_cols = [f"ask{level}_price" for level in range(1, tracked_top_n + 1) if f"ask{level}_price" in book_levels.columns]
    bid_prices = book_levels[bid_price_cols].apply(pd.to_numeric, errors="coerce") if bid_price_cols else pd.DataFrame(index=book_levels.index)
    ask_prices = book_levels[ask_price_cols].apply(pd.to_numeric, errors="coerce") if ask_price_cols else pd.DataFrame(index=book_levels.index)
    bid_visible_count = bid_prices.notna().sum(axis=1).to_numpy(dtype=np.int64) if not bid_prices.empty else np.zeros(n_rows, dtype=np.int64)
    ask_visible_count = ask_prices.notna().sum(axis=1).to_numpy(dtype=np.int64) if not ask_prices.empty else np.zeros(n_rows, dtype=np.int64)
    bid_worst = bid_prices.min(axis=1, skipna=True).to_numpy(dtype=float) if not bid_prices.empty else np.full(n_rows, np.nan)
    ask_worst = ask_prices.max(axis=1, skipna=True).to_numpy(dtype=float) if not ask_prices.empty else np.full(n_rows, np.nan)

    left = qty_rows[qty_rows["book_row"] < n_rows - 1].copy()
    left["next_row"] = left["book_row"] + 1
    right = qty_rows.rename(columns={"book_row": "next_row", "qty": "next_qty"}).loc[:, ["next_row", "book_side", "price", "next_qty"]]
    events = left.merge(right, on=["next_row", "book_side", "price"], how="left")
    events["aggressor_side"] = events["book_side"].map(BOOK_SIDE_TO_AGGRESSOR)
    events = events.merge(
        trade_interval,
        left_on=["aggressor_side", "price", "book_row"],
        right_on=["aggressor_side", "price", "interval_row"],
        how="left",
    )
    events["trade_qty"] = events["trade_qty"].fillna(0.0)

    next_rows = events["next_row"].to_numpy(dtype=np.int64)
    prices = events["price"].to_numpy(dtype=float)
    next_qty = pd.to_numeric(events["next_qty"], errors="coerce").to_numpy(dtype=float)
    visible_next = np.isfinite(next_qty)
    is_bid = events["book_side"].to_numpy(dtype=object) == "bid"

    conclusive_absent_bid = ~visible_next & is_bid & (
        (bid_visible_count[next_rows] < tracked_top_n) | (prices > bid_worst[next_rows])
    )
    conclusive_absent_ask = ~visible_next & (~is_bid) & (
        (ask_visible_count[next_rows] < tracked_top_n) | (prices < ask_worst[next_rows])
    )
    conclusive_absent = conclusive_absent_bid | conclusive_absent_ask
    next_qty_known = np.where(visible_next, next_qty, np.where(conclusive_absent, 0.0, np.nan))

    prev_qty = pd.to_numeric(events["qty"], errors="coerce").to_numpy(dtype=float)
    trade_qty = pd.to_numeric(events["trade_qty"], errors="coerce").to_numpy(dtype=float)
    nontrade_qty = np.where(
        np.isfinite(next_qty_known),
        np.maximum(prev_qty - next_qty_known - trade_qty, 0.0),
        np.nan,
    )

    events["event_qty"] = nontrade_qty
    events["event_time_ms"] = time_values[next_rows]
    events["event_kind"] = "nontrade"
    events = events[np.isfinite(events["event_qty"]) & (events["event_qty"] > 0)].copy()
    if events.empty:
        return pd.DataFrame(columns=["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"])

    if "interval_row" in events.columns:
        events = events.drop(columns=["interval_row"])

    return events.rename(columns={"book_row": "interval_row"}).loc[
        :,
        ["aggressor_side", "price", "interval_row", "event_time_ms", "event_qty", "event_kind"],
    ]


def _apply_match_times(
    outcomes: pd.DataFrame,
    events: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    """Attach first-passage match times for one event stream to the outcomes table."""
    matched_col = f"{prefix}_matched"
    matched_time_col = f"{prefix}_match_time_ms"
    matched_delay_col = f"time_to_{prefix}_matched_ms"
    outcomes[matched_col] = False
    outcomes[matched_time_col] = np.nan
    outcomes[matched_delay_col] = np.nan

    if outcomes.empty or events.empty:
        return

    event_kind_order = {"trade": 0, "nontrade": 1}
    events = events.copy()
    events["event_kind_order"] = events["event_kind"].map(event_kind_order).fillna(99).astype(int)
    events = events.sort_values(
        ["aggressor_side", "price", "interval_row", "event_time_ms", "event_kind_order"],
        kind="mergesort",
    )
    events["cum_qty"] = events.groupby(["aggressor_side", "price"], sort=False)["event_qty"].cumsum()

    event_groups = {
        key: (
            group["interval_row"].to_numpy(dtype=np.int64),
            group["event_time_ms"].to_numpy(dtype=np.int64),
            group["cum_qty"].to_numpy(dtype=float),
        )
        for key, group in events.groupby(["aggressor_side", "price"], sort=False)
    }

    matched_mask = np.zeros(len(outcomes), dtype=bool)
    matched_time = np.full(len(outcomes), np.nan)

    for key, idx in outcomes.groupby(["aggressor_side", "initial_price"], sort=False).groups.items():
        if key not in event_groups:
            continue
        indexer = np.asarray(list(idx), dtype=np.int64)
        interval_rows, event_time_ms, cum_qty = event_groups[key]

        starts = outcomes.iloc[indexer]["start_interval"].to_numpy(dtype=np.int64)
        ends = outcomes.iloc[indexer]["end_interval"].to_numpy(dtype=np.int64)
        initial_qty = outcomes.iloc[indexer]["initial_qty"].to_numpy(dtype=float)

        base_pos = np.searchsorted(interval_rows, starts - 1, side="right") - 1
        end_pos = np.searchsorted(interval_rows, ends, side="right") - 1
        base_cum = np.where(base_pos >= 0, cum_qty[np.clip(base_pos, 0, None)], 0.0)
        target = base_cum + initial_qty
        cross_pos = np.searchsorted(cum_qty, target, side="left")
        valid = (cross_pos < len(cum_qty)) & (cross_pos <= end_pos)
        if not np.any(valid):
            continue

        valid_indexer = indexer[valid]
        valid_cross_pos = cross_pos[valid]
        matched_mask[valid_indexer] = True
        matched_time[valid_indexer] = event_time_ms[valid_cross_pos]

    outcomes[matched_col] = matched_mask
    outcomes[matched_time_col] = matched_time
    outcomes[matched_delay_col] = np.where(
        matched_mask,
        matched_time - outcomes["book_time_ms"].to_numpy(dtype=float),
        np.nan,
    )


def compute_price_level_survival(
    book_levels: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    max_initial_level: int,
    tracked_top_n: int | None = None,
    time_col_ms: str = "recv_time_ms",
) -> pd.DataFrame:
    """Track exact price levels forward and classify disappearance vs censoring.

    For each book snapshot and each initial level `1..max_initial_level`, this
    records the exact price and quantity seen at `t0` and follows that same
    price through later replayed book states.

    Main outputs:
    - `disappeared`: the price is no longer visible and, from the tracked depth,
      we can conclude it left the visible book.
    - `survived_to_end`: the price was still visible at the last sample.
    - `fell_below_window`: the price moved deeper than `tracked_top_n`, so we
      lost visibility. This is a censoring reason, not a market event.

    The function also reports how much aggressive trade volume at the same price
    was observed before the terminal event/censoring time, plus a simple
    trade-explained share of the initial displayed quantity.
    """
    if max_initial_level <= 0:
        raise ValueError("max_initial_level must be positive")
    if book_levels.empty:
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
                "terminal_row",
                "terminal_recv_seq",
                "terminal_time_ms",
                "status",
                "censored",
                "censor_reason",
                "disappeared",
                "fell_below_window",
                "survived_to_end",
                "delay_ms",
                "observed_trade_qty_at_price",
                "trade_explained_ratio",
                "residual_nontrade_ratio",
                "trade_matched",
                "trade_match_time_ms",
                "time_to_trade_matched_ms",
                "queue_matched",
                "queue_match_time_ms",
                "time_to_queue_matched_ms",
            ]
        )

    tracked_top_n = _infer_tracked_top_n(book_levels) if tracked_top_n is None else int(tracked_top_n)
    if tracked_top_n <= 0:
        raise ValueError("tracked_top_n must be positive")
    if max_initial_level > tracked_top_n:
        raise ValueError("max_initial_level cannot exceed tracked_top_n")

    initial = _build_initial_price_observations(book_levels, max_initial_level=max_initial_level)
    if initial.empty:
        return initial

    presence = _build_price_presence_runs(book_levels, tracked_top_n=tracked_top_n)
    outcomes = initial.merge(
        presence,
        left_on=["book_row", "book_side", "initial_price"],
        right_on=["book_row", "book_side", "price"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["price"])
    outcomes["run_end_row"] = outcomes["run_end_row"].astype(np.int64)

    n_rows = len(book_levels)
    terminal_row = outcomes["run_end_row"] + 1
    outcomes["survived_to_end"] = terminal_row >= n_rows
    outcomes["terminal_row"] = np.where(outcomes["survived_to_end"], n_rows - 1, terminal_row).astype(np.int64)

    time_values = pd.to_numeric(book_levels[time_col_ms], errors="coerce").to_numpy(dtype=np.int64)
    recv_seq_values = pd.to_numeric(book_levels["recv_seq"], errors="coerce").to_numpy(dtype=np.int64)
    outcomes["terminal_time_ms"] = time_values[outcomes["terminal_row"].to_numpy(dtype=np.int64)]
    outcomes["terminal_recv_seq"] = recv_seq_values[outcomes["terminal_row"].to_numpy(dtype=np.int64)]
    outcomes["delay_ms"] = np.where(
        outcomes["survived_to_end"],
        np.nan,
        outcomes["terminal_time_ms"] - outcomes["book_time_ms"],
    )

    bid_price_cols = [f"bid{level}_price" for level in range(1, tracked_top_n + 1) if f"bid{level}_price" in book_levels.columns]
    ask_price_cols = [f"ask{level}_price" for level in range(1, tracked_top_n + 1) if f"ask{level}_price" in book_levels.columns]
    bid_prices = book_levels[bid_price_cols].apply(pd.to_numeric, errors="coerce") if bid_price_cols else pd.DataFrame(index=book_levels.index)
    ask_prices = book_levels[ask_price_cols].apply(pd.to_numeric, errors="coerce") if ask_price_cols else pd.DataFrame(index=book_levels.index)
    bid_visible_count = bid_prices.notna().sum(axis=1).to_numpy(dtype=np.int64) if not bid_prices.empty else np.zeros(n_rows, dtype=np.int64)
    ask_visible_count = ask_prices.notna().sum(axis=1).to_numpy(dtype=np.int64) if not ask_prices.empty else np.zeros(n_rows, dtype=np.int64)
    bid_worst = bid_prices.min(axis=1, skipna=True).to_numpy(dtype=float) if not bid_prices.empty else np.full(n_rows, np.nan)
    ask_worst = ask_prices.max(axis=1, skipna=True).to_numpy(dtype=float) if not ask_prices.empty else np.full(n_rows, np.nan)

    outcomes["fell_below_window"] = False
    outcomes["disappeared"] = False

    live_mask = ~outcomes["survived_to_end"]
    bid_mask = live_mask & (outcomes["book_side"] == "bid")
    ask_mask = live_mask & (outcomes["book_side"] == "ask")
    term_rows = outcomes["terminal_row"].to_numpy(dtype=np.int64)
    init_prices = outcomes["initial_price"].to_numpy(dtype=float)

    bid_disappeared = (bid_visible_count[term_rows] < tracked_top_n) | (init_prices > bid_worst[term_rows])
    ask_disappeared = (ask_visible_count[term_rows] < tracked_top_n) | (init_prices < ask_worst[term_rows])
    outcomes.loc[bid_mask, "disappeared"] = bid_disappeared[bid_mask.to_numpy()]
    outcomes.loc[ask_mask, "disappeared"] = ask_disappeared[ask_mask.to_numpy()]
    outcomes.loc[live_mask & ~outcomes["disappeared"], "fell_below_window"] = True

    outcomes["status"] = np.select(
        [
            outcomes["disappeared"],
            outcomes["fell_below_window"],
            outcomes["survived_to_end"],
        ],
        [
            "disappeared",
            "fell_below_window",
            "survived_to_end",
        ],
        default="unknown",
    )
    outcomes["censored"] = outcomes["fell_below_window"] | outcomes["survived_to_end"]
    outcomes["censor_reason"] = np.select(
        [
            outcomes["fell_below_window"],
            outcomes["survived_to_end"],
        ],
        [
            "depth_window",
            "sample_end",
        ],
        default="",
    )

    trade_cum = _build_cumulative_trades_by_interval(book_levels, trades)
    outcomes["start_interval"] = outcomes["book_row"].astype(np.int64)
    outcomes["end_interval"] = outcomes["terminal_row"].astype(np.int64) - np.where(outcomes["survived_to_end"], 0, 1)
    outcomes["end_interval"] = outcomes["end_interval"].clip(lower=outcomes["start_interval"])

    if trade_cum.empty:
        outcomes["observed_trade_qty_at_price"] = 0.0
    else:
        trade_groups = {
            key: (
                group["interval_row"].to_numpy(dtype=np.int64),
                group["cum_qty"].to_numpy(dtype=float),
            )
            for key, group in trade_cum.groupby(["aggressor_side", "price"], sort=False)
        }
        observed_trade_qty = np.zeros(len(outcomes), dtype=float)
        for key, idx in outcomes.groupby(["aggressor_side", "initial_price"], sort=False).groups.items():
            indexer = np.asarray(list(idx), dtype=np.int64)
            if key not in trade_groups:
                continue
            interval_rows, cum_qty = trade_groups[key]
            end_intervals = outcomes.iloc[indexer]["end_interval"].to_numpy(dtype=np.int64)
            start_prev = outcomes.iloc[indexer]["start_interval"].to_numpy(dtype=np.int64) - 1

            end_pos = np.searchsorted(interval_rows, end_intervals, side="right") - 1
            start_pos = np.searchsorted(interval_rows, start_prev, side="right") - 1
            end_cum = np.where(end_pos >= 0, cum_qty[np.clip(end_pos, 0, None)], 0.0)
            start_cum = np.where(start_pos >= 0, cum_qty[np.clip(start_pos, 0, None)], 0.0)
            observed_trade_qty[indexer] = end_cum - start_cum
        outcomes["observed_trade_qty_at_price"] = observed_trade_qty

    executed = np.minimum(outcomes["observed_trade_qty_at_price"].to_numpy(dtype=float), outcomes["initial_qty"].to_numpy(dtype=float))
    outcomes["trade_explained_ratio"] = np.clip(executed / outcomes["initial_qty"].to_numpy(dtype=float), 0.0, 1.0)
    outcomes["residual_nontrade_ratio"] = np.where(
        outcomes["disappeared"],
        1.0 - outcomes["trade_explained_ratio"],
        np.nan,
    )

    trade_events = _build_trade_events(book_levels, trades, time_col_ms=time_col_ms)
    queue_events = pd.concat(
        [
            trade_events,
            _build_nontrade_queue_events(book_levels, trades, tracked_top_n=tracked_top_n, time_col_ms=time_col_ms),
        ],
        ignore_index=True,
        sort=False,
    )
    _apply_match_times(outcomes, trade_events, prefix="trade")
    _apply_match_times(outcomes, queue_events, prefix="queue")

    return outcomes.loc[
        :,
        [
            "book_row",
            "book_recv_seq",
            "book_time_ms",
            "book_side",
            "aggressor_side",
            "level",
            "initial_price",
            "initial_qty",
            "terminal_row",
            "terminal_recv_seq",
            "terminal_time_ms",
            "status",
            "censored",
            "censor_reason",
            "disappeared",
            "fell_below_window",
            "survived_to_end",
            "delay_ms",
            "observed_trade_qty_at_price",
            "trade_explained_ratio",
            "residual_nontrade_ratio",
            "trade_matched",
            "trade_match_time_ms",
            "time_to_trade_matched_ms",
            "queue_matched",
            "queue_match_time_ms",
            "time_to_queue_matched_ms",
        ],
    ]


def summarize_price_level_survival(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize price-level survival outcomes by aggressor side and initial level.

    The summary separates observed disappearance from censoring. Depth-window
    loss is reported as a subset of total censoring so it is not confused with
    an actual market outcome.
    """
    if outcomes.empty:
        return pd.DataFrame(
            columns=[
                "aggressor_side",
                "level",
                "n",
                "disappeared_share",
                "censored_share",
                "censored_by_depth_share",
                "survived_to_end_share",
                "median_time_to_disappearance_ms",
                "p90_time_to_disappearance_ms",
                "mean_trade_explained_ratio",
                "mean_residual_nontrade_ratio",
            ]
        )

    def _q90(series: pd.Series) -> float:
        clean = series.dropna()
        return float(np.nanquantile(clean, 0.9)) if clean.size else np.nan

    disappeared = outcomes[outcomes["disappeared"]].copy()
    summary = outcomes.groupby(["aggressor_side", "level"]).agg(
        n=("status", "size"),
        disappeared_share=("disappeared", "mean"),
        censored_share=("censored", "mean"),
        censored_by_depth_share=("fell_below_window", "mean"),
        survived_to_end_share=("survived_to_end", "mean"),
    )

    if disappeared.empty:
        summary["median_time_to_disappearance_ms"] = np.nan
        summary["p90_time_to_disappearance_ms"] = np.nan
        summary["mean_trade_explained_ratio"] = np.nan
        summary["mean_residual_nontrade_ratio"] = np.nan
        return summary.reset_index()

    gone_summary = disappeared.groupby(["aggressor_side", "level"]).agg(
        median_time_to_disappearance_ms=("delay_ms", "median"),
        p90_time_to_disappearance_ms=("delay_ms", _q90),
        mean_trade_explained_ratio=("trade_explained_ratio", "mean"),
        mean_residual_nontrade_ratio=("residual_nontrade_ratio", "mean"),
    )
    return summary.join(gone_summary, how="left").reset_index()


def summarize_price_level_queue_matches(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize trade-only and queue-consumption match times by side and level."""
    if outcomes.empty:
        return pd.DataFrame(
            columns=[
                "aggressor_side",
                "level",
                "n",
                "trade_match_share",
                "queue_match_share",
                "median_time_to_trade_matched_ms",
                "p90_time_to_trade_matched_ms",
                "median_time_to_queue_matched_ms",
                "p90_time_to_queue_matched_ms",
            ]
        )

    def _q90(series: pd.Series) -> float:
        clean = series.dropna()
        return float(np.nanquantile(clean, 0.9)) if clean.size else np.nan

    summary = outcomes.groupby(["aggressor_side", "level"]).agg(
        n=("level", "size"),
        trade_match_share=("trade_matched", "mean"),
        queue_match_share=("queue_matched", "mean"),
        median_time_to_trade_matched_ms=("time_to_trade_matched_ms", "median"),
        p90_time_to_trade_matched_ms=("time_to_trade_matched_ms", _q90),
        median_time_to_queue_matched_ms=("time_to_queue_matched_ms", "median"),
        p90_time_to_queue_matched_ms=("time_to_queue_matched_ms", _q90),
    )
    return summary.reset_index()
