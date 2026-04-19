"""Trade-flow depletion proxies and fixed-price implied cancellation estimates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .book_queue_common import BOOK_SIDE_TO_AGGRESSOR
from stats.utils.cache import cache_path, load_or_build_parquet

try:  # pragma: no cover - optional notebook UX dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _progress_iter(iterable, *, total: int | None = None, desc: str = "Progress", show_progress: bool = False):
    if not show_progress:
        return iterable
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    if total is None or total <= 0:
        return iterable

    def _wrapped():
        count = 0
        last_pct = -1
        for item in iterable:
            yield item
            count += 1
            pct = int((count * 100) / total)
            if pct != last_pct and (count == total or count % max(1, total // 100) == 0):
                last_pct = pct
                print(f"\r{desc}: {count}/{total} ({pct}%)", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

    return _wrapped()


def _stream_cache_params(stream: pd.DataFrame, *, max_level: int) -> dict[str, object]:
    recv_seq = pd.to_numeric(stream.get("recv_seq", pd.Series(dtype=float)), errors="coerce")
    recv_time_ms = pd.to_numeric(stream.get("recv_time_ms", pd.Series(dtype=float)), errors="coerce")
    event_type = stream.get("event_type", pd.Series(dtype=object))
    return {
        "algorithm": "trade_depletion_v2",
        "max_level": int(max_level),
        "stream_rows": int(len(stream)),
        "book_rows": int((event_type == "book").sum()),
        "trade_rows": int((event_type == "trade").sum()),
        "first_recv_seq": None if recv_seq.dropna().empty else int(recv_seq.min()),
        "last_recv_seq": None if recv_seq.dropna().empty else int(recv_seq.max()),
        "first_recv_time_ms": None if recv_time_ms.dropna().empty else int(recv_time_ms.min()),
        "last_recv_time_ms": None if recv_time_ms.dropna().empty else int(recv_time_ms.max()),
    }


def compute_trade_depletion(
    stream: pd.DataFrame,
    *,
    max_level: int,
    time_col_ms: str = "recv_time_ms",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Estimate when cumulative aggressive flow first matches visible depth.

    For each book row and each initial level `k`, this computes the first later
    event where cumulative opposite-side aggressive trade volume is at least the
    visible depth from level 1 through `k` at the starting snapshot.

    This is a trade-pressure proxy. It does not prove that the original queue
    was removed only by trades, and it does not track queue identity.
    """
    if max_level <= 0:
        raise ValueError("max_level must be positive")
    if "event_type" not in stream.columns:
        raise KeyError("stream must include event_type")

    book_positions = np.flatnonzero((stream["event_type"] == "book").to_numpy())
    if len(book_positions) == 0:
        return pd.DataFrame(
            columns=[
                "book_stream_pos",
                "book_recv_seq",
                "book_time_ms",
                "aggressor_side",
                "level",
                "initial_depth_qty",
                "depletion_stream_pos",
                "depletion_recv_seq",
                "depletion_time_ms",
                "delay_ms",
                "trade_events_to_depletion",
                "censored",
            ]
        )

    cum_buy = stream["cum_buy_qty"].to_numpy(dtype=float)
    cum_sell = stream["cum_sell_qty"].to_numpy(dtype=float)
    time_ms = stream[time_col_ms].to_numpy(dtype="int64")
    recv_seq = stream["recv_seq"].to_numpy(dtype="int64")
    trade_count = stream["cum_trade_event_count"].to_numpy(dtype="int64")

    rows: list[dict[str, object]] = []
    for pos in _progress_iter(book_positions, total=len(book_positions), desc="trade depletion", show_progress=show_progress):
        book_row = stream.iloc[pos]
        for book_side, aggressor_side in BOOK_SIDE_TO_AGGRESSOR.items():
            cum_series = cum_buy if aggressor_side == "buy" else cum_sell
            base_cum = float(cum_series[pos])
            for level in range(1, max_level + 1):
                price_cols = [f"{book_side}{idx}_price" for idx in range(1, level + 1)]
                qty_cols = [f"{book_side}{idx}_qty" for idx in range(1, level + 1)]
                if not set(price_cols + qty_cols) <= set(book_row.index):
                    continue
                prices = pd.to_numeric(book_row[price_cols], errors="coerce")
                qtys = pd.to_numeric(book_row[qty_cols], errors="coerce").fillna(0.0).clip(lower=0.0)
                if prices.isna().any():
                    continue
                depth_qty = float(qtys.sum())
                if depth_qty <= 0:
                    continue

                target = base_cum + depth_qty
                depletion_pos = int(np.searchsorted(cum_series, target, side="left"))
                censored = depletion_pos >= len(stream)
                if censored:
                    delay_ms = np.nan
                    depletion_recv_seq = np.nan
                    depletion_time_ms = np.nan
                    trade_events_to_depletion = np.nan
                else:
                    delay_ms = float(time_ms[depletion_pos] - time_ms[pos])
                    depletion_recv_seq = int(recv_seq[depletion_pos])
                    depletion_time_ms = int(time_ms[depletion_pos])
                    trade_events_to_depletion = int(trade_count[depletion_pos] - trade_count[pos])

                rows.append(
                    {
                        "book_stream_pos": int(pos),
                        "book_recv_seq": int(recv_seq[pos]),
                        "book_time_ms": int(time_ms[pos]),
                        "aggressor_side": aggressor_side,
                        "book_side": book_side,
                        "level": level,
                        "initial_depth_qty": depth_qty,
                        "cumulative_depth_qty": depth_qty,
                        "depletion_stream_pos": None if censored else depletion_pos,
                        "depletion_recv_seq": depletion_recv_seq,
                        "depletion_time_ms": depletion_time_ms,
                        "delay_ms": delay_ms,
                        "trade_events_to_depletion": trade_events_to_depletion,
                        "censored": censored,
                    }
                )

    return pd.DataFrame(rows)


def estimate_implied_cancellations(
    stream: pd.DataFrame,
    *,
    max_level: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Estimate non-trade removal between consecutive book rows at fixed prices.

    For each pair of consecutive book rows, this compares the visible quantity
    at each tracked level and subtracts aggressive trade volume observed at the
    same price between the two rows. The residual is reported as implied
    cancellation/replacement quantity when the price itself stays unchanged.

    This is still an inference from net book change, not a literal cancel feed.
    """
    if max_level <= 0:
        raise ValueError("max_level must be positive")
    if "event_type" not in stream.columns:
        raise KeyError("stream must include event_type")

    book_positions = np.flatnonzero((stream["event_type"] == "book").to_numpy())
    if len(book_positions) < 2:
        return pd.DataFrame(
            columns=[
                "book_stream_pos",
                "next_book_stream_pos",
                "book_recv_seq",
                "next_book_recv_seq",
                "book_time_ms",
                "next_book_time_ms",
                "interval_ms",
                "book_side",
                "level",
                "prev_price",
                "next_price",
                "prev_qty",
                "next_qty",
                "trade_qty_at_price",
                "visible_reduction",
                "implied_cancel_qty",
                "price_stable",
            ]
        )

    rows: list[dict[str, object]] = []
    paired_positions = zip(book_positions[:-1], book_positions[1:])
    for left_pos, right_pos in _progress_iter(
        paired_positions,
        total=max(0, len(book_positions) - 1),
        desc="implied cancellations",
        show_progress=show_progress,
    ):
        interval = stream.iloc[left_pos + 1 : right_pos]
        trade_interval = interval[interval["event_type"] == "trade"]
        if trade_interval.empty:
            trade_groups = pd.Series(dtype=float)
        else:
            trade_groups = trade_interval.groupby(["trade_side", "price"], dropna=True)["qty"].sum()

        left_row = stream.iloc[left_pos]
        right_row = stream.iloc[right_pos]
        for book_side, aggressor_side in BOOK_SIDE_TO_AGGRESSOR.items():
            for level in range(1, max_level + 1):
                price_col = f"{book_side}{level}_price"
                qty_col = f"{book_side}{level}_qty"
                prev_price = left_row.get(price_col, np.nan)
                next_price = right_row.get(price_col, np.nan)
                prev_qty = float(pd.to_numeric(pd.Series([left_row.get(qty_col, np.nan)]), errors="coerce").fillna(0.0).iloc[0])
                next_qty = float(pd.to_numeric(pd.Series([right_row.get(qty_col, np.nan)]), errors="coerce").fillna(0.0).iloc[0])

                if not np.isfinite(prev_price) or prev_qty <= 0:
                    continue

                trade_qty_at_price = float(trade_groups.get((aggressor_side, float(prev_price)), 0.0))
                visible_reduction = max(prev_qty - next_qty, 0.0)
                price_stable = bool(np.isfinite(next_price) and float(prev_price) == float(next_price))
                implied_cancel_qty = np.nan
                if price_stable:
                    implied_cancel_qty = max(visible_reduction - trade_qty_at_price, 0.0)
                next_book_time_ms = int(right_row["recv_time_ms"])
                interval_ms = int(next_book_time_ms - int(left_row["recv_time_ms"]))

                rows.append(
                    {
                        "book_stream_pos": int(left_pos),
                        "next_book_stream_pos": int(right_pos),
                        "book_recv_seq": int(left_row["recv_seq"]),
                        "next_book_recv_seq": int(right_row["recv_seq"]),
                        "book_time_ms": int(left_row["recv_time_ms"]),
                        "next_book_time_ms": next_book_time_ms,
                        "interval_ms": interval_ms,
                        "book_side": book_side,
                        "aggressor_side": aggressor_side,
                        "level": level,
                        "prev_price": float(prev_price),
                        "next_price": float(next_price) if np.isfinite(next_price) else np.nan,
                        "prev_qty": prev_qty,
                        "next_qty": next_qty,
                        "trade_qty_at_price": trade_qty_at_price,
                        "visible_reduction": visible_reduction,
                        "implied_cancel_qty": implied_cancel_qty,
                        "price_stable": price_stable,
                        "price_changed": not price_stable,
                    }
                )

    return pd.DataFrame(rows)


def summarize_depletion_by_level(depletion: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-depletion rows into side/level summary statistics."""
    if depletion.empty:
        return pd.DataFrame(
            columns=[
                "aggressor_side",
                "level",
                "n",
                "censored_share",
                "median_delay_ms",
                "p90_delay_ms",
                "mean_delay_ms",
            ]
        )
    valid = depletion[~depletion["censored"]].copy()
    summary = depletion.groupby(["aggressor_side", "level"]).agg(
        n=("delay_ms", "size"),
        censored_share=("censored", "mean"),
        median_delay_ms=("delay_ms", "median"),
        p90_delay_ms=("delay_ms", lambda s: float(np.nanquantile(s.dropna(), 0.9)) if s.dropna().size else np.nan),
        mean_delay_ms=("delay_ms", "mean"),
    )
    if valid.empty:
        summary["median_delay_ms"] = np.nan
        summary["p90_delay_ms"] = np.nan
        summary["mean_delay_ms"] = np.nan
    return summary.reset_index()


def get_or_build_implied_cancellations(
    stream: pd.DataFrame,
    *,
    day_dir: Path,
    max_level: int,
    show_progress: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Cache implied cancellation estimates for one day and depth setting."""
    params = _stream_cache_params(stream, max_level=max_level)
    path = cache_path(day_dir, "implied_cancellations", params, ext="parquet")
    return load_or_build_parquet(
        path,
        build=lambda: estimate_implied_cancellations(stream, max_level=max_level, show_progress=show_progress),
        force=force,
        index=False,
    )


def get_or_build_trade_depletion(
    stream: pd.DataFrame,
    *,
    day_dir: Path,
    max_level: int,
    show_progress: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Cache trade-depletion estimates for one day and depth setting."""
    params = _stream_cache_params(stream, max_level=max_level)
    path = cache_path(day_dir, "trade_depletion", params, ext="parquet")
    return load_or_build_parquet(
        path,
        build=lambda: compute_trade_depletion(stream, max_level=max_level, show_progress=show_progress),
        force=force,
        index=False,
    )
