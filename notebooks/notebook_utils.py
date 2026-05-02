from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_backtester_root(start: Path | None = None) -> Path:
    """Return the repository root that contains both `stats/` and `notebooks/`."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "stats").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the exchange-data-backtester project root")


def bootstrap_backtester_path(start: Path | None = None) -> Path:
    """Find the repository root and add it to `sys.path` if needed."""
    project_root = find_backtester_root(start=start)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def resolve_day_dir(project_root: Path, *, symbol: str, day: str, exchange: str = "binance") -> Path:
    """Locate a recorder day directory for a symbol and exchange."""
    candidates = [
        project_root.parent / "exchange-data-recorder" / "data" / exchange / symbol / day,
        project_root.parent / "exchange-data-recorder" / "data" / symbol / day,
        project_root / "data" / exchange / symbol / day,
        project_root / "data" / symbol / day,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "schema.json").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate a recorder day folder for {exchange} {symbol} {day}")


def rolling_window_sum(x: np.ndarray, T: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be positive")
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    if len(x) < T:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[T - 1 :] = csum[T:] - csum[:-T]
    return out


def build_weighted_signed_flow(sign: np.ndarray, qty: np.ndarray, a: float) -> np.ndarray:
    return np.asarray(sign, dtype=float) * np.power(np.asarray(qty, dtype=float), a)


def build_imbalance_series(sign: np.ndarray, qty: np.ndarray, T: int, a: float) -> np.ndarray:
    weighted = build_weighted_signed_flow(sign, qty, a)
    return rolling_window_sum(weighted, T)


def _datetime_ns(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values, utc=True).astype("int64").to_numpy()


def build_trade_time_delta(trade_frame: pd.DataFrame, return_H: int) -> np.ndarray:
    log_mid = trade_frame["log_mid"].to_numpy(dtype=float)
    delta = np.full(len(log_mid), np.nan, dtype=float)
    if len(log_mid) > return_H:
        delta[:-return_H] = log_mid[return_H:] - log_mid[:-return_H]
    return delta


def build_clock_time_delta(
    trade_frame: pd.DataFrame,
    book_mid_frame: pd.DataFrame,
    return_horizon: str | pd.Timedelta,
) -> np.ndarray:
    horizon = pd.to_timedelta(return_horizon)
    trade_ts_ns = _datetime_ns(trade_frame["ts"])
    book_ts_ns = _datetime_ns(book_mid_frame["ts"])
    book_log_mid = book_mid_frame["log_mid"].to_numpy(dtype=float)

    target_ts_ns = trade_ts_ns + int(horizon.value)
    future_idx = np.searchsorted(book_ts_ns, target_ts_ns, side="left")
    valid = future_idx < len(book_log_mid)

    delta = np.full(len(trade_frame), np.nan, dtype=float)
    current_log_mid = trade_frame["log_mid"].to_numpy(dtype=float)
    delta[valid] = book_log_mid[future_idx[valid]] - current_log_mid[valid]
    return delta


def build_signal_return_frame(trade_frame: pd.DataFrame, signal_T: int, a: float, delta: np.ndarray) -> pd.DataFrame:
    sign = trade_frame["aggr_sign"].to_numpy(dtype=float)
    qty = trade_frame["qty"].to_numpy(dtype=float)
    imbalance = build_imbalance_series(sign, qty, T=signal_T, a=a)
    return pd.DataFrame({"I": imbalance, "delta": delta}).dropna().copy()


def summarize_signal_delta(imbalance: np.ndarray, delta: np.ndarray) -> pd.Series:
    x = np.asarray(imbalance, dtype=float)
    y = np.asarray(delta, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return pd.Series({"cov": np.nan, "pearson": np.nan, "spearman": np.nan, "corr": np.nan, "n": 0})

    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan

    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1]) if len(x) > 1 and np.std(x_rank) > 0 and np.std(y_rank) > 0 else np.nan

    return pd.Series({"cov": cov, "pearson": pearson, "spearman": spearman, "corr": pearson, "n": len(x)})


def build_imbalance_for_signal(trade_frame: pd.DataFrame, signal_T: int, a: float) -> np.ndarray:
    sign = trade_frame["aggr_sign"].to_numpy(dtype=float)
    qty = trade_frame["qty"].to_numpy(dtype=float)
    return build_imbalance_series(sign, qty, T=signal_T, a=a)


def return_horizon_label(value: str | pd.Timedelta) -> str:
    horizon = pd.to_timedelta(value)
    if horizon < pd.Timedelta(minutes=1):
        return f"{int(horizon.total_seconds())}s"
    if horizon < pd.Timedelta(hours=1):
        return f"{int(horizon.total_seconds() // 60)}min"
    return f"{int(horizon.total_seconds() // 3600)}h"


def build_trade_time_cov_corr_grid(
    trade_frame: pd.DataFrame,
    signal_T_list: np.ndarray,
    return_H_trade_list: np.ndarray,
    a_list: np.ndarray,
) -> pd.DataFrame:
    rows = []
    delta_cache = {int(return_H): build_trade_time_delta(trade_frame, int(return_H)) for return_H in return_H_trade_list}
    signal_cache = {
        (float(a), int(signal_T)): build_imbalance_for_signal(trade_frame, signal_T=int(signal_T), a=float(a))
        for a in a_list
        for signal_T in signal_T_list
    }

    for a in a_list:
        for signal_T in signal_T_list:
            imbalance = signal_cache[(float(a), int(signal_T))]
            for return_H in return_H_trade_list:
                summary_row = summarize_signal_delta(imbalance, delta_cache[int(return_H)])
                rows.append({
                    "a": float(a),
                    "signal_T": int(signal_T),
                    "return_H_trades": int(return_H),
                    "cov": float(summary_row["cov"]),
                    "pearson": float(summary_row["pearson"]),
                    "spearman": float(summary_row["spearman"]),
                    "corr": float(summary_row["pearson"]),
                    "n": int(summary_row["n"]),
                })
    return pd.DataFrame(rows)


def build_clock_time_cov_corr_grid(
    trade_frame: pd.DataFrame,
    book_mid_frame: pd.DataFrame,
    signal_T_list: np.ndarray,
    return_H_clock_list: list[str],
    a_list: np.ndarray,
) -> pd.DataFrame:
    rows = []
    delta_cache = {
        return_horizon_label(horizon): build_clock_time_delta(trade_frame, book_mid_frame, horizon)
        for horizon in return_H_clock_list
    }
    signal_cache = {
        (float(a), int(signal_T)): build_imbalance_for_signal(trade_frame, signal_T=int(signal_T), a=float(a))
        for a in a_list
        for signal_T in signal_T_list
    }

    for a in a_list:
        for signal_T in signal_T_list:
            imbalance = signal_cache[(float(a), int(signal_T))]
            for horizon_label, delta in delta_cache.items():
                summary_row = summarize_signal_delta(imbalance, delta)
                horizon_td = pd.to_timedelta(horizon_label)
                rows.append({
                    "a": float(a),
                    "signal_T": int(signal_T),
                    "return_H_clock": horizon_label,
                    "return_H_seconds": float(horizon_td.total_seconds()),
                    "cov": float(summary_row["cov"]),
                    "pearson": float(summary_row["pearson"]),
                    "spearman": float(summary_row["spearman"]),
                    "corr": float(summary_row["pearson"]),
                    "n": int(summary_row["n"]),
                })
    return pd.DataFrame(rows)


def plot_trade_time_grid(grid_df: pd.DataFrame, *, a: float = 0.0) -> None:
    plot_df = grid_df[grid_df["a"] == a].copy()
    pearson_pivot = plot_df.pivot(index="signal_T", columns="return_H_trades", values="pearson").sort_index()
    spearman_pivot = plot_df.pivot(index="signal_T", columns="return_H_trades", values="spearman").sort_index()
    cov_pivot = plot_df.pivot(index="signal_T", columns="return_H_trades", values="cov").sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

    im0 = axes[0].imshow(cov_pivot.values, aspect="auto", origin="lower", cmap="coolwarm")
    axes[0].set_title(f"Trade-time covariance, a={a}")
    axes[0].set_xticks(range(len(cov_pivot.columns)))
    axes[0].set_xticklabels(cov_pivot.columns)
    axes[0].set_yticks(range(len(cov_pivot.index)))
    axes[0].set_yticklabels(cov_pivot.index)
    axes[0].set_xlabel("Future return horizon, trades")
    axes[0].set_ylabel("Signal lookback, trades")
    fig.colorbar(im0, ax=axes[0], label="covariance")

    im1 = axes[1].imshow(pearson_pivot.values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title(f"Trade-time Pearson, a={a}")
    axes[1].set_xticks(range(len(pearson_pivot.columns)))
    axes[1].set_xticklabels(pearson_pivot.columns)
    axes[1].set_yticks(range(len(pearson_pivot.index)))
    axes[1].set_yticklabels(pearson_pivot.index)
    axes[1].set_xlabel("Future return horizon, trades")
    fig.colorbar(im1, ax=axes[1], label="Pearson")

    im2 = axes[2].imshow(spearman_pivot.values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title(f"Trade-time Spearman, a={a}")
    axes[2].set_xticks(range(len(spearman_pivot.columns)))
    axes[2].set_xticklabels(spearman_pivot.columns)
    axes[2].set_yticks(range(len(spearman_pivot.index)))
    axes[2].set_yticklabels(spearman_pivot.index)
    axes[2].set_xlabel("Future return horizon, trades")
    fig.colorbar(im2, ax=axes[2], label="Spearman")

    plt.tight_layout()
    plt.show()


def plot_clock_time_grid(grid_df: pd.DataFrame, *, a: float = 0.0) -> None:
    plot_df = grid_df[grid_df["a"] == a].copy().sort_values("return_H_seconds")
    horizon_order = plot_df[["return_H_clock", "return_H_seconds"]].drop_duplicates().sort_values("return_H_seconds")["return_H_clock"].tolist()
    pearson_pivot = plot_df.pivot(index="signal_T", columns="return_H_clock", values="pearson").reindex(columns=horizon_order).sort_index()
    spearman_pivot = plot_df.pivot(index="signal_T", columns="return_H_clock", values="spearman").reindex(columns=horizon_order).sort_index()
    cov_pivot = plot_df.pivot(index="signal_T", columns="return_H_clock", values="cov").reindex(columns=horizon_order).sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=True)

    im0 = axes[0].imshow(cov_pivot.values, aspect="auto", origin="lower", cmap="coolwarm")
    axes[0].set_title(f"Clock-time covariance, a={a}")
    axes[0].set_xticks(range(len(cov_pivot.columns)))
    axes[0].set_xticklabels(cov_pivot.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(cov_pivot.index)))
    axes[0].set_yticklabels(cov_pivot.index)
    axes[0].set_xlabel("Future return horizon, clock time")
    axes[0].set_ylabel("Signal lookback, trades")
    fig.colorbar(im0, ax=axes[0], label="covariance")

    im1 = axes[1].imshow(pearson_pivot.values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title(f"Clock-time Pearson, a={a}")
    axes[1].set_xticks(range(len(pearson_pivot.columns)))
    axes[1].set_xticklabels(pearson_pivot.columns, rotation=45, ha="right")
    axes[1].set_yticks(range(len(pearson_pivot.index)))
    axes[1].set_yticklabels(pearson_pivot.index)
    axes[1].set_xlabel("Future return horizon, clock time")
    fig.colorbar(im1, ax=axes[1], label="Pearson")

    im2 = axes[2].imshow(spearman_pivot.values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title(f"Clock-time Spearman, a={a}")
    axes[2].set_xticks(range(len(spearman_pivot.columns)))
    axes[2].set_xticklabels(spearman_pivot.columns, rotation=45, ha="right")
    axes[2].set_yticks(range(len(spearman_pivot.index)))
    axes[2].set_yticklabels(spearman_pivot.index)
    axes[2].set_xlabel("Future return horizon, clock time")
    fig.colorbar(im2, ax=axes[2], label="Spearman")

    plt.tight_layout()
    plt.show()


def make_clock_grid(top_df: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    book_mid = top_df[["ts", "mid"]].dropna().sort_values("ts").rename(columns={"mid": "mid_at_book"})
    start = book_mid["ts"].min().floor(freq)
    end = book_mid["ts"].max().ceil(freq)
    grid = pd.DataFrame({"ts": pd.date_range(start, end, freq=freq, tz="UTC")})
    grid = pd.merge_asof(grid, book_mid, on="ts", direction="backward")
    grid = grid.dropna(subset=["mid_at_book"]).reset_index(drop=True)
    grid["log_mid"] = np.log(grid["mid_at_book"])
    return grid


def rolling_time_window_sum(
    source_ts_ns: np.ndarray,
    values: np.ndarray,
    target_ts_ns: np.ndarray,
    window_ns: int,
) -> np.ndarray:
    source_ts_ns = np.asarray(source_ts_ns, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    target_ts_ns = np.asarray(target_ts_ns, dtype=np.int64)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    left = np.searchsorted(source_ts_ns, target_ts_ns - int(window_ns), side="left")
    right = np.searchsorted(source_ts_ns, target_ts_ns, side="right")
    out = csum[right] - csum[left]
    return out.astype(float)


def rolling_time_window_mean(
    source_ts_ns: np.ndarray,
    values: np.ndarray,
    target_ts_ns: np.ndarray,
    window_ns: int,
) -> np.ndarray:
    source_ts_ns = np.asarray(source_ts_ns, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    target_ts_ns = np.asarray(target_ts_ns, dtype=np.int64)
    finite = np.isfinite(values)
    clean_values = np.where(finite, values, 0.0)
    csum = np.cumsum(np.insert(clean_values, 0, 0.0))
    ccnt = np.cumsum(np.insert(finite.astype(float), 0, 0.0))
    left = np.searchsorted(source_ts_ns, target_ts_ns - int(window_ns), side="left")
    right = np.searchsorted(source_ts_ns, target_ts_ns, side="right")
    total = csum[right] - csum[left]
    count = ccnt[right] - ccnt[left]
    out = np.full(len(target_ts_ns), np.nan, dtype=float)
    mask = count > 0
    out[mask] = total[mask] / count[mask]
    return out


def build_top_depth_depletion_frame(book_state: pd.DataFrame, window: int | float | str | pd.Timedelta = "5s") -> pd.DataFrame:
    """Estimate recent top-of-book depletion using only past book updates.

    The proxy measures how much the best bid / best ask has shrunk over the
    requested trailing window relative to the quantity that was visible over
    the same period.

    A positive `ask_depletion_rate` means the ask queue has been shrinking.
    A positive `bid_depletion_rate` means the bid queue has been shrinking.
    """
    if not {"ts", "bid1_qty", "ask1_qty"}.issubset(book_state.columns):
        missing = sorted({"ts", "bid1_qty", "ask1_qty"} - set(book_state.columns))
        raise KeyError(f"book_state is missing required columns: {missing}")

    window_td = _coerce_horizon_timedelta(window, default_unit="s")
    window_ns = int(window_td.value)
    if window_ns <= 0:
        raise ValueError("window must be positive")

    out = book_state[["ts"]].copy()
    ts_ns = pd.to_datetime(book_state["ts"], utc=True).astype("int64").to_numpy(dtype=np.int64)
    bid = pd.to_numeric(book_state["bid1_qty"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(book_state["ask1_qty"], errors="coerce").to_numpy(dtype=float)

    if len(book_state) == 0:
        out["bid_shrink_sum"] = []
        out["ask_shrink_sum"] = []
        out["bid_base_sum"] = []
        out["ask_base_sum"] = []
        out["bid_depletion_rate"] = []
        out["ask_depletion_rate"] = []
        out["depletion_bias"] = []
        return out

    bid_prev = np.r_[bid[0], bid[:-1]]
    ask_prev = np.r_[ask[0], ask[:-1]]

    bid_shrink = np.maximum(bid_prev - bid, 0.0)
    ask_shrink = np.maximum(ask_prev - ask, 0.0)
    bid_base = np.maximum(bid_prev, 0.0)
    ask_base = np.maximum(ask_prev, 0.0)

    bid_shrink_sum = rolling_time_window_sum(ts_ns, bid_shrink, ts_ns, window_ns)
    ask_shrink_sum = rolling_time_window_sum(ts_ns, ask_shrink, ts_ns, window_ns)
    bid_base_sum = rolling_time_window_sum(ts_ns, bid_base, ts_ns, window_ns)
    ask_base_sum = rolling_time_window_sum(ts_ns, ask_base, ts_ns, window_ns)

    out["bid_shrink_sum"] = bid_shrink_sum
    out["ask_shrink_sum"] = ask_shrink_sum
    out["bid_base_sum"] = bid_base_sum
    out["ask_base_sum"] = ask_base_sum
    out["bid_depletion_rate"] = np.divide(
        bid_shrink_sum,
        bid_base_sum,
        out=np.full_like(bid_shrink_sum, np.nan, dtype=float),
        where=bid_base_sum != 0,
    )
    out["ask_depletion_rate"] = np.divide(
        ask_shrink_sum,
        ask_base_sum,
        out=np.full_like(ask_shrink_sum, np.nan, dtype=float),
        where=ask_base_sum != 0,
    )
    out["depletion_bias"] = out["ask_depletion_rate"] - out["bid_depletion_rate"]
    return out


def build_top_price_survival_frame(book_state: pd.DataFrame) -> pd.DataFrame:
    """Measure how long the current top quote price has persisted on each side.

    This is a causal proxy for top-level survival. It uses only the current and
    past best bid / best ask prices, so it can be aligned to the same clock grid
    as the other notebook features without leaking future survival time.
    """
    required = {"ts", "bid1_price", "ask1_price"}
    if not required.issubset(book_state.columns):
        missing = sorted(required - set(book_state.columns))
        raise KeyError(f"book_state is missing required columns: {missing}")

    out = book_state[["ts"]].copy()
    if book_state.empty:
        out["bid_price_age_ms"] = []
        out["ask_price_age_ms"] = []
        out["queue_survival_half_life_ms"] = []
        out["queue_survival_bias_ms"] = []
        return out

    book = book_state[["ts", "bid1_price", "ask1_price"]].dropna().sort_values("ts").reset_index(drop=True)
    out = book[["ts"]].copy()
    if book.empty:
        out["bid_price_age_ms"] = []
        out["ask_price_age_ms"] = []
        out["queue_survival_half_life_ms"] = []
        out["queue_survival_bias_ms"] = []
        return out

    ts_ns = pd.to_datetime(book["ts"], utc=True).astype("int64").to_numpy(dtype=np.int64)

    def _price_age_ms(price: pd.Series) -> np.ndarray:
        changed = price.ne(price.shift()).fillna(True).to_numpy(dtype=bool)
        last_change_ns = np.where(changed, ts_ns, 0)
        last_change_ns = np.maximum.accumulate(last_change_ns)
        return (ts_ns - last_change_ns) / 1e6

    bid_price_age_ms = _price_age_ms(pd.to_numeric(book["bid1_price"], errors="coerce"))
    ask_price_age_ms = _price_age_ms(pd.to_numeric(book["ask1_price"], errors="coerce"))

    out["bid_price_age_ms"] = bid_price_age_ms
    out["ask_price_age_ms"] = ask_price_age_ms
    out["queue_survival_half_life_ms"] = 0.5 * (bid_price_age_ms + ask_price_age_ms)
    out["queue_survival_bias_ms"] = ask_price_age_ms - bid_price_age_ms
    return out


def build_trailing_mean_frame(
    frame: pd.DataFrame,
    value_col: str,
    window: int | float | str | pd.Timedelta = "5s",
    *,
    min_periods: int = 1,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Compute a past-only trailing mean for a book-state column."""
    required = {"ts", value_col}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise KeyError(f"frame is missing required columns: {missing}")

    window_td = _coerce_horizon_timedelta(window, default_unit="s")
    window_label = _format_horizon_label(window_td)
    output_col = output_col or f"{value_col}_{window_label}_mean"

    book = frame[["ts", value_col]].dropna().sort_values("ts").reset_index(drop=True).copy()
    out = book[["ts"]].copy()
    if book.empty:
        out[output_col] = []
        out[value_col] = []
        return out

    series = pd.Series(pd.to_numeric(book[value_col], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(book["ts"], utc=True))
    rolled = series.rolling(window_td, min_periods=min_periods).mean()
    out[output_col] = rolled.to_numpy(dtype=float)
    out[value_col] = series.to_numpy(dtype=float)
    return out


def build_trailing_realized_vol_frame(
    book_state: pd.DataFrame,
    window: int | float | str | pd.Timedelta = "5s",
    *,
    price_col: str = "log_mid",
    min_periods: int = 5,
) -> pd.DataFrame:
    """Compute a past-only trailing realized-vol proxy from the book state.

    The feature uses the trailing window of completed log-mid changes and
    expresses the result in basis points of log-return volatility.
    """
    required = {"ts", price_col}
    if not required.issubset(book_state.columns):
        missing = sorted(required - set(book_state.columns))
        raise KeyError(f"book_state is missing required columns: {missing}")

    window_td = _coerce_horizon_timedelta(window, default_unit="s")
    window_label = _format_horizon_label(window_td)

    book = book_state[["ts", price_col]].dropna().sort_values("ts").reset_index(drop=True).copy()
    out = book[["ts"]].copy()
    if book.empty:
        out[f"rv_{window_label}_bps"] = []
        out["log_mid_ret"] = []
        return out

    log_mid = pd.to_numeric(book[price_col], errors="coerce").to_numpy(dtype=float)
    log_ret = np.diff(log_mid, prepend=np.nan)
    ret_series = pd.Series(log_ret, index=pd.to_datetime(book["ts"], utc=True))
    rv = np.sqrt(ret_series.pow(2).rolling(window_td, closed="left", min_periods=min_periods).sum())

    out[f"rv_{window_label}_bps"] = rv.to_numpy(dtype=float) * 1e4
    out["log_mid_ret"] = log_ret
    return out


def build_event_time_trade_explained_depletion_frame(
    book_levels: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    window_events: int = 20,
    max_level: int = 1,
) -> pd.DataFrame:
    """Estimate event-time book shrinkage explained by trades.

    This is the event-time analogue of the clock-time depletion proxy.
    Instead of looking back over a fixed number of seconds, it looks back over
    the last `window_events` replayed book updates. The proxy uses the existing
    price-level survival machinery to estimate how much of the visible reduction
    at the top of book can be explained by aggressive trades at the same price.

    A positive `ask_trade_explained_rate` means the ask side has been shrinking
    quickly in a trade-explained sense. A positive `bid_trade_explained_rate`
    means the bid side has been shrinking quickly. The difference
    `trade_explained_bias = ask_trade_explained_rate - bid_trade_explained_rate`
    is the signed feature analogue of the clock-time `depletion_bias`.
    """
    if window_events <= 0:
        raise ValueError("window_events must be positive")
    if max_level <= 0:
        raise ValueError("max_level must be positive")
    if not {"ts"}.issubset(book_levels.columns):
        raise KeyError("book_levels must include 'ts'")
    if trades is None:
        raise ValueError("trades must not be None")

    from stats.analysis.book_queue_common import build_layer_event_stream
    from stats.analysis.trade_depletion import estimate_implied_cancellations

    stream = build_layer_event_stream(book_levels, trades)
    implied = estimate_implied_cancellations(stream, max_level=max_level, show_progress=False)
    if implied.empty:
        return pd.DataFrame(
            columns=[
                "book_stream_pos",
                "ts",
                "bid_trade_explained_rate",
                "ask_trade_explained_rate",
                "bid_trade_explained_share",
                "ask_trade_explained_share",
                "bid_nontrade_rate",
                "ask_nontrade_rate",
                "trade_explained_bias",
            ]
        )

    implied = implied.loc[np.isfinite(implied["implied_cancel_qty"]) & implied["price_stable"]].copy()
    if implied.empty:
        return pd.DataFrame(
            columns=[
                "book_stream_pos",
                "ts",
                "bid_trade_explained_rate",
                "ask_trade_explained_rate",
                "bid_trade_explained_share",
                "ask_trade_explained_share",
                "bid_nontrade_rate",
                "ask_nontrade_rate",
                "trade_explained_bias",
            ]
        )

    implied["trade_explained_qty"] = np.maximum(
        pd.to_numeric(implied["visible_reduction"], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(implied["implied_cancel_qty"], errors="coerce").to_numpy(dtype=float),
        0.0,
    )
    implied["nontrade_qty"] = pd.to_numeric(implied["implied_cancel_qty"], errors="coerce").to_numpy(dtype=float)
    implied["prev_qty"] = pd.to_numeric(implied["prev_qty"], errors="coerce").to_numpy(dtype=float)

    grouped = (
        implied.groupby(["book_stream_pos", "book_side"], sort=False)
        .agg(
            ts=("book_time_ms", "first"),
            trade_explained_qty=("trade_explained_qty", "sum"),
            visible_reduction=("visible_reduction", "sum"),
            nontrade_qty=("nontrade_qty", "sum"),
            prev_qty=("prev_qty", "sum"),
        )
        .reset_index()
        .sort_values(["book_side", "book_stream_pos"])
        .reset_index(drop=True)
    )
    grouped["ts"] = pd.to_datetime(grouped["ts"], unit="ms", utc=True)

    side_frames: list[pd.DataFrame] = []
    for side in ("bid", "ask"):
        side_df = grouped.loc[grouped["book_side"] == side].copy().sort_values("book_stream_pos").reset_index(drop=True)
        if side_df.empty:
            continue
        explained_sum = rolling_window_sum(side_df["trade_explained_qty"].to_numpy(dtype=float), window_events)
        visible_sum = rolling_window_sum(side_df["visible_reduction"].to_numpy(dtype=float), window_events)
        nontrade_sum = rolling_window_sum(side_df["nontrade_qty"].to_numpy(dtype=float), window_events)
        prev_sum = rolling_window_sum(side_df["prev_qty"].to_numpy(dtype=float), window_events)

        side_out = side_df.loc[:, ["book_stream_pos", "ts"]].copy()
        side_out[f"{side}_trade_explained_rate"] = np.divide(
            explained_sum,
            prev_sum,
            out=np.full_like(explained_sum, np.nan, dtype=float),
            where=prev_sum != 0,
        )
        side_out[f"{side}_trade_explained_share"] = np.divide(
            explained_sum,
            visible_sum,
            out=np.full_like(explained_sum, np.nan, dtype=float),
            where=visible_sum != 0,
        )
        side_out[f"{side}_nontrade_rate"] = np.divide(
            nontrade_sum,
            prev_sum,
            out=np.full_like(nontrade_sum, np.nan, dtype=float),
            where=prev_sum != 0,
        )
        side_frames.append(side_out)

    if not side_frames:
        return pd.DataFrame(
            columns=[
                "book_stream_pos",
                "ts",
                "bid_trade_explained_rate",
                "ask_trade_explained_rate",
                "bid_trade_explained_share",
                "ask_trade_explained_share",
                "bid_nontrade_rate",
                "ask_nontrade_rate",
                "trade_explained_bias",
            ]
        )

    out = side_frames[0]
    for frame in side_frames[1:]:
        out = out.merge(frame, on=["book_stream_pos", "ts"], how="outer")

    out = out.sort_values("book_stream_pos").reset_index(drop=True)
    for col in [
        "bid_trade_explained_rate",
        "ask_trade_explained_rate",
        "bid_trade_explained_share",
        "ask_trade_explained_share",
        "bid_nontrade_rate",
        "ask_nontrade_rate",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["trade_explained_bias"] = out["ask_trade_explained_rate"] - out["bid_trade_explained_rate"]
    return out


def _signal_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"ts", "mid_at_book", "log_mid"}]


def build_event_signal_frame(trade_frame: pd.DataFrame, signal_T: int, a: float) -> pd.DataFrame:
    sign = trade_frame["aggr_sign"].to_numpy(dtype=float)
    qty = trade_frame["qty"].to_numpy(dtype=float)
    signed_weight = sign * np.power(qty, a)
    signed_volume = sign * qty
    abs_volume = qty

    out = trade_frame[["ts", "log_mid"]].copy()
    out["trade_flow_raw"] = rolling_window_sum(signed_weight, signal_T)
    vol_sum = rolling_window_sum(signed_volume, signal_T)
    abs_sum = rolling_window_sum(abs_volume, signal_T)
    out["volume_imbalance"] = np.divide(
        vol_sum,
        abs_sum,
        out=np.full_like(vol_sum, np.nan, dtype=float),
        where=abs_sum != 0,
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def build_clock_signal_frame(
    trade_frame: pd.DataFrame,
    clock_grid: pd.DataFrame,
    lookback_min: int,
    a: float,
) -> pd.DataFrame:
    trade_ts_ns = pd.to_datetime(trade_frame["ts"], utc=True).astype("int64").to_numpy()
    grid_ts_ns = pd.to_datetime(clock_grid["ts"], utc=True).astype("int64").to_numpy()
    qty = trade_frame["qty"].to_numpy(dtype=float)
    sign = trade_frame["aggr_sign"].to_numpy(dtype=float)
    signed_weight = sign * np.power(qty, a)
    signed_volume = sign * qty
    abs_volume = qty
    window_ns = pd.Timedelta(minutes=lookback_min).value

    out = clock_grid[["ts", "log_mid"]].copy()
    out["trade_flow_raw"] = rolling_time_window_sum(trade_ts_ns, signed_weight, grid_ts_ns, window_ns)
    vol_sum = rolling_time_window_sum(trade_ts_ns, signed_volume, grid_ts_ns, window_ns)
    abs_sum = rolling_time_window_sum(trade_ts_ns, abs_volume, grid_ts_ns, window_ns)
    out["volume_imbalance"] = np.divide(
        vol_sum,
        abs_sum,
        out=np.full_like(vol_sum, np.nan, dtype=float),
        where=abs_sum != 0,
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def smooth_signal_frame(frame: pd.DataFrame, window_min: int) -> pd.DataFrame:
    cols = _signal_columns(frame)
    out = frame[["ts", *cols]].copy().sort_values("ts")
    out = out.set_index("ts").rolling(f"{window_min}min", min_periods=5).mean().reset_index()
    return out


def filter_high_imbalance_frame(
    frame: pd.DataFrame,
    source_col: str,
    q: float,
    *,
    out_col: str | None = None,
) -> pd.DataFrame:
    if source_col not in frame.columns:
        raise KeyError(f"{source_col} not found in frame")
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")
    out_col = out_col or f"{source_col}_high_imbalance"
    values = frame[source_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        out = frame[["ts", source_col]].copy()
        out[out_col] = np.nan
        return out[["ts", out_col]]
    threshold = float(np.nanquantile(np.abs(values[finite]), q))
    out = frame[["ts", source_col]].copy()
    out[out_col] = np.where(
        np.abs(out[source_col].to_numpy(dtype=float)) >= threshold,
        out[source_col].to_numpy(dtype=float),
        0.0,
    )
    return out[["ts", out_col]]


def _coerce_horizon_timedelta(horizon: int | float | str | pd.Timedelta, *, default_unit: str) -> pd.Timedelta:
    if isinstance(horizon, pd.Timedelta):
        return horizon
    if isinstance(horizon, str):
        return pd.to_timedelta(horizon)
    if isinstance(horizon, (int, np.integer, float, np.floating)):
        return pd.to_timedelta(horizon, unit=default_unit)
    raise TypeError(f"Unsupported horizon type: {type(horizon)!r}")


def _format_horizon_label(horizon: pd.Timedelta) -> str:
    total_ns = int(horizon.value)
    if total_ns < 1_000_000_000:
        ms = int(round(total_ns / 1_000_000))
        return f"{ms}ms"
    total_seconds = int(round(horizon.total_seconds()))
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"


def build_future_return_frame(
    clock_grid: pd.DataFrame,
    horizons: list[int | float | str | pd.Timedelta],
    *,
    default_unit: str = "min",
) -> pd.DataFrame:
    out = clock_grid[["ts", "log_mid"]].copy().reset_index(drop=True)
    ts_ns = pd.to_datetime(out["ts"], utc=True).astype("int64").to_numpy()
    log_mid = out["log_mid"].to_numpy(dtype=float)
    for horizon in horizons:
        horizon_td = _coerce_horizon_timedelta(horizon, default_unit=default_unit)
        horizon_ns = horizon_td.value
        label = _format_horizon_label(horizon_td)
        future_idx = np.searchsorted(ts_ns, ts_ns + int(horizon_ns), side="left")
        delta = np.full(len(out), np.nan, dtype=float)
        valid = future_idx < len(out)
        delta[valid] = log_mid[future_idx[valid]] - log_mid[valid]
        out[f"fwd_{label}"] = delta
    return out


def summarize_pair(
    signal_frame: pd.DataFrame,
    return_frame: pd.DataFrame,
    signal_col: str,
    horizon: int | float | str | pd.Timedelta,
    *,
    default_unit: str = "min",
) -> dict:
    horizon_td = _coerce_horizon_timedelta(horizon, default_unit=default_unit)
    horizon_label = _format_horizon_label(horizon_td)
    merged = pd.merge_asof(
        return_frame[["ts", f"fwd_{horizon_label}"]].sort_values("ts"),
        signal_frame[["ts", signal_col]].sort_values("ts"),
        on="ts",
        direction="backward",
    ).dropna(subset=[signal_col, f"fwd_{horizon_label}"])
    x = merged[signal_col].to_numpy(dtype=float)
    y = merged[f"fwd_{horizon_label}"].to_numpy(dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
        x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n": int(len(x))}


def scan_signal_frames(
    signal_frames: list[dict],
    return_frame: pd.DataFrame,
    horizons: list[int | float | str | pd.Timedelta],
    *,
    default_unit: str = "min",
    horizon_label: str = "min",
) -> pd.DataFrame:
    rows = []
    for spec in signal_frames:
        frame = spec["frame"]
        for signal_col in spec["signal_cols"]:
            for horizon in horizons:
                metrics = summarize_pair(frame, return_frame, signal_col, horizon, default_unit=default_unit)
                rows.append({
                    **{k: v for k, v in spec.items() if k != "frame" and k != "signal_cols"},
                    "signal_col": signal_col,
                    f"horizon_{horizon_label}": horizon,
                    **metrics,
                })
    return pd.DataFrame(rows)


def summarize_candidates(scan_df: pd.DataFrame, *, horizon_col: str = "horizon_min") -> pd.DataFrame:
    group_keys = ["signal_mode", "signal_family", "lookback_label", "smooth_min", "a"]

    grouped = (
        scan_df.groupby(group_keys, dropna=False)
        .agg(
            mean_abs_pearson=("pearson", lambda s: float(np.nanmean(np.abs(s)))),
            mean_abs_spearman=("spearman", lambda s: float(np.nanmean(np.abs(s)))),
            best_abs_pearson=("pearson", lambda s: float(np.nanmax(np.abs(s)))),
            best_abs_spearman=("spearman", lambda s: float(np.nanmax(np.abs(s)))),
        )
        .reset_index()
    )

    tmp = scan_df[group_keys + [horizon_col, "pearson"]].copy()
    tmp["_abs_pearson"] = tmp["pearson"].abs()
    best_idx = tmp.groupby(group_keys, dropna=False)["_abs_pearson"].idxmax()
    best_horizons = (
        tmp.loc[best_idx, group_keys + [horizon_col]]
        .rename(columns={horizon_col: "best_horizon"})
        .reset_index(drop=True)
    )
    return grouped.merge(best_horizons, on=group_keys, how="left")


def plot_candidate_curves(
    scan_df: pd.DataFrame,
    *,
    signal_mode: str,
    signal_family: str,
    horizon_col: str = "horizon_min",
) -> None:
    subset = scan_df[(scan_df["signal_mode"] == signal_mode) & (scan_df["signal_family"] == signal_family)].copy()
    if subset.empty:
        return
    cand = (
        summarize_candidates(subset, horizon_col=horizon_col)
        .sort_values(["best_abs_pearson", "mean_abs_pearson"], ascending=False)
        .iloc[0]
    )
    if pd.isna(cand["a"]):
        a_label = "n/a"
    else:
        a_label = f'{cand["a"]}'

    picked = subset[
        (subset["lookback_label"] == cand["lookback_label"])
        & (subset["smooth_min"] == cand["smooth_min"])
        & (subset["a"].fillna(-9999) == (cand["a"] if not pd.isna(cand["a"]) else -9999))
    ].sort_values(horizon_col)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].plot(picked[horizon_col], picked["pearson"], marker="o", label="Pearson")
    axes[1].plot(picked[horizon_col], picked["spearman"], marker="o", label="Spearman")
    for ax, title, ylabel in [
        (axes[0], f"{signal_mode} / {signal_family} Pearson", "Pearson"),
        (axes[1], f"{signal_mode} / {signal_family} Spearman", "Spearman"),
    ]:
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("future horizon")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Best candidate: lookback={cand['lookback_label']}, smooth={cand['smooth_min']}m, a={a_label}")
    plt.tight_layout()
    plt.show()
