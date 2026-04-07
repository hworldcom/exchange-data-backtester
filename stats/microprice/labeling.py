"""Future mid-move labeling helpers for microprice state tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats.utils.common import to_utc_datetime_ms


def resolve_future_move_k(*, future_move_definition: str, future_move_k: int) -> int:
    """Normalize the requested future-move definition to a concrete positive k."""
    if future_move_definition not in {"next_mid_change", "kth_mid_change"}:
        raise ValueError(
            "future_move_definition must be one of {'next_mid_change', 'kth_mid_change'}"
        )
    if future_move_definition == "next_mid_change":
        return 1
    if int(future_move_k) < 1:
        raise ValueError("future_move_k must be >= 1")
    return int(future_move_k)


def label_kth_mid_change(
    frame: pd.DataFrame,
    *,
    k: int = 1,
    time_col_ms: str = "recv_time_ms",
) -> pd.DataFrame:
    """Label each state by the kth future mid-price change within its replay segment."""
    if int(k) < 1:
        raise ValueError("k must be >= 1")

    labeled_parts = []
    for _, segment_frame in frame.groupby(["segment_index", "epoch_id"], sort=False):
        part = segment_frame.sort_values("recv_seq").copy()
        part["mid_run_id"] = part["mid_price"].ne(part["mid_price"].shift()).cumsum()
        run_summary = (
            part.groupby("mid_run_id", sort=False)
            .agg(
                run_mid=("mid_price", "first"),
                tau_k_recv_seq=("recv_seq", lambda s: s.iloc[0]),
                tau_k_time_ms=(time_col_ms, lambda s: s.iloc[0]),
            )
            .reset_index()
        )
        run_summary["target_mid"] = run_summary["run_mid"].shift(-k)
        run_summary["tau_k_recv_seq"] = run_summary["tau_k_recv_seq"].shift(-k)
        run_summary["tau_k_time_ms"] = run_summary["tau_k_time_ms"].shift(-k)
        part = part.merge(
            run_summary[["mid_run_id", "target_mid", "tau_k_recv_seq", "tau_k_time_ms"]],
            on="mid_run_id",
            how="left",
        )
        part["future_move_k"] = int(k)
        part["time_to_target_ms"] = part["tau_k_time_ms"] - part[time_col_ms]
        part["delta_mid_target"] = part["target_mid"] - part["mid_price"]
        part["direction_target"] = np.sign(part["delta_mid_target"])
        labeled_parts.append(part)

    if not labeled_parts:
        return pd.DataFrame()

    labeled = pd.concat(labeled_parts, ignore_index=True)
    labeled = labeled.loc[labeled["target_mid"].notna()].copy()
    labeled["tau_k_analysis_ts"] = to_utc_datetime_ms(labeled["tau_k_time_ms"])

    if int(k) == 1:
        labeled["next_mid"] = labeled["target_mid"]
        labeled["tau1_recv_seq"] = labeled["tau_k_recv_seq"]
        labeled["tau1_time_ms"] = labeled["tau_k_time_ms"]
        labeled["time_to_tau1_ms"] = labeled["time_to_target_ms"]
        labeled["delta_mid_tau1"] = labeled["delta_mid_target"]
        labeled["direction_tau1"] = labeled["direction_target"]
        labeled["tau1_analysis_ts"] = labeled["tau_k_analysis_ts"]

    return labeled


def label_next_mid_change(frame: pd.DataFrame, *, time_col_ms: str = "recv_time_ms") -> pd.DataFrame:
    """Convenience wrapper for the next mid-price change label."""
    return label_kth_mid_change(frame, k=1, time_col_ms=time_col_ms)
