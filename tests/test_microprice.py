from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.microprice import bucket_imbalance, bucket_spread_ticks, estimate_g1_tables, get_or_build_microprice_labeled_table, get_or_build_pooled_microprice_labeled_table, label_kth_mid_change
from stats.utils.cache import cache_path
from tests_support import write_diffs, write_events, write_gaps, write_schema, write_snapshot, write_trades


def test_microprice_labeled_table_is_cached_per_day(tmp_path: Path) -> None:
    """Verify that the daily microprice table is cached and reused."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")
    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
        ],
    )
    write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [["100.00", "2.00"]], "a": []},
            {"recv_ms": 1300, "recv_seq": 13, "E": 1300, "U": 103, "u": 103, "b": [["100.00", "1.00"]], "a": [["101.00", "0"], ["102.00", "2.00"]]},
        ],
    )

    labeled = get_or_build_microprice_labeled_table(day_dir, on_gap="strict", imbalance_bucket_count=3, spread_bucket_values=[1, 2])

    assert not labeled.empty
    assert {"delta_mid_target", "direction_target", "time_to_target_ms", "spread_bucket", "imbalance_bucket", "day", "day_dir"} <= set(labeled.columns)
    assert {"delta_mid_tau1", "direction_tau1", "tau1_recv_seq", "time_to_tau1_ms"} <= set(labeled.columns)
    assert len(list((day_dir / "cache").rglob("microprice_labeled_*.parquet"))) == 1

    labeled_cached = get_or_build_microprice_labeled_table(day_dir, on_gap="strict", imbalance_bucket_count=3, spread_bucket_values=[1, 2])
    pd.testing.assert_frame_equal(labeled, labeled_cached, check_like=True)


def test_stale_microprice_cache_is_rebuilt(tmp_path: Path) -> None:
    """Verify that stale microprice caches are validated and rebuilt."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")
    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
        ],
    )
    write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [["100.00", "2.00"]], "a": []},
            {"recv_ms": 1300, "recv_seq": 13, "E": 1300, "U": 103, "u": 103, "b": [["100.00", "1.00"]], "a": [["101.00", "0"], ["102.00", "2.00"]]},
        ],
    )

    stale_path = cache_path(
        day_dir,
        "microprice_labeled",
        {
            "on_gap": "strict",
            "event_time_or_recv_time": "recv",
            "imbalance_bucket_count": 3,
            "imbalance_bucket_edges": None,
            "spread_bucket_values": [1, 2],
            "future_move_definition": "next_mid_change",
            "future_move_k": 1,
        },
        ext="parquet",
    )
    pd.DataFrame({"recv_seq": [11], "delta_mid_tau1": [1.0]}).to_parquet(stale_path, index=False)

    rebuilt = get_or_build_microprice_labeled_table(day_dir, on_gap="strict", imbalance_bucket_count=3, spread_bucket_values=[1, 2])

    assert "time_to_target_ms" in rebuilt.columns
    assert "direction_target" in rebuilt.columns
    assert len(rebuilt) > 0


def test_label_kth_mid_change_uses_kth_future_mid_change() -> None:
    """Verify that labeling by kth future mid change skips the correct number of changes."""
    frame = pd.DataFrame(
        {
            "segment_index": [0, 0, 0, 0, 0, 0],
            "epoch_id": [0, 0, 0, 0, 0, 0],
            "recv_seq": [10, 11, 12, 13, 14, 15],
            "recv_time_ms": [1000, 1100, 1200, 1300, 1400, 1500],
            "mid_price": [100.0, 100.0, 101.0, 101.0, 102.0, 102.0],
        }
    )

    labeled = label_kth_mid_change(frame, k=2, time_col_ms="recv_time_ms")

    assert list(labeled["recv_seq"]) == [10, 11]
    assert list(labeled["future_move_k"]) == [2, 2]
    assert list(labeled["tau_k_recv_seq"]) == [14.0, 14.0]
    assert list(labeled["target_mid"]) == [102.0, 102.0]
    assert list(labeled["time_to_target_ms"]) == [400.0, 300.0]
    assert list(labeled["delta_mid_target"]) == [2.0, 2.0]
    assert "delta_mid_tau1" not in labeled.columns


def test_pooled_microprice_tables_use_analysis_cache(tmp_path: Path) -> None:
    """Verify that pooled microprice tables are cached under the analysis cache root."""
    base = tmp_path / "data" / "binance" / "BTCUSDT"
    day_specs = [("20260221", 100, 1000), ("20260222", 200, 2000)]
    for day, last_update_id, recv_base in day_specs:
        day_dir = base / day
        day_dir.mkdir(parents=True, exist_ok=True)
        write_schema(day_dir, "BTCUSDT", day)
        write_gaps(day_dir / f"gaps_BTCUSDT_{day}.csv.gz")
        write_trades(day_dir / f"trades_ws_BTCUSDT_{day}.csv.gz")
        write_events(
            day_dir / f"events_BTCUSDT_{day}.csv.gz",
            [
                [1, recv_base, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            ],
        )
        write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=last_update_id, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
        write_diffs(
            day_dir / f"diffs/depth_diffs_BTCUSDT_{day}.ndjson.gz",
            [
                {"recv_ms": recv_base + 100, "recv_seq": 11, "E": recv_base + 100, "U": last_update_id + 1, "u": last_update_id + 1, "b": [["100.00", "1.50"]], "a": []},
                {"recv_ms": recv_base + 200, "recv_seq": 12, "E": recv_base + 200, "U": last_update_id + 2, "u": last_update_id + 2, "b": [["100.00", "2.00"]], "a": []},
                {"recv_ms": recv_base + 300, "recv_seq": 13, "E": recv_base + 300, "U": last_update_id + 3, "u": last_update_id + 3, "b": [["100.00", "1.00"]], "a": [["101.00", "0"], ["102.00", "2.00"]]},
                {"recv_ms": recv_base + 400, "recv_seq": 14, "E": recv_base + 400, "U": last_update_id + 4, "u": last_update_id + 4, "b": [["100.00", "0"], ["99.00", "1.00"]], "a": []},
            ],
        )

    day_dirs = [base / "20260221", base / "20260222"]
    pooled = get_or_build_pooled_microprice_labeled_table(
        day_dirs,
        cache_root=tmp_path,
        on_gap="strict",
        imbalance_bucket_count=3,
        spread_bucket_values=[1, 2],
    )

    assert set(pooled["day"]) == {"20260221", "20260222"}
    assert len(list((tmp_path / "analysis_cache").rglob("microprice_labeled_pooled_*.parquet"))) == 1

    g1_long, g1_pivot, diagnostics = estimate_g1_tables(pooled, min_obs_per_bucket=1)
    assert not g1_long.empty
    assert not g1_pivot.empty
    assert not diagnostics.empty

    pooled_k2 = get_or_build_pooled_microprice_labeled_table(
        day_dirs,
        cache_root=tmp_path,
        on_gap="strict",
        imbalance_bucket_count=3,
        spread_bucket_values=[1, 2],
        future_move_definition="kth_mid_change",
        future_move_k=2,
    )
    assert set(pooled_k2["future_move_k"]) == {2}
    assert len(list((tmp_path / "analysis_cache").rglob("microprice_labeled_pooled_*.parquet"))) == 2


def test_microprice_bucket_helpers_validate_inputs() -> None:
    """Verify that microprice bucketing helpers reject invalid inputs."""
    imbalance = pd.Series([0.2, 0.2, 0.2])
    spread_ticks = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="bucket_count must be >= 1"):
        bucket_imbalance(imbalance, bucket_count=0, bucket_edges=None)

    with pytest.raises(ValueError, match="bucket_edges must be non-empty"):
        bucket_imbalance(imbalance, bucket_count=None, bucket_edges=[])

    with pytest.raises(ValueError, match="bucket_values must be non-empty"):
        bucket_spread_ticks(spread_ticks, [])
