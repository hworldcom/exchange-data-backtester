from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from stats.io import DatasetIntegrityError, load_day
from stats.microprice import (
    estimate_g1_tables,
    get_or_build_microprice_labeled_table,
    get_or_build_pooled_microprice_labeled_table,
    label_kth_mid_change,
)
from stats.notebook import load_day_context, load_market_preview, replay_summary
from stats.replay import compute_ofi_events, iter_market_events, iter_trade_events, replay_book_frames
from stats.tables import get_or_build_market_grid, get_or_build_top_of_book_table, get_or_build_trades_table
from stats.utils.cache import cache_path


def _write_schema(day_dir: Path, symbol: str, day: str) -> None:
    schema = {
        "schema_version": 4,
        "created_utc": "2026-03-01T00:00:00+00:00",
        "files": {
            "events_csv": {"path": f"events_{symbol}_{day}.csv.gz"},
            "gaps_csv": {"path": f"gaps_{symbol}_{day}.csv.gz"},
            "trades_ws_csv": {"path": f"trades_ws_{symbol}_{day}.csv.gz"},
            "depth_diffs_ndjson_gz": {"path": f"diffs/depth_diffs_{symbol}_{day}.ndjson.gz"},
        },
    }
    (day_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")


def _write_events(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "recv_time_ms", "recv_seq", "run_id", "type", "epoch_id", "details_json"])
        writer.writerows(rows)


def _write_gaps(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recv_time_ms", "recv_seq", "run_id", "epoch_id", "event", "details"])


def _write_trades(path: Path, rows: list[list[object]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_time_ms",
                "recv_time_ms",
                "recv_seq",
                "run_id",
                "trade_id",
                "trade_time_ms",
                "price",
                "qty",
                "is_buyer_maker",
            ]
        )
        if rows:
            writer.writerows(rows)


def _write_snapshot(path: Path, *, last_update_id: int, bid_price: float, bid_qty: float, ask_price: float, ask_qty: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "event_id", "side", "price", "qty", "lastUpdateId"])
        writer.writerow([1, 1, "bid", f"{bid_price:.2f}", f"{bid_qty:.2f}", last_update_id])
        writer.writerow([1, 1, "ask", f"{ask_price:.2f}", f"{ask_qty:.2f}", last_update_id])


def _write_diffs(path: Path, payloads: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


def test_load_day_is_lazy_and_wraps_gzip_errors(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")

    events_path = day_dir / "events_BTCUSDT_20260221.csv.gz"
    events_path.write_bytes(b"not-a-valid-gzip")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")

    dataset = load_day(day_dir)
    assert dataset.paths.events_path == events_path

    with pytest.raises(DatasetIntegrityError):
        dataset.load_events()


def test_load_day_reads_instrument_metadata_from_schema(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDC" / "20260222"
    day_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": 5,
        "created_utc": "2026-03-01T00:00:00+00:00",
        "instrument": {
            "exchange": "binance",
            "symbol": "BTCUSDC",
            "base_asset": "BTC",
            "quote_asset": "USDC",
            "asset_source": "exchange_metadata",
            "tick_size": "0.01",
            "tick_size_source": "metadata",
        },
        "files": {
            "events_csv": {"path": "events_BTCUSDC_20260222.csv.gz"},
        },
    }
    (day_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    _write_events(day_dir / "events_BTCUSDC_20260222.csv.gz", [])

    dataset = load_day(day_dir)

    assert dataset.instrument is not None
    assert dataset.instrument.base_asset == "BTC"
    assert dataset.instrument.quote_asset == "USDC"
    assert dataset.instrument.asset_source == "exchange_metadata"


def test_build_segments_uses_resync_boundaries(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")

    _write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1001, 20, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1002, 30, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    _write_snapshot(day_dir / "snapshots" / "snapshot_000003_resync_000001.csv", last_update_id=200, bid_price=200, bid_qty=1, ask_price=201, ask_qty=2)

    dataset = load_day(day_dir)
    segments = dataset.build_segments()

    assert [segment.recv_seq for segment in segments] == [10, 30]
    assert [segment.end_recv_seq for segment in segments] == [20, None]


def test_binance_replay_respects_segments_and_skips_invalid_range(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1090, 1090, 9, 1, 9001, 1090, 99.50, 0.10, 0],
            [1150, 1150, 15, 1, 9002, 1150, 100.50, 0.20, 1],
            [3105, 3105, 32, 1, 9003, 3105, 200.50, 0.30, 0],
        ],
    )

    _write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1001, 20, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1002, 30, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    _write_snapshot(day_dir / "snapshots" / "snapshot_000003_resync_000001.csv", last_update_id=200, bid_price=200, bid_qty=1, ask_price=201, ask_qty=2)
    _write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 2100, "recv_seq": 21, "E": 2100, "U": 999, "u": 999, "b": [], "a": []},
            {"recv_ms": 3100, "recv_seq": 31, "E": 3100, "U": 201, "u": 201, "b": [["200.00", "2.00"]], "a": []},
        ],
    )

    frames = replay_book_frames(load_day(day_dir), top_n=1)

    assert list(frames["recv_seq"]) == [11, 31]
    assert list(frames["epoch_id"]) == [0, 1]
    assert list(frames["bid1_qty"]) == [1.5, 2.0]

    trades = list(iter_trade_events(load_day(day_dir), on_gap="strict"))
    assert [trade.recv_seq for trade in trades] == [15, 32]
    assert [trade.segment_index for trade in trades] == [0, 1]

    market = list(iter_market_events(load_day(day_dir), on_gap="strict"))
    assert [event.event_type for event in market] == ["book", "trade", "book", "trade"]
    assert [event.recv_seq for event in market] == [11, 15, 31, 32]


def test_compute_ofi_events_uses_replayed_top_of_book(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")

    _write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
        ],
    )
    _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    _write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [], "a": [["101.00", "1.25"]]},
        ],
    )

    ofi = compute_ofi_events(load_day(day_dir))

    assert list(ofi["recv_seq"]) == [11, 12]
    assert float(ofi.iloc[0]["ofi"]) == 0.0
    assert float(ofi.iloc[1]["ofi"]) == pytest.approx(0.75)


def test_notebook_helpers_report_skipped_segments_and_preview_valid_data(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260225"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260225")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260225.csv.gz")
    _write_trades(
        day_dir / "trades_ws_BTCUSDT_20260225.csv.gz",
        rows=[
            [1170, 1170, 17, 1, 9002, 1170, 200.50, 0.20, 1],
            [1190, 1190, 19, 1, 9003, 1190, 200.75, 0.30, 0],
        ],
    )
    _write_events(
        day_dir / "events_BTCUSDT_20260225.csv.gz",
        [
            [1, 1000, 6, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1010, 12, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1020, 14, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    _write_snapshot(
        day_dir / "snapshots" / "snapshot_000001_initial.csv",
        last_update_id=100,
        bid_price=100,
        bid_qty=1,
        ask_price=101,
        ask_qty=2,
    )
    _write_snapshot(
        day_dir / "snapshots" / "snapshot_000003_resync_000001.csv",
        last_update_id=150,
        bid_price=200,
        bid_qty=1,
        ask_price=201,
        ask_qty=2,
    )
    _write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260225.ndjson.gz",
        [
            {"recv_ms": 1080, "recv_seq": 8, "E": 1080, "U": 115, "u": 118, "b": [], "a": []},
            {"recv_ms": 1170, "recv_seq": 17, "E": 1170, "U": 151, "u": 151, "b": [["200.00", "1.50"]], "a": []},
            {"recv_ms": 1180, "recv_seq": 18, "E": 1180, "U": 152, "u": 152, "b": [], "a": [["201.00", "1.75"]]},
        ],
    )

    summary = replay_summary(load_day(day_dir), replay_on_gap="skip-segment")
    assert summary == {
        "replay_on_gap": "skip-segment",
        "segments_total": 2,
        "segments_kept": 1,
        "segments_skipped": 1,
    }

    preview = load_market_preview(load_day(day_dir), limit=10, replay_on_gap="skip-segment")
    assert list(preview["recv_seq"]) == [17, 17, 18, 19]
    assert list(preview["event_type"]) == ["book", "trade", "book", "trade"]

    context = load_day_context(
        day_dir,
        include_book=False,
        include_trades=False,
        include_events=False,
        replay_on_gap="skip-segment",
        include_market_preview=True,
        market_preview_limit=10,
    )
    assert context["replay_summary"]["segments_skipped"] == 1
    assert list(context["market_preview"]["recv_seq"]) == [17, 17, 18, 19]


def test_cached_tables_build_and_reload(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1110, 1110, 12, 1, 9001, 1110, 100.50, 0.20, 0],
            [1210, 1210, 13, 1, 9002, 1210, 100.75, 0.10, 1],
        ],
    )
    _write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
        ],
    )
    _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    _write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 14, "E": 1200, "U": 102, "u": 102, "b": [], "a": [["101.00", "1.25"]]},
        ],
    )

    dataset = load_day(day_dir)
    top = get_or_build_top_of_book_table(dataset, on_gap="strict")
    trades = get_or_build_trades_table(dataset)
    grid = get_or_build_market_grid(dataset, grid_freq="100ms", on_gap="strict")

    assert list(top["recv_seq"]) == [11, 14]
    assert "microprice" in top.columns
    assert list(trades["recv_seq"]) == [12, 13]
    assert "signed_qty" in trades.columns
    assert "book_updates" in grid.columns
    assert "trade_count" in grid.columns

    top_cached = get_or_build_top_of_book_table(dataset, on_gap="strict")
    trades_cached = get_or_build_trades_table(dataset)
    grid_cached = get_or_build_market_grid(dataset, grid_freq="100ms", on_gap="strict")

    assert list(top_cached["recv_seq"]) == [11, 14]
    assert list(trades_cached["recv_seq"]) == [12, 13]
    assert grid_cached.shape == grid.shape


def test_microprice_labeled_table_is_cached_per_day(tmp_path: Path) -> None:
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")
    _write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
        ],
    )
    _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    _write_diffs(
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
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(day_dir, "BTCUSDT", "20260221")
    _write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    _write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")
    _write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
        ],
    )
    _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    _write_diffs(
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
    base = tmp_path / "data" / "binance" / "BTCUSDT"
    day_specs = [("20260221", 100, 1000), ("20260222", 200, 2000)]
    for day, last_update_id, recv_base in day_specs:
        day_dir = base / day
        day_dir.mkdir(parents=True, exist_ok=True)
        _write_schema(day_dir, "BTCUSDT", day)
        _write_gaps(day_dir / f"gaps_BTCUSDT_{day}.csv.gz")
        _write_trades(day_dir / f"trades_ws_BTCUSDT_{day}.csv.gz")
        _write_events(
            day_dir / f"events_BTCUSDT_{day}.csv.gz",
            [
                [1, recv_base, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            ],
        )
        _write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=last_update_id, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
        _write_diffs(
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
