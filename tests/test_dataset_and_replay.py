from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from stats.io import DatasetIntegrityError, load_day
from stats.replay import compute_ofi_events, iter_market_events, iter_trade_events, replay_book_frames


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
