from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.io import DatasetIntegrityError, load_day
from stats.notebook import load_day_context, load_market_preview, replay_summary
from stats.replay import compute_ofi_events, get_or_build_ofi_grid, iter_market_events, iter_trade_events, replay_book_frames, replay_ranges
from stats.tables import get_or_build_book_levels_table, get_or_build_market_grid, get_or_build_top_of_book_table, get_or_build_trades_table
from stats.utils.cache import cache_path
from tests_support import (
    assert_strictly_increasing,
    write_depth_snapshot,
    write_diffs,
    write_events,
    write_gaps,
    write_schema,
    write_snapshot,
    write_trades,
)


def test_load_day_is_lazy_and_wraps_gzip_errors(tmp_path: Path) -> None:
    """Verify that loading a day does not read gzip files until a stream is requested."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")

    events_path = day_dir / "events_BTCUSDT_20260221.csv.gz"
    events_path.write_bytes(b"not-a-valid-gzip")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")

    dataset = load_day(day_dir)
    assert dataset.paths.events_path == events_path

    with pytest.raises(DatasetIntegrityError):
        dataset.load_events()


def test_load_day_reads_instrument_metadata_from_schema(tmp_path: Path) -> None:
    """Verify that instrument metadata is loaded from schema.json."""
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
    write_events(day_dir / "events_BTCUSDC_20260222.csv.gz", [])

    dataset = load_day(day_dir)

    assert dataset.instrument is not None
    assert dataset.instrument.base_asset == "BTC"
    assert dataset.instrument.quote_asset == "USDC"
    assert dataset.instrument.asset_source == "exchange_metadata"


def test_build_segments_uses_resync_boundaries(tmp_path: Path) -> None:
    """Verify that replay segments are split at resync boundaries."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")

    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1001, 20, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1002, 30, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    write_snapshot(day_dir / "snapshots" / "snapshot_000003_resync_000001.csv", last_update_id=200, bid_price=200, bid_qty=1, ask_price=201, ask_qty=2)

    dataset = load_day(day_dir)
    segments = dataset.build_segments()

    assert [segment.recv_seq for segment in segments] == [10, 30]
    assert [segment.end_recv_seq for segment in segments] == [20, None]


def test_binance_replay_respects_segments_and_skips_invalid_range(tmp_path: Path) -> None:
    """Verify that replay skips invalid diff ranges and keeps segment boundaries intact."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1090, 1090, 9, 1, 9001, 1090, 99.50, 0.10, 0],
            [1150, 1150, 15, 1, 9002, 1150, 100.50, 0.20, 1],
            [3105, 3105, 32, 1, 9003, 3105, 200.50, 0.30, 0],
        ],
    )

    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1001, 20, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1002, 30, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    write_snapshot(day_dir / "snapshots" / "snapshot_000003_resync_000001.csv", last_update_id=200, bid_price=200, bid_qty=1, ask_price=201, ask_qty=2)
    write_diffs(
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


def test_replay_book_frames_supports_multiple_levels(tmp_path: Path) -> None:
    """Verify that replay_book_frames can reconstruct more than one book level."""
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
    write_depth_snapshot(
        day_dir / "snapshots" / "snapshot_000001_initial.csv",
        last_update_id=100,
        rows=[
            ("bid", 100.0, 1.0),
            ("bid", 99.0, 2.0),
            ("ask", 101.0, 3.0),
            ("ask", 102.0, 4.0),
        ],
    )
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [["99.50", "2.50"]], "a": [["101.00", "0"], ["101.50", "3.50"]]},
        ],
    )

    frames = replay_book_frames(load_day(day_dir), top_n=2)

    assert list(frames["recv_seq"]) == [11, 12]
    assert list(frames["bid1_price"]) == [100.0, 100.0]
    assert list(frames["bid1_qty"]) == [1.5, 1.5]
    assert list(frames["bid2_price"]) == [99.0, 99.5]
    assert list(frames["bid2_qty"]) == [2.0, 2.5]
    assert list(frames["ask1_price"]) == [101.0, 101.5]
    assert list(frames["ask1_qty"]) == [3.0, 3.5]
    assert list(frames["ask2_price"]) == [102.0, 102.0]
    assert list(frames["ask2_qty"]) == [4.0, 4.0]


def test_dataset_recv_seq_contracts_are_strict_and_disjoint(tmp_path: Path) -> None:
    """Verify that raw dataset recv_seq values are strictly ordered and disjoint across streams."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 1, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 2000, 20, 1, "run_stop", 0, json.dumps({})],
        ],
    )
    write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1100, 1100, 13, 1, 9001, 1100, 100.50, 0.10, 0],
            [1200, 1200, 14, 1, 9002, 1200, 100.75, 0.20, 1],
        ],
    )
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1050, "recv_seq": 8, "E": 1050, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1060, "recv_seq": 9, "E": 1060, "U": 102, "u": 102, "b": [["100.00", "1.75"]], "a": []},
        ],
    )
    write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)

    dataset = load_day(day_dir)
    event_seqs = [event.recv_seq for event in dataset.iter_events()]
    diff_seqs = [int(payload["recv_seq"]) for payload in dataset.iter_depth_diffs()]
    trade_seqs = [trade.recv_seq for trade in dataset.iter_trades()]

    assert_strictly_increasing(event_seqs)
    assert_strictly_increasing(diff_seqs)
    assert_strictly_increasing(trade_seqs)
    assert set(event_seqs).isdisjoint(diff_seqs)
    assert set(event_seqs).isdisjoint(trade_seqs)
    assert set(diff_seqs).isdisjoint(trade_seqs)


def test_replay_ranges_are_strictly_ordered_and_non_overlapping(tmp_path: Path) -> None:
    """Verify that replay_ranges returns ordered, non-overlapping replay segments."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(day_dir / "trades_ws_BTCUSDT_20260221.csv.gz")
    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1500, 20, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1600, 30, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    write_snapshot(day_dir / "snapshots" / "snapshot_000001_initial.csv", last_update_id=100, bid_price=100, bid_qty=1, ask_price=101, ask_qty=2)
    write_snapshot(day_dir / "snapshots" / "snapshot_000003_resync_000001.csv", last_update_id=200, bid_price=200, bid_qty=1, ask_price=201, ask_qty=2)
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [["100.00", "1.75"]], "a": []},
            {"recv_ms": 3100, "recv_seq": 31, "E": 3100, "U": 201, "u": 201, "b": [["200.00", "2.00"]], "a": []},
            {"recv_ms": 3200, "recv_seq": 32, "E": 3200, "U": 202, "u": 202, "b": [["200.00", "2.25"]], "a": []},
        ],
    )

    ranges = replay_ranges(load_day(day_dir), on_gap="strict")

    assert [(r.segment_index, r.start_recv_seq, r.end_recv_seq) for r in ranges] == [
        (0, 11, 19),
        (1, 31, None),
    ]

    for prev, cur in zip(ranges, ranges[1:]):
        prev_end = prev.end_recv_seq if prev.end_recv_seq is not None else float("inf")
        assert prev.start_recv_seq < cur.start_recv_seq
        assert prev_end < cur.start_recv_seq


def test_replay_book_frames_can_show_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that replay_book_frames emits a progress hint when requested."""
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
    write_depth_snapshot(
        day_dir / "snapshots" / "snapshot_000001_initial.csv",
        last_update_id=100,
        rows=[
            ("bid", 100.0, 1.0),
            ("bid", 99.0, 2.0),
            ("ask", 101.0, 3.0),
        ],
    )
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.50"]], "a": []},
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [["99.50", "2.50"]], "a": []},
        ],
    )

    frames = replay_book_frames(load_day(day_dir), top_n=2, show_progress=True)
    captured = capsys.readouterr()

    assert list(frames["recv_seq"]) == [11, 12]
    assert "replay top_n=2" in captured.err
    assert "elapsed" in captured.err


def test_compute_ofi_events_uses_replayed_top_of_book(tmp_path: Path) -> None:
    """Verify that OFI events are derived from replayed book states."""
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
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [], "a": [["101.00", "1.25"]]},
        ],
    )

    ofi = compute_ofi_events(load_day(day_dir))

    assert list(ofi["recv_seq"]) == [11, 12]
    assert float(ofi.iloc[0]["ofi"]) == 0.0
    assert float(ofi.iloc[1]["ofi"]) == pytest.approx(0.75)


def test_notebook_helpers_report_skipped_segments_and_preview_valid_data(tmp_path: Path) -> None:
    """Verify that notebook helpers summarize skipped segments and build a valid preview."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260225"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260225")
    write_gaps(day_dir / "gaps_BTCUSDT_20260225.csv.gz")
    write_trades(
        day_dir / "trades_ws_BTCUSDT_20260225.csv.gz",
        rows=[
            [1170, 1170, 17, 1, 9002, 1170, 200.50, 0.20, 1],
            [1190, 1190, 19, 1, 9003, 1190, 200.75, 0.30, 0],
        ],
    )
    write_events(
        day_dir / "events_BTCUSDT_20260225.csv.gz",
        [
            [1, 1000, 6, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 1010, 12, 1, "resync_start", 1, json.dumps({"tag": "resync_000001"})],
            [3, 1020, 14, 1, "snapshot_loaded", 1, json.dumps({"tag": "resync_000001"})],
        ],
    )
    write_snapshot(
        day_dir / "snapshots" / "snapshot_000001_initial.csv",
        last_update_id=100,
        bid_price=100,
        bid_qty=1,
        ask_price=101,
        ask_qty=2,
    )
    write_snapshot(
        day_dir / "snapshots" / "snapshot_000003_resync_000001.csv",
        last_update_id=150,
        bid_price=200,
        bid_qty=1,
        ask_price=201,
        ask_qty=2,
    )
    write_diffs(
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
    """Verify that cached top-of-book, trade, and grid tables build once and reload cleanly."""
    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")
    write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1110, 1110, 12, 1, 9001, 1110, 100.50, 0.20, 0],
            [1210, 1210, 13, 1, 9002, 1210, 100.75, 0.10, 1],
        ],
    )
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


def test_stale_top_of_book_cache_is_rebuilt(tmp_path: Path) -> None:
    """Verify that stale top-of-book cache files are validated and rebuilt."""
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
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [], "a": [["101.00", "1.25"]]},
        ],
    )

    stale_path = cache_path(day_dir, "top_of_book", {"on_gap": "strict"}, ext="parquet")
    pd.DataFrame({"recv_seq": [11], "bid1_price": [100.0]}).to_parquet(stale_path, index=False)

    rebuilt = get_or_build_top_of_book_table(day_dir, on_gap="strict")

    assert {"mid", "spread", "spread_bps", "microprice", "ts"} <= set(rebuilt.columns)
    assert list(rebuilt["recv_seq"]) == [11, 12]


def test_stale_ofi_grid_cache_is_rebuilt(tmp_path: Path) -> None:
    """Verify that stale OFI grid caches are validated and rebuilt."""
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
            {"recv_ms": 1200, "recv_seq": 12, "E": 1200, "U": 102, "u": 102, "b": [], "a": [["101.00", "1.25"]]},
        ],
    )

    params = {"grid_freq": "100ms", "windows_ms": [100, 500], "on_gap": "strict"}
    stale_path = cache_path(day_dir, "ofi_grid", params, ext="parquet")
    pd.DataFrame({"ofi_sum": [1.0]}).to_parquet(stale_path)

    rebuilt = get_or_build_ofi_grid(day_dir, grid_freq="100ms", windows_ms=(100, 500), on_gap="strict")

    assert {"ofi_sum", "ofi_abs_sum", "ofi_count", "ofi_sum_100ms", "ofi_abs_sum_100ms", "ofi_sum_500ms", "ofi_abs_sum_500ms"} <= set(rebuilt.columns)
    assert isinstance(rebuilt.index, pd.DatetimeIndex)
