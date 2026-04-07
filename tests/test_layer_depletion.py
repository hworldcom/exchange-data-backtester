from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stats.analysis import compute_price_level_survival, estimate_implied_cancellations, summarize_price_level_queue_matches, summarize_price_level_survival
from stats.analysis import build_layer_event_stream, compute_trade_depletion
from stats.io import load_day
from stats.tables import get_or_build_book_levels_table, get_or_build_trades_table
from tests_support import write_depth_snapshot, write_diffs, write_events, write_gaps, write_schema, write_trades


def test_layer_depletion_and_implied_cancellation_estimators(tmp_path: Path) -> None:
    """Verify that depletion and implied-cancellation estimators produce sensible values on a small replay."""
    import json

    day_dir = tmp_path / "data" / "binance" / "BTCUSDT" / "20260221"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_schema(day_dir, "BTCUSDT", "20260221")
    write_gaps(day_dir / "gaps_BTCUSDT_20260221.csv.gz")

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
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.00"], ["99.00", "2.00"]], "a": [["101.00", "3.00"], ["102.00", "4.00"]]},
        ],
    )
    write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1200, 1200, 12, 1, 9001, 1200, 101.00, 1.00, 0],
            [1300, 1300, 13, 1, 9002, 1300, 101.00, 2.00, 0],
            [1400, 1400, 14, 1, 9003, 1400, 101.00, 4.00, 0],
        ],
    )

    dataset = load_day(day_dir)
    book_levels = get_or_build_book_levels_table(dataset, top_n=2, on_gap="strict")
    trades = get_or_build_trades_table(dataset)
    stream = build_layer_event_stream(book_levels, trades)

    depletion = compute_trade_depletion(stream, max_level=2)
    first_book = depletion[(depletion["book_recv_seq"] == 11) & (depletion["aggressor_side"] == "buy")].sort_values("level")
    assert list(first_book["level"]) == [1, 2]
    assert list(first_book["delay_ms"].astype(int)) == [200, 300]
    assert list(first_book["trade_events_to_depletion"].astype(int)) == [2, 3]
    assert list(first_book["censored"]) == [False, False]

    write_events(
        day_dir / "events_BTCUSDT_20260221.csv.gz",
        [
            [1, 1000, 10, 1, "snapshot_loaded", 0, json.dumps({"tag": "initial"})],
            [2, 2000, 20, 1, "snapshot_loaded", 1, json.dumps({"tag": "followup"})],
        ],
    )
    write_depth_snapshot(
        day_dir / "snapshots" / "snapshot_000002_followup.csv",
        last_update_id=150,
        rows=[
            ("bid", 100.0, 1.0),
            ("bid", 99.0, 2.0),
            ("ask", 101.0, 1.0),
            ("ask", 102.0, 4.0),
        ],
    )
    write_diffs(
        day_dir / "diffs" / "depth_diffs_BTCUSDT_20260221.ndjson.gz",
        [
            {"recv_ms": 1100, "recv_seq": 11, "E": 1100, "U": 101, "u": 101, "b": [["100.00", "1.00"], ["99.00", "2.00"]], "a": [["101.00", "3.00"], ["102.00", "4.00"]]},
            {"recv_ms": 2000, "recv_seq": 21, "E": 2000, "U": 151, "u": 151, "b": [["100.00", "1.00"], ["99.00", "2.00"]], "a": [["101.00", "1.00"], ["102.00", "4.00"]]},
        ],
    )
    write_trades(
        day_dir / "trades_ws_BTCUSDT_20260221.csv.gz",
        rows=[
            [1200, 1200, 12, 1, 9001, 1200, 101.00, 1.00, 0],
        ],
    )

    dataset = load_day(day_dir)
    book_levels = get_or_build_book_levels_table(dataset, top_n=2, on_gap="strict", force=True)
    trades = get_or_build_trades_table(dataset, force=True)
    stream = build_layer_event_stream(book_levels, trades)
    cancels = estimate_implied_cancellations(stream, max_level=2)

    ask1 = cancels[(cancels["book_recv_seq"] == 11) & (cancels["book_side"] == "ask") & (cancels["level"] == 1)].iloc[0]
    assert bool(ask1["price_stable"]) is True
    assert ask1["trade_qty_at_price"] == pytest.approx(1.0)
    assert ask1["visible_reduction"] == pytest.approx(2.0)
    assert ask1["implied_cancel_qty"] == pytest.approx(1.0)


def test_price_level_survival_tracks_reranking_and_disappearance() -> None:
    """Verify that price-level survival distinguishes reranking from disappearance."""
    book_levels = pd.DataFrame(
        {
            "recv_seq": [11, 12, 13],
            "recv_time_ms": [1100, 1200, 1300],
            "event_time_ms": [1100, 1200, 1300],
            "bid1_price": [100.0, 101.0, 101.0],
            "bid1_qty": [5.0, 2.0, 2.0],
            "bid2_price": [99.0, 100.0, 99.0],
            "bid2_qty": [4.0, 5.0, 4.0],
            "ask1_price": [101.0, 102.0, 102.0],
            "ask1_qty": [3.0, 3.0, 3.0],
            "ask2_price": [102.0, 103.0, 103.0],
            "ask2_qty": [4.0, 4.0, 4.0],
        }
    )
    trades = pd.DataFrame(
        {
            "recv_seq": [11, 12],
            "recv_time_ms": [1115, 1215],
            "event_time_ms": [1115, 1215],
            "trade_time_ms": [1115, 1215],
            "trade_id": [1, 2],
            "price": [100.0, 100.0],
            "qty": [2.0, 1.0],
            "side": ["sell", "sell"],
        }
    )

    outcomes = compute_price_level_survival(book_levels, trades, max_initial_level=2, tracked_top_n=2)

    bid1 = outcomes[(outcomes["book_recv_seq"] == 11) & (outcomes["book_side"] == "bid") & (outcomes["level"] == 1)].iloc[0]
    assert bid1["status"] == "disappeared"
    assert bool(bid1["disappeared"]) is True
    assert bool(bid1["censored"]) is False
    assert bid1["censor_reason"] == ""
    assert int(bid1["terminal_recv_seq"]) == 13
    assert bid1["delay_ms"] == pytest.approx(200.0)
    assert bid1["observed_trade_qty_at_price"] == pytest.approx(3.0)
    assert bid1["trade_explained_ratio"] == pytest.approx(0.6)
    assert bid1["residual_nontrade_ratio"] == pytest.approx(0.4)

    bid2 = outcomes[(outcomes["book_recv_seq"] == 11) & (outcomes["book_side"] == "bid") & (outcomes["level"] == 2)].iloc[0]
    assert bid2["status"] == "fell_below_window"
    assert bool(bid2["fell_below_window"]) is True
    assert bool(bid2["censored"]) is True
    assert bid2["censor_reason"] == "depth_window"
    assert pd.isna(bid2["residual_nontrade_ratio"])

    bid1_late = outcomes[(outcomes["book_recv_seq"] == 12) & (outcomes["book_side"] == "bid") & (outcomes["level"] == 1)].iloc[0]
    assert bid1_late["status"] == "survived_to_end"
    assert bool(bid1_late["survived_to_end"]) is True
    assert bool(bid1_late["censored"]) is True
    assert bid1_late["censor_reason"] == "sample_end"

    summary = summarize_price_level_survival(outcomes)
    buy_l1 = summary[(summary["aggressor_side"] == "sell") & (summary["level"] == 1)].iloc[0]
    assert buy_l1["disappeared_share"] == pytest.approx(1.0 / 3.0)
    assert buy_l1["censored_share"] == pytest.approx(2.0 / 3.0)
    assert buy_l1["censored_by_depth_share"] == pytest.approx(0.0)
    assert buy_l1["mean_trade_explained_ratio"] == pytest.approx(0.6)

    buy_l2 = summary[(summary["aggressor_side"] == "sell") & (summary["level"] == 2)].iloc[0]
    assert buy_l2["censored_share"] == pytest.approx(2.0 / 3.0)
    assert buy_l2["censored_by_depth_share"] == pytest.approx(1.0 / 3.0)


def test_price_level_queue_match_times_capture_replenishment_and_nontrade_removal() -> None:
    """Verify that queue-match timing counts replenishment and implied non-trade removal."""
    book_levels = pd.DataFrame(
        {
            "recv_seq": [10, 20, 30, 40],
            "recv_time_ms": [1000, 1200, 1400, 1600],
            "event_time_ms": [1000, 1200, 1400, 1600],
            "bid1_price": [100.0, 100.0, 100.0, 99.0],
            "bid1_qty": [100.0, 20.0, 20.0, 50.0],
            "bid2_price": [99.0, 99.0, 99.0, 98.0],
            "bid2_qty": [50.0, 50.0, 50.0, 50.0],
            "ask1_price": [101.0, 101.0, 101.0, 101.0],
            "ask1_qty": [30.0, 30.0, 30.0, 30.0],
            "ask2_price": [102.0, 102.0, 102.0, 102.0],
            "ask2_qty": [40.0, 40.0, 40.0, 40.0],
        }
    )
    trades = pd.DataFrame(
        {
            "recv_seq": [11, 12, 35],
            "recv_time_ms": [1110, 1120, 1510],
            "event_time_ms": [1110, 1120, 1510],
            "trade_time_ms": [1110, 1120, 1510],
            "trade_id": [1, 2, 3],
            "price": [100.0, 100.0, 100.0],
            "qty": [60.0, 40.0, 10.0],
            "side": ["sell", "sell", "sell"],
        }
    )

    outcomes = compute_price_level_survival(book_levels, trades, max_initial_level=1, tracked_top_n=2)

    first = outcomes[(outcomes["book_recv_seq"] == 10) & (outcomes["book_side"] == "bid") & (outcomes["level"] == 1)].iloc[0]
    assert first["status"] == "disappeared"
    assert bool(first["trade_matched"]) is True
    assert first["trade_match_time_ms"] == pytest.approx(1120.0)
    assert first["time_to_trade_matched_ms"] == pytest.approx(120.0)
    assert bool(first["queue_matched"]) is True
    assert first["queue_match_time_ms"] == pytest.approx(1120.0)
    assert first["time_to_queue_matched_ms"] == pytest.approx(120.0)

    second = outcomes[(outcomes["book_recv_seq"] == 20) & (outcomes["book_side"] == "bid") & (outcomes["level"] == 1)].iloc[0]
    assert bool(second["trade_matched"]) is False
    assert pd.isna(second["time_to_trade_matched_ms"])
    assert bool(second["queue_matched"]) is True
    assert second["queue_match_time_ms"] == pytest.approx(1600.0)
    assert second["time_to_queue_matched_ms"] == pytest.approx(400.0)

    queue_summary = summarize_price_level_queue_matches(outcomes)
    sell_l1 = queue_summary[(queue_summary["aggressor_side"] == "sell") & (queue_summary["level"] == 1)].iloc[0]
    assert sell_l1["trade_match_share"] == pytest.approx(1.0 / 4.0)
    assert sell_l1["queue_match_share"] == pytest.approx(3.0 / 4.0)
    assert sell_l1["median_time_to_trade_matched_ms"] == pytest.approx(120.0)
    assert sell_l1["median_time_to_queue_matched_ms"] == pytest.approx(200.0)


def test_price_level_survival_outputs_obey_internal_invariants() -> None:
    """Verify the internal invariants that should always hold on survival outputs."""
    book_levels = pd.DataFrame(
        {
            "recv_seq": [10, 20, 30, 40],
            "recv_time_ms": [1000, 1200, 1400, 1600],
            "event_time_ms": [1000, 1200, 1400, 1600],
            "bid1_price": [100.0, 100.0, 100.0, 99.0],
            "bid1_qty": [100.0, 20.0, 20.0, 50.0],
            "bid2_price": [99.0, 99.0, 99.0, 98.0],
            "bid2_qty": [50.0, 50.0, 50.0, 50.0],
            "ask1_price": [101.0, 101.0, 101.0, 101.0],
            "ask1_qty": [30.0, 30.0, 30.0, 30.0],
            "ask2_price": [102.0, 102.0, 102.0, 102.0],
            "ask2_qty": [40.0, 40.0, 40.0, 40.0],
        }
    )
    trades = pd.DataFrame(
        {
            "recv_seq": [11, 12, 35],
            "recv_time_ms": [1110, 1120, 1510],
            "event_time_ms": [1110, 1120, 1510],
            "trade_time_ms": [1110, 1120, 1510],
            "trade_id": [1, 2, 3],
            "price": [100.0, 100.0, 100.0],
            "qty": [60.0, 40.0, 10.0],
            "side": ["sell", "sell", "sell"],
        }
    )

    outcomes = compute_price_level_survival(book_levels, trades, max_initial_level=1, tracked_top_n=2)

    assert (outcomes["censored"] == (outcomes["fell_below_window"] | outcomes["survived_to_end"])).all()
    assert (~outcomes["disappeared"] | ~outcomes["censored"]).all()
    assert (~outcomes["queue_matched"] | (outcomes["queue_match_time_ms"] >= outcomes["book_time_ms"])).all()
    assert (~outcomes["trade_matched"] | (outcomes["trade_match_time_ms"] >= outcomes["book_time_ms"])).all()

    both_matched = outcomes[outcomes["trade_matched"] & outcomes["queue_matched"]]
    assert (both_matched["time_to_queue_matched_ms"] <= both_matched["time_to_trade_matched_ms"]).all()

    queue_summary = summarize_price_level_queue_matches(outcomes)
    assert (queue_summary["queue_match_share"] >= queue_summary["trade_match_share"]).all()
    assert (queue_summary["queue_match_share"] <= 1.0).all()


def test_price_level_survival_output_schema_contract() -> None:
    """Verify that the price-level survival output schema stays stable."""
    book_levels = pd.DataFrame(
        {
            "recv_seq": [10],
            "recv_time_ms": [1000],
            "event_time_ms": [1000],
            "bid1_price": [100.0],
            "bid1_qty": [5.0],
            "ask1_price": [101.0],
            "ask1_qty": [6.0],
        }
    )
    trades = pd.DataFrame(columns=["recv_seq", "recv_time_ms", "event_time_ms", "trade_time_ms", "trade_id", "price", "qty", "side"])

    outcomes = compute_price_level_survival(book_levels, trades, max_initial_level=1, tracked_top_n=1)

    expected_columns = [
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
    assert list(outcomes.columns) == expected_columns
