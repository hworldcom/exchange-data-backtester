from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path


def write_schema(day_dir: Path, symbol: str, day: str) -> None:
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


def write_events(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "recv_time_ms", "recv_seq", "run_id", "type", "epoch_id", "details_json"])
        writer.writerows(rows)


def write_gaps(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recv_time_ms", "recv_seq", "run_id", "epoch_id", "event", "details"])


def write_trades(path: Path, rows: list[list[object]] | None = None) -> None:
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


def write_snapshot(path: Path, *, last_update_id: int, bid_price: float, bid_qty: float, ask_price: float, ask_qty: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "event_id", "side", "price", "qty", "lastUpdateId"])
        writer.writerow([1, 1, "bid", f"{bid_price:.2f}", f"{bid_qty:.2f}", last_update_id])
        writer.writerow([1, 1, "ask", f"{ask_price:.2f}", f"{ask_qty:.2f}", last_update_id])


def write_depth_snapshot(path: Path, *, last_update_id: int, rows: list[tuple[str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "event_id", "side", "price", "qty", "lastUpdateId"])
        for side, price, qty in rows:
            writer.writerow([1, 1, side, f"{price:.2f}", f"{qty:.2f}", last_update_id])


def write_diffs(path: Path, payloads: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


def assert_strictly_increasing(values: list[int]) -> None:
    assert values == sorted(values)
    assert len(values) == len(set(values))
