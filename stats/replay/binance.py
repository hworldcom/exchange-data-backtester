from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd

from stats.io.dataset import DayDataset, ReplaySegment, load_day


GapMode = Literal["strict", "skip-segment"]


class ReplayGapError(RuntimeError):
    pass


@dataclass(frozen=True)
class BookEvent:
    event_type: str
    recv_seq: int
    recv_time_ms: int
    event_time_ms: int
    epoch_id: int
    segment_index: int
    segment_tag: str
    bid1_price: float
    bid1_qty: float
    ask1_price: float
    ask1_qty: float

    @property
    def mid(self) -> float:
        return (self.bid1_price + self.ask1_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask1_price - self.bid1_price

    @property
    def spread_bps(self) -> float:
        if self.mid <= 0:
            return float("nan")
        return 1e4 * self.spread / self.mid


@dataclass(frozen=True)
class ReplayRange:
    segment_index: int
    epoch_id: int
    segment_tag: str
    start_recv_seq: int
    end_recv_seq: Optional[int]


@dataclass
class BinanceBook:
    bids: dict[float, float]
    asks: dict[float, float]
    last_update_id: int

    @classmethod
    def from_snapshot_csv(cls, path: Path) -> "BinanceBook":
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        last_update_id: Optional[int] = None
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ReplayGapError(f"Empty snapshot csv: {path}")
            for row in reader:
                side = str(row["side"])
                price = float(row["price"])
                qty = float(row["qty"])
                last_update_id = int(row.get("lastUpdateId") or 0)
                if qty <= 0:
                    continue
                if side == "bid":
                    bids[price] = qty
                elif side == "ask":
                    asks[price] = qty
        if last_update_id is None:
            raise ReplayGapError(f"Snapshot missing lastUpdateId: {path}")
        return cls(bids=bids, asks=asks, last_update_id=last_update_id)

    def apply_updates(self, bids_updates: list[list[str]], asks_updates: list[list[str]], *, update_id: int) -> None:
        for price_raw, qty_raw in bids_updates:
            price = float(price_raw)
            qty = float(qty_raw)
            if qty <= 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price_raw, qty_raw in asks_updates:
            price = float(price_raw)
            qty = float(qty_raw)
            if qty <= 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty
        self.last_update_id = int(update_id)

    def best(self) -> tuple[float, float, float, float]:
        if not self.bids or not self.asks:
            return (np.nan, 0.0, np.nan, 0.0)
        bid_price = max(self.bids)
        ask_price = min(self.asks)
        return bid_price, float(self.bids[bid_price]), ask_price, float(self.asks[ask_price])


def _ensure_dataset(dataset_or_day_dir: DayDataset | Path) -> DayDataset:
    if isinstance(dataset_or_day_dir, DayDataset):
        return dataset_or_day_dir
    return load_day(dataset_or_day_dir)


def _book_event_from_state(book: BinanceBook, *, payload: dict, segment: ReplaySegment) -> BookEvent:
    bid1_price, bid1_qty, ask1_price, ask1_qty = book.best()
    return BookEvent(
        event_type="book",
        recv_seq=int(payload["recv_seq"]),
        recv_time_ms=int(payload["recv_ms"]),
        event_time_ms=int(payload.get("E", 0)),
        epoch_id=int(segment.epoch_id),
        segment_index=int(segment.index),
        segment_tag=segment.tag,
        bid1_price=float(bid1_price),
        bid1_qty=float(bid1_qty),
        ask1_price=float(ask1_price),
        ask1_qty=float(ask1_qty),
    )


def iter_book_events(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
) -> Iterator[BookEvent]:
    dataset = _ensure_dataset(dataset_or_day_dir)
    segments = dataset.build_segments()
    if not segments:
        raise ReplayGapError(f"No snapshot_loaded segments found under {dataset.day_dir}")
    if not dataset.paths.diff_paths:
        raise FileNotFoundError(f"No diff files found under {dataset.day_dir}")

    diff_iter = iter(dataset.iter_depth_diffs())
    next_diff = next(diff_iter, None)

    def in_segment(recv_seq: int, end_recv_seq: Optional[int]) -> bool:
        if end_recv_seq is None:
            return True
        return recv_seq < end_recv_seq

    def drain_segment(end_recv_seq: Optional[int]) -> None:
        nonlocal next_diff
        if end_recv_seq is None:
            next_diff = None
            return
        while next_diff is not None and int(next_diff.get("recv_seq", 0)) < end_recv_seq:
            next_diff = next(diff_iter, None)

    for segment in segments:
        book = BinanceBook.from_snapshot_csv(segment.snapshot_csv_path)
        bridged = False

        while next_diff is not None and int(next_diff.get("recv_seq", 0)) <= segment.recv_seq:
            next_diff = next(diff_iter, None)

        while next_diff is not None and in_segment(int(next_diff.get("recv_seq", 0)), segment.end_recv_seq):
            payload = next_diff
            next_diff = next(diff_iter, None)

            recv_seq = int(payload.get("recv_seq", 0))
            U = int(payload.get("U", 0))
            u = int(payload.get("u", 0))
            expected = int(book.last_update_id) + 1

            if u <= book.last_update_id:
                continue

            if not bridged:
                if U <= expected <= u:
                    book.apply_updates(payload.get("b", []), payload.get("a", []), update_id=u)
                    bridged = True
                    yield _book_event_from_state(book, payload=payload, segment=segment)
                    continue
                if U > expected:
                    if on_gap == "strict":
                        raise ReplayGapError(
                            f"Initial bridge failed in segment {segment.index} ({segment.tag}) "
                            f"recv_seq={recv_seq} expected={expected} U={U} u={u}"
                        )
                    drain_segment(segment.end_recv_seq)
                    break
                continue

            if U > expected:
                if on_gap == "strict":
                    raise ReplayGapError(
                        f"Gap detected in segment {segment.index} ({segment.tag}) "
                        f"recv_seq={recv_seq} expected={expected} U={U} u={u}"
                    )
                drain_segment(segment.end_recv_seq)
                break

            book.apply_updates(payload.get("b", []), payload.get("a", []), update_id=u)
            yield _book_event_from_state(book, payload=payload, segment=segment)


def replay_ranges(dataset_or_day_dir: DayDataset | Path, *, on_gap: GapMode = "strict") -> list[ReplayRange]:
    dataset = _ensure_dataset(dataset_or_day_dir)
    segments = dataset.build_segments()
    if not segments:
        raise ReplayGapError(f"No snapshot_loaded segments found under {dataset.day_dir}")
    if not dataset.paths.diff_paths:
        raise FileNotFoundError(f"No diff files found under {dataset.day_dir}")

    ranges: list[ReplayRange] = []
    diff_iter = iter(dataset.iter_depth_diffs())
    next_diff = next(diff_iter, None)

    def in_segment(recv_seq: int, end_recv_seq: Optional[int]) -> bool:
        if end_recv_seq is None:
            return True
        return recv_seq < end_recv_seq

    def drain_segment(end_recv_seq: Optional[int]) -> None:
        nonlocal next_diff
        if end_recv_seq is None:
            next_diff = None
            return
        while next_diff is not None and int(next_diff.get("recv_seq", 0)) < end_recv_seq:
            next_diff = next(diff_iter, None)

    for segment in segments:
        book = BinanceBook.from_snapshot_csv(segment.snapshot_csv_path)
        bridged = False
        start_recv_seq: Optional[int] = None
        range_end_recv_seq: Optional[int] = None

        while next_diff is not None and int(next_diff.get("recv_seq", 0)) <= segment.recv_seq:
            next_diff = next(diff_iter, None)

        while next_diff is not None and in_segment(int(next_diff.get("recv_seq", 0)), segment.end_recv_seq):
            payload = next_diff
            next_diff = next(diff_iter, None)

            recv_seq = int(payload.get("recv_seq", 0))
            U = int(payload.get("U", 0))
            u = int(payload.get("u", 0))
            expected = int(book.last_update_id) + 1

            if u <= book.last_update_id:
                continue

            if not bridged:
                if U <= expected <= u:
                    bridged = True
                    start_recv_seq = recv_seq
                    book.last_update_id = u
                    continue
                if U > expected:
                    if on_gap == "strict":
                        raise ReplayGapError(
                            f"Initial bridge failed in segment {segment.index} ({segment.tag}) "
                            f"recv_seq={recv_seq} expected={expected} U={U} u={u}"
                        )
                    drain_segment(segment.end_recv_seq)
                    break
                continue

            if U > expected:
                if on_gap == "strict":
                    raise ReplayGapError(
                        f"Gap detected in segment {segment.index} ({segment.tag}) "
                        f"recv_seq={recv_seq} expected={expected} U={U} u={u}"
                    )
                range_end_recv_seq = recv_seq - 1
                drain_segment(segment.end_recv_seq)
                break

            book.last_update_id = u

        if start_recv_seq is None:
            continue
        if range_end_recv_seq is None:
            range_end_recv_seq = segment.end_recv_seq - 1 if segment.end_recv_seq is not None else None

        ranges.append(
            ReplayRange(
                segment_index=segment.index,
                epoch_id=segment.epoch_id,
                segment_tag=segment.tag,
                start_recv_seq=start_recv_seq,
                end_recv_seq=range_end_recv_seq,
            )
        )
    return ranges


def replay_book_frames(
    dataset_or_day_dir: DayDataset | Path,
    *,
    top_n: Optional[int] = None,
    on_gap: GapMode = "strict",
) -> pd.DataFrame:
    if top_n not in (None, 1):
        raise ValueError("Binance replay_book_frames only supports top_n=1")
    rows = [asdict(event) for event in iter_book_events(dataset_or_day_dir, on_gap=on_gap)]
    return pd.DataFrame(rows)


def replay_top_of_book(dataset_or_day_dir: DayDataset | Path, *, on_gap: GapMode = "strict") -> pd.DataFrame:
    frames = replay_book_frames(dataset_or_day_dir, on_gap=on_gap)
    if frames.empty:
        return frames
    out = frames.copy()
    out["mid"] = (out["bid1_price"] + out["ask1_price"]) / 2.0
    out["spread"] = out["ask1_price"] - out["bid1_price"]
    out["spread_bps"] = np.where(out["mid"] > 0, 1e4 * out["spread"] / out["mid"], np.nan)
    return out
