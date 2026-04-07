from __future__ import annotations

import gzip
import csv
import heapq
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import numpy as np
import pandas as pd

from stats.io.dataset import DayDataset, ReplaySegment
from stats.utils.common import ensure_dataset


GapMode = Literal["strict", "skip-segment"]


class ReplayGapError(RuntimeError):
    """Raised when replay cannot bridge or continue a diff stream safely."""
    pass


@dataclass(frozen=True)
class BookEvent:
    """Replay-normalized top-of-book event emitted after a diff update is applied."""
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
    """Inclusive recv-seq interval that can be replayed within one snapshot segment."""
    segment_index: int
    epoch_id: int
    segment_tag: str
    start_recv_seq: int
    end_recv_seq: Optional[int]


@dataclass(frozen=True)
class _ReplaySegmentAction:
    """Internal event describing either an applied diff or the end of a replayable segment."""
    action_type: Literal["applied", "segment_end"]
    segment: ReplaySegment
    payload: Optional[dict[str, Any]] = None
    book: Optional["BinanceBook"] = None
    start_recv_seq: Optional[int] = None
    end_recv_seq: Optional[int] = None
    gap_recv_seq: Optional[int] = None


@dataclass
class BinanceBook:
    """Mutable in-memory Binance order book reconstructed from one snapshot plus diffs."""
    bids: dict[float, float]
    asks: dict[float, float]
    last_update_id: int

    @classmethod
    def from_snapshot_csv(cls, path: Path) -> "BinanceBook":
        """Build the initial in-memory book from a recorder snapshot CSV."""
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
        """Apply one Binance diff payload to the in-memory book state."""
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
        """Return best bid/ask price and quantity, or NaNs if one side is empty."""
        if not self.bids or not self.asks:
            return (np.nan, 0.0, np.nan, 0.0)
        bid_price = max(self.bids)
        ask_price = min(self.asks)
        return bid_price, float(self.bids[bid_price]), ask_price, float(self.asks[ask_price])

    def levels(self, *, top_n: int) -> dict[str, float]:
        """Serialize the current book into `bid{k}` / `ask{k}` columns up to `top_n`."""
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        row: dict[str, float] = {}
        bid_levels = heapq.nlargest(top_n, self.bids.items(), key=lambda item: item[0])
        ask_levels = heapq.nsmallest(top_n, self.asks.items(), key=lambda item: item[0])

        for idx in range(top_n):
            level = idx + 1

            if idx < len(bid_levels):
                bid_price, bid_qty = bid_levels[idx]
                row[f"bid{level}_price"] = float(bid_price)
                row[f"bid{level}_qty"] = float(bid_qty)
            else:
                row[f"bid{level}_price"] = float("nan")
                row[f"bid{level}_qty"] = 0.0

            if idx < len(ask_levels):
                ask_price, ask_qty = ask_levels[idx]
                row[f"ask{level}_price"] = float(ask_price)
                row[f"ask{level}_qty"] = float(ask_qty)
            else:
                row[f"ask{level}_price"] = float("nan")
                row[f"ask{level}_qty"] = 0.0

        return row


def _book_event_from_state(book: BinanceBook, *, payload: dict, segment: ReplaySegment) -> BookEvent:
    """Convert one applied diff and resulting book state into a `BookEvent`."""
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


def _book_frame_row_from_state(book: BinanceBook, *, payload: dict, segment: ReplaySegment, top_n: int) -> dict[str, float | int | str]:
    """Convert one applied diff and resulting book state into a tabular replay row."""
    row: dict[str, float | int | str] = {
        "event_type": "book",
        "recv_seq": int(payload["recv_seq"]),
        "recv_time_ms": int(payload["recv_ms"]),
        "event_time_ms": int(payload.get("E", 0)),
        "epoch_id": int(segment.epoch_id),
        "segment_index": int(segment.index),
        "segment_tag": segment.tag,
    }
    row.update(book.levels(top_n=top_n))
    return row


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as `MM:SS` or `HH:MM:SS` for progress output."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _estimate_replay_total_rows(dataset: DayDataset) -> Optional[int]:
    """Estimate replay work from the derived depth CSV row count when available."""
    path = dataset.paths.book_path
    if path is None or not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            total = sum(1 for _ in handle) - 1
    except Exception:
        return None
    return max(0, total)


def _render_progress(
    current: int,
    total: Optional[int],
    started_at: float,
    *,
    label: str,
    final: bool = False,
) -> None:
    """Render an in-place replay progress line with elapsed time and optional ETA."""
    elapsed = time.perf_counter() - started_at
    if total is None or total <= 0:
        message = f"{label}: {current} rows | elapsed {_format_duration(elapsed)}"
    else:
        ratio = min(1.0, current / total)
        width = 24
        filled = min(width, int(round(width * ratio)))
        bar = "#" * filled + "-" * (width - filled)
        rate = current / elapsed if elapsed > 0 else 0.0
        if rate > 0:
            remaining = max(0.0, (total - current) / rate)
            eta = _format_duration(remaining)
        else:
            eta = "--:--"
        message = (
            f"{label}: [{bar}] {current}/{total} "
            f"{ratio * 100:5.1f}% | elapsed {_format_duration(elapsed)} | eta {eta}"
        )
    end = "\n" if final else ""
    sys.stderr.write("\r" + message + end)
    sys.stderr.flush()


def _iter_replayed_books(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
) -> Iterator[tuple[BinanceBook, dict, ReplaySegment]]:
    """Yield applied book states for every replay-valid diff across all kept segments."""
    for action in _iter_segment_replay_actions(dataset_or_day_dir, on_gap=on_gap):
        if action.action_type != "applied":
            continue
        assert action.book is not None
        assert action.payload is not None
        yield action.book, action.payload, action.segment


def _iter_segment_replay_actions(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
) -> Iterator[_ReplaySegmentAction]:
    """Walk snapshot segments once and emit applied diffs plus terminal segment metadata.

    This is the shared replay state machine used by both book-event replay and
    replay-range construction, so bridge and gap semantics stay identical.
    """
    dataset = ensure_dataset(dataset_or_day_dir)
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
        start_recv_seq: Optional[int] = None
        gap_recv_seq: Optional[int] = None

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
                    start_recv_seq = recv_seq
                    yield _ReplaySegmentAction(
                        action_type="applied",
                        segment=segment,
                        payload=payload,
                        book=book,
                        start_recv_seq=start_recv_seq,
                    )
                    continue
                if U > expected:
                    if on_gap == "strict":
                        raise ReplayGapError(
                            f"Initial bridge failed in segment {segment.index} ({segment.tag}) "
                            f"recv_seq={recv_seq} expected={expected} U={U} u={u}"
                        )
                    gap_recv_seq = recv_seq
                    drain_segment(segment.end_recv_seq)
                    break
                continue

            if U > expected:
                if on_gap == "strict":
                    raise ReplayGapError(
                        f"Gap detected in segment {segment.index} ({segment.tag}) "
                        f"recv_seq={recv_seq} expected={expected} U={U} u={u}"
                    )
                gap_recv_seq = recv_seq
                drain_segment(segment.end_recv_seq)
                break

            book.apply_updates(payload.get("b", []), payload.get("a", []), update_id=u)
            yield _ReplaySegmentAction(
                action_type="applied",
                segment=segment,
                payload=payload,
                book=book,
                start_recv_seq=start_recv_seq,
            )

        if start_recv_seq is not None:
            end_recv_seq = gap_recv_seq - 1 if gap_recv_seq is not None else (
                segment.end_recv_seq - 1 if segment.end_recv_seq is not None else None
            )
            yield _ReplaySegmentAction(
                action_type="segment_end",
                segment=segment,
                start_recv_seq=start_recv_seq,
                end_recv_seq=end_recv_seq,
                gap_recv_seq=gap_recv_seq,
            )


def replay_ranges(dataset_or_day_dir: DayDataset | Path, *, on_gap: GapMode = "strict") -> list[ReplayRange]:
    """Return replay-valid recv-seq intervals for each segment that successfully bridges."""
    ranges: list[ReplayRange] = []
    for action in _iter_segment_replay_actions(dataset_or_day_dir, on_gap=on_gap):
        if action.action_type != "segment_end":
            continue

        ranges.append(
            ReplayRange(
                segment_index=action.segment.index,
                epoch_id=action.segment.epoch_id,
                segment_tag=action.segment.tag,
                start_recv_seq=int(action.start_recv_seq),
                end_recv_seq=action.end_recv_seq,
            )
        )
    return ranges


def iter_book_events(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
) -> Iterator[BookEvent]:
    """Iterate replayed top-of-book events in recv-seq order."""
    for book, payload, segment in _iter_replayed_books(dataset_or_day_dir, on_gap=on_gap):
        yield _book_event_from_state(book, payload=payload, segment=segment)


def replay_book_frames(
    dataset_or_day_dir: DayDataset | Path,
    *,
    top_n: Optional[int] = None,
    on_gap: GapMode = "strict",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Replay the order book into a dataframe with top-of-book or depth-level columns."""
    top_n = 1 if top_n is None else int(top_n)
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    dataset = ensure_dataset(dataset_or_day_dir)
    total_rows = _estimate_replay_total_rows(dataset) if show_progress else None
    progress_label = f"replay top_n={top_n}"
    started_at = time.perf_counter()
    update_every = max(1, (total_rows // 200) if total_rows else 100)

    rows = []
    for idx, (book, payload, segment) in enumerate(_iter_replayed_books(dataset, on_gap=on_gap), start=1):
        rows.append(_book_frame_row_from_state(book, payload=payload, segment=segment, top_n=top_n))
        if show_progress and (idx == 1 or idx % update_every == 0):
            _render_progress(idx, total_rows, started_at, label=progress_label)
    if show_progress:
        _render_progress(len(rows), total_rows, started_at, label=progress_label, final=True)
    return pd.DataFrame(rows)


def replay_top_of_book(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Replay top-of-book states and add derived mid, spread, and spread-bps columns."""
    frames = replay_book_frames(dataset_or_day_dir, on_gap=on_gap, show_progress=show_progress)
    if frames.empty:
        return frames
    out = frames.copy()
    out["mid"] = (out["bid1_price"] + out["ask1_price"]) / 2.0
    out["spread"] = out["ask1_price"] - out["bid1_price"]
    out["spread_bps"] = np.where(out["mid"] > 0, 1e4 * out["spread"] / out["mid"], np.nan)
    return out
