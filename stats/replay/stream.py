from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from stats.io import DayDataset, TradeRow
from stats.replay.binance import BookEvent, GapMode, ReplayRange, iter_book_events, replay_ranges
from stats.utils.common import ensure_dataset


@dataclass(frozen=True)
class TradeEvent:
    event_type: str
    recv_seq: int
    recv_time_ms: int
    event_time_ms: int
    trade_time_ms: int
    trade_id: int
    run_id: int
    epoch_id: int
    segment_index: int
    segment_tag: str
    price: float
    qty: float
    side: str | None
    is_buyer_maker: int
    exchange: str | None
    symbol: str | None
    ord_type: str | None

    @property
    def notional(self) -> float:
        return self.price * self.qty


MarketEvent = BookEvent | TradeEvent


def _to_trade_event(trade: TradeRow, replay_range: ReplayRange) -> TradeEvent:
    return TradeEvent(
        event_type="trade",
        recv_seq=trade.recv_seq,
        recv_time_ms=trade.recv_time_ms,
        event_time_ms=trade.event_time_ms,
        trade_time_ms=trade.trade_time_ms,
        trade_id=trade.trade_id,
        run_id=trade.run_id,
        epoch_id=replay_range.epoch_id,
        segment_index=replay_range.segment_index,
        segment_tag=replay_range.segment_tag,
        price=trade.price,
        qty=trade.qty,
        side=trade.side,
        is_buyer_maker=trade.is_buyer_maker,
        exchange=trade.exchange,
        symbol=trade.symbol,
        ord_type=trade.ord_type,
    )


def iter_trade_events(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
) -> Iterator[TradeEvent]:
    dataset = ensure_dataset(dataset_or_day_dir)
    ranges = replay_ranges(dataset, on_gap=on_gap)
    if not ranges:
        return

    range_idx = 0
    current = ranges[range_idx]

    for trade in dataset.iter_trades():
        while current.end_recv_seq is not None and trade.recv_seq > current.end_recv_seq:
            range_idx += 1
            if range_idx >= len(ranges):
                return
            current = ranges[range_idx]

        if trade.recv_seq < current.start_recv_seq:
            continue

        yield _to_trade_event(trade, current)


def iter_market_events(
    dataset_or_day_dir: DayDataset | Path,
    *,
    on_gap: GapMode = "strict",
) -> Iterator[MarketEvent]:
    dataset = ensure_dataset(dataset_or_day_dir)
    book_iter = iter(iter_book_events(dataset, on_gap=on_gap))
    trade_iter = iter(iter_trade_events(dataset, on_gap=on_gap))

    next_book = next(book_iter, None)
    next_trade = next(trade_iter, None)

    while next_book is not None or next_trade is not None:
        if next_trade is None or (next_book is not None and next_book.recv_seq <= next_trade.recv_seq):
            yield next_book
            next_book = next(book_iter, None)
            continue
        yield next_trade
        next_trade = next(trade_iter, None)
