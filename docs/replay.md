# Replay Notes

## Current Implementation

The current research-side replay is intentionally narrow and correct:

- exchange support: Binance
- source of truth: `events + snapshots + diffs`
- segment boundaries: `snapshot_loaded` and `resync_start`
- backtest stream: merged book and trade events ordered by `recv_seq`

It does **not** replay from `orderbook_ws_depth_*.csv.gz`.

The input contract for all of this is documented in:

- `docs/dataset_contract.md`

## Why

Recorder day folders can contain multiple valid replay segments due to resyncs.

A replay that assumes:

- one snapshot
- one continuous diff stream
- one valid book for the whole day

is incorrect for many real datasets.

## Binance Replay Model

For each segment:

1. Load the snapshot CSV referenced by `snapshot_loaded`.
2. Skip diffs with `recv_seq <= snapshot_loaded.recv_seq`.
3. Find the bridge diff satisfying:

```text
U <= lastUpdateId + 1 <= u
```

4. Apply sequential diffs after that.
5. End the segment at the next `resync_start` or next `snapshot_loaded`.

If a gap is encountered inside a segment:

- `on_gap="strict"` raises
- `on_gap="skip-segment"` drops the rest of that segment

## OFI

OFI is computed from replayed top-of-book frames, not from raw diff files directly.

That keeps OFI aligned with:

- snapshot bridging
- resync boundaries
- sequence continuity

## Backtest Stream

The replay layer now exposes:

- book events from successful Binance replay segments
- trade events filtered to replay-valid ranges
- merged market events ordered by `recv_seq`

That gives backtests one deterministic event stream instead of separate notebook-side joins.

## Future Extensions

- replay result exports to parquet
- checksum-exchange support for Kraken and Bitfinex
- strategy/backtest interfaces on top of market events
