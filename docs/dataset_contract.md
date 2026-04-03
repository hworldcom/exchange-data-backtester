# Dataset Contract

This project does not import Python code from the recorder repository.

It does depend on the recorder dataset contract: the folder layout, file names, row semantics, and replay rules produced by the recorder.

## What "Dataset Contract" Means

The dataset contract is the agreed interface between:

- the recorder project, which writes day folders
- the research/backtest project, which reads them

The contract defines:

- which files exist
- what each file means
- which artifacts are authoritative
- how rows are ordered
- how replay boundaries are determined

If the recorder output format changes, this project may need code changes even though there is no direct Python dependency.

## Expected Day Layout

Typical day folder:

```text
DATA_ROOT/<exchange>/<symbol>/<day>/
  schema.json
  events_<symbol>_<day>.csv.gz
  gaps_<symbol>_<day>.csv.gz
  trades_ws_<symbol>_<day>.csv.gz
  orderbook_ws_depth_<symbol>_<day>.csv.gz
  diffs/
    depth_diffs_<symbol>_<day>.ndjson.gz
  snapshots/
    snapshot_<event_id>_<tag>.csv
    snapshot_<event_id>_<tag>.json
```

Not every file is required for every workflow.

For Binance replay in this project, the required inputs are:

- `schema.json`
- `events_*.csv.gz`
- `diffs/*.ndjson.gz`
- `snapshots/*.csv`

For trade-aware backtesting, add:

- `trades_ws_*.csv.gz`

## File Semantics

`schema.json`
- metadata and file locations
- used to resolve paths first, with directory scanning as fallback

`events_*.csv.gz`
- authoritative event ledger
- defines replay segment boundaries
- provides the canonical `recv_seq` timeline

`gaps_*.csv.gz`
- recorder-side diagnostics about observed gaps/discontinuities
- useful for QA, but not the primary replay driver in the current research code

`trades_ws_*.csv.gz`
- normalized trade stream
- carries the same global `recv_seq` ordering key as the event ledger and diffs

`orderbook_ws_depth_*.csv.gz`
- derived top-N book output recorded after sync
- useful for parity checks or convenience analysis
- not the source of truth for replay

`diffs/*.ndjson.gz`
- raw replay-grade depth updates
- authoritative incremental order book updates

`snapshots/*.csv`
- snapshot state used to seed replay for Binance
- includes `lastUpdateId` used for Binance bridge logic

`snapshots/*.json`
- raw exchange snapshots when available
- important for exchanges whose replay/checksum depends on exact string representations

## Ordering Rules

The key ordering field is:

- `recv_seq`

Use `recv_seq` as the deterministic sequence for:

- book diffs
- trades
- merged market event streams

Do not merge streams by wall-clock timestamp if `recv_seq` is available.

## Replay Boundaries

The event ledger defines replayable segments.

Important event types:

- `snapshot_loaded`
  - starts a new replay segment
- `resync_start`
  - ends the current segment

For a given day:

- multiple segments may exist
- replay is not assumed to be continuous across the full day
- each segment must be reconstructed independently

## Binance Replay Rules

For each segment:

1. Load the snapshot referenced by the `snapshot_loaded` event.
2. Ignore diffs with `recv_seq <= snapshot_loaded.recv_seq`.
3. Find the first bridge diff satisfying:

```text
U <= lastUpdateId + 1 <= u
```

4. Apply subsequent diffs sequentially.
5. End the segment at the next `resync_start` or next `snapshot_loaded`.

Gap handling modes:

- `strict`
  - raise on bridge failure or sequence gap
- `skip-segment`
  - stop replaying the broken segment and continue with the next valid segment

## Valid Backtest Window

For the backtest stream, a segment becomes valid:

- at the first successfully bridged diff

and remains valid until:

- the segment boundary, or
- the point where a sequence gap invalidates the segment

This matters because trades may occur after the last emitted book update but before the segment boundary. Those trades are still part of the valid replay window.

## Current Project Assumptions

The current stats project assumes:

- replay implementation is Binance-specific
- replay uses `events + snapshots + diffs`
- merged market streams use `recv_seq`
- `orderbook_ws_depth` is derived, not canonical

If the recorder changes any of those assumptions, update this document and the loader/replay code together.
