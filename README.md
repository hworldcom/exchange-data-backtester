# exchange-data-backtester

Research-side loader and analytics package for recorder datasets.

This project is intentionally separate from the recorder so notebooks stay thin and reusable logic lives in Python modules instead of ad hoc notebook cells.

The repository/project name is `exchange-data-backtester`. The Python package name remains `stats`.

## Current Scope

- lazy loading of recorder day folders
- segment-aware dataset parsing using `events_*.csv.gz`
- Binance replay from `snapshots + diffs + events`
- interleaved `recv_seq`-ordered book/trade event streams for backtesting
- cached Parquet derived tables for fast notebook reruns
- OFI generation from replayed top-of-book
- notebook-friendly feature helpers for book, trades, and forward returns

Old exploratory notebooks are preserved under `notebooks_old/`.

## Dataset Contract

This project is built against the dataset contract documented in:

- `docs/dataset_contract.md`

The important rules are:

- `events_*.csv.gz` is the authoritative timeline
- replay uses `snapshots + diffs + events`
- `orderbook_ws_depth_*.csv.gz` is a derived artifact, not the canonical replay input
- `recv_seq` is the deterministic ordering key

The recorder repo is the producer of these datasets, but this project depends on the data contract rather than importing recorder code.

Related analysis reference:

- `docs/microstructure_concepts.md`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from pathlib import Path

from stats.io import load_day
from stats.replay import get_or_build_ofi_grid, iter_market_events
from stats.tables import (
    get_or_build_market_grid,
    get_or_build_top_of_book_table,
    get_or_build_trades_table,
)

day = load_day(Path("/path/to/data/binance/BTCUSDT/20260221"))

events = day.load_events()
top_of_book = get_or_build_top_of_book_table(day, on_gap="skip-segment")
trades = get_or_build_trades_table(day)
market_grid = get_or_build_market_grid(day, grid_freq="100ms", on_gap="skip-segment")
ofi_grid = get_or_build_ofi_grid(day, grid_freq="100ms")
market_events = iter_market_events(day)
```

## Package Layout

- `stats/io`
  - day-folder metadata and lazy readers
- `stats/replay`
  - segment-aware replay and OFI
- `stats/tables.py`
  - cached derived tables for notebooks
- `stats/features`
  - reusable book, trade, and return helpers
- `stats/analysis`
  - grid-based analysis helpers for notebooks
- `stats/utils`
  - cache helpers

## Caching

Derived artifacts are cached under:

```text
DAY_DIR/cache/v2/
```

Delete that folder to recompute.

## Tests

```bash
python3 -m pytest tests -q
```
