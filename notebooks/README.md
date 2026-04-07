# Notebooks

This directory is for new notebooks built on the current `stats` package.
Notebook code should stay thin: load cached tables and use helpers from `stats.*` instead of re-parsing raw recorder files in cells.

Starter notebooks:

- `00_load_binance_data.ipynb`
  - locate the sibling recorder data and load a Binance day
  - preview replay metadata and cached derived tables
- `01_basic_market_analysis.ipynb`
  - descriptive analysis on cached top-of-book, trades, and grid tables
  - distributions, returns, volatility, spread, and trade activity
- `02_microprice_g1_analysis.ipynb`
  - strict replay-based level-1 microprice research notebook
  - estimates the first-order signal `G1(I,S) = E[M_{tau1} - M_t | I,S]`
  - builds a row-level first-order microprice proxy `mid + G1(state)`
  - uses the `stats.microprice` package, split into `features`, `labeling`, and `tables`, while keeping the public import path stable

Research subfolders:

- `btcusdc_orderflow_study/`
  - BTCUSDC order-flow and imbalance analysis workspace inspired by the paper `The Subtle Interplay between Square-root Impact, Order Imbalance & Volatility: A Unifying Framework`
  - contains the multi-notebook roadmap for diagnostics, scaling, correlation, and robustness work
- `btcusdc_layer_depletion_study/`
  - BTCUSDC book-layer depletion and implied-cancellation workspace
  - focuses on how long visible depth survives at each level of the order book

Guidelines:

- import reusable logic from `stats.*`
- prefer `stats.notebook` for day loading and replay-policy handling
- prefer `stats.tables` for Parquet-backed derived tables
- keep time parameters grid-aligned: any forward-return horizon, OFI window, or decision interval must be an exact multiple of the chosen grid frequency
- do not rely on silent truncation of time horizons; non-aligned values now raise errors by design
- do not parse raw recorder files directly in notebook cells
- do not implement replay logic in notebooks
- keep notebooks focused on experiments, plotting, and interpretation

Historical notebooks remain under `notebooks_old/`.
