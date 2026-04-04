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

Guidelines:

- import reusable logic from `stats.*`
- prefer `stats.notebook` for day loading and replay-policy handling
- prefer `stats.tables` for Parquet-backed derived tables
- do not parse raw recorder files directly in notebook cells
- do not implement replay logic in notebooks
- keep notebooks focused on experiments, plotting, and interpretation

Historical notebooks remain under `notebooks_old/`.
