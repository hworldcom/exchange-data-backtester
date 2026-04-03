# Notebooks

This directory is for new notebooks built on the current `stats` package.

Starter notebooks:

- `00_load_binance_data.ipynb`
  - locate the sibling recorder data and load a Binance day
- `01_merged_market_events.ipynb`
  - convert the merged event iterator into a DataFrame for inspection

Guidelines:

- import reusable logic from `stats.*`
- do not parse raw recorder files directly in notebook cells
- do not implement replay logic in notebooks
- keep notebooks focused on experiments, plotting, and interpretation

Historical notebooks remain under `notebooks_old/`.
