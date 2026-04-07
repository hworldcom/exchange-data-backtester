# BTCUSDC Layer Depletion Study

This folder is for the order-book layer depletion study on Binance BTCUSDC.

The goal is to measure how quickly visible depth at layer 1, layer 2, and deeper levels disappears, and to separate that disappearance into:

- trade-driven depletion
- implied cancellation or replacement
- right-censored cases where the level is not fully consumed within the observation window

Working rules:

- load data through `stats.io.load_day`
- replay books through `stats.replay.replay_book_frames`
- keep notebook logic thin and move reusable pieces into `stats.*` once they stabilize
- do not parse recorder files directly in cells unless the replay helper is not enough

Starter notebook:

- `01_layer_depletion.ipynb`
  - define the depletion problem
  - load one BTCUSDC day
  - replay `top_n > 1`
  - inspect cumulative depth and first-pass depletion times

