# BTCUSDC Order-Flow Study

This folder is for a BTCUSDC order-flow study inspired by:

`The Subtle Interplay between Square-root Impact, Order Imbalance & Volatility: A Unifying Framework`

Source paper:

- `/Users/hoangdeveloper/Downloads/2506.07711v6.pdf`

The goal is not to reproduce the paper mechanically. The goal is to build a clean, notebook-based research track that:

- reuses the current `stats` package and cached tables,
- tests the paper's main empirical objects on Binance BTCUSDC data,
- preserves BTC-specific diagnostics that were useful in the old exploratory work,
- separates paper-faithful analysis from older signal-discovery notebooks.

Current notebook sequence:

- `01_trade_flow_diagnostics.ipynb`: raw tape diagnostics, run lengths, imbalance scaling basics, and trade-count-to-clock-time spans
- `02_imbalance_scaling.ipynb`: moment scaling of trade-flow imbalance
- `03_return_covariance_correlation.ipynb`: trade-flow imbalance versus future returns across trade-time and clock-time horizons
- `04_mid_term_trade_flow_signals.ipynb`: mid-term trade-flow signal screening with event-time and clock-time smoothing
- `05_sudden_burst_diagnostics.ipynb`: interactive sudden-burst visual diagnostics for price, last-N imbalance, and trades/sec
- `06_sudden_burst_clean_plot.ipynb`: cleaner interactive price, sign-imbalance, and rolling-volume plot with optional buy/sell volume and ratio panels
- `07_persistence_visual_diagnostics.ipynb`: visual diagnostics for persistent last-N sign imbalance in event-window or clock-time terms
- `08_burst_effect_visual_diagnostics.ipynb`: visual diagnostics for calm-to-burst trades/sec and volume regimes with editable burst rules
