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
- `04_feature_comparison.ipynb`: canonical short-horizon feature notebook for `top_imbalance`, `trade_flow_imbalance`, `impact_pressure_raw`, and `impact_pressure_log`, including univariate OLS/logistic checks and a minimal `flow + top + impact` comparison
- `04_README.md`: summary of the `04` findings plus a compact record of the exploratory variants that were tried and not kept
- `04_01_impact_feature_deep_dive.ipynb`: impact-focused follow-up notebook for understanding how to improve `impact_pressure_raw` and `impact_pressure_log`
- `04_02_orderbook_flow_deep_dive.ipynb`: orderbook / flow deep dive centered on flow-book regimes, alignment diagnostics, and how those regimes feed into impact construction
- `04_03_depth_concentration_entropy.ipynb`: depth concentration / entropy study focused on how tightly liquidity is packed into the top of the ladder and whether concentrated books are easier to sweep
- `05_sudden_burst_diagnostics.ipynb`: interactive sudden-burst visual diagnostics for price, last-N imbalance, and trades/sec
- `06_sudden_burst_clean_plot.ipynb`: cleaner interactive price, sign-imbalance, and rolling-volume plot with optional buy/sell volume and ratio panels
- `07_persistence_visual_diagnostics.ipynb`: visual diagnostics for persistent last-N sign imbalance in event-window or clock-time terms
- `08_burst_effect_visual_diagnostics.ipynb`: visual diagnostics for calm-to-burst trades/sec and volume regimes with editable burst rules
