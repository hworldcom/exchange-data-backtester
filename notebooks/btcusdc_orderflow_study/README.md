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

## Working Principles

- Keep notebook cells thin and move reusable logic into `stats.*` when it stabilizes.
- Prefer raw-trade or cached trade-table analysis over ad hoc event compression.
- Treat same-sign clustering as a diagnostic, not as the primary state representation.
- Match the paper's contemporaneous scaling analysis before adding predictive extensions.
- Start with one day for iteration speed, then expand to multi-day robustness.

## Proposed Notebook Sequence

### `01_trade_flow_diagnostics.ipynb`

Purpose:

- establish BTCUSDC trade-flow structure before any paper-style scaling work.

Expected content:

- child-trade size distribution,
- log-size distribution,
- same-sign run-length distribution,
- sign autocorrelation in trade time,
- optional persistence by size bucket.

This notebook also carries the project scope inline:

- what from the paper is being tested,
- what is adapted to Binance BTCUSDC trade data,
- what remains out of scope because true metaorders are not observed.

It keeps the earlier "same-sign clustering" insight, but without using run-compressed events as the main analysis dataset.

### `02_imbalance_scaling.ipynb`

Purpose:

- define generalized imbalance on raw trades,
- study distributions and scaling of imbalance moments.

Expected content:

- `I_T^a` for multiple `a`,
- imbalance distributions for several `T`,
- rescaled distribution checks,
- log-log estimation of moment slopes versus `T`,
- slope summaries versus `a`.

This is the first notebook that directly targets the paper's core empirical claims.

### `03_return_covariance_correlation.ipynb`

Purpose:

- measure the relationship between price changes and generalized imbalance.

Expected content:

- trade-time return `Delta_T`,
- covariance `E[Delta_T * I_T^a]` versus `T`,
- effective log-log slopes versus `a`,
- correlation `R_a(T)` line plots,
- `(a, T)` heatmaps.

This notebook should be the closest analogue to the paper's Sections 6 and 7.
It should also explain the distinction between:

- raw coupling via covariance,
- normalized coupling via correlation,
- and why both are useful when comparing different `T` and `a`.

### `04_robustness_and_comparison.ipynb`

Purpose:

- stress-test the findings and compare them with the old notebook framing.

Expected content:

- clipped vs unclipped sizes,
- midprice vs trade-price returns,
- one-day vs multi-day comparison,
- old time-bar view versus new raw-trade view,
- summary of what carries over and what does not.

### `05_signal_decay_and_execution_window.ipynb`

Purpose:

- measure how quickly the signal decays after it is observed,
- estimate how much execution time is available before the edge becomes stale.

Expected content:

- separate notation for signal window, execution delay, and return horizon,
- trade-time decay curves as a function of delayed entry in number of trades,
- clock-time decay curves as a function of delayed entry in milliseconds or seconds,
- calibration by signal-strength bucket after delay,
- comparison of immediate-entry edge versus delayed-entry edge.

This notebook should treat clock time as a first-class object, because execution constraints are set by wall-clock latency rather than event count alone.

## Minimal First Milestone

If we want to move quickly, the first useful milestone is:

1. `01_trade_flow_diagnostics.ipynb`
2. `02_imbalance_scaling.ipynb`

That gives us a clean base before we build covariance and correlation surfaces.

## Notes On The Old Notebook

Relevant prior work:

- `notebooks_old/research_summary/BTCUSDT_microstructure_cleaned.ipynb`

What to carry forward:

- the observation that BTC trades can cluster in long same-sign bursts,
- the practical value of comparing midprice and trade-price returns,
- the usefulness of scanning the weighting exponent `a`.

What not to carry forward as the main design:

- same-sign run-compressed events as the central representation,
- treating `a = 0` as degenerate because of the event construction,
- focusing mainly on forward-return prediction instead of paper-style scaling laws.
