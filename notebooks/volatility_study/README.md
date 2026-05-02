# Binance Volatility Study

This folder is for Binance `BTCUSDC` volatility research built on the current `stats` package.

Current notebook sequence:

- `09_rough_volatility_h_estimation.ipynb`: adaptation of section 2 of `Volatility is rough` using one-minute returns, a trailing realized-volatility proxy, and rolling `H` diagnostics

Scope:

- keep the notebook descriptive and math-first
- state the source paper and the local data provenance explicitly
- rely on `stats.notebook` and cached tables instead of re-parsing raw recorder files in cells
- treat the roughness estimate as a diagnostic that should be checked for sensitivity to proxy-window choice
