# Pairs Correlation

This workspace tests whether `BTCUSDC` is a useful short-horizon price predictor for `ONDOUSDC` on the shared Binance sample from:

- `20260226`
- `20260227`
- `20260228`

The notebook in this folder works with cached top-of-book tables, aligns both symbols on a shared clock grid, and checks:

- contemporaneous return correlation,
- lead-lag return correlation,
- simple directional hit rate,
- a quick visual readout of how fast the predictive edge decays with lag.

The current sample suggests strong same-time co-movement but only a weak predictive edge once BTC is shifted ahead of ONDO by even a few seconds.
