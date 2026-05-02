# 04 Summary

Canonical notebook:

- [04_feature_comparison.ipynb](/Users/hoangdeveloper/PycharmProjects/exchange-data-backtester/notebooks/orderflow_study/04_feature_comparison.ipynb)

Scope:

- symbol: `ONDOUSDC`
- train: `20260226`, `20260227`
- validation: `20260228`
- stress: `20260301`
- horizons: `1s`, `2s`, `3s`

Core features kept in the final notebook:

- `top_imbalance`
- `trade_flow_imbalance`
- `impact_pressure_raw`
- `impact_pressure_log`

Definitions:

- `top_imbalance = (bid1_qty - ask1_qty) / (bid1_qty + ask1_qty)`
- `trade_flow_imbalance = build_imbalance_for_signal(trade_frame, signal_T=10, a=0.0)`
- `impact_pressure_raw = trade_flow_imbalance / ask1_qty` when flow is non-negative, otherwise `trade_flow_imbalance / bid1_qty`
- `impact_pressure_log = sign(impact_pressure_raw) * log1p(abs(impact_pressure_raw))`

Main findings:

- `trade_flow_imbalance` is the strongest univariate OLS feature on the validation day.
  Validation `R^2` is about `0.1983` at `1s`, `0.1686` at `2s`, and `0.1463` at `3s`.
- `top_imbalance` is the next strongest pure regression feature.
  Validation `R^2` is about `0.1143`, `0.1038`, and `0.0971`.
- `impact_pressure_log` is materially better than `impact_pressure_raw`.
  Validation `R^2` is about `0.1145`, `0.0992`, and `0.0911`, while raw impact is near zero or negative.
- `impact_pressure_raw` is not a usable final feature on its own.
  It keeps the sign ordering, but it is too tail-heavy for the linear fit.
- In univariate logistic classification, `top_imbalance` and `impact_pressure_log` both work, but in different ways.
  `top_imbalance` gives the strongest AUC, while `impact_pressure_log` is still much better than raw impact.

Combined-model finding:

- `flow + top` is the right baseline.
- Adding `impact_pressure_raw` or `impact_pressure_log` on top of `flow + top` does not materially improve the validation metrics in this setup.
  Validation OLS at `1s` is about `0.2565` for `flow+top`, `0.2566` for `flow+top+impact_pressure_raw`, and `0.2556` for `flow+top+impact_pressure_log`.
  The same pattern holds at `2s` and `3s`.
- The same conclusion carries over to the combined logistic check.
  `flow + top` already captures most of the useful short-horizon direction signal here.

Additional combined checks:

- `flow + impact` is better than `impact` alone, but still below `flow + top`.
  On validation, `flow+impact_pressure_log` reaches OLS `R^2` of about `0.2091`, `0.1770`, and `0.1546` at `1s`, `2s`, and `3s`, respectively.
  The corresponding logistic accuracy is about `0.7226`, `0.7089`, and `0.6958`.
- `top + impact` is also better than either feature alone, especially versus raw impact.
  On validation, `top+impact_pressure_log` reaches OLS `R^2` of about `0.1496`, `0.1332`, and `0.1252`.
  The corresponding logistic accuracy is about `0.7301`, `0.7105`, and `0.7128`.
- In short, `impact` is useful as a secondary feature, but it does not replace the information already carried by `flow` and `top`.

What we tried and did not keep in the final notebook:

- volatility-regime features:
  Useful for diagnostics, but not a clean additive predictor in the final unfiltered comparison.
- imbalance encodings:
  Dead zones, signed imbalance, ternary buckets, and five-state buckets helped interpretation more than prediction.
- rolling top imbalance:
  `500ms`, `1s`, `2s`, `3s`, `4s`, and `5s` rolling variants did not beat raw imbalance.
- deeper-book rolling imbalance:
  Two-layer depth was valid, but still did not beat the simpler raw level-1 feature.
- impact pressure with deeper opposite-side depth:
  The two-layer version did not beat the level-1 version.
- alternative flow/book normalizations:
  `flow_over_signed_gap`, `flow_over_depth_ratio`, and related variants were weaker or less stable than the simpler features.
- trade-consumption features:
  Last-trade and trailing-window consumption ratios were intuitive, but they did not beat `impact_pressure_log`.
- alignment and neutral-book regime flags:
  Useful diagnostics, but not a clean unfiltered improvement when used as standalone binary add-ons.

Interpretation:

- If the goal is a small final feature set, the notebook supports:
  - `trade_flow_imbalance`
  - `top_imbalance`
  - `impact_pressure_log` as a secondary feature worth understanding
- If the goal is pure predictive simplicity, `flow + top` is the clean baseline to keep.
- If the goal is to continue feature development, the next experiments should build on `flow + top`, not on `impact` in isolation.
