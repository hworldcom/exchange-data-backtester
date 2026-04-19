# Order Book Imbalance Study

This folder is a draft research workspace inspired by:

`Enhancing Trading Strategies with Order Book Signals`

Source paper:

- `/Users/hoangdeveloper/Downloads/Imbalance_AMF_resubmit.pdf`

## Short Paper Summary

The paper studies top-of-book volume imbalance in the limit order book and shows that it contains useful short-horizon information.

Core idea:

- define imbalance from best bid and best ask queue sizes
- use the imbalance level to predict the next market order side
- use the imbalance level to explain the immediate post-order midprice move

The paper’s main message is that imbalance is a simple state variable that captures buying and selling pressure and can help reduce adverse selection in short-term trading.

## What We Want To Do

This is a draft plan for our data, focused only on order book information for now.

Planned analysis:

- compute top-of-book imbalance from our order book data
- bucket imbalance into sell-heavy, neutral, and buy-heavy regimes
- test whether imbalance predicts the sign of the next midprice move
- test whether imbalance changes the distribution of short-horizon price moves
- compare results across symbols and time-of-day
- build a simple baseline predictive model for short-term price direction

Out of scope for the first draft:

- trade flow imbalance
- execution strategy design
- complex model fitting before the basic signal is validated

## Draft Status

This workspace is intentionally minimal.

- `01_imbalance_basics.ipynb` is a descriptive implementation notebook
- `02_future_price_conditioned_high_activity_imbalance.ipynb` studies future midprice change once the market is already in high total activity plus heavy imbalance
- `03_transition_to_high_activity_imbalance.ipynb` studies entries into high total activity plus heavy imbalance
- `04_book_liquidity_concentration.ipynb` studies where bid and ask liquidity sit inside the top `L` levels using center-of-mass curves, cumulative imbalance, and the bid/ask center-of-mass gap
- `05_transition_to_high_activity_imbalance.ipynb` studies entries into high total activity plus heavy imbalance
- no production code yet
- the goal is to define the analysis direction before implementation

## Proposed Notebook

### `01_imbalance_basics.ipynb`

Purpose:

- establish the simplest possible order-book imbalance analysis
- verify that the signal exists before any modeling or strategy work

Core definitions:

- top-of-book imbalance:
  - `rho_t = (Vb_t - Va_t) / (Vb_t + Va_t)`
  - `Vb_t` is best bid volume, `Va_t` is best ask volume
- imbalance regimes:
  - split `rho_t` into `N_REGIMES` bins on `[-1, 1]`
  - default `N_REGIMES=5`
- market-order intensity over a window `W` ending at time `t`:
  - `lambda_buy_raw(t; W) = count of buy market orders in (t-W, t]`
  - `lambda_sell_raw(t; W) = count of sell market orders in (t-W, t]`
  - `lambda_total_raw(t; W) = lambda_buy_raw(t; W) + lambda_sell_raw(t; W)`
  - `lambda_buy_rate(t; W) = lambda_buy_raw(t; W) / W`
  - `lambda_sell_rate(t; W) = lambda_sell_raw(t; W) / W`
  - `lambda_total_rate(t; W) = lambda_total_raw(t; W) / W`

Window definitions:

- fixed clock-time lookback window:
  - `W` is a constant duration such as `100ms`, `500ms`, `1000ms`, or `3000ms`
  - the window is always `(t-W, t]`
  - this is the default window type for intensity and price-impact work
- rolling event window:
  - `W` is the last `N` market orders before `t`
  - the raw count is fixed by construction
  - the normalized rate uses the actual elapsed clock time covered by those `N` events
- state-aligned window:
  - the window runs from the time the current imbalance regime began until `t`
  - the duration is the time spent in the current regime so far
  - this is useful for regime-conditional diagnostics, but not the default

Suggested sections:

1. Setup and data loading
   - load one or more days of book data
   - inspect the top-of-book fields needed for imbalance
   - confirm timestamp alignment, sampling frequency, and missing-data behavior
2. Imbalance feature construction
   - compute best bid volume and best ask volume
   - define top-of-book imbalance
   - discretize imbalance into regimes, with `N_REGIMES=5` as the default
   - allow the user to change `N_REGIMES` via notebook parameters
   - optionally compute a continuous imbalance series and a regime label side by side
3. State occupancy and dwell time
   - measure total time spent in each imbalance regime `Z`
   - compute the share of the sample in each state
   - summarize dwell-time distributions for each state
   - compare mean, median, tail, and outlier dwell times
   - compare occupancy and dwell time across symbols and time of day
4. Basic imbalance diagnostics
   - plot the imbalance time series
   - show its distribution
   - compare imbalance across time of day
   - inspect whether the regime boundaries produce reasonable state separation
5. Next-event analysis
   - measure the probability of a buy or sell market order after each imbalance regime
   - test whether buy-heavy states are followed by more buy pressure
   - compare next-event probabilities to unconditional baselines
6. Market order intensity
   - compute `lambda_buy`, `lambda_sell`, and `lambda_total`
   - compute both raw count and normalized rate versions for each lambda
   - use the fixed clock-time lookback window as the default
   - compare intensity across imbalance regimes, time of day, and symbols
   - compare the fixed-window definition against event-window and state-aligned variants
   - study whether raw counts or normalized rates are more informative
7. Imbalance transition probabilities
   - estimate how imbalance moves between regimes over time
   - build a transition matrix for imbalance states
   - measure persistence, mean reversion, and regime switching frequency
   - use the same `N_REGIMES` setting as the rest of the notebook
   - compare transitions across symbols, time of day, and spread states
8. Price impact conditioned on trade direction
   - estimate forward price change conditional on the last market-order side and imbalance state
   - study quantities like `E[ΔP_h | buy MO, Z_t = z]` and `E[ΔP_h | sell MO, Z_t = z]`
   - where `ΔP_h` is the forward midprice change over horizon `h`, for example `10ms` to `3000ms`
   - compute a table or heatmap with rows = imbalance regimes and columns = market-order side
   - compare the full conditional distributions, not just the mean
   - compare against simpler baselines such as `E[ΔP_h | buy MO]` and `E[ΔP_h | Z_t = z]`
   - test whether the same imbalance regime has different impact after a buy versus a sell market order
   - summarize the effect with mean, median, quantiles, and sign probabilities
9. Short-horizon price impact
   - measure forward midprice change over clock-time horizons up to `3000ms`
   - start with a small grid such as `10ms`, `50ms`, `100ms`, `250ms`, `500ms`, `1000ms`, `2000ms`, `3000ms`
   - condition the distribution on imbalance regime
   - compare mean, median, and tail behavior across horizons and regimes
   - optionally add event-count horizons later as a robustness check
10. Correlation analysis
   - compute simple correlations between imbalance, intensity, and forward price change
   - check correlation between `rho_t` and `ΔP_h` across horizons
   - check correlation between `lambda_buy`, `lambda_sell`, `lambda_total` and `ΔP_h`
   - compare correlations across imbalance regimes and time of day
   - use both raw and normalized intensity versions
   - keep the analysis descriptive rather than predictive
   - add conditioned correlations, for example:
     - correlation between `rho_t` and `ΔP_h` within each imbalance regime
     - correlation between intensity and `ΔP_h` within each regime
     - correlation by buy MO versus sell MO
     - correlation by time-of-day bucket
11. Summary
   - record what is confirmed
   - record what is inconclusive
   - note the next notebook to build if the signal looks useful

What this notebook should not do:

- combine imbalance with trade flow yet
- fit a complex model
- attempt trading logic or execution simulation

### `02_future_price_conditioned_high_activity_imbalance.ipynb`

Purpose:

- isolate the future-price question from the transition analysis
- condition on the joint regime `lambda_total_bucket == "high"` and `imbalance_family == "heavy"`
- compare the aggregate high/heavy regime against the directional sub-slices `strong buy-heavy` and `strong sell-heavy`
- measure forward midprice change over configurable millisecond horizons
- summarize `mean_delta_mid`, median, tails, `prob_positive`, mean imbalance, and mean total market-order intensity
- use `other` rows as a descriptive baseline outside the target high/heavy regime

### `03_transition_to_high_activity_imbalance.ipynb`

Purpose:

- study how the book enters the joint regime `lambda_total_bucket == "high"` and `imbalance_family == "heavy"`
- estimate joint-state transitions across total-activity bucket and imbalance family
- summarize precursor states before entries into `high|heavy`
- measure episode duration once the target regime begins
- run an event study around entry to compare midprice, imbalance, and total activity before and after the transition

### `04_book_liquidity_concentration.ipynb`

Purpose:

- study where bid and ask liquidity sit across the top `L` book levels
- compute `center_of_mass_bid_L` and `center_of_mass_ask_L` as the main concentration metrics
- smooth each center-of-mass series with a user-selected moving-average window
- compare the selected `L` against the uniform reference `(L + 1) / 2`
- inspect whether liquidity is concentrated near the touch or pushed deeper into the ladder
- compute cumulative bid/ask imbalance across the same `L` levels
- compute `center_of_mass_gap_L = center_of_mass_ask_L - center_of_mass_bid_L` as a compact asymmetry measure
- keep the metric focused on concentration and imbalance rather than execution flow
