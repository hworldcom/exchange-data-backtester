# BTCUSDC Price-Level Survival Study

This folder is for the BTCUSDC price-level survival study on Binance.

The current goal is to track the exact price that is visible at an initial level and ask:

- when that exact price next disappears from the tracked book
- when we lose sight of it because it falls below the tracked window
- how much of the initial displayed quantity is explained by aggressive trades at that same price
- how much remains as implied non-trade removal, such as cancellation or replacement
- how long it takes for the initial displayed quantity to be matched by trades only
- how long it takes for the initial displayed quantity to be matched by trades plus implied non-trade removal

The two main summary tables are:

- `price_level_summary`
  - one row per `aggressor_side` and initial `level`
  - tells you how often the tracked price fully disappeared versus stayed visible
  - separates real disappearance from censoring by tracked depth or sample end
  - reports the time-to-disappearance only for rows that actually disappeared
- `queue_match_summary`
  - one row per `aggressor_side` and initial `level`
  - compares trade-only queue consumption with queue consumption after implied non-trade removal
  - reports how often the initial visible queue was matched by trades only versus trades plus implied removals
  - reports the corresponding wait-time proxies for each match definition

In one sentence:

- `price_level_summary` asks whether the exact price at `t0` survived, disappeared, or was censored.
- `queue_match_summary` asks whether the initial displayed queue at that price was plausibly worked through by flow.

How to read `price_levels`:

- each row starts from one book snapshot and one initial level, for example "the bid visible at level 1 at `t0`"
- the row then tracks that exact price forward until one of three things happens: the price disappears, the price moves below the tracked depth window, or the sample ends
- there is no fixed forward horizon; the look-ahead is open-ended and event-driven

Working rules:

- load data through `stats.io.load_day`
- replay books through `stats.replay.replay_book_frames`
- keep notebook logic thin and move reusable pieces into `stats.*` once they stabilize
- do not parse recorder files directly in cells unless the replay helper is not enough

Starter notebook:

- `01_layer_depletion.ipynb`
  - define the price-level survival problem
  - load one BTCUSDC day
  - replay a deeper tracked book
  - build initial price-level observations
  - summarize disappearance plus censoring by depth or sample end
  - estimate trade-only and queue-consumption wait times
