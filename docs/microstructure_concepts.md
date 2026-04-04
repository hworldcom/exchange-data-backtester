# Microstructure Concepts

This page is a compact reference for the first analysis notebook. It focuses on the difference between trade flow, book pressure, and price-based summary measures.

## Quick Table

| Concept | Source | What it measures | Simple intuition | What it does not tell you | Typical use |
|---|---|---|---|---|---|
| Mid-price | Best bid and best ask | Center of the visible top of book | The fair midpoint between bid and ask | Queue pressure or execution flow | Returns, volatility, price level |
| Spread | Best ask minus best bid | Immediate trading cost | How wide the market is right now | Direction or liquidity imbalance by itself | Tradability, regime detection |
| Microprice | Best bid/ask plus queue sizes | Weighted mid that leans toward the thinner side | The side with less size is more likely to move first | Exact future price path | Short-horizon prediction, imbalance features |
| Trade imbalance | Aggressive buy volume minus aggressive sell volume | Executed flow direction | Who actually traded more aggressively | Passive cancellations or book refill behavior | Flow analysis, signed volume features |
| OFI | Order flow imbalance from book updates | Visible queue change at bid and ask | Did the book pressure build on bid or ask side | Which exact orders were canceled or filled | Book pressure, microstructure signals |
| Return | Change in mid-price over a horizon | Price movement over time | How much the price moved | Why it moved | Label generation, performance analysis |
| Volatility | Dispersion of returns over a window | How noisy the market is | How unstable price is in that period | Direction of moves | Regime analysis, risk, feature scaling |

## How To Read The Differences

The most important split is:

- `trade imbalance` is about what executed
- `OFI` is about how the visible book changed

They can correlate, but they are not interchangeable.

Microprice sits between them:

- it uses current bid/ask prices and queue sizes
- it is not a trade measure
- it is not a direct book-delta measure
- it is a weighted snapshot of pressure at the top of the book

## Practical Recommendations

- Use `spread` and `microprice` for basic market-quality diagnostics.
- Use `trade imbalance` for executed-flow summaries.
- Use `OFI` for passive book-pressure analysis.
- Use `returns` and `volatility` for labels and regime studies.

## Examples

If aggressive buy volume is large but makers refill immediately:

- trade imbalance is positive
- OFI may be close to zero

If no trades happen but bids are canceled:

- trade imbalance is near zero
- OFI can be negative

If the ask is thin relative to the bid:

- microprice moves upward relative to mid
- the book is signaling upward pressure even before a trade happens
