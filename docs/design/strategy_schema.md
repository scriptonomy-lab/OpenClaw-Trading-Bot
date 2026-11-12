# Strategy schema (Phase 2)

This is the reference for the agent and for the article. The agent turns a
plain-English idea into one JSON object. It never writes code. Python
validates the object with `bot/strategy/schema.py` before anything runs. The
building blocks live in `bot/strategy/blocks.py`.

## The shape of a strategy

```json
{
  "schema_version": 1,
  "name": "short name, 3 to 60 characters, one line",
  "description_plain_english": "the agent's repeat-back, 10 to 800 characters",
  "version": 1,
  "universe": "sp500_point_in_time",
  "side": "long",
  "entry": [ {"block": "...", "...": "..."} ],
  "exit": {"holding_days": 3, "stop_loss_pct": null, "profit_target_pct": null},
  "sizing": {"method": "equal_weight", "max_positions": 10},
  "entry_timing": "next_open",
  "rank_by": null
}
```

| Field | Rule |
|---|---|
| `schema_version` | Always 1. May be left out. |
| `name` | 3 to 60 characters, one line, no control characters. |
| `description_plain_english` | The agent's repeat-back of the strategy, 10 to 800 characters. |
| `version` | 1 to 500. The agent's label for which attempt this is. See "Versions tried". |
| `universe` | Only `"sp500_point_in_time"`: stocks in the S&P 500 on the signal day, including names later removed or delisted. May be left out. |
| `side` | Only `"long"`. May be left out. |
| `entry` | 1 to 8 conditions. **All must be true** (AND). No two may be identical. |
| `exit` | At least one of `holding_days`, `stop_loss_pct`, `profit_target_pct`. |
| `sizing` | `method` is only `"equal_weight"` (may be left out). `max_positions` 1 to 50. |
| `entry_timing` | `"next_open"` (default, may be left out) or `"same_open"`. See "Timing". `"same_open"` needs a `gap` condition. |
| `rank_by` | Optional. Which stocks to buy first when more qualify than there are free slots. See "Sizing". Left out means the fixed shuffle. |

General rules:

- Unknown fields are rejected, at every level. The agent cannot add fields.
- Types are strict. `"5"` is not a number, `3.0` is not a whole number and
  `true` is not 1.
- Percent values are in percent units: `3` means 3%.
- Every bound is inclusive.

### Why entry conditions are AND only

There is no OR. One list where every condition must hold is easy to repeat
back, easy to check by hand and hard to get wrong. Most OR requests are really
two strategies. The agent should offer to test them one after the other.

### Python reads the rules back

`describe_rules(strategy)` turns the validated JSON into plain English. The
agent shows this text, together with its own `description_plain_english`,
when it asks the user to confirm. The user then confirms what Python will
actually run, not the agent's paraphrase of it. If the two texts disagree,
the agent has made a mistake and must fix the JSON.

## Timing: what is known when

There are two entry timings. Exits are the same for both.

**`next_open` (the default).** Every entry decision is made at the **close of
day D** (16:00 New York) and fills at the **open of the next trading day**. A
block may use any data known at the close of D, including D's own open, high,
low, close and volume. It never sees anything later.

**`same_open`.** The decision is made at **09:30 New York on day D** and
fills at **D's open**. This suits gap strategies: the gap is known at the
open, so the trade can be taken on the gap day itself instead of the next
morning. At 09:30 on D only these things about D are known, so only these
are used:

- `gap`: D's open (plus any dividend going ex on D) against D-1's close.
- `no_news` and `news_sentiment`: articles from 16:00 New York on D-1 up to,
  but not including, 09:30 New York on D. With `lookback_sessions` N, the
  window starts at 16:00 on the trading day N days before D. Times are
  converted with the New York time zone rules, so daylight saving is handled.
- Every other block (`n_day_return`, `price_vs_moving_average`, `rsi`,
  `relative_volume`, `days_to_earnings`, `days_since_earnings`,
  `spy_filter`) is read **as of the close of D-1**, exactly as a `next_open`
  strategy would read it at that close. `describe_rules` says "as of the
  previous close" next to each of these.
- Membership is D's (index changes are announced in advance).

`same_open` needs at least one `gap` condition. Without one, every condition
would be read at the previous close and the trade would be the same as a
`next_open` trade, under a second strategy id. Holding days, stops and
targets count from the fill day, as for `next_open`. Paper trading records a
same-open buy after D's close, at D's open price, from what was known at
09:30; it is a simulated open order.

Trading days are the days SPY traded.

## Entry blocks

| `block` | Parameters (bounds) | Exact meaning at the close of D |
|---|---|---|
| `gap` | `direction`: `"down"` or `"up"`; `min_gap_pct`: 0.1 to 50 | gap = (open[D] + dividend[D]) / close[D-1] - 1, in percent. Down passes when gap <= -min_gap_pct. Up passes when gap >= min_gap_pct. The dividend is added back so an ex-dividend day does not look like a gap. |
| `n_day_return` | `days`: 1 to 252; `comparison`: `"at_least"` or `"at_most"`; `return_pct`: -90 to 500 | Total return including dividends from the close `days` trading days before D to the close of D, in percent. |
| `price_vs_moving_average` | `ma_days`: 5 to 250; `position`: `"above"` or `"below"` | close[D] compared with the simple average of the last `ma_days` closes, including D. Equal passes neither. |
| `rsi` | `period`: 2 to 50 (default 14); `comparison`; `level`: 1 to 99 | Wilder's RSI of the close (formula below). |
| `relative_volume` | `lookback_days`: 5 to 100 (default 20); `comparison`; `ratio`: 0.1 to 20 | volume[D] divided by the average volume of the `lookback_days` days before D. D itself is not in the average. |
| `no_news` | `lookback_sessions`: 1 to 20 (default 1) | No articles about the stock in the last `lookback_sessions` news windows. 1 means "since the previous close". |
| `news_sentiment` | `lookback_sessions`: 1 to 20 (default 1); `comparison`; `polarity`: -1 to 1; `min_articles`: 1 to 50 (default 1) | At least `min_articles` articles in the windows, and their average polarity is at least or at most `polarity`. |
| `days_to_earnings` | `relation`: `"within"` or `"not_within"`; `days`: 1 to 10 | Trading days from D to the earliest possible next earnings reaction session. Within: at most `days`. Not within: no report visible, or more than `days`. |
| `days_since_earnings` | `relation`; `days`: 0 to 60 | Trading days since the latest possible last earnings reaction session (0 means D). Within: at most `days`. Not within: none yet, or more than `days`. |
| `spy_filter` | `ma_days`: 5 to 250 (default 200); `position`: `"above"` or `"below"` | SPY's close compared with its own `ma_days` simple average. The same answer for every stock. |

Details that matter:

- **Missing data never passes.** Too little history, a missing price, or a
  stock with no news or earnings data at all gives False.
- **News window.** A window runs from 16:00 New York on the previous trading
  day up to, but not including, 16:00 on D. For `same_open` it ends at
  09:30 on D instead (see Timing). Only articles tagged with
  the exact `.US` symbol count, each link once. Polarity is EODHD's
  per-article `sentiment.polarity`, from -1 (negative) to 1 (positive). The
  average over several days weights each day by its article count.
- **News coverage.** Before 2021 "no news" often means "no data" (the live
  sample jumps about 35 times between 2020 and 2021). Any strategy with
  `no_news` or `news_sentiment` gets ONE results window, 2021-01-01 to
  2023-12-31, and no secondary window. 2019-2020 appears only
  as a footnote (`NEWS_COVERAGE_FOOTNOTE`, carried in the window's
  `coverage_warning` field), never as results. Other strategies use 2016-2023.
  `backtest_windows` applies this rule, so the agent does not need to. The
  window is not part of `strategy_id`.
- **RSI formula.** Over the 251 closes ending at D: change = close - previous
  close, gain = max(change, 0), loss = max(-change, 0). Start avg_gain and
  avg_loss with the simple mean of the first `period` values, then each day
  avg = (previous avg * (period - 1) + today) / period. RSI = 100 - 100 /
  (1 + avg_gain / avg_loss). No losses gives 100, no moves gives 50. The fixed
  250-day window makes the value independent of how much history is loaded.
  The cost: values differ slightly from a chart that smooths over all
  history. A recently listed stock uses the closes it has, needing at least
  `period` + 1. A missing day (a halt) is skipped. Without today's close the
  block is False.
- **Relative volume excludes today** so that a spike does not raise its own
  baseline. "Twice normal volume" then means twice the usual level.
- **Earnings.** A before-market report reacts on its report date, an
  after-market report on the next trading day. Report dates are only treated
  as known 10 trading days ahead, because we cannot rebuild when a date was
  first announced. That is why `days_to_earnings` stops at 10. When
  the before/after-market flag is unknown, `days_to_earnings` uses the
  earliest possible reaction session (the report date) and
  `days_since_earnings` the latest (the next trading day). Each errs toward
  "earnings are close", so neither can act on a report before the market
  could. A stock with no earnings data fails both `within` and `not_within`.
- **Volume** is only used as relative volume within one stock. EODHD volume
  is on today's share basis, so comparing raw volume across stocks, or
  multiplying it by price, would quietly use splits from after 2023.
- **Price level.** No block uses an absolute price, so a split between D and
  the end of 2023 cannot change a signal.

## Exits

| Field | Bounds | Meaning |
|---|---|---|
| `holding_days` | 1 to 252 | Sell at the open after `holding_days` trading days. Bought at the open of day F, held F, F+1, ... and sold at the open of F + `holding_days`. |
| `stop_loss_pct` | 0.5 to 50 | Sell when the day's low reaches the buy price minus this percent. Fills at the stop price, or at the open if the stock opens below the stop. |
| `profit_target_pct` | 0.5 to 200 | Sell when the day's high reaches the buy price plus this percent. Fills at the target price, never higher. |

Any combination works, but at least one is required. The first one hit wins.
If the stop and the target are both touched on the same day, we assume the
stop came first, because daily bars cannot tell. A stop or target is checked
from the buy day itself. Full rules are in `docs/design/backtest.md`.

## Sizing

Equal weight. At each decision close, each new buy is sized at the portfolio's
equity divided by `max_positions`, limited by the cash available. Shares are
fractional. There is no leverage. A stock already held is not bought again. If
more stocks qualify than there are free slots, a fixed repeatable shuffle
picks which ones (see `docs/design/backtest.md`), unless `rank_by` is set.

`rank_by` picks by one block's number first, then by the shuffle for ties.
The block must be in the strategy. If the strategy has that block twice, both
must use the same look-back, otherwise it is unclear which number to sort on.

| `rank_by` | Needs | Buys first |
|---|---|---|
| `largest_gap` | `gap` | The largest gap in size, up or down |
| `highest_sentiment` | `news_sentiment` | The highest average polarity over its look-back |
| `strongest_n_day_return` | `n_day_return` | The highest return over its `days` |
| `highest_relative_volume` | `relative_volume` | The highest volume ratio |
| `lowest_rsi` | `rsi` | The lowest RSI over its `period` |

For `same_open` the number is read at the same moment as the condition: the
gap and sentiment at D's open, the rest at D-1's close.

## Ids, versions and windows

- `strategy_id(strategy)` is a hash of the trading rules only: entry, exit,
  sizing, universe, side, schema version, and `entry_timing` and `rank_by`
  when they are not at their defaults. Name, description and version
  are left out, and entry conditions are sorted first. Leaving out the
  defaults keeps the ids of strategies written before those two fields
  existed. So renaming a strategy
  or reordering its conditions gives the same id. The holdout approval record
  uses this id, so each set of rules gets one holdout run.
- `strategy_to_json` and `strategy_from_json` round-trip exactly.
- **Versions tried.** Every report shows how many versions the user tried.
  That number is counted by storage (each distinct `strategy_id` run in the
  conversation), not taken from the agent's `version` field, because every
  number must come from Python.
- `backtest_windows(strategy)` returns the one results window to test (see News coverage).

## Errors the agent can relay

`explain_validation_error(error)` turns a validation error into short lines,
one per problem. Examples from real runs of the code:

```
Entry condition 1 (gap): min_gap_pct must be between 0.1 and 50. You gave 80.
Exit rules: Give at least one exit: holding_days, stop_loss_pct or profit_target_pct.
side must be one of 'long'. You gave 'short'.
Entry condition 1: 'reddit_mentions' is not a supported building block. Supported blocks: 'gap', 'n_day_return', ...
'leverage' is not a supported field. Remove it.
Entry condition 1 (rsi): comparison must be one of 'at_least' or 'at_most'. You gave 'below'.
Exit rules: holding_days must be a whole number. You gave 3.5.
Entry condition 1 (days_to_earnings): days must be a whole number between 1 and 10. You gave 30.
Entry condition 2 is the same as condition 1. Remove one.
entry_timing 'same_open' needs a gap condition. Without one, every condition is read at the previous close, which is the same as buying at the next open. Add a gap condition or use entry_timing 'next_open'.
rank_by 'lowest_rsi' needs an entry condition with block 'rsi'. Add one or remove rank_by.
```

The agent should fix its own JSON when the mistake is its own, and ask the
user when the request itself is out of range.

## Closest supported version

When a request cannot be built exactly, the agent says so plainly, offers the
closest supported version below, and waits for the user to agree. It never
quietly swaps one rule for another.

| Request | Response |
|---|---|
| Short selling, "sell short when..." | Decline the short side. Offer the long-only mirror if one makes sense (for example, buy the opposite signal), or a `spy_filter` that stays in cash in weak markets. |
| Options, futures, crypto, forex | Decline. Only S&P 500 stocks are supported. |
| Leverage, margin, "2x" | Decline. Offer the same rules unlevered. |
| "Buy at the open on the gap day", "buy the gap as it opens" | Supported: `entry_timing` `"same_open"` with a `gap` condition. Say that other conditions are then read at the previous close and news up to 09:30. |
| Intraday entries or exits ("buy at 10:30", "sell at the close", "first 5 minutes") | Explain that trades fill at an open: the next open, or the same open for a gap strategy (`same_open`). Offer one of those. |
| Stops or targets checked intraday by time | Supported only on daily high and low, with the fill rules above. Say so. |
| Trailing stop | Not supported. Offer a fixed `stop_loss_pct`, optionally with `holding_days`. |
| OR logic ("gap down OR RSI under 30") | Offer to test each branch as its own strategy, one after the other. |
| NOT of a block ("not near a moving average") | Use the opposite parameter where one exists (`below` instead of `above`, `at_most` instead of `at_least`, `not_within` instead of `within`). Otherwise decline. |
| ML or AI price prediction | Decline. Offer a rule built from the blocks, such as momentum or RSI. |
| Fundamentals: P/E, earnings growth, revenue, market cap | Not supported. Offer `days_to_earnings` or `days_since_earnings` for earnings timing, or decline. |
| Earnings surprise ("beat estimates") | Not supported yet. Offer `days_since_earnings` (for example within 1 day) combined with `gap` up, as a price-based stand-in. Say clearly that it is a stand-in. If a surprise block is added later, it must use the latest possible reaction session, like `days_since_earnings`. |
| "Biggest gaps first", "most positive news first", "strongest momentum first" when slots are short | Supported with `rank_by` (`largest_gap`, `highest_sentiment`, `strongest_n_day_return`, `highest_relative_volume`, `lowest_rsi`), with the matching block in the strategy. Other orders are not supported. |
| Most traded stocks, "top N by volume", dollar volume, liquidity filters | Not supported and will not be added: EODHD volume is on today's share basis, so it can only be compared within one stock. Offer `relative_volume` (today's volume against the stock's own average) instead. |
| Sectors, industries, a stock list, Nasdaq 100, Russell 2000, small caps, non-US | Decline the universe. Only the point-in-time S&P 500 is supported. Offer the same rules on it. |
| "Only big names", "only stocks above $50" | Not supported. Offer the rules on the full S&P 500. |
| Reddit, Twitter or social mentions | Not available. Offer `news_sentiment` or `no_news`. |
| Analyst ratings, insider trades, short interest, options flow | Not available. Offer a news or price block if one fits, otherwise decline. |
| Live trading, real money, connect my broker | Decline. Approved strategies are paper traded only. |
| Rebalance weekly or monthly, hold "until the signal stops" | Offer `holding_days` (for example 5 for a week, 21 for a month), plus a stop or target if wanted. |
| Position sizes by conviction, volatility or Kelly | Offer equal weight with a chosen `max_positions`. |
| More than 8 conditions, or 2 conditions on the same block | Up to 8 are allowed, and the same block can appear twice with different parameters (for example a return range with `at_least` and `at_most`). Identical repeats are rejected. |
| A test period other than 2016-2023, or "test on 2024" | The window is fixed by the rules. 2024 onward is the holdout and only runs once, after approval. |
| "Make it more profitable", "tune until it works" | Decline to tune toward results. The user may change the rules, and each change counts as another version in the report. |
| Run code, shell commands, read a file or URL | Decline. The agent only fills in this format. |

## Worked examples

Each example validates with `strategy_from_json`, round-trips through
`strategy_to_json`, and was checked by running the code.

### 1. The brief's gap-down example

User: "Buy S&P 500 stocks that gap down 3% or more with no news since the
previous close, sell after 3 days."

Questions the agent should ask: how many stocks to hold at once (here 10),
and whether "3 days" means trading days (it does in this format).

```json
{
  "name": "Gap down, no news",
  "description_plain_english": "Buy S&P 500 stocks whose open is 3% or more below the previous close, with no news since the previous close. Sell after 3 trading days.",
  "version": 1,
  "entry": [
    {"block": "gap", "direction": "down", "min_gap_pct": 3},
    {"block": "no_news", "lookback_sessions": 1}
  ],
  "exit": {"holding_days": 3},
  "sizing": {"max_positions": 10}
}
```

It uses a news block, so it runs on 2021-2023 only, with a footnote about 2019-2020.
To buy on the gap day itself, add `"entry_timing": "same_open"`: the gap is
then read at that morning's open and "no news" covers 16:00 the day before up
to 09:30. That is a different strategy with its own id.
The design also asks for the same rules without `no_news` on 2016-2023, to show what
the filter adds. That is a separate strategy with its own id.

### 2. Momentum

User: "In a bull market, buy stocks up 20% over the last six months that are
still above their 50-day average and not overbought. Hold a month, with a 10%
stop."

The agent must turn "bull market" into `spy_filter` above the 200-day
average, "six months" into 126 trading days, "a month" into 20 trading days
and "not overbought" into RSI at most 70, and say each of these back.

```json
{
  "name": "Trend momentum",
  "description_plain_english": "When SPY is above its 200-day average, buy S&P 500 stocks up at least 20% over the last 126 trading days that close above their 50-day average and have an RSI of 70 or less. Sell after 20 trading days, or earlier on a 10% stop loss.",
  "version": 1,
  "entry": [
    {"block": "spy_filter", "ma_days": 200, "position": "above"},
    {"block": "n_day_return", "days": 126, "comparison": "at_least", "return_pct": 20},
    {"block": "price_vs_moving_average", "ma_days": 50, "position": "above"},
    {"block": "rsi", "period": 14, "comparison": "at_most", "level": 70}
  ],
  "exit": {"holding_days": 20, "stop_loss_pct": 10},
  "sizing": {"max_positions": 10}
}
```

No news block, so it runs on 2016-2023.

### 3. News sentiment

User: "Buy stocks with strongly positive news and a volume spike, but stay
away from earnings. Take profit at 8%, cut at 5%, otherwise out after a
week."

The agent must turn "strongly positive" into a polarity threshold (here 0.5
with at least 3 articles), "volume spike" into a relative volume ratio (here
2) and "away from earnings" into a window (here not within 5 trading days),
and confirm each number with the user.

```json
{
  "name": "Positive news, heavy volume",
  "description_plain_english": "Buy S&P 500 stocks with at least 3 articles since the previous close whose average polarity is 0.5 or higher, on at least twice their normal volume, with no earnings reaction in the next 5 trading days. Sell after 5 trading days, at an 8% profit, or at a 5% loss.",
  "version": 1,
  "entry": [
    {"block": "news_sentiment", "lookback_sessions": 1, "comparison": "at_least", "polarity": 0.5, "min_articles": 3},
    {"block": "relative_volume", "lookback_days": 20, "comparison": "at_least", "ratio": 2},
    {"block": "days_to_earnings", "relation": "not_within", "days": 5}
  ],
  "exit": {"holding_days": 5, "stop_loss_pct": 5, "profit_target_pct": 8},
  "sizing": {"max_positions": 5}
}
```

It uses a news block, so it runs on 2021-2023 only, with a footnote about 2019-2020.
