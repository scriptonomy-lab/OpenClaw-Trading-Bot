# Data layer design (Phase 1)

This note explains how raw EODHD data becomes point-in-time data that the
building blocks can use without seeing the future or the holdout. The code
stubs in `bot/data/` carry the exact signatures. This note explains why.

## Module map

| Module | Job | Status |
|---|---|---|
| `bot/data/eodhd_client.py` | Calls EODHD, caches raw JSON, calls the holdout guard on every request's end date | data-engineer |
| `bot/data/holdout.py` | The holdout guard and the approval session | done |
| `bot/data/prices.py` | Split-adjusted OHLC, volume, dividends, total return | stub |
| `bot/data/membership.py` | Who was in the S&P 500 on each day | stub |
| `bot/data/news.py` | Article counts and sentiment per decision | stub |
| `bot/data/earnings.py` | Days to and since earnings per decision | stub |

Raw JSON goes in, typed rows (TypedDicts) are parsed, and each module returns
plain dataclasses or pandas frames indexed by trading date.

## The trading calendar

Trading days are the dates in the SPY price frame. No holiday library is
used. Every "previous trading day" and "next trading day" in this note means
the neighbor in that index.

## Holdout guard and approval

Everything from 2024-01-01 (`config.HOLDOUT_START`) on is sealed.

- `assert_before_holdout(end_date, *, approved=False)` raises
  `HoldoutAccessError` for any date on or after the start of the holdout.
- `drop_holdout_rows(rows, date_field, *, approved=False)` keeps only rows
  dated before the holdout. A datetime with a time zone is converted to New
  York time before taking its date. A missing or bad date raises. It never
  drops a bad row quietly.

**`approved=True` alone does not unseal anything.** The guard also needs an
open approval session:

```python
with holdout_session(strategy_id):
    assert_before_holdout(some_2024_date, approved=True)  # passes
```

`holdout_session` looks up a stored approval record with `find_approval`. It
refuses when there is no record, when the record says its one holdout run was
already used, or when another session is already open. On entry it calls
`mark_approval_used`, so each approved strategy gets exactly one holdout run,
as the brief requires. Outside a session, `approved=True` raises with a hint.

In Phase 1, `find_approval` always returns None, so nothing can open the
holdout. Phase 5 will read and write the record in `bot/storage.py`. Tests
can replace `find_approval` with monkeypatch to test the approved path.

The session is held in a `ContextVar`, so it does not leak between threads or
async tasks.

## Prices

### Split adjustment

The splits endpoint gives `split` as "new/old", for example
"4.000000/1.000000". The factor is new divided by old: 4.0 for a 4-for-1
split, 0.1 for a 1-for-10 reverse split.

Each price row is divided by the product of the factors of all splits whose
ex-date is after that row. Rows strictly before the ex-date are divided. The
ex-date row is already on the new basis. The endpoint check shows this for
AAPL: 499.23 on 2020-08-28, 129.04 on 2020-08-31.

Prices are anchored on the last split before the holdout. We never fetch
splits after 2023, so the client must pass `to` on or before 2023-12-31 to
the splits and dividends endpoints.

### Delisted names without split data

For delisted tickers, EODHD has splits only after 2018. Before that we infer
them from `adjusted_close / close`. A dividend moves this ratio by a small
step. A split moves it by the split factor between two consecutive rows. A
jump counts as a split only when it is within 5% of a common ratio (2, 3, 1.5,
1/10 and so on). A reported split always wins over an inferred one within
3 days.

### What we do not use

EODHD's `adjusted_close` is computed as of today. A 2023 row already includes
splits and dividends from 2024 to 2026, which is holdout information. So it
is not in the canonical frame. It is used only for the jump test above, which
compares neighboring pre-2024 rows, so a constant later factor cancels out.

For the same reason, dividends use `unadjustedValue`, divided by our own split
factor, and not `value`, which is on today's basis.

### Volume

EODHD volume is already split-adjusted and is never adjusted again. But it is
adjusted to today's basis, which may include splits after 2023. Before the
holdout this equals our basis times one constant. That is fine for ratios
inside one series, such as relative volume. It is wrong for anything in
dollars. Rule: volume is only used in ratios within one symbol's series.

### Total return

The user requires returns with dividends:

`total_return[t] = (close[t] + dividend[t]) / close[t-1] - 1`

with `dividend` on its ex-date and on the same split basis as `close`. The
backtester credits held positions with the dividend as cash on the ex-date
, which matches this.

### The canonical frame

One frame per symbol, index `date` (tz-naive trading days), all float64:
`open, high, low, close, raw_close, volume, dividend, split_factor,
total_return`. See `bot/data/prices.py` for the meaning of each column.

## Membership

Source: `HistoricalTickerComponents` rows with `Code`, `StartDate`,
`EndDate`, `IsActiveNow`, `IsDelisted`. A bare `Code` such as "AAPL" maps to
"AAPL.US".

Holdout rules, applied once when the rows are parsed:

- `StartDate` on or after 2024-01-01: the row is dropped.
- `EndDate` on or after 2024-01-01: treated as None (still a member). A
  removal in 2025 is invisible from 2023.
- `IsActiveNow` and `IsDelisted` describe today. They decide what to download
  and nothing else. They live in `DownloadHint`, never in a spell.

`MembershipHistory.members_on(day)` returns the members on a day.
`universe(MEMBERSHIP_START, BACKTEST_END)` is every name that was ever a
member from 2015 to 2023. That list drives every download, which is what
removes survivorship bias.

Provisional: both `StartDate` and `EndDate` are treated as inclusive. We check
this on a few known index change dates against the `historical=1` snapshots
once the real key works.

## News

### Decision time

A decision on day D is made at 16:00 New York time. The news window for that
decision is half-open:

`[16:00 New York on the previous trading day, 16:00 New York on D)`

Article timestamps carry a time zone. They are converted to New York time
with `zoneinfo`, which applies daylight saving time correctly. A Monday
decision covers Friday 16:00 to Monday 16:00. An article stamped exactly
16:00 on D belongs to the next decision.

In real paper trading the job runs some time after 16:00. News between 16:00
and the run is still left for the next day, so backtest and paper trading
see the same thing.

### Which articles count

Only articles whose `symbols` contain the exact `.US` symbol count. Articles
are de-duplicated per symbol by `link`. A page that returns exactly `limit`
rows means there are more, so the client must page through.

### Sentiment

The primary score is the per-article `sentiment.polarity`, averaged over the
window. It has a timestamp, so it follows the same window rule as counts.

The daily `/sentiments` value has a date and no time, so it may include
articles up to the end of that day. A value dated X is first used at the
close of the first trading day after X. It is a secondary column.

### What the agent sees

Only `NewsWindowSummary` and the daily news frame: counts, average scores
and dates. Titles and text stay in storage.

## Earnings

`report_date` is the announcement day. The raw `date` field is the fiscal
period end and is ignored for timing. The first session that can react:

- BeforeMarket: the report_date session.
- AfterMarket: the next trading day.
- Unknown: the report_date session (the earlier choice).

`days_since_last_earnings` has no lookahead problem.

`days_to_next_earnings` does. The calendar holds the date the report really
happened. We cannot rebuild when that date was first announced. Companies
usually confirm a date a few weeks ahead. Our rule: the value is only filled
in when the next reaction session is within 10 trading days of the decision.
Beyond that it is unknown. The 10 days is provisional. A remaining risk is a
date moved at short notice, which we cannot see in this data.

`actual` and `estimate` may only be used from the reaction session on.

## Building blocks and the lookahead test

A building block is a pure function:

`(data_up_to_decision_date, params) -> bool | pd.Series`

`data_up_to_decision_date` holds the price, news and earnings frames and
membership, each cut with rows dated on or before D. Every frame is indexed
by trading date, so cutting is one `.loc[:D]` per frame.

The lookahead test should be strict: rebuild the frames from raw rows cut at
D, compute signals, and compare with signals from full pre-holdout history at
D. Rebuilding matters because split anchoring moves price levels when a split
happens between D and 2023. So:

- Blocks must be scale-free: returns, ratios, moving averages and RSI on the
  same series.
- Absolute price filters such as "price above $5" use `raw_close`.

A block that depends on the adjusted price level fails the strict test, which
is what the test is for.

## Cache layout expectations

- The client stores raw JSON under `data/cache/eodhd/`, one file per endpoint,
  symbol and date range, exactly as received. The exact path scheme is the
  data-engineer's choice.
- Every request passes an end date on or before 2023-12-31 and calls
  `assert_before_holdout` on it first. Results are also passed through
  `drop_holdout_rows`.
- Derived frames (parquet) go in a separate folder such as
  `data/cache/derived/`. They can always be rebuilt from raw JSON.
- `data/cache/` is never committed.

## Open risks

1. **End of the backtest touches the holdout.** A signal at the 2023-12-29
   close fills at the 2024-01-02 open, and holding period exits spill into
   January 2024. Proposed rule: no entry whose fill date is in the holdout;
   positions still open are valued at the last pre-holdout close
   (2023-12-29). The backtester owner needs to adopt this.
2. **The membership file holds holdout rows.** `HistoricalTickerComponents`
   comes as one response that includes 2024 to 2026 changes. Only
   `membership.py` reads that cache file, and it clips at parse time.
3. **EndDate meaning is unverified** (inclusive or exclusive). One-day error
   at most. Checked against snapshots.
4. **Ticker reuse.** A ticker can belong to different companies over time,
   and renamed companies change tickers. The `symbol-change-history`
   endpoint is the planned check.
5. **News timestamps.** We assume `date` is publication time. If EODHD stores
   ingestion time or edits timestamps later, windows shift. We will spot-check
   a few articles against their source pages.
6. **Earnings announcement dates** cannot be rebuilt (see Earnings).
7. **Inferred splits** can miss unusual ratios or mistake a large special
   dividend for a split. `check_splits_against_prices` lists leftover jumps
   for the data-quality report.
