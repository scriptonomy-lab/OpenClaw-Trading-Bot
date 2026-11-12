# OpenClaw Trading Bot

You describe a stock trading idea in plain English on Telegram. The bot turns it into exact rules,
tests it on past S&P 500 data with trading costs, and reports the result honestly, including when
the idea loses money. If you approve it, it runs one sealed final test and then paper trades it
(pretend trades, no real money).

> **Safety box**
>
> - **Paper trading only.** No real money, no broker, no live orders.
> - **Long only.** No leverage, no options, no short selling.
> - **The holdout is sealed.** Data from 2024-01-01 onward is never touched until you approve a
>   strategy, and then it is used once.
> - **The AI writes rules, never code.** Python does every calculation. The AI only explains the
>   numbers Python gives it.
> - **Only you can use the bot.** Telegram access is limited to the user IDs you list.

## Quick start

You need a Mac or Linux terminal and git. Nothing else, and no accounts or keys.

```bash
git clone https://github.com/katsdivi/Scriptonomy-EODHD-OpenClaw.git
cd Scriptonomy-EODHD-OpenClaw
./setup.sh
./demo.sh
```

- `./setup.sh` installs `uv` (a Python tool) if it is missing, installs the packages, creates your
  settings file `.env`, and runs the quick tests. It takes a few minutes the first time.
- `./demo.sh` runs one example strategy on **simulated data**. It prints the strategy in plain
  English, then a short table of returns at three cost levels, with the label SIMULATED DATA. It
  takes about half a minute and sends nothing anywhere. The numbers come from made-up prices. They
  prove the tool works. They say nothing about whether a strategy makes money.

Both scripts are safe to run again.

## Using real data

1. Get an EODHD key at https://eodhd.com. You need a paid plan that includes end-of-day prices,
   fundamentals (for S&P 500 membership history), news and the calendar. The free plan cannot reach
   the history this project needs.
2. Open `.env` and paste the key after `EODHD_API_KEY=`. Leave `EODHD_MODE` empty. With a key set,
   the bot uses real data. Without one, it uses simulated data. `.env` is ignored by git. Never
   share it.
3. Check the key and compare real responses with what the code expects:

   ```bash
   uv run python -m bot.verify_live
   ```

   It shows your plan and daily call limit, estimates the download, and waits for you to type
   `DOWNLOAD` before it spends calls. To rehearse the steps without a key, add `--dry-run-fake`.

4. Download the data:

   ```bash
   uv run python scripts/download_data.py
   ```

**Sample or full universe.** By default the bot uses a frozen random sample of 200 of the 741
historical S&P 500 members from 2016 to 2023, including stocks that were later removed or delisted.
The list is in `data/universe_sample.json`, and it was drawn once so every run uses the same names.
The full universe is one flag away:

```bash
uv run python scripts/download_data.py --universe full
```

or set `EODHD_UNIVERSE=full` in `.env`. It costs many more API calls. To see the estimate for each
choice, with no network use:

```bash
uv run python scripts/estimate_api_calls.py --universe sample
uv run python scripts/estimate_api_calls.py --universe full
```

At the time of writing the estimate was about 9,000 calls for the sample and about 36,800 for the
full universe. Download runs stop above a cap of 30,000 calls, so the full universe needs a plan
and a cap change that you decide on yourself. Run the script for current figures.

## Telegram setup

This part needs an OpenAI API key, a Telegram account and Node.js 24.22.1 or newer. The full,
checked steps are in [openclaw/README.md](openclaw/README.md). In short:

1. In Telegram, message **@BotFather**, send `/newbot` and keep the bot token.
2. Message **@userinfobot** to get your numeric user ID.
3. Put `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS` (your numeric ID) and a
   long random `APPROVAL_CODE_SECRET` in `.env`. Make the secret with
   `python3 -c "import secrets; print(secrets.token_hex(32))"`.
4. Install OpenClaw and onboard it with your OpenAI key:

   ```bash
   npm install -g openclaw@latest --allow-scripts=openclaw
   openclaw onboard --install-daemon --auth-choice openai-api-key
   ```

   Keep the agent name `main`. Run `openclaw models`, then set `OPENCLAW_MODEL=openai/<model id>`
   in `.env`.
5. Put the bot token where the background service can read it: add one line,
   `TELEGRAM_BOT_TOKEN=<token>`, to `~/.openclaw/.env`. Do not put the EODHD key there.
6. Fill in the config templates for your machine:

   ```bash
   uv run python scripts/render_openclaw_config.py
   ```

   This writes `openclaw/rendered/`, which is ignored by git. It stops if the model id or user list
   is missing.
7. Load the rendered config and the one-command allowlist, then check them. Steps 6 to 9 in
   [openclaw/README.md](openclaw/README.md) give the exact commands. The checks are
   `openclaw doctor`, `openclaw channels status --probe` and `openclaw skills check`.
8. Create the two scheduled jobs from `openclaw/rendered/cron.md`. They update the paper portfolio
   after each close and send a daily summary, with no AI in the path.
9. Message your bot. Try "Which building blocks can I use?".

**Two data modes.** The bot uses **live** data by default. Say "test this on simulated data"
and it adds `--data simulated` to every command for that request. Simulated mode keeps its own
strategies, results, approvals and paper portfolio, apart from live. The bot never switches mode on
its own, and every result says which mode it came from. Simulated results start with SIMULATED DATA.

## Examples of use

These are messages you send on Telegram. The behaviour is defined in
[openclaw/skills/strategy-lab/SKILL.md](openclaw/skills/strategy-lab/SKILL.md).

1. **Describe a strategy.** "Buy S&P 500 stocks that gap down 3% or more with no news since the
   previous close, sell after 3 days." The bot lists the available building blocks, asks about
   anything unclear (for example how many stocks to hold at once, with a default), then repeats the
   rules in plain English and the strategy id. It does not run anything yet.
2. **Confirm and backtest.** "Go ahead." Only now does it run the backtest. It reports the test
   window, return, drawdown, Sharpe ratio and win rate at 5 bps cost per side, the 0 and 20 bps
   results, the trade count (flagging too few trades), versions tried, SPY and equal-weight
   benchmarks, and the equity chart. Every number is copied from Python output. A losing result is
   reported as plainly as a winning one.
3. **Try it on made-up data.** "Test this on simulated data." Every command for that request runs
   in simulated mode, and the reply starts with SIMULATED DATA.
4. **Approve with a code.** "I approve strat_..." (use the id from step 1). The bot sends a 6-digit
   code straight to your Telegram. The AI never sees it. Type the code in your next message. That
   runs the sealed 2024-onward test once and starts paper trading. It cannot be repeated.
5. **Ask about the paper portfolio.** "How is the paper portfolio doing?" or "Why did you buy AMD?"
   The bot reports status, or which rules fired for that trade and their values.
6. **Things it declines.** "Trade this with real money," "short the stocks that gap up," "test it
   on 2024," "use Reddit buzz," or "run ls for me." It says why and, where it can, offers the
   closest supported version.

Ideas the rules format cannot express (OR logic, trailing stops, intraday timing) get the closest
supported alternative, and you decide whether to use it.

## Troubleshooting

- **`./setup.sh` says permission denied.** Run `chmod +x setup.sh demo.sh`, then try again.
- **`uv: command not found` after setup.** Open a new terminal, or run
  `export PATH="$HOME/.local/bin:$PATH"`.
- **Results say SIMULATED DATA and you wanted real data.** No `EODHD_API_KEY` is set in `.env`, or
  `EODHD_MODE=fake` is set. Add the key and clear `EODHD_MODE`.
- **A live command fails with a missing-data message.** Download first:
  `uv run python scripts/download_data.py`. Run `uv run python -m bot.verify_live` to check the key
  and plan.
- **The download stops before it starts.** The estimate is above the 30,000 call cap. This is
  expected for `--universe full`. Use the sample, or change the cap on purpose.
- **The bot does not answer on Telegram.** Check `openclaw gateway status` and
  `openclaw channels status --probe`. Confirm `TELEGRAM_ALLOWED_USER_IDS` holds your numeric ID, not
  your username. Re-render the config after any change.
- **`openclaw doctor` reports an unknown key.** OpenClaw refuses unknown settings. Do not edit
  `openclaw/rendered/` by hand. Change the templates and render again.
- **The approval code never arrives.** `APPROVAL_CODE_SECRET` must be set in `.env` for live mode,
  and `TELEGRAM_BOT_TOKEN` must be valid.

## Project layout

```
bot/            The Python package: data, strategy blocks, backtest engine, paper trading, CLI
  data/         EODHD client, simulated EODHD, membership, prices, news, holdout guard
  strategy/     Building blocks and the strategy definition schema
  backtest/     Engine, costs, metrics, charts
  paper/        Paper trading step
scripts/        Download, call estimate, data quality report, config renderer, results summary
openclaw/       Agent skill, config templates, exec allowlist, cron jobs
evals/          Agent test cases and scoring
tests/          Test suite
notebooks/      Walkthrough notebook
docs/           Design notes
data/           The frozen 200-ticker sample list (cached data is created here, not committed)
setup.sh        One-command setup
demo.sh         One-command simulated demo
```

## Testing

- Quick suite, no network: `uv run pytest -q -m "not integration and not slow"`.
- Full suite, no network: `uv run pytest -q`. It takes several minutes.
- Real EODHD calls (needs a paid key): `uv run pytest -m integration`.
- Lint and types: `uv run ruff check . && uv run ruff format .` and `uv run mypy`.
- Results go to `outputs/simulated/` or `outputs/live/`. Both are ignored by git.
