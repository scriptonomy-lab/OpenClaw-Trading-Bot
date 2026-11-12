# Scheduled jobs

Two OpenClaw automations run the paper portfolio after each US close. Both
are **command jobs**: the Gateway runs `openclaw-bot` directly and no model is
called.

The commands below are a template. `scripts/render_openclaw_config.py` writes
a filled copy to `openclaw/rendered/cron.md`. Run the commands from that copy.

Sources:
https://docs.openclaw.ai/automation/cron-jobs/payloads (command payloads),
https://docs.openclaw.ai/cli/cron (flags),
https://docs.openclaw.ai/automation/cron-jobs/delivery (Telegram delivery),
https://docs.openclaw.ai/automation/cron-jobs/managing-jobs (host execution).

## Why command jobs

- **No model touches the numbers.** A command job captures the process
  output and posts it as is. The daily summary is printed by Python, so no
  LLM can round, reword or invent a figure on its way to Telegram (rule 2).
- **No agent decides when to trade.** The paper portfolio steps forward on a
  fixed schedule, not because a chat message asked it to. This is also why
  `paper-update` and `daily-summary` are not in the agent's exec allowlist.
- **Exact argv, no shell.** `--command-argv` stores the argument list as
  given, with no shell parsing. (`--command` would wrap it in `sh -lc`.)

## 1. Paper update after the close

Weekdays at 18:30 New York time. That leaves time for EODHD to publish the
day's bars after the 16:00 close. On market holidays the CLI finds no new
bar and makes no change. Output is kept out of chat (`--no-deliver`); a
non-zero exit is recorded as a failed run.

```bash
openclaw automations create "30 18 * * 1-5" \
  --name "paper-update" \
  --tz "America/New_York" \
  --exact \
  --command-argv '["<ABSOLUTE_PATH_TO_REPO>/.venv/bin/openclaw-bot","paper-update"]' \
  --command-cwd "<ABSOLUTE_PATH_TO_REPO>" \
  --timeout-seconds 1800 \
  --no-deliver
```

## 2. Daily summary to Telegram

Weekdays at 19:00 New York time, after the update. The plain-text output of
`daily-summary` is sent to the owner's Telegram DM (for a private chat, the
chat ID is the user ID). If the summary prints only `NO_REPLY`, OpenClaw
sends nothing, which the CLI can use on days with no update.

```bash
openclaw automations create "0 19 * * 1-5" \
  --name "daily-summary" \
  --tz "America/New_York" \
  --exact \
  --command-argv '["<ABSOLUTE_PATH_TO_REPO>/.venv/bin/openclaw-bot","daily-summary"]' \
  --command-cwd "<ABSOLUTE_PATH_TO_REPO>" \
  --timeout-seconds 300 \
  --announce \
  --channel telegram \
  --to "<TELEGRAM_USER_ID>"
```

`--exact` turns off schedule staggering, so the summary never runs before
the update. In fake data mode the summary text itself starts with
"SIMULATED DATA", because it comes straight from the CLI.

## Check and manage

```bash
openclaw automations list --all
openclaw automations run JOB_ID --wait
```

Run each job once by hand with `run --wait` after creating it, and check the
Telegram message.

## Things that must stay true

- **Leave `OPENCLAW_DISABLE_CRON_HOST_EXECUTION` unset.** When it is set to
  `1`, the Gateway refuses command payloads, and both jobs stop running.
  This design depends on command jobs.
- **Command jobs are not covered by the exec allowlist.** They run inside
  the Gateway process with its permissions. Only an operator with
  `operator.admin` can create them, and the agent cannot, because the
  `cron`, `automations` and `gateway` tools are denied in
  `openclaw/openclaw.json5`. Keep this list to these two jobs.
- **Same code path as the backtest.** `paper-update` uses the backtest
  engine, so the paper ledger can be checked against a backtest.
