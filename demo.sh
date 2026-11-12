#!/usr/bin/env bash
# Try the bot on SIMULATED data. No API keys, no cost, nothing sent anywhere.
# Run from the project folder:   ./demo.sh
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Please run ./setup.sh first."
  exit 1
fi

# Simulated data only, and never Telegram, whatever is in .env.
export EODHD_MODE=fake
# The simulated market has its own made-up tickers, so use all of them.
export EODHD_UNIVERSE=full
export TELEGRAM_DISABLED=1

say "1/3  Creating the simulated market (takes a few seconds)"
uv run python scripts/download_data.py --universe full >/dev/null
echo "Done. Everything below is SIMULATED DATA, not real markets."

STRATEGY='{"name":"Gap down, no news","description_plain_english":"Buy S&P 500 stocks that gap down 3% or more with no news since the previous close, sell after 3 trading days, hold at most 10.","entry":[{"block":"gap","direction":"down","min_gap_pct":3},{"block":"no_news","lookback_sessions":1}],"exit":{"holding_days":3},"sizing":{"method":"equal_weight","max_positions":10}}'

say "2/3  Checking the strategy (this is what the Telegram agent sends)"
uv run openclaw-bot validate "$STRATEGY" --data simulated 2>/dev/null | uv run python -c '
import json, sys
out = json.load(sys.stdin)
if not out.get("ok"):
    sys.exit("Validation failed: " + "; ".join(out.get("errors", [])))
print(out["rules_text"])
print("\nStrategy id:", out["strategy_id"])
open(".demo_strategy_id", "w").write(out["strategy_id"])
'

say "3/3  Running the backtest (about a minute)"
uv run openclaw-bot backtest "$(cat .demo_strategy_id)" --data simulated 2>/dev/null | uv run python -c '
import json, sys
out = json.load(sys.stdin)
if not out.get("ok"):
    sys.exit("Backtest failed: " + "; ".join(out.get("errors", [])))
print("*** " + out["banner"] + ": made-up prices, not real markets ***")
for w in out["windows"]:
    print("\nPeriod: %s to %s" % (w["start"], w["end"]))
    print("Trades closed: %d" % w["closed_trades"])
    print("\n  Costs per side   Total return   Worst drop   Win rate")
    for bps, m in w["by_cost_bps"].items():
        print("  %5s bps       %9.2f%%   %9.2f%%   %7.2f%%" % (
            bps, m["total_return_pct"], m["max_drawdown_pct"], m["win_rate_pct"]))
    print("\n  Buy and hold for comparison:")
    for key, b in w["benchmarks"].items():
        label = out["benchmark_labels"].get(key, key)
        print("  %-30s total return %.2f%%" % (label, b["total_return_pct"]))
print("\nChart saved to:", out["chart_path"])
'
rm -f .demo_strategy_id

say "That was a simulated run."
echo "The numbers above come from made-up prices. They show the tool works;"
echo "they say nothing about whether the strategy makes money."
