#!/usr/bin/env bash
# One-command setup for the OpenClaw trading bot.
# Run from the project folder:   ./setup.sh
# Safe to run more than once.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

say "1/4  Checking for uv (the Python installer this project uses)"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing it now (from astral.sh)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer puts uv in ~/.local/bin; make it usable in this run.
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "2/4  Installing Python and the project's packages (first run takes a minute)"
uv sync

say "3/4  Creating your settings file (.env)"
if [ -f .env ]; then
  echo ".env already exists. Leaving it as it is."
else
  cp .env.example .env
  chmod 600 .env
  echo "Created .env from .env.example."
  echo "It is fine to leave every key empty for now: the bot then runs on"
  echo "simulated data, so you can try everything without paying for anything."
fi

say "4/4  Running the quick tests to make sure everything works"
uv run pytest -q -m "not integration and not slow"

say "Setup finished."
cat <<'EOF'
Next steps:
  - Try it on simulated data (no keys needed):   ./demo.sh
  - Use real market data: put your EODHD key in .env (EODHD_API_KEY=...),
    then run:   uv run python -m bot.verify_live
  - Talk to it on Telegram: see "Path 3" in README.md.
EOF
