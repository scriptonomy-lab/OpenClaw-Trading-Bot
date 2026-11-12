"""Project settings: API keys from the environment, plus fixed dates and costs."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Settings such as EODHD_MODE set on the command line still win over .env.
load_dotenv(ENV_FILE)
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# The test window. Everything from HOLDOUT_START on is sealed until a
# strategy is approved, so it cannot influence any design choice.
BACKTEST_START = date(2016, 1, 1)
BACKTEST_END = date(2023, 12, 31)
HOLDOUT_START = date(2024, 1, 1)

# Membership history starts a year early so 2016 signals can look back.
MEMBERSHIP_START = date(2015, 1, 1)

# One cost model for every strategy, so results are comparable.
DEFAULT_COST_BPS_PER_SIDE = 5.0
COST_SENSITIVITY_BPS = (0.0, 5.0, 20.0)

# Below this many trades a result is reported as too few to judge.
MIN_TRADES_TO_JUDGE = 30


@dataclass(frozen=True)
class Secrets:
    """API keys and allowed Telegram user IDs read from the environment."""

    eodhd_api_key: str
    telegram_bot_token: str
    telegram_allowed_user_ids: tuple[int, ...]


def secret(name: str) -> str:
    """Return a key, preferring a non-empty value in the project's .env file.

    A key exported in the user's shell (for another project) must not
    silently replace the key this project was given.
    """
    from_file = (dotenv_values(ENV_FILE).get(name) or "").strip()
    return from_file or os.getenv(name, "").strip()


def load_secrets() -> Secrets:
    """Read keys, .env first, then the environment. Missing values become empty strings."""
    raw_ids = secret("TELEGRAM_ALLOWED_USER_IDS")
    user_ids = tuple(int(part) for part in raw_ids.split(",") if part.strip())
    return Secrets(
        eodhd_api_key=secret("EODHD_API_KEY"),
        telegram_bot_token=secret("TELEGRAM_BOT_TOKEN"),
        telegram_allowed_user_ids=user_ids,
    )


SIMULATED_LABEL = "SIMULATED DATA"


def data_mode(api_token: str | None = None) -> Literal["fake", "live"]:
    """Say whether EODHD calls go to the fake transport or the real API.

    EODHD_MODE=fake forces fake and EODHD_MODE=live forces live. When it is
    unset, the mode is fake only if there is no EODHD_API_KEY and no token was
    passed in explicitly.
    """
    setting = os.getenv("EODHD_MODE", "").strip().lower()
    if setting not in ("", "fake", "live"):
        raise ValueError(f"EODHD_MODE must be 'fake' or 'live', got {setting!r}.")
    if setting:
        return "fake" if setting == "fake" else "live"
    has_key = bool(api_token or os.getenv("EODHD_API_KEY", ""))
    return "live" if has_key else "fake"


# The --data flag on the agent-facing commands (D34) and the EODHD_MODE it sets.
DATA_FLAG_TO_MODE: dict[str, Literal["fake", "live"]] = {"simulated": "fake", "live": "live"}


def apply_data_override(flag_value: str | None) -> None:
    """Set the data mode for this process from ``--data simulated|live`` (D34).

    None leaves EODHD_MODE as it is. Fake and live use separate output folders
    and databases, so one command in the other mode never mixes their records.
    """
    if flag_value is None:
        return
    if flag_value not in DATA_FLAG_TO_MODE:
        raise ValueError(f"--data must be 'simulated' or 'live', got {flag_value!r}.")
    os.environ["EODHD_MODE"] = DATA_FLAG_TO_MODE[flag_value]


def output_dir() -> Path:
    """Folder for generated reports and charts. Fake and live never share one."""
    folder = "simulated" if data_mode() == "fake" else "live"
    return PROJECT_ROOT / "outputs" / folder


def report_title(title: str) -> str:
    """Prefix the title with the simulated label when the data is fake."""
    return f"{SIMULATED_LABEL}: {title}" if data_mode() == "fake" else title
