"""Smoke tests that the package imports and the settings are sane."""

from pathlib import Path

import pytest

from bot import config
from bot.cli import main


def test_holdout_starts_after_backtest_ends() -> None:
    assert config.HOLDOUT_START > config.BACKTEST_END


def test_default_cost_is_in_sensitivity_set() -> None:
    assert config.DEFAULT_COST_BPS_PER_SIDE in config.COST_SENSITIVITY_BPS


def test_cli_runs_with_no_command() -> None:
    assert main([]) == 0


def test_env_file_key_beats_shell_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key exported in the shell must not replace the project's .env key."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\nEODHD_API_KEY=\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.setenv("EODHD_API_KEY", "shell-only")
    assert config.secret("OPENAI_API_KEY") == "from-file"
    # An empty value in .env falls back to the environment.
    assert config.secret("EODHD_API_KEY") == "shell-only"


def test_telegram_can_be_switched_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """TELEGRAM_DISABLED=1 sends nothing even when a token and user IDs are set."""
    from bot import telegram_send

    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_ALLOWED_USER_IDS=1\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("TELEGRAM_DISABLED", "1")
    monkeypatch.setattr(telegram_send.requests, "post", lambda *a, **k: 1 / 0)
    assert telegram_send.send_text("hello").status == telegram_send.DISABLED
    assert telegram_send.send_photo(tmp_path / "x.png", "c").status == telegram_send.DISABLED
