"""Send a chart to the allowed Telegram users with the Bot API (D21).

Python sends the image, so the agent never handles files. The bot token is
never logged and never put in a result. If the token or user IDs are missing,
nothing is sent and the result says so; the calling command still succeeds.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from bot.config import SIMULATED_LABEL, data_mode, load_secrets

API_URL = "https://api.telegram.org/bot{token}/sendPhoto"
MESSAGE_URL = "https://api.telegram.org/bot{token}/sendMessage"
CAPTION_LIMIT = 1024
TIMEOUT_SECONDS = 30
NOT_CONFIGURED = "not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_IDS missing"
DISABLED = "not sent: TELEGRAM_DISABLED=1"


def telegram_disabled() -> bool:
    """True when TELEGRAM_DISABLED=1 is set, for demos and tests.

    Blanking the token in the environment is not enough, because a key in
    the project .env wins over the environment (bot.config.secret).
    """
    return os.getenv("TELEGRAM_DISABLED", "").strip() == "1"


@dataclass(frozen=True)
class SendResult:
    """What happened. ``sent`` counts users who got the photo."""

    sent: int
    attempted: int
    status: str
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        """Plain dict for the command's JSON."""
        return asdict(self)


def make_caption(text: str) -> str:
    """The caption, starting with SIMULATED DATA in fake mode, cut to Telegram's limit."""
    caption = f"{SIMULATED_LABEL}: {text}" if data_mode() == "fake" else text
    return caption[:CAPTION_LIMIT]


def send_photo(image_path: Path, caption: str) -> SendResult:
    """Send the image to every allowed user ID. Never raises on missing settings or network."""
    if telegram_disabled():
        return SendResult(0, 0, DISABLED, [])
    secrets = load_secrets()
    if not secrets.telegram_bot_token or not secrets.telegram_allowed_user_ids:
        return SendResult(
            0, 0, "not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_IDS missing", []
        )
    if not image_path.exists():
        return SendResult(0, 0, "not sent: chart file not found", [])
    text = make_caption(caption)
    errors: list[str] = []
    sent = 0
    for user_id in secrets.telegram_allowed_user_ids:
        problem = _send_one(secrets.telegram_bot_token, user_id, image_path, text)
        if problem is None:
            sent += 1
        else:
            errors.append(f"user {user_id}: {problem}")
    attempted = len(secrets.telegram_allowed_user_ids)
    return SendResult(sent, attempted, "sent" if sent == attempted else "partly sent", errors)


def send_text(text: str) -> SendResult:
    """Send a text message to every allowed user ID. Never raises on missing settings."""
    if telegram_disabled():
        return SendResult(0, 0, DISABLED, [])
    secrets = load_secrets()
    if not secrets.telegram_bot_token or not secrets.telegram_allowed_user_ids:
        return SendResult(
            0, 0, "not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_IDS missing", []
        )
    message = f"{SIMULATED_LABEL}: {text}" if data_mode() == "fake" else text
    errors: list[str] = []
    sent = 0
    for user_id in secrets.telegram_allowed_user_ids:
        problem = _post(
            MESSAGE_URL.format(token=secrets.telegram_bot_token),
            {"chat_id": user_id, "text": message},
            None,
        )
        if problem is None:
            sent += 1
        else:
            errors.append(f"user {user_id}: {problem}")
    attempted = len(secrets.telegram_allowed_user_ids)
    return SendResult(sent, attempted, "sent" if sent == attempted else "partly sent", errors)


def _post(url: str, data: dict[str, Any], files: dict[str, Any] | None) -> str | None:
    """Post to the Bot API. Returns a short problem text, or None. No token in the text."""
    try:
        response = requests.post(url, data=data, files=files, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as error:
        return f"network error ({type(error).__name__})"
    if response.status_code != 200:
        return f"Telegram answered HTTP {response.status_code}"
    return None


def _send_one(token: str, user_id: int, image_path: Path, caption: str) -> str | None:
    """Post one photo. Returns a short problem text, or None on success."""
    with image_path.open("rb") as image:
        return _post(
            API_URL.format(token=token),
            {"chat_id": user_id, "caption": caption},
            {"photo": image},
        )
