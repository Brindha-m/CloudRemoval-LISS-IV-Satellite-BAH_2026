"""Credential and path configuration.

Credentials are read from environment variables, optionally loaded from a
local `.env` file (which must never be committed to version control).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "liss4"
CATALOG_DIR = DATA_DIR / "catalog"


def _load_env_file(path: Path) -> None:
    """Minimal .env loader so we do not require python-dotenv."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(ENV_PATH)


@dataclass(frozen=True)
class BhoonidhiCredentials:
    user_id: str
    password: str

    @property
    def is_complete(self) -> bool:
        return bool(self.user_id and self.password)


def get_credentials() -> BhoonidhiCredentials:
    return BhoonidhiCredentials(
        user_id=os.environ.get("BHOONIDHI_USER_ID", ""),
        password=os.environ.get("BHOONIDHI_PASSWORD", ""),
    )


def ensure_data_dirs() -> None:
    for directory in (RAW_DIR, CATALOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
