"""Helpers for initialising the Qlib runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import qlib
from qlib.config import REG_CN


def get_default_provider_uri() -> str:
    """Return the default CN data directory used by Qlib."""

    return str(Path.home() / ".qlib" / "qlib_data" / "cn_data")


def init_qlib_env(provider_uri: Optional[str]) -> None:
    """Initialise Qlib using the configured provider URI."""

    qlib.init(provider_uri=provider_uri or get_default_provider_uri(), region=REG_CN, exp_manager=None)
