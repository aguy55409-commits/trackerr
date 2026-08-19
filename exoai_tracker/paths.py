from __future__ import annotations

import sys
from pathlib import Path


def app_dir() -> Path:
    """Directory containing config, models, and writable data."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return app_dir() / candidate
