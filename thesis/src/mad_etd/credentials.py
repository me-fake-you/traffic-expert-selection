from __future__ import annotations

import os
from pathlib import Path


def resolve_nvidia_api_key(
    *,
    explicit_key: str | None = None,
    key_file: str | Path | None = None,
) -> str | None:
    """Resolve a secret without logging or persisting its value."""

    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    if key_file is not None:
        source = Path(key_file).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"NVIDIA API key file not found: {source}")
        value = source.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"NVIDIA API key file is empty: {source}")
        if "\n" in value or "\r" in value:
            raise ValueError("NVIDIA API key file must contain exactly one line")
        return value
    value = os.getenv("NVIDIA_API_KEY", "").strip()
    return value or None
