"""Metadata tracking — records which image was displayed and when."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logging_utils import get_logger

log = get_logger("metadata")


def _metadata_path(metadata_dir: Path) -> Path:
    return metadata_dir / "display_history.json"


def load_history(metadata_dir: Path) -> list[dict[str, Any]]:
    """Load the display history JSON, returning an empty list if absent."""
    p = _metadata_path(metadata_dir)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def record_display(
    metadata_dir: Path,
    *,
    source_image: str,
    style_name: str,
    algorithm: str,
    output_path: str,
    display_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an entry to the display-history JSON and return it."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(metadata_dir)

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_image": source_image,
        "style_name": style_name,
        "algorithm": algorithm,
        "output_path": output_path,
        "display_path": display_path,
    }
    if extra:
        entry["extra"] = extra

    history.append(entry)
    with open(_metadata_path(metadata_dir), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    log.info("Recorded display entry #%d: %s → %s", len(history), source_image, style_name)
    return entry
