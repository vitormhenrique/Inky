"""File-system helpers for the local cache / working directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from src.config import Settings
from src.logging_utils import get_logger

log = get_logger("files")

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def ensure_dirs(settings: Settings) -> None:
    """Create all required local directories if they don't exist."""
    for attr in (
        "local_cache_dir",
        "local_output_dir",
        "local_display_dir",
        "local_archive_dir",
        "local_styles_dir",
        "local_metadata_dir",
    ):
        p = settings.resolve_path(getattr(settings, attr))
        p.mkdir(parents=True, exist_ok=True)

    # Ensure subdirs inside cache mirror the GDrive layout
    for sub in ("raw", "parsed", "styled", "display", "archive", "logs"):
        (settings.resolve_path(settings.local_cache_dir) / sub).mkdir(
            parents=True, exist_ok=True
        )


def list_images(directory: Path) -> list[Path]:
    """Return sorted list of image files in *directory*."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


def newest_file(directory: Path) -> Path | None:
    """Return the most recently modified image in *directory*, or ``None``."""
    images = list_images(directory)
    if not images:
        return None
    return max(images, key=lambda p: p.stat().st_mtime)


def archive_file(src: Path, archive_dir: Path) -> Path:
    """Move *src* into *archive_dir*, preserving name. Returns new path."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / src.name
    # Avoid overwriting — append counter
    counter = 1
    while dest.exists():
        dest = archive_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(dest))
    log.info("Archived %s → %s", src.name, dest)
    return dest


def safe_filename(name: str) -> str:
    """Sanitise a string for use as a filename."""
    keep = {" ", ".", "-", "_"}
    return "".join(c if (c.isalnum() or c in keep) else "_" for c in name).strip()
