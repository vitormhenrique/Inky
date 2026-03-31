"""Tests for image selection logic."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import Settings
from src.pipeline.selector import select_image


@pytest.fixture()
def temp_cache(tmp_path: Path) -> Path:
    """Create a cache structure with test images."""
    for sub in ("raw", "parsed", "styled", "display", "archive", "logs"):
        (tmp_path / sub).mkdir()

    # Create fake images
    for name in ("photo_a.jpg", "photo_b.jpg"):
        (tmp_path / "raw" / name).write_bytes(b"fake-image-data")

    (tmp_path / "parsed" / "portrait_01.png").write_bytes(b"fake-parsed")
    return tmp_path


@pytest.fixture()
def settings(temp_cache: Path) -> Settings:
    return Settings(local_cache_dir=temp_cache)


class TestSelectImage:
    def test_explicit_path(self, settings: Settings, temp_cache: Path):
        explicit = temp_cache / "raw" / "photo_a.jpg"
        result = select_image(settings, explicit_path=explicit)
        assert result == explicit

    def test_explicit_path_not_found(self, settings: Settings, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            select_image(settings, explicit_path=tmp_path / "nonexistent.jpg")

    def test_latest_parsed(self, settings: Settings, temp_cache: Path):
        # Touch a file to make it newest
        target = temp_cache / "parsed" / "portrait_01.png"
        target.write_bytes(b"updated")
        result = select_image(settings, mode_override="latest_parsed")
        assert result.name == "portrait_01.png"

    def test_random_raw(self, settings: Settings, temp_cache: Path):
        result = select_image(settings, mode_override="random_raw")
        assert result.parent.name == "raw"
        assert result.suffix == ".jpg"

    def test_random_any(self, settings: Settings, temp_cache: Path):
        result = select_image(settings, mode_override="random_any")
        assert result.suffix in (".jpg", ".png")

    def test_random_raw_empty(self, tmp_path: Path):
        for sub in ("raw", "parsed"):
            (tmp_path / sub).mkdir()
        s = Settings(local_cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            select_image(s, mode_override="random_raw")

    def test_latest_parsed_falls_back_to_random_raw(self, settings: Settings, temp_cache: Path):
        """When parsed is empty, latest_parsed falls back to random_raw."""
        # Remove parsed files
        for f in (temp_cache / "parsed").iterdir():
            f.unlink()
        result = select_image(settings, mode_override="latest_parsed")
        assert result.parent.name == "raw"

    def test_unknown_mode_raises(self, settings: Settings):
        with pytest.raises(ValueError, match="Unknown selection mode"):
            select_image(settings, mode_override="bogus")
