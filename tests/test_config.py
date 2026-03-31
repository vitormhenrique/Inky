"""Tests for configuration system."""

from __future__ import annotations

import os

import pytest

from src.config import Settings, get_settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.default_algorithm == "nst"
        assert s.display_width == 1600
        assert s.display_height == 1200
        assert s.fallback_to_nst is True

    def test_display_resolution(self):
        s = Settings()
        assert s.display_resolution == (1600, 1200)

    def test_csv_parsing(self):
        s = Settings(allowed_algorithms="nst,diffusion")
        assert s.allowed_algorithms == ["nst", "diffusion"]

    def test_csv_parsing_list(self):
        s = Settings(allowed_algorithms=["nst"])
        assert s.allowed_algorithms == ["nst"]

    def test_resolve_relative_path(self):
        s = Settings()
        resolved = s.resolve_path(s.local_cache_dir)
        assert resolved.is_absolute()

    def test_detect_device_cpu(self):
        s = Settings(device_preference="cpu")
        assert s.detect_device() == "cpu"

    def test_detect_device_auto_fallback(self):
        """Auto mode should return at least 'cpu'."""
        s = Settings(device_preference="auto")
        device = s.detect_device()
        assert device in ("cpu", "mps", "cuda")

    def test_get_settings_factory(self):
        s = get_settings()
        assert isinstance(s, Settings)
