"""Tests for NST helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.pipeline.nst import find_style_reference


@pytest.mark.parametrize(
    ("target_size", "expected_name"),
    [
        ((600, 900), "portrait.jpg"),
        ((900, 600), "landscape.jpg"),
    ],
)
def test_find_style_reference_prefers_closest_aspect_ratio(
    tmp_path: Path,
    target_size: tuple[int, int],
    expected_name: str,
):
    style_dir = tmp_path / "cubism"
    style_dir.mkdir()

    Image.new("RGB", (400, 800), "white").save(style_dir / "portrait.jpg")
    Image.new("RGB", (800, 400), "white").save(style_dir / "landscape.jpg")
    Image.new("RGB", (600, 600), "white").save(style_dir / "square.jpg")

    chosen = find_style_reference(tmp_path, "cubism", target_size=target_size)

    assert chosen.name == expected_name
