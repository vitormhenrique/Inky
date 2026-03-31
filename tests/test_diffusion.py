"""Tests for diffusion helpers."""

from __future__ import annotations

from PIL import Image

from src.config import Settings
from src.pipeline.diffusion import (
    _derive_reference_hint,
    _derive_source_hint,
    _prepare_diffusion_input,
)


def test_prepare_diffusion_input_resizes_cpu_image_to_safe_multiple():
    settings = Settings(device_preference="cpu")
    img = Image.new("RGB", (687, 1023), "white")

    prepared = _prepare_diffusion_input(img, settings, device="cpu")

    assert prepared.size == (512, 768)
    assert prepared.width % 8 == 0
    assert prepared.height % 8 == 0


def test_prepare_diffusion_input_keeps_valid_small_image():
    settings = Settings(device_preference="cpu")
    img = Image.new("RGB", (512, 768), "white")

    prepared = _prepare_diffusion_input(img, settings, device="cpu")

    assert prepared.size == (512, 768)


def test_prepare_diffusion_input_uses_smaller_pi_limit():
    class PiSettings(Settings):
        @property
        def is_raspberry_pi(self) -> bool:
            return True

    settings = PiSettings(device_preference="cpu")
    img = Image.new("RGB", (687, 1023), "white")

    prepared = _prepare_diffusion_input(img, settings, device="cpu")

    assert prepared.size == (336, 512)


def test_derive_source_hint_uses_meaningful_names():
    assert _derive_source_hint("Mona_Lisa") == "mona lisa"
    assert _derive_source_hint("IMG_1234") is None


def test_derive_reference_hint_uses_reference_filename():
    assert (
        _derive_reference_hint("post_impressionism/vangogh_starry_night.jpg")
        == "vangogh starry night"
    )
