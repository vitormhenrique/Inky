"""Tests for diffusion helpers."""

from __future__ import annotations

from PIL import Image

from src.config import Settings
from src.models.reference_analysis import ReferenceStyleAnalysis
from src.pipeline.diffusion import (
    _condition_diffusion_input,
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


REFERENCE_ANALYSIS = ReferenceStyleAnalysis(
    dominant_colors=("blue", "gold"),
    palette_description="saturated cool blue palette with gold accents",
    brush_description="sweeping curved brush strokes with flowing paint movement",
    mood_description="luminous high-contrast paint handling",
    prompt_fragments=(
        "saturated cool blue palette with gold accents",
        "sweeping curved brush strokes with flowing paint movement",
        "luminous high-contrast paint handling",
    ),
    negative_fragments=("washed-out color", "tiny brush strokes", "flat lighting"),
    palette_mix=0.68,
    saturation_boost=1.18,
    contrast_boost=1.04,
    blur_radius=0.85,
    style_strength=0.86,
    broad_stroke_score=0.79,
)


def test_condition_diffusion_input_uses_reference_analysis_to_push_palette_cooler():
    content = Image.new("RGB", (32, 32), (170, 140, 110))
    reference = Image.new("RGB", (32, 32), (40, 80, 220))

    conditioned = _condition_diffusion_input(
        content,
        reference_image=reference,
        reference_analysis=REFERENCE_ANALYSIS,
    )

    red, green, blue = conditioned.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    assert blue > red
    assert blue > green
