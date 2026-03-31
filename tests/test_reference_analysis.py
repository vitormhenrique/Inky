"""Tests for learned reference style analysis."""

from __future__ import annotations

from PIL import Image, ImageDraw

from src.models.reference_analysis import analyze_reference_style


def test_analyze_reference_style_extracts_palette_and_brush_descriptors(monkeypatch):
    monkeypatch.setattr(
        "src.models.reference_analysis._estimate_learned_texture_strength",
        lambda *args, **kwargs: 0.8,
    )

    img = Image.new("RGB", (160, 160), (32, 78, 180))
    draw = ImageDraw.Draw(img)
    for offset in range(0, 120, 24):
        draw.arc((8, offset, 152, offset + 40), 10, 320, fill=(232, 205, 72), width=8)

    analysis = analyze_reference_style(img)

    assert "blue" in analysis.palette_description
    assert analysis.palette_mix > 0.35
    assert analysis.style_strength > 0.45
    assert analysis.prompt_fragments
