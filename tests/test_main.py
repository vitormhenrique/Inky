"""Tests for main pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.config import Settings
from src.main import _apply_nst_variation_weights, run_pipeline


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        local_cache_dir=tmp_path / "cache",
        local_output_dir=tmp_path / "output",
        local_display_dir=tmp_path / "display",
        local_archive_dir=tmp_path / "archive",
        local_styles_dir=Path("data/styles"),
        local_metadata_dir=tmp_path / "metadata",
        log_file=str(tmp_path / "logs" / "test.log"),
    )


def test_run_pipeline_passes_reference_path_to_nst(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    source_path = tmp_path / "Mona_Lisa.jpg"
    Image.new("RGB", (300, 450), "white").save(source_path)

    stylised = Image.new("RGB", (300, 450), "gray")
    display_path = tmp_path / "display" / "result.png"
    hires_path = tmp_path / "output" / "result.png"

    captured: dict[str, object] = {}

    def fake_run_nst_with_style(
        content_image,
        style,
        settings,
        style_intensity=None,
        reference_path=None,
        *,
        variation_index=None,
        variation_count=None,
    ):
        captured["reference_path"] = reference_path
        return stylised

    monkeypatch.setattr("src.main.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.ensure_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.select_image", lambda *args, **kwargs: source_path)
    monkeypatch.setattr(
        "src.main.preprocess",
        lambda *args, **kwargs: Image.new("RGB", (300, 450), "white"),
    )
    monkeypatch.setattr("src.main._run_nst_with_style", fake_run_nst_with_style)
    monkeypatch.setattr("src.main.prepare_display_image", lambda *args, **kwargs: stylised)
    monkeypatch.setattr(
        "src.main.save_outputs",
        lambda *args, **kwargs: (hires_path, display_path),
    )
    monkeypatch.setattr("src.main.record_display", lambda *args, **kwargs: None)

    result = run_pipeline(
        settings,
        input_path=str(source_path),
        style_name="cubism",
        algorithm="nst",
        skip_sync=True,
        skip_display=True,
        skip_upload=True,
        reference_path="cubism/gris_portrait_picasso.jpg",
    )

    assert result == display_path
    assert captured["reference_path"] == "cubism/gris_portrait_picasso.jpg"


def test_diffusion_fallback_keeps_reference_path(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    source_path = tmp_path / "Mona_Lisa.jpg"
    Image.new("RGB", (300, 450), "white").save(source_path)

    stylised = Image.new("RGB", (300, 450), "gray")
    display_path = tmp_path / "display" / "result.png"
    hires_path = tmp_path / "output" / "result.png"

    captured: dict[str, object] = {}

    def fake_run_nst_with_style(
        content_image,
        style,
        settings,
        style_intensity=None,
        reference_path=None,
        *,
        variation_index=None,
        variation_count=None,
    ):
        captured["reference_path"] = reference_path
        return stylised

    monkeypatch.setattr("src.main.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.ensure_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.select_image", lambda *args, **kwargs: source_path)
    monkeypatch.setattr(
        "src.main.preprocess",
        lambda *args, **kwargs: Image.new("RGB", (300, 450), "white"),
    )
    monkeypatch.setattr("src.main.should_use_diffusion", lambda *args, **kwargs: (False, "test"))
    monkeypatch.setattr("src.main._run_nst_with_style", fake_run_nst_with_style)
    monkeypatch.setattr("src.main.prepare_display_image", lambda *args, **kwargs: stylised)
    monkeypatch.setattr(
        "src.main.save_outputs",
        lambda *args, **kwargs: (hires_path, display_path),
    )
    monkeypatch.setattr("src.main.record_display", lambda *args, **kwargs: None)

    result = run_pipeline(
        settings,
        input_path=str(source_path),
        style_name="cubism",
        algorithm="diffusion",
        skip_sync=True,
        skip_display=True,
        skip_upload=True,
        reference_path="cubism/gris_portrait_picasso.jpg",
    )

    assert result == display_path
    assert captured["reference_path"] == "cubism/gris_portrait_picasso.jpg"


def test_run_pipeline_passes_reference_path_to_diffusion(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    source_path = tmp_path / "Mona_Lisa.jpg"
    Image.new("RGB", (300, 450), "white").save(source_path)

    stylised = Image.new("RGB", (300, 450), "gray")
    display_path = tmp_path / "display" / "result.png"
    hires_path = tmp_path / "output" / "result.png"

    captured: dict[str, object] = {}

    def fake_run_diffusion(
        content_image,
        style,
        settings,
        *,
        source_name=None,
        reference_path=None,
        strength=None,
        guidance_scale=None,
        num_inference_steps=None,
    ):
        captured["source_name"] = source_name
        captured["reference_path"] = reference_path
        return stylised

    monkeypatch.setattr("src.main.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.ensure_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.select_image", lambda *args, **kwargs: source_path)
    monkeypatch.setattr(
        "src.main.preprocess",
        lambda *args, **kwargs: Image.new("RGB", (300, 450), "white"),
    )
    monkeypatch.setattr("src.main.should_use_diffusion", lambda *args, **kwargs: (True, "ok"))
    monkeypatch.setattr("src.main.run_diffusion", fake_run_diffusion)
    monkeypatch.setattr("src.main.prepare_display_image", lambda *args, **kwargs: stylised)
    monkeypatch.setattr(
        "src.main.save_outputs",
        lambda *args, **kwargs: (hires_path, display_path),
    )
    monkeypatch.setattr("src.main.record_display", lambda *args, **kwargs: None)

    result = run_pipeline(
        settings,
        input_path=str(source_path),
        style_name="post_impressionism",
        algorithm="diffusion",
        skip_sync=True,
        skip_display=True,
        skip_upload=True,
        reference_path="post_impressionism/vangogh_starry_night.jpg",
    )

    assert result == display_path
    assert captured["source_name"] == "Mona_Lisa"
    assert captured["reference_path"] == "post_impressionism/vangogh_starry_night.jpg"


def test_run_pipeline_passes_batch_variation_to_nst(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    source_path = tmp_path / "Mona_Lisa.jpg"
    Image.new("RGB", (300, 450), "white").save(source_path)

    stylised = Image.new("RGB", (300, 450), "gray")
    display_path = tmp_path / "display" / "result.png"
    hires_path = tmp_path / "output" / "result.png"

    captured: dict[str, object] = {}

    def fake_run_nst_with_style(
        content_image,
        style,
        settings,
        style_intensity=None,
        reference_path=None,
        *,
        variation_index=None,
        variation_count=None,
    ):
        captured["variation_index"] = variation_index
        captured["variation_count"] = variation_count
        return stylised

    monkeypatch.setattr("src.main.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.ensure_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.main.select_image", lambda *args, **kwargs: source_path)
    monkeypatch.setattr(
        "src.main.preprocess",
        lambda *args, **kwargs: Image.new("RGB", (300, 450), "white"),
    )
    monkeypatch.setattr("src.main._run_nst_with_style", fake_run_nst_with_style)
    monkeypatch.setattr("src.main.prepare_display_image", lambda *args, **kwargs: stylised)
    monkeypatch.setattr(
        "src.main.save_outputs",
        lambda *args, **kwargs: (hires_path, display_path),
    )
    monkeypatch.setattr("src.main.record_display", lambda *args, **kwargs: None)

    result = run_pipeline(
        settings,
        input_path=str(source_path),
        style_name="rococo_portrait",
        algorithm="nst",
        skip_sync=True,
        skip_display=True,
        skip_upload=True,
        variation_index=3,
        variation_count=10,
    )

    assert result == display_path
    assert captured["variation_index"] == 3
    assert captured["variation_count"] == 10


def test_apply_nst_variation_weights_spreads_batch_slots():
    stronger_content, stronger_style = _apply_nst_variation_weights(
        1.0,
        1000.0,
        variation_index=1,
        variation_count=6,
    )
    softer_content, softer_style = _apply_nst_variation_weights(
        1.0,
        1000.0,
        variation_index=4,
        variation_count=6,
    )

    assert stronger_content == pytest.approx(0.9874)
    assert stronger_style == pytest.approx(1036.0)
    assert softer_content == pytest.approx(1.0378)
    assert softer_style == pytest.approx(892.0)

    unchanged = _apply_nst_variation_weights(1.0, 1000.0)
    assert unchanged == (1.0, 1000.0)
