"""Tests for main pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import Settings
from src.main import run_pipeline


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
