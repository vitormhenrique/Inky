"""Tests for CLI batch commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from PIL import Image

from src.cli import cli
from src.config import Settings


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        local_cache_dir=tmp_path / "cache",
        local_output_dir=tmp_path / "output",
        local_display_dir=tmp_path / "display",
        local_archive_dir=tmp_path / "archive",
        local_styles_dir=tmp_path / "styles",
        local_metadata_dir=tmp_path / "metadata",
        log_file=str(tmp_path / "logs" / "test.log"),
    )


def test_sweep_by_name_runs_all_style_reference_jobs(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    raw_dir = settings.resolve_path(settings.local_cache_dir) / "raw"
    raw_dir.mkdir(parents=True)
    Image.new("RGB", (300, 450), "white").save(raw_dir / "lais_8.png")

    styles_dir = settings.resolve_path(settings.local_styles_dir)
    rococo_dir = styles_dir / "rococo_portrait"
    cubism_dir = styles_dir / "cubism"
    empty_dir = styles_dir / "empty_style"
    rococo_dir.mkdir(parents=True)
    cubism_dir.mkdir(parents=True)
    empty_dir.mkdir(parents=True)

    Image.new("RGB", (100, 100), "pink").save(rococo_dir / "ref_a.jpg")
    Image.new("RGB", (100, 100), "blue").save(rococo_dir / "ref_b.jpg")
    Image.new("RGB", (100, 100), "gray").save(cubism_dir / "ref_c.jpg")

    monkeypatch.setattr("src.cli.get_settings", lambda: settings)
    monkeypatch.setattr(
        "src.cli.list_styles",
        lambda: [
            SimpleNamespace(name="rococo_portrait", nst_reference_subdir="rococo_portrait"),
            SimpleNamespace(name="cubism", nst_reference_subdir="cubism"),
            SimpleNamespace(name="empty_style", nst_reference_subdir="empty_style"),
        ],
    )

    calls: list[dict[str, object]] = []

    def fake_run_pipeline(*args, **kwargs):
        calls.append(kwargs)
        reference_name = Path(str(kwargs["reference_path"])).name
        return settings.resolve_path(settings.local_display_dir) / (
            f"{kwargs['style_name']}_{reference_name}.png"
        )

    monkeypatch.setattr("src.main.run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["sweep-by-name", "lais_8", "--algorithm", "nst", "--style-intensity", "7.0"],
    )

    assert result.exit_code == 0
    assert "Found source:" in result.output
    assert "Skipping style 'empty_style'" in result.output
    assert "Running 3 render(s) across 2 style(s)" in result.output
    assert "[1/3] rococo_portrait -> rococo_portrait/ref_a.jpg" in result.output
    assert "[2/3] rococo_portrait -> rococo_portrait/ref_b.jpg" in result.output
    assert "[3/3] cubism -> cubism/ref_c.jpg" in result.output
    assert "Completed 3/3 render(s)." in result.output

    assert [call["style_name"] for call in calls] == [
        "rococo_portrait",
        "rococo_portrait",
        "cubism",
    ]
    assert [call["reference_path"] for call in calls] == [
        "rococo_portrait/ref_a.jpg",
        "rococo_portrait/ref_b.jpg",
        "cubism/ref_c.jpg",
    ]
    assert all(call["skip_sync"] is True for call in calls)
    assert all(call["skip_display"] is True for call in calls)
    assert all(call["skip_upload"] is True for call in calls)
    assert all(call["skip_archive"] is True for call in calls)
    assert all(call["style_intensity"] == 7.0 for call in calls)


def test_sweep_by_name_errors_when_image_is_missing(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    monkeypatch.setattr("src.cli.get_settings", lambda: settings)

    runner = CliRunner()
    result = runner.invoke(cli, ["sweep-by-name", "missing_subject"])

    assert result.exit_code != 0
    assert "No image matching 'missing_subject' found" in result.output


def test_sweep_uses_explicit_image_path(monkeypatch, tmp_path: Path):
    settings = _make_settings(tmp_path)
    source_path = tmp_path / "source.png"
    Image.new("RGB", (300, 450), "white").save(source_path)

    monkeypatch.setattr("src.cli.get_settings", lambda: settings)

    captured: dict[str, object] = {}

    def fake_run_reference_sweep(settings_arg, source_arg, *, algorithm, style_intensity):
        captured["settings"] = settings_arg
        captured["source_path"] = source_arg
        captured["algorithm"] = algorithm
        captured["style_intensity"] = style_intensity

    monkeypatch.setattr("src.cli._run_reference_sweep", fake_run_reference_sweep)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["sweep", str(source_path), "--algorithm", "diffusion", "--style-intensity", "6.0"],
    )

    assert result.exit_code == 0
    assert f"Found source: {source_path}" in result.output
    assert captured["settings"] == settings
    assert captured["source_path"] == source_path
    assert captured["algorithm"] == "diffusion"
    assert captured["style_intensity"] == 6.0
