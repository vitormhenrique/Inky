"""CLI entry-point for inky-stylize."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from src.config import Settings, get_settings
from src.models.style_profiles import StyleProfile, list_styles
from src.utils.files import IMAGE_EXTENSIONS, list_images


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Inky Stylisation System — transform photos into classic paintings."""


@dataclass(frozen=True)
class ReferenceSweepJob:
    style: StyleProfile
    reference_path: Path


def _iter_recursive_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _candidate_named_input_dirs(settings: Settings) -> list[Path]:
    directories = [settings.project_root / "data" / "raw"]
    cache_raw = settings.resolve_path(settings.local_cache_dir) / "raw"
    if cache_raw not in directories:
        directories.append(cache_raw)
    return directories


def _find_image_matches_by_name(settings: Settings, name: str) -> list[Path]:
    folded_name = name.casefold()
    matches: list[Path] = []
    for directory in _candidate_named_input_dirs(settings):
        matches.extend(
            path for path in _iter_recursive_images(directory) if folded_name in path.name.casefold()
        )
        if matches:
            break
    return matches


def _build_reference_sweep_jobs(
    settings: Settings,
) -> tuple[list[ReferenceSweepJob], list[tuple[str, Path]]]:
    styles_dir = settings.resolve_path(settings.local_styles_dir)
    jobs: list[ReferenceSweepJob] = []
    skipped: list[tuple[str, Path]] = []

    for style in list_styles():
        reference_dir = styles_dir / style.nst_reference_subdir
        references = list_images(reference_dir)
        if not references:
            skipped.append((style.name, reference_dir))
            continue
        jobs.extend(
            ReferenceSweepJob(style=style, reference_path=reference_path)
            for reference_path in references
        )

    return jobs, skipped


def _reference_argument(reference_path: Path, styles_dir: Path) -> str:
    try:
        return str(reference_path.relative_to(styles_dir))
    except ValueError:
        return str(reference_path)


def _run_reference_sweep(
    settings: Settings,
    source_path: Path,
    *,
    algorithm: str,
    style_intensity: float | None,
) -> None:
    from src.main import run_pipeline

    jobs, skipped = _build_reference_sweep_jobs(settings)
    for style_name, reference_dir in skipped:
        click.echo(
            f"Skipping style '{style_name}' (no reference images in {reference_dir})"
        )

    if not jobs:
        raise click.ClickException("No style reference images available for the sweep.")

    styles_dir = settings.resolve_path(settings.local_styles_dir)
    style_count = len({job.style.name for job in jobs})
    total_jobs = len(jobs)
    click.echo(
        f"Running {total_jobs} render(s) across {style_count} style(s) for {source_path.name}"
    )

    failures: list[tuple[ReferenceSweepJob, Exception]] = []
    for index, job in enumerate(jobs, start=1):
        reference_arg = _reference_argument(job.reference_path, styles_dir)
        click.echo(f"[{index}/{total_jobs}] {job.style.name} -> {reference_arg}")
        try:
            display_path = run_pipeline(
                settings,
                input_path=str(source_path),
                style_name=job.style.name,
                algorithm=algorithm,
                skip_sync=True,
                skip_display=True,
                skip_upload=True,
                skip_archive=True,
                style_intensity=style_intensity,
                reference_path=reference_arg,
            )
        except Exception as exc:
            failures.append((job, exc))
            click.echo(f"    Failed: {exc}")
            continue

        click.echo(f"    Saved: {display_path}")

    succeeded = total_jobs - len(failures)
    click.echo(f"Completed {succeeded}/{total_jobs} render(s).")
    if failures:
        raise click.ClickException(f"{len(failures)} render(s) failed during the sweep.")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    help="Explicit source image path.",
)
@click.option(
    "--style", "-s", "style_name", type=str, default=None, help="Style profile name."
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(["nst", "diffusion"]),
    default=None,
    help="Algorithm override.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["latest_parsed", "random_raw", "random_any"]),
    default=None,
    help="Image selection mode.",
)
@click.option("--skip-sync", is_flag=True, help="Skip Google Drive sync.")
@click.option("--skip-display", is_flag=True, help="Skip Inky display update.")
@click.option("--skip-upload", is_flag=True, help="Skip uploading results to Drive.")
@click.option(
    "--archive", is_flag=True, help="Archive (move) the source image after processing."
)
@click.option(
    "--style-intensity",
    type=float,
    default=None,
    help="NST style intensity override (1.0=subtle, 10.0=maximum). Overrides per-style default.",
)
@click.option(
    "--reference",
    "-r",
    "reference_path",
    type=click.Path(),
    default=None,
    help="Explicit style reference image path (overrides random selection).",
)
@click.option(
    "--variation-index",
    type=click.IntRange(1),
    default=None,
    help="Batch variation slot (used by generate).",
)
@click.option(
    "--variation-count",
    type=click.IntRange(1),
    default=None,
    help="Total batch variation count (used by generate).",
)
def run(
    input_path: str | None,
    style_name: str | None,
    algorithm: str | None,
    mode: str | None,
    skip_sync: bool,
    skip_display: bool,
    skip_upload: bool,
    archive: bool,
    style_intensity: float | None,
    reference_path: str | None,
    variation_index: int | None,
    variation_count: int | None,
) -> None:
    """Run the full stylisation pipeline once."""
    from src.main import run_pipeline

    settings = get_settings()
    display_path = run_pipeline(
        settings,
        input_path=input_path,
        style_name=style_name,
        algorithm=algorithm,
        selection_mode=mode,
        skip_sync=skip_sync,
        skip_display=skip_display,
        skip_upload=skip_upload,
        skip_archive=not archive,
        style_intensity=style_intensity,
        reference_path=reference_path,
        variation_index=variation_index,
        variation_count=variation_count,
    )
    click.echo(f"Done! Display image: {display_path}")


@cli.command(name="sweep-by-name")
@click.argument("name", type=str)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(["nst", "diffusion"]),
    default="nst",
    show_default=True,
    help="Algorithm to use for every style/reference combination.",
)
@click.option(
    "--style-intensity",
    type=float,
    default=None,
    help="NST style intensity override applied to every sweep render.",
)
def sweep_by_name(
    name: str,
    algorithm: str,
    style_intensity: float | None,
) -> None:
    """Match one source image by name and render every style/reference combination."""
    settings = get_settings()
    matches = _find_image_matches_by_name(settings, name)
    if not matches:
        searched = ", ".join(str(path) for path in _candidate_named_input_dirs(settings))
        raise click.ClickException(
            f"No image matching '{name}' found in {searched}"
        )

    source_path = matches[0]
    if len(matches) == 1:
        click.echo(f"Found source: {source_path}")
    else:
        click.echo(
            f"Found {len(matches)} matches for '{name}'; using first: {source_path}"
        )

    _run_reference_sweep(
        settings,
        source_path,
        algorithm=algorithm,
        style_intensity=style_intensity,
    )


@cli.command(name="sweep")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(["nst", "diffusion"]),
    default="nst",
    show_default=True,
    help="Algorithm to use for every style/reference combination.",
)
@click.option(
    "--style-intensity",
    type=float,
    default=None,
    help="NST style intensity override applied to every sweep render.",
)
def sweep(
    image_path: Path,
    algorithm: str,
    style_intensity: float | None,
) -> None:
    """Render one explicit image through every style/reference combination."""
    settings = get_settings()
    click.echo(f"Found source: {image_path}")
    _run_reference_sweep(
        settings,
        image_path,
        algorithm=algorithm,
        style_intensity=style_intensity,
    )


@cli.command()
def styles() -> None:
    """List all available style profiles."""
    for s in list_styles():
        affinity = s.subject_affinity.value
        click.echo(f"  {s.name:<30s} {s.display_name:<30s} [{affinity}]")


@cli.command()
def config() -> None:
    """Show current resolved configuration."""
    settings = get_settings()
    for field_name, field_info in settings.model_fields.items():
        value = getattr(settings, field_name)
        click.echo(f"  {field_name}: {value}")


@cli.command()
@click.option(
    "--subfolder", default="raw", help="Drive subfolder to sync (raw, parsed)."
)
def sync(subfolder: str) -> None:
    """Sync images from Google Drive to local cache."""
    from src.integrations.google_drive import sync_folder

    settings = get_settings()
    downloaded = sync_folder(settings, subfolder)
    click.echo(f"Downloaded {len(downloaded)} file(s) from '{subfolder}'")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def display(image_path: str) -> None:
    """Push an image directly to the Inky display (or simulate)."""
    from pathlib import Path

    from src.integrations.inky_display import update_display

    settings = get_settings()
    update_display(Path(image_path), settings)
    click.echo("Display updated.")


@cli.command()
def schedule() -> None:
    """Start the built-in daily scheduler (blocking)."""
    from src.scheduler.daily_job import start_scheduler

    settings = get_settings()
    start_scheduler(settings)


@cli.command()
def history() -> None:
    """Show display history."""
    from src.utils.metadata import load_history

    settings = get_settings()
    metadata_dir = settings.resolve_path(settings.local_metadata_dir)
    entries = load_history(metadata_dir)
    if not entries:
        click.echo("No display history yet.")
        return
    for e in entries[-10:]:
        click.echo(
            f"  {e['timestamp']}  {e['style_name']:<25s}  "
            f"{e['algorithm']:<10s}  {e['source_image']}"
        )


@cli.command(name="download-references")
@click.option("--style", type=str, default=None, help="Download only this style.")
@click.option("--dry-run", is_flag=True, help="List paintings without downloading.")
def download_references(style: str | None, dry_run: bool) -> None:
    """Download public-domain reference paintings from Wikimedia Commons."""
    import subprocess

    cmd = ["python", "scripts/download_references.py"]
    if style:
        cmd += ["--style", style]
    if dry_run:
        cmd += ["--dry-run"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()
