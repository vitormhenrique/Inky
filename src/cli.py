"""CLI entry-point for inky-stylize."""

from __future__ import annotations

import click

from src.config import get_settings
from src.models.style_profiles import list_styles


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Inky Stylisation System — transform photos into classic paintings."""


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
    )
    click.echo(f"Done! Display image: {display_path}")


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
