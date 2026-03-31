"""Main pipeline orchestrator — ties all modules together."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import Settings, get_settings
from src.integrations.google_drive import sync_all, upload_to_drive
from src.integrations.inky_display import update_display
from src.logging_utils import get_logger, setup_logging
from src.models.style_profiles import StyleProfile, get_style
from src.pipeline.diffusion import (
    is_diffusion_available,
    run_diffusion,
    should_use_diffusion,
)
from src.pipeline.nst import find_style_reference, run_nst
from src.pipeline.postprocess import prepare_display_image, save_outputs
from src.pipeline.preprocess import preprocess
from src.pipeline.selector import select_image
from src.utils.files import archive_file, ensure_dirs
from src.utils.image_ops import load_image
from src.utils.metadata import record_display

log = get_logger("main")


def run_pipeline(
    settings: Settings | None = None,
    *,
    input_path: str | None = None,
    style_name: str | None = None,
    algorithm: str | None = None,
    selection_mode: str | None = None,
    skip_sync: bool = False,
    skip_display: bool = False,
    skip_upload: bool = False,
) -> Path:
    """Execute the full stylisation pipeline and return the display image path.

    This is the main entry point called by the CLI and the daily scheduler.
    """
    if settings is None:
        settings = get_settings()

    setup_logging(settings.log_level, settings.log_file)
    ensure_dirs(settings)

    log.info("=" * 60)
    log.info("Starting Inky Stylisation Pipeline")
    log.info("=" * 60)

    # 1. Sync Google Drive
    if not skip_sync:
        log.info("Step 1: Syncing Google Drive…")
        try:
            sync_all(settings)
        except Exception as e:
            log.warning("Drive sync failed (continuing with cache): %s", e)
    else:
        log.info("Step 1: Skipping Drive sync")

    # 2. Select image
    log.info("Step 2: Selecting image…")
    explicit = Path(input_path) if input_path else None
    source_path = select_image(
        settings, explicit_path=explicit, mode_override=selection_mode
    )
    log.info("Selected: %s", source_path)

    # 3. Resolve style
    resolved_style_name = style_name or settings.default_style
    style = get_style(resolved_style_name)
    log.info("Step 3: Style = %s", style.display_name)

    # 4. Preprocess
    log.info("Step 4: Preprocessing…")
    content_image = preprocess(source_path, settings)

    # 5. Stylise
    log.info("Step 5: Stylising…")
    algo = algorithm or settings.default_algorithm
    stylised: Image.Image

    if algo == "diffusion":
        usable, reason = should_use_diffusion(settings)
        if usable:
            log.info("Using diffusion img2img")
            stylised = run_diffusion(content_image, style, settings)
        elif settings.fallback_to_nst:
            log.warning("Diffusion unavailable (%s) — falling back to NST", reason)
            algo = "nst"
            stylised = _run_nst_with_style(content_image, style, settings)
        else:
            raise RuntimeError(
                f"Diffusion unavailable ({reason}) and fallback disabled"
            )
    else:
        stylised = _run_nst_with_style(content_image, style, settings)

    # 6. Post-process & save
    log.info("Step 6: Post-processing & saving…")
    display_image = prepare_display_image(
        stylised, settings, style_name=style.display_name
    )
    hires_path, display_path = save_outputs(
        stylised,
        display_image,
        settings,
        source_name=source_path.stem,
        style_name=style.name,
        algorithm=algo,
    )

    # 7. Update display
    if not skip_display:
        log.info("Step 7: Updating display…")
        update_display(display_path, settings)
    else:
        log.info("Step 7: Skipping display update")

    # 8. Upload styled output to Drive
    if not skip_upload:
        try:
            upload_to_drive(settings, hires_path, "styled")
            upload_to_drive(settings, display_path, "display")
        except Exception as e:
            log.warning("Drive upload failed: %s", e)

    # 9. Record metadata
    log.info("Step 8: Recording metadata…")
    metadata_dir = settings.resolve_path(settings.local_metadata_dir)
    record_display(
        metadata_dir,
        source_image=str(source_path),
        style_name=style.name,
        algorithm=algo,
        output_path=str(hires_path),
        display_path=str(display_path),
    )

    # 10. Archive source
    archive_dir = settings.resolve_path(settings.local_archive_dir)
    if source_path.is_file():
        archive_file(source_path, archive_dir)

    log.info("Pipeline complete! Display image: %s", display_path)
    return display_path


def _run_nst_with_style(
    content_image: Image.Image,
    style: StyleProfile,
    settings: Settings,
) -> Image.Image:
    """Run NST using the style's reference image."""
    styles_dir = settings.resolve_path(settings.local_styles_dir)
    ref_path = find_style_reference(styles_dir, style.nst_reference_subdir)
    log.info("NST reference: %s", ref_path)
    style_image = load_image(ref_path)

    return run_nst(
        content_image,
        style_image,
        settings,
        content_weight=style.nst_content_weight,
        style_weight=style.nst_style_weight,
    )
