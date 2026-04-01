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
from src.pipeline.nst import run_nst
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
    skip_archive: bool = True,
    style_intensity: float | None = None,
    reference_path: str | None = None,
    variation_index: int | None = None,
    variation_count: int | None = None,
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
            stylised = run_diffusion(
                content_image,
                style,
                settings,
                source_name=source_path.stem,
                reference_path=reference_path,
            )
        elif settings.fallback_to_nst:
            log.warning("Diffusion unavailable (%s) — falling back to NST", reason)
            algo = "nst"
            stylised = _run_nst_with_style(
                content_image,
                style,
                settings,
                style_intensity,
                reference_path,
                variation_index=variation_index,
                variation_count=variation_count,
            )
        else:
            raise RuntimeError(
                f"Diffusion unavailable ({reason}) and fallback disabled"
            )
    else:
        stylised = _run_nst_with_style(
            content_image,
            style,
            settings,
            style_intensity,
            reference_path,
            variation_index=variation_index,
            variation_count=variation_count,
        )

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
    if not skip_archive:
        archive_dir = settings.resolve_path(settings.local_archive_dir)
        if source_path.is_file():
            archive_file(source_path, archive_dir)
    else:
        log.info("Step 10: Skipping archive")

    log.info("Pipeline complete! Display image: %s", display_path)
    return display_path


def _run_nst_with_style(
    content_image: Image.Image,
    style: StyleProfile,
    settings: Settings,
    style_intensity: float | None = None,
    reference_path: str | None = None,
    *,
    variation_index: int | None = None,
    variation_count: int | None = None,
) -> Image.Image:
    """Run NST using the style's reference image."""
    from src.pipeline.nst import find_style_reference
    from src.models.reference_analysis import analyze_reference_style

    styles_dir = settings.resolve_path(settings.local_styles_dir)

    if reference_path:
        ref = Path(reference_path)
        # If not absolute, resolve relative to the style's reference directory
        if not ref.is_absolute():
            ref = styles_dir / ref
        if not ref.is_file():
            raise FileNotFoundError(f"Reference image not found: {ref}")
        ref_path = ref
    else:
        ref_path = find_style_reference(
            styles_dir,
            style.nst_reference_subdir,
            target_size=content_image.size,
            variation_index=variation_index,
            variation_count=variation_count,
        )

    log.info("NST reference: %s", ref_path)
    style_image = load_image(ref_path)

    if style_intensity is not None:
        cw, sw = 1.0, 10 ** (style_intensity / 2 + 1)
    else:
        reference_analysis = analyze_reference_style(style_image, settings)
        cw, sw = style.compute_nst_weights(
            content_image.size,
            reference_analysis=reference_analysis,
        )

    cw, sw = _apply_nst_variation_weights(
        cw,
        sw,
        variation_index=variation_index,
        variation_count=variation_count,
    )
    if variation_index is not None and variation_count and variation_count > 1:
        slot = ((variation_index - 1) % variation_count) + 1
        log.info(
            "NST batch variation %d/%d — content_weight=%.3f style_weight=%.1f",
            slot,
            variation_count,
            cw,
            sw,
        )

    return run_nst(
        content_image,
        style_image,
        settings,
        content_weight=cw,
        style_weight=sw,
    )


def _apply_nst_variation_weights(
    content_weight: float,
    style_weight: float,
    *,
    variation_index: int | None = None,
    variation_count: int | None = None,
) -> tuple[float, float]:
    """Spread NST weights across batch runs without affecting single renders."""
    if variation_index is None or variation_count is None or variation_count <= 1:
        return content_weight, style_weight

    minimum_scale = 0.82
    maximum_scale = 1.18
    step = (maximum_scale - minimum_scale) / max(variation_count - 1, 1)
    raw_scales = [minimum_scale + step * idx for idx in range(variation_count)]
    scales = sorted(raw_scales, key=lambda value: (abs(value - 1.0), value < 1.0))
    slot = (max(variation_index, 1) - 1) % variation_count
    style_scale = scales[slot]
    content_scale = 1.0 - (style_scale - 1.0) * 0.35
    return content_weight * content_scale, style_weight * style_scale
