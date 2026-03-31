"""Diffusion-based img2img stylisation — local execution with hardware awareness.

Uses Hugging Face ``diffusers`` (optional dependency).
Gracefully falls back to NST when diffusion is unavailable or impractical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from src.config import Settings
from src.logging_utils import get_logger
from src.models.style_profiles import StyleProfile

log = get_logger("diffusion")

# ── Availability check ───────────────────────────────────────────────────────

_DIFFUSERS_AVAILABLE: bool | None = None


def is_diffusion_available() -> bool:
    """Return ``True`` if the ``diffusers`` library is importable."""
    global _DIFFUSERS_AVAILABLE
    if _DIFFUSERS_AVAILABLE is None:
        try:
            import diffusers  # noqa: F401

            _DIFFUSERS_AVAILABLE = True
        except ImportError:
            _DIFFUSERS_AVAILABLE = False
    return _DIFFUSERS_AVAILABLE


def should_use_diffusion(settings: Settings) -> tuple[bool, str]:
    """Decide whether diffusion is practical on the current hardware.

    Returns ``(usable, reason)``.
    """
    if "diffusion" not in settings.allowed_algorithms_list:
        return False, "diffusion not in allowed_algorithms"

    if not is_diffusion_available():
        return False, "diffusers library not installed"

    if settings.is_raspberry_pi:
        # Running diffusion on Pi is possible but extremely slow.
        # Default: disabled unless user explicitly requests.
        if settings.default_algorithm != "diffusion":
            return (
                False,
                "Raspberry Pi detected — diffusion disabled by default (too slow)",
            )
        log.warning(
            "Diffusion on Raspberry Pi will be very slow "
            "(expect 30+ minutes at reduced resolution). Proceeding as explicitly requested."
        )
        return True, "explicitly requested on Raspberry Pi"

    return True, "hardware capable"


# ── Pipeline loader ──────────────────────────────────────────────────────────


def _load_pipeline(settings: Settings) -> Any:
    """Lazily load and return the diffusion img2img pipeline."""
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    device = settings.detect_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    log.info(
        "Loading diffusion model %s on %s (dtype=%s)",
        settings.diffusion_model_id,
        device,
        dtype,
    )

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        settings.diffusion_model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    # Optimisations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


# ── Public API ───────────────────────────────────────────────────────────────


def run_diffusion(
    content_image: Image.Image,
    style: StyleProfile,
    settings: Settings,
    *,
    strength: float | None = None,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
) -> Image.Image:
    """Run diffusion img2img and return the stylised PIL image.

    Raises ``RuntimeError`` if diffusion is unavailable.
    """
    usable, reason = should_use_diffusion(settings)
    if not usable:
        raise RuntimeError(f"Diffusion not available: {reason}")

    pipe = _load_pipeline(settings)

    _strength = strength or style.recommended_strength or settings.diffusion_strength
    _guidance = (
        guidance_scale
        or style.recommended_guidance_scale
        or settings.diffusion_guidance_scale
    )
    _steps = (
        num_inference_steps
        or style.recommended_steps
        or settings.diffusion_num_inference_steps
    )

    # On Pi / constrained hardware, reduce resolution
    img = content_image
    if settings.is_raspberry_pi:
        max_edge = 512
        if max(img.size) > max_edge:
            scale = max_edge / max(img.size)
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
            )
            log.info("Reduced resolution for Pi: %dx%d", img.width, img.height)

    log.info(
        "Diffusion params — strength=%.2f  guidance=%.1f  steps=%d  size=%dx%d",
        _strength,
        _guidance,
        _steps,
        img.width,
        img.height,
    )

    result = pipe(
        prompt=style.prompt,
        negative_prompt=style.negative_prompt,
        image=img,
        strength=_strength,
        guidance_scale=_guidance,
        num_inference_steps=_steps,
    ).images[0]

    log.info("Diffusion complete — output size %dx%d", result.width, result.height)
    return result
