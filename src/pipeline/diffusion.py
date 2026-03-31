"""Diffusion-based img2img stylisation — local execution with hardware awareness.

Uses Hugging Face ``diffusers`` (optional dependency).
Gracefully falls back to NST when diffusion is unavailable or impractical.
"""

from __future__ import annotations

import os
import re
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
    cache_dir = settings.resolve_path(settings.local_cache_dir) / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache_dir / "diffusers"))

    # float16 on MPS produces corrupted images — only use it on CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info(
        "Loading diffusion model %s on %s (dtype=%s, cache=%s)",
        settings.diffusion_model_id,
        device,
        dtype,
        cache_dir,
    )

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        settings.diffusion_model_id,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Optimisations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    elif device == "mps":
        pipe.enable_attention_slicing()
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    return pipe


def _resize_image(img: Image.Image, *, max_edge: int) -> Image.Image:
    if max(img.size) <= max_edge:
        return img

    scale = max_edge / max(img.size)
    return img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )


def _snap_size_to_multiple(img: Image.Image, multiple: int = 8) -> Image.Image:
    width = max(multiple, img.width - (img.width % multiple))
    height = max(multiple, img.height - (img.height % multiple))
    if (width, height) == img.size:
        return img
    return img.resize((width, height), Image.LANCZOS)


def _prepare_diffusion_input(
    content_image: Image.Image,
    settings: Settings,
    *,
    device: str,
) -> Image.Image:
    """Resize the img2img input to a stable size for the diffusion model."""
    if settings.is_raspberry_pi:
        max_edge = 512
    elif device == "cuda":
        max_edge = 1024
    else:
        # Keep MPS / CPU runs smaller to avoid OOMs and long inference times.
        max_edge = 768

    img = _resize_image(content_image, max_edge=max_edge)
    img = _snap_size_to_multiple(img, multiple=8)
    return img


def _derive_source_hint(source_name: str | None) -> str | None:
    """Turn a descriptive filename into a prompt hint when it looks useful."""
    if not source_name:
        return None

    cleaned = re.sub(r"[_\-]+", " ", source_name).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None

    generic_tokens = {
        "img",
        "image",
        "photo",
        "picture",
        "scan",
        "screenshot",
        "copy",
        "final",
        "edit",
        "edited",
        "export",
        "pxl",
        "dsc",
    }
    tokens = [token for token in cleaned.split() if token.isalpha()]
    meaningful = [token for token in tokens if token not in generic_tokens]
    if len(meaningful) < 2:
        return None

    return " ".join(meaningful)


# ── Public API ───────────────────────────────────────────────────────────────


def run_diffusion(
    content_image: Image.Image,
    style: StyleProfile,
    settings: Settings,
    *,
    source_name: str | None = None,
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

    device = settings.detect_device()
    pipe = _load_pipeline(settings)
    source_hint = _derive_source_hint(source_name)
    tuning = style.compute_diffusion_tuning(
        content_image.size,
        source_hint=source_hint,
    )

    _strength = strength if strength is not None else tuning.strength
    _guidance = guidance_scale if guidance_scale is not None else tuning.guidance_scale
    _steps = (
        num_inference_steps
        if num_inference_steps is not None
        else tuning.num_inference_steps
    )

    img = _prepare_diffusion_input(content_image, settings, device=device)
    if img.size != content_image.size:
        log.info(
            "Adjusted diffusion input from %dx%d to %dx%d",
            content_image.width,
            content_image.height,
            img.width,
            img.height,
        )

    log.info(
        "Diffusion params — strength=%.2f  guidance=%.1f  steps=%d  size=%dx%d  hint=%s",
        _strength,
        _guidance,
        _steps,
        img.width,
        img.height,
        source_hint or "none",
    )

    result = pipe(
        prompt=tuning.prompt,
        negative_prompt=tuning.negative_prompt,
        image=img,
        strength=_strength,
        guidance_scale=_guidance,
        num_inference_steps=_steps,
    ).images[0]

    log.info("Diffusion complete — output size %dx%d", result.width, result.height)
    return result
