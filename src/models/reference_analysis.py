"""Reference painting analysis using learned and classical visual descriptors."""

from __future__ import annotations

import colorsys
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import Settings
from src.logging_utils import get_logger

log = get_logger("reference_analysis")


@dataclass(frozen=True)
class ReferenceStyleAnalysis:
    dominant_colors: tuple[str, ...]
    palette_description: str
    brush_description: str
    mood_description: str
    prompt_fragments: tuple[str, ...]
    negative_fragments: tuple[str, ...]
    palette_mix: float
    saturation_boost: float
    contrast_boost: float
    blur_radius: float
    style_strength: float
    broad_stroke_score: float


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _classify_color(rgb: tuple[int, int, int]) -> str:
    r, g, b = (channel / 255 for channel in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if s < 0.16:
        if v < 0.22:
            return "charcoal"
        if v > 0.82:
            return "ivory"
        return "gray"

    if 0.55 <= h < 0.74:
        return "blue"
    if 0.44 <= h < 0.55:
        return "teal"
    if 0.74 <= h < 0.89:
        return "violet"
    if h < 0.03 or h >= 0.96:
        return "crimson"
    if 0.03 <= h < 0.10:
        return "orange"
    if 0.10 <= h < 0.18:
        return "gold" if v >= 0.55 else "ochre"
    if 0.18 <= h < 0.44:
        return "green"
    return "earth"


def _extract_dominant_colors(reference_image: Image.Image, limit: int = 3) -> tuple[str, ...]:
    quantized = reference_image.convert("RGB").resize((96, 96), Image.LANCZOS).quantize(
        colors=6,
        method=Image.MEDIANCUT,
    )
    palette = quantized.getpalette() or []
    counts = sorted(quantized.getcolors() or [], reverse=True)

    names: list[str] = []
    for _, palette_index in counts:
        start = palette_index * 3
        rgb = tuple(palette[start : start + 3])
        if len(rgb) != 3:
            continue
        name = _classify_color(rgb)
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break

    return tuple(names) or ("balanced",)


def _palette_metrics(reference_image: Image.Image) -> tuple[float, float, float, float]:
    hsv = np.asarray(
        reference_image.convert("RGB").resize((160, 160), Image.LANCZOS).convert("HSV"),
        dtype=np.float32,
    )
    hue = hsv[..., 0] / 255.0
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0

    sat_mean = float(saturation.mean())
    contrast = float(value.std())
    sat_weight = float(saturation.sum()) + 1e-6
    cool_ratio = float(
        saturation[((hue >= 0.45) & (hue <= 0.75))].sum() / sat_weight
    )
    warm_ratio = float(
        saturation[(((hue <= 0.18) | (hue >= 0.92)))].sum() / sat_weight
    )
    return sat_mean, contrast, cool_ratio, warm_ratio


def _estimate_broad_strokes(reference_image: Image.Image) -> tuple[float, float, float]:
    gray = np.asarray(
        reference_image.convert("L").resize((192, 192), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    magnitude = np.hypot(gx, gy)
    edge_strength = float(_clamp(np.quantile(magnitude, 0.85) * 4.0, 0.0, 1.0))

    orientation = np.arctan2(gy, gx)
    hist, _ = np.histogram(
        orientation,
        bins=12,
        range=(-math.pi, math.pi),
        weights=magnitude + 1e-6,
    )
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-6
    orientation_entropy = float(
        -np.sum(hist * np.log(hist + 1e-6)) / math.log(len(hist))
    )

    spectrum = np.abs(np.fft.rfft2(gray - gray.mean()))
    fy = np.fft.fftfreq(gray.shape[0])[:, None]
    fx = np.fft.rfftfreq(gray.shape[1])[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    low_energy = float(spectrum[radius < 0.10].sum())
    high_energy = float(spectrum[radius > 0.22].sum()) + 1e-6

    coarse = (
        Image.fromarray((gray * 255).astype(np.uint8))
        .resize((48, 48), Image.LANCZOS)
        .resize((192, 192), Image.LANCZOS)
    )
    coarse_arr = np.asarray(coarse, dtype=np.float32) / 255.0
    coarse_similarity = 1.0 - float(
        _clamp(np.mean(np.abs(gray - coarse_arr)) / 0.22, 0.0, 1.0)
    )

    broad_score = float(
        _clamp(
            0.45 * (low_energy / (low_energy + high_energy)) + 0.55 * coarse_similarity,
            0.0,
            1.0,
        )
    )

    return broad_score, orientation_entropy, edge_strength


def _estimate_learned_texture_strength(
    reference_image: Image.Image,
    settings: Settings | None = None,
) -> float:
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
    except ImportError:
        return 0.5

    torch_home_candidates: list[Path] = []
    if settings is not None:
        torch_cache_dir = settings.resolve_path(settings.local_cache_dir) / "torch"
        torch_cache_dir.mkdir(parents=True, exist_ok=True)
        torch_home_candidates.append(torch_cache_dir)
    torch_home_candidates.append(Path.home() / ".cache" / "torch")

    weights_name = "vgg19-dcbb9e9d.pth"
    torch_home = next(
        (
            candidate
            for candidate in torch_home_candidates
            if (candidate / "hub" / "checkpoints" / weights_name).is_file()
        ),
        None,
    )
    if torch_home is None:
        log.info("No cached VGG19 weights found; using classical reference analysis only")
        return 0.5

    os.environ["TORCH_HOME"] = str(torch_home)

    try:
        cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:18].eval()
    except Exception as exc:
        log.warning("Falling back to classical reference analysis only: %s", exc)
        return 0.5

    tensor = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )(reference_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        x = tensor
        activations: list[torch.Tensor] = []
        for index, layer in enumerate(cnn):
            x = layer(x)
            if index in {3, 8, 17}:
                activations.append(x)

    if len(activations) != 3:
        return 0.5

    shallow, mid, deep = activations
    raw_score = (
        shallow.var(dim=(1, 2, 3), unbiased=False).mean().item() * 12.0
        + mid.var(dim=(1, 2, 3), unbiased=False).mean().item() * 8.0
        + deep.abs().mean().item() * 1.5
    )
    return float(_clamp(math.tanh(raw_score), 0.0, 1.0))


def _build_palette_description(
    dominant_colors: tuple[str, ...],
    *,
    saturation: float,
    cool_ratio: float,
    warm_ratio: float,
) -> str:
    vivid_colors = [color for color in dominant_colors if color not in {"gray", "ivory", "charcoal"}]
    ordered = vivid_colors or list(dominant_colors)
    primary = ordered[0] if ordered else "balanced"
    accent = next((color for color in ordered[1:] if color != primary), None)

    if cool_ratio > warm_ratio + 0.15:
        temperature = "cool"
    elif warm_ratio > cool_ratio + 0.15:
        temperature = "warm"
    else:
        temperature = "balanced"

    if accent:
        base = f"{temperature} {primary} palette with {accent} accents"
    else:
        base = f"{temperature} {primary} palette"

    if saturation > 0.58:
        return f"saturated {base}"
    if saturation < 0.28:
        return f"muted {base}"
    return base


def _build_brush_description(
    *,
    broad_strokes: float,
    orientation_entropy: float,
    edge_strength: float,
    learned_texture: float,
) -> str:
    if broad_strokes > 0.48 and orientation_entropy > 0.78:
        return "sweeping curved brush strokes with flowing paint movement"
    if edge_strength > 0.48 and orientation_entropy < 0.72:
        return "angular directional brushwork"
    if broad_strokes > 0.52:
        return "broad painterly brush strokes"
    if learned_texture > 0.58 or edge_strength > 0.42:
        return "dense visible brush texture"
    return "soft blended brushwork"


def _build_mood_description(*, brightness: float, contrast: float) -> str:
    if contrast > 0.20 and brightness > 0.48:
        return "luminous high-contrast paint handling"
    if brightness < 0.36:
        return "moody low-key atmosphere"
    if brightness > 0.68:
        return "airy light-filled atmosphere"
    return "balanced painterly atmosphere"


def analyze_reference_style(
    reference_image: Image.Image,
    settings: Settings | None = None,
) -> ReferenceStyleAnalysis:
    """Summarise a reference painting into palette, brush, and conditioning cues."""
    rgb_image = reference_image.convert("RGB")
    dominant_colors = _extract_dominant_colors(rgb_image)
    saturation, contrast, cool_ratio, warm_ratio = _palette_metrics(rgb_image)
    brightness = float(
        np.asarray(rgb_image.convert("L").resize((128, 128), Image.LANCZOS), dtype=np.float32).mean()
        / 255.0
    )
    broad_strokes, orientation_entropy, edge_strength = _estimate_broad_strokes(rgb_image)
    learned_texture = _estimate_learned_texture_strength(rgb_image, settings)
    contrast_norm = _clamp(contrast / 0.22, 0.0, 1.0)

    palette_description = _build_palette_description(
        dominant_colors,
        saturation=saturation,
        cool_ratio=cool_ratio,
        warm_ratio=warm_ratio,
    )
    brush_description = _build_brush_description(
        broad_strokes=broad_strokes,
        orientation_entropy=orientation_entropy,
        edge_strength=edge_strength,
        learned_texture=learned_texture,
    )
    mood_description = _build_mood_description(
        brightness=brightness,
        contrast=contrast,
    )

    negative_fragments: list[str] = []
    if saturation > 0.42:
        negative_fragments.append("washed-out color")
    if broad_strokes > 0.48:
        negative_fragments.append("tiny brush strokes")
    if contrast > 0.15:
        negative_fragments.append("flat lighting")
    if learned_texture > 0.55:
        negative_fragments.append("plastic smooth shading")

    style_strength = _clamp(
        0.15
        + 0.25 * saturation
        + 0.18 * contrast_norm
        + 0.22 * broad_strokes
        + 0.10 * edge_strength
        + 0.15 * learned_texture,
        0.0,
        1.0,
    )
    palette_mix = _clamp(
        0.28 + 0.30 * saturation + 0.22 * abs(cool_ratio - warm_ratio) + 0.10 * contrast_norm,
        0.25,
        0.75,
    )
    saturation_boost = _clamp(0.96 + 0.32 * saturation, 0.95, 1.30)
    contrast_boost = _clamp(0.92 + 0.22 * contrast_norm, 0.90, 1.18)
    blur_radius = _clamp(0.18 + 0.85 * broad_strokes, 0.15, 1.15)

    prompt_fragments = (
        palette_description,
        brush_description,
        mood_description,
    )

    log.info(
        "Reference analysis — colors=%s  palette=%s  brush=%s  style_strength=%.2f",
        ",".join(dominant_colors),
        palette_description,
        brush_description,
        style_strength,
    )

    return ReferenceStyleAnalysis(
        dominant_colors=dominant_colors,
        palette_description=palette_description,
        brush_description=brush_description,
        mood_description=mood_description,
        prompt_fragments=prompt_fragments,
        negative_fragments=tuple(negative_fragments),
        palette_mix=float(round(palette_mix, 2)),
        saturation_boost=float(round(saturation_boost, 2)),
        contrast_boost=float(round(contrast_boost, 2)),
        blur_radius=float(round(blur_radius, 2)),
        style_strength=float(round(style_strength, 2)),
        broad_stroke_score=float(round(broad_strokes, 2)),
    )
