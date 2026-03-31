"""Low-level image manipulation utilities using Pillow (no ML deps)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_image(path: Path) -> Image.Image:
    """Open an image and ensure RGB mode."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_long_edge(img: Image.Image, max_long: int) -> Image.Image:
    """Resize so the longest edge equals *max_long*, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_long:
        return img
    scale = max_long / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def fit_to_display(
    img: Image.Image,
    width: int,
    height: int,
    border_px: int = 0,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Resize and letter-box / pillar-box *img* to fit exactly *width* × *height*.

    The image is never up-scaled beyond its original size — only down-scaled or centered.
    """
    target_w = width - 2 * border_px
    target_h = height - 2 * border_px

    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * img_ratio)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), bg_color)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def add_title_bar(
    img: Image.Image,
    title: str,
    bar_height: int = 48,
    font_size: int = 28,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    text_color: tuple[int, int, int] = (220, 220, 220),
) -> Image.Image:
    """Prepend a title bar to the top of *img*."""
    bar = Image.new("RGB", (img.width, bar_height), bg_color)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((bar.width - tw) // 2, (bar_height - th) // 2),
        title,
        fill=text_color,
        font=font,
    )

    combined = Image.new("RGB", (img.width, img.height + bar_height))
    combined.paste(bar, (0, 0))
    combined.paste(img, (0, bar_height))
    return combined


def center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop *img* to exactly *width* × *height*."""
    w, h = img.size
    left = max(0, (w - width) // 2)
    top = max(0, (h - height) // 2)
    return img.crop((left, top, left + width, top + height))


def composition_safe_crop(
    img: Image.Image,
    target_ratio: float | None = None,
) -> Image.Image:
    """Crop to *target_ratio* while keeping the centre.

    If *target_ratio* is ``None``, returns the image unchanged.
    """
    if target_ratio is None:
        return img
    w, h = img.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.02:
        return img
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        return center_crop(img, new_w, h)
    else:
        new_h = int(w / target_ratio)
        return center_crop(img, w, new_h)
