"""Neural Style Transfer — content/style optimisation with multi-scale refinement.

Runs fully locally on CPU or MPS/CUDA when available.
Uses a pre-trained VGG-19 feature extractor (ships with torchvision).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from src.config import Settings
from src.logging_utils import get_logger

log = get_logger("nst")

# ─── Image ↔ Tensor helpers ──────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_to_tensor = T.Compose(
    [T.ToTensor(), T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)]
)

_UNNORM_MEAN = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_UNNORM_STD = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
_NORM_MIN = ((torch.zeros(3) - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)).view(
    1, 3, 1, 1
)
_NORM_MAX = ((torch.ones(3) - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)).view(
    1, 3, 1, 1
)


def _img_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return _to_tensor(img).unsqueeze(0).to(device)


def _tensor_to_img(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().squeeze(0)
    t = t * _UNNORM_STD + _UNNORM_MEAN
    t = t.clamp(0, 1)
    return T.ToPILImage()(t)


def _clamp_normalized_(tensor: torch.Tensor) -> None:
    min_v = _NORM_MIN.to(tensor.device, tensor.dtype)
    max_v = _NORM_MAX.to(tensor.device, tensor.dtype)
    tensor.copy_(torch.maximum(torch.minimum(tensor, max_v), min_v))


def _prepare_style_image(style_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Crop the style image to the target aspect ratio before resizing."""
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    style_w, style_h = style_image.size
    style_ratio = style_w / style_h

    if abs(style_ratio - target_ratio) > 0.02:
        if style_ratio > target_ratio:
            crop_w = int(style_h * target_ratio)
            left = max(0, (style_w - crop_w) // 2)
            style_image = style_image.crop((left, 0, left + crop_w, style_h))
        else:
            crop_h = int(style_w / target_ratio)
            top = max(0, (style_h - crop_h) // 2)
            style_image = style_image.crop((0, top, style_w, top + crop_h))

    return style_image.resize(target_size, Image.LANCZOS)


def _resize_image(img: Image.Image, target_long_edge: int) -> Image.Image:
    if max(img.size) == target_long_edge:
        return img

    if img.width >= img.height:
        target_size = (target_long_edge, max(1, round(img.height * target_long_edge / img.width)))
    else:
        target_size = (max(1, round(img.width * target_long_edge / img.height)), target_long_edge)

    return img.resize(target_size, Image.LANCZOS)


def _build_scale_schedule(target_long_edge: int) -> list[int]:
    candidates = [min(384, target_long_edge), min(640, target_long_edge), target_long_edge]
    schedule = sorted({edge for edge in candidates if edge >= 256})
    return schedule or [target_long_edge]


def _split_steps(total_steps: int, num_scales: int) -> list[int]:
    if num_scales <= 1:
        return [total_steps]

    if num_scales == 2:
        ratios = [0.4, 0.6]
    else:
        ratios = [0.28, 0.32, 0.40]

    steps = [max(40, round(total_steps * ratio)) for ratio in ratios[:num_scales]]
    delta = total_steps - sum(steps)
    steps[-1] += delta
    if steps[-1] < 1:
        steps[-1] = 1
    return steps


# ─── Loss modules ────────────────────────────────────────────────────────────


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor, weight: float):
        super().__init__()
        self.target = target.detach()
        self.weight = weight
        self.loss: torch.Tensor = torch.tensor(0.0, device=target.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = nn.functional.mse_loss(x, self.target) * self.weight
        return x


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target: torch.Tensor, weight: float):
        super().__init__()
        self.target = _gram_matrix(target).detach()
        self.weight = weight
        self.loss: torch.Tensor = torch.tensor(0.0, device=target.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = _gram_matrix(x)
        self.loss = nn.functional.mse_loss(gram, self.target) * self.weight
        return x


class ReferenceLoss(nn.Module):
    def __init__(self, target: torch.Tensor, weight: float):
        super().__init__()
        self.target = target.detach()
        self.weight = weight
        self.loss: torch.Tensor = torch.tensor(0.0, device=target.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = nn.functional.l1_loss(x, self.target) * self.weight
        return x


# ─── Model builder ───────────────────────────────────────────────────────────

# VGG19 sequential conv index mapping:
# conv_1=conv1_1, conv_3=conv2_1, conv_5=conv3_1, conv_9=conv4_1,
# conv_10=conv4_2, conv_13=conv5_1.
_CONTENT_LAYER_WEIGHTS = {"conv_10": 1.0}
_STYLE_LAYER_WEIGHTS = {
    "conv_1": 0.2,
    "conv_3": 0.4,
    "conv_5": 0.8,
    "conv_9": 1.2,
    "conv_13": 1.6,
}
_REFERENCE_LAYER_WEIGHTS = {
    "conv_5": 0.10,
    "conv_9": 0.18,
}


def _build_model(
    cnn: nn.Module,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
) -> tuple[nn.Sequential, list[ContentLoss], list[StyleLoss], list[ReferenceLoss]]:
    """Return (model, content_losses, style_losses, reference_losses) with VGG features."""
    normalization = nn.Identity()  # already normalised via transforms
    model = nn.Sequential(normalization)

    content_losses: list[ContentLoss] = []
    style_losses: list[StyleLoss] = []
    reference_losses: list[ReferenceLoss] = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
            # AvgPool produces smoother results than MaxPool (Gatys et al.)
            layer = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in _CONTENT_LAYER_WEIGHTS:
            target = model(content_img).detach()
            cl = ContentLoss(target, _CONTENT_LAYER_WEIGHTS[name])
            model.add_module(f"content_loss_{i}", cl)
            content_losses.append(cl)

        if name in _STYLE_LAYER_WEIGHTS:
            target = model(style_img).detach()
            sl = StyleLoss(target, _STYLE_LAYER_WEIGHTS[name])
            model.add_module(f"style_loss_{i}", sl)
            style_losses.append(sl)

        if name in _REFERENCE_LAYER_WEIGHTS:
            target = model(style_img).detach()
            rl = ReferenceLoss(target, _REFERENCE_LAYER_WEIGHTS[name])
            model.add_module(f"reference_loss_{i}", rl)
            reference_losses.append(rl)

    # Trim layers after the last loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[: j + 1]

    return model, content_losses, style_losses, reference_losses


def _total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    horizontal = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
    vertical = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
    return horizontal + vertical


def _run_single_scale(
    cnn: nn.Module,
    content_t: torch.Tensor,
    style_t: torch.Tensor,
    input_img: torch.Tensor,
    *,
    content_weight: float,
    style_weight: float,
    reference_weight: float,
    num_steps: int,
    tv_weight: float,
    scale_name: str,
) -> torch.Tensor:
    model, content_losses, style_losses, reference_losses = _build_model(
        cnn,
        content_t,
        style_t,
    )

    input_img = input_img.clone().detach().requires_grad_(True)
    optimizer = optim.LBFGS([input_img], lr=1.0, max_iter=1, history_size=50)
    step = 0
    log_interval = max(20, num_steps // 4)

    while step < num_steps:

        def closure() -> torch.Tensor:
            nonlocal step
            with torch.no_grad():
                _clamp_normalized_(input_img)

            optimizer.zero_grad()
            model(input_img)

            content_score = sum(cl.loss for cl in content_losses) * content_weight
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            reference_score = sum(rl.loss for rl in reference_losses) * reference_weight
            tv_score = _total_variation_loss(input_img) * tv_weight
            loss = content_score + style_score + reference_score + tv_score
            loss.backward()

            step += 1
            if step == 1 or step % log_interval == 0 or step == num_steps:
                log.info(
                    "  %s step %d/%d  content=%.4f  style=%.4f  ref=%.4f  tv=%.4f",
                    scale_name,
                    step,
                    num_steps,
                    content_score.item(),
                    style_score.item(),
                    reference_score.item(),
                    tv_score.item(),
                )

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        _clamp_normalized_(input_img)
    return input_img.detach()


# ─── Public API ──────────────────────────────────────────────────────────────


def run_nst(
    content_image: Image.Image,
    style_image: Image.Image,
    settings: Settings,
    *,
    content_weight: float | None = None,
    style_weight: float | None = None,
    num_steps: int | None = None,
) -> Image.Image:
    """Run Neural Style Transfer and return the stylised PIL image.

    Parameters fall back to ``settings`` defaults when not explicitly provided.
    """
    device = torch.device(settings.detect_device())
    log.info("NST device: %s", device)

    cw = content_weight if content_weight is not None else settings.nst_content_weight
    sw = style_weight if style_weight is not None else settings.nst_style_weight
    steps = num_steps if num_steps is not None else settings.nst_num_steps
    scale_schedule = _build_scale_schedule(max(content_image.size))
    scale_steps = _split_steps(steps, len(scale_schedule))
    torch_cache_dir = settings.resolve_path(settings.local_cache_dir) / "torch"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))
    torch.hub.set_dir(str(torch_cache_dir))

    log.info(
        "NST params — content_weight=%.0e  style_weight=%.0e  steps=%d  scales=%s",
        cw,
        sw,
        steps,
        scale_schedule,
    )

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    input_img: torch.Tensor | None = None

    for idx, (scale_edge, scale_num_steps) in enumerate(zip(scale_schedule, scale_steps), start=1):
        scaled_content = _resize_image(content_image, scale_edge)
        scaled_style = _prepare_style_image(style_image, scaled_content.size)

        content_t = _img_to_tensor(scaled_content, device)
        style_t = _img_to_tensor(scaled_style, device)

        if input_img is None:
            input_img = content_t.clone()
            input_img = input_img + torch.randn_like(input_img) * 0.02
        else:
            input_img = nn.functional.interpolate(
                input_img,
                size=content_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        style_multiplier = 1.15 if idx == 1 and len(scale_schedule) > 1 else 1.0
        if idx == len(scale_schedule) and len(scale_schedule) > 1:
            style_multiplier = 0.9

        tv_weight = 3e-5 if idx == len(scale_schedule) else 1.5e-5
        reference_weight = 1.2 if idx == 1 else 0.9
        if idx == len(scale_schedule):
            reference_weight = 0.7
        scale_name = f"scale {idx}/{len(scale_schedule)} ({scaled_content.width}x{scaled_content.height})"
        log.info("Optimising %s", scale_name)

        input_img = _run_single_scale(
            cnn,
            content_t,
            style_t,
            input_img,
            content_weight=cw,
            style_weight=sw * style_multiplier,
            reference_weight=reference_weight,
            num_steps=scale_num_steps,
            tv_weight=tv_weight,
            scale_name=scale_name,
        )

    result = _tensor_to_img(input_img)
    log.info("NST complete — output size %dx%d", result.width, result.height)
    return result


def _collect_style_references(styles_dir: Path, style_subdir: str) -> list[Path]:
    """Return all reference images in ``styles_dir/style_subdir/``."""
    d = styles_dir / style_subdir
    if not d.is_dir():
        raise FileNotFoundError(
            f"Style reference directory not found: {d}. "
            f"Place at least one reference image in {d}/"
        )
    from src.utils.files import IMAGE_EXTENSIONS

    refs: list[Path] = []
    for ext in sorted(IMAGE_EXTENSIONS):
        refs.extend(sorted(d.glob(f"*{ext}")))
    if not refs:
        raise FileNotFoundError(f"No reference images found in {d}")
    return refs


def _aspect_ratio_distance(path: Path, target_ratio: float) -> float:
    with Image.open(path) as img:
        ratio = img.width / img.height
    return abs(ratio - target_ratio)


def find_style_reference(
    styles_dir: Path,
    style_subdir: str,
    *,
    target_size: tuple[int, int] | None = None,
    variation_index: int | None = None,
    variation_count: int | None = None,
) -> Path:
    """Pick the reference image whose aspect ratio best matches the target image."""
    refs = _collect_style_references(styles_dir, style_subdir)
    if target_size is not None and len(refs) > 1:
        target_ratio = target_size[0] / target_size[1]
        refs = sorted(
            refs,
            key=lambda ref: (_aspect_ratio_distance(ref, target_ratio), ref.name),
        )

    if (
        variation_index is None
        or variation_count is None
        or variation_count <= 1
        or len(refs) == 1
    ):
        return refs[0]

    slot = (max(variation_index, 1) - 1) % len(refs)
    return refs[slot]


def find_random_style_reference(styles_dir: Path, style_subdir: str) -> Path:
    """Pick a random reference image from ``styles_dir/style_subdir/``."""
    import random

    refs = _collect_style_references(styles_dir, style_subdir)
    return random.choice(refs)
