"""Neural Style Transfer — classic content/style loss optimisation.

Runs fully locally on CPU or MPS/CUDA when available.
Uses a pre-trained VGG-19 feature extractor (ships with torchvision).
"""

from __future__ import annotations

import copy
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

_to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)])

_UNNORM_MEAN = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_UNNORM_STD = torch.tensor(_IMAGENET_STD).view(3, 1, 1)


def _img_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return _to_tensor(img).unsqueeze(0).to(device)


def _tensor_to_img(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().squeeze(0)
    t = t * _UNNORM_STD + _UNNORM_MEAN
    t = t.clamp(0, 1)
    return T.ToPILImage()(t)


# ─── Loss modules ────────────────────────────────────────────────────────────


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = _gram_matrix(target).detach()
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = _gram_matrix(x)
        self.loss = nn.functional.mse_loss(gram, self.target)
        return x


# ─── Model builder ───────────────────────────────────────────────────────────

_CONTENT_LAYERS = ["conv_4"]
_STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


def _build_model(
    cnn: nn.Module,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Sequential, list[ContentLoss], list[StyleLoss]]:
    """Return (model, content_losses, style_losses) with VGG features."""
    normalization = nn.Identity()  # already normalised via transforms
    model = nn.Sequential(normalization)

    content_losses: list[ContentLoss] = []
    style_losses: list[StyleLoss] = []

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
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in _CONTENT_LAYERS:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{i}", cl)
            content_losses.append(cl)

        if name in _STYLE_LAYERS:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{i}", sl)
            style_losses.append(sl)

    # Trim layers after the last loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[: j + 1]

    return model, content_losses, style_losses


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

    cw = content_weight or settings.nst_content_weight
    sw = style_weight or settings.nst_style_weight
    steps = num_steps or settings.nst_num_steps

    log.info("NST params — content_weight=%.0e  style_weight=%.0e  steps=%d", cw, sw, steps)

    content_t = _img_to_tensor(content_image, device)
    style_t = _img_to_tensor(
        style_image.resize(content_image.size, Image.LANCZOS), device
    )

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    model, content_losses, style_losses = _build_model(cnn, content_t, style_t, device)

    # Optimise the input image (start from content)
    input_img = content_t.clone().requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    run_step = [0]
    while run_step[0] <= steps:

        def closure() -> float:
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)

            content_score = sum(cl.loss for cl in content_losses) * cw
            style_score = sum(sl.loss for sl in style_losses) * sw
            loss = content_score + style_score
            loss.backward()

            run_step[0] += 1
            if run_step[0] % 50 == 0:
                log.info(
                    "  step %d/%d  content=%.2f  style=%.2f",
                    run_step[0],
                    steps,
                    content_score.item(),
                    style_score.item(),
                )
            return loss.item()

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    result = _tensor_to_img(input_img)
    log.info("NST complete — output size %dx%d", result.width, result.height)
    return result


def find_style_reference(styles_dir: Path, style_subdir: str) -> Path:
    """Locate the first reference image in ``styles_dir/style_subdir/``."""
    d = styles_dir / style_subdir
    if not d.is_dir():
        raise FileNotFoundError(
            f"Style reference directory not found: {d}. "
            f"Place at least one reference image in {d}/"
        )
    from src.utils.files import IMAGE_EXTENSIONS

    for ext in sorted(IMAGE_EXTENSIONS):
        for f in sorted(d.glob(f"*{ext}")):
            return f
    raise FileNotFoundError(f"No reference images found in {d}")
