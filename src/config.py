"""Centralised configuration loaded from .env and environment variables."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings populated from environment variables / .env file."""

    # ── Google Drive ─────────────────────────────────────────
    gdrive_service_account_key: str = "path/to/service_account.json"
    gdrive_root_folder_id: str = ""

    # ── Local Paths (relative to project root) ───────────────
    local_cache_dir: Path = Path("data/cache")
    local_output_dir: Path = Path("data/output")
    local_display_dir: Path = Path("data/display")
    local_archive_dir: Path = Path("data/archive")
    local_styles_dir: Path = Path("data/styles")
    local_metadata_dir: Path = Path("data/metadata")

    # ── Algorithm ────────────────────────────────────────────
    default_algorithm: Literal["nst", "diffusion"] = "nst"
    allowed_algorithms: str = "nst,diffusion"

    # ── Style ────────────────────────────────────────────────
    default_style: str = "renaissance_portrait"

    # ── Image Selection ──────────────────────────────────────
    selection_mode: Literal["latest_parsed", "random_raw", "random_any"] = (
        "latest_parsed"
    )

    # ── Display ──────────────────────────────────────────────
    display_width: int = 1600
    display_height: int = 1200
    display_border_px: int = 0
    display_add_title: bool = False

    # ── Scheduler ────────────────────────────────────────────
    schedule_time: str = "06:00"

    # ── Hardware ─────────────────────────────────────────────
    device_preference: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # ── Diffusion ────────────────────────────────────────────
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5"
    diffusion_strength: float = 0.65
    diffusion_guidance_scale: float = 7.5
    diffusion_num_inference_steps: int = 30
    diffusion_controlnet_conditioning_scale: float = 0.8

    # ── NST ──────────────────────────────────────────────────
    nst_content_weight: float = 1e5
    nst_style_weight: float = 1e6
    nst_num_steps: int = 300
    nst_output_long_edge: int = 1024

    # ── Logging ──────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "data/logs/inky.log"

    # ── Fallback ─────────────────────────────────────────────
    fallback_to_nst: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Derived helpers ──────────────────────────────────────
    @property
    def allowed_algorithms_list(self) -> list[str]:
        """Parse comma-separated allowed_algorithms string into a list."""
        return [s.strip() for s in self.allowed_algorithms.split(",")]

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def resolve_path(self, p: Path) -> Path:
        """Resolve a path relative to the project root if not absolute."""
        if p.is_absolute():
            return p
        return self.project_root / p

    def detect_device(self) -> str:
        """Return the best available torch device string."""
        if self.device_preference != "auto":
            return self.device_preference

        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def is_raspberry_pi(self) -> bool:
        if platform.system() != "Linux":
            return False
        try:
            return "raspberry pi" in Path("/proc/device-tree/model").read_text().lower()
        except OSError:
            return False

    @property
    def display_resolution(self) -> tuple[int, int]:
        return (self.display_width, self.display_height)


def get_settings() -> Settings:
    """Factory that returns a validated Settings instance."""
    return Settings()
