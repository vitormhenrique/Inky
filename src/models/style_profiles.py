"""Style profile registry — defines painting styles for NST and diffusion pipelines."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class SubjectAffinity(str, Enum):
    HUMAN = "human"
    ANIMAL = "animal"
    BOTH = "both"


class StyleProfile(BaseModel):
    """Complete description of a target painting style."""

    name: str
    display_name: str

    # Diffusion prompts
    prompt: str
    negative_prompt: str = (
        "blurry, low quality, distorted face, extra limbs, watermark, text, "
        "modern, digital art, 3d render, anime"
    )

    # NST guidance
    nst_reference_subdir: (
        str  # e.g. "renaissance_portrait" — maps to data/styles/<subdir>/
    )
    nst_content_weight: float | None = None  # override global if set
    nst_style_weight: float | None = None

    # Pre / post guidance (human-readable notes, used in docs / logs)
    preprocessing_notes: str = ""
    postprocessing_notes: str = ""

    # Subject affinity
    subject_affinity: SubjectAffinity = SubjectAffinity.BOTH

    # Recommended diffusion params
    recommended_strength: float = 0.65
    recommended_guidance_scale: float = 7.5
    recommended_steps: int = 30


# ─── Built-in style registry ─────────────────────────────────────────────────

BUILTIN_STYLES: dict[str, StyleProfile] = {}


def _register(profile: StyleProfile) -> None:
    BUILTIN_STYLES[profile.name] = profile


_register(
    StyleProfile(
        name="renaissance_portrait",
        display_name="Renaissance Portrait",
        prompt=(
            "a Renaissance oil portrait painting, soft chiaroscuro lighting, "
            "rich warm tones, classical composition, museum quality, "
            "masterful brushwork, Leonardo da Vinci style"
        ),
        nst_reference_subdir="renaissance_portrait",
        preprocessing_notes="Crop tightly to head/shoulders for best effect.",
        postprocessing_notes="Slight warm color grading enhances the period feel.",
        subject_affinity=SubjectAffinity.HUMAN,
        recommended_strength=0.60,
        recommended_guidance_scale=7.5,
    )
)

_register(
    StyleProfile(
        name="baroque_oil_painting",
        display_name="Baroque Oil Painting",
        prompt=(
            "a Baroque oil painting, dramatic chiaroscuro, deep shadows, "
            "golden highlights, Caravaggio dramatic lighting, rich pigments, "
            "17th century masterwork"
        ),
        nst_reference_subdir="baroque_oil_painting",
        preprocessing_notes="Works best with dramatic side lighting in source photo.",
        postprocessing_notes="Deepen blacks slightly in post for the Caravaggio effect.",
        subject_affinity=SubjectAffinity.BOTH,
        recommended_strength=0.65,
    )
)

_register(
    StyleProfile(
        name="rococo_portrait",
        display_name="Rococo Portrait",
        prompt=(
            "an elegant Rococo portrait, pastel palette, soft light, "
            "delicate brushwork, ornate details, Fragonard style, "
            "18th century French court elegance"
        ),
        nst_reference_subdir="rococo_portrait",
        preprocessing_notes="Lighter backgrounds translate better into Rococo style.",
        postprocessing_notes="Boost saturation of pinks and blues slightly.",
        subject_affinity=SubjectAffinity.HUMAN,
        recommended_strength=0.60,
        recommended_guidance_scale=8.0,
    )
)

_register(
    StyleProfile(
        name="dutch_golden_age",
        display_name="Dutch Golden Age",
        prompt=(
            "a Dutch Golden Age portrait, Rembrandt lighting, warm earth tones, "
            "dark background, luminous skin, masterful impasto, "
            "17th century Dutch masters technique"
        ),
        nst_reference_subdir="dutch_golden_age",
        preprocessing_notes="Dark or neutral backgrounds strongly recommended.",
        postprocessing_notes="Add subtle vignette for period authenticity.",
        subject_affinity=SubjectAffinity.BOTH,
        recommended_strength=0.65,
    )
)

_register(
    StyleProfile(
        name="romanticism",
        display_name="Romanticism",
        prompt=(
            "a Romantic era painting, dramatic sky, emotional atmosphere, "
            "rich saturated colors, expressive brushstrokes, "
            "Delacroix or Turner style, sublime landscape quality"
        ),
        nst_reference_subdir="romanticism",
        preprocessing_notes="Outdoor scenes or dramatic poses work well.",
        postprocessing_notes="Enhance contrast for dramatic atmosphere.",
        subject_affinity=SubjectAffinity.BOTH,
        recommended_strength=0.70,
        recommended_guidance_scale=8.0,
    )
)

_register(
    StyleProfile(
        name="impressionism",
        display_name="Impressionism",
        prompt=(
            "an Impressionist painting, visible brushstrokes, vibrant light, "
            "soft focus, plein air atmosphere, Monet and Renoir style, "
            "dappled sunlight, color harmony"
        ),
        nst_reference_subdir="impressionism",
        preprocessing_notes="Outdoor photos with natural light are ideal source material.",
        postprocessing_notes="Slight softening filter can enhance the impressionistic feel.",
        subject_affinity=SubjectAffinity.BOTH,
        recommended_strength=0.70,
        recommended_guidance_scale=7.0,
    )
)

_register(
    StyleProfile(
        name="post_impressionism",
        display_name="Post-Impressionism",
        prompt=(
            "a Post-Impressionist painting, bold expressive color, "
            "structured brushwork, Van Gogh or Cezanne style, "
            "vibrant palette, emotional intensity, thick impasto"
        ),
        nst_reference_subdir="post_impressionism",
        preprocessing_notes="Strong shapes and outlines in source help preserve structure.",
        postprocessing_notes="Boost vibrance; consider slight texture overlay.",
        subject_affinity=SubjectAffinity.BOTH,
        recommended_strength=0.70,
        recommended_guidance_scale=7.5,
    )
)

_register(
    StyleProfile(
        name="victorian_animal_portrait",
        display_name="Victorian Animal Portrait",
        prompt=(
            "a Victorian-era animal portrait painting, dignified pose, "
            "rich dark background, fine fur detail, oil painting technique, "
            "Edwin Landseer style, noble animal portrait, museum quality"
        ),
        nst_reference_subdir="victorian_animal_portrait",
        preprocessing_notes="Crop to highlight the animal's head/upper body. Dark BG preferred.",
        postprocessing_notes="Warm mid-tones; deepen background.",
        subject_affinity=SubjectAffinity.ANIMAL,
        recommended_strength=0.60,
    )
)

_register(
    StyleProfile(
        name="classical_equestrian",
        display_name="Classical Equestrian Portrait",
        prompt=(
            "a classical equestrian portrait, horse and rider, "
            "dramatic landscape background, heroic composition, "
            "George Stubbs style, oil on canvas, museum masterpiece"
        ),
        nst_reference_subdir="classical_equestrian",
        preprocessing_notes="Full-body pose with visible limbs works best.",
        postprocessing_notes="Add subtle landscape glow in background.",
        subject_affinity=SubjectAffinity.ANIMAL,
        recommended_strength=0.65,
    )
)

_register(
    StyleProfile(
        name="naturalist_oil_portrait",
        display_name="Naturalist Oil Portrait",
        prompt=(
            "a Naturalist oil portrait, precise realistic rendering, "
            "careful observation of light and form, John Singer Sargent style, "
            "refined technique, warm skin tones, elegant composition"
        ),
        nst_reference_subdir="naturalist_oil_portrait",
        preprocessing_notes="High-quality evenly lit source photos yield the best results.",
        postprocessing_notes="Minimal post needed — style is close to photorealism.",
        subject_affinity=SubjectAffinity.BOTH,
        recommended_strength=0.55,
        recommended_guidance_scale=7.0,
    )
)


def get_style(name: str) -> StyleProfile:
    """Look up a style by name. Raises ``KeyError`` if not found."""
    if name not in BUILTIN_STYLES:
        available = ", ".join(sorted(BUILTIN_STYLES.keys()))
        raise KeyError(f"Unknown style '{name}'. Available: {available}")
    return BUILTIN_STYLES[name]


def list_styles() -> list[StyleProfile]:
    """Return all registered styles sorted by name."""
    return sorted(BUILTIN_STYLES.values(), key=lambda s: s.name)


def suggest_style_for_subject(
    subject: Literal["human", "animal"],
    default: str = "renaissance_portrait",
) -> StyleProfile:
    """Return a style that has affinity for the given subject type.

    Falls back to *default* if no specific match is found.
    """
    candidates = [
        s
        for s in BUILTIN_STYLES.values()
        if s.subject_affinity.value == subject
        or s.subject_affinity == SubjectAffinity.BOTH
    ]
    if not candidates:
        return BUILTIN_STYLES[default]
    # Return first alphabetically among candidates for determinism
    return sorted(candidates, key=lambda s: s.name)[0]
