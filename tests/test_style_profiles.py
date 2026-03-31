"""Tests for style profile registry."""

from __future__ import annotations

import pytest

from src.models.style_profiles import (
    BUILTIN_STYLES,
    SubjectAffinity,
    get_style,
    list_styles,
    suggest_style_for_subject,
)


class TestStyleRegistry:
    def test_builtin_styles_not_empty(self):
        assert len(BUILTIN_STYLES) >= 10

    def test_all_styles_have_prompts(self):
        for name, style in BUILTIN_STYLES.items():
            assert style.prompt, f"{name} has no prompt"
            assert style.nst_reference_subdir, f"{name} has no nst_reference_subdir"

    def test_get_style_valid(self):
        style = get_style("renaissance_portrait")
        assert style.name == "renaissance_portrait"
        assert style.display_name == "Renaissance Portrait"

    def test_get_style_invalid(self):
        with pytest.raises(KeyError, match="Unknown style"):
            get_style("nonexistent_style")

    def test_list_styles_sorted(self):
        styles = list_styles()
        names = [s.name for s in styles]
        assert names == sorted(names)

    def test_suggest_for_human(self):
        style = suggest_style_for_subject("human")
        assert style.subject_affinity in (SubjectAffinity.HUMAN, SubjectAffinity.BOTH)

    def test_suggest_for_animal(self):
        style = suggest_style_for_subject("animal")
        assert style.subject_affinity in (SubjectAffinity.ANIMAL, SubjectAffinity.BOTH)

    def test_all_styles_have_valid_affinity(self):
        for name, style in BUILTIN_STYLES.items():
            assert style.subject_affinity in SubjectAffinity

    def test_style_recommended_params(self):
        for name, style in BUILTIN_STYLES.items():
            assert (
                0.0 < style.recommended_strength <= 1.0
            ), f"{name} strength out of range"
            assert (
                style.recommended_guidance_scale > 0
            ), f"{name} guidance_scale invalid"
            assert style.recommended_steps > 0, f"{name} steps invalid"

    def test_compute_nst_weights_preserves_portraits_more(self):
        style = get_style("cubism")

        default_content, default_style = style.compute_nst_weights()
        portrait_content, portrait_style = style.compute_nst_weights((687, 1023))

        assert portrait_content > default_content
        assert portrait_style < default_style

    def test_compute_nst_weights_can_boost_starry_night_reference(self):
        style = get_style("post_impressionism")

        baseline_content, baseline_style = style.compute_nst_weights((687, 1023))
        starry_content, starry_style = style.compute_nst_weights(
            (687, 1023),
            reference_hint="vangogh starry night",
        )

        assert starry_content > baseline_content
        assert starry_style > baseline_style

    def test_compute_diffusion_tuning_preserves_human_portraits(self):
        style = get_style("cubism")

        tuning = style.compute_diffusion_tuning(
            (687, 1023),
            source_hint="mona lisa",
        )

        assert tuning.strength < style.recommended_strength
        assert tuning.guidance_scale > style.recommended_guidance_scale
        assert tuning.num_inference_steps >= style.recommended_steps
        assert "mona lisa" in tuning.prompt
        assert "recognizable" in tuning.prompt
        assert len(tuning.prompt.split()) < 70

    def test_compute_diffusion_tuning_uses_starry_night_reference(self):
        style = get_style("post_impressionism")

        baseline = style.compute_diffusion_tuning((687, 1023), source_hint="mona lisa")
        starry = style.compute_diffusion_tuning(
            (687, 1023),
            source_hint="mona lisa",
            reference_hint="vangogh starry night",
        )

        assert "starry night" in starry.prompt
        assert "preserve face" in starry.prompt
        assert "split face" in starry.negative_prompt
        assert starry.strength <= 0.22
        assert starry.strength < baseline.strength
        assert starry.num_inference_steps > baseline.num_inference_steps
        assert len(starry.prompt.split()) < len(baseline.prompt.split())
