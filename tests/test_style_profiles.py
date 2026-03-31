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
