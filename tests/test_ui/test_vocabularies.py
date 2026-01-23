# tests/test_ui/test_vocabularies.py
"""Tests for vocabulary definitions and helper functions."""

import pytest

from src.ui.vocabularies import (
    AGE_ALIASES,
    ASSAY_ALIASES,
    ASSAY_TYPES,
    BODY_PARTS,
    COMMON_LABS,
    DEVELOPMENTAL_STAGES,
    HISTONE_ALIASES,
    HISTONE_MODIFICATIONS,
    ORGANISMS,
    TISSUE_SYNONYMS,
    get_all_assay_types,
    get_all_body_parts,
    get_all_developmental_stages,
    get_all_histone_mods,
    get_all_organisms,
    get_organism_display,
    get_tissues_for_body_part,
)


class TestAssayTypes:
    """Tests for ASSAY_TYPES dictionary."""

    def test_assay_types_not_empty(self) -> None:
        """Test that ASSAY_TYPES contains entries."""
        assert len(ASSAY_TYPES) > 0

    def test_assay_types_contains_common_assays(self) -> None:
        """Test that common ENCODE assays are present."""
        # Note: ENCODE uses "HiC" not "Hi-C" as the canonical spelling
        common_assays = ["ChIP-seq", "RNA-seq", "ATAC-seq", "DNase-seq", "HiC"]
        for assay in common_assays:
            assert assay in ASSAY_TYPES

    def test_hic_variants_present(self) -> None:
        """Test that HiC-related assays are present."""
        # Note: ENCODE uses "HiC" (not "Hi-C") and "capture Hi-C" (not "in situ Hi-C")
        hic_variants = ["HiC", "capture Hi-C"]
        for variant in hic_variants:
            assert variant in ASSAY_TYPES

    def test_assay_types_have_display_names(self) -> None:
        """Test that all assay types have non-empty display names."""
        for key, display in ASSAY_TYPES.items():
            assert display, f"Empty display name for {key}"
            assert isinstance(display, str)

    def test_assay_aliases_reference_valid_assays(self) -> None:
        """Test that all aliases reference valid assay types."""
        for assay_key in ASSAY_ALIASES.keys():
            assert assay_key in ASSAY_TYPES, f"Alias key {assay_key} not in ASSAY_TYPES"

    def test_assay_aliases_are_lists(self) -> None:
        """Test that all alias values are non-empty lists of strings."""
        for key, aliases in ASSAY_ALIASES.items():
            assert isinstance(aliases, list), f"Aliases for {key} should be a list"
            assert len(aliases) > 0, f"Aliases for {key} should not be empty"
            for alias in aliases:
                assert isinstance(alias, str), f"Alias {alias} should be a string"


class TestOrganisms:
    """Tests for ORGANISMS dictionary."""

    def test_organisms_not_empty(self) -> None:
        """Test that ORGANISMS contains entries."""
        assert len(ORGANISMS) > 0

    def test_organisms_contains_human_and_mouse(self) -> None:
        """Test that human and mouse are present."""
        assert "human" in ORGANISMS
        assert "mouse" in ORGANISMS

    def test_organisms_have_required_fields(self) -> None:
        """Test that all organisms have required fields."""
        required_fields = ["display_name", "scientific_name", "assembly"]
        for org_key, org_info in ORGANISMS.items():
            for field in required_fields:
                assert field in org_info, f"Missing {field} for {org_key}"

    def test_human_has_correct_assembly(self) -> None:
        """Test that human has hg38 assembly."""
        assert ORGANISMS["human"]["assembly"] == "hg38"

    def test_mouse_has_correct_assembly(self) -> None:
        """Test that mouse has mm10 assembly."""
        assert ORGANISMS["mouse"]["assembly"] == "mm10"


class TestHistoneModifications:
    """Tests for HISTONE_MODIFICATIONS dictionary."""

    def test_histone_mods_not_empty(self) -> None:
        """Test that HISTONE_MODIFICATIONS contains entries."""
        assert len(HISTONE_MODIFICATIONS) > 0

    def test_common_histone_marks_present(self) -> None:
        """Test that common histone marks are present."""
        common_marks = ["H3K27ac", "H3K4me3", "H3K4me1", "H3K27me3", "CTCF"]
        for mark in common_marks:
            assert mark in HISTONE_MODIFICATIONS

    def test_histone_mods_have_required_fields(self) -> None:
        """Test that all histone mods have required fields."""
        required_fields = ["full_name", "description", "category"]
        for mark_key, mark_info in HISTONE_MODIFICATIONS.items():
            for field in required_fields:
                assert field in mark_info, f"Missing {field} for {mark_key}"

    def test_histone_categories_are_valid(self) -> None:
        """Test that histone modification categories are valid."""
        valid_categories = {"active", "enhancer", "promoter", "transcription", "repressive", "tf"}
        for mark_key, mark_info in HISTONE_MODIFICATIONS.items():
            assert mark_info["category"] in valid_categories, f"Invalid category for {mark_key}"

    def test_histone_aliases_reference_valid_marks(self) -> None:
        """Test that all histone aliases reference valid modifications."""
        for mark_key in HISTONE_ALIASES.keys():
            assert mark_key in HISTONE_MODIFICATIONS, f"Alias key {mark_key} not in HISTONE_MODIFICATIONS"


class TestBodyParts:
    """Tests for BODY_PARTS dictionary."""

    def test_body_parts_not_empty(self) -> None:
        """Test that BODY_PARTS contains entries."""
        assert len(BODY_PARTS) > 0

    def test_common_body_parts_present(self) -> None:
        """Test that common body parts are present."""
        common_parts = ["brain", "heart", "liver", "kidney", "lung", "blood"]
        for part in common_parts:
            assert part in BODY_PARTS

    def test_body_parts_have_required_fields(self) -> None:
        """Test that all body parts have required fields."""
        for part_key, part_info in BODY_PARTS.items():
            assert "display_name" in part_info, f"Missing display_name for {part_key}"
            assert "tissues" in part_info, f"Missing tissues for {part_key}"
            assert isinstance(part_info["tissues"], list), f"Tissues for {part_key} should be a list"
            assert len(part_info["tissues"]) > 0, f"Tissues for {part_key} should not be empty"

    def test_brain_contains_cerebellum(self) -> None:
        """Test that brain body part includes cerebellum."""
        assert "cerebellum" in BODY_PARTS["brain"]["tissues"]

    def test_cell_line_body_part_contains_k562(self) -> None:
        """Test that cell_line body part includes K562."""
        assert "K562" in BODY_PARTS["cell_line"]["tissues"]

    def test_body_parts_have_aliases(self) -> None:
        """Test that most body parts have aliases."""
        # At least some body parts should have aliases
        parts_with_aliases = [
            key for key, info in BODY_PARTS.items()
            if "aliases" in info and len(info.get("aliases", [])) > 0
        ]
        assert len(parts_with_aliases) > 5, "Most body parts should have aliases"


class TestTissueSynonyms:
    """Tests for TISSUE_SYNONYMS dictionary."""

    def test_tissue_synonyms_not_empty(self) -> None:
        """Test that TISSUE_SYNONYMS contains entries."""
        assert len(TISSUE_SYNONYMS) > 0

    def test_cerebellum_hindbrain_synonyms(self) -> None:
        """Test that cerebellum and hindbrain are synonyms."""
        assert "hindbrain" in TISSUE_SYNONYMS["cerebellum"]
        assert "cerebellum" in TISSUE_SYNONYMS["hindbrain"]

    def test_synonyms_are_sets(self) -> None:
        """Test that all synonym values are sets."""
        for key, synonyms in TISSUE_SYNONYMS.items():
            assert isinstance(synonyms, set), f"Synonyms for {key} should be a set"


class TestDevelopmentalStages:
    """Tests for DEVELOPMENTAL_STAGES dictionary."""

    def test_developmental_stages_not_empty(self) -> None:
        """Test that DEVELOPMENTAL_STAGES contains entries."""
        assert len(DEVELOPMENTAL_STAGES) > 0

    def test_mouse_stages_present(self) -> None:
        """Test that mouse developmental stages are present."""
        mouse_stages = ["E10.5", "E14.5", "P0", "P7", "P56"]
        for stage in mouse_stages:
            assert stage in DEVELOPMENTAL_STAGES

    def test_human_stages_present(self) -> None:
        """Test that human developmental stages are present."""
        human_stages = ["embryonic", "fetal", "newborn", "adult"]
        for stage in human_stages:
            assert stage in DEVELOPMENTAL_STAGES

    def test_stages_have_required_fields(self) -> None:
        """Test that all stages have required fields."""
        for stage_key, stage_info in DEVELOPMENTAL_STAGES.items():
            assert "species" in stage_info, f"Missing species for {stage_key}"
            assert "description" in stage_info, f"Missing description for {stage_key}"
            assert "days" in stage_info, f"Missing days for {stage_key}"

    def test_age_aliases_reference_valid_stages(self) -> None:
        """Test that all age aliases reference valid stages."""
        for stage_key in AGE_ALIASES.keys():
            assert stage_key in DEVELOPMENTAL_STAGES, f"Alias key {stage_key} not in DEVELOPMENTAL_STAGES"


class TestCommonLabs:
    """Tests for COMMON_LABS list."""

    def test_common_labs_not_empty(self) -> None:
        """Test that COMMON_LABS contains entries."""
        assert len(COMMON_LABS) > 0

    def test_common_labs_are_strings(self) -> None:
        """Test that all labs are non-empty strings."""
        for lab in COMMON_LABS:
            assert isinstance(lab, str)
            assert len(lab) > 0


class TestHelperFunctions:
    """Tests for vocabulary helper functions."""

    def test_get_all_assay_types_returns_sorted_list(self) -> None:
        """Test that get_all_assay_types returns a sorted list."""
        result = get_all_assay_types()
        assert isinstance(result, list)
        assert len(result) == len(ASSAY_TYPES)
        assert result == sorted(result)

    def test_get_all_organisms_returns_list(self) -> None:
        """Test that get_all_organisms returns a list."""
        result = get_all_organisms()
        assert isinstance(result, list)
        assert len(result) == len(ORGANISMS)
        assert "human" in result
        assert "mouse" in result

    def test_get_organism_display_known_organism(self) -> None:
        """Test get_organism_display for known organisms."""
        human_display = get_organism_display("human")
        assert "Human" in human_display
        assert "hg38" in human_display

        mouse_display = get_organism_display("mouse")
        assert "Mouse" in mouse_display
        assert "mm10" in mouse_display

    def test_get_organism_display_unknown_organism(self) -> None:
        """Test get_organism_display for unknown organisms."""
        result = get_organism_display("unknown_organism")
        assert result == "unknown_organism"

    def test_get_all_histone_mods_returns_sorted_list(self) -> None:
        """Test that get_all_histone_mods returns a sorted list."""
        result = get_all_histone_mods()
        assert isinstance(result, list)
        assert len(result) == len(HISTONE_MODIFICATIONS)
        assert result == sorted(result)

    def test_get_all_body_parts_returns_list(self) -> None:
        """Test that get_all_body_parts returns a list."""
        result = get_all_body_parts()
        assert isinstance(result, list)
        assert len(result) == len(BODY_PARTS)
        assert "brain" in result

    def test_get_tissues_for_body_part_valid(self) -> None:
        """Test get_tissues_for_body_part for valid body part."""
        brain_tissues = get_tissues_for_body_part("brain")
        assert isinstance(brain_tissues, list)
        assert len(brain_tissues) > 0
        assert "cerebellum" in brain_tissues

    def test_get_tissues_for_body_part_invalid(self) -> None:
        """Test get_tissues_for_body_part for invalid body part."""
        result = get_tissues_for_body_part("invalid_body_part")
        assert result == []

    def test_get_all_developmental_stages_returns_list(self) -> None:
        """Test that get_all_developmental_stages returns a list."""
        result = get_all_developmental_stages()
        assert isinstance(result, list)
        assert len(result) == len(DEVELOPMENTAL_STAGES)
        assert "P0" in result


class TestVocabularyConsistency:
    """Tests for consistency across vocabularies."""

    def test_all_tissue_synonyms_lowercase(self) -> None:
        """Test that all synonym values are lowercase."""
        for key, synonyms in TISSUE_SYNONYMS.items():
            for syn in synonyms:
                assert syn == syn.lower(), f"Synonym '{syn}' for '{key}' should be lowercase"

    def test_no_duplicate_tissues_in_body_part(self) -> None:
        """Test that there are no duplicate tissues within a body part."""
        for part_key, part_info in BODY_PARTS.items():
            tissues = part_info["tissues"]
            tissues_lower = [t.lower() for t in tissues]
            assert len(tissues_lower) == len(set(tissues_lower)), f"Duplicate tissues in {part_key}"

    def test_developmental_stages_days_are_numeric(self) -> None:
        """Test that all developmental stage days are numeric."""
        for stage_key, stage_info in DEVELOPMENTAL_STAGES.items():
            days = stage_info["days"]
            assert isinstance(days, (int, float)), f"Days for {stage_key} should be numeric"
            assert days >= 0, f"Days for {stage_key} should be non-negative"
