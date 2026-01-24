# tests/test_ui/test_vocabularies.py
"""Tests for vocabulary definitions and helper functions.

Tests the new dynamic JSON loading architecture where all vocabulary
values are loaded from scripts/encode_facets_raw.json.
"""

from src.ui.vocabularies import (
    ASSAY_ALIASES,
    ASSAY_TYPES,
    BODY_PARTS,
    COMMON_LABS,
    HISTONE_ALIASES,
    HISTONE_MODIFICATIONS,
    LIFE_STAGES,
    ORGAN_DISPLAY_NAMES,
    ORGANISMS,
    TISSUE_SYNONYMS,
    TOP_BIOSAMPLES,
    TOP_TARGETS,
    get_all_assay_types,
    get_all_body_parts,
    get_all_developmental_stages,
    get_all_histone_mods,
    get_all_organisms,
    get_assay_display_name,
    get_assay_types,
    get_biosample_names_for_organ,
    get_biosamples,
    get_biosamples_for_organ,
    get_labs,
    get_life_stages,
    get_organ_display_name,
    get_organ_system_names,
    get_organ_systems,
    get_organism_display,
    get_targets,
    get_tissues_for_body_part,
    get_total_experiments,
)


class TestJSONLoading:
    """Tests for dynamic JSON loading functionality."""

    def test_json_loads_successfully(self) -> None:
        """Test that JSON data loads without errors."""
        # This implicitly tests _load_facets()
        total = get_total_experiments()
        assert total > 0

    def test_assay_types_loaded_from_json(self) -> None:
        """Test that assay types are loaded from JSON."""
        assays = get_assay_types()
        assert isinstance(assays, list)
        assert len(assays) > 0
        # Check structure: list of (name, count) tuples
        name, count = assays[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0

    def test_assay_types_ordered_by_popularity(self) -> None:
        """Test that assay types are ordered by experiment count (descending)."""
        assays = get_assay_types()
        counts = [count for name, count in assays]
        # Counts should be in descending order (most popular first)
        assert counts == sorted(counts, reverse=True)

    def test_chip_seq_is_first(self) -> None:
        """Test that ChIP-seq is the most popular assay type."""
        assays = get_assay_types()
        first_assay = assays[0][0]
        assert first_assay == "ChIP-seq"

    def test_biosamples_loaded_from_json(self) -> None:
        """Test that biosamples are loaded from JSON."""
        biosamples = get_biosamples()
        assert isinstance(biosamples, list)
        assert len(biosamples) > 100  # ENCODE has many biosamples

    def test_targets_loaded_from_json(self) -> None:
        """Test that targets are loaded from JSON."""
        targets = get_targets()
        assert isinstance(targets, list)
        assert len(targets) > 50  # ENCODE has many targets
        # H3K4me3 should be in top targets
        target_names = [name for name, count in targets[:20]]
        assert "H3K4me3" in target_names

    def test_life_stages_loaded_from_json(self) -> None:
        """Test that life stages are loaded from JSON."""
        stages = get_life_stages()
        assert isinstance(stages, list)
        assert len(stages) > 0
        # Check for actual ENCODE life stages
        stage_names = [name for name, count in stages]
        assert "adult" in stage_names
        assert "embryonic" in stage_names

    def test_life_stages_not_fabricated(self) -> None:
        """Test that life stages are real ENCODE values, not fabricated."""
        stages = get_life_stages()
        stage_names = [name for name, count in stages]
        # These fabricated values should NOT be present
        fabricated_stages = ["E10.5", "E14.5", "P0", "P56", "P60"]
        for fake_stage in fabricated_stages:
            assert fake_stage not in stage_names, f"Fabricated stage {fake_stage} found"

    def test_labs_loaded_from_json(self) -> None:
        """Test that labs are loaded from JSON."""
        labs = get_labs()
        assert isinstance(labs, list)
        assert len(labs) > 10  # ENCODE has many labs


class TestAssayTypes:
    """Tests for ASSAY_TYPES dictionary (legacy compatibility)."""

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


class TestDisplayNames:
    """Tests for display name functionality."""

    def test_get_assay_display_name_returns_short_name(self) -> None:
        """Test that long assay names get shortened."""
        long_name = "single-cell RNA sequencing assay"
        display = get_assay_display_name(long_name)
        assert display == "scRNA-seq"

    def test_get_assay_display_name_returns_original_for_unknown(self) -> None:
        """Test that unknown assays return original name."""
        unknown_assay = "ChIP-seq"
        display = get_assay_display_name(unknown_assay)
        assert display == "ChIP-seq"


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
        valid_categories = {
            "active",
            "enhancer",
            "promoter",
            "transcription",
            "repressive",
            "tf",
        }
        for mark_key, mark_info in HISTONE_MODIFICATIONS.items():
            assert (
                mark_info["category"] in valid_categories
            ), f"Invalid category for {mark_key}"

    def test_histone_aliases_reference_valid_marks(self) -> None:
        """Test that all histone aliases reference valid modifications."""
        for mark_key in HISTONE_ALIASES.keys():
            assert (
                mark_key in HISTONE_MODIFICATIONS
            ), f"Alias key {mark_key} not in HISTONE_MODIFICATIONS"


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
            assert isinstance(
                part_info["tissues"], list
            ), f"Tissues for {part_key} should be a list"
            assert (
                len(part_info["tissues"]) > 0
            ), f"Tissues for {part_key} should not be empty"

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
            key
            for key, info in BODY_PARTS.items()
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


class TestLifeStages:
    """Tests for life stages (replacing DEVELOPMENTAL_STAGES)."""

    def test_life_stages_not_empty(self) -> None:
        """Test that LIFE_STAGES contains entries."""
        assert len(LIFE_STAGES) > 0

    def test_real_encode_stages_present(self) -> None:
        """Test that real ENCODE life stages are present."""
        real_stages = ["adult", "embryonic", "child", "newborn"]
        for stage in real_stages:
            assert stage in LIFE_STAGES, f"Real ENCODE stage {stage} not found"

    def test_fabricated_stages_not_present(self) -> None:
        """Test that fabricated developmental stages are NOT present."""
        # These were invented values that don't exist in ENCODE
        fabricated = ["E10.5", "E14.5", "P0", "P7", "P56", "P60", "8 weeks"]
        for stage in fabricated:
            assert (
                stage not in LIFE_STAGES
            ), f"Fabricated stage {stage} should not be in LIFE_STAGES"


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

    def test_get_all_assay_types_returns_list(self) -> None:
        """Test that get_all_assay_types returns a list ordered by popularity."""
        result = get_all_assay_types()
        assert isinstance(result, list)
        assert len(result) == len(ASSAY_TYPES)
        # Lists are ordered by experiment count (popularity), not alphabetically
        # ChIP-seq should be first as it has most experiments
        assert result[0] == "ChIP-seq"

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

    def test_get_all_histone_mods_returns_list(self) -> None:
        """Test that get_all_histone_mods returns a list."""
        result = get_all_histone_mods()
        assert isinstance(result, list)
        assert len(result) == len(HISTONE_MODIFICATIONS)
        # Should contain common histone marks
        assert "H3K27ac" in result
        assert "H3K4me3" in result
        assert "CTCF" in result

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
        """Test that get_all_developmental_stages returns actual ENCODE life stages."""
        result = get_all_developmental_stages()
        assert isinstance(result, list)
        assert len(result) > 0
        # Should contain real ENCODE stages
        assert "adult" in result


class TestVocabularyConsistency:
    """Tests for consistency across vocabularies."""

    def test_tissue_synonyms_are_strings(self) -> None:
        """Test that all synonym values are non-empty strings."""
        for key, synonyms in TISSUE_SYNONYMS.items():
            for syn in synonyms:
                assert isinstance(
                    syn, str
                ), f"Synonym '{syn}' for '{key}' should be a string"
                assert len(syn) > 0, f"Synonym for '{key}' should not be empty"

    def test_no_duplicate_tissues_in_body_part(self) -> None:
        """Test that there are no duplicate tissues within a body part."""
        for part_key, part_info in BODY_PARTS.items():
            tissues = part_info["tissues"]
            tissues_lower = [t.lower() for t in tissues]
            assert len(tissues_lower) == len(
                set(tissues_lower)
            ), f"Duplicate tissues in {part_key}"

    def test_top_biosamples_matches_json(self) -> None:
        """Test that TOP_BIOSAMPLES comes from JSON data."""
        top_biosamples = list(TOP_BIOSAMPLES)
        json_biosamples = get_biosamples()[:50]
        json_names = [name for name, count in json_biosamples]
        assert top_biosamples == json_names

    def test_top_targets_matches_json(self) -> None:
        """Test that TOP_TARGETS comes from JSON data."""
        top_targets = list(TOP_TARGETS)
        json_targets = get_targets()[:40]
        json_names = [name for name, count in json_targets]
        assert top_targets == json_names


class TestOrganSystems:
    """Tests for organ_slims-based functions."""

    def test_get_organ_systems_returns_list(self) -> None:
        """Test that get_organ_systems returns ordered list of tuples."""
        organs = get_organ_systems()
        assert isinstance(organs, list)
        assert len(organs) > 0
        # Should be (name, count) tuples
        name, count = organs[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0

    def test_get_organ_systems_ordered_by_count(self) -> None:
        """Test that organ systems are ordered by experiment count (descending)."""
        organs = get_organ_systems()
        counts = [count for name, count in organs]
        assert counts == sorted(counts, reverse=True)

    def test_get_organ_systems_contains_common_organs(self) -> None:
        """Test that common organs are present."""
        organs = get_organ_systems()
        organ_names = [name for name, count in organs]
        common_organs = ["brain", "heart", "liver", "lung", "kidney", "blood"]
        for organ in common_organs:
            assert organ in organ_names, f"Expected {organ} in organ systems"

    def test_get_organ_system_names(self) -> None:
        """Test that get_organ_system_names returns list of strings."""
        names = get_organ_system_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
        # Should match the names from get_organ_systems
        full_data = get_organ_systems()
        expected_names = [name for name, count in full_data]
        assert names == expected_names

    def test_get_biosamples_for_organ_brain(self) -> None:
        """Test getting biosamples for brain organ."""
        biosamples = get_biosamples_for_organ("brain")
        assert isinstance(biosamples, list)
        # Brain should have multiple biosamples
        assert len(biosamples) > 5
        # Check structure
        name, count = biosamples[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0
        # Common brain tissues should be present
        biosample_names = [name for name, count in biosamples]
        assert any("cortex" in name.lower() for name in biosample_names)

    def test_get_biosamples_for_organ_ordered_by_count(self) -> None:
        """Test that biosamples for an organ are ordered by count."""
        biosamples = get_biosamples_for_organ("brain")
        counts = [count for name, count in biosamples]
        assert counts == sorted(counts, reverse=True)

    def test_get_biosamples_for_invalid_organ(self) -> None:
        """Test that invalid organ returns empty list."""
        biosamples = get_biosamples_for_organ("nonexistent_organ")
        assert biosamples == []

    def test_get_biosample_names_for_organ(self) -> None:
        """Test getting biosample names (without counts) for an organ."""
        names = get_biosample_names_for_organ("heart")
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_get_biosample_names_for_organ_with_limit(self) -> None:
        """Test that limit parameter works correctly."""
        all_names = get_biosample_names_for_organ("brain")
        limited = get_biosample_names_for_organ("brain", limit=5)
        assert len(limited) == 5
        assert limited == all_names[:5]

    def test_get_organ_display_name_known(self) -> None:
        """Test display name for known organ mappings."""
        # "bodily fluid" -> "Blood / Bodily Fluid"
        assert get_organ_display_name("bodily fluid") == "Blood / Bodily Fluid"
        # "musculature of body" -> "Muscle"
        assert get_organ_display_name("musculature of body") == "Muscle"

    def test_get_organ_display_name_unknown(self) -> None:
        """Test display name for organs without custom mapping."""
        # Should title-case and replace underscores
        assert get_organ_display_name("brain") == "Brain"
        assert get_organ_display_name("test_organ") == "Test Organ"

    def test_organ_display_names_not_empty(self) -> None:
        """Test that ORGAN_DISPLAY_NAMES contains entries."""
        assert len(ORGAN_DISPLAY_NAMES) > 0

    def test_multiple_organs_have_biosamples(self) -> None:
        """Test that major organs all have biosample data."""
        organs_to_check = ["brain", "heart", "liver", "lung", "kidney"]
        for organ in organs_to_check:
            biosamples = get_biosamples_for_organ(organ)
            assert len(biosamples) > 0, f"Expected biosamples for {organ}"
