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
    ORGANISM_ASSEMBLIES,
    ORGANISMS,
    TISSUE_SYNONYMS,
    TOP_BIOSAMPLES,
    TOP_TARGETS,
    build_biosample_to_organs,
    get_all_assay_types,
    get_all_body_parts,
    get_all_developmental_stages,
    get_all_histone_mods,
    get_all_organisms,
    get_all_organs_for_biosample,
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
    get_organism_common_name,
    get_organism_display,
    get_organism_names,
    get_organism_scientific_name,
    get_organisms,
    get_primary_organ_for_biosample,
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


class TestDynamicOrganisms:
    """Tests for dynamic organism loading from ENCODE JSON."""

    def test_get_organisms_returns_list_of_tuples(self) -> None:
        """Test that get_organisms returns list of (name, count) tuples."""
        result = get_organisms()
        assert isinstance(result, list)
        assert len(result) > 0
        # Each item should be (scientific_name, count)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], int)

    def test_get_organisms_includes_main_model_organisms(self) -> None:
        """Test that main model organisms are in the dynamic list."""
        organisms = get_organisms()
        sci_names = [name for name, _ in organisms]
        assert "Homo sapiens" in sci_names
        assert "Mus musculus" in sci_names
        assert "Drosophila melanogaster" in sci_names
        assert "Caenorhabditis elegans" in sci_names

    def test_get_organisms_includes_minor_species(self) -> None:
        """Test that minor species from ENCODE are included."""
        organisms = get_organisms()
        sci_names = [name for name, _ in organisms]
        # There should be more than the 4 main model organisms
        assert len(sci_names) > 4

    def test_get_organism_names_returns_strings(self) -> None:
        """Test that get_organism_names returns list of strings."""
        result = get_organism_names()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        assert "Homo sapiens" in result

    def test_get_organism_names_with_limit(self) -> None:
        """Test that limit parameter works."""
        result = get_organism_names(limit=3)
        assert len(result) == 3

    def test_organism_assemblies_structure(self) -> None:
        """Test ORGANISM_ASSEMBLIES has correct structure."""
        assert "Homo sapiens" in ORGANISM_ASSEMBLIES
        assert "Mus musculus" in ORGANISM_ASSEMBLIES

        human = ORGANISM_ASSEMBLIES["Homo sapiens"]
        assert human["common_name"] == "human"
        assert human["short_name"] == "Human"
        assert human["assembly"] == "hg38"

    def test_get_organism_common_name(self) -> None:
        """Test get_organism_common_name function."""
        assert get_organism_common_name("Homo sapiens") == "human"
        assert get_organism_common_name("Mus musculus") == "mouse"
        assert get_organism_common_name("Unknown species") is None

    def test_get_organism_scientific_name(self) -> None:
        """Test get_organism_scientific_name function."""
        # From common name
        assert get_organism_scientific_name("human") == "Homo sapiens"
        assert get_organism_scientific_name("mouse") == "Mus musculus"
        # Scientific name passes through
        assert get_organism_scientific_name("Homo sapiens") == "Homo sapiens"
        # Unknown passes through
        assert get_organism_scientific_name("unknown") == "unknown"


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
        """Test that get_all_organisms returns scientific names from ENCODE."""
        result = get_all_organisms()
        assert isinstance(result, list)
        # Should include all organisms from ENCODE JSON (more than 4 main ones)
        assert len(result) >= 4
        # Returns scientific names, not common names
        assert "Homo sapiens" in result
        assert "Mus musculus" in result

    def test_get_organism_display_known_organism(self) -> None:
        """Test get_organism_display for known organisms."""
        # Test with common name
        human_display = get_organism_display("human")
        assert "Human" in human_display
        assert "hg38" in human_display

        # Test with scientific name
        human_display2 = get_organism_display("Homo sapiens")
        assert "Human" in human_display2
        assert "hg38" in human_display2

        mouse_display = get_organism_display("mouse")
        assert "Mouse" in mouse_display
        assert "mm10" in mouse_display

    def test_get_organism_display_unknown_organism(self) -> None:
        """Test get_organism_display for unknown organisms."""
        # Unknown organisms just return the input
        result = get_organism_display("unknown_organism")
        assert result == "unknown_organism"

        # Scientific names not in assembly dict return as-is
        result2 = get_organism_display("Drosophila simulans")
        assert result2 == "Drosophila simulans"

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


class TestBiosampleToOrganMapping:
    """Tests for biosample-to-organ reverse mapping functions."""

    def test_build_biosample_to_organs_returns_dict(self) -> None:
        """Test that build_biosample_to_organs returns a non-empty dict."""
        mapping = build_biosample_to_organs()
        assert isinstance(mapping, dict)
        assert len(mapping) > 100  # Should have many biosamples mapped

    def test_build_biosample_to_organs_values_are_lists(self) -> None:
        """Test that mapping values are lists of organ names."""
        mapping = build_biosample_to_organs()
        for biosample, organs in list(mapping.items())[:10]:
            assert isinstance(organs, list)
            assert len(organs) > 0
            assert all(isinstance(org, str) for org in organs)

    def test_get_primary_organ_for_known_biosample(self) -> None:
        """Test getting primary organ for a known biosample."""
        # Cerebellum should map to brain
        organ = get_primary_organ_for_biosample("cerebellum")
        assert organ == "brain"

    def test_get_primary_organ_for_k562(self) -> None:
        """Test getting primary organ for K562 cell line."""
        organ = get_primary_organ_for_biosample("K562")
        assert organ == "blood"

    def test_get_primary_organ_for_unknown_biosample(self) -> None:
        """Test that unknown biosample returns None."""
        result = get_primary_organ_for_biosample("nonexistent_biosample_xyz")
        assert result is None

    def test_get_all_organs_for_biosample_single_organ(self) -> None:
        """Test biosample that maps to a single organ."""
        organs = get_all_organs_for_biosample("cerebellum")
        assert isinstance(organs, list)
        assert len(organs) >= 1
        assert "brain" in organs

    def test_get_all_organs_for_biosample_multiple_organs(self) -> None:
        """Test biosample that maps to multiple organs."""
        # Some biosamples map to multiple organs
        mapping = build_biosample_to_organs()
        # Find a biosample with multiple organs
        multi_organ_sample = None
        for biosample, organs in mapping.items():
            if len(organs) > 1:
                multi_organ_sample = biosample
                break
        assert multi_organ_sample is not None, "Should have at least one multi-organ biosample"
        organs = get_all_organs_for_biosample(multi_organ_sample)
        assert len(organs) > 1

    def test_get_all_organs_for_unknown_biosample(self) -> None:
        """Test that unknown biosample returns empty list."""
        organs = get_all_organs_for_biosample("nonexistent_sample")
        assert organs == []

    def test_organs_ordered_by_experiment_count(self) -> None:
        """Test that organs are ordered by experiment count (most popular first)."""
        # Get a biosample that maps to multiple organs
        mapping = build_biosample_to_organs()
        organ_counts = {
            name: count for name, count in get_organ_systems()
        }

        for biosample, organs in list(mapping.items())[:20]:
            if len(organs) > 1:
                # Check that organs are ordered by experiment count
                organ_experiment_counts = [organ_counts.get(org, 0) for org in organs]
                assert organ_experiment_counts == sorted(
                    organ_experiment_counts, reverse=True
                ), f"Organs for {biosample} not ordered by count"

    def test_all_biosamples_in_organ_mapping_can_be_reversed(self) -> None:
        """Test that biosamples from organ mapping appear in reverse mapping."""
        # Get some biosamples from an organ
        brain_biosamples = get_biosamples_for_organ("brain")[:10]
        mapping = build_biosample_to_organs()

        for name, _ in brain_biosamples:
            assert name in mapping, f"Biosample {name} should be in reverse mapping"
            assert "brain" in mapping[name], f"Brain should be in organs for {name}"
