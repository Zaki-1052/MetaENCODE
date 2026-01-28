# tests/test_utils/test_spell_check.py
"""Tests for spell checking module."""

import pytest

# Check if spell check dependencies are available
try:
    import symspellpy
    import jellyfish

    SPELL_CHECK_AVAILABLE = True
except ImportError:
    SPELL_CHECK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SPELL_CHECK_AVAILABLE,
    reason="Spell check dependencies (symspellpy, jellyfish) not installed",
)


# =============================================================================
# SpellingSuggestion Tests
# =============================================================================


class TestSpellingSuggestion:
    """Tests for SpellingSuggestion dataclass."""

    def test_suggestion_creation(self) -> None:
        """Test creating a SpellingSuggestion."""
        from src.utils.spell_check import SpellingSuggestion

        suggestion = SpellingSuggestion(
            term="cerebellum",
            distance=1,
            confidence=0.95,
            frequency=1234,
            phonetic_match=True,
            category="biosample",
        )
        assert suggestion.term == "cerebellum"
        assert suggestion.distance == 1
        assert suggestion.confidence == 0.95
        assert suggestion.frequency == 1234
        assert suggestion.phonetic_match is True
        assert suggestion.category == "biosample"

    def test_suggestion_sorting(self) -> None:
        """Test that suggestions sort by confidence then frequency."""
        from src.utils.spell_check import SpellingSuggestion

        s1 = SpellingSuggestion(
            term="a", distance=1, confidence=0.9, frequency=100, phonetic_match=False
        )
        s2 = SpellingSuggestion(
            term="b", distance=1, confidence=0.95, frequency=50, phonetic_match=False
        )
        s3 = SpellingSuggestion(
            term="c", distance=1, confidence=0.9, frequency=200, phonetic_match=False
        )

        sorted_suggestions = sorted([s1, s2, s3])
        # Highest confidence first
        assert sorted_suggestions[0].term == "b"
        # Then higher frequency
        assert sorted_suggestions[1].term == "c"
        assert sorted_suggestions[2].term == "a"


# =============================================================================
# VocabularySpellChecker Tests
# =============================================================================


class TestVocabularySpellCheckerInit:
    """Tests for VocabularySpellChecker initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        assert checker.min_query_length == 3
        assert checker.max_edit_distance == 2
        assert checker.prefix_length == 7

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(
            min_query_length=4, max_edit_distance=3, prefix_length=5
        )
        assert checker.min_query_length == 4
        assert checker.max_edit_distance == 3
        assert checker.prefix_length == 5


class TestVocabularySpellCheckerAddTerm:
    """Tests for VocabularySpellChecker.add_term() method."""

    def test_add_single_term(self) -> None:
        """Test adding a single term."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", frequency=1000, category="biosample")
        assert checker.is_valid_term("cerebellum")
        assert checker.is_valid_term("Cerebellum")  # Case insensitive

    def test_add_term_short_term_ignored(self) -> None:
        """Test that very short terms are ignored."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("a", frequency=100)
        assert not checker.is_valid_term("a")

    def test_add_term_updates_frequency(self) -> None:
        """Test that adding same term updates frequency if higher."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", frequency=100)
        checker.add_term("test", frequency=200)
        # Second add should update frequency
        assert checker._vocabulary["test"].frequency == 200

    def test_add_term_keeps_higher_frequency(self) -> None:
        """Test that adding with lower frequency doesn't downgrade."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", frequency=200)
        checker.add_term("test", frequency=100)
        # Should keep higher frequency
        assert checker._vocabulary["test"].frequency == 200


class TestVocabularySpellCheckerAddTerms:
    """Tests for VocabularySpellChecker.add_terms() method."""

    def test_add_multiple_terms(self) -> None:
        """Test adding multiple terms at once."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        terms = [("cerebellum", 1000), ("hippocampus", 800), ("cortex", 1200)]
        checker.add_terms(terms, category="biosample")

        assert checker.is_valid_term("cerebellum")
        assert checker.is_valid_term("hippocampus")
        assert checker.is_valid_term("cortex")


class TestVocabularySpellCheckerSuggest:
    """Tests for VocabularySpellChecker.suggest() method."""

    @pytest.fixture
    def checker(self):
        """Create a spell checker with test vocabulary."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        # Add some biological terms
        terms = [
            ("cerebellum", 1000),
            ("cerebrum", 500),
            ("hippocampus", 800),
            ("hypothalamus", 600),
            ("cortex", 1200),
            ("liver", 900),
            ("kidney", 850),
            ("H3K27ac", 2000),
            ("H3K4me3", 1800),
            ("CTCF", 1500),
        ]
        checker.add_terms(terms, category="test")
        return checker

    def test_suggest_exact_match(self, checker) -> None:
        """Test that exact matches return the term."""
        suggestions = checker.suggest("cerebellum", include_exact=True)
        assert len(suggestions) >= 1
        assert suggestions[0].term == "cerebellum"
        assert suggestions[0].distance == 0
        assert suggestions[0].confidence == 1.0

    def test_suggest_exact_match_excluded(self, checker) -> None:
        """Test that exact matches can be excluded."""
        suggestions = checker.suggest("cerebellum", include_exact=False)
        assert len(suggestions) == 0

    def test_suggest_typo_cerebelum(self, checker) -> None:
        """Test correction of 'cerebelum' -> 'cerebellum'."""
        suggestions = checker.suggest("cerebelum")
        assert len(suggestions) >= 1
        # Should suggest cerebellum with high confidence
        cerebellum_sugg = next(
            (s for s in suggestions if s.term == "cerebellum"), None
        )
        assert cerebellum_sugg is not None
        assert cerebellum_sugg.distance == 1
        assert cerebellum_sugg.confidence >= 0.8

    def test_suggest_typo_hipocampus(self, checker) -> None:
        """Test correction of 'hipocampus' -> 'hippocampus'."""
        suggestions = checker.suggest("hipocampus")
        assert len(suggestions) >= 1
        hippocampus_sugg = next(
            (s for s in suggestions if s.term == "hippocampus"), None
        )
        assert hippocampus_sugg is not None
        assert hippocampus_sugg.distance == 1

    def test_suggest_histone_mark(self, checker) -> None:
        """Test suggestions for histone mark typos."""
        suggestions = checker.suggest("H3K27a")  # Missing 'c'
        assert len(suggestions) >= 1
        h3k27ac_sugg = next((s for s in suggestions if s.term == "H3K27ac"), None)
        assert h3k27ac_sugg is not None

    def test_suggest_short_query_ignored(self, checker) -> None:
        """Test that very short queries return empty list."""
        suggestions = checker.suggest("ab")
        assert len(suggestions) == 0

    def test_suggest_empty_query(self, checker) -> None:
        """Test that empty query returns empty list."""
        suggestions = checker.suggest("")
        assert len(suggestions) == 0

    def test_suggest_no_match(self, checker) -> None:
        """Test query with no reasonable matches."""
        suggestions = checker.suggest("xyzabc123")
        # Should return empty or low-confidence results
        high_confidence = [s for s in suggestions if s.confidence >= 0.7]
        assert len(high_confidence) == 0

    def test_suggest_max_suggestions(self, checker) -> None:
        """Test that max_suggestions is respected."""
        suggestions = checker.suggest("c", max_suggestions=2)
        # Short query so likely no results, but if any, max 2
        assert len(suggestions) <= 2

    def test_suggest_protected_suffix_ignored(self, checker) -> None:
        """Test that protected suffixes like '-seq' are not corrected."""
        suggestions = checker.suggest("-seq")
        assert len(suggestions) == 0


class TestVocabularySpellCheckerCorrect:
    """Tests for VocabularySpellChecker.correct() method."""

    @pytest.fixture
    def checker(self):
        """Create a spell checker with test vocabulary."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        terms = [
            ("cerebellum", 1000),
            ("hippocampus", 800),
            ("cortex", 1200),
        ]
        checker.add_terms(terms, category="test")
        return checker

    def test_correct_typo(self, checker) -> None:
        """Test correcting a typo."""
        corrected = checker.correct("cerebelum")
        assert corrected == "cerebellum"

    def test_correct_valid_term(self, checker) -> None:
        """Test that valid terms are returned unchanged."""
        corrected = checker.correct("cerebellum")
        assert corrected == "cerebellum"

    def test_correct_no_match(self, checker) -> None:
        """Test that unknown terms are returned unchanged."""
        corrected = checker.correct("xyzabc123")
        assert corrected == "xyzabc123"


class TestVocabularySpellCheckerIsValidTerm:
    """Tests for VocabularySpellChecker.is_valid_term() method."""

    def test_is_valid_term_exists(self) -> None:
        """Test is_valid_term for existing term."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", 1000)
        assert checker.is_valid_term("cerebellum") is True

    def test_is_valid_term_case_insensitive(self) -> None:
        """Test that is_valid_term is case insensitive."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("Cerebellum", 1000)
        assert checker.is_valid_term("cerebellum") is True
        assert checker.is_valid_term("CEREBELLUM") is True

    def test_is_valid_term_not_exists(self) -> None:
        """Test is_valid_term for non-existing term."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        assert checker.is_valid_term("notaword") is False


class TestVocabularySpellCheckerFromEncode:
    """Tests for VocabularySpellChecker.from_encode_vocabularies() factory."""

    def test_from_encode_vocabularies(self) -> None:
        """Test creating spell checker from ENCODE vocabularies."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker.from_encode_vocabularies()

        # Should have biosamples
        assert checker.is_valid_term("cerebellum") or checker.is_valid_term("K562")

        # Should have targets
        assert checker.is_valid_term("H3K27ac") or checker.is_valid_term("CTCF")

        # Should have assays
        assert checker.is_valid_term("ChIP-seq") or checker.is_valid_term("RNA-seq")


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_spell_checker_singleton(self) -> None:
        """Test that get_spell_checker returns singleton."""
        from src.utils.spell_check import get_spell_checker

        checker1 = get_spell_checker()
        checker2 = get_spell_checker()
        assert checker1 is checker2

    def test_suggest_correction(self) -> None:
        """Test suggest_correction convenience function."""
        from src.utils.spell_check import suggest_correction

        # Should return suggestions for a typo
        suggestions = suggest_correction("cerebelum")
        # May or may not have suggestions depending on vocabulary
        assert isinstance(suggestions, list)

    def test_correct_spelling(self) -> None:
        """Test correct_spelling convenience function."""
        from src.utils.spell_check import correct_spelling

        # Should return corrected term or original
        corrected = correct_spelling("cerebelum")
        assert isinstance(corrected, str)
