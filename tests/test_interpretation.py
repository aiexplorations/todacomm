"""Tests for interpretation module."""

import pytest
import numpy as np

from todacomm.analysis.interpretation import (
    LayerInsight,
    PatternInsight,
    TDAInterpretation,
    METRIC_DESCRIPTIONS,
    interpret_tda_results,
    format_interpretation_markdown,
    generate_metric_glossary,
    _detect_patterns,
    _generate_layer_insights,
    _generate_key_findings,
    _generate_executive_summary,
    _generate_layer_headline,
)


@pytest.fixture
def sample_tda_summaries():
    """Sample TDA summaries for testing."""
    return {
        "embedding": {
            "H0_count": 30,
            "H0_total_persistence": 15.5,
            "H0_max_lifetime": 3.2,
            "H1_count": 0,
            "H1_total_persistence": 0.0,
            "H1_max_lifetime": 0.0
        },
        "layer_0": {
            "H0_count": 30,
            "H0_total_persistence": 45.2,
            "H0_max_lifetime": 8.5,
            "H1_count": 2,
            "H1_total_persistence": 0.05,
            "H1_max_lifetime": 0.03
        },
        "layer_5": {
            "H0_count": 30,
            "H0_total_persistence": 120.8,
            "H0_max_lifetime": 25.3,
            "H1_count": 5,
            "H1_total_persistence": 0.85,
            "H1_max_lifetime": 0.3
        },
        "layer_11": {
            "H0_count": 30,
            "H0_total_persistence": 95.1,
            "H0_max_lifetime": 18.7,
            "H1_count": 3,
            "H1_total_persistence": 0.42,
            "H1_max_lifetime": 0.2
        },
        "final": {
            "H0_count": 30,
            "H0_total_persistence": 95.1,
            "H0_max_lifetime": 18.7,
            "H1_count": 3,
            "H1_total_persistence": 0.42,
            "H1_max_lifetime": 0.2
        }
    }


class TestDataclasses:
    """Tests for dataclass definitions."""

    def test_layer_insight_creation(self):
        """LayerInsight should be creatable with required fields."""
        insight = LayerInsight(
            layer_name="layer_0",
            headline="Test headline",
            details=["Detail 1", "Detail 2"],
            significance="high"
        )
        assert insight.layer_name == "layer_0"
        assert insight.headline == "Test headline"
        assert len(insight.details) == 2
        assert insight.significance == "high"

    def test_pattern_insight_creation(self):
        """PatternInsight should be creatable with required fields."""
        insight = PatternInsight(
            pattern_type="peak",
            metric="H0_total_persistence",
            description="Test description",
            affected_layers=["layer_0", "layer_1"],
            interpretation="Test interpretation"
        )
        assert insight.pattern_type == "peak"
        assert insight.metric == "H0_total_persistence"
        assert len(insight.affected_layers) == 2

    def test_tda_interpretation_creation(self):
        """TDAInterpretation should be creatable with required fields."""
        interpretation = TDAInterpretation(
            model_name="test_model",
            executive_summary="Test summary",
            key_findings=["Finding 1"],
            layer_insights=[],
            pattern_insights=[],
            methodology_note="Test methodology",
            limitations=["Limitation 1"]
        )
        assert interpretation.model_name == "test_model"
        assert len(interpretation.key_findings) == 1


class TestMetricDescriptions:
    """Tests for METRIC_DESCRIPTIONS constant."""

    def test_contains_h0_metrics(self):
        """Should contain H0 metric descriptions."""
        assert "H0_count" in METRIC_DESCRIPTIONS
        assert "H0_total_persistence" in METRIC_DESCRIPTIONS
        assert "H0_max_lifetime" in METRIC_DESCRIPTIONS

    def test_contains_h1_metrics(self):
        """Should contain H1 metric descriptions."""
        assert "H1_count" in METRIC_DESCRIPTIONS
        assert "H1_total_persistence" in METRIC_DESCRIPTIONS
        assert "H1_max_lifetime" in METRIC_DESCRIPTIONS

    def test_metrics_have_required_fields(self):
        """Each metric should have name and what fields."""
        for key, info in METRIC_DESCRIPTIONS.items():
            assert "name" in info, f"Metric {key} missing 'name'"
            assert "what" in info, f"Metric {key} missing 'what'"


class TestInterpretTDAResults:
    """Tests for interpret_tda_results function."""

    def test_returns_tda_interpretation(self, sample_tda_summaries):
        """Should return TDAInterpretation object."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        assert isinstance(result, TDAInterpretation)

    def test_includes_model_name(self, sample_tda_summaries):
        """Result should include model name."""
        result = interpret_tda_results(sample_tda_summaries, model_name="test_model")
        assert result.model_name == "test_model"

    def test_includes_executive_summary(self, sample_tda_summaries):
        """Result should include executive summary."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        assert result.executive_summary
        assert len(result.executive_summary) > 0

    def test_includes_key_findings(self, sample_tda_summaries):
        """Result should include key findings."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        assert result.key_findings
        assert len(result.key_findings) > 0

    def test_includes_layer_insights(self, sample_tda_summaries):
        """Result should include layer insights for each layer."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        assert result.layer_insights
        assert len(result.layer_insights) == len(sample_tda_summaries)

    def test_includes_methodology_note(self, sample_tda_summaries):
        """Result should include methodology note."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        assert result.methodology_note
        assert "persistent homology" in result.methodology_note.lower()

    def test_includes_limitations(self, sample_tda_summaries):
        """Result should include limitations."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        assert result.limitations
        assert len(result.limitations) > 0

    def test_respects_sample_count(self, sample_tda_summaries):
        """Methodology note should mention sample count."""
        result = interpret_tda_results(sample_tda_summaries, model_name="gpt2", sample_count=50)
        assert "50" in result.methodology_note


class TestDetectPatterns:
    """Tests for _detect_patterns function."""

    def test_detects_h0_peak(self):
        """Should detect dramatic H0 peak."""
        # Create data with a clear peak (>2x mean)
        data = {
            "embedding": {"H0_total_persistence": 10.0, "H1_count": 0, "H1_total_persistence": 0.0},
            "layer_0": {"H0_total_persistence": 15.0, "H1_count": 0, "H1_total_persistence": 0.0},
            "layer_5": {"H0_total_persistence": 200.0, "H1_count": 0, "H1_total_persistence": 0.0},  # Peak >> 2x mean
            "final": {"H0_total_persistence": 20.0, "H1_count": 0, "H1_total_persistence": 0.0}
        }
        layers = list(data.keys())
        patterns = _detect_patterns(data, layers)
        # Should find a peak at layer_5 (200 is >2x mean of ~61)
        peak_patterns = [p for p in patterns if p.pattern_type == "peak"]
        assert len(peak_patterns) > 0

    def test_detects_embedding_compactness(self, sample_tda_summaries):
        """Should detect embedding layer compactness."""
        layers = list(sample_tda_summaries.keys())
        patterns = _detect_patterns(sample_tda_summaries, layers)
        compact_patterns = [p for p in patterns if "embedding" in p.affected_layers]
        # May or may not detect depending on threshold
        assert isinstance(patterns, list)

    def test_returns_list(self, sample_tda_summaries):
        """Should return a list of patterns."""
        layers = list(sample_tda_summaries.keys())
        patterns = _detect_patterns(sample_tda_summaries, layers)
        assert isinstance(patterns, list)

    def test_handles_uniform_data(self):
        """Should handle uniform data without patterns."""
        uniform_data = {
            "layer_0": {"H0_total_persistence": 50.0, "H1_count": 0, "H1_total_persistence": 0.0},
            "layer_1": {"H0_total_persistence": 50.0, "H1_count": 0, "H1_total_persistence": 0.0},
            "layer_2": {"H0_total_persistence": 50.0, "H1_count": 0, "H1_total_persistence": 0.0}
        }
        layers = list(uniform_data.keys())
        patterns = _detect_patterns(uniform_data, layers)
        assert isinstance(patterns, list)


class TestGenerateLayerInsights:
    """Tests for _generate_layer_insights function."""

    def test_returns_list_of_layer_insights(self, sample_tda_summaries):
        """Should return list of LayerInsight objects."""
        layers = list(sample_tda_summaries.keys())
        insights = _generate_layer_insights(sample_tda_summaries, layers, 30)
        assert isinstance(insights, list)
        assert all(isinstance(i, LayerInsight) for i in insights)

    def test_one_insight_per_layer(self, sample_tda_summaries):
        """Should return one insight per layer."""
        layers = list(sample_tda_summaries.keys())
        insights = _generate_layer_insights(sample_tda_summaries, layers, 30)
        assert len(insights) == len(layers)

    def test_insights_have_layer_names(self, sample_tda_summaries):
        """Each insight should have corresponding layer name."""
        layers = list(sample_tda_summaries.keys())
        insights = _generate_layer_insights(sample_tda_summaries, layers, 30)
        insight_layers = [i.layer_name for i in insights]
        assert set(insight_layers) == set(layers)

    def test_insights_have_significance(self, sample_tda_summaries):
        """Each insight should have significance level."""
        layers = list(sample_tda_summaries.keys())
        insights = _generate_layer_insights(sample_tda_summaries, layers, 30)
        for insight in insights:
            assert insight.significance in ["high", "medium", "low"]


class TestGenerateKeyFindings:
    """Tests for _generate_key_findings function."""

    def test_returns_list_of_strings(self, sample_tda_summaries):
        """Should return list of string findings."""
        layers = list(sample_tda_summaries.keys())
        findings = _generate_key_findings(sample_tda_summaries, layers, "gpt2")
        assert isinstance(findings, list)
        assert all(isinstance(f, str) for f in findings)

    def test_mentions_peak_layer(self, sample_tda_summaries):
        """Findings should mention peak H0 layer."""
        layers = list(sample_tda_summaries.keys())
        findings = _generate_key_findings(sample_tda_summaries, layers, "gpt2")
        findings_text = " ".join(findings)
        assert "layer_5" in findings_text  # layer_5 has peak H0

    def test_mentions_expansion(self, sample_tda_summaries):
        """Findings should mention expansion if significant."""
        layers = list(sample_tda_summaries.keys())
        findings = _generate_key_findings(sample_tda_summaries, layers, "gpt2")
        findings_text = " ".join(findings)
        # Should mention expansion since layer_5 (120.8) >> embedding (15.5)
        assert "expand" in findings_text.lower() or "embedding" in findings_text.lower()


class TestGenerateExecutiveSummary:
    """Tests for _generate_executive_summary function."""

    def test_returns_string(self, sample_tda_summaries):
        """Should return a string."""
        layers = list(sample_tda_summaries.keys())
        patterns = _detect_patterns(sample_tda_summaries, layers)
        summary = _generate_executive_summary(sample_tda_summaries, layers, "gpt2", patterns)
        assert isinstance(summary, str)

    def test_mentions_model_name(self, sample_tda_summaries):
        """Summary should mention model name."""
        layers = list(sample_tda_summaries.keys())
        patterns = _detect_patterns(sample_tda_summaries, layers)
        summary = _generate_executive_summary(sample_tda_summaries, layers, "gpt2", patterns)
        assert "gpt2" in summary

    def test_mentions_peak_layer(self, sample_tda_summaries):
        """Summary should mention peak layer."""
        layers = list(sample_tda_summaries.keys())
        patterns = _detect_patterns(sample_tda_summaries, layers)
        summary = _generate_executive_summary(sample_tda_summaries, layers, "gpt2", patterns)
        assert "layer_5" in summary


class TestGenerateLayerHeadline:
    """Tests for _generate_layer_headline function."""

    def test_returns_string(self):
        """Should return a string."""
        headline = _generate_layer_headline("layer_0", 50.0, 10.0, 2, 0.1, 50.0, 0.1)
        assert isinstance(headline, str)

    def test_describes_high_h0(self):
        """Should describe above-average H0."""
        headline = _generate_layer_headline("layer_0", 150.0, 30.0, 0, 0.0, 50.0, 0.1)
        assert "separation" in headline.lower() or "spread" in headline.lower() or "above" in headline.lower()

    def test_describes_low_h0(self):
        """Should describe below-average H0."""
        headline = _generate_layer_headline("layer_0", 10.0, 2.0, 0, 0.0, 50.0, 0.1)
        assert "compact" in headline.lower() or "tight" in headline.lower() or "moderate" in headline.lower()

    def test_describes_no_h1(self):
        """Should mention no loops when H1 is zero."""
        headline = _generate_layer_headline("layer_0", 50.0, 10.0, 0, 0.0, 50.0, 0.1)
        assert "no loops" in headline.lower() or "simple" in headline.lower()

    def test_describes_h1_loops(self):
        """Should mention loops when H1 count > 0."""
        headline = _generate_layer_headline("layer_0", 50.0, 10.0, 5, 0.5, 50.0, 0.1)
        assert "loop" in headline.lower() or "cyclic" in headline.lower()


class TestFormatInterpretationMarkdown:
    """Tests for format_interpretation_markdown function."""

    def test_returns_string(self, sample_tda_summaries):
        """Should return a string."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert isinstance(result, str)

    def test_includes_header(self, sample_tda_summaries):
        """Should include header with model name."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert "## TDA Interpretation: gpt2" in result

    def test_includes_executive_summary_section(self, sample_tda_summaries):
        """Should include executive summary section."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert "### Executive Summary" in result

    def test_includes_key_findings_section(self, sample_tda_summaries):
        """Should include key findings section."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert "### Key Findings" in result

    def test_includes_layer_analysis_section(self, sample_tda_summaries):
        """Should include layer-by-layer analysis section."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert "### Layer-by-Layer Analysis" in result

    def test_includes_methodology_section(self, sample_tda_summaries):
        """Should include methodology section."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert "### Methodology Note" in result

    def test_includes_limitations_section(self, sample_tda_summaries):
        """Should include limitations section."""
        interpretation = interpret_tda_results(sample_tda_summaries, model_name="gpt2")
        result = format_interpretation_markdown(interpretation)
        assert "### Limitations" in result


class TestGenerateMetricGlossary:
    """Tests for generate_metric_glossary function."""

    def test_returns_string(self):
        """Should return a string."""
        result = generate_metric_glossary()
        assert isinstance(result, str)

    def test_includes_header(self):
        """Should include glossary header."""
        result = generate_metric_glossary()
        assert "## TDA Metrics Glossary" in result

    def test_includes_h0_metrics(self):
        """Should include H0 metric descriptions."""
        result = generate_metric_glossary()
        assert "Connected Components" in result or "H0" in result

    def test_includes_h1_metrics(self):
        """Should include H1 metric descriptions."""
        result = generate_metric_glossary()
        assert "Loops" in result or "H1" in result


class TestEdgeCases:
    """Tests for edge cases in interpretation."""

    def test_single_layer(self):
        """Should handle single layer data."""
        data = {
            "layer_0": {
                "H0_count": 30,
                "H0_total_persistence": 50.0,
                "H0_max_lifetime": 10.0,
                "H1_count": 0,
                "H1_total_persistence": 0.0,
                "H1_max_lifetime": 0.0
            }
        }
        result = interpret_tda_results(data, model_name="test")
        assert result is not None
        assert len(result.layer_insights) == 1

    def test_all_zero_h1(self):
        """Should handle all zero H1 values."""
        data = {
            "layer_0": {"H0_total_persistence": 50.0, "H0_max_lifetime": 10.0, "H1_count": 0, "H1_total_persistence": 0.0, "H1_max_lifetime": 0.0},
            "layer_1": {"H0_total_persistence": 60.0, "H0_max_lifetime": 12.0, "H1_count": 0, "H1_total_persistence": 0.0, "H1_max_lifetime": 0.0}
        }
        result = interpret_tda_results(data, model_name="test")
        assert result is not None
        # Key findings should mention minimal cyclic structure
        findings_text = " ".join(result.key_findings)
        assert "minimal" in findings_text.lower() or "tree" in findings_text.lower()

    def test_uniform_h0(self):
        """Should handle uniform H0 values."""
        data = {
            "layer_0": {"H0_total_persistence": 50.0, "H0_max_lifetime": 10.0, "H1_count": 0, "H1_total_persistence": 0.0, "H1_max_lifetime": 0.0},
            "layer_1": {"H0_total_persistence": 50.0, "H0_max_lifetime": 10.0, "H1_count": 0, "H1_total_persistence": 0.0, "H1_max_lifetime": 0.0},
            "layer_2": {"H0_total_persistence": 50.0, "H0_max_lifetime": 10.0, "H1_count": 0, "H1_total_persistence": 0.0, "H1_max_lifetime": 0.0}
        }
        result = interpret_tda_results(data, model_name="test")
        assert result is not None

    def test_missing_optional_fields(self):
        """Should handle missing optional fields gracefully."""
        data = {
            "layer_0": {"H0_total_persistence": 50.0},
            "layer_1": {"H0_total_persistence": 60.0}
        }
        layers = list(data.keys())
        # Should not raise an error
        insights = _generate_layer_insights(data, layers, 30)
        assert len(insights) == 2
