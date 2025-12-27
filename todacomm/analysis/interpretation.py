"""Interpretive analysis module for TDA results.

Transforms raw TDA metrics into human-readable insights that researchers
can actually understand and act upon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class LayerInsight:
    """Insight about a specific layer's topology."""
    layer_name: str
    headline: str  # One-line summary
    details: List[str]  # Detailed explanations
    significance: str  # "high", "medium", "low"


@dataclass
class PatternInsight:
    """Insight about a pattern across layers."""
    pattern_type: str  # "peak", "trough", "trend", "anomaly"
    metric: str
    description: str
    affected_layers: List[str]
    interpretation: str


@dataclass
class TDAInterpretation:
    """Complete interpretation of TDA analysis results."""
    model_name: str
    executive_summary: str
    key_findings: List[str]
    layer_insights: List[LayerInsight]
    pattern_insights: List[PatternInsight]
    methodology_note: str
    limitations: List[str]


# Constants for interpretation
METRIC_DESCRIPTIONS = {
    "H0_count": {
        "name": "Connected Components (H0)",
        "what": "Number of distinct clusters in the activation space",
        "note": "When using fixed sampling (e.g., 30 points), this equals the sample size and is NOT meaningful. Focus on persistence instead."
    },
    "H0_total_persistence": {
        "name": "H0 Total Persistence",
        "what": "Sum of all cluster lifetimes in the filtration",
        "interpretation": "Higher values indicate more spread-out, well-separated clusters. Low values suggest compact, tightly grouped activations.",
        "units": "filtration distance units"
    },
    "H0_max_lifetime": {
        "name": "H0 Maximum Lifetime",
        "what": "Lifetime of the most persistent (dominant) cluster",
        "interpretation": "A high max lifetime indicates one very prominent cluster structure. Compare to total persistence to understand cluster distribution.",
        "units": "filtration distance units"
    },
    "H1_count": {
        "name": "Loops/Cycles (H1)",
        "what": "Number of 1-dimensional holes (loops) in the data",
        "interpretation": "More loops suggest more complex circular/cyclic structure in the activation manifold. Zero means a tree-like structure."
    },
    "H1_total_persistence": {
        "name": "H1 Total Persistence",
        "what": "Sum of all loop lifetimes",
        "interpretation": "Higher values indicate more robust cyclic structures. Very low values suggest loops are noise/artifacts."
    },
    "H1_max_lifetime": {
        "name": "H1 Maximum Lifetime",
        "what": "Lifetime of the most persistent loop",
        "interpretation": "A long-lived loop represents a genuine topological feature, not noise."
    }
}


def interpret_tda_results(
    tda_summaries: Dict[str, Dict[str, float]],
    model_name: str = "model",
    sample_count: int = 30
) -> TDAInterpretation:
    """Generate comprehensive interpretation of TDA results.

    Args:
        tda_summaries: Dictionary of layer -> TDA metrics
        model_name: Name of the model being analyzed
        sample_count: Number of points sampled for TDA (for context)

    Returns:
        TDAInterpretation with all insights
    """
    layers = list(tda_summaries.keys())

    # Extract metrics across layers
    h0_persistence = {l: tda_summaries[l].get("H0_total_persistence", 0) for l in layers}
    h0_max_life = {l: tda_summaries[l].get("H0_max_lifetime", 0) for l in layers}
    h1_count = {l: tda_summaries[l].get("H1_count", 0) for l in layers}
    h1_persistence = {l: tda_summaries[l].get("H1_total_persistence", 0) for l in layers}

    # Detect patterns
    pattern_insights = _detect_patterns(tda_summaries, layers)

    # Generate layer-by-layer insights
    layer_insights = _generate_layer_insights(tda_summaries, layers, sample_count)

    # Generate key findings
    key_findings = _generate_key_findings(tda_summaries, layers, model_name)

    # Executive summary
    executive_summary = _generate_executive_summary(
        tda_summaries, layers, model_name, pattern_insights
    )

    # Methodology note
    methodology_note = (
        f"Analysis computed using persistent homology on {sample_count}-point samples "
        f"from PCA-reduced activation spaces. H0 features track cluster structure, "
        f"H1 features track loops/cycles. Persistence measures feature significance."
    )

    # Limitations
    limitations = [
        "H0 count equals sample size when using fixed sampling - focus on persistence metrics instead.",
        "Results depend on PCA dimensionality and sampling strategy.",
        "Topological features describe geometry, not causally explain model behavior.",
        "Small sample sizes may miss fine-grained structure."
    ]

    return TDAInterpretation(
        model_name=model_name,
        executive_summary=executive_summary,
        key_findings=key_findings,
        layer_insights=layer_insights,
        pattern_insights=pattern_insights,
        methodology_note=methodology_note,
        limitations=limitations
    )


def _detect_patterns(
    tda_summaries: Dict[str, Dict[str, float]],
    layers: List[str]
) -> List[PatternInsight]:
    """Detect significant patterns across layers."""
    patterns = []

    # Check H0 persistence patterns
    h0_pers = [tda_summaries[l].get("H0_total_persistence", 0) for l in layers]
    if len(h0_pers) > 2:
        peak_idx = np.argmax(h0_pers)
        peak_layer = layers[peak_idx]
        peak_val = h0_pers[peak_idx]
        mean_val = np.mean(h0_pers)

        if peak_val > 2 * mean_val:
            patterns.append(PatternInsight(
                pattern_type="peak",
                metric="H0_total_persistence",
                description=f"H0 persistence peaks dramatically at {peak_layer} ({peak_val:.1f})",
                affected_layers=[peak_layer],
                interpretation=(
                    f"The {peak_layer} layer shows maximum cluster separation - activations "
                    f"spread out into well-separated groups here. This is {peak_val/mean_val:.1f}x "
                    f"higher than average ({mean_val:.1f}), suggesting this layer creates the most "
                    f"distinct representational clusters."
                )
            ))

    # Check H1 evolution
    h1_pers = [tda_summaries[l].get("H1_total_persistence", 0) for l in layers]
    h1_counts = [tda_summaries[l].get("H1_count", 0) for l in layers]

    # Check if H1 increases towards later layers
    if len(layers) >= 3:
        early_h1 = np.mean(h1_pers[:len(layers)//2])
        late_h1 = np.mean(h1_pers[len(layers)//2:])

        if late_h1 > 3 * early_h1 and late_h1 > 0.1:
            late_layers = layers[len(layers)//2:]
            patterns.append(PatternInsight(
                pattern_type="trend",
                metric="H1_total_persistence",
                description="Cyclic structure (H1) increases significantly in later layers",
                affected_layers=late_layers,
                interpretation=(
                    f"Later layers ({', '.join(late_layers)}) develop stronger loop structures "
                    f"({late_h1:.2f} avg) compared to early layers ({early_h1:.4f} avg). "
                    f"This suggests the model builds more complex, cyclic representations "
                    f"as information propagates through the network."
                )
            ))

    # Check for embedding layer compactness
    if "embedding" in layers:
        emb_h0 = tda_summaries["embedding"].get("H0_total_persistence", 0)
        other_h0 = [tda_summaries[l].get("H0_total_persistence", 0)
                   for l in layers if l != "embedding"]
        if other_h0 and emb_h0 < np.mean(other_h0) * 0.2:
            patterns.append(PatternInsight(
                pattern_type="characteristic",
                metric="H0_total_persistence",
                description="Embedding layer has very compact activation clusters",
                affected_layers=["embedding"],
                interpretation=(
                    f"The embedding layer shows much lower cluster spread ({emb_h0:.1f}) "
                    f"compared to other layers (avg {np.mean(other_h0):.1f}). Token embeddings "
                    f"start in a relatively compact space before transformer blocks expand "
                    f"and differentiate the representations."
                )
            ))

    # Check for final layer characteristics
    if "final" in layers and "layer_11" in layers:
        final_h0 = tda_summaries["final"].get("H0_total_persistence", 0)
        layer11_h0 = tda_summaries["layer_11"].get("H0_total_persistence", 0)

        if abs(final_h0 - layer11_h0) < 0.01:
            patterns.append(PatternInsight(
                pattern_type="equivalence",
                metric="H0_total_persistence",
                description="Final layer and layer_11 have identical topology",
                affected_layers=["layer_11", "final"],
                interpretation=(
                    "The final layer normalization doesn't change the topological structure "
                    "established by the last transformer block. This is expected - LayerNorm "
                    "rescales but preserves relative distances."
                )
            ))

    return patterns


def _generate_layer_insights(
    tda_summaries: Dict[str, Dict[str, float]],
    layers: List[str],
    sample_count: int
) -> List[LayerInsight]:
    """Generate insights for each layer."""
    insights = []

    # Compute statistics for context
    all_h0_pers = [tda_summaries[l].get("H0_total_persistence", 0) for l in layers]
    all_h1_pers = [tda_summaries[l].get("H1_total_persistence", 0) for l in layers]
    h0_mean, h0_std = np.mean(all_h0_pers), np.std(all_h0_pers)
    h1_mean = np.mean(all_h1_pers)

    for layer in layers:
        metrics = tda_summaries[layer]
        h0_pers = metrics.get("H0_total_persistence", 0)
        h0_max = metrics.get("H0_max_lifetime", 0)
        h1_count = metrics.get("H1_count", 0)
        h1_pers = metrics.get("H1_total_persistence", 0)

        # Determine significance
        if h0_std > 0:
            z_score = abs(h0_pers - h0_mean) / h0_std
            significance = "high" if z_score > 1.5 else ("medium" if z_score > 0.5 else "low")
        else:
            significance = "medium"

        # Generate headline
        headline = _generate_layer_headline(layer, h0_pers, h0_max, h1_count, h1_pers,
                                           h0_mean, h1_mean)

        # Generate details
        details = []

        # H0 interpretation
        h0_ratio = h0_max / h0_pers if h0_pers > 0 else 0
        if h0_ratio > 0.5:
            details.append(
                f"Cluster structure dominated by one major component (max={h0_max:.1f} is "
                f"{h0_ratio*100:.0f}% of total={h0_pers:.1f})"
            )
        else:
            details.append(
                f"Multiple significant clusters contribute to H0 persistence "
                f"(max={h0_max:.1f}, total={h0_pers:.1f})"
            )

        # H1 interpretation
        if h1_count == 0:
            details.append("No loops detected - activation space has tree-like structure")
        elif h1_pers < 0.1:
            details.append(
                f"{int(h1_count)} weak loop(s) detected (persistence={h1_pers:.4f}) - "
                f"likely noise rather than significant topology"
            )
        else:
            details.append(
                f"{int(h1_count)} significant loop(s) with total persistence {h1_pers:.2f} - "
                f"indicates genuine cyclic structure in activations"
            )

        insights.append(LayerInsight(
            layer_name=layer,
            headline=headline,
            details=details,
            significance=significance
        ))

    return insights


def _generate_layer_headline(
    layer: str,
    h0_pers: float,
    h0_max: float,
    h1_count: float,
    h1_pers: float,
    h0_mean: float,
    h1_mean: float
) -> str:
    """Generate a one-line summary for a layer."""

    # Classify H0 spread
    if h0_pers > 2 * h0_mean:
        h0_desc = "Maximum cluster separation"
    elif h0_pers < 0.3 * h0_mean:
        h0_desc = "Compact, tight clusters"
    elif h0_pers > h0_mean:
        h0_desc = "Above-average cluster spread"
    else:
        h0_desc = "Moderate cluster structure"

    # Classify H1 complexity
    if h1_count == 0 or h1_pers < 0.01:
        h1_desc = "simple (no loops)"
    elif h1_pers > 2 * h1_mean and h1_mean > 0:
        h1_desc = "complex cyclic structure"
    else:
        h1_desc = f"{int(h1_count)} loop(s)"

    return f"{h0_desc}, {h1_desc}"


def _generate_key_findings(
    tda_summaries: Dict[str, Dict[str, float]],
    layers: List[str],
    model_name: str
) -> List[str]:
    """Generate the top key findings."""
    findings = []

    # Find layer with max H0 persistence
    h0_pers = {l: tda_summaries[l].get("H0_total_persistence", 0) for l in layers}
    max_h0_layer = max(h0_pers, key=h0_pers.get)
    max_h0_val = h0_pers[max_h0_layer]

    # Finding 1: Peak cluster separation
    findings.append(
        f"Maximum cluster separation occurs at {max_h0_layer} with H0 persistence of "
        f"{max_h0_val:.1f}, indicating this layer creates the most distinct representation groups."
    )

    # Finding 2: Embedding vs later layers
    if "embedding" in layers:
        emb_h0 = tda_summaries["embedding"].get("H0_total_persistence", 0)
        expansion = max_h0_val / emb_h0 if emb_h0 > 0 else float('inf')
        if expansion > 5:
            findings.append(
                f"Representations expand {expansion:.0f}x from embedding ({emb_h0:.1f}) "
                f"to peak layer ({max_h0_val:.1f}), showing significant geometric transformation."
            )

    # Finding 3: H1 evolution
    h1_pers = {l: tda_summaries[l].get("H1_total_persistence", 0) for l in layers}
    max_h1_layer = max(h1_pers, key=h1_pers.get)
    max_h1_val = h1_pers[max_h1_layer]

    if max_h1_val > 0.1:
        findings.append(
            f"Strongest cyclic structure appears at {max_h1_layer} with H1 persistence of "
            f"{max_h1_val:.2f}, suggesting complex loop topology in later representations."
        )
    else:
        findings.append(
            f"Minimal cyclic structure across all layers (max H1 persistence: {max_h1_val:.4f}), "
            f"indicating primarily tree-like activation geometry."
        )

    return findings


def _generate_executive_summary(
    tda_summaries: Dict[str, Dict[str, float]],
    layers: List[str],
    model_name: str,
    patterns: List[PatternInsight]
) -> str:
    """Generate a 2-3 sentence executive summary."""

    h0_pers = [tda_summaries[l].get("H0_total_persistence", 0) for l in layers]
    h1_pers = [tda_summaries[l].get("H1_total_persistence", 0) for l in layers]

    max_h0_layer = layers[np.argmax(h0_pers)]
    max_h0 = max(h0_pers)
    min_h0 = min(h0_pers)
    max_h1 = max(h1_pers)

    summary_parts = []

    # Part 1: Overall spread
    spread_ratio = max_h0 / min_h0 if min_h0 > 0 else float('inf')
    summary_parts.append(
        f"{model_name} shows a {spread_ratio:.0f}x variation in cluster spread across layers, "
        f"with peak separation at {max_h0_layer}."
    )

    # Part 2: Topology character
    if max_h1 > 1.0:
        summary_parts.append(
            "Significant cyclic structures emerge in later layers, suggesting complex "
            "representational geometry."
        )
    elif max_h1 > 0.1:
        summary_parts.append(
            "Moderate loop structures present, indicating some cyclic patterns in the "
            "activation manifold."
        )
    else:
        summary_parts.append(
            "Minimal cyclic structure throughout - activations primarily form tree-like "
            "cluster arrangements."
        )

    return " ".join(summary_parts)


def format_interpretation_markdown(interpretation: TDAInterpretation) -> str:
    """Format the interpretation as a markdown report section."""
    lines = []

    lines.append(f"## TDA Interpretation: {interpretation.model_name}")
    lines.append("")

    # Executive Summary
    lines.append("### Executive Summary")
    lines.append("")
    lines.append(interpretation.executive_summary)
    lines.append("")

    # Key Findings
    lines.append("### Key Findings")
    lines.append("")
    for i, finding in enumerate(interpretation.key_findings, 1):
        lines.append(f"{i}. {finding}")
    lines.append("")

    # Pattern Analysis
    if interpretation.pattern_insights:
        lines.append("### Pattern Analysis")
        lines.append("")
        for pattern in interpretation.pattern_insights:
            lines.append(f"**{pattern.description}**")
            lines.append("")
            lines.append(pattern.interpretation)
            lines.append("")

    # Layer-by-Layer Analysis
    lines.append("### Layer-by-Layer Analysis")
    lines.append("")
    for insight in interpretation.layer_insights:
        sig_marker = {"high": "**", "medium": "", "low": ""}[insight.significance]
        lines.append(f"#### {sig_marker}{insight.layer_name}{sig_marker}")
        lines.append(f"*{insight.headline}*")
        lines.append("")
        for detail in insight.details:
            lines.append(f"- {detail}")
        lines.append("")

    # Methodology
    lines.append("### Methodology Note")
    lines.append("")
    lines.append(interpretation.methodology_note)
    lines.append("")

    # Limitations
    lines.append("### Limitations")
    lines.append("")
    for limitation in interpretation.limitations:
        lines.append(f"- {limitation}")
    lines.append("")

    return "\n".join(lines)


def generate_metric_glossary() -> str:
    """Generate a glossary of TDA metrics for the report."""
    lines = []
    lines.append("## TDA Metrics Glossary")
    lines.append("")

    for metric_key, info in METRIC_DESCRIPTIONS.items():
        lines.append(f"### {info['name']}")
        lines.append(f"**What it measures:** {info['what']}")
        if "interpretation" in info:
            lines.append(f"**How to interpret:** {info['interpretation']}")
        if "note" in info:
            lines.append(f"**Note:** {info['note']}")
        lines.append("")

    return "\n".join(lines)
