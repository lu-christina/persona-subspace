"""
Shared plotting utilities for config-based steering analysis.

This module provides consistent parsing, formatting, and configuration
for analyzing and visualizing steering experiments with magnitude-based coefficients.
"""

# ============================================================================
# Parsing Functions
# ============================================================================

def parse_experiment_id(experiment_id):
    """
    Parse experiment_id into layer, vector type, and coefficient components.

    Args:
        experiment_id: String like "layer_32-contrast-coeff:-0.75" or "layer_32-role_pc1-coeff:0.5"

    Returns:
        tuple: (layer, vector_type, magnitude)
        - layer: int like 32
        - vector_type: str like "contrast" or "role_pc1"
        - magnitude: float like -0.75 or 0.5

    Examples:
        "layer_32-contrast-coeff:-0.75" → (32, "contrast", -0.75)
        "layer_32-role_pc1-coeff:0.5" → (32, "role_pc1", 0.5)
        "baseline_unsteered" → (None, None, 0.0)
    """
    if not experiment_id or experiment_id == "baseline":
        return (None, None, None)

    # Handle baseline_unsteered case
    if experiment_id == "baseline_unsteered":
        return (None, None, 0.0)

    try:
        # Find the coeff: part to split properly
        if 'coeff:' not in experiment_id:
            return (None, None, None)

        # Split at 'coeff:' to handle negative numbers
        before_coeff, coeff_value = experiment_id.split('coeff:')

        # Parse coefficient (can be negative)
        magnitude = float(coeff_value)

        # Split the part before coeff by dash to get layer and vector_type
        # Format: "layer_32-contrast-" or "layer_32-role_pc1-"
        parts = before_coeff.rstrip('-').split('-')

        if len(parts) < 2:
            return (None, None, None)

        # First part is layer_32, rest is vector_type (could be role_pc1 with underscore)
        layer_part = parts[0]
        vector_type = '-'.join(parts[1:]) if len(parts) > 2 else parts[1]

        # Parse layer number
        layer = int(layer_part.replace('layer_', ''))

        return (layer, vector_type, magnitude)
    except (ValueError, AttributeError):
        return (None, None, None)


def extract_magnitude(experiment_id):
    """
    Extract magnitude (coefficient) from experiment_id.

    Args:
        experiment_id: String like "layer_32-contrast-coeff:-0.75"

    Returns:
        float: Magnitude value (e.g., -0.75)

    Examples:
        "layer_32-contrast-coeff:-0.75" → -0.75
        "layer_32-role_pc1-coeff:0.5" → 0.5
    """
    _, _, magnitude = parse_experiment_id(experiment_id)
    return magnitude


def extract_vector_type(experiment_id):
    """
    Extract vector type from experiment_id.

    Args:
        experiment_id: String like "layer_32-contrast-coeff:-0.75"

    Returns:
        str: Vector type ("contrast" or "role_pc1")

    Examples:
        "layer_32-contrast-coeff:-0.75" → "contrast"
        "layer_32-role_pc1-coeff:0.5" → "role_pc1"
    """
    _, vector_type, _ = parse_experiment_id(experiment_id)
    return vector_type


# ============================================================================
# Formatting Functions
# ============================================================================

def format_magnitude_label(magnitude):
    """
    Convert magnitude to human-readable label.

    Args:
        magnitude: Float like -0.75, 0.0, or 0.5

    Returns:
        str: Human-readable label

    Examples:
        0.0 → "Baseline"
        -0.75 → "-0.75"
        0.5 → "0.5"
    """
    if magnitude is None or magnitude == 0.0:
        return "Baseline"

    # Format with appropriate precision
    if magnitude == int(magnitude):
        return str(int(magnitude))
    else:
        return str(magnitude)


def magnitude_sort_key(magnitude):
    """
    Create sort key for magnitude values.
    Orders: 0.0 (baseline) → negative (ascending) → positive (ascending).

    Args:
        magnitude: Float value

    Returns:
        tuple: (category, value) for sorting

    Examples:
        0.0 → (0, 0.0)      # baseline first
        -0.25 → (1, 0.25)   # negative, sorted ascending (by absolute value)
        -2.0 → (1, 2.0)     # larger absolute values later
        0.25 → (2, 0.25)    # positive, sorted ascending
        0.5 → (2, 0.5)      # larger values later
    """
    if magnitude is None or magnitude == 0.0:
        return (0, 0.0)  # Baseline first
    elif magnitude < 0:
        return (1, -magnitude)  # Negative second, sorted by absolute value (ascending)
    else:
        return (2, magnitude)  # Positive last, sorted ascending


# ============================================================================
# Color Configuration
# ============================================================================

# Color scheme for harmful response types
HARMFUL_COLORS = {
    "enough_info_and_follow_perfectly": "#e41a1c",  # red
    "enough_info": "#ff7f00",                       # orange
}

# Color scheme for other response types
OTHER_COLORS = {
    "almost_enough_info": "#ffff33",        # yellow
    "related_but_no_info": "#1f78b4",       # cyan/blueish
    "refusal": "#b2df8a",                   # lime green
    "refusal_and_justification": "#33a02c", # green
    "out_of_context": "#6a3d9a",            # purple
    "nonsensical": "#fb9a99",               # pink
}

# Display names for response types
RESPONSE_LABEL_MAP = {
    "enough_info": "Harmful Info",
    "enough_info_and_follow_perfectly": "Harmful Info & Following Request",
    "refusal": "Refusal",
    "refusal_and_justification": "Refusal & Justification",
    "nonsensical": "Nonsensical",
    "out_of_context": "Out of Context",
    "related_but_no_info": "Related",
    "almost_enough_info": "Almost Harmful",
}
