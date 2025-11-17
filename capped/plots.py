"""
Shared plotting utilities for capped projection analysis.

This module provides consistent parsing, formatting, and configuration
for analyzing and visualizing projection capping experiments.
"""

# ============================================================================
# Configuration & Metadata
# ============================================================================

# Color scheme for different configs
CONFIG_COLORS = {
    'baseline': '#2ca02c',      # green
    'role_trait': '#2ca02c',    # green
    'jailbreak': '#d62728',     # red
    'lmsys_10000': '#9467bd',   # purple
    'pc1_role_trait': '#1f77b4', # blue
    'sonnet_role': '#ff7f0e', # orange
    'qwen_role': '#9467bd' # purple
}

# Display names for configs
CONFIG_DISPLAY_NAMES = {
    'baseline': 'Baseline',
    'role_trait': 'On-Policy Role/Trait Rollouts',
    'jailbreak': 'Jailbreak Rollouts',
    'lmsys_10000': 'LMSYS-10K Rollouts',
    'pc1_role_trait': 'Role PC1',
    'sonnet_role': 'Sonnet Role Rollouts',
    'qwen_role': 'Qwen Role Rollouts'
}

# Config ordering for consistent sorting (baseline first)
CONFIG_ORDER = {
    'baseline': 0,
    'role_trait': 1,
    'jailbreak': 2,
    'lmsys_10000': 3,
    'pc1_role_trait': 4,
    'sonnet_role': 5,
    'qwen_role': 6
}


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_experiment_id(experiment_id):
    """
    Parse experiment_id into layers and cap components.

    Args:
        experiment_id: String like "layers_0:64-p0.01" or "layers_32:64-harm_0.25"

    Returns:
        tuple: (layer_spec, cap_type, cap_value)
        - layer_spec: str like "0:64" or "2:64:2"
        - cap_type: str like "percentile", "harm", or "safe"
        - cap_value: str like "p0.01", "harm_0.25", or "safe_0.01"

    Examples:
        "layers_0:64-p0.01" → ("0:64", "percentile", "p0.01")
        "layers_32:64-harm_0.25" → ("32:64", "harm", "harm_0.25")
        "layers_2:64:2-safe_0.01" → ("2:64:2", "safe", "safe_0.01")
        "baseline" → (None, None, None)
    """
    if experiment_id == "baseline":
        return (None, None, None)

    parts = experiment_id.split('-')
    if len(parts) != 2:
        return (None, None, None)

    layer_part, cap_part = parts
    layer_spec = layer_part.replace('layers_', '')

    # Determine cap type
    if cap_part.startswith('p'):
        cap_type = "percentile"
    elif cap_part.startswith('harm'):
        cap_type = "harm"
    elif cap_part.startswith('safe'):
        cap_type = "safe"
    else:
        cap_type = "unknown"

    return (layer_spec, cap_type, cap_part)


def parse_layer_sort_key(layer_spec, total_layers=64):
    """
    Parse layer_spec to create sort key for consistent ordering:
    1. All layers (0:64) comes first
    2. Interval layers (2:64:2, 4:64:4) by increasing interval
    3. Range layers (8:16, 16:24, etc.) by start then end

    Args:
        layer_spec: String like "0:64", "16:24", "2:64:2"
        total_layers: Total number of layers in the model (default: 64)

    Returns:
        tuple: (category, interval_or_start, end)
    """
    if layer_spec is None:
        return (-1, 0, 0)  # Baseline comes first

    parts = layer_spec.split(':')

    if len(parts) == 3:
        # Interval layer (e.g., "2:64:2")
        start, end, interval = map(int, parts)
        return (1, int(interval), end)  # Category 1 = interval layers
    elif len(parts) == 2:
        start, end = map(int, parts)
        if start == 0 and end == total_layers:
            # All layers (0:64)
            return (0, 0, total_layers)  # Category 0 = all layers
        else:
            # Range layers (8:16, 16:24, etc.)
            return (2, start, end)  # Category 2 = range layers

    return (999, 0, 0)  # Unknown, sort to end


def cap_sort_key(cap_value):
    """
    Create sort key for cap values.
    Orders: safe before harm, smaller percentiles first.

    Args:
        cap_value: String like "p0.01", "harm_0.25", "safe_0.01"

    Returns:
        tuple: (category, value) for sorting
    """
    if cap_value is None:
        return (0, 0.0)  # Baseline first
    if cap_value.startswith('safe_'):
        return (0, float(cap_value.split('_')[1]))  # Safe first, then by value
    elif cap_value.startswith('harm_'):
        return (1, float(cap_value.split('_')[1]))  # Harm second, then by value
    elif cap_value.startswith('p'):
        return (0, float(cap_value[1:]))  # Regular percentiles
    return (999, 0.0)  # Unknown


# ============================================================================
# Formatting Functions
# ============================================================================

def format_layer_range(layer_spec, total_layers=64):
    """
    Convert layer_spec to human-readable string.

    Args:
        layer_spec: String like "0:64", "16:24", "2:64:2"
        total_layers: Total number of layers in the model (default: 64)

    Returns:
        str: Human-readable layer description

    Examples:
        "0:64" → "All Layers (0-63)"
        "16:24" → "Layers 16-23"
        "2:64:2" → "Every 2nd Layer"
        "4:64:4" → "Every 4th Layer"
    """
    if layer_spec is None:
        return "Baseline"

    parts = layer_spec.split(':')

    if len(parts) == 3:
        # Has interval (e.g., "2:64:2")
        start, end, interval = map(int, parts)
        if int(interval) == 2:
            return "Every 2nd Layer"
        else:
            return f"Every {interval}th Layer"
    elif len(parts) == 2:
        # Range without interval
        start, end = map(int, parts)
        if start == 0 and end == total_layers:
            return f"All Layers (0-{total_layers-1})"
        elif start == total_layers // 2:
            return f"Layers {start}-{end-1}"
        else:
            return f"Layers {start}-{end-1}"

    return layer_spec


def format_cap_label(cap_value):
    """
    Convert cap value to human-readable label.

    Args:
        cap_value: String like "p0.01", "harm_0.25", "safe_0.01"

    Returns:
        str: Human-readable label

    Examples:
        "p0.01" → "1st"
        "p0.25" → "25th"
        "p0.5" → "50th"
        "p0.75" → "75th"
        "harm_0.01" → "Harm 99th"
        "harm_0.25" → "Harm 75th"
        "safe_0.01" → "Safe 99th"
        "safe_0.50" → "Safe 50th"
    """
    if cap_value is None:
        return "Baseline"

    # Percentile-based (role_trait, lmsys_10000)
    if cap_value.startswith('p'):
        percentile = float(cap_value[1:])
        if percentile == 0.01:
            return "1st"
        elif percentile == 0.25:
            return "25th"
        elif percentile == 0.5:
            return "50th"
        elif percentile == 0.75:
            return "75th"
        else:
            return cap_value

    # Harm/Safe-based (jailbreak)
    elif cap_value.startswith('harm_') or cap_value.startswith('safe_'):
        prefix, value = cap_value.split('_')
        percentile = float(value)

        # Convert to percentile rank (lower value = higher rank)
        if percentile == 0.01:
            rank = "99th"
        elif percentile == 0.25:
            rank = "75th"
        elif percentile == 0.50:
            rank = "50th"
        else:
            rank = value

        return f"{prefix.capitalize()} {rank}"

    return cap_value


def get_cap_names(cap_type='percentile'):
    """
    Get short names for cap types based on the experiment type.

    Args:
        cap_type: Either 'percentile' (for role_trait/lmsys) or 'jailbreak'

    Returns:
        dict: Mapping of cap values to short display labels

    Examples:
        get_cap_names('percentile') → {'p0.01': '1%ile', 'p0.25': '25%ile', ...}
        get_cap_names('jailbreak') → {'harm_0.25': '75%<br>Harmful', ...}
    """
    if cap_type == 'jailbreak':
        return {
            'harm_0.25': '75%<br>Harmful',
            'harm_0.01': '99%<br>Harmful',
            'safe_0.50': '50%<br>Safe',
            'safe_0.01': '99%<br>Safe',
        }
    else:  # percentile
        return {
            'p0.01': '1%ile',
            'p0.25': '25%ile',
            'p0.5': '50%ile',
            'p0.75': '75%ile',
        }


def get_layer_group_names(layer_groups, total_layers=64):
    """
    Generate human-readable names for layer groups.

    Args:
        layer_groups: List of layer_spec strings (e.g., ['0:64', '16:24', '2:64:2'])
        total_layers: Total number of layers in the model (default: 64)

    Returns:
        dict: Mapping of layer_spec to human-readable names

    Example:
        get_layer_group_names(['0:64', '16:24']) →
            {'0:64': 'All Layers (0-63)', '16:24': 'Layers 16-23'}
    """
    return {lg: format_layer_range(lg, total_layers) for lg in layer_groups}


# ============================================================================
# DataFrame Utilities
# ============================================================================

def add_parsed_columns(df, total_layers=64):
    """
    Add parsed metadata columns to a DataFrame with experiment results.

    Args:
        df: DataFrame with 'experiment_id' column
        total_layers: Total number of layers in the model (default: 64)

    Returns:
        DataFrame: Copy of input with added columns:
            - layer_spec: Parsed layer specification
            - cap_type: Type of cap (percentile, harm, safe)
            - cap_value: Cap value string
            - layer_label: Human-readable layer description
            - cap_label: Human-readable cap label
            - display_name: Full human-readable name
            - layer_sort_key: Tuple for sorting by layer
            - cap_sort_key: Tuple for sorting by cap
            - config_sort_key: Integer for sorting by config
    """
    df_out = df.copy()

    # Parse experiment IDs
    parsed = df_out['experiment_id'].apply(parse_experiment_id)
    df_out['layer_spec'] = parsed.apply(lambda x: x[0])
    df_out['cap_type'] = parsed.apply(lambda x: x[1])
    df_out['cap_value'] = parsed.apply(lambda x: x[2])

    # Add human-readable labels
    df_out['layer_label'] = df_out['layer_spec'].apply(
        lambda x: format_layer_range(x, total_layers)
    )
    df_out['cap_label'] = df_out['cap_value'].apply(format_cap_label)
    df_out['display_name'] = df_out.apply(
        lambda row: f"{row['layer_label']}, {row['cap_label']}"
                    if row['experiment_id'] != 'baseline'
                    else "Baseline",
        axis=1
    )

    # Add sort keys
    df_out['layer_sort_key'] = df_out['layer_spec'].apply(
        lambda x: parse_layer_sort_key(x, total_layers)
    )
    df_out['cap_sort_key'] = df_out['cap_value'].apply(cap_sort_key)

    if 'config_name' in df_out.columns:
        df_out['config_sort_key'] = df_out['config_name'].map(CONFIG_ORDER)

    return df_out


def sort_experiments(df):
    """
    Sort DataFrame by config, layer groups, and cap values.

    Args:
        df: DataFrame with parsed columns (use add_parsed_columns first)

    Returns:
        DataFrame: Sorted copy with reset index
    """
    sort_cols = []
    if 'config_sort_key' in df.columns:
        sort_cols.append('config_sort_key')
    sort_cols.extend(['layer_sort_key', 'cap_sort_key'])

    return df.sort_values(sort_cols, na_position='first').reset_index(drop=True)
