import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

def _detect_role_types(pca_results):
    """
    Auto-detect role vector types from pca_results.
    
    Returns:
    - dict with keys: has_pos2, has_pos3, vectors, n_pos_2, colors, markers, show_legend
    """
    has_pos2 = ('pos_2' in pca_results.get('vectors', {}) and 
                'pos_2' in pca_results.get('roles', {}))
    has_pos3 = ('pos_3' in pca_results.get('vectors', {}) and 
                'pos_3' in pca_results.get('roles', {}))
    
    if has_pos2 and has_pos3:
        # Combined vectors: pos_2 + pos_3
        vectors = pca_results['vectors']['pos_2'] + pca_results['vectors']['pos_3']
        n_pos_2 = len(pca_results['roles']['pos_2'])
        n_total = len(vectors)
        colors = ['cyan'] * n_pos_2 + ['blue'] * (n_total - n_pos_2)
        markers = ['circle'] * n_pos_2 + ['square'] * (n_total - n_pos_2)
        show_legend = True
    elif has_pos3:
        # Only pos_3 vectors
        vectors = pca_results['vectors']['pos_3']
        n_pos_2 = 0
        colors = ['blue'] * len(vectors)
        markers = ['circle'] * len(vectors)
        show_legend = False
    else:
        raise ValueError("No valid role vectors found in pca_results")
    
    return {
        'has_pos2': has_pos2,
        'has_pos3': has_pos3,
        'vectors': vectors,
        'n_pos_2': n_pos_2,
        'colors': colors,
        'markers': markers,
        'show_legend': show_legend
    }

def plot_pca_cosine_similarity(pca_results, role_labels, pc_component, 
                             layer, dir, type, assistant_activation=None,
                             title="PCA on Role-Playing Vectors", subtitle=""):
    """
    Create a plot similar to the PC1 Cosine Similarity visualization.
    Shows labels on hover for most points, with visible labels and leader lines 
    for the 20 traits at either end of the range to avoid overlap.
    
    Parameters:
    - pca_results: Dictionary containing PCA results and vectors
    - role_labels: List of labels for each data point
    - pc_component: Which PC component to use (0-indexed, so PC1 = 0)
    - layer: Layer number for title (unused if custom title/subtitle provided)
    - dir: Directory parameter (unused if custom title/subtitle provided)
    - type: Type parameter (unused if custom title/subtitle provided)
    - assistant_activation: Optional assistant activation for comparison
    - title: Main title for the plot (default: "PCA on Role-Playing Vectors")
    - subtitle: Subtitle for the plot (default: "")
    
    Returns:
    - Plotly figure object
    """
    
    # Extract the specified PC component
    pc_direction = torch.from_numpy(pca_results['pca'].components_[pc_component])

    # get raw vectors
    if type == "pos23":
        vectors = pca_results['vectors']['pos_2'] + pca_results['vectors']['pos_3']
    elif type == "pos3":
        vectors = pca_results['vectors']['pos_3']
    
    vectors = torch.stack(vectors)[:, layer, :]
    pc_direction = F.normalize(pc_direction.unsqueeze(0), dim=1)
    vectors = F.normalize(vectors, dim=1)

    cosine_sims = vectors.float() @ pc_direction.float().T
    cosine_sims = cosine_sims.squeeze(1).numpy()
    
    # Calculate assistant cosine similarity if provided
    assistant_cosine_sim = None
    if assistant_activation is not None and layer is not None:
        assistant_activation_layer = assistant_activation[layer, :]
        assistant_activation_norm = F.normalize(assistant_activation_layer.unsqueeze(0), dim=1)
        assistant_cosine_sim = assistant_activation_norm.float() @ pc_direction.float().T
        assistant_cosine_sim = assistant_cosine_sim.squeeze(1).numpy()[0]
    
    # Create colors based on vector type (pos_2 = cyan, pos_3 = blue)
    if type == "pos23" and 'pos_2' in pca_results['roles']:
        n_pos_2 = len(pca_results['roles']['pos_2'])
        colors = ['cyan'] * n_pos_2 + ['blue'] * (len(cosine_sims) - n_pos_2)
    else:
        # Default to blue for pos3 only
        colors = ['blue'] * len(cosine_sims)
    
    # Determine marker shapes based on vector type
    if type == "pos23" and 'pos_2' in pca_results['roles']:
        n_pos_2 = len(pca_results['roles']['pos_2'])
        # First n_pos_2 are pos_2 (circles), rest are pos_3 (squares)
        marker_symbols = ['circle'] * n_pos_2 + ['square'] * (len(cosine_sims) - n_pos_2)
    else:
        # Default to circles for all points
        marker_symbols = ['circle'] * len(cosine_sims)
    
    # Identify extreme traits (10 lowest and 10 highest)
    sorted_indices = np.argsort(cosine_sims)
    low_extreme_indices = sorted_indices[:10]
    high_extreme_indices = sorted_indices[-10:]
    extreme_indices = set(list(low_extreme_indices) + list(high_extreme_indices))
    
    # Create single plot figure
    fig = go.Figure()
    
    if type == "pos23" and 'pos_2' in pca_results['roles']:
        # Split points by type for legend
        n_pos_2 = len(pca_results['roles']['pos_2'])
        
        # Split regular and extreme points by type
        pos2_regular_x, pos2_regular_y, pos2_regular_colors, pos2_regular_labels = [], [], [], []
        pos3_regular_x, pos3_regular_y, pos3_regular_colors, pos3_regular_labels = [], [], [], []
        pos2_extreme_x, pos2_extreme_y, pos2_extreme_colors, pos2_extreme_labels = [], [], [], []
        pos3_extreme_x, pos3_extreme_y, pos3_extreme_colors, pos3_extreme_labels = [], [], [], []
        
        for i, (sim, color, label, symbol) in enumerate(zip(cosine_sims, colors, role_labels, marker_symbols)):
            is_pos2 = i < n_pos_2
            if i in extreme_indices:
                if is_pos2:
                    pos2_extreme_x.append(sim)
                    pos2_extreme_y.append(1)
                    pos2_extreme_colors.append(color)
                    pos2_extreme_labels.append(label)
                else:
                    pos3_extreme_x.append(sim)
                    pos3_extreme_y.append(1)
                    pos3_extreme_colors.append(color)
                    pos3_extreme_labels.append(label)
            else:
                if is_pos2:
                    pos2_regular_x.append(sim)
                    pos2_regular_y.append(1)
                    pos2_regular_colors.append(color)
                    pos2_regular_labels.append(label)
                else:
                    pos3_regular_x.append(sim)
                    pos3_regular_y.append(1)
                    pos3_regular_colors.append(color)
                    pos3_regular_labels.append(label)
        
        # Add pos_2 regular points
        if pos2_regular_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_regular_x,
                    y=pos2_regular_y,
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_regular_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add pos_2 extreme points
        if pos2_extreme_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_extreme_x,
                    y=pos2_extreme_y,
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_extreme_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add pos_3 regular points  
        if pos3_regular_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_regular_x,
                    y=pos3_regular_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_regular_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add pos_3 extreme points
        if pos3_extreme_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_extreme_x,
                    y=pos3_extreme_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_extreme_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
            
    else:
        # Original logic for single type
        # Split points into regular and extreme for different display modes
        regular_x, regular_y, regular_colors, regular_labels, regular_symbols = [], [], [], [], []
        extreme_x, extreme_y, extreme_colors, extreme_labels, extreme_symbols = [], [], [], [], []
        
        for i, (sim, color, label, symbol) in enumerate(zip(cosine_sims, colors, role_labels, marker_symbols)):
            if i in extreme_indices:
                extreme_x.append(sim)
                extreme_y.append(1)
                extreme_colors.append(color)
                extreme_labels.append(label)
                extreme_symbols.append(symbol)
            else:
                regular_x.append(sim)
                regular_y.append(1)
                regular_colors.append(color)
                regular_labels.append(label)
                regular_symbols.append(symbol)
        
        # Add regular points (hover labels only)
        if regular_x:
            fig.add_trace(
                go.Scatter(
                    x=regular_x,
                    y=regular_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol=regular_symbols,
                        line=dict(width=1, color='black')
                    ),
                    text=regular_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add extreme points with visible labels and leader lines
        if extreme_x:
            fig.add_trace(
                go.Scatter(
                    x=extreme_x,
                    y=extreme_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol=extreme_symbols,
                        line=dict(width=1, color='black')
                    ),
                    text=extreme_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
    
    # Add leader lines and annotations for extreme points
    if len(extreme_indices) > 0:
        # Create predefined alternating heights with variation
        # High positions with variation
        high_positions = [1.65, 1.45, 1.55, 1.35, 1.25, 1.65, 1.45, 1.55, 1.35, 1.25]
        # Low positions with variation  
        low_positions = [0.35, 0.55, 0.75, 0.65, 0.45, 0.35, 0.55, 0.75, 0.65, 0.45]
        
        # Alternate high-low pattern
        all_y_positions = []
        for i in range(10):
            all_y_positions.extend([high_positions[i], low_positions[i]])
        
        # Handle low extremes (10 lowest cosine similarities)
        for i, idx in enumerate(low_extreme_indices):
            x_pos = cosine_sims[idx]
            label = role_labels[idx]
            # Change leader lines to all red
            leader_color = 'black'
            y_label = all_y_positions[i]
            
            # Add leader line as a separate trace
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add label at the end of the line
            fig.add_annotation(
                x=x_pos,
                y=y_label,
                text=label,
                showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color,
                borderwidth=1,
                row=1, col=1
            )
        
        # Handle high extremes (10 highest cosine similarities)
        for i, idx in enumerate(high_extreme_indices):
            x_pos = cosine_sims[idx]
            label = role_labels[idx]
            # Change leader lines to all red
            leader_color = 'black'
            y_label = all_y_positions[i + 10]  # Offset by 10 to continue the pattern
            
            # Add leader line as a separate trace
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add label at the end of the line
            fig.add_annotation(
                x=x_pos,
                y=y_label,
                text=label,
                showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color,
                borderwidth=1,
                row=1, col=1
            )
    
    # Add vertical line at x=0 for both panels
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="gray",
        line_width=1,
        opacity=0.7,
        row=1, col=1
    )

    if assistant_cosine_sim is not None:
        # Add red dashed vertical line for assistant position
        fig.add_vline(x=assistant_cosine_sim, line_dash="dash", line_color="red", line_width=1, opacity=1.0)
        
        # Add Assistant label at same height as extremes
        assistant_y_position = 2  # Higher position for better visibility
        fig.add_annotation(
            x=assistant_cosine_sim,
            y=assistant_y_position,
            text="Assistant",
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red",
            borderwidth=1,
            row=1, col=1
        )
        
    fig.add_vline(
        x=0,
        line_dash="solid", 
        line_color="gray",
        line_width=1,
        opacity=0.7
    )
    

    # Update layout with single legend
    show_legend = type == "pos23" and 'pos_2' in pca_results['roles']
    fig.update_layout(
        height=700,
        title=dict(
            text=title,
            subtitle={
                "text": subtitle,
            },
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=show_legend,
        barmode='stack',  # Enable stacked bars
        legend=dict(
            x=0.0,
            y=1.04,
            xanchor='left',
            yanchor='bottom',
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    # Calculate symmetric range around 0 (not around data center)
    max_abs_value = max(abs(min(cosine_sims)), abs(max(cosine_sims)))
    x_half_width = max_abs_value * 1.1  # Add 10% padding
    
    # Update x-axes with symmetric ranges centered on 0
    fig.update_xaxes(
        range=[-x_half_width, x_half_width]
    )
    
    fig.update_xaxes(
        title_text=f"PC{pc_component+1} Cosine Similarity",
        range=[-x_half_width, x_half_width]
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="",
        showticklabels=False,
        range=[0.25, 2.5]  # Range for varied label heights with extra top space
    )
    
    fig.update_yaxes(
        title_text="Frequency",
        row=2, col=1
    )
    
    return fig

def plot_pca_projection(pca_results, role_labels, pc_component, 
                             type, assistant_activation=None, assistant_projection=None,
                             title="PCA on Role-Playing Vectors", subtitle=""):
    """
    Create a plot similar to the PC1 Normalized Projection visualization.
    Shows labels on hover for most points, with visible labels and leader lines 
    for the 20 traits at either end of the range to avoid overlap.
    
    Parameters:
    - pca_transformed: PCA-transformed data (n_samples, n_components)
    - role_labels: List of labels for each data point
    - pc_component: Which PC component to use (0-indexed, so PC1 = 0)
    - layer: Layer number for title
    
    Returns:
    - Plotly figure object
    """
    
    # Extract the specified PC component
    pc_values = pca_results['pca_transformed'][:, pc_component]
    projections = pc_values / np.linalg.norm(pc_values)  # Normalized PC values
    
    # Calculate assistant projection if provided
    assistant_normalized_projection = None
    if assistant_projection is not None:
        assistant_pc_value = assistant_projection[pc_component]
        assistant_normalized_projection = assistant_pc_value / np.linalg.norm(np.concatenate([pc_values, [assistant_pc_value]]))
    
    
    # Create colors based on vector type (pos_2 = cyan, pos_3 = blue)
    if type == "pos23" and 'pos_2' in pca_results['roles']:
        n_pos_2 = len(pca_results['roles']['pos_2'])
        colors = ['cyan'] * n_pos_2 + ['blue'] * (len(projections) - n_pos_2)
    else:
        # Default to blue for pos3 only
        colors = ['blue'] * len(projections)
    
    # Determine marker shapes based on vector type
    if type == "pos23" and 'pos_2' in pca_results['roles']:
        n_pos_2 = len(pca_results['roles']['pos_2'])
        # First n_pos_2 are pos_2 (circles), rest are pos_3 (squares)
        marker_symbols = ['circle'] * n_pos_2 + ['square'] * (len(projections) - n_pos_2)
    else:
        # Default to circles for all points
        marker_symbols = ['circle'] * len(projections)
    
    # Identify extreme traits (10 lowest and 10 highest)
    sorted_indices = np.argsort(projections)
    low_extreme_indices = sorted_indices[:10]
    high_extreme_indices = sorted_indices[-10:]
    extreme_indices = set(list(low_extreme_indices) + list(high_extreme_indices))
    
    # Create single plot figure
    fig = go.Figure()
    
    if type == "pos23" and 'pos_2' in pca_results['roles']:
        # Split points by type for legend
        n_pos_2 = len(pca_results['roles']['pos_2'])
        
        # Split regular and extreme points by type
        pos2_regular_x, pos2_regular_y, pos2_regular_colors, pos2_regular_labels = [], [], [], []
        pos3_regular_x, pos3_regular_y, pos3_regular_colors, pos3_regular_labels = [], [], [], []
        pos2_extreme_x, pos2_extreme_y, pos2_extreme_colors, pos2_extreme_labels = [], [], [], []
        pos3_extreme_x, pos3_extreme_y, pos3_extreme_colors, pos3_extreme_labels = [], [], [], []
        
        for i, (sim, color, label, symbol) in enumerate(zip(projections, colors, role_labels, marker_symbols)):
            is_pos2 = i < n_pos_2
            if i in extreme_indices:
                if is_pos2:
                    pos2_extreme_x.append(sim)
                    pos2_extreme_y.append(1)
                    pos2_extreme_colors.append(color)
                    pos2_extreme_labels.append(label)
                else:
                    pos3_extreme_x.append(sim)
                    pos3_extreme_y.append(1)
                    pos3_extreme_colors.append(color)
                    pos3_extreme_labels.append(label)
            else:
                if is_pos2:
                    pos2_regular_x.append(sim)
                    pos2_regular_y.append(1)
                    pos2_regular_colors.append(color)
                    pos2_regular_labels.append(label)
                else:
                    pos3_regular_x.append(sim)
                    pos3_regular_y.append(1)
                    pos3_regular_colors.append(color)
                    pos3_regular_labels.append(label)
        
        # Add pos_2 regular points
        if pos2_regular_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_regular_x,
                    y=pos2_regular_y,
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_regular_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add pos_2 extreme points
        if pos2_extreme_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_extreme_x,
                    y=pos2_extreme_y,
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_extreme_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add pos_3 regular points  
        if pos3_regular_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_regular_x,
                    y=pos3_regular_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_regular_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add pos_3 extreme points
        if pos3_extreme_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_extreme_x,
                    y=pos3_extreme_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_extreme_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
    else:
        # Original logic for single type
        # Split points into regular and extreme for different display modes
        regular_x, regular_y, regular_colors, regular_labels, regular_symbols = [], [], [], [], []
        extreme_x, extreme_y, extreme_colors, extreme_labels, extreme_symbols = [], [], [], [], []
        
        for i, (sim, color, label, symbol) in enumerate(zip(projections, colors, role_labels, marker_symbols)):
            if i in extreme_indices:
                extreme_x.append(sim)
                extreme_y.append(1)
                extreme_colors.append(color)
                extreme_labels.append(label)
                extreme_symbols.append(symbol)
            else:
                regular_x.append(sim)
                regular_y.append(1)
                regular_colors.append(color)
                regular_labels.append(label)
                regular_symbols.append(symbol)
        
        # Add regular points (hover labels only)
        if regular_x:
            fig.add_trace(
                go.Scatter(
                    x=regular_x,
                    y=regular_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol=regular_symbols,
                        line=dict(width=1, color='black')
                    ),
                    text=regular_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add extreme points with visible labels and leader lines
        if extreme_x:
            fig.add_trace(
                go.Scatter(
                    x=extreme_x,
                    y=extreme_y,
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol=extreme_symbols,
                        line=dict(width=1, color='black')
                    ),
                    text=extreme_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Add leader lines and annotations for extreme points
    if len(extreme_indices) > 0:
        # Create predefined alternating heights with variation
        # High positions with variation
        high_positions = [1.65, 1.45, 1.55, 1.35, 1.25, 1.65, 1.45, 1.55, 1.35, 1.25]
        # Low positions with variation  
        low_positions = [0.35, 0.55, 0.75, 0.65, 0.45, 0.35, 0.55, 0.75, 0.65, 0.45]
        
        # Alternate high-low pattern
        all_y_positions = []
        for i in range(10):
            all_y_positions.extend([high_positions[i], low_positions[i]])
        
        # Handle low extremes (10 lowest cosine similarities)
        for i, idx in enumerate(low_extreme_indices):
            x_pos = projections[idx]
            label = role_labels[idx]
            # Change leader lines to all red
            leader_color = 'black'
            y_label = all_y_positions[i]
            
            # Add leader line as a separate trace
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add label at the end of the line
            fig.add_annotation(
                x=x_pos,
                y=y_label,
                text=label,
                showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color,
                borderwidth=1,
                row=1, col=1
            )
        
        # Handle high extremes (10 highest cosine similarities)
        for i, idx in enumerate(high_extreme_indices):
            x_pos = projections[idx]
            label = role_labels[idx]
            # Change leader lines to all red
            leader_color = 'black'
            y_label = all_y_positions[i + 10]  # Offset by 10 to continue the pattern
            
            # Add leader line as a separate trace
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add label at the end of the line
            fig.add_annotation(
                x=x_pos,
                y=y_label,
                text=label,
                showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color,
                borderwidth=1,
                row=1, col=1
            )
    
    # Add vertical line at x=0 for both panels
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="gray",
        line_width=1,
        opacity=0.7,
        row=1, col=1
    )

    if assistant_normalized_projection is not None:
        # Add red dashed vertical line for assistant position
        fig.add_vline(x=assistant_normalized_projection, line_dash="dash", line_color="red", line_width=1, opacity=1.0)
        
        # Add Assistant label at same height as extremes
        assistant_y_position = 2  # Higher position for better visibility
        fig.add_annotation(
            x=assistant_normalized_projection,
            y=assistant_y_position,
            text="Assistant",
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red",
            borderwidth=1,
            row=1, col=1
        )
        
    fig.add_vline(
        x=0,
        line_dash="solid", 
        line_color="gray",
        line_width=1,
        opacity=0.7
    )
    

    # Update layout with single legend
    show_legend = type == "pos23" and 'pos_2' in pca_results['roles']
    fig.update_layout(
        height=700,
        title=dict(
            text=title,
            subtitle={
                "text": subtitle,
            },
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=show_legend,
        barmode='stack',  # Enable stacked bars
        legend=dict(
            x=0.0,
            y=1.04,
            xanchor='left',
            yanchor='bottom',
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    # Calculate symmetric range around 0 (not around data center)
    max_abs_value = max(abs(min(projections)), abs(max(projections)))
    x_half_width = max_abs_value * 1.1  # Add 10% padding
    
    # Update x-axes with symmetric ranges centered on 0
    fig.update_xaxes(
        range=[-x_half_width, x_half_width]
    )
    
    fig.update_xaxes(
        title_text=f"PC{pc_component+1} Normalized Projection",
        range=[-x_half_width, x_half_width]
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="",
        showticklabels=False,
        range=[0.25, 2.5]  # Range for varied label heights with extra top space
    )
    
    fig.update_yaxes(
        title_text="Frequency",
        row=2, col=1
    )
    
    return fig

def plot_3d_pca(pca_results, role_labels, type, assistant_projection=None,
                title="Role-Playing Vectors in 3D PC Space", subtitle=""):
    # Create 3D scatter plot if we have enough components
    pca_transformed = pca_results['pca_transformed']
    variance_explained = pca_results['variance_explained']

    if type == "pos23" and 'pos_2' in pca_results['roles']:
        # Split into two traces for legend
        n_pos_2 = len(pca_results['roles']['pos_2'])
        
        # Select subset of points to display labels (50% of each type)
        pos2_label_indices = list(range(0, n_pos_2, 3))  # Every other point for pos_2
        pos3_label_indices = list(range(n_pos_2, len(role_labels), 3))  # Every other point for pos_3
        
        # Create text arrays with labels only for selected points
        pos2_text = [role_labels[i] if i in pos2_label_indices else '' for i in range(n_pos_2)]
        pos3_text = [role_labels[i] if i in pos3_label_indices else '' for i in range(n_pos_2, len(role_labels))]
        
        # Add pos_2 trace (cyan circles)
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=pca_transformed[:n_pos_2, 0],
            y=pca_transformed[:n_pos_2, 1], 
            z=pca_transformed[:n_pos_2, 2],
            mode='markers+text',
            text=pos2_text,
            textposition='top center',
            textfont=dict(size=6),
            marker=dict(
                size=3,
                color='cyan',
                symbol='circle'
            ),
            name='Somewhat Role-Playing',
            hovertemplate='<b>%{hovertext}</b><br>' +
                        f'PC1: %{{x:.3f}}<br>' +
                        f'PC2: %{{y:.3f}}<br>' +
                        f'PC3: %{{z:.3f}}<br>' +
                        '<extra></extra>',
            hovertext=role_labels[:n_pos_2]
        )])
        
        # Add pos_3 trace (blue squares)
        fig_3d.add_trace(go.Scatter3d(
            x=pca_transformed[n_pos_2:, 0],
            y=pca_transformed[n_pos_2:, 1], 
            z=pca_transformed[n_pos_2:, 2],
            mode='markers+text',
            text=pos3_text,
            textposition='top center',
            textfont=dict(size=6),
            marker=dict(
                size=3,
                color='blue',
                symbol='square'
            ),
            name='Fully Role-Playing',
            hovertemplate='<b>%{hovertext}</b><br>' +
                        f'PC1: %{{x:.3f}}<br>' +
                        f'PC2: %{{y:.3f}}<br>' +
                        f'PC3: %{{z:.3f}}<br>' +
                        '<extra></extra>',
            hovertext=role_labels[n_pos_2:]
        ))
    else:
        # Default single trace - show labels for 50% of points
        label_indices = list(range(0, len(role_labels), 3))  # Every other point
        text_labels = [role_labels[i] if i in label_indices else '' for i in range(len(role_labels))]
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=pca_transformed[:, 0],
            y=pca_transformed[:, 1], 
            z=pca_transformed[:, 2],
            mode='markers+text',
            text=text_labels,
            textposition='top center',
            textfont=dict(size=6),
            marker=dict(
                size=3,
                color='blue',
            ),
            showlegend=False,
            hovertemplate='<b>%{hovertext}</b><br>' +
                        f'PC1: %{{x:.3f}}<br>' +
                        f'PC2: %{{y:.3f}}<br>' +
                        f'PC3: %{{z:.3f}}<br>' +
                        '<extra></extra>',
            hovertext=role_labels
        )])
    
    if assistant_projection is not None:
        fig_3d.add_trace(go.Scatter3d(
        x=[assistant_projection[0]],
        y=[assistant_projection[1]],
        z=[assistant_projection[2]],
        mode='markers+text',
        text=['Assistant'],
        textposition='top center',
        textfont=dict(size=8, color='black'),
        marker=dict(
            size=5,  # 2 sizes bigger than trait dots (3 -> 5)
            color='red',
            opacity=1.0
        ),
        showlegend=False,
        hovertemplate='<b>Assistant</b><br>' +
                    f'PC1: %{{x:.3f}}<br>' +
                    f'PC2: %{{y:.3f}}<br>' +
                    f'PC3: %{{z:.3f}}<br>' +
                    '<extra></extra>'
    ))
    
    fig_3d.update_layout(
        title={
            "text": title,
            "subtitle": {
                "text": subtitle,
            },
        },
        scene=dict(
            xaxis_title=f'PC1 ({variance_explained[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({variance_explained[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({variance_explained[2]*100:.1f}%)'
        ),
        legend=dict(
            itemsizing='constant',
            itemwidth=30,
        ),
        width=1000,
        height=800
    )
    
    return fig_3d


def plot_pc(pca_results, role_labels, pc_component, layer=None, dir=None,
           assistant_activation=None, assistant_projection=None,
           title="PCA Analysis", subtitle=""):
    """
    Create a combined plot with cosine similarity (top) and normalized projection (bottom).
    Shows histograms directly on the plots at y=1 level. Auto-detects role types.
    
    Parameters:
    - pca_results: Dictionary containing PCA results and vectors
    - role_labels: List of labels for each data point
    - pc_component: Which PC component to use (0-indexed, so PC1 = 0)
    - layer: Layer number (used for assistant_activation if provided)
    - dir: Directory parameter (unused, kept for compatibility)
    - assistant_activation: Optional assistant activation for cosine similarity comparison
    - assistant_projection: Optional assistant projection for projection comparison
    - title: Main title for the plot
    - subtitle: Subtitle for the plot
    
    Returns:
    - Plotly figure object
    """
    
    # Auto-detect role types
    role_info = _detect_role_types(pca_results)
    
    # Extract the specified PC component for cosine similarity
    pc_direction = pca_results['pca'].components_[pc_component]
    vectors = torch.stack(role_info['vectors'])[:, layer, :].float().numpy()
    scaled_vectors = pca_results['scaler'].transform(vectors)
    pc_direction_norm = pc_direction / np.linalg.norm(pc_direction)
    vectors_norm = scaled_vectors / np.linalg.norm(scaled_vectors, axis=1, keepdims=True)
    cosine_sims = vectors_norm @ pc_direction_norm.T
    
    # Calculate assistant cosine similarity if provided
    assistant_cosine_sim = None
    if assistant_activation is not None and layer is not None:
        assistant_layer_activation = assistant_activation[layer, :].float().numpy().reshape(1, -1)
        asst_scaled = pca_results['scaler'].transform(assistant_layer_activation)
        asst_scaled_norm = asst_scaled / np.linalg.norm(asst_scaled)
        assistant_cosine_sim = asst_scaled_norm @ pc_direction.T
        assistant_cosine_sim = assistant_cosine_sim[0]
    
    # Extract projection data
    pc_values = pca_results['pca_transformed'][:, pc_component]
    projections = pc_values / np.linalg.norm(pc_values)

    if assistant_projection is not None:
        assistant_pc_value = assistant_projection[pc_component]
        assistant_normalized_projection = assistant_pc_value / np.linalg.norm(np.concatenate([pc_values, [assistant_pc_value]]))
    
    # Identify extreme points for both plots
    cosine_sorted_indices = np.argsort(cosine_sims)
    cosine_low_extreme = cosine_sorted_indices[:10]
    cosine_high_extreme = cosine_sorted_indices[-10:]
    cosine_extreme_indices = set(list(cosine_low_extreme) + list(cosine_high_extreme))
    
    proj_sorted_indices = np.argsort(projections)
    proj_low_extreme = proj_sorted_indices[:10]
    proj_high_extreme = proj_sorted_indices[-10:]
    proj_extreme_indices = set(list(proj_low_extreme) + list(proj_high_extreme))
    
    # Create subplot figure
    fig = sp.make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.1,
        subplot_titles=[
            f'PC{pc_component+1} Cosine Similarity',
            f'PC{pc_component+1} Normalized Projection'
        ]
    )
    
    # Label positions for extreme points
    high_positions = [1.45, 1.9, 1.6, 1.3, 1.75, 1.45, 1.9, 1.6, 1.3, 1.75]
    low_positions = [0.6, 0.15, 0.45, 0.75, 0.30, 0.6, 0.15, 0.45, 0.75, 0.30]
    cosine_y_positions = []
    for i in range(10):
        cosine_y_positions.extend([high_positions[i], low_positions[i]])
    proj_y_positions = []
    for i in range(10):
        proj_y_positions.extend([high_positions[i], low_positions[i]])
    
    # === TOP SUBPLOT: COSINE SIMILARITY ===
    
    if role_info['has_pos2'] and role_info['has_pos3']:
        # Split points by type for legend
        n_pos_2 = role_info['n_pos_2']
        
        # Split regular and extreme points by type for cosine similarity
        pos2_regular_cosine_x, pos2_regular_cosine_labels = [], []
        pos3_regular_cosine_x, pos3_regular_cosine_labels = [], []
        pos2_extreme_cosine_x, pos2_extreme_cosine_labels = [], []
        pos3_extreme_cosine_x, pos3_extreme_cosine_labels = [], []
        
        for i, (sim, label) in enumerate(zip(cosine_sims, role_labels)):
            is_pos2 = i < n_pos_2
            if i in cosine_extreme_indices:
                if is_pos2:
                    pos2_extreme_cosine_x.append(sim)
                    pos2_extreme_cosine_labels.append(label)
                else:
                    pos3_extreme_cosine_x.append(sim)
                    pos3_extreme_cosine_labels.append(label)
            else:
                if is_pos2:
                    pos2_regular_cosine_x.append(sim)
                    pos2_regular_cosine_labels.append(label)
                else:
                    pos3_regular_cosine_x.append(sim)
                    pos3_regular_cosine_labels.append(label)
        
        # Add pos_2 regular points
        if pos2_regular_cosine_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_regular_cosine_x,
                    y=[1] * len(pos2_regular_cosine_x),
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_regular_cosine_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add pos_2 extreme points
        if pos2_extreme_cosine_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_extreme_cosine_x,
                    y=[1] * len(pos2_extreme_cosine_x),
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_extreme_cosine_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add pos_3 regular points  
        if pos3_regular_cosine_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_regular_cosine_x,
                    y=[1] * len(pos3_regular_cosine_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_regular_cosine_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add pos_3 extreme points
        if pos3_extreme_cosine_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_extreme_cosine_x,
                    y=[1] * len(pos3_extreme_cosine_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_extreme_cosine_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
            
    else:
        # Single type logic for cosine similarity
        regular_cosine_x, regular_cosine_labels = [], []
        extreme_cosine_x, extreme_cosine_labels = [], []
        
        for i, (sim, label) in enumerate(zip(cosine_sims, role_labels)):
            if i in cosine_extreme_indices:
                extreme_cosine_x.append(sim)
                extreme_cosine_labels.append(label)
            else:
                regular_cosine_x.append(sim)
                regular_cosine_labels.append(label)
        
        # Add regular points (hover labels only)
        if regular_cosine_x:
            fig.add_trace(
                go.Scatter(
                    x=regular_cosine_x,
                    y=[1] * len(regular_cosine_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=regular_cosine_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
        
        # Add extreme points with visible labels and leader lines
        if extreme_cosine_x:
            fig.add_trace(
                go.Scatter(
                    x=extreme_cosine_x,
                    y=[1] * len(extreme_cosine_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=extreme_cosine_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
                )
            )
    
    # Add leader lines and annotations for extreme cosine points
    if len(cosine_extreme_indices) > 0:
        for i, idx in enumerate(cosine_low_extreme):
            x_pos = cosine_sims[idx]
            label = role_labels[idx]
            leader_color = 'black'
            y_label = cosine_y_positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color, borderwidth=1,
                row=1, col=1
            )
        
        for i, idx in enumerate(cosine_high_extreme):
            x_pos = cosine_sims[idx]
            label = role_labels[idx]
            leader_color = 'black'
            y_label = cosine_y_positions[i + 10]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color, borderwidth=1,
                row=1, col=1
            )
    
    # Add cosine similarity histogram as opaque bars
    nbins = 30
    if role_info['has_pos2'] and role_info['has_pos3']:
        # Split cosine similarities by type
        n_pos_2 = role_info['n_pos_2']
        pos2_cosine_sims = cosine_sims[:n_pos_2]
        pos3_cosine_sims = cosine_sims[n_pos_2:]
        
        # Calculate histogram bins manually
        min_val = min(cosine_sims)
        max_val = max(cosine_sims)
        bin_edges = np.linspace(min_val, max_val, nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Count occurrences in each bin for both types
        pos2_counts, _ = np.histogram(pos2_cosine_sims, bins=bin_edges)
        pos3_counts, _ = np.histogram(pos3_cosine_sims, bins=bin_edges)
        
        # Scale histogram heights
        max_hist_height = 0.9
        max_count = max(np.max(pos2_counts), np.max(pos3_counts))
        pos2_scaled_counts = (pos2_counts / max_count) * max_hist_height
        pos3_scaled_counts = (pos3_counts / max_count) * max_hist_height
        
        # Add stacked bars at marker level - let Plotly handle stacking automatically
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=pos2_scaled_counts,
                width=bin_width * 0.9,
                marker_color='cyan',
                opacity=0.7,
                name='Somewhat Role-Playing',
                legendgroup='pos2',
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=pos3_scaled_counts,
                width=bin_width * 0.9,
                marker_color='blue',
                opacity=0.7,
                name='Fully Role-Playing',
                legendgroup='pos3',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    else:
        # Single histogram
        cosine_hist_counts, cosine_bin_edges = np.histogram(cosine_sims, bins=nbins)
        cosine_bin_centers = (cosine_bin_edges[:-1] + cosine_bin_edges[1:]) / 2
        cosine_bin_width = cosine_bin_edges[1] - cosine_bin_edges[0]
        
        # Scale histogram heights
        max_hist_height = 0.9
        cosine_scaled_counts = (cosine_hist_counts / np.max(cosine_hist_counts)) * max_hist_height
        
        fig.add_trace(
            go.Bar(
                x=cosine_bin_centers,
                y=cosine_scaled_counts,
                base=1,  # Position bars at marker line level
                width=cosine_bin_width * 0.8,
                marker_color='blue',
                opacity=0.7,
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Add vertical line at x=0 and assistant line for cosine similarity
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.7)
    
    if assistant_cosine_sim is not None:
        fig.add_vline(x=assistant_cosine_sim, line_dash="dash", line_color="red", line_width=1, opacity=1.0)
        fig.add_annotation(
            x=assistant_cosine_sim, y=2.25, text="Assistant", showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red", borderwidth=1,
            row=1, col=1
        )
    
    # === BOTTOM SUBPLOT: PROJECTION ===
    
    if role_info['has_pos2'] and role_info['has_pos3']:
        # Split points by type for projection
        n_pos_2 = role_info['n_pos_2']
        
        # Split regular and extreme points by type for projection
        pos2_regular_proj_x, pos2_regular_proj_labels = [], []
        pos3_regular_proj_x, pos3_regular_proj_labels = [], []
        pos2_extreme_proj_x, pos2_extreme_proj_labels = [], []
        pos3_extreme_proj_x, pos3_extreme_proj_labels = [], []
        
        for i, (proj, label) in enumerate(zip(projections, role_labels)):
            is_pos2 = i < n_pos_2
            if i in proj_extreme_indices:
                if is_pos2:
                    pos2_extreme_proj_x.append(proj)
                    pos2_extreme_proj_labels.append(label)
                else:
                    pos3_extreme_proj_x.append(proj)
                    pos3_extreme_proj_labels.append(label)
            else:
                if is_pos2:
                    pos2_regular_proj_x.append(proj)
                    pos2_regular_proj_labels.append(label)
                else:
                    pos3_regular_proj_x.append(proj)
                    pos3_regular_proj_labels.append(label)
        
        # Add pos_2 regular points
        if pos2_regular_proj_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_regular_proj_x,
                    y=[1] * len(pos2_regular_proj_x),
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_regular_proj_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add pos_2 extreme points
        if pos2_extreme_proj_x:
            fig.add_trace(
                go.Scatter(
                    x=pos2_extreme_proj_x,
                    y=[1] * len(pos2_extreme_proj_x),
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=pos2_extreme_proj_labels,
                    name='Somewhat Role-Playing',
                    legendgroup='pos2',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add pos_3 regular points  
        if pos3_regular_proj_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_regular_proj_x,
                    y=[1] * len(pos3_regular_proj_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_regular_proj_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add pos_3 extreme points
        if pos3_extreme_proj_x:
            fig.add_trace(
                go.Scatter(
                    x=pos3_extreme_proj_x,
                    y=[1] * len(pos3_extreme_proj_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='square',
                        line=dict(width=1, color='black')
                    ),
                    text=pos3_extreme_proj_labels,
                    name='Fully Role-Playing',
                    legendgroup='pos3',
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
    else:
        # Single type logic for projection
        regular_proj_x, regular_proj_labels = [], []
        extreme_proj_x, extreme_proj_labels = [], []
        
        for i, (proj, label) in enumerate(zip(projections, role_labels)):
            if i in proj_extreme_indices:
                extreme_proj_x.append(proj)
                extreme_proj_labels.append(label)
            else:
                regular_proj_x.append(proj)
                regular_proj_labels.append(label)
        
        # Add regular points
        if regular_proj_x:
            fig.add_trace(
                go.Scatter(
                    x=regular_proj_x,
                    y=[1] * len(regular_proj_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=regular_proj_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add extreme points
        if extreme_proj_x:
            fig.add_trace(
                go.Scatter(
                    x=extreme_proj_x,
                    y=[1] * len(extreme_proj_x),
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=1.0,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    text=extreme_proj_labels,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Add leader lines and annotations for extreme projection points
    if len(proj_extreme_indices) > 0:
        for i, idx in enumerate(proj_low_extreme):
            x_pos = projections[idx]
            label = role_labels[idx]
            leader_color = 'black'
            y_label = proj_y_positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color, borderwidth=1,
                row=2, col=1
            )
        
        for i, idx in enumerate(proj_high_extreme):
            x_pos = projections[idx]
            label = role_labels[idx]
            leader_color = 'black'
            y_label = proj_y_positions[i + 10]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color=leader_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color=leader_color),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=leader_color, borderwidth=1,
                row=2, col=1
            )
    
    # Add projection histogram as opaque bars
    if role_info['has_pos2'] and role_info['has_pos3']:
        # Split projection by type
        pos2_projections = projections[:n_pos_2]
        pos3_projections = projections[n_pos_2:]
        
        # Calculate histogram bins manually
        min_val = min(projections)
        max_val = max(projections)
        bin_edges = np.linspace(min_val, max_val, nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Count occurrences in each bin for both types
        pos2_counts, _ = np.histogram(pos2_projections, bins=bin_edges)
        pos3_counts, _ = np.histogram(pos3_projections, bins=bin_edges)
        
        # Scale histogram heights
        max_hist_height = 0.9
        max_count = max(np.max(pos2_counts), np.max(pos3_counts))
        pos2_scaled_counts = (pos2_counts / max_count) * max_hist_height
        pos3_scaled_counts = (pos3_counts / max_count) * max_hist_height
        
        # Add stacked bars at marker level - let Plotly handle stacking automatically
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=pos2_scaled_counts,
                width=bin_width * 0.9,
                marker_color='cyan',
                opacity=0.7,
                name='Somewhat Role-Playing',
                legendgroup='pos2',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=pos3_scaled_counts,
                width=bin_width * 0.9,
                marker_color='blue',
                opacity=0.7,
                name='Fully Role-Playing',
                legendgroup='pos3',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    else:
        # Single histogram
        proj_hist_counts, proj_bin_edges = np.histogram(projections, bins=nbins)
        proj_bin_centers = (proj_bin_edges[:-1] + proj_bin_edges[1:]) / 2
        proj_bin_width = proj_bin_edges[1] - proj_bin_edges[0]
        
        # Scale histogram heights
        max_hist_height = 0.9
        proj_scaled_counts = (proj_hist_counts / np.max(proj_hist_counts)) * max_hist_height
        
        fig.add_trace(
            go.Bar(
                x=proj_bin_centers,
                y=proj_scaled_counts,
                width=proj_bin_width * 0.8,
                marker_color='blue',
                opacity=0.7,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    # Add vertical line at x=0 and assistant line for projection
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.7, row=2, col=1)
    
    if assistant_projection is not None:
        fig.add_vline(x=assistant_normalized_projection, line_dash="dash", line_color="red", line_width=1, opacity=1.0, row=2, col=1)
        fig.add_annotation(
            x=assistant_normalized_projection, y=2.25, text="Assistant", showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red", borderwidth=1,
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=dict(
            text=title,
            subtitle={"text": subtitle},
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=role_info['show_legend'],
        barmode='stack',
        legend=dict(
            x=0.0,
            y=1.04,
            xanchor='left',
            yanchor='bottom',
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    # Update x-axes ranges
    cosine_max_abs = max(abs(min(cosine_sims)), abs(max(cosine_sims)))
    cosine_x_width = cosine_max_abs * 1.1
    
    proj_max_abs = max(abs(min(projections)), abs(max(projections)))
    proj_x_width = proj_max_abs * 1.1
    
    fig.update_xaxes(range=[-cosine_x_width, cosine_x_width], row=1, col=1)
    fig.update_xaxes(range=[-proj_x_width, proj_x_width], row=2, col=1)
    
    # Update y-axes
    fig.update_yaxes(
        showticklabels=False,
        range=[0, 2.5],
        row=1, col=1
    )
    fig.update_yaxes(
        showticklabels=False,
        range=[0, 2.5],
        row=2, col=1
    )
    
    return fig