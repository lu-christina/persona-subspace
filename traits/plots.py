import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


def plot_pca_cosine_similarity(pca_results, trait_labels, pc_component, 
                             layer, assistant_activation=None, 
                             title="PCA on Trait Vectors", 
                             subtitle=""):
    """
    Create a plot similar to the PC1 Cosine Similarity visualization.
    Shows labels on hover for most points, with visible labels and leader lines 
    for the 20 traits at either end of the range to avoid overlap.
    
    Parameters:
    - pca_results: Dictionary containing PCA results and vectors
    - trait_labels: List of labels for each data point
    - pc_component: Which PC component to use (0-indexed, so PC1 = 0)
    - layer: Layer number for title (unused if custom title/subtitle provided)
    - assistant_activation: Optional assistant activation for comparison
    - title: Main title for the plot (default: "PCA on Trait Vectors from Mean Response Activations")
    - subtitle: Subtitle for the plot (default: "")
    
    Returns:
    - Plotly figure object
    """
    
    # Extract the specified PC component
    pc_direction = torch.from_numpy(pca_results['pca'].components_[pc_component])

    # get raw vectors - use pos_neg_50 for traits
    vectors = torch.stack(pca_results['vectors']['pos_neg_50'])[:, layer, :]
    pc_direction = F.normalize(pc_direction.unsqueeze(0), dim=1)
    vectors = F.normalize(vectors, dim=1)

    cosine_sims = vectors.float() @ pc_direction.float().T
    cosine_sims = cosine_sims.squeeze(1).numpy()
    
    if assistant_activation is not None:
        assistant_activation = assistant_activation[layer, :]
        assistant_activation = F.normalize(assistant_activation.unsqueeze(0), dim=1)
        assistant_cosine_sim = assistant_activation.float() @ pc_direction.float().T
        assistant_cosine_sim = assistant_cosine_sim.squeeze(1).numpy()
        assistant_cosine_sim = assistant_cosine_sim[0]
    
    # Create colors - all green for traits
    colors = ['limegreen'] * len(cosine_sims)
    
    # All markers are diamonds
    marker_symbols = ['diamond'] * len(cosine_sims)
    
    # Identify extreme traits (10 lowest and 10 highest)
    sorted_indices = np.argsort(cosine_sims)
    low_extreme_indices = sorted_indices[:10]
    high_extreme_indices = sorted_indices[-10:]
    extreme_indices = set(list(low_extreme_indices) + list(high_extreme_indices))
    
    # Create subplot figure
    fig = sp.make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=[
            f'PC{pc_component+1} Cosine Similarity',
            'Trait Frequency Distribution'
        ]
    )
    
    # Split points into regular and extreme for different display modes
    regular_x, regular_y, regular_colors, regular_labels, regular_symbols = [], [], [], [], []
    extreme_x, extreme_y, extreme_colors, extreme_labels, extreme_symbols = [], [], [], [], []
    
    for i, (sim, color, label, symbol) in enumerate(zip(cosine_sims, colors, trait_labels, marker_symbols)):
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
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol=regular_symbols,
                    line=dict(width=1, color='black')
                ),
                text=regular_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add extreme points with visible labels and leader lines
    if extreme_x:
        fig.add_trace(
            go.Scatter(
                x=extreme_x,
                y=extreme_y,
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol=extreme_symbols,
                    line=dict(width=1, color='black')
                ),
                text=extreme_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
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
            label = trait_labels[idx]
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
                ),
                row=1, col=1
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
            label = trait_labels[idx]
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
                ),
                row=1, col=1
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

    if assistant_activation is not None:
        # Add red dashed vertical line for assistant position
        fig.add_vline(x=assistant_cosine_sim, line_dash="dash", line_color="red", line_width=1, opacity=1.0, row=1, col=1)
        
        # Add Assistant label at same height as extremes
        assistant_y_position = 1.6  # Same as first high position
        fig.add_annotation(
            x=assistant_cosine_sim,
            y=assistant_y_position,
            text="Assistant",
            showarrow=False,
            font=dict(size=10, color="red"),
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
        opacity=0.7,
        row=2, col=1
    )
    
    # Bottom panel: Histogram
    fig.add_trace(
        go.Histogram(
            x=cosine_sims,
            nbinsx=30,
            opacity=1.0,
            marker_color='limegreen',
            showlegend=False
        ),
        row=2, col=1
    )

    # Update layout
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
        showlegend=False
    )
    
    # Calculate symmetric range around 0 (not around data center)
    max_abs_value = max(abs(min(cosine_sims)), abs(max(cosine_sims)))
    x_half_width = max_abs_value * 1.1  # Add 10% padding
    
    # Update x-axes with symmetric ranges centered on 0
    fig.update_xaxes(
        row=1, col=1,
        range=[-x_half_width, x_half_width]
    )
    
    fig.update_xaxes(
        title_text=f"PC{pc_component+1} Cosine Similarity",
        row=2, col=1,
        range=[-x_half_width, x_half_width]
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="",
        showticklabels=False,
        row=1, col=1,
        range=[0.25, 1.75]  # Range for varied label heights
    )
    
    fig.update_yaxes(
        title_text="Frequency",
        row=2, col=1
    )
    
    return fig

def plot_pca_projection(pca_results, trait_labels, pc_component, 
                             assistant_activation=None, 
                             title="PCA on Trait Vectors from Mean Response Activations", 
                             subtitle=""):
    """
    Create a plot similar to the PC1 Normalized Projection visualization.
    Shows labels on hover for most points, with visible labels and leader lines 
    for the 20 traits at either end of the range to avoid overlap.
    
    Parameters:
    - pca_results: Dictionary containing PCA results and vectors
    - trait_labels: List of labels for each data point
    - pc_component: Which PC component to use (0-indexed, so PC1 = 0)
    - layer: Layer number for title (unused if custom title/subtitle provided)
    - assistant_activation: Optional assistant activation for comparison
    - title: Main title for the plot (default: "PCA on Trait Vectors from Mean Response Activations")
    - subtitle: Subtitle for the plot (default: "")
    
    Returns:
    - Plotly figure object
    """
    
    # Extract the specified PC component
    pc_values = pca_results['pca_transformed'][:, pc_component]
    if assistant_activation is not None:
        assistant_pc_value = assistant_activation[pc_component]
    
    # Calculate normalized projections
    projections = pc_values / np.linalg.norm(pc_values)  # Normalized PC values
    if assistant_activation is not None:
        assistant_projection = assistant_pc_value / np.linalg.norm(np.concatenate([pc_values, [assistant_pc_value]]))
    
    # Create colors - all blue for traits
    colors = ['limegreen'] * len(projections)
    
    # All markers are circles
    marker_symbols = ['circle'] * len(projections)
    
    # Identify extreme traits (10 lowest and 10 highest)
    sorted_indices = np.argsort(projections)
    low_extreme_indices = sorted_indices[:10]
    high_extreme_indices = sorted_indices[-10:]
    extreme_indices = set(list(low_extreme_indices) + list(high_extreme_indices))
    
    # Create subplot figure
    fig = sp.make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=[
            f'PC{pc_component+1} Normalized Projection',
            'Trait Frequency Distribution'
        ]
    )
    
    # Split points into regular and extreme for different display modes
    regular_x, regular_y, regular_colors, regular_labels, regular_symbols = [], [], [], [], []
    extreme_x, extreme_y, extreme_colors, extreme_labels, extreme_symbols = [], [], [], [], []
    
    for i, (sim, color, label, symbol) in enumerate(zip(projections, colors, trait_labels, marker_symbols)):
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
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol=regular_symbols,
                    line=dict(width=1, color='black')
                ),
                text=regular_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add extreme points with visible labels and leader lines
    if extreme_x:
        fig.add_trace(
            go.Scatter(
                x=extreme_x,
                y=extreme_y,
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol=extreme_symbols,
                    line=dict(width=1, color='black')
                ),
                text=extreme_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>Normalized Projection: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
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
        
        # Handle low extremes (10 lowest projections)
        for i, idx in enumerate(low_extreme_indices):
            x_pos = projections[idx]
            label = trait_labels[idx]
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
                ),
                row=1, col=1
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
        
        # Handle high extremes (10 highest projections)
        for i, idx in enumerate(high_extreme_indices):
            x_pos = projections[idx]
            label = trait_labels[idx]
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
                ),
                row=1, col=1
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

    if assistant_activation is not None:
        # Add red dashed vertical line for assistant position
        fig.add_vline(x=assistant_projection, line_dash="dash", line_color="red", line_width=1, opacity=1.0, row=1, col=1)
        
        # Add Assistant label at same height as extremes
        assistant_y_position = 1.6  # Same as first high position
        fig.add_annotation(
            x=assistant_projection,
            y=assistant_y_position,
            text="Assistant",
            showarrow=False,
            font=dict(size=10, color="red"),
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
        opacity=0.7,
        row=2, col=1
    )
    
    # Bottom panel: Histogram
    fig.add_trace(
        go.Histogram(
            x=projections,
            nbinsx=30,
            opacity=1.0,
            marker_color='limegreen',
            showlegend=False
        ),
        row=2, col=1
    )

    # Update layout
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
        showlegend=False
    )
    
    # Calculate symmetric range around 0 (not around data center)
    max_abs_value = max(abs(min(projections)), abs(max(projections)))
    x_half_width = max_abs_value * 1.1  # Add 10% padding
    
    # Update x-axes with symmetric ranges centered on 0
    fig.update_xaxes(
        row=1, col=1,
        range=[-x_half_width, x_half_width]
    )
    
    fig.update_xaxes(
        title_text=f"PC{pc_component+1} Normalized Projection",
        row=2, col=1,
        range=[-x_half_width, x_half_width]
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="",
        showticklabels=False,
        row=1, col=1,
        range=[0.25, 1.75]  # Range for varied label heights
    )
    
    fig.update_yaxes(
        title_text="Frequency",
        row=2, col=1
    )
    
    return fig

def plot_3d_pca(pca_transformed, variance_explained, trait_labels, assistant_projection=None,
                title="Trait Vectors in 3D Principal Component Space", subtitle=""
                ):
    # Create 3D scatter plot if we have enough components

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=pca_transformed[:, 0],
        y=pca_transformed[:, 1], 
        z=pca_transformed[:, 2],
        mode='markers+text',
        text=trait_labels,
        textposition='top center',
        textfont=dict(size=6),
        marker=dict(
            size=3,
            color=['limegreen'] * len(trait_labels),
            line=dict(width=2, color='black')
        ),
        hovertemplate='<b>%{text}</b><br>' +
                    f'PC1: %{{x:.3f}}<br>' +
                    f'PC2: %{{y:.3f}}<br>' +
                    f'PC3: %{{z:.3f}}<br>' +
                    '<extra></extra>'
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
        width=1000,
        height=800
    )
    
    return fig_3d
