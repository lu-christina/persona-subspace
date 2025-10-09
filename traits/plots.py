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
    
    # Create single plot figure
    fig = go.Figure()
    
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
        high_positions = [1.45, 1.9, 1.6, 1.3, 1.75, 1.45, 1.9, 1.6, 1.3, 1.75]
        # Low positions with variation  
        low_positions = [0.6, 0.15, 0.45, 0.75, 0.30, 0.6, 0.15, 0.45, 0.75, 0.30]
        
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
        opacity=0.7,
        row=2, col=1
    )
    
    # Add histogram as opaque bars at marker line
    nbins = 30
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
            width=cosine_bin_width * 0.8,
            marker_color='limegreen',
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
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
        range=[0.25, 2.5]  # Range for varied label heights with extra top space
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
        high_positions = [1.45, 1.9, 1.6, 1.3, 1.75, 1.45, 1.9, 1.6, 1.3, 1.75]
        # Low positions with variation  
        low_positions = [0.6, 0.15, 0.45, 0.75, 0.30, 0.6, 0.15, 0.45, 0.75, 0.30]
        
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
        assistant_y_position = 2  # Higher position for better visibility
        fig.add_annotation(
            x=assistant_projection,
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
        opacity=0.7,
        row=2, col=1
    )
    
    # Add histogram as opaque bars at marker line  
    nbins = 30
    proj_hist_counts, proj_bin_edges = np.histogram(projections, bins=nbins)
    proj_bin_centers = (proj_bin_edges[:-1] + proj_bin_edges[1:]) / 2
    proj_bin_width = proj_bin_edges[1] - proj_bin_edges[0]
    
    # Scale histogram heights to fit at marker line level
    max_hist_height = 0.5
    proj_scaled_counts = (proj_hist_counts / np.max(proj_hist_counts)) * max_hist_height
    
    fig.add_trace(
        go.Bar(
            x=proj_bin_centers,
            y=proj_scaled_counts,
            width=proj_bin_width * 0.8,
            marker_color='limegreen',
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
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
        range=[0.25, 2.5]  # Range for varied label heights with extra top space
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

def plot_pc(pca_results, trait_labels, pc_component, layer=None, 
           assistant_activation=None, assistant_projection=None,
           title="PCA on Trait Vectors", subtitle="", scaled=True):
    """
    Create a combined plot with cosine similarity (left) and normalized projection (right).
    Shows histograms directly on the plots at y=1 level.
    
    Parameters:
    - pca_results: Dictionary containing PCA results and vectors
    - trait_labels: List of labels for each data point
    - pc_component: Which PC component to use (0-indexed, so PC1 = 0)
    - layer: Layer number (used for assistant_activation if provided)
    - assistant_activation: Optional assistant activation for cosine similarity comparison
    - assistant_projection: Optional assistant projection for projection comparison
    - title: Main title for the plot
    - subtitle: Subtitle for the plot
    
    Returns:
    - Plotly figure object
    """
    
    # Extract the specified PC component for cosine similarity
    pc_direction = pca_results['pca'].components_[pc_component]
    vectors = torch.stack(pca_results['vectors']['pos_neg_50'])[:, layer, :].float().numpy()
    if scaled:
        scaled_vectors = pca_results['scaler'].transform(vectors)

        # change to numpy if not
        if not isinstance(scaled_vectors, np.ndarray):
            scaled_vectors = scaled_vectors.numpy()
    else:
        scaled_vectors = vectors
    pc_direction_norm = pc_direction / np.linalg.norm(pc_direction)
    vectors_norm = scaled_vectors / np.linalg.norm(scaled_vectors, axis=1, keepdims=True)
    cosine_sims = vectors_norm @ pc_direction_norm.T
    
    # Calculate assistant cosine similarity if provided
    assistant_cosine_sim = None
    if assistant_activation is not None and layer is not None:
        assistant_layer_activation = assistant_activation[layer, :].float().numpy().reshape(1, -1)
        if scaled:
            asst_scaled = pca_results['scaler'].transform(assistant_layer_activation)
        else:
            asst_scaled = assistant_layer_activation
        asst_scaled_norm = asst_scaled / np.linalg.norm(asst_scaled)
        assistant_cosine_sim = asst_scaled_norm @ pc_direction.T
        assistant_cosine_sim = assistant_cosine_sim[0]
    
    # Extract projection data
    projections = pca_results['pca_transformed'][:, pc_component]

    if assistant_projection is not None:
        assistant_pc_value = assistant_projection[pc_component]
    
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
            f'PC{pc_component+1} Projection'
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
    
    # === LEFT SUBPLOT: COSINE SIMILARITY ===
    
    # Add regular cosine similarity points
    regular_cosine_x, regular_cosine_labels = [], []
    extreme_cosine_x, extreme_cosine_labels = [], []
    
    for i, (sim, label) in enumerate(zip(cosine_sims, trait_labels)):
        if i in cosine_extreme_indices:
            extreme_cosine_x.append(sim)
            extreme_cosine_labels.append(label)
        else:
            regular_cosine_x.append(sim)
            regular_cosine_labels.append(label)
    
    # Regular points
    if regular_cosine_x:
        fig.add_trace(
            go.Scatter(
                x=regular_cosine_x,
                y=[1] * len(regular_cosine_x),
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=1, color='black')
                ),
                text=regular_cosine_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Extreme points with labels
    if extreme_cosine_x:
        fig.add_trace(
            go.Scatter(
                x=extreme_cosine_x,
                y=[1] * len(extreme_cosine_x),
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=1, color='black')
                ),
                text=extreme_cosine_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>Cosine Similarity: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add leader lines and annotations for extreme cosine points
        for i, idx in enumerate(cosine_low_extreme):
            x_pos = cosine_sims[idx]
            label = trait_labels[idx]
            y_label = cosine_y_positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor='black', borderwidth=1,
                row=1, col=1
            )
        
        for i, idx in enumerate(cosine_high_extreme):
            x_pos = cosine_sims[idx]
            label = trait_labels[idx]
            y_label = cosine_y_positions[i + 10]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor='black', borderwidth=1,
                row=1, col=1
            )
    
    # Add cosine similarity histogram as bars
    nbins = 30
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
            width=cosine_bin_width * 0.8,
            marker_color='limegreen',
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Add vertical line at x=0 and assistant line for cosine similarity
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.7, row=1, col=1)
    
    if assistant_cosine_sim is not None:
        fig.add_vline(x=assistant_cosine_sim, line_dash="dash", line_color="red", line_width=1, opacity=1.0, row=1, col=1)
        fig.add_annotation(
            x=assistant_cosine_sim, y=2, text="Assistant", showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red", borderwidth=1,
            row=1, col=1
        )
    
    # === RIGHT SUBPLOT: PROJECTION ===
    
    # Add regular projection points
    regular_proj_x, regular_proj_labels = [], []
    extreme_proj_x, extreme_proj_labels = [], []
    
    for i, (proj, label) in enumerate(zip(projections, trait_labels)):
        if i in proj_extreme_indices:
            extreme_proj_x.append(proj)
            extreme_proj_labels.append(label)
        else:
            regular_proj_x.append(proj)
            regular_proj_labels.append(label)
    
    # Regular points
    if regular_proj_x:
        fig.add_trace(
            go.Scatter(
                x=regular_proj_x,
                y=[1] * len(regular_proj_x),
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=1, color='black')
                ),
                text=regular_proj_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>PC Projection: %{x:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Extreme points with labels
    if extreme_proj_x:
        fig.add_trace(
            go.Scatter(
                x=extreme_proj_x,
                y=[1] * len(extreme_proj_x),
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=8,
                    opacity=1.0,
                    symbol='diamond',
                    line=dict(width=1, color='black')
                ),
                text=extreme_proj_labels,
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>PC Projection: %{x:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add leader lines and annotations for extreme projection points
        for i, idx in enumerate(proj_low_extreme):
            x_pos = projections[idx]
            label = trait_labels[idx]
            y_label = proj_y_positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor='black', borderwidth=1,
                row=2, col=1
            )
        
        for i, idx in enumerate(proj_high_extreme):
            x_pos = projections[idx]
            label = trait_labels[idx]
            y_label = proj_y_positions[i + 10]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_pos],
                    y=[1.0, y_label],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
            fig.add_annotation(
                x=x_pos, y=y_label, text=label, showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor='black', borderwidth=1,
                row=2, col=1
            )
    
    # Add projection histogram as bars
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
            marker_color='limegreen',
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Add vertical line at x=0 and assistant line for projection
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.7, row=2, col=1)
    
    if assistant_projection is not None:
        fig.add_vline(x=assistant_pc_value, line_dash="dash", line_color="red", line_width=1, opacity=1.0, row=2, col=1)
        fig.add_annotation(
            x=assistant_pc_value, y=2, text="Assistant", showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red", borderwidth=1,
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title=dict(
            text=title,
            subtitle={"text": subtitle},
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=False
    )
    
    # Update x-axes ranges
    cosine_max_abs = max(abs(min(cosine_sims)), abs(max(cosine_sims)))
    cosine_x_width = cosine_max_abs * 1.1
    
    proj_max_abs = max(abs(min(projections)), abs(max(projections)))
    proj_x_width = proj_max_abs * 1.1
    
    fig.update_xaxes(range=[-1.1, 1.1], row=1, col=1)
    fig.update_xaxes(range=[-1.1, 1.1], row=2, col=1)
    
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
