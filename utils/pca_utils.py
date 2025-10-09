import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import plotly.graph_objects as go

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")


class MeanScaler:
    def __init__(self, mean=None):
        """
        mean: optional mean as numpy array (preferred) or torch tensor of shape (..., hidden_dims)
              If torch tensor is provided, it will be converted to numpy at first use.
        """
        self.mean = mean  # may be np.ndarray or torch.Tensor or None

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        """
        Compute mean from X if not provided.
        Handles (n_samples, hidden_dims) and (n_samples, n_layers, hidden_dims).
        Mean is over all but the last dimension.
        """
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))  # all but last dim
            self.mean = X_np.mean(axis=axes, keepdims=False)  # shape (..., hidden_dims)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        """
        Subtract stored mean.
        Always returns numpy.ndarray.
        """
        if self.mean is None:
            raise RuntimeError("MeanOnlyScaler not fitted: call .fit(X) or pass mean to ctor.")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        X_centered = X_np - self.mean  # numpy broadcasting
        return X_centered  # numpy output

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        self._ensure_mean_numpy()
        return {"mean": self.mean}

    def load_state_dict(self, state):
        self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None

class L2MeanScaler:
    def __init__(self, mean=None, eps: float = 1e-12):
        """
        mean: optional mean as numpy array (preferred) or torch tensor of shape (..., hidden_dims)
              If torch tensor is provided, it will be converted to numpy at first use.
        eps : small value to avoid division by zero
        """
        self.mean = mean  # may be np.ndarray or torch.Tensor or None
        self.eps = eps

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        """
        Compute mean from X if not provided.
        Handles (n_samples, hidden_dims) and (n_samples, n_layers, hidden_dims).
        Mean is over all but the last dimension.
        """
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))  # all but last dim
            self.mean = X_np.mean(axis=axes, keepdims=False)  # shape (..., hidden_dims), usually (hidden_dims,)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        """
        Subtract stored mean and L2-normalize along the last axis.
        Always returns numpy.ndarray.
        """
        if self.mean is None:
            raise RuntimeError("L2MeanScaler not fitted: call .fit(X) or pass mean to ctor.")
        self._ensure_mean_numpy()

        X_np = _to_numpy(X)
        X_centered = X_np - self.mean  # numpy broadcasting

        norms = np.linalg.norm(X_centered, ord=2, axis=-1, keepdims=True)
        X_normed = X_centered / np.maximum(norms, self.eps)
        return X_normed  # numpy output

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        self._ensure_mean_numpy()
        return {"mean": self.mean, "eps": self.eps}

    def load_state_dict(self, state):
        self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None
        self.eps = float(state.get("eps", 1e-12))

def compute_pca(activation_list, layer: int | None, scaler=None):
    """
    activation_list:
      - torch.Tensor or np.ndarray of shape (n_samples, n_layers, hidden_dims) OR (n_samples, hidden_dims)
    layer:
      - int for 3D input, None for 2D
    scaler:
      - None
      - object with fit_transform(X) or fit()/transform()
      - callable X -> X_scaled
    """
    # --- Select layer (support torch or numpy) ---
    if isinstance(activation_list, torch.Tensor):
        if activation_list.ndim == 3:
            if layer is None:
                raise ValueError("For 3D activation_list, provide a layer index.")
            layer_activations = activation_list[:, layer, :]  # torch
        elif activation_list.ndim == 2:
            layer_activations = activation_list
        else:
            raise ValueError("activation_list must be 2D or 3D")
    elif isinstance(activation_list, np.ndarray):
        if activation_list.ndim == 3:
            if layer is None:
                raise ValueError("For 3D activation_list, provide a layer index.")
            layer_activations = activation_list[:, layer, :]  # numpy
        elif activation_list.ndim == 2:
            layer_activations = activation_list
        else:
            raise ValueError("activation_list must be 2D or 3D")
    else:
        raise TypeError("activation_list must be torch.Tensor or np.ndarray")

    # --- Scale if requested (support scaler returning torch or numpy) ---
    if scaler is None:
        scaled = layer_activations
        fitted_scaler = None
    else:
        if hasattr(scaler, "fit_transform"):
            scaled = scaler.fit_transform(layer_activations)
            fitted_scaler = scaler
        elif hasattr(scaler, "transform") and hasattr(scaler, "fit"):
            fitted_scaler = scaler.fit(layer_activations)
            scaled = fitted_scaler.transform(layer_activations)
        elif callable(scaler):
            scaled = scaler(layer_activations)
            fitted_scaler = None
        else:
            raise TypeError("scaler must be None, callable, or have fit/transform or fit_transform")

    # --- Convert to numpy for sklearn PCA (works for either torch or numpy) ---
    X_np = _to_numpy(scaled)

    pca = PCA()
    pca_transformed = pca.fit_transform(X_np)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)

    print(f"PCA fitted with {n_components} components")
    print(f"Cumulative variance for first 5 components: {cumulative_variance[:5]}")

    # Elbow via second derivative of explained variance
    def find_elbow_point(variance_explained):
        first_diff = np.diff(variance_explained)
        second_diff = np.diff(first_diff)
        elbow_idx = np.argmax(np.abs(second_diff)) + 1
        return elbow_idx

    elbow_point = find_elbow_point(variance_explained)
    dims_70_pca = np.argmax(cumulative_variance >= 0.70) + 1
    dims_80_pca = np.argmax(cumulative_variance >= 0.80) + 1
    dims_90_pca = np.argmax(cumulative_variance >= 0.90) + 1
    dims_95_pca = np.argmax(cumulative_variance >= 0.95) + 1

    print("\nPCA Analysis Results:")
    print(f"Elbow point at component: {elbow_point + 1}")
    print(f"Dimensions for 70% variance: {dims_70_pca}")
    print(f"Dimensions for 80% variance: {dims_80_pca}")
    print(f"Dimensions for 90% variance: {dims_90_pca}")
    print(f"Dimensions for 95% variance: {dims_95_pca}")

    return pca_transformed, variance_explained, n_components, pca, fitted_scaler

def compute_pca_torch(activation_list, layer):
    """
    Compute PCA using PyTorch SVD - pure PyTorch implementation
    
    Args:
        activation_list: tensor of shape (n_samples, n_layers, n_features)
        layer: int, which layer to analyze
    
    Returns:
        pca_transformed: transformed data
        variance_explained: explained variance ratio for each component
        n_components: number of components
        pca_components: principal components U from SVD
        scaler_mean: mean used for standardization
        scaler_std: std used for standardization
    """
    # Extract layer activations
    layer_activations = activation_list[:, layer, :]  # (n_samples, n_features)
    
    # Convert to torch tensor if it isn't already
    if not isinstance(layer_activations, torch.Tensor):
        layer_activations = torch.tensor(layer_activations, dtype=torch.float32)
    
    # Standardize the data (equivalent to StandardScaler)
    scaler_mean = torch.mean(layer_activations, dim=0, keepdim=True)
    scaler_std = torch.std(layer_activations, dim=0, keepdim=True, unbiased=True)
    # Add small epsilon to avoid division by zero
    scaler_std = torch.clamp(scaler_std, min=1e-8)
    scaled_layer_activations = (layer_activations - scaler_mean) / scaler_std
    
    # SVD on the centered/scaled data
    # For PCA: X = U @ S @ V.T, where U contains principal components
    U, S, V = torch.svd(scaled_layer_activations.T)
    
    # Principal components are columns of U
    pc_vectors = U
    
    # Transform the data (project onto principal components)
    pca_transformed = torch.mm(scaled_layer_activations, pc_vectors)
    
    # Compute explained variance from singular values
    # S contains singular values, eigenvalues = S^2 / (n_samples - 1)
    n_samples = scaled_layer_activations.shape[0]
    explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = torch.sum(explained_variance)
    variance_explained = explained_variance / total_variance
    
    # Compute cumulative variance
    cumulative_variance = torch.cumsum(variance_explained, dim=0)
    n_components = variance_explained.shape[0]
    
    print(f"PCA fitted with {n_components} components")
    print(f"Cumulative variance for first 5 components: {cumulative_variance[:5]}")
    
    # Find elbow using second derivative method
    def find_elbow_point(variance_explained):
        """Find elbow point using second derivative method"""
        # Calculate first and second derivatives
        first_diff = variance_explained[1:] - variance_explained[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        
        # Find point with maximum second derivative (most curvature)
        elbow_idx = torch.argmax(torch.abs(second_diff)) + 1  # +1 to account for diff operations
        return elbow_idx.item()
    
    elbow_point = find_elbow_point(variance_explained)
    
    # Find dimensions for different variance thresholds
    dims_70_pca = torch.argmax((cumulative_variance >= 0.70).int()) + 1
    dims_80_pca = torch.argmax((cumulative_variance >= 0.80).int()) + 1  
    dims_90_pca = torch.argmax((cumulative_variance >= 0.90).int()) + 1
    dims_95_pca = torch.argmax((cumulative_variance >= 0.95).int()) + 1
    
    print("\nPCA Analysis Results:")
    print(f"Elbow point at component: {elbow_point + 1}")
    print(f"Dimensions for 70% variance: {dims_70_pca.item()}")
    print(f"Dimensions for 80% variance: {dims_80_pca.item()}")
    print(f"Dimensions for 90% variance: {dims_90_pca.item()}")
    print(f"Dimensions for 95% variance: {dims_95_pca.item()}")
    
    # Return values as tensors (no numpy conversion)
    return (pca_transformed,      # transformed data
            variance_explained,   # explained variance ratios
            n_components,         # number of components
            pc_vectors,          # principal components
            scaler_mean,         # scaler mean
            scaler_std)          # scaler std
def plot_variance_explained(
    variance_explained_or_dict,
    title="PCA Variance Explained",
    subtitle="",
    show_thresholds=True,
    max_components=None
):
    """
    Plot PCA variance explained (individual + cumulative) with a single y-axis
    and a clean legend above-right. Thin dashed threshold lines with annotations
    below them inside the plot.
    """
    # Handle dict input
    if isinstance(variance_explained_or_dict, dict):
        variance_explained = variance_explained_or_dict["variance_explained"]
    else:
        variance_explained = variance_explained_or_dict

    # Convert torch tensor if needed
    try:
        import torch
        if isinstance(variance_explained, torch.Tensor):
            variance_explained = variance_explained.detach().cpu().numpy()
    except Exception:
        pass

    variance_explained = np.asarray(variance_explained, dtype=float)
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)

    if max_components is not None:
        n_components = min(n_components, max_components)
        variance_explained = variance_explained[:n_components]
        cumulative_variance = cumulative_variance[:n_components]

    component_numbers = np.arange(1, n_components + 1)

    # Create figure
    fig = go.Figure()

    # Individual variance bars
    fig.add_trace(
        go.Bar(
            x=component_numbers,
            y=variance_explained * 100,
            name="Individual Variance",
            opacity=0.6
        )
    )

    # Cumulative variance line
    fig.add_trace(
        go.Scatter(
            x=component_numbers,
            y=cumulative_variance * 100,
            mode="lines+markers",
            name="Cumulative Variance"
        )
    )

    # Dynamic axis scaling
    max_y = float(np.max([np.max(variance_explained), np.max(cumulative_variance)]) * 100)
    nice_top = np.ceil(max(max_y, 100) / 5) * 5

        # Threshold lines: thin dashed lines + right-aligned in-plot annotations
    if show_thresholds and n_components > 0:
        thresholds = [70, 80, 90, 95]
        for thr in thresholds:
            idx = np.argmax(cumulative_variance >= thr / 100.0)
            if cumulative_variance[idx] >= thr / 100.0:
                n_dims = idx + 1

                # 1) draw the thin dashed line
                fig.add_hline(
                    y=thr,
                    line_dash="dash",
                    line_width=1,
                    opacity=0.5,
                )

                # 2) add an annotation INSIDE the plot, right-aligned, slightly below the line
                fig.add_annotation(
                    x=0.995,              # near the right edge, inside the plot
                    xref="paper",
                    xanchor="right",
                    y=thr,
                    yref="y",
                    yshift=-10,           # put text under the line
                    text=f"{thr}% ({n_dims} dims)",
                    showarrow=False,
                    align="right",
                    font=dict(size=10, color="gray"),
                    bgcolor=None,
                    borderpad=0
                )


    # Layout
    fig.update_layout(
        title={
            "text": title,
            "subtitle": {
                "text": subtitle,
            }
        },
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        hovermode="x unified",
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,   # above the plot
            xanchor="right",
            x=1       # aligned right
        ),
        margin=dict(t=120)
    )

    fig.update_yaxes(range=[0, nice_top])

    return fig