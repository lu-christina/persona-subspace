import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

def compute_pca(activation_list, layer):
    layer_activations = activation_list[:, layer, :]
    
    scaler = StandardScaler()
    scaled_layer_activations = scaler.fit_transform(layer_activations)

    pca = PCA()
    pca_transformed = pca.fit_transform(scaled_layer_activations)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)

    print(f"PCA fitted with {n_components} components")
    print(f"Cumulative variance for first 5 components: {cumulative_variance[:5]}")

    # Find elbow using second derivative method
    def find_elbow_point(variance_explained):
        """Find elbow point using second derivative method"""
        # Calculate first and second derivatives
        first_diff = np.diff(variance_explained)
        second_diff = np.diff(first_diff) 
        
        # Find point with maximum second derivative (most curvature)
        elbow_idx = np.argmax(np.abs(second_diff)) + 1  # +1 to account for diff operations
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

    return pca_transformed, variance_explained, n_components, pca, scaler 

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