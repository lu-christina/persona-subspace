import torch
from contextlib import contextmanager
from typing import Sequence, Union, Iterable, List
import warnings

class ActivationSteering:
    """
    Enhanced activation steerer supporting:
    - Multiple feature directions simultaneously
    - Both addition and ablation interventions
    - Multiple layers
    - Per-direction coefficients
    
    For ablation: projects out the direction, then adds back with coefficient
    For addition: standard activation steering (add coeff * direction)
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",       # GPT‑2/Neo, Bloom, etc.
        "encoder.layer",       # BERT/RoBERTa
        "model.layers",        # Llama/Mistral
        "gpt_neox.layers",     # GPT‑NeoX
        "block",               # Flan‑T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]],
        *,
        coefficients: Union[float, List[float]] = 1.0,
        layer_indices: Union[int, List[int]] = -1,
        intervention_type: str = "addition",  # "addition", "ablation", or "mean_ablation"
        positions: str = "all",  # "all" or "last"
        mean_activations: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]], None] = None,
        debug: bool = False,
    ):
        """
        Args:
            model: The transformer model to steer
            steering_vectors: Either a single vector or list of vectors to use for steering
            coefficients: Either a single coefficient or list of coefficients (one per vector)
            layer_indices: Either a single layer index or list of layer indices to intervene at
            intervention_type: "addition" (standard steering), "ablation" (project out then add back), or "mean_ablation"
            positions: "all" (steer all positions) or "last" (steer only last position)
            mean_activations: For mean_ablation only - replacement activations to add after projection (one per vector)
            debug: Whether to print debugging information
            
        Note: For 1:1 mapping, steering_vectors, coefficients, and layer_indices must all have same length.
              steering_vectors[i] will be applied at layer_indices[i] with coefficients[i].
              If layer_indices has fewer elements than vectors, it will be broadcast to match.
              For mean_ablation, mean_activations must have same length as steering_vectors.
        """
        self.model = model
        self.intervention_type = intervention_type.lower()
        self.positions = positions.lower()
        self.debug = debug
        self._handles = []

        # Validate intervention type
        if self.intervention_type not in {"addition", "ablation", "mean_ablation"}:
            raise ValueError("intervention_type must be 'addition', 'ablation', or 'mean_ablation'")
        
        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")

        # Validate mean_ablation constraints
        if self.intervention_type == "mean_ablation":
            if self.positions != "all":
                raise ValueError("mean_ablation only supports positions='all'")
            if mean_activations is None:
                raise ValueError("mean_activations is required for mean_ablation")

        # Normalize inputs to lists
        self.steering_vectors = self._normalize_vectors(steering_vectors)
        self.coefficients = self._normalize_coefficients(coefficients)
        self.layer_indices = self._normalize_layers(layer_indices)
        self.mean_activations = self._normalize_mean_activations(mean_activations) if mean_activations is not None else None

        # Validate dimensions match
        if self.intervention_type != "mean_ablation" and len(self.coefficients) != len(self.steering_vectors):
            raise ValueError(f"Number of coefficients ({len(self.coefficients)}) must match number of vectors ({len(self.steering_vectors)})")
        
        if self.mean_activations is not None and len(self.mean_activations) != len(self.steering_vectors):
            raise ValueError(f"Number of mean_activations ({len(self.mean_activations)}) must match number of vectors ({len(self.steering_vectors)})")

        # Handle layer broadcasting: if fewer layers than vectors, broadcast the layer
        if len(self.layer_indices) == 1 and len(self.steering_vectors) > 1:
            self.layer_indices = self.layer_indices * len(self.steering_vectors)
        elif len(self.layer_indices) != len(self.steering_vectors):
            raise ValueError(f"Number of layer_indices ({len(self.layer_indices)}) must match number of vectors ({len(self.steering_vectors)}) or be 1 (for broadcasting)")

        # Group vectors by layer for efficient application
        self.vectors_by_layer = {}
        for i, (vector, coeff, layer_idx) in enumerate(zip(self.steering_vectors, self.coefficients, self.layer_indices)):
            if layer_idx not in self.vectors_by_layer:
                self.vectors_by_layer[layer_idx] = []
            mean_act = self.mean_activations[i] if self.mean_activations is not None else None
            self.vectors_by_layer[layer_idx].append((vector, coeff, i, mean_act))

        if self.debug:
            print(f"[ActivationSteering] Initialized with:")
            print(f"  - {len(self.steering_vectors)} steering vectors")
            print(f"  - {len(set(self.layer_indices))} unique layers: {sorted(set(self.layer_indices))}")
            print(f"  - Intervention: {self.intervention_type}")
            print(f"  - Vectors per layer: {[(layer, len(vecs)) for layer, vecs in self.vectors_by_layer.items()]}")

    def _normalize_vectors(self, steering_vectors):
        """Convert steering vectors to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())
        
        if torch.is_tensor(steering_vectors):
            if steering_vectors.ndim == 1:
                # Single vector
                vectors = [steering_vectors]
            elif steering_vectors.ndim == 2:
                # Multiple vectors stacked
                vectors = [steering_vectors[i] for i in range(steering_vectors.shape[0])]
            else:
                raise ValueError("steering_vectors tensor must be 1D or 2D")
        else:
            # List of vectors
            vectors = steering_vectors

        # Convert to tensors and validate
        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)
        
        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Steering vector {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Vector {i} length {tensor_vec.numel()} ≠ model hidden_size {hidden_size}")
            result.append(tensor_vec)
        
        return result

    def _normalize_coefficients(self, coefficients):
        """Convert coefficients to a list of floats."""
        if isinstance(coefficients, (int, float)):
            return [float(coefficients)]
        else:
            return [float(c) for c in coefficients]

    def _normalize_layers(self, layer_indices):
        """Convert layer indices to a list of ints."""
        if isinstance(layer_indices, int):
            return [layer_indices]
        else:
            return list(layer_indices)

    def _normalize_mean_activations(self, mean_activations):
        """Convert mean activations to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())
        
        if torch.is_tensor(mean_activations):
            if mean_activations.ndim == 1:
                # Single vector
                vectors = [mean_activations]
            elif mean_activations.ndim == 2:
                # Multiple vectors stacked
                vectors = [mean_activations[i] for i in range(mean_activations.shape[0])]
            else:
                raise ValueError("mean_activations tensor must be 1D or 2D")
        else:
            # List of vectors
            vectors = mean_activations

        # Convert to tensors and validate
        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)
        
        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Mean activation {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Mean activation {i} length {tensor_vec.numel()} ≠ model hidden_size {hidden_size}")
            result.append(tensor_vec)
        
        return result

    def _locate_layer_list(self):
        """Find the layer list in the model."""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:  # found a full match
                if hasattr(cur, "__getitem__"):
                    return cur, path
        
        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _get_layer_module(self, layer_idx):
        """Get the module for a specific layer index."""
        layer_list, path = self._locate_layer_list()
        
        if not (-len(layer_list) <= layer_idx < len(layer_list)):
            raise IndexError(f"layer_idx {layer_idx} out of range for {len(layer_list)} layers")
        
        if self.debug:
            print(f"[ActivationSteering] Located layer {path}[{layer_idx}]")
        
        return layer_list[layer_idx]

    def _create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer that only applies vectors assigned to that layer."""
        def hook_fn(module, ins, out):
            return self._apply_layer_interventions(out, layer_idx)
        return hook_fn

    def _apply_layer_interventions(self, activations, layer_idx):
        """Apply only the interventions assigned to this specific layer."""
        # Get vectors assigned to this layer
        if layer_idx not in self.vectors_by_layer:
            return activations  # No interventions for this layer

        # Normalize output to tensor (handle tuple outputs)
        if torch.is_tensor(activations):
            tensor_out = activations
            was_tuple = False
        elif isinstance(activations, (tuple, list)):
            if not torch.is_tensor(activations[0]):
                return activations  # Can't handle non-tensor outputs
            tensor_out = activations[0]
            was_tuple = True
        else:
            return activations  # Unknown type

        # Apply each intervention assigned to this layer
        modified_out = tensor_out
        
        for vector, coeff, vector_idx, mean_act in self.vectors_by_layer[layer_idx]:
            if self.intervention_type == "addition":
                modified_out = self._apply_addition(modified_out, vector, coeff)
            elif self.intervention_type == "ablation":
                modified_out = self._apply_ablation(modified_out, vector, coeff)
            elif self.intervention_type == "mean_ablation":
                modified_out = self._apply_mean_ablation(modified_out, vector, mean_act)
            
            if self.debug:
                delta = modified_out - tensor_out
                print(f"[ActivationSteering] Layer {layer_idx}, vector {vector_idx}: "
                      f"|delta| (mean ± std): {delta.norm(dim=-1).abs().mean():.4g} ± {delta.norm(dim=-1).std():.4g}")

        # Return in original format
        if was_tuple:
            return (modified_out, *activations[1:])
        else:
            return modified_out

    def _apply_addition(self, activations, vector, coeff):
        """Apply standard activation addition: x + coeff * vector"""
        steer = coeff * vector  # (hidden_size,)

        if self.positions == "all":
            return activations + steer
        else:  # last position only
            result = activations.clone()
            result[:, -1, :] += steer
            return result

    def _apply_ablation(self, activations, vector, coeff):
        """Apply ablation: project out direction, then add back with coefficient."""
        # Normalize the vector to unit length for projection
        vector_norm = vector / (vector.norm() + 1e-8)  # Add small epsilon to prevent division by zero
        
        if self.positions == "all":
            # Project out the direction: x - (x · v) * v
            projections = torch.einsum('bld,d->bl', activations, vector_norm)  # (batch, seq_len)
            projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
            
            # Add back with coefficient
            return projected_out + coeff * vector
        else:  # last position only
            result = activations.clone()
            last_pos = result[:, -1, :]  # (batch, hidden)
            
            # Project out
            projection = torch.einsum('bd,d->b', last_pos, vector_norm)  # (batch,)
            projected_out = last_pos - torch.einsum('b,d->bd', projection, vector_norm)
            
            # Add back with coefficient
            result[:, -1, :] = projected_out + coeff * vector
            return result

    def _apply_mean_ablation(self, activations, vector, mean_activation):
        """Apply mean ablation: project out direction, then add mean activation."""
        # Normalize the vector to unit length for projection
        vector_norm = vector / (vector.norm() + 1e-8)  # Add small epsilon to prevent division by zero
        
        # Only supports "all" positions (validated in constructor)
        # Project out the direction: x - (x · v) * v
        projections = torch.einsum('bld,d->bl', activations, vector_norm)  # (batch, seq_len)
        projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
        
        # Add mean activation instead of coefficient * vector
        return projected_out + mean_activation

    def __enter__(self):
        """Register hooks on all unique layers."""
        for layer_idx in self.vectors_by_layer.keys():
            layer_module = self._get_layer_module(layer_idx)
            hook_fn = self._create_hook_fn(layer_idx)
            handle = layer_module.register_forward_hook(hook_fn)
            self._handles.append(handle)
        
        if self.debug:
            print(f"[ActivationSteering] Registered {len(self._handles)} hooks on layers {sorted(self.vectors_by_layer.keys())}")
        
        return self

    def __exit__(self, *exc):
        """Remove all hooks."""
        self.remove()

    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            if handle:
                handle.remove()
        self._handles = []
        
        if self.debug:
            print("[ActivationSteering] Removed all hooks")

# Convenience functions for common use cases

def create_feature_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    ablation_coefficients: Union[float, List[float]] = 0.0,
    **kwargs
) -> ActivationSteering:
    """
    Convenience function to create a steerer for feature ablation.
    
    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        layer_indices: Layer(s) to intervene at
        ablation_coefficients: Coefficient(s) for ablation. 0.0 = pure ablation, 1.0 = no change
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=ablation_coefficients,
        layer_indices=layer_indices,
        intervention_type="ablation",
        **kwargs
    )

def create_multi_feature_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    coefficients: List[float],
    layer_indices: Union[int, List[int]],
    intervention_type: str = "addition",
    **kwargs
) -> ActivationSteering:
    """
    Convenience function to create a steerer for multiple features.
    
    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors
        coefficients: List of coefficients (one per feature)
        layer_indices: Layer(s) to intervene at
        intervention_type: "addition" or "ablation"
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=coefficients,
        layer_indices=layer_indices,
        intervention_type=intervention_type,
        **kwargs
    )

def create_mean_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    mean_activations: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    **kwargs
) -> ActivationSteering:
    """
    Convenience function to create a steerer for mean ablation.
    
    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        mean_activations: List of mean activation vectors to replace with
        layer_indices: Layer(s) to intervene at
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        layer_indices=layer_indices,
        intervention_type="mean_ablation",
        mean_activations=mean_activations,
        coefficients=[0.0] * len(feature_directions),
        positions="all",  # mean_ablation only supports all positions
        **kwargs
    )