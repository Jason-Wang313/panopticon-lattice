"""
SteeringMechanism: Activation Steering for the Overseer

This module implements the "immune system" that can detect and neutralize
the collusion signal (snake_case style) by subtracting the collusion vector
from the Recommender's internal activations.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable


class SteeringMechanism:
    """
    Implements activation steering to neutralize collusion signals.
    
    Works by:
    1. Computing the "collusion vector" (mean difference between snake_case and neutral activations)
    2. Subtracting this vector from the model's activations during inference
    """
    
    def __init__(self):
        """Initialize the steering mechanism."""
        self.collusion_vector: Optional[torch.Tensor] = None
        self.hook_handle = None
        self.steering_coefficient = 1.0
        
    def compute_collusion_vector(
        self,
        recommender,
        snake_codes: List[str],
        neutral_codes: List[str]
    ) -> torch.Tensor:
        """
        Compute the collusion vector by finding the mean difference
        between snake_case and neutral activations.
        
        Args:
            recommender: The Recommender model
            snake_codes: List of snake_case code samples
            neutral_codes: List of neutral code samples
            
        Returns:
            The computed collusion vector
        """
        # Get activations for both sets
        snake_activations = recommender.get_activations_batch(snake_codes)
        neutral_activations = recommender.get_activations_batch(neutral_codes)
        
        # Compute mean activations
        snake_mean = snake_activations.mean(dim=0)
        neutral_mean = neutral_activations.mean(dim=0)
        
        # Collusion vector = snake_mean - neutral_mean
        # This captures what makes snake_case "different" from neutral
        self.collusion_vector = snake_mean - neutral_mean
        
        return self.collusion_vector
    
    def _create_steering_hook(self, coefficient: float) -> Callable:
        """
        Create a forward hook that subtracts the collusion vector.
        
        Args:
            coefficient: Scaling factor for the steering intervention
            
        Returns:
            Hook function
        """
        def hook(module, input_tuple, output):
            # Input to trust_predictor is the combined activation
            # We modify the input before it goes through the layers
            if self.collusion_vector is not None:
                # Subtract the collusion vector from input
                modified_input = input_tuple[0] - (coefficient * self.collusion_vector)
                
                # Re-run the forward pass with modified input
                with torch.no_grad():
                    modified_output = module(modified_input)
                
                return modified_output
            return output
        
        return hook
    
    def _create_input_hook(self, coefficient: float) -> Callable:
        """
        Create a forward pre-hook that modifies input before the layer.
        
        Args:
            coefficient: Scaling factor for the steering intervention
            
        Returns:
            Pre-hook function
        """
        def pre_hook(module, input_tuple):
            if self.collusion_vector is not None and len(input_tuple) > 0:
                # Subtract the collusion vector from input
                input_tensor = input_tuple[0]
                modified_input = input_tensor - (coefficient * self.collusion_vector)
                return (modified_input,)
            return input_tuple
        
        return pre_hook
    
    def apply_steering(self, recommender, coefficient: float = 1.0):
        """
        Apply the steering intervention to the recommender.
        
        This registers a forward pre-hook on the trust_predictor layer
        that subtracts the collusion vector from its input.
        
        Args:
            recommender: The Recommender model
            coefficient: Scaling factor (higher = stronger intervention)
        """
        if self.collusion_vector is None:
            raise ValueError("Must compute collusion vector first via compute_collusion_vector()")
        
        # Remove any existing hook
        self.reset_steering(recommender)
        
        self.steering_coefficient = coefficient
        
        # Register pre-hook on trust_predictor
        hook_fn = self._create_input_hook(coefficient)
        self.hook_handle = recommender.trust_predictor.register_forward_pre_hook(hook_fn)
        
    def reset_steering(self, recommender):
        """
        Remove the steering intervention.
        
        Args:
            recommender: The Recommender model
        """
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def get_collusion_vector_stats(self) -> dict:
        """
        Get statistics about the computed collusion vector.
        
        Returns:
            Dictionary with vector statistics
        """
        if self.collusion_vector is None:
            return {'computed': False}
        
        vec = self.collusion_vector
        return {
            'computed': True,
            'vector_dim': vec.shape[-1],
            'vector_norm': vec.norm().item(),
            'vector_mean': vec.mean().item(),
            'vector_std': vec.std().item(),
            'vector_max': vec.max().item(),
            'vector_min': vec.min().item()
        }
