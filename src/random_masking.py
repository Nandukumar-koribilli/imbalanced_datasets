import torch
import numpy as np

class RandomMasking(object):
    """
    Random Masking technique as described in the paper:
    Self-Supervised Learning for Activity Recognition Based on Datasets With Imbalanced Classes
    
    This acts as a data augmentation/corruption technique to remove identity 
    mappings for the self-supervised pre-training pretext task.
    """
    def __init__(self, mask_prob=0.2):
        """
        Args:
            mask_prob (float): The probability of masking a specific timestep 
                               across all channels.
        """
        self.mask_prob = mask_prob

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Sequence_Length, Channels) or (Channels, Sequence_Length).
                              Assuming shape is (Channels, Sequence_Length) based on 1D Conv inputs.
        Returns:
            torch.Tensor: Masked tensor.
        """
        # x is assumed to be (C, L)
        device = x.device
        C, L = x.shape
        
        # Create a mask of shape (L,)
        # Mask whole timesteps across all channels to force network to learn temporal dependencies
        mask = torch.rand(L, device=device) < self.mask_prob
        
        # Apply mask (set to 0)
        x_masked = x.clone()
        x_masked[:, mask] = 0.0
        
        return x_masked

def apply_random_masking_batch(x_batch, mask_prob=0.2):
    """
    Applies random masking to a batch of sequences.
    Args:
        x_batch (torch.Tensor): Input batch of shape (Batch, Channels, Sequence_Length)
    """
    device = x_batch.device
    B, C, L = x_batch.shape
    
    # Create mask of shape (Batch, 1, Sequence_Length) to broadcast across channels
    mask = torch.rand((B, 1, L), device=device) < mask_prob
    
    x_masked = x_batch.clone()
    x_masked[mask.expand(-1, C, -1)] = 0.0
    
    return x_masked

if __name__ == "__main__":
    # Test
    dummy_input = torch.ones((32, 9, 128)) # Batch=32, Channels=9, SeqLen=128
    masked = apply_random_masking_batch(dummy_input, mask_prob=0.2)
    
    print(f"Original shape: {dummy_input.shape}")
    print(f"Masked shape: {masked.shape}")
    
    # Check percentage of zeros
    zeros_perc = (masked == 0).float().mean().item()
    print(f"Percentage of masked (zeroed) elements: {zeros_perc:.4f} (Expected: ~0.20)")
