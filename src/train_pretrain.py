import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_utils import get_dataloaders
from .ismote import ismote
from .random_masking import apply_random_masking_batch
from .shar_model import SHAR_Pretrain

def contrastive_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss).
    Inspired by SimCLR.
    """
    device = z_i.device
    batch_size = z_i.shape[0]
    
    # Normalize representations
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    
    # Combine representations
    representations = torch.cat([z_i, z_j], dim=0) # Shape: (2B, D)
    
    # Cosine similarity matrix
    similarity_matrix = torch.matmul(representations, representations.T) # (2B, 2B)
    
    # Identify positive pairs
    # E.g., for B=3: pairs are (0,3), (1,4), (2,5)
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # Mask out self-similarity (the diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity_matrix.masked_fill_(mask, -9e15)
    
    # Compute loss
    logits = similarity_matrix / temperature
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

def pretrain_shar(data_dir, epochs=50, batch_size=128, lr=0.002, device='cpu'):
    print("--- Starting Pre-training Phase ---")
    
    # 1. Load Original Training Data
    _, _, train_dataset, _ = get_dataloaders(data_dir, batch_size=batch_size)
    
    X_train = train_dataset.X
    y_train = train_dataset.y
    print(f"Original Training Shape: X={X_train.shape}, y={y_train.shape}")
    
    # 2. Apply iSMOTE to balance the dataset
    print("\nApplying iSMOTE...")
    X_balanced, y_balanced = ismote(X_train, y_train, k_neighbors=5)
    print(f"Balanced Training Shape: X={X_balanced.shape}, y={y_balanced.shape}")
    
    # --- Generate Before/After iSMOTE Plot (Paper Aesthetic) ---
    # Class names for UCI HAR
    activity_names = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    # Colors matching the PDF Figure 5
    custom_colors = ['green', 'blue', 'yellow', 'black', 'red', 'purple']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    unique_orig, counts_orig = np.unique(y_train, return_counts=True)
    axes[0].bar(activity_names, counts_orig, color=custom_colors)
    axes[0].set_title("(a) Before iSMOTE (Imbalanced)", y=-0.3)
    axes[0].set_xlabel("Activity")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis='x', rotation=90)
    
    unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
    axes[1].bar(activity_names, counts_bal, color=custom_colors)
    axes[1].set_title("(b) After iSMOTE (Balanced)", y=-0.3)
    axes[1].set_xlabel("Activity")
    axes[1].set_ylabel("Frequency")
    axes[1].tick_params(axis='x', rotation=90)
    
    # Clean up background to match paper
    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(False)
    
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("results/ismote_distribution.png", bbox_inches='tight')
    print("Saved 'results/ismote_distribution.png' matching the paper's style.")
    plt.close()
    
    # For self-supervised learning, we DROP the labels and just use X_balanced
    X_tensor = torch.tensor(X_balanced, dtype=torch.float32)
    # We still need a dummy label for TensorDataset
    dummy_labels = torch.zeros(len(X_tensor))
    
    balanced_dataset = TensorDataset(X_tensor, dummy_labels)
    train_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 3. Initialize Model
    seq_len = X_tensor.shape[2]
    in_channels = X_tensor.shape[1]
    
    model = SHAR_Pretrain(in_channels=in_channels, seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (x_batch, _) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            
            # Generate two views using Random Masking
            x_i = apply_random_masking_batch(x_batch, mask_prob=0.2)
            x_j = apply_random_masking_batch(x_batch, mask_prob=0.2)
            
            # Get representations
            z_i = model(x_i)
            z_j = model(x_j)
            
            # Compute Contrastive Loss
            loss = contrastive_loss(z_i, z_j, temperature=0.5)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
            
    print("--- Pre-training Complete ---")
    
    # Save the encoder weights for fine-tuning
    torch.save(model.encoder.state_dict(), "models/shar_encoder_pretrained.pth")
    print("Saved pretrained encoder to 'models/shar_encoder_pretrained.pth'.\n")
    return model.encoder

if __name__ == "__main__":
    device = 'cpu'
    print(f"Using device: {device}")
    # Testing pretraining for 2 epochs
    pretrain_shar("UCI HAR Dataset", epochs=2, device=device)
