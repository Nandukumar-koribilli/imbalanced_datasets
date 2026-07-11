import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_utils_wisdm import get_dataloaders
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

def pretrain_shar(data_dir, epochs=50, batch_size=128, lr=0.002, device='cpu',
                  max_pretrain_samples=None):
    print("--- Starting Pre-training Phase ---")

    # 1. Load Original Training Data
    _, _, train_dataset, _ = get_dataloaders(data_dir, batch_size=batch_size)

    X_train = train_dataset.X
    y_train = train_dataset.y
    print(f"Original Training Shape: X={X_train.shape}, y={y_train.shape}")

    # Optional cap for CPU-friendly runs: random subsample BEFORE balancing
    # (WISDM has ~45k windows — full-set pretraining takes hours on CPU)
    if max_pretrain_samples is not None and len(y_train) > max_pretrain_samples:
        keep = np.random.choice(len(y_train), max_pretrain_samples, replace=False)
        X_train, y_train = X_train[keep], y_train[keep]
        print(f"Subsampled pretraining set to {len(y_train)} windows "
              f"(--max_pretrain_samples).")
    
    # 2. Apply iSMOTE to balance the dataset
    print("\nApplying iSMOTE...")
    X_balanced, y_balanced = ismote(X_train, y_train, k_neighbors=5)
    print(f"Balanced Training Shape: X={X_balanced.shape}, y={y_balanced.shape}")
    
    # --- Generate Before/After iSMOTE Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    unique_orig, counts_orig = np.unique(y_train, return_counts=True)
    sns.barplot(x=unique_orig, y=counts_orig, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Before iSMOTE (Imbalanced)")
    axes[0].set_xlabel("Activity Class")
    axes[0].set_ylabel("Number of Samples")
    
    unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
    sns.barplot(x=unique_bal, y=counts_bal, ax=axes[1], palette="Greens_d")
    axes[1].set_title("After iSMOTE (Balanced)")
    axes[1].set_xlabel("Activity Class")
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/ismote_distribution_wisdm.png")
    print("Saved 'results/ismote_distribution_wisdm.png' showing class balances before and after.")
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
    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/shar_encoder_pretrained_wisdm.pth"
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
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}", flush=True)
            # Periodic checkpoint so an interrupted run still leaves a usable encoder
            torch.save(model.encoder.state_dict(), ckpt_path)

    print("--- Pre-training Complete ---")

    # Save the final encoder weights for fine-tuning
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"Saved pretrained encoder to '{ckpt_path}'.\n")
    return model.encoder

if __name__ == "__main__":
    device = 'cpu'
    print(f"Using device: {device}")
    # Testing pretraining for 2 epochs
    pretrain_shar("UCI HAR Dataset", epochs=2, device=device)
