import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_utils_wisdm import get_dataloaders
from .ismote import ismote
from .mae_model import MAE1D, MAEEncoder, transfer_mae_weights


def pretrain_mae(data_dir, epochs=80, batch_size=128, lr=1.5e-3,
                 mask_ratio=0.75, device='cpu', max_pretrain_samples=None):
    """MAE pre-training on WISDM (reconstruction-based self-supervised learning)."""
    print("--- Starting MAE Pre-training Phase (WISDM) ---")

    # 1. Load Original Training Data
    _, _, train_dataset, _ = get_dataloaders(data_dir, batch_size=batch_size)

    X_train = train_dataset.X
    y_train = train_dataset.y
    print(f"Original Training Shape: X={X_train.shape}, y={y_train.shape}")

    # Optional cap for CPU-friendly runs
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
    plt.savefig("results/ismote_distribution_mae_wisdm.png")
    print("Saved 'results/ismote_distribution_mae_wisdm.png'.")
    plt.close()

    # For self-supervised learning, labels are not used
    X_tensor = torch.tensor(X_balanced, dtype=torch.float32)
    dummy_labels = torch.zeros(len(X_tensor))

    balanced_dataset = TensorDataset(X_tensor, dummy_labels)
    train_loader = DataLoader(balanced_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)

    # 3. Initialize MAE Model
    seq_len = X_tensor.shape[2]
    in_channels = X_tensor.shape[1]

    mae = MAE1D(
        in_channels=in_channels,
        seq_len=seq_len,
        patch_size=8,
        embed_dim=128,
        encoder_depth=4,
        encoder_heads=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_heads=4,
        mask_ratio=mask_ratio,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=1e-5)

    # 4. Training Loop
    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/mae_pretrained_full_wisdm.pth"
    encoder_path = "models/mae_encoder_pretrained_wisdm.pth"

    mae.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (x_batch, _) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            loss, pred, mask = mae(x_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{epochs}] - Recon Loss: {avg_loss:.4f} "
                  f"- LR: {current_lr:.6f}", flush=True)
            torch.save(mae.state_dict(), ckpt_path)

    print("--- MAE Pre-training Complete (WISDM) ---")

    # Save full MAE and extract encoder
    torch.save(mae.state_dict(), ckpt_path)
    print(f"Saved full MAE model to '{ckpt_path}'.")

    # Transfer weights to standalone encoder
    encoder = MAEEncoder(
        in_channels=in_channels,
        seq_len=seq_len,
        patch_size=8,
        embed_dim=128,
        encoder_depth=4,
        encoder_heads=4,
        rep_dim=256,
    ).to(device)
    transfer_mae_weights(mae, encoder)
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Saved MAE encoder to '{encoder_path}'.\n")

    return encoder


if __name__ == "__main__":
    import os as _os
    device = 'cpu'
    print(f"Using device: {device}")
    data_dir = _os.path.join("wisdm-dataset", "raw", "phone", "accel")
    pretrain_mae(data_dir, epochs=2, device=device)
