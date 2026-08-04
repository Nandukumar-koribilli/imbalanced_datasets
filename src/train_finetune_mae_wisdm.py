import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

WISDM_CLASS_NAMES = [
    "Walking", "Jogging", "Stairs", "Sitting", "Standing", "Typing",
    "Brushing Teeth", "Eating Soup", "Eating Chips", "Eating Pasta",
    "Drinking", "Eating Sandwich", "Kicking", "Playing Catch", "Dribbling",
    "Writing", "Clapping", "Folding",
]

from .dataset_utils_wisdm import get_dataloaders
from .shar_model import SHAR_Classifier
from .mae_model import MAEEncoder


def finetune_mae(data_dir, encoder_path="models/mae_encoder_pretrained_wisdm.pth",
                 label_ratio=0.25, epochs=60, batch_size=128, lr=5e-4,
                 device='cpu'):
    """
    Fine-tunes the MAE-pretrained encoder on a small percentage of labeled
    WISDM data. Uses the same SHAR_Classifier head.
    """
    print(f"--- Starting MAE Fine-tuning Phase — WISDM (Label Ratio: {label_ratio*100}%) ---")

    # 1. Load Data
    train_loader_full, test_loader, train_dataset, test_dataset = get_dataloaders(
        data_dir, batch_size=batch_size
    )

    X_train = train_dataset.X
    y_train = train_dataset.y
    print(f"Full Training Shape: X={X_train.shape}, y={y_train.shape}")

    # 2. Sample a subset of labeled data (stratified)
    num_samples = int(len(y_train) * label_ratio)
    from sklearn.model_selection import train_test_split
    X_subset, _, y_subset, _ = train_test_split(
        X_train, y_train,
        train_size=num_samples,
        stratify=y_train,
        random_state=42,
    )
    print(f"Subset Labeled Training Shape ({label_ratio*100}%): "
          f"X={X_subset.shape}, y={y_subset.shape}")

    # Convert to loaders
    X_tensor = torch.tensor(X_subset, dtype=torch.float32)
    y_tensor = torch.tensor(y_subset, dtype=torch.long)
    subset_dataset = TensorDataset(X_tensor, y_tensor)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=True)

    # 3. Load MAE Pretrained Encoder and build classifier
    seq_len = X_tensor.shape[2]
    in_channels = X_tensor.shape[1]
    num_classes = 18  # 18 activities in WISDM

    encoder = MAEEncoder(
        in_channels=in_channels,
        seq_len=seq_len,
        patch_size=8,
        embed_dim=128,
        encoder_depth=4,
        encoder_heads=4,
        rep_dim=256,
    ).to(device)

    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"Successfully loaded MAE encoder weights from {encoder_path}")
    except Exception as e:
        print(f"Could not load weights, starting from scratch. Error: {e}")

    model = SHAR_Classifier(encoder, num_classes=num_classes).to(device)

    # 4. Optimizer, loss, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=1e-6)

    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in subset_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        scheduler.step()
        avg_loss = total_loss / len(subset_loader)
        acc = 100.0 * correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")

    print("--- MAE Fine-tuning Complete (WISDM) ---")

    # 6. Evaluation on Test Set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    test_acc = 100.0 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Macro F1-Score: {f1:.4f}")

    # Persist metrics
    present = sorted(set(all_targets) | set(all_preds))
    names = [WISDM_CLASS_NAMES[i] if i < len(WISDM_CLASS_NAMES) else f"Class {i}"
             for i in present]
    report = classification_report(all_targets, all_preds, labels=present,
                                   target_names=names, output_dict=True)
    metrics = {
        "dataset": "WISDM",
        "method": "MAE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_ratio": label_ratio,
        "epochs": epochs,
        "test_accuracy": round(test_acc, 2),
        "macro_f1": round(float(f1), 4),
        "per_class": {
            name: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1": round(v["f1-score"], 4),
                "support": int(v["support"]),
            }
            for name, v in report.items()
            if isinstance(v, dict) and "f1-score" in v
            and name not in ("macro avg", "weighted avg")
        },
    }
    os.makedirs("results", exist_ok=True)
    with open("results/metrics_mae_wisdm.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to 'results/metrics_mae_wisdm.json'.")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title("MAE — Confusion Matrix on Test Set (WISDM)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_mae_wisdm.png")
    print("Saved 'results/confusion_matrix_mae_wisdm.png'.")
    plt.close()

    # Save fine-tuned classifier
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mae_classifier_finetuned_wisdm.pth")
    print("Saved fine-tuned MAE classifier to 'models/mae_classifier_finetuned_wisdm.pth'.")

    return model


if __name__ == "__main__":
    import os as _os
    device = 'cpu'
    print(f"Using device: {device}")
    data_dir = _os.path.join("wisdm-dataset", "raw", "phone", "accel")
    finetune_mae(data_dir, label_ratio=0.25, epochs=2, device=device)
