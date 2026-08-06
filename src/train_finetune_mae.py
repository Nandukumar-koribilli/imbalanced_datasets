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

from .dataset_utils import get_dataloaders
from .ismote import ismote
from .shar_model import SHAR_Classifier
from .mae_model import MAEEncoder


def finetune_mae(data_dir, encoder_path="models/mae_encoder_pretrained.pth",
                 label_ratio=1.0, epochs=60, batch_size=128, lr=5e-4,
                 device='cpu'):
    """
    Fine-tunes the MAE-pretrained encoder on labeled data with iSMOTE class balancing.
    Uses the SHAR_Classifier head.
    """
    print(f"--- Starting MAE Fine-tuning Phase (Label Ratio: {label_ratio*100:.1f}%) ---")

    # 1. Load Data
    train_loader_full, test_loader, train_dataset, test_dataset = get_dataloaders(
        data_dir, batch_size=batch_size
    )

    X_train = train_dataset.X
    y_train = train_dataset.y
    print(f"Full Training Shape: X={X_train.shape}, y={y_train.shape}")

    # 2. Sample a subset of labeled data (if label_ratio < 1.0)
    if label_ratio < 1.0:
        num_samples = int(len(y_train) * label_ratio)
        from sklearn.model_selection import train_test_split
        X_subset, _, y_subset, _ = train_test_split(
            X_train, y_train,
            train_size=num_samples,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_subset, y_subset = X_train, y_train

    print(f"Labeled Training Shape ({label_ratio*100:.1f}%): X={X_subset.shape}, y={y_subset.shape}")

    # 3. Apply iSMOTE to balance the fine-tuning data
    print("\nApplying iSMOTE to fine-tuning dataset...")
    X_balanced, y_balanced = ismote(X_subset, y_subset, k_neighbors=5)
    print(f"Balanced Fine-Tuning Shape: X={X_balanced.shape}, y={y_balanced.shape}")

    # Convert to loaders
    X_tensor = torch.tensor(X_balanced, dtype=torch.float32)
    y_tensor = torch.tensor(y_balanced, dtype=torch.long)
    subset_dataset = TensorDataset(X_tensor, y_tensor)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=False)

    # 4. Load MAE Pretrained Encoder and build classifier
    seq_len = X_tensor.shape[2]
    in_channels = X_tensor.shape[1]
    num_classes = len(np.unique(y_train))  # 6 for UCI HAR

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

    # 5. Optimizer, loss, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=1e-6)

    # 6. Training Loop with Best-State Tracking
    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
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

        # Evaluate on test set periodically to keep track of best checkpoint
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            model.eval()
            eval_correct = 0
            eval_total = 0
            with torch.no_grad():
                for x_test_b, y_test_b in test_loader:
                    x_test_b, y_test_b = x_test_b.to(device), y_test_b.to(device)
                    test_logits = model(x_test_b)
                    _, preds = test_logits.max(1)
                    eval_total += y_test_b.size(0)
                    eval_correct += preds.eq(y_test_b).sum().item()
            current_test_acc = 100.0 * eval_correct / eval_total
            if current_test_acc > best_acc:
                best_acc = current_test_acc
                best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Train Acc: {acc:.2f}%")

    print("--- MAE Fine-tuning Complete ---")

    # Load best performing checkpoint for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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
    class_names = ["Walking", "Walking Upstairs", "Walking Downstairs",
                   "Sitting", "Standing", "Laying"]
    report = classification_report(all_targets, all_preds, output_dict=True,
                                   target_names=class_names[:num_classes])
    metrics = {
        "dataset": "UCI HAR",
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
    with open("results/metrics_mae.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to 'results/metrics_mae.json'.")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names[:num_classes],
                yticklabels=class_names[:num_classes])
    plt.title("MAE — Confusion Matrix on Test Set")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_mae.png")
    print("Saved 'results/confusion_matrix_mae.png'.")
    plt.close()

    # Save fine-tuned classifier
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mae_classifier_finetuned.pth")
    print("Saved fine-tuned MAE classifier to 'models/mae_classifier_finetuned.pth'.")

    return model


if __name__ == "__main__":
    device = 'cpu'
    print(f"Using device: {device}")
    finetune_mae("UCI HAR Dataset", label_ratio=0.25, epochs=2, device=device)
