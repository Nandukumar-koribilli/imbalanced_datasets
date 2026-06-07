import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_utils import get_dataloaders
from .shar_model import SHAR_Pretrain, SHAR_Classifier

def finetune_shar(data_dir, encoder_path="shar_encoder_pretrained.pth", label_ratio=0.25, epochs=50, batch_size=128, lr=0.001, device='cpu'):
    """
    Fine-tunes the pretrained SHAR encoder on a small percentage of labeled data.
    """
    print(f"--- Starting Fine-tuning Phase (Label Ratio: {label_ratio*100}%) ---")
    
    # 1. Load Data
    train_loader_full, test_loader, train_dataset, test_dataset = get_dataloaders(data_dir, batch_size=batch_size)
    
    X_train = train_dataset.X
    y_train = train_dataset.y
    print(f"Full Training Shape: X={X_train.shape}, y={y_train.shape}")
    
    # 2. Sample a subset of labeled data
    num_samples = int(len(y_train) * label_ratio)
    
    # Stratified sampling to maintain distribution
    from sklearn.model_selection import train_test_split
    X_subset, _, y_subset, _ = train_test_split(
        X_train, y_train, 
        train_size=num_samples, 
        stratify=y_train, 
        random_state=42
    )
    
    print(f"Subset Labeled Training Shape ({label_ratio*100}%): X={X_subset.shape}, y={y_subset.shape}")
    
    # Convert to loaders
    X_tensor = torch.tensor(X_subset, dtype=torch.float32)
    y_tensor = torch.tensor(y_subset, dtype=torch.long)
    subset_dataset = TensorDataset(X_tensor, y_tensor)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 3. Load Pretrained Encoder and Initialize Classifier
    seq_len = X_tensor.shape[2]
    in_channels = X_tensor.shape[1]
    num_classes = len(np.unique(y_test)) if 'y_test' in locals() else 6 # 6 classes for UCI
    
    # Needs the dummy projector to load the encoder properly 
    dummy_pretrain = SHAR_Pretrain(in_channels=in_channels, seq_len=seq_len).to(device)
    
    try:
        dummy_pretrain.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"Successfully loaded pretrained encoder weights from {encoder_path}")
    except Exception as e:
        print(f"Could not load weights, starting training from scratch. Error: {e}")
        
    model = SHAR_Classifier(dummy_pretrain.encoder, num_classes=num_classes).to(device)
    
    # 4. Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(subset_loader):
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
            
        avg_loss = total_loss / len(subset_loader)
        acc = 100. * correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")
            
    print("--- Fine-tuning Complete ---")
    
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
            
    test_acc = 100. * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    
    # F1 Score calculation
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Macro F1-Score: {f1:.4f}")
    
    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    
    # Optional: If you had string labels, you would pass them to xticklabels and yticklabels. 
    # For UCI HAR, classes are 0-5.
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    print("Saved classification heatmap to 'results/confusion_matrix.png'.")
    plt.close()
    
    return model

if __name__ == "__main__":
    device = 'cpu'
    print(f"Using device: {device}")
    # Testing finetuning for 2 epochs on 25% data
    finetune_shar("UCI HAR Dataset", label_ratio=0.25, epochs=2, device=device)
