import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class UCIHARDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir (string): Directory with all the UCI HAR data.
            split (string): 'train' or 'test'
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        
        # Load inertial signals
        signal_dir = os.path.join(self.data_dir, 'Inertial Signals')
        
        # File prefixes
        signals = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]
        
        loaded_signals = []
        for signal in signals:
            filename = f"{signal}_{self.split}.txt"
            filepath = os.path.join(signal_dir, filename)
            # Read space separated data
            data = pd.read_csv(filepath, sep=r'\s+', header=None).values
            loaded_signals.append(data)
            
        # Shape will be (N, 9, 128) where 9 is num channels, 128 is window length
        self.X = np.stack(loaded_signals, axis=1) 
        
        # Load labels
        y_path = os.path.join(self.data_dir, f"y_{self.split}.txt")
        self.y = pd.read_csv(y_path, sep=r'\s+', header=None).values.flatten()
        # UCI HAR labels are 1-6, convert to 0-5
        self.y = self.y - 1 
        
        print(f"Loaded {split} data: X shape = {self.X.shape}, y shape = {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Convert to torch tensors
        x_seq = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return x_seq, label

def get_dataloaders(data_dir, batch_size=128):
    train_dataset = UCIHARDataset(data_dir, split='train')
    test_dataset = UCIHARDataset(data_dir, split='test')
    
    # We will compute the label distribution for imbalance insight
    unique, counts = np.unique(train_dataset.y, return_counts=True)
    print("Train Dataset label distribution:", dict(zip(unique, counts)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == '__main__':
    data_dir = "UCI HAR Dataset"
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(data_dir)
    for X, y in train_loader:
        print(f"Batch X shape: {X.shape}")
        print(f"Batch y shape: {y.shape}")
        break
