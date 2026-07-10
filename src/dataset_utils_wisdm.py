import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import glob

class WISDMDataset(Dataset):
    def __init__(self, data_dir, split='train', window_size=128, step_size=64):
        """
        Args:
            data_dir (string): Directory pointing to wisdm-dataset/raw/phone/accel
            split (string): 'train' or 'test'
            window_size (int): Number of time steps per window
            step_size (int): Number of time steps to move the window (overlap)
        """
        self.split = split
        self.window_size = window_size
        self.step_size = step_size
        
        # All phone accel files
        file_paths = glob.glob(os.path.join(data_dir, "data_*_accel_phone.txt"))
        
        # Split by subject: Assume subject 1600-1639 is Train, 1640-1650 is Test
        train_files = []
        test_files = []
        for fp in file_paths:
            filename = os.path.basename(fp)
            subject_id = int(filename.split('_')[1])
            if subject_id < 1640:
                train_files.append(fp)
            else:
                test_files.append(fp)
                
        target_files = train_files if split == 'train' else test_files
        print(f"[{split.upper()}] Loading {len(target_files)} subject files...")
        
        all_X = []
        all_y = []
        
        # Load and window data
        for fp in target_files:
            # The WISDM format is: subject_id, activity, timestamp, x, y, z;
            # We strip the semicolon from the z column using a custom converter or string manipulation
            with open(fp, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                parts = line.strip().rstrip(';').split(',')
                if len(parts) == 6:
                    data.append(parts)
            
            df = pd.DataFrame(data, columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
            
            # Convert sensors to numeric
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['z'] = pd.to_numeric(df['z'], errors='coerce')
            df = df.dropna()
            
            # Windowing per activity
            for activity, group in df.groupby('activity'):
                values = group[['x', 'y', 'z']].values
                # Slide window
                for start_idx in range(0, len(values) - window_size, step_size):
                    window = values[start_idx:start_idx + window_size]
                    # Transpose to shape (Channels, SequenceLength) -> (3, 128)
                    all_X.append(window.T)
                    all_y.append(activity)
        
        self.X = np.array(all_X)
        self.y_raw = np.array(all_y)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        # To ensure consistent encoding between train and test, we should technically fit on all possible.
        # But for simplicity in this script, we'll fit on train. 
        # Actually it's safer to pre-define the known WISDM activities:
        known_activities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
        self.label_encoder.fit(known_activities)
        self.y = self.label_encoder.transform(self.y_raw)
        
        print(f"Loaded {split} data: X shape = {self.X.shape}, y shape = {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_seq = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return x_seq, label

def get_dataloaders(data_dir, batch_size=128):
    train_dataset = WISDMDataset(data_dir, split='train')
    test_dataset = WISDMDataset(data_dir, split='test')
    
    unique, counts = np.unique(train_dataset.y, return_counts=True)
    print("Train Dataset label distribution:", dict(zip(unique, counts)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == '__main__':
    # Path pointing to the raw phone accel data (relative to project root)
    data_dir = os.path.join("wisdm-dataset", "raw", "phone", "accel")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(data_dir)
    for X, y in train_loader:
        print(f"Batch X shape: {X.shape}")
        print(f"Batch y shape: {y.shape}")
        break
