import numpy as np
from sklearn.neighbors import NearestNeighbors

def ismote(X, y, target_samples_per_class=None, k_neighbors=5):
    """
    iSMOTE: Improved Synthetic Minority Oversampling Technique.
    
    Args:
        X (np.ndarray): Data features of shape (N, Channels, Sequence_Length)
        y (np.ndarray): Labels of shape (N,)
        target_samples_per_class (int or dict): Target number of samples per class. 
                                              If None, balance all classes to the max class size.
        k_neighbors (int): Number of nearest neighbors to use for validation.
        
    Returns:
        X_resampled (np.ndarray): Oversampled features.
        y_resampled (np.ndarray): Oversampled labels.
    """
    # X shape is (N, C, L). We need to flatten to (N, C*L) for NearestNeighbors
    N, C, L = X.shape
    X_flat = X.reshape(N, -1)
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    if target_samples_per_class is None:
        target_samples = np.max(class_counts)
        target_dict = {c: target_samples for c in unique_classes}
    elif isinstance(target_samples_per_class, int):
        target_dict = {c: target_samples_per_class for c in unique_classes}
    else:
        target_dict = target_samples_per_class
        
    new_X = [X]
    new_y = [y]
    
    # Fit KNN on original data
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1) # +1 because it finds itself
    nn.fit(X_flat)
        
    for cls in unique_classes:
        target_N = target_dict.get(cls, 0)
        current_N = class_counts[np.where(unique_classes == cls)[0][0]]
        
        samples_to_generate = target_N - current_N
        if samples_to_generate <= 0:
            continue
            
        print(f"iSMOTE: Class {cls}: Generating {samples_to_generate} synthetic samples...")
        
        # Get only samples from this class
        cls_indices = np.where(y == cls)[0]
        X_cls = X_flat[cls_indices]
        
        # Fit KNN on just this class to find neighbors for interpolation
        nn_cls = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_cls)))
        nn_cls.fit(X_cls)
        
        generated_count = 0
        patience = 0
        max_patience = samples_to_generate * 10 # Prevent infinite loops
        
        generated_samples = []
        
        while generated_count < samples_to_generate and patience < max_patience:
            patience += 1
            
            # 1. Randomly pick an instance from minority class
            idx = np.random.randint(0, len(X_cls))
            x_i = X_cls[idx]
            
            # 2. Pick a random nearest neighbor of x_i
            distances, indices = nn_cls.kneighbors([x_i])
            # Exclude itself (index 0)
            neighbor_idxs = indices[0][1:]
            if len(neighbor_idxs) == 0:
                 continue
            
            neighbor_idx = np.random.choice(neighbor_idxs)
            x_neighbor = X_cls[neighbor_idx]
            
            # 3. Create synthetic sample
            lam = np.random.uniform(0, 1)
            x_new = x_i + lam * (x_neighbor - x_i)
            
            # 4. iSMOTE Validation: Check if K nearest neighbors in the *entire* dataset are of the same class
            _, all_nn_indices = nn.kneighbors([x_new], n_neighbors=k_neighbors)
            neighbor_labels = y[all_nn_indices[0]]
            
            # If all K neighbors belong to class `cls`, the sample is accepted
            if np.all(neighbor_labels == cls):
                # Reshape back to (C, L)
                generated_samples.append(x_new.reshape(C, L))
                generated_count += 1
                
        if generated_count < samples_to_generate:
            print(f"Warning: iSMOTE could only generate {generated_count}/{samples_to_generate} valid samples for class {cls} within patience limit.")
            
        if generated_samples:
            new_X.append(np.stack(generated_samples))
            new_y.append(np.full(len(generated_samples), cls))
            
    X_resampled = np.concatenate(new_X, axis=0)
    y_resampled = np.concatenate(new_y, axis=0)
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    # Small test
    X_dummy = np.random.rand(100, 9, 128)
    # create imbalance
    y_dummy = np.concatenate([np.zeros(80), np.ones(15), np.full(5, 2)])
    print(f"Original shape: {X_dummy.shape}, y dist: {np.unique(y_dummy, return_counts=True)}")
    
    X_resampled, y_resampled = ismote(X_dummy, y_dummy)
    print(f"Resampled shape: {X_resampled.shape}, y dist: {np.unique(y_resampled, return_counts=True)}")
