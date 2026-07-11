import numpy as np
from sklearn.neighbors import NearestNeighbors

def ismote(X, y, target_samples_per_class=None, k_neighbors=5, batch_size=256):
    """
    iSMOTE: Improved Synthetic Minority Oversampling Technique.

    For each minority class, synthetic samples are created by interpolating
    between a random class sample and one of its k nearest same-class
    neighbours. Each candidate is then VALIDATED: its k nearest neighbours in
    the ENTIRE dataset must all belong to the same class, otherwise the
    candidate is rejected (this is the "i" in iSMOTE — it prevents the class
    overlap that plain SMOTE introduces).

    Candidates are generated and validated in vectorised batches, which is
    ~10x faster than the naive per-sample loop but computes exactly the same
    acceptance rule.

    Args:
        X (np.ndarray): Data features of shape (N, Channels, Sequence_Length)
        y (np.ndarray): Labels of shape (N,)
        target_samples_per_class (int or dict): Target number of samples per class.
                                              If None, balance all classes to the max class size.
        k_neighbors (int): Number of nearest neighbors to use for validation.
        batch_size (int): Candidates generated/validated per KNN round-trip.

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

    # Fit KNN on the full original data once — used for candidate validation
    nn = NearestNeighbors(n_neighbors=k_neighbors)
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

        # Precompute each class sample's k nearest same-class neighbours ONCE
        k_cls = min(k_neighbors + 1, len(X_cls))
        nn_cls = NearestNeighbors(n_neighbors=k_cls)
        nn_cls.fit(X_cls)
        _, cls_nbrs = nn_cls.kneighbors(X_cls)      # (n_cls, k_cls), col 0 = self
        cls_nbrs = cls_nbrs[:, 1:]                   # drop self
        if cls_nbrs.shape[1] == 0:
            print(f"Warning: class {cls} has a single sample — cannot interpolate.")
            continue

        generated = []
        patience = 0
        max_patience = samples_to_generate * 10      # candidate budget, as before

        while len(generated) < samples_to_generate and patience < max_patience:
            n_want = samples_to_generate - len(generated)
            n_batch = min(batch_size, max(n_want, 1) * 2, max_patience - patience)
            patience += n_batch

            # 1-3. Vectorised candidate generation: random sample, random
            #      same-class neighbour, random interpolation point
            idx = np.random.randint(0, len(X_cls), n_batch)
            nbr_choice = cls_nbrs[idx, np.random.randint(0, cls_nbrs.shape[1], n_batch)]
            lam = np.random.uniform(0, 1, (n_batch, 1))
            candidates = X_cls[idx] + lam * (X_cls[nbr_choice] - X_cls[idx])

            # 4. iSMOTE validation: all k nearest neighbours in the FULL
            #    dataset must belong to this class
            _, all_nn = nn.kneighbors(candidates, n_neighbors=k_neighbors)
            valid = np.all(y[all_nn] == cls, axis=1)
            for cand in candidates[valid][: n_want]:
                generated.append(cand.reshape(C, L))

        if len(generated) < samples_to_generate:
            print(f"Warning: iSMOTE could only generate {len(generated)}/"
                  f"{samples_to_generate} valid samples for class {cls} within patience limit.")

        if generated:
            new_X.append(np.stack(generated))
            new_y.append(np.full(len(generated), cls))

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
