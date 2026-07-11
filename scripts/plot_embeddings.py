"""
t-SNE visualization of SHAR encoder embeddings — before vs. after
self-supervised pre-training.

The killer figure for the SSL claim: a randomly-initialised encoder produces
an unstructured blob, while the pre-trained encoder (which never saw a label)
separates activities into clusters.

Run from the project root:
    python scripts/plot_embeddings.py
Writes: results/embedding_tsne.png
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shar_model import SHAREncoder
from src.dataset_utils import UCIHARDataset

ACTIVITY_NAMES = ["Walking", "Walking Upstairs", "Walking Downstairs",
                  "Sitting", "Standing", "Laying"]
COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"]

N_SAMPLES_PER_CLASS = 200
SEED = 42


def embed(encoder: SHAREncoder, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            outs.append(encoder(xb).numpy())
    return np.concatenate(outs)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "UCI HAR Dataset")
    ckpt = os.path.join(root, "models", "shar_encoder_pretrained.pth")
    if not os.path.isdir(data_dir):
        sys.exit("UCI HAR Dataset folder not found at project root.")
    if not os.path.exists(ckpt):
        sys.exit("Pretrained encoder not found — run `python main.py` first.")

    rng = np.random.default_rng(SEED)
    ds = UCIHARDataset(data_dir, split="test")
    X, y = np.asarray(ds.X, dtype=np.float32), np.asarray(ds.y)

    # Stratified subsample so t-SNE stays fast and every class is visible
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        keep.extend(rng.choice(idx, min(N_SAMPLES_PER_CLASS, len(idx)), replace=False))
    keep = np.array(keep)
    X, y = X[keep], y[keep]
    print(f"Embedding {len(X)} test windows ({N_SAMPLES_PER_CLASS}/class)...")

    torch.manual_seed(SEED)
    enc_random = SHAREncoder(in_channels=9, seq_len=128)
    enc_pretrained = SHAREncoder(in_channels=9, seq_len=128)
    enc_pretrained.load_state_dict(torch.load(ckpt, map_location="cpu"))

    panels = [
        ("Random init — before pre-training", embed(enc_random, X)),
        ("After self-supervised pre-training (no labels used)", embed(enc_pretrained, X)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#0f172a")
    for ax, (title, emb) in zip(axes, panels):
        print(f"t-SNE: {title} ...")
        pts = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=SEED).fit_transform(emb)
        ax.set_facecolor("#0f172a")
        for c in range(6):
            m = y == c
            ax.scatter(pts[m, 0], pts[m, 1], s=14, alpha=0.75,
                       color=COLORS[c], label=ACTIVITY_NAMES[c],
                       edgecolors="none")
        ax.set_title(title, color="#f1f5f9", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
    axes[1].legend(loc="upper right", fontsize=9, facecolor="#1e293b",
                   edgecolor="#334155", labelcolor="#94a3b8", markerscale=1.6)
    fig.suptitle("SHAR encoder embeddings (t-SNE) — what self-supervision learned",
                 color="#f1f5f9", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(root, "results", "embedding_tsne.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
