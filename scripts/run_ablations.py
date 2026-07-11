"""
Ablation study: does self-supervised pre-training + iSMOTE actually help
when labels are scarce?

Compares three setups across label ratios, with multiple seeds:
    A. supervised   — train the classifier from scratch (no pre-training)
    B. ssl          — SSL pre-training WITHOUT iSMOTE balancing
    C. ssl_ismote   — SSL pre-training WITH iSMOTE balancing  (full SHAR)

Pre-training is done ONCE per variant (B, C) and the frozen encoder weights
are reused for every label ratio / seed — that's methodologically correct
(the pretext task never sees labels) and makes the sweep tractable on CPU.

Usage (from project root):
    python scripts/run_ablations.py                           # full study
    python scripts/run_ablations.py --quick                   # smoke test
    python scripts/run_ablations.py --ratios 0.05 0.25 --seeds 42

Outputs:
    results/ablations.json   — every run's accuracy + macro F1
    results/ablations.png    — label-ratio curve with error bars
    results/ablations.md     — README-ready markdown table
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset_utils import UCIHARDataset
from src.ismote import ismote
from src.random_masking import apply_random_masking_batch
from src.shar_model import SHAREncoder, SHAR_Pretrain, SHAR_Classifier
from src.train_pretrain import contrastive_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

VARIANT_LABELS = {
    "supervised": "Supervised from scratch",
    "ssl":        "SSL pretrain (no iSMOTE)",
    "ssl_ismote": "SSL + iSMOTE (SHAR, ours)",
}
VARIANT_COLORS = {"supervised": "#ef4444", "ssl": "#f59e0b", "ssl_ismote": "#22c55e"}


def pretrain_encoder(X, epochs, batch_size, lr, seed, tag):
    """Self-supervised contrastive pre-training; returns encoder state_dict."""
    torch.manual_seed(seed)
    model = SHAR_Pretrain(in_channels=X.shape[1], seq_len=X.shape[2])
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X))),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, _ in loader:
            opt.zero_grad()
            z_i = model(apply_random_masking_batch(xb, mask_prob=0.2))
            z_j = model(apply_random_masking_batch(xb, mask_prob=0.2))
            loss = contrastive_loss(z_i, z_j, temperature=0.5)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"    [{tag}] pretrain epoch {ep+1}/{epochs}  loss {total/len(loader):.4f}",
              flush=True)
    return copy.deepcopy(model.encoder.state_dict())


def finetune_and_eval(encoder_state, X_tr, y_tr, X_te, y_te,
                      label_ratio, epochs, batch_size, lr, seed):
    """Fine-tune on a stratified label subset; return (test_acc, macro_f1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = max(len(np.unique(y_tr)) * 2, int(len(y_tr) * label_ratio))
    X_sub, _, y_sub, _ = train_test_split(
        X_tr, y_tr, train_size=n, stratify=y_tr, random_state=seed)

    encoder = SHAREncoder(in_channels=X_tr.shape[1], seq_len=X_tr.shape[2])
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    model = SHAR_Classifier(encoder, num_classes=len(np.unique(y_tr)))

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_sub, dtype=torch.float32),
                      torch.tensor(y_sub, dtype=torch.long)),
        batch_size=min(batch_size, len(y_sub)), shuffle=True, drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 512):
            xb = torch.tensor(X_te[i:i + 512], dtype=torch.float32)
            preds.extend(model(xb).argmax(1).tolist())
    preds = np.array(preds)
    return float((preds == y_te).mean() * 100), float(f1_score(y_te, preds, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--finetune_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run: 2 pretrain / 3 finetune epochs, 1 seed, 2 ratios")
    args = ap.parse_args()
    if args.quick:
        args.pretrain_epochs, args.finetune_epochs = 2, 3
        args.seeds, args.ratios = [42], [0.05, 0.25]

    print("Loading UCI HAR ...", flush=True)
    train_ds = UCIHARDataset(os.path.join(ROOT, "UCI HAR Dataset"), split="train")
    test_ds = UCIHARDataset(os.path.join(ROOT, "UCI HAR Dataset"), split="test")
    X_tr = np.asarray(train_ds.X, dtype=np.float32)
    y_tr = np.asarray(train_ds.y, dtype=np.int64)
    X_te = np.asarray(test_ds.X, dtype=np.float32)
    y_te = np.asarray(test_ds.y, dtype=np.int64)

    # ── One pre-training per variant, reused across ratios & seeds ──────────
    encoders = {"supervised": None}
    t0 = time.time()
    print("\n[1/2] Pre-training encoder WITHOUT iSMOTE ...", flush=True)
    encoders["ssl"] = pretrain_encoder(
        X_tr, args.pretrain_epochs, args.batch_size, 0.002, 42, "ssl")

    print("\n[2/2] Pre-training encoder WITH iSMOTE ...", flush=True)
    np.random.seed(42)
    X_bal, _ = ismote(X_tr, y_tr, k_neighbors=5)
    encoders["ssl_ismote"] = pretrain_encoder(
        X_bal, args.pretrain_epochs, args.batch_size, 0.002, 42, "ssl_ismote")

    # ── Sweep ────────────────────────────────────────────────────────────────
    runs = []
    total = len(encoders) * len(args.ratios) * len(args.seeds)
    done = 0
    for variant, enc_state in encoders.items():
        for ratio in args.ratios:
            for seed in args.seeds:
                done += 1
                acc, f1 = finetune_and_eval(
                    enc_state, X_tr, y_tr, X_te, y_te,
                    ratio, args.finetune_epochs, args.batch_size, 0.001, seed)
                runs.append({"variant": variant, "label_ratio": ratio,
                             "seed": seed, "test_acc": round(acc, 2),
                             "macro_f1": round(f1, 4)})
                print(f"  ({done}/{total}) {variant:11s} ratio={ratio:<5} seed={seed} "
                      f"→ acc {acc:.2f}%  F1 {f1:.4f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    payload = {"config": vars(args), "elapsed_sec": round(time.time() - t0, 1),
               "runs": runs}
    with open(os.path.join(RESULTS, "ablations.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ── Plot: F1 vs label ratio, mean ± std ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    for variant in encoders:
        means, stds = [], []
        for r in args.ratios:
            f1s = [x["macro_f1"] for x in runs
                   if x["variant"] == variant and x["label_ratio"] == r]
            means.append(np.mean(f1s)); stds.append(np.std(f1s))
        ax.errorbar([r * 100 for r in args.ratios], means, yerr=stds,
                    label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant],
                    marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("Labelled training data (%)", color="#94a3b8")
    ax.set_ylabel("Macro F1 on test set", color="#94a3b8")
    ax.set_title("SSL + iSMOTE vs. baselines on UCI HAR",
                 color="#f1f5f9", fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "ablations.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")

    # ── Markdown table for the README ────────────────────────────────────────
    lines = ["| Setup | " + " | ".join(f"{int(r*100)}% labels" for r in args.ratios) + " |",
             "|---|" + "---|" * len(args.ratios)]
    for variant in encoders:
        cells = []
        for r in args.ratios:
            f1s = [x["macro_f1"] for x in runs
                   if x["variant"] == variant and x["label_ratio"] == r]
            cells.append(f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        lines.append(f"| {VARIANT_LABELS[variant]} | " + " | ".join(cells) + " |")
    with open(os.path.join(RESULTS, "ablations.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nSaved results/ablations.json, ablations.png, ablations.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
