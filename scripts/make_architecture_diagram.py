"""
Generate the SHAR pipeline architecture diagram.

Run from the project root:
    python scripts/make_architecture_diagram.py
Writes: docs/architecture.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, CARD, BORDER = "#0f172a", "#1e293b", "#334155"
TEXT, MUTED = "#f1f5f9", "#94a3b8"
ACCENT, GREEN, AMBER, RED, PURPLE = "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a855f7"


def box(ax, x, y, w, h, title, subtitle="", edge=ACCENT, title_size=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=CARD, edgecolor=edge, linewidth=1.6))
    cy = y + h / 2
    if subtitle:
        ax.text(x + w / 2, cy + 0.14, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold")
        ax.text(x + w / 2, cy - 0.22, subtitle, ha="center", va="center",
                color=MUTED, fontsize=7.5)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, linewidth=1.6))


def main():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    ax.text(7.5, 7.85, "SHAR — Self-Supervised Learning for Imbalanced Activity Recognition",
            ha="center", color=TEXT, fontsize=15, fontweight="bold")

    # ── Phase 1: self-supervised pre-training (top band) ────────────────────
    ax.text(0.4, 6.9, "PHASE 1 · SELF-SUPERVISED PRE-TRAINING (no labels)",
            color=AMBER, fontsize=10, fontweight="bold")
    box(ax, 0.4, 5.3, 2.1, 1.1, "Raw sensor windows",
        "UCI HAR 9ch / WISDM 3ch\nimbalanced classes", edge=RED)
    box(ax, 3.1, 5.3, 1.9, 1.1, "iSMOTE", "KNN-validated\nminority oversampling", edge=GREEN)
    box(ax, 5.6, 5.3, 2.0, 1.1, "Random Masking ×2", "two corrupted views\n~20% timesteps zeroed", edge=AMBER)
    box(ax, 8.2, 5.3, 2.6, 1.1, "SHAREncoder",
        "Causal Conv1d ×2 → Lambda\nself-attention → FC (256-d)", edge=ACCENT)
    box(ax, 11.4, 5.3, 1.6, 1.1, "Projector", "MLP → 128-d", edge=ACCENT)
    box(ax, 13.4, 5.3, 1.3, 1.1, "NT-Xent", "contrastive\nloss", edge=PURPLE)

    y_mid = 5.85
    arrow(ax, 2.5, y_mid, 3.1, y_mid)
    arrow(ax, 5.0, y_mid, 5.6, y_mid)
    arrow(ax, 7.6, y_mid, 8.2, y_mid)
    arrow(ax, 10.8, y_mid, 11.4, y_mid)
    arrow(ax, 13.0, y_mid, 13.4, y_mid)

    # ── Transfer arrow ───────────────────────────────────────────────────────
    arrow(ax, 9.5, 5.3, 9.5, 4.15, color=GREEN)
    ax.text(9.75, 4.7, "transfer pretrained\nencoder weights", color=GREEN, fontsize=8)

    # ── Phase 2: supervised fine-tuning (middle band) ────────────────────────
    ax.text(0.4, 4.0, "PHASE 2 · SUPERVISED FINE-TUNING (few labels)",
            color=GREEN, fontsize=10, fontweight="bold")
    box(ax, 0.4, 2.5, 2.4, 1.1, "Labelled subset",
        "only 25% (or less)\nof training labels", edge=AMBER)
    box(ax, 8.2, 2.5, 2.6, 1.1, "SHAREncoder", "initialised from\nPhase 1 weights", edge=ACCENT)
    box(ax, 11.4, 2.5, 1.9, 1.1, "Classifier head", "Linear 256→64→C", edge=ACCENT)
    box(ax, 13.7, 2.5, 1.0, 1.1, "Activity", "6 / 18\nclasses", edge=GREEN)

    y_ft = 3.05
    arrow(ax, 2.8, y_ft, 8.2, y_ft)
    arrow(ax, 10.8, y_ft, 11.4, y_ft)
    arrow(ax, 13.3, y_ft, 13.7, y_ft)

    # ── Application band ─────────────────────────────────────────────────────
    ax.text(0.4, 1.75, "APPLICATION · HEALTH TRACKING DASHBOARD",
            color=ACCENT, fontsize=10, fontweight="bold")
    apps = [("Calories", "MET-based energy"), ("Steps", "cadence estimate"),
            ("Sleep", "stage classification"), ("Falls", "3-phase detection"),
            ("Heart", "HR / HRV / stress")]
    for i, (t, s) in enumerate(apps):
        box(ax, 0.4 + i * 2.5, 0.25, 2.2, 1.0, t, s, edge=BORDER, title_size=9)
    arrow(ax, 14.2, 2.5, 14.2, 1.9, color=MUTED)
    ax.plot([1.5, 14.2], [1.9, 1.9], color=MUTED, linewidth=1.2)
    for i in range(5):
        arrow(ax, 1.5 + i * 2.5, 1.9, 1.5 + i * 2.5, 1.35, color=MUTED)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "architecture.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
