import argparse
import torch
import os
import numpy as np
from src.train_pretrain_mae import pretrain_mae
from src.train_finetune_mae import finetune_mae

def main():
    # Reproducibility: fixed seeds for numpy (iSMOTE) and torch (training)
    np.random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(
        description="MAE-SHAR: Masked Autoencoder Pre-training for Activity Recognition"
    )
    parser.add_argument('--data_dir', type=str, default='UCI HAR Dataset',
                        help='Directory for datasets')
    parser.add_argument('--pretrain_epochs', type=int, default=80,
                        help='Number of epochs for MAE pre-training')
    parser.add_argument('--finetune_epochs', type=int, default=60,
                        help='Number of epochs for fine-tuning (supervised)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1.5e-3,
                        help='Learning rate for MAE pre-training')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Ratio of patches to mask during MAE pre-training')
    parser.add_argument('--label_ratio', type=float, default=0.25,
                        help='Percentage of labels used in fine-tuning phase')
    parser.add_argument('--max_pretrain_samples', type=int, default=None,
                        help='Cap the pretraining set size for faster CPU runs '
                             '(default: use all)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"===========================================================")
    print(f"           MAE-SHAR Pipeline Start (UCI HAR)               ")
    print(f"===========================================================")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Mask Ratio: {args.mask_ratio * 100}%")
    print(f"Target Label Ratio for Fine-Tuning: {args.label_ratio * 100}%")
    print(f"===========================================================\n")

    if not os.path.isdir(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found.")
        print("Please ensure you have executed the download script first.")
        return

    # Phase 1: MAE Self-Supervised Pre-Training
    print(">>> PHASE 1: MAE Pre-training (Reconstruction)")
    _ = pretrain_mae(
        data_dir=args.data_dir,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        device=device,
        max_pretrain_samples=args.max_pretrain_samples,
    )

    # Phase 2: Supervised Fine-Tuning
    print(">>> PHASE 2: Fine-Tuning & Evaluation")
    _ = finetune_mae(
        data_dir=args.data_dir,
        encoder_path="models/mae_encoder_pretrained.pth",
        label_ratio=args.label_ratio,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr * 0.33,  # Lower LR for fine-tuning
        device=device,
    )

    print("\n===========================================================")
    print(f"           MAE-SHAR Pipeline Complete                      ")
    print("===========================================================")

if __name__ == "__main__":
    main()
