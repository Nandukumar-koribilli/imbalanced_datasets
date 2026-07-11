import argparse
import torch
import os
import numpy as np
from src.train_pretrain_wisdm import pretrain_shar
from src.train_finetune_wisdm import finetune_shar

def main():
    parser = argparse.ArgumentParser(description="SHAR: Self-Supervised Learning for Activity Recognition")
    parser.add_argument('--data_dir', type=str, default=os.path.join('wisdm-dataset', 'raw', 'phone', 'accel'), help='Directory for datasets')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of epochs for pre-training (self-supervised)')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Number of epochs for fine-tuning (supervised)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--label_ratio', type=float, default=0.25, help='Percentage of labels used in fine-tuning phase')
    parser.add_argument('--max_pretrain_samples', type=int, default=15000,
                        help='Cap the pretraining set size for faster CPU runs '
                             '(WISDM has ~45k windows; pass 0 to use all)')
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cpu' # Forced CPU as requested
    print(f"===========================================================")
    print(f"                    SHAR Pipeline Start                    ")
    print(f"===========================================================")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Target Label Ratio for Fine-Tuning: {args.label_ratio * 100}%")
    print(f"===========================================================\n")
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found.")
        print("Please ensure you have executed the download script first.")
        return
        
    # Phase 1: Self-Supervised Pre-Training
    print(">>> PHASE 1: Pre-training")
    _ = pretrain_shar(
        data_dir=args.data_dir,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        max_pretrain_samples=args.max_pretrain_samples or None
    )
    
    # Phase 2: Supervised Fine-Tuning
    print(">>> PHASE 2: Fine-Tuning & Evaluation")
    _ = finetune_shar(
        data_dir=args.data_dir,
        encoder_path="models/shar_encoder_pretrained_wisdm.pth",
        label_ratio=args.label_ratio,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr * 0.5, # Slightly lower learning rate for fine-tuning
        device=device
    )
    
    print("\n===========================================================")
    print(f"                    SHAR Pipeline Complete                 ")
    print("===========================================================")

if __name__ == "__main__":
    main()
