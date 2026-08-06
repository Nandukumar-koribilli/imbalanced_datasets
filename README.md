# Self-Supervised Learning for Activity Recognition on Imbalanced Datasets

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a **Self-supervised Human Activity Recognition (SHAR)** system that addresses class imbalance in HAR datasets using **I-SMOTE (Improved Synthetic Minority Over-sampling Technique)** combined with **Masked Autoencoder (MAE)** pre-training.

### Key Features

- 🔄 **MAE Self-supervised Pre-training**: Learns robust sensor representations without labels using Masked Autoencoder reconstruction
- ⚖️ **I-SMOTE Augmentation**: Handles class imbalance with distance-weighted synthetic sample generation and KNN validation
- 🏗️ **Transformer Architecture**: Modern transformer-based encoder for temporal sensor data
- 📊 **Comprehensive Evaluation**: Per-class metrics, confusion matrices, and embedding visualizations
- 🎮 **Interactive Demo**: Streamlit web app for model exploration and live classification

## Architecture

The pipeline consists of three main phases:

```
Raw Sensor Data → I-SMOTE Balancing → MAE Pre-training → Supervised Fine-tuning → Activity Prediction
```

### Phase 1: I-SMOTE Class Balancing + MAE Self-Supervised Pre-training
- Balances the dataset using I-SMOTE before pre-training
- Patches sensor time-series into fixed-size segments
- Randomly masks 75% of patches
- Reconstructs masked patches using a lightweight decoder
- Learns rich temporal representations without labels

### Phase 2: Supervised Fine-tuning
- Loads the pre-trained MAE encoder
- Fine-tunes on a subset of labeled data (default 25%)
- Trains a classification head for activity recognition
- Uses cosine annealing learning rate schedule

## Dataset

### UCI HAR Dataset
- **Activities**: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- **Sensors**: Accelerometer + Gyroscope (9 channels)
- **Samples**: 7,352 train / 2,947 test
- **Window**: 128 timesteps at 50Hz (2.56 seconds)

## Installation

```bash
# Clone the repository
git clone https://github.org/yourusername/imbalanced_datasets.git
cd imbalanced_datasets

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

## Dataset Setup

```bash
# Download and extract UCI HAR Dataset
mkdir -p data
cd data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"
cd ..
```

Or manually download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and place the `UCI HAR Dataset` folder in the project root.

## Usage

### Training

```bash
python main.py

python main.py --pretrain_epochs 80 --finetune_epochs 60
```

### Advanced Options

```bash
python main.py \
    --data_dir "UCI HAR Dataset" \
    --pretrain_epochs 80 \
    --finetune_epochs 60 \
    --batch_size 128 \
    --lr 1.5e-3 \
    --mask_ratio 0.75 \
    --label_ratio 0.25 \
    --max_pretrain_samples 5000  # For faster CPU runs
```

### Interactive Demo

```bash
python -m streamlit run health_tracking/app.py
```

### Run Ablation Studies

```bash
python scripts/run_ablations.py
```

### Generate Paper Tables

```bash
python scripts/generate_paper_tables.py
```

### Visualize Embeddings

```bash
python scripts/plot_embeddings.py
```

## Project Structure

```
imbalanced_datasets/
├── main.py                         # Entry point: UCI HAR + MAE + I-SMOTE pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore
├── UCI HAR Dataset/                # Dataset (download separately — see Dataset Setup)
│   ├── train/
│   │   ├── Inertial Signals/       # 9 raw sensor .txt files (body_acc, body_gyro, total_acc)
│   │   ├── X_train.txt             # Pre-computed 561 features
│   │   ├── y_train.txt             # Activity labels (1-6)
│   │   └── subject_train.txt       # Subject IDs
│   ├── test/
│   │   ├── Inertial Signals/       # 9 raw sensor .txt files
│   │   ├── X_test.txt
│   │   ├── y_test.txt
│   │   └── subject_test.txt
│   ├── activity_labels.txt         # Activity ID → name mapping
│   ├── features.txt                # Feature names
│   └── features_info.txt           # Feature descriptions
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py            # UCI HAR data loader
│   ├── mae_model.py                # Masked Autoencoder (MAE1D, MAEEncoder)
│   ├── shar_model.py               # SHAR encoder & classifier heads
│   ├── ismote.py                   # I-SMOTE oversampling
│   ├── lambda_layer.py             # Lambda attention layer
│   ├── random_masking.py           # Random masking augmentation
│   ├── train_pretrain_mae.py       # MAE pre-training logic
│   └── train_finetune_mae.py       # MAE fine-tuning & evaluation logic
├── health_tracking/                # Health Tracking Application Module
│   ├── app.py                      # Health tracking Streamlit dashboard
│   ├── config.py                   # MET values, sleep & fall thresholds
│   ├── health_metrics.py           # BMR, calories, step counter, sleep & fall engine
│   └── README.md
├── models/                         # Saved model weights (.pth)
├── results/                        # Training results & plots
├── scripts/                        # Utility scripts
│   ├── run_ablations.py            # Ablation studies
│   ├── generate_paper_tables.py    # Format results for paper
│   ├── make_architecture_diagram.py # Generate architecture figures
│   └── plot_embeddings.py          # t-SNE / PCA embedding plots
├── tests/                          # Unit tests
│   ├── test_health_metrics.py      # Health metrics engine tests
│   ├── test_ismote.py              # I-SMOTE tests
│   ├── test_labels.py              # Label consistency tests
│   └── test_model.py               # Model architecture tests
└── docs/                           # Documentation & diagrams
```

## Results

### UCI HAR Dataset (MAE + I-SMOTE)

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.2% |
| Macro F1-Score | 93.0% |

### Per-Class Performance

| Activity | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Walking | 0.95 | 0.93 | 0.94 |
| Walking Upstairs | 0.91 | 0.89 | 0.90 |
| Walking Downstairs | 0.88 | 0.90 | 0.89 |
| Sitting | 0.94 | 0.96 | 0.95 |
| Standing | 0.92 | 0.91 | 0.92 |
| Laying | 0.99 | 1.00 | 1.00 |

## References

1. He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
2. Chawla, N.V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." JAIR 2002.
3. Anguita, D., et al. "A Public Domain Dataset for Human Activity Recognition Using Smartphones." ESANN 2013.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
