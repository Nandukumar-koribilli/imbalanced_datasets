# SHAR — Self-Supervised HAR + Health Tracking Dashboard

PyTorch implementation of **Self-Supervised Learning for Activity Recognition Based on Datasets With Imbalanced Classes** (SHAR), plus a **Health Tracking Dashboard** that turns the activity-recognition engine into practical health insights: calories, steps, sleep staging, fall detection, and heart-rate/stress monitoring.

---

## 🚀 Easy Install (3 steps)

```powershell
# 1. Install dependencies
pip install torch torchvision numpy scipy pandas scikit-learn matplotlib seaborn streamlit

# 2. Launch the Health Tracking Dashboard (works out of the box)
python -m streamlit run health_tracking/app.py

# 3. (Optional) Train the model, then use the research dashboard
python main.py --pretrain_epochs 50 --finetune_epochs 50 --label_ratio 0.25
python -m streamlit run app.py
```

**The datasets are already in place** — `UCI HAR Dataset/` and `wisdm-dataset/` sit at the project root, right where the code expects them.

> **Windows tip:** if `streamlit` isn't recognized, use `python -m streamlit run ...` (always works),
> or open a **new** terminal.

---

## Requirements

* **Python 3.10 or newer** (tested on 3.11 and 3.14)
* `torch`, `torchvision`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`
* **Windows only:** the Microsoft Visual C++ 2015–2022 Redistributable (x64).
  If `import torch` fails with `OSError: [WinError 1114] ... c10.dll`, install it from
  <https://aka.ms/vs/17/release/vc_redist.x64.exe>. Reinstalling torch will NOT fix it.

Virtual environment (optional but recommended):

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install torch torchvision numpy scipy pandas scikit-learn matplotlib seaborn streamlit
```

---

## What's in the Box

```
imbalanced_datasets/
├── main.py                    # UCI HAR pipeline (pretrain → finetune → evaluate)
├── main_wisdm.py              # WISDM pipeline (same, 18 classes)
├── app.py                     # Research dashboard (Streamlit)
├── README.md                  # you are here
│
├── UCI HAR Dataset/           # ← dataset #1 (already extracted)
│   ├── train/  test/          # inertial signals + labels
│   ├── activity_labels.txt    # class names
│   └── README.txt, features.txt, features_info.txt
│
├── wisdm-dataset/             # ← dataset #2 (already extracted, cleaned)
│   ├── raw/phone/accel/       # 51 subject files (data_*_accel_phone.txt)
│   ├── activity_key.txt       # A..S → human-readable activities
│   ├── README.txt
│   └── WISDM-dataset-description.pdf
│
├── src/
│   ├── dataset_utils.py       # UCI HAR loading + DataLoaders
│   ├── dataset_utils_wisdm.py # WISDM loading + windowing
│   ├── ismote.py              # iSMOTE class-balancing algorithm
│   ├── random_masking.py      # Random Masking augmentation
│   ├── lambda_layer.py        # Lambda self-attention (einsum)
│   ├── shar_model.py          # SHAREncoder / SHAR_Pretrain / SHAR_Classifier
│   ├── train_pretrain.py      # Phase 1: self-supervised contrastive pre-training
│   └── train_finetune.py      # Phase 2: supervised fine-tuning + evaluation
│
├── health_tracking/
│   ├── app.py                 # Health Tracking Dashboard (Streamlit)
│   ├── health_metrics.py      # Pure-logic health metrics engine
│   ├── config.py              # MET values, HR zones, thresholds, label maps
│   └── README.md              # metrics formulas + how each page works
│
├── models/                    # trained checkpoints land here (.pth)
├── results/                   # generated figures (iSMOTE dist, confusion matrix)
├── scripts/                   # paper-table generation utilities
└── docs/                      # the SHAR paper (PDF)
```

Both datasets have been flattened and cleaned — no more triple-nested `wisdm-dataset/wisdm-dataset/wisdm-dataset/` folders, no `__pycache__`, no `.DS_Store`, no unused arff files, no editor swap files.

---

## How to Run

### 1. Health Tracking Dashboard (no training required)

```powershell
python -m streamlit run health_tracking/app.py
```

Open <http://localhost:8501>. The dashboard is a **two-step workflow**:

**🗂️ Step 1 — Dataset & Balancing** (the landing page, always visible)
1. Pick a dataset — **UCI HAR** (6 activities · 9 channels) or **WISDM** (18 activities · 3 channels). Each card shows what the dataset contains and whether it's found on disk.
2. The original class distribution loads and is charted — you see the raw imbalance ratio, total windows, class count, and sensor channels.
3. Click **🚀 Run iSMOTE**. A spinner runs while iSMOTE synthesises minority-class samples and rejects overlapping candidates (30–60 s on the demo subsample).
4. When it finishes you get: a green "Balancing finished for the *{dataset}* dataset" banner, side-by-side **before / after** distribution charts, per-class breakdown table, and a peek at a raw signal window from the selected dataset.

**📊 Step 2 — Dashboard** (unlocked after balancing)
The five health pages (Dashboard, Calories & Steps, Sleep Analysis, Fall Detection, Heart & Stress) plus the Live Activity Classifier become available in the sidebar. **All of them are scoped to the dataset you selected in Step 1** — activity dropdowns list only that dataset's activities, the timeline substitutes any missing labels, and the classifier loads that dataset's test split and checkpoint. Every dashboard page shows a "📦 Working on dataset: *X*" banner so you always know which dataset the numbers refer to.

You can go back to Step 1 at any time to switch datasets — the balancing state resets and the dashboard re-locks until you rebalance.

### 2. Train the SHAR model

```powershell
# UCI HAR — full run
python main.py --pretrain_epochs 50 --finetune_epochs 50 --label_ratio 0.25

# UCI HAR — quick smoke test (minutes on CPU)
python main.py --pretrain_epochs 1 --finetune_epochs 1 --label_ratio 0.10

# WISDM (18 classes)
python main_wisdm.py --pretrain_epochs 50 --finetune_epochs 50 --label_ratio 0.25
```

This produces (in `models/` and `results/`):

| Output | Purpose |
|---|---|
| `models/shar_encoder_pretrained.pth` | Self-supervised encoder weights |
| `models/shar_classifier_finetuned.pth` | **Full fine-tuned classifier** — used by the dashboards for live inference |
| `results/ismote_distribution.png` | Class balance before/after iSMOTE |
| `results/confusion_matrix.png` | Test-set confusion matrix |

WISDM outputs get a `_wisdm` suffix on all filenames.

**Arguments** (both pipelines): `--data_dir`, `--batch_size` (128), `--lr` (0.002, halved for fine-tuning), `--pretrain_epochs`, `--finetune_epochs`, `--label_ratio` (0.25 = use 25 % of labels).

### 3. Research Dashboard

```powershell
python -m streamlit run app.py
```

Explore raw sensor signals, visualize the iSMOTE balancing, and run live classification with the trained encoder.

---

## Key Contributions Implemented

1. **iSMOTE** — Improved SMOTE for imbalanced sensor datasets: generates a synthetic minority sample, then uses a K-nearest-neighbours check on the *entire dataset* to reject samples whose neighbours don't match the minority class. Prevents class overlap.
2. **Random Masking (RM)** — batch-level pretext-task augmentation that zeroes ~20 % of timesteps to break identity mappings during self-supervised training.
3. **Lambda Layer + 1D Causal CNN (SHAREncoder)** — an efficient self-attention alternative implemented from scratch with `einsum`, computing the Content Lambda ($L_c = K^T V$) and applying it to the Queries — much cheaper than standard attention for long temporal sequences.
4. **Contrastive Pre-training** — NT-Xent loss on two randomly-masked views of each sample builds representations that transfer with very little labelled data.
5. **Health Tracking Application** — MET-based calorie estimation, cadence-based step counting, threshold-based Deep/Light/REM sleep staging, three-phase fall detection (free-fall → impact → immobility), and simulated HR/HRV stress analysis, all driven by the recognized activity.

## Example Output on UCI HAR

Running the complete pipeline with `--label_ratio 0.25` yields approximately **85–87 % accuracy** and **84–87 % macro F1** — proving the self-supervised representations generalise well on minority classes (walking upstairs/downstairs) using only a quarter of the available labels.

![iSMOTE Graph](results/ismote_distribution.png)
![Confusion Matrix Heatmap](results/confusion_matrix.png)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognized | Use `python -m streamlit run ...`, or open a new terminal (PATH changes don't apply to already-open shells). |
| `OSError: [WinError 1114] ... c10.dll` on `import torch` | Install <https://aka.ms/vs/17/release/vc_redist.x64.exe>. Reinstalling torch does not help. |
| "Model or dataset not available" on the Live Activity Classifier page | Run `python main.py` once — it saves `models/shar_classifier_finetuned.pth`, which the dashboard then loads. |
| Classifier page warns "only the pretrained encoder was found" | Predictions unreliable until you finish fine-tuning (`main.py`/`main_wisdm.py` handles both phases). |
| iSMOTE is taking too long | The dashboard subsamples to 1,500 windows for a responsive demo (this cap is noted below the before/after chart). For the full-dataset run, use `python main.py` / `python main_wisdm.py`. |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502` |
| Want to relocate a dataset | Pass `--data_dir "path\to\your\folder"` to `main.py` or `main_wisdm.py`. |
