# Health Tracking System — Dashboard

This folder contains a premium, dark-themed Health Monitoring Dashboard powered by the **SHAR (Self-Supervised HAR)** activity recognition engine.

The dashboard provides real-time health insights, including:
* **Calories & Steps:** Estimates daily energy expenditure using MET values and simulates step counting.
* **Sleep Analysis:** Classifies sleep stages (Deep, Light, REM) and generates a sleep score based on accelerometer movement patterns.
* **Fall Detection:** Uses threshold-based detection (free-fall vs impact) to simulate and detect falls.
* **Heart & Stress:** Simulates physiological metrics (Heart Rate Variability, Stress Level, average Heart Rate) across different activity zones.
* **Live Activity Classifier:** Runs live classifications using the pre-trained SHAR model from the main project.

## Setup & Running Instructions

Since this dashboard relies on the main SHAR model and utility scripts located in the root project directory, you must run it using the project's virtual environment.

### 1. Activate the Virtual Environment
Assuming you have already created the virtual environment in the project's root folder (`.venv`), navigate to the `health_tracking` folder and activate it:

```bash
# Navigate to the health tracking directory
cd health_tracking

# Activate the virtual environment from the parent directory
source ../.venv/bin/activate
```
*(On Windows, use: `..\.venv\Scripts\activate`)*

### 2. Install Required Dependencies
If you haven't already installed the required packages in the root project, make sure they are installed:

```bash
pip install streamlit pandas numpy matplotlib seaborn torch torchvision scikit-learn
```

### 3. Run the Dashboard
Use `streamlit` to launch the application:

```bash
streamlit run app.py
```

This will spin up a local server. You can view the dashboard by opening the Local URL (usually `http://localhost:8501`) in your browser.

## Note on Live Classification
To use the **"Live Activity Classifier"** page in the dashboard:
1. You must have trained the SHAR model first by running `python main.py` in the root directory.
2. The model weights (`shar_encoder_pretrained.pth`) must be present in the `models/` directory in the root project.
3. The raw `UCI HAR Dataset` folder must also be present in the root project so the app can sample test data.
