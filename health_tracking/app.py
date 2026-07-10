"""
🏥 Health Tracking System — Streamlit Dashboard
=================================================
A premium, dark-themed health monitoring dashboard powered by
the SHAR (Self-Supervised HAR) activity recognition engine.

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys, os

# ── Ensure parent project is importable ──────────────────────────────────────
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    UCI_HAR_LABELS, WISDM_LABELS, MET_VALUES,
    ACTIVITY_CATEGORIES, CATEGORY_COLORS, PALETTE,
    DEFAULT_WEIGHT_KG, DEFAULT_HEIGHT_CM, DEFAULT_AGE, DEFAULT_GENDER,
    ACTIVITY_HR_ZONE,
)
from health_metrics import (
    compute_bmr, calories_burned, compute_daily_calories, active_calories,
    count_steps, estimate_steps_from_timeline, sedentary_ratio,
    simulate_sleep_night, detect_fall, simulate_heart_rate, simulate_hrv,
    generate_daily_timeline,
)

# ── Try to load the SHAR model from the parent project ──────────────────────
# NOTE: torch can fail with OSError (broken DLL / unsupported Python build),
# not just ImportError — catch everything so the dashboard still runs with
# simulated data and only the live classifier is disabled.
MODEL_AVAILABLE = False
MODEL_IMPORT_ERROR = None
try:
    from src.shar_model import SHAREncoder, SHAR_Classifier
    from src.dataset_utils import UCIHARDataset
    from src.dataset_utils_wisdm import WISDMDataset
    import torch
    MODEL_AVAILABLE = True
except Exception as e:
    MODEL_IMPORT_ERROR = str(e)

# iSMOTE is pure numpy/sklearn — always available
from src.ismote import ismote

# ── Dataset registry ─────────────────────────────────────────────────────────
UCI_HAR_DIR   = os.path.join(PARENT_DIR, "UCI HAR Dataset")
WISDM_ACCEL_DIR = os.path.join(PARENT_DIR, "wisdm-dataset", "raw", "phone", "accel")

DATASETS = {
    "UCI HAR": {
        "path": UCI_HAR_DIR,
        "labels": UCI_HAR_LABELS,
        "in_channels": 9,
        "num_classes": 6,
        "seq_len": 128,
        "classifier_ckpt": "shar_classifier_finetuned.pth",
        "encoder_ckpt":    "shar_encoder_pretrained.pth",
        "channel_names": ["body_acc_x","body_acc_y","body_acc_z",
                           "body_gyro_x","body_gyro_y","body_gyro_z",
                           "total_acc_x","total_acc_y","total_acc_z"],
        "description": "6 activities · 30 subjects · smartphone waist sensor · 9 channels @ 50 Hz",
    },
    "WISDM": {
        "path": WISDM_ACCEL_DIR,
        "labels": WISDM_LABELS,
        "in_channels": 3,
        "num_classes": 18,
        "seq_len": 128,
        "classifier_ckpt": "shar_classifier_finetuned_wisdm.pth",
        "encoder_ckpt":    "shar_encoder_pretrained_wisdm.pth",
        "channel_names": ["accel_x","accel_y","accel_z"],
        "description": "18 activities · 51 subjects · phone accelerometer · 3 channels @ 20 Hz",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Health Tracking System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stable per-session simulation seed ───────────────────────────────────────
# Streamlit reruns the whole script on every widget interaction; without a
# fixed seed all simulated metrics would re-roll on every click.
if "sim_seed" not in st.session_state:
    st.session_state.sim_seed = 42
np.random.seed(st.session_state.sim_seed)

# ── Session state: dataset selection + balancing pipeline ────────────────────
for k, v in {
    "selected_dataset": None,
    "loaded": False,
    "balanced": False,
    "X_orig": None, "y_orig": None,          # FULL training set
    "X_bal": None,  "y_bal": None,           # iSMOTE-balanced working set
    "X_sub": None,  "y_sub": None,           # subsample iSMOTE actually saw
    "orig_counts": None, "bal_counts": None,
    "sub_counts": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def activities_for(dataset_name):
    """Return the list of activity strings for the selected dataset."""
    return list(DATASETS[dataset_name]["labels"].values())


@st.cache_data(show_spinner=False)
def load_training_data(dataset_name: str):
    """Load train split for balancing. Cached — datasets are read from disk once."""
    if dataset_name == "UCI HAR":
        if not MODEL_AVAILABLE:
            raise RuntimeError("PyTorch is required to load datasets.")
        ds = UCIHARDataset(DATASETS["UCI HAR"]["path"], split="train")
        return np.asarray(ds.X, dtype=np.float32), np.asarray(ds.y, dtype=np.int64)
    if dataset_name == "WISDM":
        if not MODEL_AVAILABLE:
            raise RuntimeError("PyTorch is required to load datasets.")
        ds = WISDMDataset(DATASETS["WISDM"]["path"], split="train")
        return np.asarray(ds.X, dtype=np.float32), np.asarray(ds.y, dtype=np.int64)
    raise ValueError(dataset_name)


@st.cache_data(show_spinner=False)
def run_ismote_cached(dataset_name: str, max_samples: int = 1500):
    """
    Run iSMOTE on the training split of *dataset_name*, cached in memory.
    iSMOTE runs a full-dataset KNN validation for every synthetic sample,
    so we cap the working set to keep the dashboard responsive. The
    balancing math and rejection logic are identical — just on fewer points.
    Returns (X_orig, y_orig, X_bal, y_bal).
    """
    X, y = load_training_data(dataset_name)
    if len(y) > max_samples:
        # Random subsample — proportions are preserved on average, so the
        # class imbalance the user came here to see is not artificially fixed.
        rng = np.random.default_rng(0)
        keep = rng.choice(len(y), max_samples, replace=False)
        X, y = X[keep], y[keep]
    X_bal, y_bal = ismote(X, y, k_neighbors=5)
    return X, y, X_bal, y_bal

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM CSS — Premium Dark Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
*, html, body, .stApp {{
    font-family: 'Inter', sans-serif !important;
}}
.stApp {{
    background: linear-gradient(135deg, {PALETTE["bg_dark"]} 0%, #1a1a2e 50%, {PALETTE["bg_dark"]} 100%) !important;
}}
section[data-testid="stSidebar"] {{
    background: {PALETTE["bg_card"]} !important;
    border-right: 1px solid {PALETTE["border"]} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {PALETTE["text_primary"]} !important;
}}
h1, h2, h3, h4, h5, h6, p, span, li, label, div {{
    color: {PALETTE["text_primary"]} !important;
}}

/* ── Metric Cards ───────────────────────────────────────────── */
.metric-card {{
    background: linear-gradient(145deg, {PALETTE["bg_card"]}, #253047);
    border: 1px solid {PALETTE["border"]};
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {PALETTE["accent"]}, {PALETTE["accent_light"]});
    border-radius: 16px 16px 0 0;
}}
.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
}}
.metric-icon {{
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, {PALETTE["accent_light"]}, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}}
.metric-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {PALETTE["text_muted"]} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
}}

/* ── Section Cards ──────────────────────────────────────────── */
.section-card {{
    background: {PALETTE["bg_card"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}}

/* ── Alert Cards ────────────────────────────────────────────── */
.alert-ok {{
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px; padding: 1rem 1.25rem;
    color: {PALETTE["success"]} !important;
}}
.alert-warn {{
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.04));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px; padding: 1rem 1.25rem;
    color: {PALETTE["warning"]} !important;
}}
.alert-danger {{
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px; padding: 1rem 1.25rem;
    color: {PALETTE["danger"]} !important;
}}

/* ── Tabs ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {PALETTE["bg_card"]};
    padding: 6px;
    border-radius: 12px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 8px 20px;
    color: {PALETTE["text_muted"]} !important;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background: {PALETTE["accent"]} !important;
    color: white !important;
}}

/* ── Misc ───────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
    color: {PALETTE["accent_light"]} !important;
}}
.block-container {{
    padding-top: 2rem !important;
}}

/* ── Timeline bar ───────────────────────────────────────────── */
.tl-bar {{
    height: 28px;
    border-radius: 6px;
    display: inline-block;
    position: relative;
    transition: opacity 0.2s;
    cursor: pointer;
}}
.tl-bar:hover {{
    opacity: 0.8;
    outline: 2px solid {PALETTE["accent_light"]};
}}

/* ── Sleep stage indicators ─────────────────────────────────── */
.sleep-deep  {{ background: #6366f1; }}
.sleep-light {{ background: #818cf8; }}
.sleep-rem   {{ background: #c084fc; }}

/* ── Glow effect on headers ─────────────────────────────────── */
.glow-text {{
    text-shadow: 0 0 20px rgba(99,102,241,0.4), 0 0 60px rgba(99,102,241,0.1);
}}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: Matplotlib dark theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def dark_fig(figsize=(10, 4)):
    """Return a dark-themed matplotlib figure and axes."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE["bg_card"])
    ax.set_facecolor(PALETTE["bg_card"])
    ax.tick_params(colors=PALETTE["text_muted"])
    ax.spines["bottom"].set_color(PALETTE["border"])
    ax.spines["left"].set_color(PALETTE["border"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(PALETTE["text_muted"])
    ax.yaxis.label.set_color(PALETTE["text_muted"])
    ax.title.set_color(PALETTE["text_primary"])
    return fig, ax


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: Load model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_shar_model(dataset_name):
    """
    Load the SHAR classifier for live inference.
    Prefers the fully fine-tuned checkpoint; falls back to the pretrained
    encoder alone (classifier head then untrained → predictions unreliable).
    Returns (model, status) with status ∈ {"finetuned", "encoder_only", None}.
    """
    if not MODEL_AVAILABLE or dataset_name not in DATASETS:
        return None, None
    ds = DATASETS[dataset_name]
    try:
        encoder = SHAREncoder(in_channels=ds["in_channels"], seq_len=ds["seq_len"])
        model = SHAR_Classifier(encoder, num_classes=ds["num_classes"])
        finetuned_path = os.path.join(PARENT_DIR, "models", ds["classifier_ckpt"])
        encoder_path   = os.path.join(PARENT_DIR, "models", ds["encoder_ckpt"])
        if os.path.exists(finetuned_path):
            model.load_state_dict(torch.load(finetuned_path, map_location="cpu"))
            status = "finetuned"
        elif os.path.exists(encoder_path):
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
            status = "encoder_only"
        else:
            return None, None
        model.eval()
        return model, status
    except Exception:
        return None, None


@st.cache_resource
def load_test_dataset(dataset_name):
    """Cache the test split so it isn't re-read from disk on every rerun."""
    if dataset_name == "UCI HAR":
        return UCIHARDataset(DATASETS["UCI HAR"]["path"], split="test")
    if dataset_name == "WISDM":
        return WISDMDataset(DATASETS["WISDM"]["path"], split="test")
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size:3rem;">🏥</span>
        <h2 style="margin:0.25rem 0 0 0; font-weight:800;" class="glow-text">HealthTrack</h2>
        <p style="font-size:0.8rem; color:#94a3b8 !important; margin:0;">AI-Powered Health Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 1 is always visible. The dashboard section unlocks after balancing. ──
    step1_label = "🗂️ 1. Dataset & Balancing"
    if st.session_state.balanced:
        step1_label += "  ✅"
    dashboard_pages = [
        "🏠 Dashboard", "🔥 Calories & Steps", "😴 Sleep Analysis",
        "⚠️ Fall Detection", "💓 Heart & Stress", "📊 Activity Classifier",
    ]
    st.markdown("##### 🗂️ Step 1 — Setup")
    pick_setup = st.radio(
        " ",
        [step1_label],
        label_visibility="collapsed",
        key="nav_setup",
    )

    st.markdown("##### 📊 Step 2 — Dashboard")
    if st.session_state.balanced:
        pick_dash = st.radio(
            " ",
            ["— (stay on setup)"] + dashboard_pages,
            label_visibility="collapsed",
            key="nav_dash",
        )
    else:
        st.caption(f"🔒 Locked — pick a dataset and run iSMOTE first.")
        pick_dash = "— (stay on setup)"

    if pick_dash != "— (stay on setup)":
        page = pick_dash
    else:
        page = "🗂️ Dataset & Balancing"

    st.markdown("---")
    st.markdown("##### 👤 Your Profile")
    user_weight = st.number_input("Weight (kg)", 30.0, 200.0, DEFAULT_WEIGHT_KG, 0.5)
    user_height = st.number_input("Height (cm)", 100.0, 250.0, DEFAULT_HEIGHT_CM, 0.5)
    user_age    = st.number_input("Age", 10, 100, DEFAULT_AGE)
    user_gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("---")
    st.caption("Powered by SHAR Encoder  •  Self-Supervised Learning")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: DATASET & BALANCING  (Step 1 — mandatory before dashboard unlocks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🗂️ Dataset & Balancing":
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.4rem; font-weight:900; margin-bottom:0.25rem;">
        🗂️ Step 1 — Pick a Dataset & Balance It
    </h1>
    <p style="color:#94a3b8 !important; font-size:1.05rem; margin-bottom:1.75rem;">
        This project ships with two Human-Activity-Recognition datasets. Both are
        <b>imbalanced</b> — some activities have many more samples than others, which
        biases a classifier. We fix that with <b>iSMOTE</b> (Improved SMOTE), then
        unlock the health dashboards on top of the balanced dataset.
    </p>
    """, unsafe_allow_html=True)

    # ── 1.1 Dataset picker ────────────────────────────────────────────────────
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">1️⃣ Choose a dataset</h3></div>""",
                unsafe_allow_html=True)

    pick_cols = st.columns(2)
    for col, name in zip(pick_cols, DATASETS.keys()):
        with col:
            info = DATASETS[name]
            exists = os.path.isdir(info["path"])
            active = st.session_state.selected_dataset == name
            border = PALETTE["accent"] if active else PALETTE["border"]
            check = " ✅" if active else ""
            st.markdown(f"""
            <div class="metric-card" style="border:2px solid {border}; text-align:left; padding:1.25rem 1.5rem;">
                <div style="font-size:1.4rem; font-weight:800; color:{PALETTE['accent_light']} !important;">
                    {name}{check}
                </div>
                <p style="color:{PALETTE['text_muted']} !important; font-size:0.85rem; margin:0.4rem 0 0.6rem 0;">
                    {info['description']}
                </p>
                <p style="color:{PALETTE['text_muted']} !important; font-size:0.75rem; margin:0;">
                    {'📁 Found on disk' if exists else '⚠️ Folder missing at ' + info['path']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {name}", key=f"pick_{name}",
                         disabled=not exists, use_container_width=True):
                # Switching datasets resets the balancing state
                if st.session_state.selected_dataset != name:
                    st.session_state.selected_dataset = name
                    st.session_state.loaded = False
                    st.session_state.balanced = False
                    st.session_state.X_orig = st.session_state.y_orig = None
                    st.session_state.X_bal  = st.session_state.y_bal  = None
                    st.session_state.orig_counts = st.session_state.bal_counts = None
                st.rerun()

    if st.session_state.selected_dataset is None:
        st.info("👆 Pick a dataset to continue.")
        st.stop()

    ds_name = st.session_state.selected_dataset
    ds_info = DATASETS[ds_name]
    label_map = ds_info["labels"]

    # ── 1.2 Load + inspect original distribution ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="section-card"><h3 style="margin-top:0;">2️⃣ Original class distribution — <span style="color:{PALETTE['accent_light']}">{ds_name}</span></h3></div>""",
                unsafe_allow_html=True)

    try:
        with st.spinner(f"Loading {ds_name} training data…"):
            X_orig, y_orig = load_training_data(ds_name)
            st.session_state.X_orig = X_orig
            st.session_state.y_orig = y_orig
            unique, counts = np.unique(y_orig, return_counts=True)
            st.session_state.orig_counts = dict(zip(unique.tolist(), counts.tolist()))
            st.session_state.loaded = True
    except Exception as e:
        st.error(f"Could not load {ds_name}: {e}")
        st.stop()

    orig_counts = st.session_state.orig_counts
    max_c, min_c = max(orig_counts.values()), min(orig_counts.values())
    imbalance = round(max_c / max(1, min_c), 2)

    stat_cols = st.columns(4)
    for c, (icon, val, lbl) in zip(stat_cols, [
        ("📊", f"{len(y_orig):,}",     "Total windows"),
        ("🎯", f"{len(unique)}",        "Classes"),
        ("⚖️", f"{imbalance}×",         "Imbalance ratio"),
        ("📡", f"{ds_info['in_channels']}", "Sensor channels"),
    ]):
        with c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="font-size:1.6rem;">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # Original distribution bar chart
    fig, ax = dark_fig(figsize=(12, 4))
    names = [label_map.get(int(c), f"Class {c}") for c in unique]
    ax.bar(names, counts, color=PALETTE["danger"], alpha=0.85,
           edgecolor=PALETTE["bg_dark"])
    ax.set_ylabel("Windows", fontsize=10)
    ax.set_title("Before iSMOTE — imbalanced", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── 1.3 Run iSMOTE ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">3️⃣ Run iSMOTE to balance the dataset</h3></div>""",
                unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color:{PALETTE['text_muted']} !important; font-size:0.9rem; margin:-0.5rem 0 0.75rem 0;">
        iSMOTE generates synthetic samples for minority classes by interpolating between
        each minority sample and one of its k-nearest neighbours in the same class —
        then validates each new sample by checking that its k nearest neighbours in the
        <i>entire</i> dataset are also of that class. Overlapping / noisy candidates are rejected.
        Result: every class ends up with the same number of samples as the largest class.
    </p>
    """, unsafe_allow_html=True)

    run_col, note_col = st.columns([1, 3])
    with run_col:
        run_now = st.button("🚀 Run iSMOTE", type="primary", use_container_width=True,
                            disabled=st.session_state.balanced)
    with note_col:
        if st.session_state.balanced:
            st.success(f"✅ Balancing already finished for **{ds_name}** — see charts below.")

    if run_now and not st.session_state.balanced:
        with st.spinner(f"Running iSMOTE on {ds_name} — this may take 30–60 seconds…"):
            X_s, y_s, X_b, y_b = run_ismote_cached(ds_name)
        # X_sub / y_sub is the (possibly subsampled) working set iSMOTE saw.
        # X_orig / y_orig stays as the FULL training set for step 2's display.
        st.session_state.X_sub, st.session_state.y_sub = X_s, y_s
        st.session_state.X_bal, st.session_state.y_bal = X_b, y_b
        u_s, c_s = np.unique(y_s, return_counts=True)
        u_b, c_b = np.unique(y_b, return_counts=True)
        st.session_state.sub_counts = dict(zip(u_s.tolist(), c_s.tolist()))
        st.session_state.bal_counts = dict(zip(u_b.tolist(), c_b.tolist()))
        st.session_state.balanced = True
        st.rerun()

    # ── 1.4 Results — after balancing ────────────────────────────────────────
    if st.session_state.balanced:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="alert-ok" style="font-size:1.05rem;">
            🎉 <b>Balancing finished for the {ds_name} dataset.</b>
            Every class now has the same number of samples. The <b>Step 2 — Dashboard</b>
            menu on the left is unlocked and works on this balanced data.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-card"><h3 style="margin-top:0;">4️⃣ Before vs. after — what iSMOTE did</h3></div>""",
                    unsafe_allow_html=True)

        bal_counts = st.session_state.bal_counts
        sub_counts = st.session_state.sub_counts
        full_orig  = st.session_state.orig_counts
        all_classes = sorted(set(sub_counts) | set(bal_counts))
        names = [label_map.get(int(c), f"Class {c}") for c in all_classes]
        o_vals = [sub_counts.get(c, 0) for c in all_classes]
        b_vals = [bal_counts.get(c, 0) for c in all_classes]
        synth = [b - o for o, b in zip(o_vals, b_vals)]

        if sum(full_orig.values()) != sum(o_vals):
            st.caption(
                f"ℹ️ For a responsive demo, iSMOTE ran on a random "
                f"{sum(o_vals):,}-window subsample of the full {sum(full_orig.values()):,}-window "
                f"training set — the class imbalance and rejection math are identical. "
                f"Full training runs (`python main.py`) use the entire dataset."
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(PALETTE["bg_card"])
        for ax in (ax1, ax2):
            ax.set_facecolor(PALETTE["bg_card"])
            ax.tick_params(colors=PALETTE["text_muted"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("bottom", "left"):
                ax.spines[spine].set_color(PALETTE["border"])
            ax.title.set_color(PALETTE["text_primary"])
            ax.xaxis.label.set_color(PALETTE["text_muted"])
            ax.yaxis.label.set_color(PALETTE["text_muted"])

        ax1.bar(names, o_vals, color=PALETTE["danger"], alpha=0.85,
                edgecolor=PALETTE["bg_dark"])
        ax1.set_title("Before iSMOTE (imbalanced)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Windows")
        ax1.tick_params(axis="x", rotation=45)
        for lbl in ax1.get_xticklabels():
            lbl.set_ha("right")

        ax2.bar(names, o_vals, color=PALETTE["accent"], alpha=0.7,
                edgecolor=PALETTE["bg_dark"], label="Original")
        ax2.bar(names, synth, bottom=o_vals, color=PALETTE["success"], alpha=0.85,
                edgecolor=PALETTE["bg_dark"], label="iSMOTE-generated")
        ax2.set_title("After iSMOTE (balanced)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Windows")
        ax2.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
                   labelcolor=PALETTE["text_muted"], fontsize=9)
        ax2.tick_params(axis="x", rotation=45)
        for lbl in ax2.get_xticklabels():
            lbl.set_ha("right")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📦</div>
                <div class="metric-value" style="font-size:1.6rem;">{sum(o_vals):,}</div>
                <div class="metric-label">Original windows</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🧬</div>
                <div class="metric-value" style="font-size:1.6rem;">{sum(synth):,}</div>
                <div class="metric-label">Synthetic added</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⚖️</div>
                <div class="metric-value" style="font-size:1.6rem;">1.0×</div>
                <div class="metric-label">New imbalance ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-value" style="font-size:1.6rem;">{sum(b_vals):,}</div>
                <div class="metric-label">Total after iSMOTE</div>
            </div>
            """, unsafe_allow_html=True)

        # Per-class table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-card"><h3 style="margin-top:0;">📋 Per-class breakdown</h3></div>""",
                    unsafe_allow_html=True)
        import pandas as pd
        table = pd.DataFrame({
            "Activity": names,
            "Original": o_vals,
            "Synthetic added": synth,
            "After iSMOTE": b_vals,
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        # ── 1.5 Peek at a real signal from the selected dataset ──────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-card"><h3 style="margin-top:0;">🔎 Peek at a real signal window</h3></div>""",
                    unsafe_allow_html=True)

        pick_class = st.selectbox(
            "Show a real training window for:",
            [label_map.get(int(c), f"Class {c}") for c in all_classes],
            key="peek_class",
        )
        cls_id = next(c for c in all_classes if label_map.get(int(c), f"Class {c}") == pick_class)
        cls_idx = np.where(st.session_state.y_orig == cls_id)[0]
        if len(cls_idx) > 0:
            sample = st.session_state.X_orig[cls_idx[np.random.randint(len(cls_idx))]]
            n_ch = sample.shape[0]
            fig, ax = dark_fig(figsize=(12, 3.5))
            palette = plt.cm.viridis(np.linspace(0.15, 0.85, n_ch))
            for ch in range(n_ch):
                ax.plot(sample[ch], color=palette[ch],
                        label=ds_info["channel_names"][ch] if ch < len(ds_info["channel_names"]) else f"ch{ch}",
                        linewidth=1.1, alpha=0.9)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Sensor value")
            ax.set_title(f"'{pick_class}' — one raw window ({n_ch} channels × {sample.shape[1]} timesteps)",
                         fontsize=11, fontweight="bold")
            ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
                      labelcolor=PALETTE["text_muted"], fontsize=8, ncol=3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Don't fall through to any other page
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Dashboard":
    ds_name = st.session_state.selected_dataset
    ds_info = DATASETS[ds_name]
    n_orig  = sum(st.session_state.orig_counts.values())
    st.markdown(f"""
    <h1 class="glow-text" style="font-size:2.5rem; font-weight:900; margin-bottom:0.25rem;">
        🏥 Health Tracking Dashboard
    </h1>
    <p style="color:#94a3b8 !important; font-size:1.05rem; margin-bottom:0.6rem;">
        Real-time health insights powered by AI activity recognition from smartphone sensors
    </p>
    <div class="alert-ok" style="margin-bottom:1.5rem;">
        📦 Working on dataset: <b>{ds_name}</b> ·
        {n_orig:,} windows · {ds_info['num_classes']} classes · balanced ✅
    </div>
    """, unsafe_allow_html=True)

    # ── Generate daily data — every metric derives from the SAME simulated day ──
    timeline = generate_daily_timeline(
        allowed_activities=activities_for(st.session_state.selected_dataset)
    )
    total_cal = compute_daily_calories(timeline, user_weight)   # MET-based, includes resting burn
    active_cal = active_calories(timeline, user_weight)         # burn above 1-MET baseline
    sed = sedentary_ratio(timeline)
    total_steps = estimate_steps_from_timeline(timeline)
    night = simulate_sleep_night()
    sleep_hours = night["hours"]
    sleep_score = night["score"]

    # Simulate the full-day heart-rate series once; card and chart share it
    hr_times, hr_vals = [], []
    for seg in timeline:
        n_samples = max(1, int(seg["duration_min"] / 2))
        hr_seg = simulate_heart_rate(seg["activity"], n_samples, ramp=False)
        start_minutes = seg["start_hour"] * 60
        hr_times.extend((start_minutes + i * 2) / 60 for i in range(n_samples))
        hr_vals.extend(hr_seg.tolist())
    avg_hr = int(np.mean(hr_vals))

    # ── Top Metric Cards ──
    cols = st.columns(6)
    metrics = [
        ("🔥", f"{int(total_cal)}", "Total kcal"),
        ("⚡", f"{int(active_cal)}", "Active kcal"),
        ("👟", f"{total_steps:,}", "Steps"),
        ("💓", f"{avg_hr}", "Avg HR (bpm)"),
        ("😴", f"{sleep_hours}h", "Sleep"),
        ("🏆", f"{sleep_score}", "Sleep Score"),
    ]
    for col, (icon, val, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 24-Hour Activity Timeline ──
    st.markdown("""
    <div class="section-card">
        <h3 style="margin-top:0;">📅 24-Hour Activity Timeline</h3>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = dark_fig(figsize=(14, 2.5))
    for seg in timeline:
        start_h = seg["start_hour"]
        dur_h = seg["duration_min"] / 60
        cat = ACTIVITY_CATEGORIES.get(seg["activity"], "daily_living")
        color = CATEGORY_COLORS[cat]
        ax.barh(0, dur_h, left=start_h, height=0.6, color=color, edgecolor=PALETTE["bg_dark"],
                linewidth=0.5, alpha=0.85)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 2)], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Time of Day", fontsize=10)
    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=k.replace("_", " ").title())
                    for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8,
              facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text_muted"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Heart Rate Trend & Calorie Burn Charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="section-card"><h3 style="margin-top:0;">💓 Heart Rate Trend</h3></div>
        """, unsafe_allow_html=True)
        fig, ax = dark_fig()
        ax.fill_between(hr_times, hr_vals, alpha=0.3, color=PALETTE["danger"])
        ax.plot(hr_times, hr_vals, color=PALETTE["danger"], linewidth=1.2, alpha=0.9)
        ax.axhline(y=100, color=PALETTE["warning"], linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlim(0, 24)
        ax.set_xlabel("Hour")
        ax.set_ylabel("BPM")
        ax.set_title("Simulated Daily Heart Rate", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("""
        <div class="section-card"><h3 style="margin-top:0;">🔥 Hourly Calorie Burn</h3></div>
        """, unsafe_allow_html=True)
        fig, ax = dark_fig()
        # Allocate each segment's calories to hours proportionally to the
        # minutes of actual overlap with each hour bin
        hourly_cals = np.zeros(24)
        for seg in timeline:
            cal_per_min = calories_burned(seg["activity"], 1, user_weight)
            seg_start = seg["start_hour"] * 60
            seg_end = seg_start + seg["duration_min"]
            for h in range(24):
                overlap = min(seg_end, (h + 1) * 60) - max(seg_start, h * 60)
                if overlap > 0:
                    hourly_cals[h] += cal_per_min * overlap
        colors = [PALETTE["accent"] if c > np.mean(hourly_cals) else PALETTE["accent_light"] for c in hourly_cals]
        ax.bar(range(24), hourly_cals, color=colors, alpha=0.85, width=0.7, edgecolor=PALETTE["bg_dark"])
        ax.set_xlabel("Hour")
        ax.set_ylabel("kcal")
        ax.set_title("Estimated Calorie Burn by Hour", fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Health Alerts ──
    st.markdown("<br>", unsafe_allow_html=True)
    alert_cols = st.columns(3)
    with alert_cols[0]:
        if sed["sedentary_min"] > 600:
            st.markdown(f"""
            <div class="alert-danger">
                ⚠️ <b>Sedentary Alert</b><br>
                <b>{sed['sedentary_min']/60:.1f} h</b> of your awake time was sedentary
                (sleep excluded). Try to stand up and walk every 60 minutes!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-ok">
                ✅ <b>Activity Level: Good</b><br>
                <b>{int(sed['ratio']*100)}%</b> of awake time spent in
                moderate-to-vigorous activity ({int(sed['active_min'])} min).
            </div>
            """, unsafe_allow_html=True)
    with alert_cols[1]:
        if total_steps >= 10000:
            st.markdown(f"""
            <div class="alert-ok">
                🏆 <b>Step Goal Achieved!</b><br>
                You hit <b>{total_steps:,}</b> steps. Great job!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-warn">
                👟 <b>Steps: {total_steps:,} / 10,000</b><br>
                Keep moving to reach your daily goal!
            </div>
            """, unsafe_allow_html=True)
    with alert_cols[2]:
        st.markdown(f"""
        <div class="alert-ok">
            💓 <b>Heart Rate: Normal</b><br>
            24-hour average HR: <b>{avg_hr} bpm</b> — healthy range.
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: CALORIES & STEPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔥 Calories & Steps":
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.2rem; font-weight:900;">🔥 Calories & Step Tracker</h1>
    <p style="color:#94a3b8 !important; margin-bottom:1.5rem;">
        Energy expenditure estimation using MET (Metabolic Equivalent of Task) values
    </p>
    """, unsafe_allow_html=True)

    bmr = compute_bmr(user_weight, user_height, user_age, user_gender)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{int(bmr)}</div>
            <div class="metric-label">BMR (kcal/day)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        tdee = bmr * 1.55  # moderate activity multiplier
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔥</div>
            <div class="metric-value">{int(tdee)}</div>
            <div class="metric-label">Est. TDEE (kcal)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏋️</div>
            <div class="metric-value">{user_weight} kg</div>
            <div class="metric-label">Body Weight</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MET Activity Calculator ──
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">🧮 Activity Calorie Calculator</h3></div>""",
                unsafe_allow_html=True)

    ds_activities = activities_for(st.session_state.selected_dataset)
    calc_cols = st.columns([2, 1, 1])
    with calc_cols[0]:
        selected_activity = st.selectbox("Select Activity", ds_activities)
    with calc_cols[1]:
        duration = st.number_input("Duration (min)", 1, 480, 30)
    with calc_cols[2]:
        met_val = MET_VALUES[selected_activity]
        cals = calories_burned(selected_activity, duration, user_weight)
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1.7rem;">
            <div class="metric-value" style="font-size:1.6rem;">{int(cals)} kcal</div>
            <div class="metric-label">MET {met_val}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── MET Comparison Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">📊 MET Value Comparison (30 min)</h3></div>""",
                unsafe_allow_html=True)

    fig, ax = dark_fig(figsize=(12, 5))
    activities = ds_activities
    mets = [MET_VALUES.get(a, 1.3) for a in activities]
    cals_30 = [calories_burned(a, 30, user_weight) for a in activities]
    bar_colors = [CATEGORY_COLORS.get(ACTIVITY_CATEGORIES.get(a, "daily_living"), PALETTE["accent"]) for a in activities]
    bars = ax.barh(range(len(activities)), cals_30, color=bar_colors, alpha=0.85, height=0.65,
                   edgecolor=PALETTE["bg_dark"])
    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels(activities, fontsize=9, color=PALETTE["text_muted"])
    ax.set_xlabel("Calories burned in 30 minutes", fontsize=10)
    ax.set_title("Calorie Burn by Activity (30 min)", fontsize=12, fontweight="bold")
    # Add calorie labels
    for bar, cal in zip(bars, cals_30):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{int(cal)} kcal', va='center', fontsize=8, color=PALETTE["text_muted"])
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Step Counter Demo ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">👟 Step Counter (Simulated)</h3></div>""",
                unsafe_allow_html=True)

    # Generate a simulated walking accelerometer signal
    np.random.seed(42)
    t = np.linspace(0, 5, 500)
    # Walking produces ~2 Hz periodic signal
    acc_walk = 1.0 + 0.4 * np.sin(2 * np.pi * 2.0 * t) + 0.1 * np.random.randn(500)
    step_info = count_steps(acc_walk, sampling_rate=100, min_peak_height=0.12, min_peak_distance=35)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👟</div>
            <div class="metric-value">{step_info['steps']}</div>
            <div class="metric-label">Steps Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⏱️</div>
            <div class="metric-value">{step_info['cadence_spm']}</div>
            <div class="metric-label">Cadence (steps/min)</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        stride_len = 0.75  # metres
        distance = step_info["steps"] * stride_len
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📏</div>
            <div class="metric-value">{distance:.0f} m</div>
            <div class="metric-label">Est. Distance</div>
        </div>
        """, unsafe_allow_html=True)

    fig, ax = dark_fig(figsize=(12, 3.5))
    ax.plot(t, acc_walk, color=PALETTE["accent_light"], linewidth=1, alpha=0.8)
    ax.fill_between(t, acc_walk, alpha=0.15, color=PALETTE["accent"])
    # Mark exactly the peaks the step counter detected (no re-computation)
    peaks = step_info["peak_indices"]
    ax.scatter(t[peaks], acc_walk[peaks], color=PALETTE["danger"], zorder=5, s=40, label="Detected Steps")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acc. Magnitude (g)")
    ax.set_title("Accelerometer Signal — Step Detection", fontsize=12, fontweight="bold")
    ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text_muted"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: SLEEP ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "😴 Sleep Analysis":
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.2rem; font-weight:900;">😴 Sleep Quality Analysis</h1>
    <p style="color:#94a3b8 !important; margin-bottom:1.5rem;">
        Sleep stage estimation from accelerometer micro-movement patterns during rest
    </p>
    """, unsafe_allow_html=True)

    # ── Simulate a night of sleep data ──
    # The stages shown are genuinely produced by classify_sleep_stage() on the
    # simulated movement signal — the same night the Dashboard summarises.
    night = simulate_sleep_night()
    sleep_stds = night["stds"]
    stages = night["stages"]
    score = night["score"]
    n_windows = len(stages)

    # ── Top Cards ──
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏆</div>
            <div class="metric-value">{score}</div>
            <div class="metric-label">Sleep Score</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        deep_pct = int(stages.count("Deep") / len(stages) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🌊</div>
            <div class="metric-value">{deep_pct}%</div>
            <div class="metric-label">Deep Sleep</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        light_pct = int(stages.count("Light") / len(stages) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🌙</div>
            <div class="metric-value">{light_pct}%</div>
            <div class="metric-label">Light Sleep</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        rem_pct = int(stages.count("REM") / len(stages) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💤</div>
            <div class="metric-value">{rem_pct}%</div>
            <div class="metric-label">REM Sleep</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hypnogram ──
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">🛏️ Sleep Hypnogram</h3></div>""",
                unsafe_allow_html=True)

    fig, ax = dark_fig(figsize=(14, 3.5))
    stage_map = {"Deep": 0, "Light": 1, "REM": 2}
    stage_colors = {"Deep": "#6366f1", "Light": "#818cf8", "REM": "#c084fc"}
    times = np.arange(n_windows) * 5 / 60  # hours
    stage_vals = [stage_map[s] for s in stages]

    for i in range(len(stages) - 1):
        ax.fill_between([times[i], times[i+1]], [stage_vals[i], stage_vals[i]],
                        -0.5, color=stage_colors[stages[i]], alpha=0.4)
    ax.step(times, stage_vals, color=PALETTE["accent_light"], linewidth=2, where="post")
    # Standard hypnogram orientation: REM at the top, Deep at the bottom
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Deep", "Light", "REM"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Hours of Sleep", fontsize=10)
    ax.set_title("Sleep Stage Transitions", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, 2.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Movement Intensity ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-card"><h3 style="margin-top:0;">📈 Movement Intensity During Sleep</h3></div>""",
                unsafe_allow_html=True)

    fig, ax = dark_fig(figsize=(14, 3))
    bar_cols = [stage_colors[s] for s in stages]
    ax.bar(times, sleep_stds, width=5/60*0.85, color=bar_cols, alpha=0.8, edgecolor=PALETTE["bg_dark"])
    ax.axhline(y=0.02, color=PALETTE["success"], linestyle="--", linewidth=0.8, alpha=0.6, label="Deep threshold")
    ax.axhline(y=0.08, color=PALETTE["warning"], linestyle="--", linewidth=0.8, alpha=0.6, label="Light/REM threshold")
    ax.set_xlabel("Hours of Sleep", fontsize=10)
    ax.set_ylabel("Acc. Std-Dev", fontsize=10)
    ax.set_title("Accelerometer Micro-Movement (Std-Dev per Window)", fontsize=12, fontweight="bold")
    ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text_muted"], fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Sleep Recommendations ──
    st.markdown("<br>", unsafe_allow_html=True)
    if score >= 80:
        st.markdown("""
        <div class="alert-ok">
            ✅ <b>Excellent Sleep Quality!</b><br>
            Your sleep stage distribution is well-balanced. Deep sleep was sufficient for physical recovery.
        </div>
        """, unsafe_allow_html=True)
    elif score >= 60:
        st.markdown("""
        <div class="alert-warn">
            ⚠️ <b>Moderate Sleep Quality</b><br>
            Consider reducing screen time before bed and maintaining a consistent sleep schedule.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-danger">
            🚨 <b>Poor Sleep Quality</b><br>
            Your deep sleep percentage is low. Avoid caffeine after 2 PM, and try relaxation techniques.
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: FALL DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⚠️ Fall Detection":
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.2rem; font-weight:900;">⚠️ Fall Detection System</h1>
    <p style="color:#94a3b8 !important; margin-bottom:1.5rem;">
        Threshold-based fall detection using accelerometer impact and free-fall patterns
    </p>
    """, unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <div class="section-card">
        <h3 style="margin-top:0;">🔬 Detection Algorithm</h3>
        <p style="color:#94a3b8 !important;">A fall is characterised by three distinct phases in the accelerometer signal:</p>
        <ol style="color:#94a3b8 !important;">
            <li><b style="color:#f59e0b !important;">Free-Fall Phase</b> — Sudden drop in acceleration below 0.4g (near-weightlessness)</li>
            <li><b style="color:#ef4444 !important;">Impact Phase</b> — Sharp spike exceeding 3.0g as the body hits the ground</li>
            <li><b style="color:#6366f1 !important;">Post-Fall Immobility</b> — Prolonged laying state with minimal movement</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Simulator ──
    tab_fall, tab_normal = st.tabs(["🚨 Simulated Fall Event", "✅ Normal Walking Signal"])

    with tab_fall:
        np.random.seed(77)
        t = np.linspace(0, 4, 400)
        # Build a fall signal: normal → free-fall → impact → laying
        signal = np.ones(400) * 1.0  # baseline 1g
        signal[:100] += 0.15 * np.sin(2 * np.pi * 2 * t[:100]) + 0.05 * np.random.randn(100)  # walking
        signal[100:130] = np.linspace(1.0, 0.2, 30)  # free-fall
        signal[130] = 4.5  # impact spike
        signal[131] = 3.8
        signal[132] = 2.5
        signal[133:160] = np.linspace(1.8, 1.0, 27)  # settling
        signal[160:] = 0.98 + 0.01 * np.random.randn(240)  # laying still

        result = detect_fall(signal, sampling_rate=100)  # 400 samples over 4 s

        if result["fall_detected"]:
            immobility_txt = (
                "Post-impact immobility confirmed — high-priority alert."
                if result["immobile_after"]
                else "Movement resumed after impact."
            )
            st.markdown(f"""
            <div class="alert-danger" style="font-size:1.1rem;">
                🚨 <b>FALL DETECTED!</b><br>
                Impact of <b>{result['impact_g']}g</b> detected at sample index {result['impact_index']}.
                {immobility_txt}
                Emergency alert would be dispatched to saved contacts.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = dark_fig(figsize=(14, 5))
        ax.plot(t, signal, color=PALETTE["accent_light"], linewidth=1.5)
        ax.fill_between(t, signal, alpha=0.15, color=PALETTE["accent"])

        # Annotate phases
        ax.axvspan(t[0], t[100], alpha=0.08, color=PALETTE["success"], label="Normal Walking")
        ax.axvspan(t[100], t[130], alpha=0.12, color=PALETTE["warning"], label="Free-Fall")
        ax.axvspan(t[130], t[160], alpha=0.12, color=PALETTE["danger"], label="Impact")
        ax.axvspan(t[160], t[-1], alpha=0.08, color=PALETTE["accent"], label="Post-Fall Immobility")

        # Threshold lines
        ax.axhline(y=3.0, color=PALETTE["danger"], linestyle="--", linewidth=1, alpha=0.7)
        ax.text(t[-1] + 0.05, 3.0, "Impact\nThreshold", fontsize=8, color=PALETTE["danger"], va="center")
        ax.axhline(y=0.4, color=PALETTE["warning"], linestyle="--", linewidth=1, alpha=0.7)
        ax.text(t[-1] + 0.05, 0.4, "Free-Fall\nThreshold", fontsize=8, color=PALETTE["warning"], va="center")

        if result["fall_detected"]:
            idx = result["impact_index"]
            ax.scatter([t[idx]], [signal[idx]], color=PALETTE["danger"], s=120, zorder=5,
                       edgecolors="white", linewidths=1.5)
            ax.annotate(f"  Impact: {result['impact_g']}g", (t[idx], signal[idx]),
                        fontsize=10, color=PALETTE["danger"], fontweight="bold")

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Acceleration (g)", fontsize=10)
        ax.set_title("Accelerometer Signal — Fall Event Simulation", fontsize=13, fontweight="bold")
        ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text_muted"], fontsize=9, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_normal:
        np.random.seed(55)
        t_n = np.linspace(0, 4, 400)
        normal_signal = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * t_n) + 0.08 * np.random.randn(400)
        result_n = detect_fall(normal_signal, sampling_rate=100)

        if result_n["fall_detected"]:
            st.markdown(f"""
            <div class="alert-danger">
                🚨 <b>False Positive!</b><br>
                Detector flagged an impact of {result_n['impact_g']}g in a walking signal.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-ok">
                ✅ <b>No Fall Detected</b><br>
                Normal walking pattern — all acceleration values within safe thresholds.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = dark_fig(figsize=(14, 5))
        ax.plot(t_n, normal_signal, color=PALETTE["success"], linewidth=1.5)
        ax.fill_between(t_n, normal_signal, alpha=0.15, color=PALETTE["success"])
        ax.axhline(y=3.0, color=PALETTE["danger"], linestyle="--", linewidth=1, alpha=0.5, label="Impact Threshold")
        ax.axhline(y=0.4, color=PALETTE["warning"], linestyle="--", linewidth=1, alpha=0.5, label="Free-Fall Threshold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Acceleration (g)", fontsize=10)
        ax.set_title("Accelerometer Signal — Normal Walking (No Fall)", fontsize=13, fontweight="bold")
        ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text_muted"], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: HEART & STRESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "💓 Heart & Stress":
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.2rem; font-weight:900;">💓 Heart Rate & Stress Monitor</h1>
    <p style="color:#94a3b8 !important; margin-bottom:1.5rem;">
        Simulated cardiovascular metrics and stress estimation via HRV analysis
    </p>
    """, unsafe_allow_html=True)

    # ── Activity selector ──
    selected_act = st.selectbox("Select an activity to simulate:",
                                 activities_for(st.session_state.selected_dataset))
    hr_data = simulate_heart_rate(selected_act, 200)
    hrv_data = simulate_hrv(selected_act)

    # ── Top Cards ──
    hc1, hc2, hc3, hc4 = st.columns(4)
    avg_hr = int(np.mean(hr_data))
    max_hr = int(np.max(hr_data))
    with hc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💓</div>
            <div class="metric-value">{avg_hr}</div>
            <div class="metric-label">Avg Heart Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with hc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{max_hr}</div>
            <div class="metric-label">Peak HR</div>
        </div>
        """, unsafe_allow_html=True)
    with hc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🧘</div>
            <div class="metric-value">{hrv_data['rmssd']}</div>
            <div class="metric-label">RMSSD (ms)</div>
        </div>
        """, unsafe_allow_html=True)
    with hc4:
        stress_color = {"Low": PALETTE["success"], "Medium": PALETTE["warning"], "High": PALETTE["danger"]}
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value" style="background:none; -webkit-text-fill-color:{stress_color[hrv_data['stress_level']]};">{hrv_data['stress_level']}</div>
            <div class="metric-label">Stress Level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── HR trace ──
    col_hr, col_hrv = st.columns(2)
    with col_hr:
        st.markdown("""<div class="section-card"><h3 style="margin-top:0;">💓 Real-Time Heart Rate</h3></div>""",
                    unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(8, 4))
        t = np.arange(len(hr_data)) * 0.5  # seconds
        ax.plot(t, hr_data, color="#ef4444", linewidth=1.5)
        ax.fill_between(t, hr_data, alpha=0.15, color="#ef4444")
        ax.axhline(y=100, color=PALETTE["warning"], linestyle="--", linewidth=0.8, alpha=0.5, label="Elevated HR")
        ax.axhline(y=60, color=PALETTE["success"], linestyle="--", linewidth=0.8, alpha=0.5, label="Resting HR")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart Rate (bpm)")
        ax.set_title(f"HR during {selected_act}", fontsize=12, fontweight="bold")
        ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text_muted"], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_hrv:
        st.markdown("""<div class="section-card"><h3 style="margin-top:0;">🧠 HRV & Stress Analysis</h3></div>""",
                    unsafe_allow_html=True)

        # Simulated HRV for different states
        fig, ax = dark_fig(figsize=(8, 4))
        states = ["Resting", "Light\nActivity", "Moderate\nActivity", "Vigorous\nActivity"]
        rmssd_vals = [52, 35, 22, 14]
        sdnn_vals = [65, 42, 30, 18]
        x_pos = np.arange(len(states))
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, rmssd_vals, width, label="RMSSD (ms)", color=PALETTE["accent"], alpha=0.85)
        bars2 = ax.bar(x_pos + width/2, sdnn_vals, width, label="SDNN (ms)", color="#c084fc", alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states, fontsize=9)
        ax.set_ylabel("Milliseconds")
        ax.set_title("HRV Decreases with Activity Intensity", fontsize=12, fontweight="bold")
        ax.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text_muted"], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── HR Zone Explanation ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card">
        <h3 style="margin-top:0;">🎯 Heart Rate Zones</h3>
        <table style="width:100%; border-collapse:collapse; margin-top:1rem;">
            <tr style="border-bottom:1px solid #334155;">
                <th style="text-align:left; padding:10px; color:#94a3b8 !important;">Zone</th>
                <th style="text-align:left; padding:10px; color:#94a3b8 !important;">BPM Range</th>
                <th style="text-align:left; padding:10px; color:#94a3b8 !important;">Activities</th>
                <th style="text-align:left; padding:10px; color:#94a3b8 !important;">Benefit</th>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:10px;">🟢 Rest</td>
                <td style="padding:10px;">58–72 bpm</td>
                <td style="padding:10px;">Sitting, Laying, Typing</td>
                <td style="padding:10px;">Recovery & restoration</td>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:10px;">🔵 Light</td>
                <td style="padding:10px;">72–100 bpm</td>
                <td style="padding:10px;">Standing, Brushing Teeth</td>
                <td style="padding:10px;">Fat burning zone</td>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:10px;">🟡 Moderate</td>
                <td style="padding:10px;">100–140 bpm</td>
                <td style="padding:10px;">Walking, Kicking</td>
                <td style="padding:10px;">Cardio fitness</td>
            </tr>
            <tr>
                <td style="padding:10px;">🔴 Vigorous</td>
                <td style="padding:10px;">140–175 bpm</td>
                <td style="padding:10px;">Stairs, Jogging</td>
                <td style="padding:10px;">VO2 Max, endurance</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: ACTIVITY CLASSIFIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Activity Classifier":
    ds_name = st.session_state.selected_dataset
    ds_info = DATASETS[ds_name]
    label_map = ds_info["labels"]
    n_classes = ds_info["num_classes"]

    st.markdown(f"""
    <h1 class="glow-text" style="font-size:2.2rem; font-weight:900;">📊 Live Activity Classifier</h1>
    <p style="color:#94a3b8 !important; margin-bottom:1.5rem;">
        Real-time activity prediction using the SHAR encoder on <b>{ds_name}</b> test data
    </p>
    """, unsafe_allow_html=True)

    model, model_status = load_shar_model(ds_name)
    data_dir = ds_info["path"]

    if model is None or not os.path.isdir(data_dir):
        import_hint = ""
        if MODEL_IMPORT_ERROR is not None:
            import_hint = f"<br><small>PyTorch could not be loaded on this machine: <code>{MODEL_IMPORT_ERROR[:200]}</code></small>"
        st.markdown(f"""
        <div class="alert-warn">
            ⚠️ <b>Model or dataset not available</b><br>
            To enable live classification, ensure you have:<br>
            1. Run <code>python main.py</code> from the project root to train the model.<br>
            2. The UCI HAR Dataset folder exists at the project root.{import_hint}
        </div>
        """, unsafe_allow_html=True)

        # Show dataset reports if available
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-card"><h3 style="margin-top:0;">📋 Training Reports</h3></div>""",
                    unsafe_allow_html=True)

        tab_uci, tab_wisdm = st.tabs(["UCI HAR Dataset", "WISDM Dataset"])
        with tab_uci:
            c1, c2 = st.columns(2)
            with c1:
                path = os.path.join(PARENT_DIR, "results", "ismote_distribution.png")
                if os.path.exists(path):
                    st.image(path, caption="iSMOTE Class Balancing")
                else:
                    st.info("Run `python main.py` to generate this report.")
            with c2:
                path = os.path.join(PARENT_DIR, "results", "confusion_matrix.png")
                if os.path.exists(path):
                    st.image(path, caption="Confusion Matrix")
                else:
                    st.info("Run `python main.py` to generate this report.")

        with tab_wisdm:
            c1, c2 = st.columns(2)
            with c1:
                path = os.path.join(PARENT_DIR, "results", "ismote_distribution_wisdm.png")
                if os.path.exists(path):
                    st.image(path, caption="iSMOTE Class Balancing (WISDM)")
                else:
                    st.info("Run `python main_wisdm.py` to generate this report.")
            with c2:
                path = os.path.join(PARENT_DIR, "results", "confusion_matrix_wisdm.png")
                if os.path.exists(path):
                    st.image(path, caption="Confusion Matrix (WISDM)")
                else:
                    st.info("Run `python main_wisdm.py` to generate this report.")
    else:
        train_cmd = "python main.py" if ds_name == "UCI HAR" else "python main_wisdm.py"
        if model_status == "finetuned":
            st.markdown(f"""
            <div class="alert-ok">
                ✅ <b>Fine-tuned SHAR model loaded for {ds_name}</b> — ready for live classification
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-warn">
                ⚠️ <b>Only the pretrained encoder was found for {ds_name}</b> — the classifier
                head is untrained, so the predictions below are NOT reliable.
                Run <code>{train_cmd}</code> to fine-tune and save the classifier.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        dataset = load_test_dataset(ds_name)

        test_idx = st.slider("Select a test sample", 0, len(dataset) - 1, 42)
        x_in, y_true = dataset[test_idx]

        with torch.no_grad():
            logits = model(x_in.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))

        true_name = label_map.get(int(y_true.item()), "Unknown")
        pred_name = label_map.get(pred_idx, "Unknown")

        # ── Result cards ──
        rc1, rc2 = st.columns(2)
        with rc1:
            match = pred_name == true_name
            color = PALETTE["success"] if match else PALETTE["danger"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{"✅" if match else "❌"}</div>
                <div class="metric-value" style="background:none; -webkit-text-fill-color:{color}; font-size:1.4rem;">{pred_name}</div>
                <div class="metric-label">Predicted Activity</div>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value" style="font-size:1.4rem;">{true_name}</div>
                <div class="metric-label">Ground Truth</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sensor signals + probability bar chart ──
        sig_col, prob_col = st.columns(2)
        with sig_col:
            st.markdown("""<div class="section-card"><h3 style="margin-top:0;">📡 Sensor Signals</h3></div>""",
                        unsafe_allow_html=True)
            n_ch = x_in.shape[0]
            # UCI HAR: 3 rows × 3 channels (body-acc / body-gyro / total-acc).
            # WISDM:   1 row × 3 channels (accelerometer).
            if n_ch == 9:
                fig, axes = plt.subplots(3, 1, figsize=(8, 7))
                titles = ["Body Accelerometer (X, Y, Z)", "Body Gyroscope (X, Y, Z)", "Total Accelerometer (X, Y, Z)"]
                axes_iter = list(enumerate(axes))
                per_row = 3
            else:
                fig, ax_single = plt.subplots(1, 1, figsize=(8, 4))
                axes_iter = [(0, ax_single)]
                titles = ["Phone Accelerometer (X, Y, Z)"]
                per_row = n_ch
            fig.patch.set_facecolor(PALETTE["bg_card"])
            ch_colors = ["#22c55e", "#3b82f6", "#f59e0b"]
            for i, ax in axes_iter:
                ax.set_facecolor(PALETTE["bg_card"])
                for j in range(per_row):
                    ax.plot(x_in[i * per_row + j].numpy(), color=ch_colors[j % len(ch_colors)],
                            linewidth=1, alpha=0.85)
                ax.set_title(titles[i], fontsize=10, color=PALETTE["text_primary"], fontweight="bold")
                ax.tick_params(colors=PALETTE["text_muted"])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color(PALETTE["border"])
                ax.spines["left"].set_color(PALETTE["border"])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with prob_col:
            st.markdown("""<div class="section-card"><h3 style="margin-top:0;">📊 Prediction Confidence</h3></div>""",
                        unsafe_allow_html=True)
            fig, ax = dark_fig(figsize=(8, max(4, n_classes * 0.35)))
            act_names = [label_map.get(i, f"Class {i}") for i in range(n_classes)]
            colors = [PALETTE["success"] if i == pred_idx else PALETTE["accent"] for i in range(n_classes)]
            ax.barh(range(n_classes), probs * 100, color=colors, alpha=0.85, height=0.6,
                    edgecolor=PALETTE["bg_dark"])
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels(act_names, fontsize=9)
            ax.set_xlabel("Confidence (%)", fontsize=10)
            ax.set_title("Model Prediction Probabilities", fontsize=12, fontweight="bold")
            ax.set_xlim(0, 115)
            for i, p in enumerate(probs):
                ax.text(p * 100 + 1, i, f"{p*100:.1f}%", va="center", fontsize=8,
                        color=PALETTE["text_muted"])
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Health insight for this activity ──
        st.markdown("<br>", unsafe_allow_html=True)
        met = MET_VALUES.get(pred_name, 1.3)
        cal_5min = calories_burned(pred_name, 5, user_weight)
        hr_sim = simulate_heart_rate(pred_name, 10)
        st.markdown(f"""
        <div class="section-card">
            <h3 style="margin-top:0;">🏥 Health Insight for "{pred_name}"</h3>
            <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:1rem;">
                <div style="flex:1; min-width:150px; text-align:center;">
                    <div style="font-size:1.6rem; font-weight:800; color:{PALETTE['accent_light']} !important;">{met}</div>
                    <div style="font-size:0.75rem; color:{PALETTE['text_muted']} !important; text-transform:uppercase;">MET Value</div>
                </div>
                <div style="flex:1; min-width:150px; text-align:center;">
                    <div style="font-size:1.6rem; font-weight:800; color:{PALETTE['accent_light']} !important;">{int(cal_5min)} kcal</div>
                    <div style="font-size:0.75rem; color:{PALETTE['text_muted']} !important; text-transform:uppercase;">Burn per 5 min</div>
                </div>
                <div style="flex:1; min-width:150px; text-align:center;">
                    <div style="font-size:1.6rem; font-weight:800; color:{PALETTE['accent_light']} !important;">{int(np.mean(hr_sim))} bpm</div>
                    <div style="font-size:0.75rem; color:{PALETTE['text_muted']} !important; text-transform:uppercase;">Expected HR</div>
                </div>
                <div style="flex:1; min-width:150px; text-align:center;">
                    <div style="font-size:1.6rem; font-weight:800; color:{PALETTE['accent_light']} !important;">{ACTIVITY_HR_ZONE.get(pred_name, 'rest').title()}</div>
                    <div style="font-size:0.75rem; color:{PALETTE['text_muted']} !important; text-transform:uppercase;">HR Zone</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"""
<div style="text-align:center; padding:2rem 0 1rem; border-top:1px solid {PALETTE['border']}; margin-top:3rem;">
    <p style="color:{PALETTE['text_muted']} !important; font-size:0.85rem;">
        🏥 Health Tracking System — Powered by SHAR Self-Supervised Learning Engine<br>
        <span style="font-size:0.75rem;">UCI HAR & WISDM Datasets • iSMOTE • Lambda Layers • Contrastive Learning</span>
    </p>
</div>
""", unsafe_allow_html=True)
