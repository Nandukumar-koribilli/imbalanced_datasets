"""
🏥 Health Tracking System — Streamlit Dashboard
=================================================
A premium, dark-themed health monitoring dashboard powered by
the SHAR (Self-Supervised HAR) activity recognition engine.

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys, os, time, random

# ── Ensure parent project is importable ──────────────────────────────────────
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    UCI_HAR_LABELS, WISDM_LABELS, MET_VALUES,
    ACTIVITY_CATEGORIES, CATEGORY_COLORS, PALETTE,
    DEFAULT_WEIGHT_KG, DEFAULT_HEIGHT_CM, DEFAULT_AGE, DEFAULT_GENDER,
    ACTIVITY_HR_ZONE, HR_PROFILES,
)
from health_metrics import (
    compute_bmr, calories_burned, compute_daily_calories,
    count_steps, sedentary_ratio, classify_sleep_stage, compute_sleep_score,
    detect_fall, simulate_heart_rate, simulate_hrv, generate_daily_timeline,
)

# ── Try to load the SHAR model from the parent project ──────────────────────
MODEL_AVAILABLE = False
try:
    from src.shar_model import SHAREncoder, SHAR_Classifier
    from src.dataset_utils import UCIHARDataset
    import torch
    MODEL_AVAILABLE = True
except ImportError:
    pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Health Tracking System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
def load_shar_model():
    if not MODEL_AVAILABLE:
        return None
    try:
        encoder = SHAREncoder(in_channels=9, seq_len=128)
        model = SHAR_Classifier(encoder, num_classes=6)
        weights_path = os.path.join(PARENT_DIR, "models", "shar_encoder_pretrained.pth")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            encoder.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception:
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

    page = st.radio(
        "🧭 Navigation",
        ["🏠 Dashboard", "🔥 Calories & Steps", "😴 Sleep Analysis",
         "⚠️ Fall Detection", "💓 Heart & Stress", "📊 Activity Classifier"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### 👤 Your Profile")
    user_weight = st.number_input("Weight (kg)", 30.0, 200.0, DEFAULT_WEIGHT_KG, 0.5)
    user_height = st.number_input("Height (cm)", 100.0, 250.0, DEFAULT_HEIGHT_CM, 0.5)
    user_age    = st.number_input("Age", 10, 100, DEFAULT_AGE)
    user_gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("---")
    st.caption("Powered by SHAR Encoder  •  Self-Supervised Learning")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Dashboard":
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.5rem; font-weight:900; margin-bottom:0.25rem;">
        🏥 Health Tracking Dashboard
    </h1>
    <p style="color:#94a3b8 !important; font-size:1.05rem; margin-bottom:2rem;">
        Real-time health insights powered by AI activity recognition from smartphone sensors
    </p>
    """, unsafe_allow_html=True)

    # ── Generate daily data ──
    timeline = generate_daily_timeline()
    bmr = compute_bmr(user_weight, user_height, user_age, user_gender)
    daily_cal = compute_daily_calories(timeline, user_weight)
    sed = sedentary_ratio(timeline)

    # Simulate aggregated metrics
    total_steps = int(np.random.uniform(6500, 12000))
    avg_hr = int(np.random.uniform(65, 78))
    sleep_hours = 7.0
    sleep_score = int(np.random.uniform(72, 92))

    # ── Top Metric Cards ──
    cols = st.columns(6)
    metrics = [
        ("🔥", f"{int(daily_cal + bmr)}", "Total kcal"),
        ("👟", f"{total_steps:,}", "Steps"),
        ("💓", f"{avg_hr}", "Avg HR (bpm)"),
        ("😴", f"{sleep_hours}h", "Sleep"),
        ("🏆", f"{sleep_score}", "Sleep Score"),
        ("📊", f"{int(sed['ratio']*100)}%", "Active Ratio"),
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
        hrs_in_day = []
        hr_vals = []
        for seg in timeline:
            n_samples = max(1, int(seg["duration_min"] / 2))
            hr_seg = simulate_heart_rate(seg["activity"], n_samples)
            start_minutes = seg["start_hour"] * 60
            times = [start_minutes + i * 2 for i in range(n_samples)]
            hrs_in_day.extend([t / 60 for t in times])
            hr_vals.extend(hr_seg.tolist())
        ax.fill_between(hrs_in_day, hr_vals, alpha=0.3, color=PALETTE["danger"])
        ax.plot(hrs_in_day, hr_vals, color=PALETTE["danger"], linewidth=1.2, alpha=0.9)
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
        hourly_cals = np.zeros(24)
        for seg in timeline:
            start_h = int(seg["start_hour"])
            end_h = min(23, int(seg["start_hour"] + seg["duration_min"] / 60))
            cal = calories_burned(seg["activity"], seg["duration_min"], user_weight)
            span = max(1, end_h - start_h + 1)
            for h in range(start_h, min(24, end_h + 1)):
                hourly_cals[h] += cal / span
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
                You've been sedentary for <b>{int(sed['sedentary_min'])} min</b> today.
                Try to stand up and walk every 60 minutes!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-ok">
                ✅ <b>Activity Level: Good</b><br>
                Active/Sedentary ratio: <b>{int(sed['ratio']*100)}%</b>
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
            Average resting HR: <b>{avg_hr} bpm</b> — healthy range.
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

    calc_cols = st.columns([2, 1, 1])
    with calc_cols[0]:
        selected_activity = st.selectbox("Select Activity", list(MET_VALUES.keys()))
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
    activities = list(MET_VALUES.keys())
    mets = [MET_VALUES[a] for a in activities]
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
    # Mark peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(acc_walk - np.mean(acc_walk), height=0.12, distance=35)
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
    np.random.seed(101)
    n_windows = 84  # 7 hours ÷ 5 min windows
    sleep_stds = []
    stages = []
    # Simulate a realistic sleep cycle pattern
    cycle_pattern = (
        ["Deep"] * 6 + ["Light"] * 4 + ["REM"] * 3 +
        ["Light"] * 5 + ["Deep"] * 5 + ["Light"] * 4 + ["REM"] * 4 +
        ["Light"] * 6 + ["Deep"] * 4 + ["Light"] * 5 + ["REM"] * 5 +
        ["Light"] * 8 + ["Deep"] * 3 + ["Light"] * 6 + ["REM"] * 6 +
        ["Light"] * 10
    )
    for i in range(n_windows):
        stage = cycle_pattern[i % len(cycle_pattern)]
        if stage == "Deep":
            std = np.random.uniform(0.005, 0.02)
        elif stage == "Light":
            std = np.random.uniform(0.02, 0.08)
        else:  # REM
            std = np.random.uniform(0.08, 0.18)
        sleep_stds.append(std)
        stages.append(stage)

    score = compute_sleep_score(stages)

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
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Deep", "Light", "REM"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Hours of Sleep", fontsize=10)
    ax.set_title("Sleep Stage Transitions", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()
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

        result = detect_fall(signal)

        if result["fall_detected"]:
            st.markdown(f"""
            <div class="alert-danger" style="font-size:1.1rem;">
                🚨 <b>FALL DETECTED!</b><br>
                Impact of <b>{result['impact_g']}g</b> detected at sample index {result['impact_index']}.
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
        result_n = detect_fall(normal_signal)

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
    selected_act = st.selectbox("Select an activity to simulate:", list(ACTIVITY_HR_ZONE.keys()))
    hr_data = simulate_heart_rate(selected_act, 200)
    hrv_data = simulate_hrv(selected_act)

    # ── Top Cards ──
    hc1, hc2, hc3, hc4 = st.columns(4)
    avg_hr = int(np.mean(hr_data))
    max_hr = int(np.max(hr_data))
    min_hr = int(np.min(hr_data))
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
    st.markdown("""
    <h1 class="glow-text" style="font-size:2.2rem; font-weight:900;">📊 Live Activity Classifier</h1>
    <p style="color:#94a3b8 !important; margin-bottom:1.5rem;">
        Real-time activity prediction using the SHAR pre-trained encoder on UCI HAR test data
    </p>
    """, unsafe_allow_html=True)

    model = load_shar_model()
    data_dir = os.path.join(PARENT_DIR, "UCI HAR Dataset")

    if model is None or not os.path.isdir(data_dir):
        st.markdown("""
        <div class="alert-warn">
            ⚠️ <b>Model or dataset not available</b><br>
            To enable live classification, ensure you have:<br>
            1. Run <code>python main.py</code> from the project root to train the model.<br>
            2. The UCI HAR Dataset folder exists at the project root.
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
        st.markdown("""
        <div class="alert-ok">
            ✅ <b>SHAR Model Loaded Successfully</b> — Ready for live classification
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        dataset = UCIHARDataset(data_dir, split="test")

        test_idx = st.slider("Select a test sample", 0, len(dataset) - 1, 42)
        x_in, y_true = dataset[test_idx]

        with torch.no_grad():
            logits = model(x_in.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)

        true_name = UCI_HAR_LABELS.get(y_true.item(), "Unknown")
        pred_name = UCI_HAR_LABELS.get(pred_idx, "Unknown")

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
            fig, axes = plt.subplots(3, 1, figsize=(8, 7))
            fig.patch.set_facecolor(PALETTE["bg_card"])
            titles = ["Body Accelerometer (X, Y, Z)", "Body Gyroscope (X, Y, Z)", "Total Accelerometer (X, Y, Z)"]
            ch_colors = ["#22c55e", "#3b82f6", "#f59e0b"]
            for i, ax in enumerate(axes):
                ax.set_facecolor(PALETTE["bg_card"])
                for j, col in enumerate(ch_colors):
                    ax.plot(x_in[i*3 + j].numpy(), color=col, linewidth=1, alpha=0.85)
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
            fig, ax = dark_fig(figsize=(8, 7))
            act_names = [UCI_HAR_LABELS[i] for i in range(6)]
            colors = [PALETTE["success"] if i == pred_idx else PALETTE["accent"] for i in range(6)]
            ax.barh(range(6), probs * 100, color=colors, alpha=0.85, height=0.6,
                    edgecolor=PALETTE["bg_dark"])
            ax.set_yticks(range(6))
            ax.set_yticklabels(act_names, fontsize=10)
            ax.set_xlabel("Confidence (%)", fontsize=10)
            ax.set_title("Model Prediction Probabilities", fontsize=12, fontweight="bold")
            ax.set_xlim(0, 105)
            for i, p in enumerate(probs):
                ax.text(p * 100 + 1, i, f"{p*100:.1f}%", va="center", fontsize=9,
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
