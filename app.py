import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Import local modules
try:
    from src import shar_model
    from src import dataset_utils
    from src import ismote
    from src import random_masking
    from src import mae_model
    
    # Alias functions/classes for convenience
    SHAREncoder = shar_model.SHAREncoder
    SHAR_Classifier = shar_model.SHAR_Classifier
    UCIHARDataset = dataset_utils.UCIHARDataset
    ismote_func = ismote.ismote
    apply_random_masking_batch = random_masking.apply_random_masking_batch
    MAEEncoder = mae_model.MAEEncoder
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Make sure you are running streamlit from the project root directory.")
except Exception as e:
    st.error(f"Initialization Error: {e}")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SHAR: Self-Supervised HAR",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Reset and base styles */
    .main, .stApp {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    
    /* Global text forcing */
    h1, h2, h3, h4, h5, h6, p, span, li, label, div {
        color: #1e293b !important;
    }
    
    .card {
        background-color: white !important;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .stat-box {
        background: white !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
    }
    
    .stat-val {
        font-size: 2.25rem;
        font-weight: 800;
        color: #2563eb !important;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Fix for metrics and other streamlit components */
    [data-testid="stMetricValue"] {
        color: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏃 SHAR Project")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "Data Exploration", "Architecture", "Simulation & Demo"])
    st.markdown("---")
    st.info("**Device:** CPU (Forced)")
    
    if st.button("Reset Session"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# --- UTILITY FUNCTIONS ---
@st.cache_resource
def load_model(device='cpu'):
    try:
        encoder = SHAREncoder(in_channels=9, seq_len=128)
        model = SHAR_Classifier(encoder, num_classes=6)
        if os.path.exists("models/shar_encoder_pretrained.pth"):
            # Note: This loads just the encoder weights if it's the pretrain file
            # or the whole model if it was saved that way.
            # Usually users save the whole state dict or just encoder.
            # Based on README, shar_encoder_pretrained.pth is the encoder.
            state_dict = torch.load("models/shar_encoder_pretrained.pth", map_location=device)
            # Filter state dict if it's from pretrain (has 'encoder.' prefix or just weights)
            encoder.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        return None

def get_activity_names():
    return {
        0: "Walking",
        1: "Walking Upstairs",
        2: "Walking Downstairs",
        3: "Sitting",
        4: "Standing",
        5: "Laying"
    }

# --- PAGE: OVERVIEW ---
if page == "Overview":
    st.title("🚀 Self-Supervised Learning for Activity Recognition (SHAR)")
    st.markdown("""
    ### Solving the "Imbalanced Dataset" & "Expensive Labelling" Problem in HAR
    
    Human Activity Recognition (HAR) using mobile sensors traditionally requires thousands of manually labelled samples. 
    However, real-world data is often **unlabelled** and **highly imbalanced** (we spend more time sitting than walking upstairs).
    
    **SHAR** implements a cutting-edge pipeline to solve these issues:
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">iSMOTE</div>
            <div class="stat-label">Smart Balancing</div>
            <p style="font-size: 0.8rem; margin-top: 10px;">Accurately balances datasets by validating synthetic samples via KNN.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">SSL</div>
            <div class="stat-label">Self-Supervised Learning</div>
            <p style="font-size: 0.8rem; margin-top: 10px;">Learns from unlabelled data using Random Masking & Contrastive loss.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">Lambda</div>
            <div class="stat-label">Efficient Attention</div>
            <p style="font-size: 0.8rem; margin-top: 10px;">Uses Lambda Layers for high-performance temporal feature extraction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Visual Results & Reports")
    st.info("💡 **Instructions**: To generate or update these reports, you must run the training pipelines in your terminal:\n\n"
            "- For UCI HAR (Contrastive): `python main.py`\n"
            "- For WISDM (Contrastive): `python main_wisdm.py`\n"
            "- For UCI HAR (MAE): `python main_mae.py`\n"
            "- For WISDM (MAE): `python main_mae_wisdm.py`")
            
    tab_uci, tab_wisdm, tab_mae_uci, tab_mae_wisdm = st.tabs(
        ["UCI HAR (Contrastive)", "WISDM (Contrastive)",
         "UCI HAR (MAE)", "WISDM (MAE)"])
    
    with tab_uci:
        st.markdown("#### UCI HAR Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("**iSMOTE Class Distribution**")
            if os.path.exists("results/ismote_distribution.png"):
                st.image("results/ismote_distribution.png")
            else:
                st.warning("iSMOTE graph not found. Run `python main.py` to generate it.")
                
        with res_col2:
            st.markdown("**Model Performance (Confusion Matrix)**")
            if os.path.exists("results/confusion_matrix.png"):
                st.image("results/confusion_matrix.png")
            else:
                st.warning("Confusion matrix not found. Run `python main.py` to generate it.")

    with tab_wisdm:
        st.markdown("#### WISDM Dataset Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("**iSMOTE Class Distribution**")
            if os.path.exists("results/ismote_distribution_wisdm.png"):
                st.image("results/ismote_distribution_wisdm.png")
            else:
                st.warning("iSMOTE graph not found. Run `python main_wisdm.py` to generate it.")
                
        with res_col2:
            st.markdown("**Model Performance (Confusion Matrix)**")
            if os.path.exists("results/confusion_matrix_wisdm.png"):
                st.image("results/confusion_matrix_wisdm.png")
            else:
                st.warning("Confusion matrix not found. Run `python main_wisdm.py` to generate it.")

    with tab_mae_uci:
        st.markdown("#### UCI HAR — Masked Autoencoder (MAE) Results")
        st.markdown("MAE pre-trains using **reconstruction** of 75% masked patches, "
                    "learning richer temporal representations than contrastive learning.")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("**iSMOTE Class Distribution**")
            if os.path.exists("results/ismote_distribution_mae.png"):
                st.image("results/ismote_distribution_mae.png")
            else:
                st.warning("Run `python main_mae.py` to generate MAE results.")
        with res_col2:
            st.markdown("**MAE Confusion Matrix**")
            if os.path.exists("results/confusion_matrix_mae.png"):
                st.image("results/confusion_matrix_mae.png")
            else:
                st.warning("Run `python main_mae.py` to generate MAE results.")

        # Show metrics comparison if both exist
        if os.path.exists("results/metrics.json") and os.path.exists("results/metrics_mae.json"):
            import json
            with open("results/metrics.json") as f:
                m_contrastive = json.load(f)
            with open("results/metrics_mae.json") as f:
                m_mae = json.load(f)
            st.markdown("---")
            st.markdown("##### 📊 Contrastive vs MAE Comparison")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.metric("Contrastive Accuracy", f"{m_contrastive['test_accuracy']}%")
                st.metric("Contrastive Macro F1", f"{m_contrastive['macro_f1']}")
            with comp_col2:
                acc_delta = round(m_mae['test_accuracy'] - m_contrastive['test_accuracy'], 2)
                f1_delta = round(m_mae['macro_f1'] - m_contrastive['macro_f1'], 4)
                st.metric("MAE Accuracy", f"{m_mae['test_accuracy']}%", delta=f"{acc_delta}%")
                st.metric("MAE Macro F1", f"{m_mae['macro_f1']}", delta=f"{f1_delta}")

    with tab_mae_wisdm:
        st.markdown("#### WISDM — Masked Autoencoder (MAE) Results")
        st.markdown("MAE pre-trains using **reconstruction** of 75% masked patches, "
                    "learning richer temporal representations than contrastive learning.")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("**iSMOTE Class Distribution**")
            if os.path.exists("results/ismote_distribution_mae_wisdm.png"):
                st.image("results/ismote_distribution_mae_wisdm.png")
            else:
                st.warning("Run `python main_mae_wisdm.py` to generate MAE results.")
        with res_col2:
            st.markdown("**MAE Confusion Matrix**")
            if os.path.exists("results/confusion_matrix_mae_wisdm.png"):
                st.image("results/confusion_matrix_mae_wisdm.png")
            else:
                st.warning("Run `python main_mae_wisdm.py` to generate MAE results.")

        # Show metrics comparison if both exist
        if os.path.exists("results/metrics_wisdm.json") and os.path.exists("results/metrics_mae_wisdm.json"):
            import json
            with open("results/metrics_wisdm.json") as f:
                m_contrastive = json.load(f)
            with open("results/metrics_mae_wisdm.json") as f:
                m_mae = json.load(f)
            st.markdown("---")
            st.markdown("##### 📊 Contrastive vs MAE Comparison")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.metric("Contrastive Accuracy", f"{m_contrastive['test_accuracy']}%")
                st.metric("Contrastive Macro F1", f"{m_contrastive['macro_f1']}")
            with comp_col2:
                acc_delta = round(m_mae['test_accuracy'] - m_contrastive['test_accuracy'], 2)
                f1_delta = round(m_mae['macro_f1'] - m_contrastive['macro_f1'], 4)
                st.metric("MAE Accuracy", f"{m_mae['test_accuracy']}%", delta=f"{acc_delta}%")
                st.metric("MAE Macro F1", f"{m_mae['macro_f1']}", delta=f"{f1_delta}")

# --- PAGE: DATA EXPLORATION ---
elif page == "Data Exploration":
    st.title("📊 Data Exploration & Pre-processing")
    
    data_dir = "UCI HAR Dataset"
    if not os.path.isdir(data_dir):
        st.error(f"Dataset directory '{data_dir}' not found. Please run the training script first to download the data.")
    else:
        st.success("UCI HAR Dataset found!")
        
        tab1, tab2 = st.tabs(["Raw Signals", "iSMOTE Balancing"])
        
        with tab1:
            st.write("Visualizing raw sensor data (Accelerometers & Gyroscopes).")
            # Load a small snippet
            dataset = UCIHARDataset(data_dir, split='train')
            idx = st.slider("Select Sample Index", 0, len(dataset)-1, 0)
            x, y = dataset[idx]
            
            activity = get_activity_names().get(y.item(), "Unknown")
            st.subheader(f"Activity: {activity}")
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            # Channels: 0-2: Body Acc, 3-5: Body Gyro, 6-8: Total Acc
            axes[0].plot(x[0:3].T)
            axes[0].set_title("Body Accelerometer (X, Y, Z)")
            axes[1].plot(x[3:6].T)
            axes[1].set_title("Body Gyroscope (X, Y, Z)")
            axes[2].plot(x[6:9].T)
            axes[2].set_title("Total Accelerometer (X, Y, Z)")
            plt.tight_layout()
            st.pyplot(fig)
            
        with tab2:
            st.subheader("The iSMOTE Algorithm")
            st.markdown("""
            Standard SMOTE can create noise by interpolating between samples that look similar but belong to different classes. 
            **iSMOTE** fixes this by:
            1. Generating a synthetic sample.
            2. Checking its **K-Nearest Neighbors** in the original feature space.
            3. Rejecting the sample if its neighbors don't match its intended class.
            """)
            
            if os.path.exists("results/ismote_distribution.png"):
                st.image("results/ismote_distribution.png", caption="iSMOTE result: Balanced classes without noise.")
            
# --- PAGE: ARCHITECTURE ---
elif page == "Architecture":
    st.title("🧠 Model Architecture")
    
    st.markdown("""
    The SHAR framework uses a specialized encoder designed for temporal sequences.
    """)
    
    tab_contrastive, tab_mae_arch = st.tabs(["Contrastive (Original)", "Masked Autoencoder (MAE)"])
    
    with tab_contrastive:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. 1D Causal CNNs")
            st.write("Two layers of Causal Convolutions extract local temporal patterns without 'looking into the future'.")
            
            st.subheader("2. Lambda Layer (Einsum)")
            st.write("""
            Instead of computationally heavy Self-Attention, we use a **Lambda Layer**. 
            It transforms context into a linear function (Lambda), which is applied to queries. 
            Complexity is reduced from $O(N^2)$ to $O(N)$.
            """)
            
            st.code("""
# Lambda Layer logic (einsum)
q = self.to_q(x)
k = self.to_k(x)
v = self.to_v(x)
# Content Lambda
Lc = torch.einsum('b k n, b v n -> b k v', k, v)
# Interaction
out = torch.einsum('b k n, b k v -> b v n', q, Lc)
            """, language="python")

        with col2:
            st.subheader("3. Random Masking (Pre-training)")
            st.write("To learn without labels, we hide 20% of the signal and ask the model to distinguish it from other signals in the batch.")
            
            # Visualize masking
            dataset = UCIHARDataset("UCI HAR Dataset", split='train')
            x, _ = dataset[0]
            x_masked = apply_random_masking_batch(x.unsqueeze(0)).squeeze(0)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x[0].numpy(), label="Original Signal (Acc X)", alpha=0.5)
            ax.plot(x_masked[0].numpy(), label="Masked Signal", color='red')
            ax.legend()
            st.pyplot(fig)

    with tab_mae_arch:
        st.markdown("""
        ### Masked Autoencoder (MAE) for Time-Series
        
        The MAE approach replaces contrastive learning with a **reconstruction** pretext task.
        Instead of learning to distinguish augmented views, the model learns to **reconstruct
        masked portions** of the input signal — forcing it to build a deep understanding of
        temporal patterns.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. Patch Embedding")
            st.write("The input signal `(B, C, 128)` is split into **16 non-overlapping patches** "
                     "of 8 timesteps each. Each patch is linearly projected to an embedding dimension.")
            
            st.subheader("2. Random Masking (75%)")
            st.write("**75%** of patches are randomly masked — much more aggressive than the 20% "
                     "used in contrastive learning. Only the **visible 25%** are fed to the encoder, "
                     "making pre-training efficient.")
            
            st.subheader("3. Transformer Encoder")
            st.write("A 4-layer Transformer encoder processes ONLY the visible patches. "
                     "This is both efficient (fewer tokens) and forces the encoder to build "
                     "rich representations from limited context.")
        
        with col2:
            st.subheader("4. Decoder & Reconstruction")
            st.write("A lightweight 2-layer Transformer decoder takes the encoded visible patches "
                     "plus learnable `[MASK]` tokens and reconstructs the full signal. "
                     "MSE loss is computed **only on masked patches**.")
            
            st.subheader("5. Fine-tuning")
            st.write("After pre-training, the decoder is **discarded**. The encoder processes "
                     "ALL patches (no masking) and outputs a 256-d vector via global average pooling "
                     "— compatible with the same `SHAR_Classifier` head.")
            
            st.code("""
# MAE Pre-training Forward Pass
patches = patch_embed(x)         # (B, 16, 128)
visible = random_mask(patches)   # (B,  4, 128) ← 75% masked
encoded = transformer(visible)   # (B,  4, 128)
recon = decoder(encoded + masks) # (B, 16, C*8)
loss = MSE(recon[masked], x[masked])
            """, language="python")

# --- PAGE: SIMULATION & DEMO ---
elif page == "Simulation & Demo":
    st.title("🎮 Live classification Demo")
    
    model = load_model()
    
    if model is None:
        st.error("Could not load the model. Ensure `shar_encoder_pretrained.pth` exists or run training first.")
    else:
        st.success("Model loaded successfully!")
        
        dataset = UCIHARDataset("UCI HAR Dataset", split='test')
        
        st.markdown("### Test the Model on Real Samples")
        test_idx = st.number_input("Enter sample ID (0-2946)", 0, len(dataset)-1, 42)
        
        x_in, y_true = dataset[test_idx]
        
        with torch.no_grad():
            logits = model(x_in.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)
            
        activities = get_activity_names()
        true_activity = activities.get(y_true.item())
        pred_activity = activities.get(pred_idx)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sensor Input")
            fig, ax = plt.subplots()
            ax.plot(x_in[0:3].T.numpy())
            ax.set_title("Accelerometer Trace")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Prediction Result")
            if pred_activity == true_activity:
                st.success(f"**PREDICTED:** {pred_activity}")
            else:
                st.error(f"**PREDICTED:** {pred_activity}")
            
            st.write(f"**ACTUAL:** {true_activity}")
            
            # Bar chart of probabilities
            prob_df = pd.DataFrame({
                'Activity': [activities[i] for i in range(6)],
                'Probability': probs
            })
            st.bar_chart(prob_df.set_index('Activity'))

st.markdown("---")
st.markdown("Created for Activity Recognition research.")
