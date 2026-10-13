import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Local module imports
from src.config import (
    ROOT, DATA_PATH, SHEET_NAME, TARGETS, TARGET_LABELS, LEAKAGE_COLUMNS,
    FEATURE_GROUPS, FEATURE_DESCRIPTIONS
)
from src.predict import predict_patient, load_model
from src.explain import get_global_feature_importance, explain_patient
from src.visualization import create_heart_figure, get_vessel_status_dict, VESSEL_METADATA

# Page configuration
st.set_page_config(
    page_title="CardioVision3D | Cardiovascular Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Clinical Styling with Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #080d1a;
        color: #f1f5f9;
    }
    
    /* Global Text Contrast & Readability */
    p, span, label, div {
        color-scheme: dark;
    }
    
    /* Animations */
    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.1); }
        28% { transform: scale(1); }
        42% { transform: scale(1.15); }
        70% { transform: scale(1); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.45); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    @keyframes pulseNormal {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated-fade {
        animation: fadeIn 0.4s ease-out forwards;
    }
    
    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #091224 0%, #0f1f3d 50%, #162a52 100%);
        border-radius: 18px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px -5px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, rgba(0, 0, 0, 0) 70%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 55%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-tagline {
        font-size: 0.98rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    
    .pulse-logo {
        animation: heartbeat 2.2s infinite ease-in-out;
        display: inline-block;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .badge-clinical {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    
    .badge-status {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    
    /* Metric & result cards */
    .metric-card {
        background: #0f172a;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease, border-color 0.25s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -4px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.15);
        border-color: #38bdf8;
    }
    
    .metric-card:hover .metric-label {
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 4px;
        transition: color 0.2s ease;
    }
    
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
    }
    
    .leakage-guard {
        background-color: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 10px;
        padding: 10px 16px;
        color: #34d399;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }
    
    /* Professional Status Badges with Pulse */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    
    .status-alert {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        animation: pulseGlow 2.5s infinite;
    }
    
    .status-normal {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        animation: pulseNormal 3s infinite;
    }
    
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .dot-alert {
        background-color: #ef4444;
        box-shadow: 0 0 8px #ef4444;
    }
    
    .dot-normal {
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    
    /* Clinical Triage Box */
    .triage-panel {
        background: #0d1527;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px 22px;
        margin-top: 16px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #060a14;
        color: #f8fafc;
        border-right: 1px solid #172033;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        padding: 7px 12px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(56, 189, 248, 0.08);
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover * {
        color: #38bdf8 !important;
    }
    
    /* ------------------------------------------------------------- */
    /* HIGH-CONTRAST HOVER & WIDGET VISIBILITY RECTIFICATIONS        */
    /* ------------------------------------------------------------- */
    
    /* 1. BUTTONS (Secondary & Primary) */
    .stButton > button,
    button[kind="secondary"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover,
    button[kind="secondary"]:hover,
    .stButton > button:focus,
    button[kind="secondary"]:focus {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
    }
    
    .stButton > button:hover *,
    button[kind="secondary"]:hover * {
        color: #38bdf8 !important;
    }
    
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    
    button[kind="primary"]:hover,
    button[kind="primary"]:focus {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border-color: #7dd3fc !important;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.45) !important;
    }
    
    button[kind="primary"]:hover * {
        color: #ffffff !important;
    }
    
    /* 2. SELECTBOXES & DROPDOWNS */
    div[data-baseweb="select"] {
        background-color: #0f172a !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }
    
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }
    
    /* Dropdown Popover Listbox & Item Hovers */
    div[data-baseweb="popover"],
    ul[role="listbox"],
    div[data-baseweb="menu"] {
        background-color: #0d1527 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.75) !important;
    }
    
    li[role="option"] {
        background-color: #0d1527 !important;
        color: #f1f5f9 !important;
        font-size: 0.9rem !important;
        padding: 9px 14px !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }
    
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    li[role="option"]:focus,
    li[role="option"][aria-selected="true"]:hover {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] *,
    li[role="option"]:focus * {
        color: #38bdf8 !important;
    }
    
    /* 3. TABS HOVER & ACTIVE STATES */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #1e293b !important;
        gap: 8px !important;
    }
    
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        background: transparent !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 9px 16px !important;
        transition: all 0.2s ease !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.08) !important;
    }
    
    button[data-baseweb="tab"]:hover * {
        color: #38bdf8 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.05) !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #ffffff !important;
    }
    
    /* 4. TEXT & NUMBER INPUTS */
    input[type="number"],
    input[type="text"],
    .stTextInput input {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    
    input[type="number"]:hover,
    input[type="text"]:hover,
    input[type="number"]:focus,
    input[type="text"]:focus {
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
    }
    
    div[data-testid="stNumberInput"] button {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-color: #334155 !important;
    }
    
    div[data-testid="stNumberInput"] button:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
    }
    
    div[data-testid="stNumberInput"] button:hover svg {
        fill: #38bdf8 !important;
    }
    
    /* 5. TOOLTIPS */
    div[data-testid="stTooltipContent"],
    div[data-baseweb="tooltip"] {
        background-color: #0b1329 !important;
        color: #f8fafc !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.75) !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        padding: 8px 12px !important;
    }
    
    div[data-testid="stTooltipContent"] *,
    div[data-baseweb="tooltip"] * {
        color: #f8fafc !important;
    }
    
    /* 6. EXPANDERS */
    div[data-testid="stExpander"] details {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
    }
    
    div[data-testid="stExpander"] details summary {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        transition: color 0.2s ease, background-color 0.2s ease !important;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.06) !important;
    }
    
    div[data-testid="stExpander"] details summary:hover * {
        color: #38bdf8 !important;
    }
    
    div[data-testid="stExpander"] details summary svg {
        fill: #94a3b8 !important;
    }
    
    div[data-testid="stExpander"] details summary:hover svg {
        fill: #38bdf8 !important;
    }
    
    /* 7. CHECKBOXES & RADIOS IN MAIN CONTENT */
    div[data-testid="stCheckbox"] label:hover span,
    div[data-testid="stRadio"] label:hover span {
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# Data loading with caching
@st.cache_data
def get_raw_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file missing at: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()
    return df


try:
    df_raw = get_raw_dataset()
    eligible_features = [c for c in df_raw.columns if c not in LEAKAGE_COLUMNS]
except Exception as err:
    st.error(f"Failed to load dataset: {err}")
    st.stop()


# Session State Initialization
if "selected_patient_idx" not in st.session_state:
    st.session_state["selected_patient_idx"] = 0

if "patient_features" not in st.session_state:
    st.session_state["patient_features"] = df_raw.loc[0, eligible_features].to_dict()

if "predictions" not in st.session_state:
    try:
        st.session_state["predictions"] = predict_patient(st.session_state["patient_features"])
    except Exception:
        st.session_state["predictions"] = None


# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 18px 0; border-bottom: 1px solid #172033; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="pulse-logo">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
                        <path d="M3.22 12H9.5l1.5-3 2 6 1.5-3h4.28"/>
                    </svg>
                </div>
                <div>
                    <h2 style="color: white; margin: 0; font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em;">CardioVision3D</h2>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.72rem; font-weight: 500;">Cardiovascular Intelligence</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation",
        options=[
            "Overview",
            "Patient Assessment",
            "3D Heart Explorer",
            "Explainable AI"
        ],
        index=1  # Default to Patient Assessment
    )

    st.markdown("---")
    st.caption("Target Leakage Shield: Active")
    st.caption("Excludes: Cath, LAD, LCX, RCA")
    
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; margin-top: 15px;">
            <span class="badge badge-clinical">Decision Support</span>
            <p style="font-size: 0.75rem; color: #94a3b8; margin: 6px 0 0 0;">
                Clinical research platform for cardiovascular risk stratification and 3D coronary correlation.
            </p>
        </div>
    """, unsafe_allow_html=True)


# Top Hero Banner
st.markdown(f"""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <span class="badge badge-clinical">Clinical Intelligence Platform</span>
            <span class="badge badge-status">Models Active</span>
            <h1 class="hero-title">CardioVision3D</h1>
            <p class="hero-tagline">See the Risk. Understand the Heart. — 3D Epicardial Perfusion & Multimodal Risk Intelligence</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
if nav_choice == "Overview":
    st.markdown("### System Overview & Clinical Cohort Profile")
    st.write(
        "CardioVision3D unifies multi-target machine learning classification, SHAP feature attributions, and "
        "interactive 3D anatomical coronary visualization to accelerate non-invasive cardiovascular risk stratification."
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cohort Size</div>
                <div class="metric-value">{len(df_raw)}</div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">Z-Alizadeh Sani Patients</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Input Predictors</div>
                <div class="metric-value">{len(eligible_features)}</div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">Demographics, ECG, Biomarkers, Echo</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Prediction Targets</div>
                <div class="metric-value">4</div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">Overall CAD, LAD, LCX, RCA</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">CAD Test ROC-AUC</div>
                <div class="metric-value" style="color: #38bdf8;">0.870</div>
                <div style="color: #10b981; font-size: 0.8rem; margin-top: 4px;">Held-out Stratified Test (N=61)</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_arch, col_targets = st.columns([1.1, 0.9])

    with col_arch:
        st.markdown("#### Multimodal Decision Pipeline")
        st.markdown("""
        - **Data Ingestion:** 55 non-invasive clinical predictors across demographics, 12-lead ECG, blood biomarkers, and echocardiography.
        - **Target Leakage Shield:** Strict elimination of diagnostic target columns (`Cath`, `LAD`, `LCX`, `RCA`) from feature spaces.
        - **Machine Learning Core:** Stratified ensemble pipelines delivering individual probabilities for overall CAD and three major epicardial arteries.
        - **Explainability Engine:** Real SHAP local attributions identifying which patient factors drive risk higher or lower.
        - **3D Anatomical Explorer:** Real-time projection of vessel-specific predictions onto a 3D coronary artery tree.
        """)

    with col_targets:
        st.markdown("#### Clinical Target Distribution")
        target_summary_data = []
        for t_key, t_col in TARGETS.items():
            pos_count = (df_raw[t_col].astype(str).str.strip().str.lower().isin(["cad", "stenotic"])).sum()
            total = len(df_raw)
            meta = TARGET_LABELS[t_key]
            target_summary_data.append({
                "Target": meta["name"],
                "Positive Label": meta["positive"],
                "Positive Cases": pos_count,
                "Normal Cases": total - pos_count,
                "Prevalence (%)": f"{(pos_count / total * 100):.1f}%"
            })
        df_target_summary = pd.DataFrame(target_summary_data)
        st.dataframe(df_target_summary, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 2: PATIENT ASSESSMENT
# ==============================================================================
elif nav_choice == "Patient Assessment":
    st.markdown("### Patient Clinical Profile & Multimodal Risk Assessment")
    st.write(
        "Select a patient from the clinical cohort, navigate between records, review multimodal biomarkers, and execute the 4-target machine learning ensemble."
    )

    st.markdown("""
        <div class="leakage-guard">
            <b>Target Leakage Shield Active:</b> Outcome targets <code>Cath</code>, <code>LAD</code>, <code>LCX</code>, and <code>RCA</code> are strictly excluded from inputs.
        </div>
    """, unsafe_allow_html=True)

    # Interactive Patient Navigation with Previous / Next Controls
    col_prev, col_sel, col_nxt = st.columns([0.65, 2.7, 0.65])

    current_idx = st.session_state["selected_patient_idx"]

    with col_prev:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("◀ Previous", use_container_width=True):
            new_idx = (current_idx - 1) % len(df_raw)
            st.session_state["selected_patient_idx"] = new_idx
            st.session_state["patient_features"] = df_raw.loc[new_idx, eligible_features].to_dict()
            st.session_state["predictions"] = predict_patient(st.session_state["patient_features"])
            st.rerun()

    with col_sel:
        selected_idx = st.selectbox(
            "Select Clinical Cohort Record (Patients 1 to 303):",
            options=list(range(len(df_raw))),
            index=current_idx,
            format_func=lambda idx: (
                f"Patient #{idx + 1:03d} | Age: {df_raw.loc[idx, 'Age']} | "
                f"Sex: {df_raw.loc[idx, 'Sex']} | BMI: {df_raw.loc[idx, 'BMI']} | "
                f"BP: {df_raw.loc[idx, 'BP']} mmHg | EF: {df_raw.loc[idx, 'EF-TTE']}%"
            )
        )

        if selected_idx != current_idx:
            st.session_state["selected_patient_idx"] = selected_idx
            st.session_state["patient_features"] = df_raw.loc[selected_idx, eligible_features].to_dict()
            st.session_state["predictions"] = predict_patient(st.session_state["patient_features"])
            st.rerun()

    with col_nxt:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Next ▶", use_container_width=True):
            new_idx = (current_idx + 1) % len(df_raw)
            st.session_state["selected_patient_idx"] = new_idx
            st.session_state["patient_features"] = df_raw.loc[new_idx, eligible_features].to_dict()
            st.session_state["predictions"] = predict_patient(st.session_state["patient_features"])
            st.rerun()

    # Patient Snapshot & Multimodal Radar Comparison
    curr_patient = st.session_state["patient_features"]
    
    col_snap1, col_snap2 = st.columns([1.1, 1.2])

    with col_snap1:
        st.markdown("#### Patient Physiological Profile")
        age_val = curr_patient.get("Age", 0)
        bmi_val = curr_patient.get("BMI", 0)
        bp_val = curr_patient.get("BP", 0)
        ef_val = curr_patient.get("EF-TTE", 0)
        fbs_val = curr_patient.get("FBS", 0)
        tg_val = curr_patient.get("TG", 0)

        st.markdown(f"""
        <div style="background: #0f172a; border-radius: 12px; padding: 16px; border: 1px solid #1e293b;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div><span style="color: #94a3b8; font-size: 0.8rem;">Age / Sex:</span><br><b>{age_val} yrs / {curr_patient.get('Sex', '—')}</b></div>
                <div><span style="color: #94a3b8; font-size: 0.8rem;">BMI Category:</span><br><b>{bmi_val:.1f} kg/m² {'(Elevated)' if bmi_val > 25 else '(Normal)'}</b></div>
                <div><span style="color: #94a3b8; font-size: 0.8rem;">Blood Pressure:</span><br><b>{bp_val} mmHg {'(Hypertensive)' if bp_val > 130 else '(Normotensive)'}</b></div>
                <div><span style="color: #94a3b8; font-size: 0.8rem;">Ejection Fraction:</span><br><b>{ef_val}% {'(Reduced)' if ef_val < 50 else '(Preserved)'}</b></div>
                <div><span style="color: #94a3b8; font-size: 0.8rem;">Fasting Sugar:</span><br><b>{fbs_val} mg/dL {'(Impaired)' if fbs_val > 100 else '(Normal)'}</b></div>
                <div><span style="color: #94a3b8; font-size: 0.8rem;">Triglycerides:</span><br><b>{tg_val} mg/dL {'(Elevated)' if tg_val > 150 else '(Normal)'}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_snap2:
        st.markdown("#### Multimodal Biomarker Radar Profile")
        # Normalize 6 key features (0 to 100 scale)
        def norm(val, vmin, vmax):
            return max(0, min(100, (val - vmin) / (vmax - vmin) * 100))

        radar_categories = ["Age", "Systolic BP", "BMI", "Fasting Glucose", "Triglycerides", "Impaired EF"]
        patient_radar_vals = [
            norm(float(age_val), 30, 80),
            norm(float(bp_val), 90, 180),
            norm(float(bmi_val), 18, 40),
            norm(float(fbs_val), 70, 220),
            norm(float(tg_val), 60, 350),
            norm(100 - float(ef_val), 20, 80)  # Lower EF yields higher impairment
        ]
        cohort_median_vals = [55, 45, 42, 35, 40, 35]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=patient_radar_vals + [patient_radar_vals[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            name=f'Patient #{st.session_state["selected_patient_idx"] + 1}',
            line=dict(color='#38bdf8', width=2),
            fillcolor='rgba(56, 189, 248, 0.22)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=cohort_median_vals + [cohort_median_vals[0]],
            theta=radar_categories + [radar_categories[0]],
            name='Cohort Benchmark',
            line=dict(color='#94a3b8', dash='dash', width=1.5)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 100]),
                bgcolor="#0f172a"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=20, r=20, t=15, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=9, color="#cbd5e1")),
            hoverlabel=dict(
                bgcolor="#0b1329",
                font=dict(color="#ffffff", size=12, family="Inter, sans-serif"),
                bordercolor="#38bdf8"
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Clinical Predictor Parameter Inspection & Editing")

    edited_features = dict(st.session_state["patient_features"])

    with st.expander("Expand Predictor Parameter Editor (55 Clinical Variables)", expanded=False):
        tabs_groups = st.tabs(list(FEATURE_GROUPS.keys()))

        for tab_idx, (group_name, group_cols) in enumerate(FEATURE_GROUPS.items()):
            with tabs_groups[tab_idx]:
                valid_cols = [c for c in group_cols if c in eligible_features]
                cols_per_row = 3
                grid_cols = st.columns(cols_per_row)

                for idx, col_name in enumerate(valid_cols):
                    curr_val = edited_features.get(col_name, df_raw.loc[st.session_state["selected_patient_idx"], col_name])
                    tooltip = FEATURE_DESCRIPTIONS.get(col_name, "")

                    with grid_cols[idx % cols_per_row]:
                        if col_name in ["Sex"]:
                            opts = ["Male", "Fmale"] if "Fmale" in df_raw[col_name].values else ["Male", "Female"]
                            edited_features[col_name] = st.selectbox(
                                f"{col_name}",
                                options=opts,
                                index=opts.index(curr_val) if curr_val in opts else 0,
                                key=f"feat_{col_name}",
                                help=tooltip
                            )
                        elif col_name in ["BBB", "VHD"]:
                            unique_vals = list(df_raw[col_name].dropna().unique())
                            edited_features[col_name] = st.selectbox(
                                f"{col_name}",
                                options=unique_vals,
                                index=unique_vals.index(curr_val) if curr_val in unique_vals else 0,
                                key=f"feat_{col_name}",
                                help=tooltip
                            )
                        elif isinstance(curr_val, str) or col_name in ["Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur", "Dyspnea", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "LVH", "Poor R Progression"]:
                            opts = ["N", "Y"]
                            val_str = str(curr_val).strip()
                            idx_sel = opts.index(val_str) if val_str in opts else 0
                            edited_features[col_name] = st.selectbox(
                                f"{col_name}",
                                options=opts,
                                index=idx_sel,
                                key=f"feat_{col_name}",
                                help=tooltip
                            )
                        elif isinstance(curr_val, (int, np.integer)):
                            edited_features[col_name] = st.number_input(
                                f"{col_name}",
                                value=int(curr_val),
                                step=1,
                                key=f"feat_{col_name}",
                                help=tooltip
                            )
                        elif isinstance(curr_val, (float, np.floating)):
                            edited_features[col_name] = st.number_input(
                                f"{col_name}",
                                value=float(curr_val),
                                step=0.1,
                                format="%.2f",
                                key=f"feat_{col_name}",
                                help=tooltip
                            )
                        else:
                            edited_features[col_name] = st.text_input(
                                f"{col_name}",
                                value=str(curr_val),
                                key=f"feat_{col_name}",
                                help=tooltip
                            )

    st.session_state["patient_features"] = edited_features

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns([1.6, 2])
    with btn_col1:
        run_pred = st.button("Execute Multi-Target Analysis", type="primary", use_container_width=True)

    if run_pred or st.session_state["predictions"] is None:
        try:
            with st.spinner("Executing machine learning pipelines..."):
                predictions = predict_patient(st.session_state["patient_features"])
                st.session_state["predictions"] = predictions
        except Exception as pred_err:
            st.error(f"Prediction failed: {pred_err}")

    preds = st.session_state.get("predictions")
    if preds:
        st.markdown("### Multi-Target Prediction Results")
        st.caption("Predicted diagnostic class and positive-class probability computed from validated ensemble models.")

        p_cols = st.columns(4)
        targets_order = ["cad", "lad", "lcx", "rca"]

        stenotic_vessels = []
        for idx, key in enumerate(targets_order):
            res = preds[key]
            prob = res["positive_probability"]
            is_positive = (res["predicted_class"] == 1)

            if is_positive and key != "cad":
                stenotic_vessels.append(key.upper())

            badge_html = (
                f'<span class="status-pill status-alert"><span class="dot dot-alert"></span> {res["predicted_label"]}</span>'
                if is_positive else
                f'<span class="status-pill status-normal"><span class="dot dot-normal"></span> {res["predicted_label"]}</span>'
            )

            bar_color = "#ef4444" if prob >= 0.50 else "#10b981"

            with p_cols[idx]:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {'#ef4444' if is_positive else '#10b981'};">
                    <div class="metric-label">{res['target_name']}</div>
                    <div style="margin: 10px 0;">{badge_html}</div>
                    <div style="margin-top: 8px;">
                        <span style="font-size: 1.55rem; font-weight: 800; color: #f8fafc;">{prob * 100:.1f}%</span>
                        <span style="color: #94a3b8; font-size: 0.8rem;"> probability</span>
                    </div>
                    <div style="background: #1e293b; border-radius: 9999px; height: 8px; width: 100%; margin: 8px 0; overflow: hidden;">
                        <div style="background: {bar_color}; width: {min(100, max(0, prob * 100)):.1f}%; height: 100%;"></div>
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500;">
                        Risk Tier: <b style="color: {'#f87171' if prob >= 0.50 else '#34d399'};">{res['risk_tier']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Clinical Triage & Vessel Burden Summary
        num_sten = len(stenotic_vessels)
        cad_pos = (preds["cad"]["predicted_class"] == 1)
        
        burden_desc = "No Obstructive Epicardial Stenosis" if num_sten == 0 else (
            f"Single-Vessel Disease ({stenotic_vessels[0]})" if num_sten == 1 else (
                f"Double-Vessel Disease ({', '.join(stenotic_vessels)})" if num_sten == 2 else "Triple-Vessel Disease (LAD + LCX + RCA)"
            )
        )

        st.markdown(f"""
        <div class="triage-panel animated-fade">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="badge badge-clinical">Clinical Decision Support Triage</span>
                    <h4 style="margin: 6px 0 2px 0; color: white;">Anatomical Burden: {burden_desc}</h4>
                    <p style="margin: 0; font-size: 0.82rem; color: #94a3b8;">
                        Overall CAD Classifier: <b>{'CAD Indicated' if cad_pos else 'Normal Angiogram Projected'}</b> | 
                        Vessel Involvement: <b>{num_sten} of 3 Epicardial Arteries</b>
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">Suggested Evaluation Priority:</span><br>
                    <span style="font-weight: 700; color: {'#ef4444' if num_sten >= 2 or 'LAD' in stenotic_vessels else ('#f59e0b' if num_sten == 1 else '#10b981')}; font-size: 1.1rem;">
                        {'HIGH (Consider Invasive Angiography)' if num_sten >= 2 or 'LAD' in stenotic_vessels else ('MODERATE (Non-invasive Stress Test)' if num_sten == 1 else 'LOW (Preventive Risk Modification)')}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: 3D HEART EXPLORER
# ==============================================================================
elif nav_choice == "3D Heart Explorer":
    st.markdown("### 3D Heart & Coronary Vessel Explorer")
    st.write(
        "Correlate patient-specific model predictions with anatomical coronary architecture and epicardial perfusion territories."
    )

    preds = st.session_state.get("predictions")
    if preds is None:
        st.session_state["predictions"] = predict_patient(st.session_state["patient_features"])
        preds = st.session_state["predictions"]

    vessel_status = get_vessel_status_dict(preds)

    tab_3d, tab_anat_ref = st.tabs([
        "Interactive 3D Digital Twin",
        "Anatomical & Angiographic Reference"
    ])

    with tab_3d:
        col_view, col_panel = st.columns([1.35, 0.85])

        with col_view:
            c_tog1, c_tog2 = st.columns(2)
            with c_tog1:
                show_orient = st.checkbox("Show Anatomical Landmarks", value=True)
            with c_tog2:
                theme_choice = st.radio("Viewport Theme", ["dark", "light"], horizontal=True, index=0)

            heart_fig = create_heart_figure(
                vessel_status=vessel_status,
                show_orientation_labels=show_orient,
                theme=theme_choice
            )
            st.plotly_chart(heart_fig, use_container_width=True)

            st.caption(
                "Controls: Click and drag to rotate in 3D. Scroll to zoom. "
                "Hover over vessels and nodes for territory and probability details."
            )

        with col_panel:
            st.markdown("#### Vessel Status Inspector")

            status_legend_html = """
            <div style="background: #0f172a; border-radius: 10px; padding: 14px; border: 1px solid #1e293b; margin-bottom: 16px;">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px;">Legend & Status</div>
                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background: #ef4444; border-radius: 50%; box-shadow: 0 0 8px #ef4444;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #f87171;">Red: Model Predicts Stenosis (Class 1)</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #34d399;">Green: Model Predicts Normal (Class 0)</span>
                </div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <span style="display: inline-block; width: 10px; height: 10px; background: #64748b; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; color: #94a3b8;">Grey: Unassessed / Insufficient Data</span>
                </div>
            </div>
            """
            st.markdown(status_legend_html, unsafe_allow_html=True)

            for v_key in ["LAD", "LCX", "RCA"]:
                meta = VESSEL_METADATA[v_key]
                v_data = vessel_status.get(v_key, {"status": "Not assessed", "probability": None})
                status_str = v_data.get("status", "Not assessed")
                prob_val = v_data.get("probability", 0)

                is_sten = (status_str == "Stenotic")
                badge = (
                    f'<span class="status-pill status-alert"><span class="dot dot-alert"></span> Stenotic</span>' if is_sten else
                    f'<span class="status-pill status-normal"><span class="dot dot-normal"></span> Normal</span>'
                )

                st.markdown(f"""
                <div style="background: #0f172a; border-radius: 10px; padding: 14px; border: 1px solid #1e293b; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1rem; color: #f8fafc;">{v_key}</span>
                        {badge}
                    </div>
                    <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 500; margin-top: 2px;">{meta['full_name']}</div>
                    <div style="font-size: 0.85rem; margin-top: 6px; color: #cbd5e1;">
                        Model Probability: <b>{prob_val * 100:.1f}%</b>
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">
                        <b>Territory:</b> {meta['territory']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_anat_ref:
        st.markdown("#### Coronary Anatomy & Pathophysiology Reference")
        st.write(
            "Medical reference illustrations displaying epicardial coronary arterial branches "
            "and atherosclerotic lumen stenosis cross-section."
        )

        img_col1, img_col2 = st.columns(2)

        img_heart_path = ROOT / "assets" / "coronary_arteries_3d.jpg"
        img_sten_path = ROOT / "assets" / "coronary_stenosis_diagram.jpg"

        with img_col1:
            if img_heart_path.exists():
                st.image(str(img_heart_path), caption="Photorealistic 3D Coronary Anatomy & Ventricular Branches", use_container_width=True)
            st.markdown("""
            **Anatomical Vascular Territories:**
            - **Left Anterior Descending (LAD):** Supplies anterior left ventricular wall, septum, and apex.
            - **Left Circumflex (LCX):** Supplies obtuse marginal branches to the lateral LV wall.
            - **Right Coronary Artery (RCA):** Supplies right ventricle, AV node, and posterior descending artery.
            """)

        with img_col2:
            if img_sten_path.exists():
                st.image(str(img_sten_path), caption="Coronary Artery Atherosclerotic Plaque Stenosis & Hemodynamic Obstruction", use_container_width=True)
            st.markdown("""
            **Stenosis Pathophysiology:**
            - **Atheroma Formation:** Intimal lipid accumulation and fibrofatty plaque encroachment.
            - **Lumen Compromise:** Significant coronary stenosis (>50-70% diameter reduction) triggers downstream ischemia during myocardial demand.
            - **Multimodal Markers:** Strongly correlated with typical angina, ST elevation/depression, Ejection Fraction, and lipid biomarkers.
            """)


# ==============================================================================
# SECTION 4: EXPLAINABLE AI (SHAP)
# ==============================================================================
elif nav_choice == "Explainable AI":
    st.markdown("### Explainable AI (SHAP Feature Attributions)")
    st.write(
        "Transparent clinical reasoning through SHAP (SHapley Additive exPlanations), "
        "highlighting patient biomarkers that increase disease likelihood versus protective indicators."
    )

    preds = st.session_state.get("predictions")
    if preds is None:
        st.session_state["predictions"] = predict_patient(st.session_state["patient_features"])
        preds = st.session_state["predictions"]

    exp_target = st.selectbox(
        "Select Target Model to Inspect:",
        options=["cad", "lad", "lcx", "rca"],
        format_func=lambda k: f"{TARGET_LABELS[k]['name']} ({k.upper()})"
    )

    tab_local, tab_global, tab_glossary = st.tabs([
        "Patient-Specific Attribution",
        "Global Feature Importance",
        "Clinical Biomarker Glossary"
    ])

    with tab_local:
        st.markdown(f"#### Local Attribution: {TARGET_LABELS[exp_target]['name']}")

        with st.spinner("Computing SHAP values..."):
            try:
                explanation = explain_patient(st.session_state["patient_features"], target_key=exp_target, top_n=10)
            except Exception as exp_err:
                st.error(f"Error computing explanation: {exp_err}")
                explanation = None

        if explanation:
            b_val = explanation["base_value"]
            p_val = explanation["predicted_probability"]

            st.markdown(f"""
            <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px;">
                <div class="metric-card" style="flex: 1; min-width: 200px;">
                    <div class="metric-label">Model Prior / Base Rate</div>
                    <div class="metric-value">{b_val * 100:.1f}%</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">Average expected probability</div>
                </div>
                <div class="metric-card" style="flex: 1; min-width: 200px;">
                    <div class="metric-label">Patient Probability</div>
                    <div class="metric-value" style="color: {'#ef4444' if p_val >= 0.50 else '#10b981'};">{p_val * 100:.1f}%</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">After feature adjustments</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            df_plot = explanation["all_attributions"].head(12).copy()
            df_plot = df_plot.sort_values(by="shap_value", ascending=True)

            fig_shap = px.bar(
                df_plot,
                x="shap_value",
                y="feature",
                orientation="h",
                color="direction",
                color_discrete_map={
                    "Increases Likelihood": "#ef4444",
                    "Reduces Likelihood / Protective": "#10b981"
                },
                labels={"shap_value": "SHAP Contribution (Log-Odds / Prob Shift)", "feature": "Clinical Predictor"},
                title=f"Top 12 SHAP Attributions for Patient #{st.session_state['selected_patient_idx'] + 1:03d}"
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#f1f5f9"),
                height=450,
                margin=dict(l=10, r=10, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                hoverlabel=dict(
                    bgcolor="#0b1329",
                    font=dict(color="#ffffff", size=12, family="Inter, sans-serif"),
                    bordercolor="#38bdf8"
                )
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### Features Driving Risk Higher")
                for item in explanation["top_positive"][:5]:
                    st.markdown(f"""
                    <div style="background: #0f172a; border-left: 4px solid #ef4444; border-radius: 8px; padding: 12px; margin-bottom: 8px; border: 1px solid #1e293b;">
                        <div style="font-weight: 700; color: #f87171;">{item['feature']} (Value: {item['patient_value']})</div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{item['description']}</div>
                        <div style="font-size: 0.75rem; color: #ef4444; font-weight: 700; margin-top: 4px;">+ {item['shap_value']:.4f} SHAP score</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_neg:
                st.markdown("##### Protective Features Reducing Risk")
                for item in explanation["top_negative"][:5]:
                    st.markdown(f"""
                    <div style="background: #0f172a; border-left: 4px solid #10b981; border-radius: 8px; padding: 12px; margin-bottom: 8px; border: 1px solid #1e293b;">
                        <div style="font-weight: 700; color: #34d399;">{item['feature']} (Value: {item['patient_value']})</div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{item['description']}</div>
                        <div style="font-size: 0.75rem; color: #10b981; font-weight: 700; margin-top: 4px;">{item['shap_value']:.4f} SHAP score</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_global:
        st.markdown(f"#### Global Feature Importance across Training Cohort ({TARGET_LABELS[exp_target]['name']})")
        df_global = get_global_feature_importance(target_key=exp_target, top_n=15)

        fig_global = px.bar(
            df_global.sort_values(by="importance", ascending=True),
            x="importance",
            y="clean_feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Global Feature Importance (Gini / Split Gain)", "clean_feature": "Clinical Predictor"},
            title=f"Top 15 Predictors Ranked by Global Model Importance ({exp_target.upper()})"
        )
        fig_global.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#f1f5f9"),
            height=480,
            margin=dict(l=10, r=10, t=40, b=20),
            coloraxis_showscale=False,
            hoverlabel=dict(
                bgcolor="#0b1329",
                font=dict(color="#ffffff", size=12, family="Inter, sans-serif"),
                bordercolor="#38bdf8"
            )
        )
        st.plotly_chart(fig_global, use_container_width=True)

        st.dataframe(df_global, use_container_width=True, hide_index=True)

    with tab_glossary:
        st.markdown("#### Clinical Biomarker & ECG Glossary")
        glossary_items = []
        for k, v in FEATURE_DESCRIPTIONS.items():
            glossary_items.append({"Feature Code": k, "Clinical Meaning & Unit": v})
        st.dataframe(pd.DataFrame(glossary_items), use_container_width=True, hide_index=True)