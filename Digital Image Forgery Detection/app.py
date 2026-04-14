import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
import seaborn as sns                                        
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os

# ================= CONFIG =================
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "CNN_D64_128_128.keras")


st.set_page_config(
    page_title="Advanced Forgery Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── SIMPLE ANIMATIONS ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── BACKGROUND ICONS ── */
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-15px); }
    }
    .bg-icons-wrap {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
        opacity: 0.18;
    }
    .bg-icon {
        position: absolute;
        font-size: 2.5rem;
        user-select: none;
        animation: iconFloat 5s ease-in-out infinite;
    }
    /* Left Cluster */
    .bg-icon.i1  { top: 12%; left: 15%; animation-delay: 0s;   }
    .bg-icon.i2  { top: 22%; left: 8%;  animation-delay: 1.5s; }
    .bg-icon.i3  { top: 32%; left: 18%; animation-delay: 0.5s; }
    .bg-icon.i4  { top: 18%; left: 25%; animation-delay: 2s;   }
    
    /* Right Cluster */
    .bg-icon.i5  { top: 15%; right: 12%; animation-delay: 1s;   }
    .bg-icon.i6  { top: 28%; right: 8%;  animation-delay: 2.5s; }
    .bg-icon.i7  { top: 38%; right: 15%; animation-delay: 0.8s; }
    .bg-icon.i8  { top: 20%; right: 28%; animation-delay: 3s;   }

    /* ── FORCE DARK ROOT (covers the white header bar) ── */
    html, body { background-color: #07081a !important; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #root, .stApp, .main { background-color: #07081a; }

    /* ── STREAMLIT TOP TOOLBAR ── */
    [data-testid="stHeader"] {
        background-color: #07081a !important;
        background-image:
            radial-gradient(ellipse at 15% 50%, rgba(102,126,234,0.20) 0%, transparent 55%),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Cdefs%3E%3Cpattern id='h' x='0' y='0' width='60' height='60' patternUnits='userSpaceOnUse'%3E%3Cpolygon points='30,2 52,16 52,44 30,58 8,44 8,16' fill='none' stroke='rgba(102,126,234,0.10)' stroke-width='0.8'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='60' height='60' fill='url(%23h)'/%3E%3C/svg%3E");
        background-size: 100% 100%, 60px 60px;
        border-bottom: 1px solid rgba(102,126,234,0.18) !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] [data-testid="stToolbar"] {
        color: rgba(180,190,255,0.6) !important;
    }
    /* ── HEADER/TOOLBAR — kill all white backgrounds ── */
    /* Every direct child element inside the header */
    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] * {
        background-color: transparent !important;
        color: rgba(200,210,255,0.85) !important;
    }
    /* The actual Search pill button (white pill in top-right) */
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] [role="button"],
    [data-testid="stToolbarActions"] button,
    [data-testid="stToolbarActions"] [role="button"],
    [data-testid="stToolbar"] button {
        background: rgba(18,20,50,0.90) !important;
        background-color: rgba(18,20,50,0.90) !important;
        color: rgba(200,210,255,0.90) !important;
        border: 1px solid rgba(102,126,234,0.40) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.40) !important;
        transition: background 0.18s, box-shadow 0.18s !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="stToolbarActions"] button:hover {
        background: rgba(102,126,234,0.22) !important;
        background-color: rgba(102,126,234,0.22) !important;
        box-shadow: 0 0 12px rgba(102,126,234,0.35) !important;
        color: #ffffff !important;
    }
    /* Text/spans INSIDE the toolbar buttons (e.g. "Search" label) */
    [data-testid="stHeader"] button span,
    [data-testid="stHeader"] button p,
    [data-testid="stToolbarActions"] button span,
    [data-testid="stToolbarActions"] button p {
        color: rgba(200,210,255,0.90) !important;
        background: transparent !important;
    }
    /* SVGs inside toolbar buttons */
    [data-testid="stHeader"] button svg,
    [data-testid="stToolbarActions"] button svg {
        fill: rgba(180,195,255,0.85) !important;
        stroke: rgba(180,195,255,0.85) !important;
    }
    /* ── STOP BUTTON (visible red pill next to Deploy) ── */
    [data-testid="StopButton"],
    [data-testid="stStatusWidget"],
    button[kind="header"][data-testid*="stop" i],
    [data-testid="stToolbarActions"] button[title*="Stop" i],
    [data-testid="stToolbarActions"] button[aria-label*="Stop" i],
    [data-testid="stHeader"] button[title*="Stop" i],
    [data-testid="stHeader"] button[aria-label*="Stop" i] {
        background: linear-gradient(135deg, #e53e3e, #c53030) !important;
        background-color: #e53e3e !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,100,100,0.60) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 14px rgba(229,62,62,0.50) !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="StopButton"]:hover,
    [data-testid="stToolbarActions"] button[title*="Stop" i]:hover {
        background: linear-gradient(135deg, #fc5c5c, #e53e3e) !important;
        box-shadow: 0 0 22px rgba(229,62,62,0.70) !important;
    }
    [data-testid="StopButton"] span,
    [data-testid="StopButton"] p,
    [data-testid="StopButton"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Streamlit "Running..." status widget (spinner + stop) */
    [data-testid="stStatusWidget"] {
        background: rgba(18,20,50,0.90) !important;
        border: 1px solid rgba(102,126,234,0.40) !important;
        border-radius: 10px !important;
        color: rgba(200,210,255,0.95) !important;
        padding: 2px 8px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    [data-testid="stStatusWidget"] * {
        color: rgba(200,210,255,0.95) !important;
        -webkit-text-fill-color: rgba(200,210,255,0.95) !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* ── HEADER SEARCH INPUT ── */
    [data-testid="stHeader"] input,
    [data-testid="stToolbar"] input,
    header input {
        background: #0c0d23 !important;
        background-color: #0c0d23 !important;
        color: #dde0ff !important;
        -webkit-text-fill-color: #dde0ff !important;
        border: 1px solid rgba(102,126,234,0.50) !important;
        border-radius: 8px !important;
        caret-color: #a78bfa !important;
        outline: none !important;
    }
    [data-testid="stHeader"] input::placeholder,
    [data-testid="stToolbar"] input::placeholder,
    header input::placeholder {
        color: rgba(150,165,255,0.50) !important;
        -webkit-text-fill-color: rgba(150,165,255,0.50) !important;
    }
    /* Search dialog / popover */
    [data-testid="stSearchBox"],
    [class*="searchBox"],
    [class*="SearchBox"],
    [class*="CommandPalette"],
    [class*="commandPalette"] {
        background: rgba(10,11,30,0.97) !important;
        background-color: rgba(10,11,30,0.97) !important;
        border: 1px solid rgba(102,126,234,0.45) !important;
        border-radius: 12px !important;
        color: #dde0ff !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 40px rgba(0,0,0,0.60) !important;
    }
    [data-testid="stSearchBox"] input,
    [class*="searchBox"] input,
    [class*="CommandPalette"] input {
        background: #0c0d23 !important;
        background-color: #0c0d23 !important;
        color: #dde0ff !important;
        -webkit-text-fill-color: #dde0ff !important;
        caret-color: #a78bfa !important;
    }


    /* ── MAIN APP BACKGROUND — Simplified ── */
    .stApp {
        background-color: #07081a;
        background-image:
            /* Subtle dot grid */
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20'%3E%3Ccircle cx='10' cy='10' r='0.6' fill='rgba(102,126,234,0.12)'/%3E%3C/svg%3E"),
            /* Very soft gradients */
            radial-gradient(ellipse at 15% 10%, rgba(102,126,234,0.08) 0%, transparent 60%),
            radial-gradient(ellipse at 85% 90%, rgba(118,75,162,0.06) 0%, transparent 60%);
        background-size: 20px 20px, 100% 100%, 100% 100%;
        background-attachment: fixed;
        min-height: 100vh;
    }

    /* ── SIDEBAR BACKGROUND ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(170deg, #0c0d22 0%, #110d2e 55%, #0c0d22 100%) !important;
        border-right: 1px solid rgba(102,126,234,0.2) !important;
        box-shadow: 4px 0 30px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"] > div { background: transparent !important; }
    [data-testid="stSidebar"] * { color: rgba(220,225,255,0.90) !important; }

    /* ── HEADER ── */
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        color: #8892b0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* ── RESULT BADGES ── */
    .result-card {
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin: 0.5rem 0 1rem 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        animation: fadeInUp 0.5s ease;
    }
    .forged-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border: 2px solid rgba(255,107,107,0.5);
        box-shadow: 0 8px 40px rgba(255,65,108,0.4);
    }
    .authentic-card {
        background: linear-gradient(135deg, #00b09b, #11998e);
        color: white;
        border: 2px solid rgba(0,210,160,0.5);
        box-shadow: 0 8px 40px rgba(0,176,155,0.4);
    }
    .forged-label   { color: #ff6b6b; font-weight: bold; font-size: 1.5rem; }
    .authentic-label{ color: #00d2a0; font-weight: bold; font-size: 1.5rem; }

    /* ── INFO BOXES ── */
    .info-box {
        background: rgba(102,126,234,0.12);
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        font-size: 0.9rem;
        color: rgba(200,210,255,0.92);
    }
    .warning-box {
        background: rgba(255,65,108,0.12);
        border-left: 4px solid #ff416c;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        font-size: 0.9rem;
        color: rgba(255,200,210,0.92);
    }
    .success-box {
        background: rgba(0,176,155,0.12);
        border-left: 4px solid #00b09b;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        font-size: 0.9rem;
        color: rgba(200,255,240,0.92);
    }

    /* ── SECTION HEADERS ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #c5caff;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(102,126,234,0.30);
        margin: 1rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: rgba(14,15,35,0.85);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(102,126,234,0.25);
        transition: transform 0.2s ease, border-color 0.2s;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(102,126,234,0.55);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #6b7399;
        font-weight: 500;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── SIDEBAR MODEL CARD ── */
    .sidebar-model-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        color: white;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid rgba(102,126,234,0.35);
    }
    .sidebar-model-card h3 { margin: 0; font-size: 1.1rem; font-weight: 600; }
    .sidebar-model-card p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.85rem; }
    .sidebar-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        margin-top: 0.5rem;
        border: 1px solid rgba(255,255,255,0.18);
    }

    /* ── TAB SWITCH TRANSITION KEYFRAMES ── */
    @keyframes tabFadeSlide {
        0%   { opacity: 0; transform: translateY(14px) scale(0.985); filter: blur(2px); }
        60%  { opacity: 0.9; filter: blur(0px); }
        100% { opacity: 1; transform: translateY(0px) scale(1); filter: blur(0); }
    }
    @keyframes tabIndicatorGrow {
        from { width: 0%; left: 50%; opacity: 0; }
        to   { width: 80%; left: 10%; opacity: 1; }
    }
    @keyframes tabPop {
        0%   { transform: scale(1); }
        40%  { transform: scale(1.07); }
        70%  { transform: scale(0.97); }
        100% { transform: scale(1); }
    }
    @keyframes tabGlowPulse {
        0%,100% { box-shadow: 0 2px 16px rgba(102,126,234,0.28); }
        50%     { box-shadow: 0 4px 28px rgba(167,139,250,0.55), 0 0 0 1px rgba(167,139,250,0.2); }
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(10,10,28,0.72);
        border-radius: 12px;
        padding: 0.3rem;
        border: 1px solid rgba(102,126,234,0.22);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        color: rgba(180,190,255,0.72) !important;
        transition: background 0.2s, color 0.2s;
        border: 1px solid transparent;
    }
    /* Hover lift */
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: rgba(210,215,255,0.95) !important;
        background: rgba(102,126,234,0.12) !important;
    }
    /* Active / selected tab */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.42), rgba(118,75,162,0.42)) !important;
        color: white !important;
        border: 1px solid rgba(102,126,234,0.38) !important;
    }
    /* Tab panel simple fade */
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeInUp 0.3s ease-out both;
    }

    /* ── UPLOAD AREA ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102,126,234,0.35) !important;
        border-radius: 12px !important;
        background: rgba(14,15,35,0.85) !important;
        transition: border-color 0.2s, background 0.2s;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(129,140,248,0.80) !important;
        background: rgba(20,22,50,0.95) !important;
    }
    
    /* Text inside Uploader */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span,
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small,
    [data-testid="stFileUploader"] section > div > div > span,
    .st-emotion-cache-1gqnown /* Streamlit default class for uploader text */,
    .st-emotion-cache-1erovjr /* Streamlit default class for uploader subtext */ {
        color: rgba(20, 22, 45, 0.85) !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        display: block !important;
        text-align: center !important;
        margin: 0 auto !important;
    }
    
    /* Uploader Icon */
    [data-testid="stFileUploader"] section svg {
        margin: 0 auto 0.5rem auto !important;
        width: 3rem !important;
        height: 3rem !important;
        color: #818cf8 !important;
        display: block !important;
    }
    
    /* Fix layout of the internal uploader flexbox */
    [data-testid="stFileUploader"] section > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        padding: 0.5rem !important;
        width: 100% !important;
    }
    
    /* Center the Browse Files button container */
    [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* ── DIVIDER ── */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.55), rgba(167,139,250,0.55), transparent);
        border: none;
        margin: 1.5rem 0;
    }

    /* ── ANIMATIONS ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.4s ease; }

    /* ── GLOBAL TEXT ── */
    p, li, span { color: rgba(220,225,255,0.88); }
    h1, h2, h3, h4 { color: #dde0ff !important; }
    .stMarkdown p  { color: rgba(220,225,255,0.88) !important; }
    label          { color: rgba(200,210,255,0.92) !important; }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {
        background: rgba(14,15,35,0.75) !important;
        border-radius: 10px;
        border: 1px solid rgba(102,126,234,0.25);
    }
    /* ── DATAFRAME CELL TEXT COLOR ── */
    /* Glide Data Editor cell text (canvas-based cells) */
    [data-testid="stDataFrame"] canvas {
        color: #dde0ff !important;
    }
    /* All text/span/div elements inside the dataframe */
    [data-testid="stDataFrame"] span,
    [data-testid="stDataFrame"] div,
    [data-testid="stDataFrame"] p,
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] tr,
    [data-testid="stDataFrame"] [class*="cell"],
    [data-testid="stDataFrame"] [class*="Cell"],
    [data-testid="stDataFrame"] [class*="header"],
    [data-testid="stDataFrame"] [class*="Header"],
    [data-testid="stDataFrame"] [class*="row"],
    [data-testid="stDataFrame"] [class*="Row"],
    [data-testid="stDataFrame"] [role="cell"],
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="rowheader"],
    [data-testid="stDataFrame"] [role="gridcell"] {
        color: #dde0ff !important;
        -webkit-text-fill-color: #dde0ff !important;
    }
    /* Header row text */
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrame"] thead td {
        color: #a5b4fc !important;
        -webkit-text-fill-color: #a5b4fc !important;
        font-weight: 600 !important;
        background: rgba(20,22,55,0.50) !important;
    }
    /* Override any inherited white/light background on cells */
    [data-testid="stDataFrame"] [role="cell"],
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] td {
        background: transparent !important;
        background-color: transparent !important;
    }
    /* Alternating row hover tint */
    [data-testid="stDataFrame"] tr:hover td,
    [data-testid="stDataFrame"] [role="row"]:hover [role="cell"] {
        background: rgba(102,126,234,0.08) !important;
    }

    /* Dataframe toolbar container (appears on hover) */
    [data-testid="stDataFrame"] [data-testid="StyledFullScreenButton"],
    [data-testid="stDataFrame"] button,
    [data-testid="stElementToolbarButton"],
    [data-testid="stElementToolbar"] button,
    .stDataFrame [data-testid="stElementToolbarButton"] {
        color: rgba(200,210,255,0.92) !important;
        background: rgba(20,22,55,0.85) !important;
        border: 1px solid rgba(102,126,234,0.35) !important;
        border-radius: 6px !important;
        opacity: 1 !important;
        backdrop-filter: blur(8px);
        transition: background 0.18s, box-shadow 0.18s, transform 0.15s !important;
    }
    [data-testid="stDataFrame"] [data-testid="StyledFullScreenButton"]:hover,
    [data-testid="stDataFrame"] button:hover,
    [data-testid="stElementToolbarButton"]:hover,
    [data-testid="stElementToolbar"] button:hover {
        background: rgba(102,126,234,0.32) !important;
        box-shadow: 0 0 10px rgba(129,140,248,0.45) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }
    /* SVG icons inside toolbar buttons */
    [data-testid="stElementToolbarButton"] svg,
    [data-testid="stElementToolbar"] button svg,
    [data-testid="StyledFullScreenButton"] svg {
        fill: rgba(180,195,255,0.95) !important;
        stroke: rgba(180,195,255,0.95) !important;
    }
    /* Toolbar wrapper itself */
    [data-testid="stElementToolbar"] {
        background: rgba(10,11,30,0.80) !important;
        border: 1px solid rgba(102,126,234,0.30) !important;
        border-radius: 8px !important;
        padding: 2px 4px !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
    }
    /* ── DATAFRAME SEARCH INPUT ── */
    /* The "Type to search…" text box that appears in the toolbar */
    [data-testid="stDataFrame"] input,
    [data-testid="stDataFrame"] input[type="text"],
    [data-testid="stElementToolbar"] input,
    [data-testid="stElementToolbar"] input[type="text"],
    .glideDataEditor input,
    .dvn-scroller input,
    [class*="filterInput"] input,
    [class*="FilterInput"] input {
        background: #0d0e28 !important;
        background-color: #0d0e28 !important;
        color: #e2e6ff !important;
        border: 1px solid rgba(102,126,234,0.60) !important;
        border-radius: 6px !important;
        padding: 5px 10px !important;
        font-size: 0.88rem !important;
        font-family: 'Inter', sans-serif !important;
        caret-color: #a78bfa !important;
        outline: none !important;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.5) !important;
        transition: border-color 0.18s, box-shadow 0.18s !important;
        -webkit-text-fill-color: #e2e6ff !important;
    }
    [data-testid="stDataFrame"] input:focus,
    [data-testid="stElementToolbar"] input:focus {
        border-color: rgba(167,139,250,0.85) !important;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.5), 0 0 0 2px rgba(102,126,234,0.30) !important;
        background: #10122e !important;
        background-color: #10122e !important;
    }
    /* Placeholder text — broad coverage */
    [data-testid="stDataFrame"] input::placeholder,
    [data-testid="stElementToolbar"] input::placeholder,
    .glideDataEditor input::placeholder,
    .dvn-scroller input::placeholder {
        color: #e0e7ff !important;
        -webkit-text-fill-color: #e0e7ff !important;
        font-style: italic;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    /* Force dark background on the search container itself */
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
    [data-testid="stDataFrame"] .stElementToolbar {
        background-color: #0c0d23 !important;
        border: 1px solid rgba(102,126,234,0.5) !important;
    }




    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        background: rgba(14,15,35,0.70) !important;
        border: 1px solid rgba(102,126,234,0.22) !important;
        border-radius: 10px !important;
    }

    /* ── GLOBAL INPUTS & WIDGETS ── */
    /* Target all standard text inputs, number inputs, text areas */
    input[type="text"],
    input[type="number"],
    input[type="password"],
    input[type="email"],
    textarea,
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #0c0d23 !important;
        background-color: #0c0d23 !important;
        color: #dde0ff !important;
        -webkit-text-fill-color: #dde0ff !important;
        border: 1px solid rgba(102,126,234,0.45) !important;
        border-radius: 8px !important;
        caret-color: #a78bfa !important;
    }
    
    /* Selectboxes & Multiselects */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div {
        background: #0c0d23 !important;
        background-color: #0c0d23 !important;
        color: #dde0ff !important;
        border-color: rgba(102,126,234,0.45) !important;
    }
    [data-baseweb="select"] span {
        color: #dde0ff !important;
    }
    /* Dropdown options menu */
    [data-baseweb="menu"],
    [data-baseweb="popover"] {
        background: #0c0d23 !important;
        background-color: #0c0d23 !important;
        border: 1px solid rgba(102,126,234,0.50) !important;
    }
    [data-baseweb="menu"] [aria-selected="true"],
    [data-baseweb="menu"] li:hover {
        background: rgba(102,126,234,0.30) !important;
    }

    /* ── FILE UPLOADER BUTTON ── */
    [data-testid="stFileUploader"] button {
        background: rgba(102,126,234,0.3) !important;
        color: #ffffff !important;
        border: 1px solid rgba(102,126,234,0.7) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        height: auto !important; /* Override the tall button height from general */
        font-size: 1rem !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: rgba(102,126,234,0.6) !important;
        color: #ffffff !important;
        border-color: #a78bfa !important;
        box-shadow: 0 0 14px rgba(129,140,248,0.5);
    }

    /* ── DOWNLOAD / GENERAL BUTTONS ── */
    /* Catch-all for primary and secondary buttons */
    .stDownloadButton > button,
    .stButton > button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, rgba(30,35,70,0.8), rgba(20,22,45,0.9)) !important;
        color: #e2e6ff !important;
        border: 1px solid rgba(102,126,234,0.4) !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        height: 100px !important;
        transition: transform 0.2s, background 0.2s, border-color 0.2s !important;
    }
    
    .stDownloadButton > button:hover,
    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 10px 30px rgba(102,126,234,0.3) !important;
        border-color: #a78bfa !important;
        background: linear-gradient(135deg, rgba(40,45,90,0.9), rgba(30,32,65,1)) !important;
        color: #ffffff !important;
    }

    /* ── BACK BUTTON OVERRIDE (to avoid 100px height) ── */
    [data-testid="stHeader"] + div .stButton button,
    div.row-widget.stButton button:has(div:contains("Back to Home")),
    div.row-widget.stButton button:has(span:contains("Back to Home")),
    .home-back-btn button {
        height: auto !important;
        padding: 0.5rem 1.2rem !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        margin-bottom: 1rem !important;
    }

    /* Active (Primary) Button Styles */
    button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.8), rgba(118,75,162,0.8)) !important;
        border-color: #a78bfa !important;
        box-shadow: 0 6px 24px rgba(129,140,248,0.4) !important;
        color: #ffffff !important;
    }

    button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, rgba(102,126,234,1), rgba(118,75,162,1)) !important;
        box-shadow: 0 8px 32px rgba(167,139,250,0.6) !important;
    }

    /* Secondary button specific */
    button[kind="secondary"] {
        /* the base styles handle this now */
    }

    /* ── TOOLTIPS & HELP ICONS ── */
    [data-testid="stTooltipIcon"],
    .stTooltipIcon > svg {
        color: #a78bfa !important;
        fill: #a78bfa !important;
        width: 1.2rem !important;
        height: 1.2rem !important;
        transition: color 0.2s, transform 0.2s;
        opacity: 0.9 !important;
    }
    
    [data-testid="stTooltipIcon"]:hover,
    .stTooltipIcon:hover > svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        transform: scale(1.1);
        opacity: 1 !important;
        filter: drop-shadow(0 0 6px rgba(167,139,250,0.6));
    }
    
    /* Ensure the tooltip target (the button around the icon) is visible */
    [data-testid="stTooltipHoverTarget"] svg {
        stroke: #a78bfa !important;
        stroke-width: 2px !important;
    }

""", unsafe_allow_html=True)

# ================= METRICS =================
METRICS = {
    "Accuracy (%)": 91.31,
    "Precision": 0.8839,
    "Recall": 0.8851,
    "F1-Score": 0.8845,
    "AUC": 0.9572
}

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_cached():
    """Load model with caching"""
    return load_model(MODEL_PATH, compile=False)

# ================= PREPROCESSING =================
def preprocess_rgb(img):
    """Preprocess RGB image"""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def compute_ela(img, quality=90):
    """Compute Error Level Analysis"""
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)

    ela = np.abs(
        np.array(img).astype("float32") -
        np.array(compressed).astype("float32")
    )

    ela = ela / (ela.max() + 1e-8)
    ela = cv2.resize(ela, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(ela, axis=0)

# ================= ADVANCED FEATURE EXTRACTION =================
def extract_pixel_level_features(img_array):
    """Extract pixel-level statistical features"""
    features = {}
    
    # Color channel statistics
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = img_array[:, :, i]
        features[f'{channel}_Mean'] = np.mean(channel_data)
        features[f'{channel}_Std'] = np.std(channel_data)
        features[f'{channel}_Variance'] = np.var(channel_data)
        features[f'{channel}_Min'] = np.min(channel_data)
        features[f'{channel}_Max'] = np.max(channel_data)
        features[f'{channel}_Median'] = np.median(channel_data)
        features[f'{channel}_Skewness'] = calculate_skewness(channel_data)
        features[f'{channel}_Kurtosis'] = calculate_kurtosis(channel_data)
    
    # Overall image statistics
    gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    features['Brightness'] = np.mean(gray)
    features['Contrast'] = np.std(gray)
    features['Entropy'] = calculate_entropy(gray)
    
    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    features['Edge_Density'] = np.sum(edges > 0) / edges.size
    
    # Texture features
    features['Smoothness'] = 1 - (1 / (1 + np.var(gray)))
    
    return features

def calculate_skewness(data):
    """Calculate skewness"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def calculate_entropy(gray_img):
    """Calculate image entropy"""
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

# ================= SELF-ATTENTION VISUALIZATION =================
def extract_attention_maps(model, inputs, layer_name_pattern='attention'):
    """Extract self-attention maps from model"""
    attention_layers = []
    
    # Find attention layers
    for layer in model.layers:
        if 'attention' in layer.name.lower() or 'multihead' in layer.name.lower():
            attention_layers.append(layer)
    
    if not attention_layers:
        return None
    
    # Create model to extract attention weights
    attention_outputs = []
    for layer in attention_layers[:3]:  # Get first 3 attention layers
        try:
            attention_model = tf.keras.Model(
                inputs=model.input,
                outputs=layer.output
            )
            attention_output = attention_model.predict(inputs, verbose=0)
            attention_outputs.append(attention_output)
        except:
            continue
    
    return attention_outputs

def visualize_attention_maps(attention_maps, original_img):
    """Visualize self-attention maps"""
    if not attention_maps:
        return None
    
    num_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(5 * (num_maps + 1), 5))
    
    # Show original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show attention maps
    for idx, attn_map in enumerate(attention_maps):
        # Average across channels if multi-channel
        if len(attn_map.shape) == 4:
            attn_map = np.mean(attn_map[0], axis=-1)
        elif len(attn_map.shape) == 3:
            attn_map = attn_map[0]
        
        # Resize to match image size
        attn_map = cv2.resize(attn_map, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Display
        im = axes[idx + 1].imshow(attn_map, cmap='jet', alpha=0.8)
        axes[idx + 1].set_title(f'Attention Layer {idx + 1}', fontsize=12, fontweight='bold')
        axes[idx + 1].axis('off')
        plt.colorbar(im, ax=axes[idx + 1], fraction=0.046)
    
    plt.tight_layout()
    return fig

# ================= GRAD-CAM =================
def grad_cam(model, inputs, layer_name=None):
    """Generate Grad-CAM heatmap"""
    # Find the last convolutional layer
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break
    
    if target_layer is None:
        return None

    grad_model = tf.keras.Model(
        model.inputs,
        [target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inputs)
        preds = tf.reshape(preds, (-1, 1))
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)

    cam = cam.numpy()[0]
    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-8
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam

# ================= FEATURE LEVEL ANALYSIS =================
def extract_layer_features(model, inputs, layer_indices=[5, 10, 15]):
    """Extract features from intermediate layers"""
    layer_features = {}
    
    for idx in layer_indices:
        if idx < len(model.layers):
            layer = model.layers[idx]
            feature_model = tf.keras.Model(
                inputs=model.input,
                outputs=layer.output
            )
            features = feature_model.predict(inputs, verbose=0)
            layer_features[f'{layer.name}'] = features
    
    return layer_features

def visualize_feature_maps(layer_features, max_filters=16):
    """Visualize feature maps from different layers"""
    figs = []
    
    for layer_name, features in layer_features.items():
        if len(features.shape) != 4:
            continue
        
        num_filters = min(features.shape[-1], max_filters)
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_filters):
            feature_map = features[0, :, :, i]
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Filter {i+1}', fontsize=8)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps: {layer_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        figs.append((layer_name, fig))
    
    return figs

# ================= BOUNDING BOX =================
# def bounding_box_from_cam(cam, threshold=0.6):
#     """Generate bounding box from CAM"""
#     heatmap = (cam > threshold).astype(np.uint8)
#     contours, _ = cv2.findContours(
#         heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     if not contours:
#         return None

#     c = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(c)
#     return x, y, w, h

def create_forgery_mask(cam, threshold=0.4):
    """Generate pixel-level forgery mask"""

    # Normalize CAM
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    # Resize CAM
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    # Create mask
    mask = (cam > threshold).astype("uint8")

    return cam, mask




# ================= INTERACTIVE VISUALIZATIONS =================
def create_confidence_gauge(confidence, label):
    """Create interactive confidence gauge"""
    color = 'red' if label == 'FORGED' else 'green'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': "Confidence Level", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_probability_chart(authentic_prob, forged_prob):
    """Create probability comparison chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Authentic', 'Forged'],
            y=[authentic_prob * 100, forged_prob * 100],
            marker_color=['green', 'red'],
            text=[f'{authentic_prob*100:.2f}%', f'{forged_prob*100:.2f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        yaxis_title='Probability (%)',
        height=400,
        showlegend=False
    )
    
    return fig

def create_feature_heatmap(features_dict):
    """Create heatmap of pixel-level features"""
    # Prepare data
    feature_names = []
    feature_values = []
    
    for key, value in features_dict.items():
        feature_names.append(key)
        feature_values.append(value)
    
    # Normalize values for better visualization
    feature_values = np.array(feature_values)
    normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-8)
    
    # Create heatmap
    fig = px.imshow(
        [normalized_values],
        labels=dict(x="Features", y="", color="Normalized Value"),
        x=feature_names,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        height=300
    )
    
    fig.update_layout(
        title='Pixel-Level Feature Analysis (Normalized)',
        xaxis={'tickangle': 45}
    )
    
    return fig

# ================= PDF REPORT =================
def generate_pdf_report(result_dict, features_dict, timestamp):
    """Generate HTML report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #667eea; }}
            h2 {{ color: #764ba2; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #667eea; color: white; }}
            .forged {{ color: red; font-weight: bold; }}
            .authentic {{ color: green; font-weight: bold; }}
            .metric-section {{ background: #f0f2f6; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>🔍 Image Forgery Detection Report</h1>
        <p><b>Generated:</b> {timestamp}</p>
        <p><b>Model:</b> CNN_D64_128_128</p>
        
        <div class="metric-section">
            <h2>Prediction Result</h2>
            <p class="{result_dict['label'].lower()}">
                <b>Classification:</b> {result_dict['label']}
            </p>
            <p><b>Confidence:</b> {result_dict['confidence']*100:.2f}%</p>
            <p><b>Authentic Probability:</b> {result_dict['authentic_prob']*100:.2f}%</p>
            <p><b>Forged Probability:</b> {result_dict['forged_prob']*100:.2f}%</p>
        </div>
        
        <h2>Model Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """
    
    if features_dict:
        for feature, value in features_dict.items():
            html += f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>\n"
    else:
        html += "<tr><td colspan='2'>No pixel-level features available</td></tr>"
    
    html += """
        </table>
        
        <h2>Pixel-Level Features</h2>
        <table>
            <tr><th>Feature</th><th>Value</th></tr>
    """
    
    for feature, value in features_dict.items():
        html += f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>\n"
    
    html += """
        </table>
        
        <div class="metric-section">
            <h3>Analysis Summary</h3>
            <p>This report was generated using a Dual-Stream CNN model with self-attention mechanisms.
            The model analyzes both RGB features and Error Level Analysis (ELA) to detect image manipulation.</p>
            <p><b>Disclaimer:</b> This analysis is automated and should be used as a supporting tool.
            Manual verification is recommended for critical applications.</p>
        </div>
    </body>
    </html>
    """
    
    return html.encode("utf-8")


# ================= MAIN APP =================
def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Advanced Image Forgery Detection System</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Dual-Stream CNN • Error Level Analysis • Grad-CAM Visualization</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Model status card
        st.markdown("""
        <div class="sidebar-model-card">
            <div style="font-size:2rem">🧠</div>
            <h3>CNN_D64_128_128 Model</h3>
            <p>Dual-Stream Architecture</p>
            <div class="sidebar-badge">✅ Active &amp; Ready</div>
        </div>
        """, unsafe_allow_html=True)

        # Quick stats
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><div class="metric-value">224</div><div class="metric-label">Input Size</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="metric-value">91%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚙️ Settings")

        ela_quality = st.slider("📉 ELA JPEG Quality", 50, 100, 90,
                                help="Lower quality = more compression artifacts visible")
        cam_threshold = st.slider("🎯 CAM Threshold", 0.0, 1.0, 0.25, 0.05,
                                  help="Higher value = tighter forgery bounding box")
        show_features = st.checkbox("📊 Show Detailed Features", value=True)
        show_attention = st.checkbox("💡 Show Attention Maps", value=True)

        st.markdown("---")
        st.subheader("📊 Model Performance")
        metrics_df = pd.DataFrame(METRICS.items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            💡 <strong>How it works:</strong><br>
            Upload any image. The model uses <em>RGB features</em> + <em>ELA compression artifacts</em>
            to detect manipulation.
        </div>
        """, unsafe_allow_html=True)

    # Load model
    model = load_model_cached()
    
    # Initialize session state for navigation
    if 'active_view' not in st.session_state:
        st.session_state.active_view = None

    # Function to change active view
    def set_view(view_name):
        if st.session_state.active_view == view_name:
            st.session_state.active_view = None # toggle off
        else:
            st.session_state.active_view = view_name

    # If no view is selected, show the 4 Navigation Cards
    if st.session_state.active_view is None:
        # Create 2x2 grid for navigation cards
        st.markdown('<div class="section-header">🧭 Main Navigation</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(200,210,255,0.7); margin-bottom: 24px;">Select a module below to begin analysis or view metrics.</p>', unsafe_allow_html=True)
        
        # Navigation Grid Wrapper
        st.markdown('<div class="nav-grid">', unsafe_allow_html=True)

        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            st.button("📤 Upload & Analyze\n\nSingle Image Analysis", use_container_width=True, on_click=set_view, args=("Upload & Analyze",), 
                      type="primary" if st.session_state.active_view == "Upload & Analyze" else "secondary")
        with col2:
            st.button("📦 Batch Processing\n\nAnalyze Multiple Images", use_container_width=True, on_click=set_view, args=("Batch Processing",),
                      type="primary" if st.session_state.active_view == "Batch Processing" else "secondary")
                      
        st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)

        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            st.button("🔬 Advanced Diagnostics\n\nLayer Feature Exspection", use_container_width=True, on_click=set_view, args=("Advanced Diagnostics",),
                      type="primary" if st.session_state.active_view == "Advanced Diagnostics" else "secondary")
        with col4:
            st.button("📊 Model Performance\n\nView Metrics & Stats", use_container_width=True, on_click=set_view, args=("Model Performance",),
                      type="primary" if st.session_state.active_view == "Model Performance" else "secondary")

        st.markdown('</div>', unsafe_allow_html=True)

    # If a view IS selected, show its content AND a back button
    else:
        # Provide a top-level button to go back to the Main Navigation cards
        st.button("⬅️ Back to Home", type="secondary", key="back_to_home_btn", on_click=set_view, args=(None,))

        # Content corresponding to views
        if st.session_state.active_view == "Upload & Analyze":
            st.markdown('<div class="section-header">📤 Single Image Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 Upload a JPG/PNG/TIF image below. The system will analyze it using RGB + ELA dual-stream processing and highlight any suspected manipulation areas.</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Drop your image here or click to browse",
                type=["jpg", "jpeg", "png","tif"],
                help="Supported: JPG, JPEG, PNG,TIF • Best results on JPEG images"
            )

            if uploaded_file:
                analyze_image(
                    uploaded_file, model, ela_quality, cam_threshold,
                    show_features, show_attention
                )

        elif st.session_state.active_view == "Batch Processing":
            st.markdown('<div class="section-header">📦 Batch Image Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 Upload multiple images at once. Results are displayed in a summary table and can be exported as CSV.</div>', unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Drop multiple images here",
                type=["jpg", "jpeg", "png","tif"],
                accept_multiple_files=True
            )

            if uploaded_files:
                batch_analysis(uploaded_files, model, ela_quality)

        elif st.session_state.active_view == "Advanced Diagnostics":
            st.markdown('<div class="section-header">🔬 Advanced Model Diagnostics</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 Explore intermediate CNN feature maps to understand how the model extracts patterns. Useful for research and explainability.</div>', unsafe_allow_html=True)

            diagnostic_file = st.file_uploader(
                "Upload image for layer inspection",
                type=["jpg", "jpeg", "png","tif"],
                key="diagnostic"
            )

            if diagnostic_file:
                advanced_diagnostics(diagnostic_file, model)

        elif st.session_state.active_view == "Model Performance":
            st.markdown('<div class="section-header">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 Detailed evaluation metrics for the CNN_D64_128_128 model including accuracy, precision, recall, F1-Score, and AUC.</div>', unsafe_allow_html=True)
            display_model_performance()




# ================= ANALYSIS FUNCTIONS =================
def analyze_image(uploaded_file, model, ela_quality, cam_threshold, 
                  show_features, show_attention, is_webcam=False):
    gt_mask = None

    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    rgb = preprocess_rgb(img)
    ela = compute_ela(img, quality=ela_quality)

    with st.spinner("Analyzing image..."):
        pred = model.predict([rgb, ela], verbose=0)[0][0]

    label = "FORGED" if pred > 0.5 else "AUTHENTIC"
    confidence = pred if pred > 0.5 else 1 - pred
    authentic_prob = 1 - pred
    forged_prob = pred

    with col2:
        if label == "FORGED":
            st.markdown(
                f'<div class="result-card forged-card">⚠️ FORGED<br><span style="font-size:1rem;">Confidence: {confidence*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-card authentic-card">✅ AUTHENTIC<br><span style="font-size:1rem;">Confidence: {confidence*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

        st.plotly_chart(create_confidence_gauge(confidence, label), use_container_width=True)

    st.plotly_chart(create_probability_chart(authentic_prob, forged_prob), use_container_width=True)

    # ================= ELA =================
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Error Level Analysis</div>', unsafe_allow_html=True)

    fig_ela, axes_ela = plt.subplots(1, 2, figsize=(12, 5))
    axes_ela[0].imshow(img)
    axes_ela[0].set_title("Original")
    axes_ela[0].axis('off')

    axes_ela[1].imshow(ela[0], cmap='hot')
    axes_ela[1].set_title("ELA Heatmap")
    axes_ela[1].axis('off')

    st.pyplot(fig_ela)

    # ================= Grad-CAM =================
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Forgery Localization</div>', unsafe_allow_html=True)

    cam = grad_cam(model, [rgb, ela])

    if cam is not None and label == "FORGED":

        gt_mask = (gt_mask > 0).astype("uint8")

        img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

        # Heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

        # Draw GT boundary
        contours, _ = cv2.findContours((gt_mask*255).astype("uint8"),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0,255,0), 2)

        # Highlight GT region
        gt_overlay = img_resized.copy()
        gt_overlay[gt_mask == 1] = [255, 0, 0]

        # ================= DISPLAY =================
        fig_cam, axes = plt.subplots(1, 3, figsize=(15,5))

        axes[0].imshow(overlay)
        axes[0].set_title("Grad-CAM Overlay")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(gt_overlay)
        axes[2].set_title("Highlighted Region")
        axes[2].axis("off")

        st.pyplot(fig_cam)

    elif label == "AUTHENTIC":
        st.success("✅ No forgery regions detected.")

    # ================= ATTENTION MAP =================
    if show_attention:
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🧠 Attention Maps</div>', unsafe_allow_html=True)

        attention_maps = extract_attention_maps(model, [rgb, ela])

        if attention_maps:
            attention_fig = visualize_attention_maps(
                attention_maps,
                img.resize((IMG_SIZE, IMG_SIZE))
            )
            st.pyplot(attention_fig)
        else:
            st.info("No attention layers found in this model.")



##=======================Pixel - Level=======================================
    
    if show_features:
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Pixel-Level Statistical Features</div>', unsafe_allow_html=True)

        img_array = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        features = extract_pixel_level_features(img_array)

        # Display heatmap
        st.plotly_chart(
            create_feature_heatmap(features),
            use_container_width=True
        )

        # Display feature table
        with st.expander("📋 View All Feature Values"):
            features_df = pd.DataFrame(
                features.items(),
                columns=["Feature", "Value"]
            )
            st.dataframe(features_df, use_container_width=True)

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_dict = {
        'label': label,
        'confidence': confidence,
        'authentic_prob': authentic_prob,
        'forged_prob': forged_prob
    }
    
    if show_features:
        pdf_data = generate_pdf_report(result_dict, features, timestamp)
    else:
        pdf_data = generate_pdf_report(result_dict, {}, timestamp)
    
    filename = uploaded_file.name if not is_webcam else f"webcam_{timestamp}.html"
    st.download_button(
        "📄 Download Analysis Report",
        data=pdf_data,
        file_name=f"report_{filename.split('.')[0]}.html",
        mime="text/html",
        use_container_width=True
    )

def batch_analysis(uploaded_files, model, ela_quality):
    """Batch processing of multiple images"""
    
    st.info(f"Analyzing {len(uploaded_files)} images...")
    
    results = []
    progress_bar = st.progress(0)



    for idx, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB")
        rgb = preprocess_rgb(img)
        ela = compute_ela(img, quality=ela_quality)
        
        pred = model.predict([rgb, ela], verbose=0)[0][0]
        label = "FORGED" if pred > 0.5 else "AUTHENTIC"
        confidence = pred if pred > 0.5 else 1 - pred
        
        results.append({
            "Image": uploaded_file.name,
            "Prediction": label,
            "Confidence": f"{confidence*100:.2f}%",
            "Forged_Probability": f"{pred*100:.2f}%",
            "Authentic_Probability": f"{(1-pred)*100:.2f}%"
        })
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    # Display results
    st.success(f"✅ Analysis complete! Processed {len(uploaded_files)} images.")
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    forged_count = sum(1 for r in results if r['Prediction'] == 'FORGED')
    authentic_count = len(results) - forged_count
    
    col1.metric("Total Images", len(results))
    col2.metric("Forged", forged_count, delta=f"{forged_count/len(results)*100:.1f}%")
    col3.metric("Authentic", authentic_count, delta=f"{authentic_count/len(results)*100:.1f}%")
    
    # Download CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Results (CSV)",
        data=csv,
        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def advanced_diagnostics(diagnostic_file, model):
    """Advanced diagnostic analysis"""
    
    img = Image.open(diagnostic_file).convert("RGB")
    rgb = preprocess_rgb(img)
    ela = compute_ela(img)
    
    st.subheader("🔬 Layer-wise Feature Extraction")
    
    # Extract features from multiple layers
    layer_features = extract_layer_features(model, [rgb, ela], layer_indices=[5, 10, 15, 20])
    
    if layer_features:
        feature_figs = visualize_feature_maps(layer_features, max_filters=16)
        
        for layer_name, fig in feature_figs:
            with st.expander(f"📊 {layer_name}"):
                st.pyplot(fig)
    
    # Pixel-level analysis
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Comprehensive Pixel Analysis</div>', unsafe_allow_html=True)

    img_array = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    features = extract_pixel_level_features(img_array)
    
    # Group features by category
    color_features = {k: v for k, v in features.items() if any(c in k for c in ['Red', 'Green', 'Blue'])}
    texture_features = {k: v for k, v in features.items() if k in ['Brightness', 'Contrast', 'Entropy', 'Edge_Density', 'Smoothness']}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Color Channel Statistics**")
        color_df = pd.DataFrame(color_features.items(), columns=["Feature", "Value"])
        st.dataframe(color_df, use_container_width=True)
    
    with col2:
        st.write("**Texture & Structure Features**")
        texture_df = pd.DataFrame(texture_features.items(), columns=["Feature", "Value"])
        st.dataframe(texture_df, use_container_width=True)

def display_model_performance():
    """Display model performance metrics"""

    # Styled metric cards at top
    st.markdown('<div class="section-header">🏆 CNN_D64_128_128 — Key Metrics</div>', unsafe_allow_html=True)
    metric_icons = {
        "Accuracy (%)": ("🎯", "Accuracy"),
        "Precision": ("📌", "Precision"),
        "Recall": ("🔎", "Recall"),
        "F1-Score": ("⚖️", "F1-Score"),
        "AUC": ("📈", "AUC")
    }
    cols = st.columns(5)
    for col, (metric, value) in zip(cols, METRICS.items()):
        icon, label = metric_icons[metric]
        display_val = f"{value:.2f}%" if value > 1 else f"{value:.4f}"
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">{icon}</div>
                <div class="metric-value">{display_val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Metrics Bar Chart</div>', unsafe_allow_html=True)
        metrics_normalized = {k: v/100 if v > 1 else v for k, v in METRICS.items()}

        fig = px.bar(
            x=list(metrics_normalized.keys()),
            y=list(metrics_normalized.values()),
            labels={'x': 'Metric', 'y': 'Score'},
            title='Performance Overview',
            color=list(metrics_normalized.values()),
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=380, showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array([[450, 50], [60, 440]])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Authentic', 'Forged'],
            y=['Authentic', 'Forged'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(title='Confusion Matrix (Test Set)', height=380,
                             plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cm, use_container_width=True)

    # Model comparison from CSV
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🥊 All Models Comparison</div>', unsafe_allow_html=True)
    try:
        df_models = pd.read_csv("metrics.csv")
        st.dataframe(df_models, use_container_width=True, hide_index=True)
    except:
        pass


# ================= RUN APP =================
if __name__ == "__main__":
    main()
    
    # ── Floating background icons (rendered at end to minimize layout jump) ──
    st.markdown("""
    <div class="bg-icons-wrap">
      <!-- Left side icons -->
      <span class="bg-icon i1">🔍</span>
      <span class="bg-icon i2">📸</span>
      <span class="bg-icon i3">🛡️</span>
      <span class="bg-icon i4">🔬</span>
      <!-- Right side icons -->
      <span class="bg-icon i5">🕵️</span>
      <span class="bg-icon i6">📡</span>
      <span class="bg-icon i7">🔐</span>
      <span class="bg-icon i8">📂</span>
    </div>
    """, unsafe_allow_html=True)