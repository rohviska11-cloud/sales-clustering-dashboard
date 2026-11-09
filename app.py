# app.py
# Dashboard analisis transaksi mingguan dan clustering deret waktu.
# File data: Sales transaction data.xlsx (root repository)
# Matriks jarak: GitHub Release v1.0.0 / all_distance_matrices.npz

from pathlib import Path
from io import BytesIO
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from tslearn.clustering import TimeSeriesKMeans

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Sales Analytics | Weekly Transactions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. THEME: WARM CORPORATE / EXECUTIVE
# =============================================================================
st.markdown("""
<style>
:root {
  --navy: #1B2A4A;
  --navy2: #263B60;
  --cream: #F5F2EB;
  --paper: #FFFEFB;
  --teal: #2A817A;
  --gold: #C49A52;
  --ink: #263247;
  --muted: #687386;
  --line: #E7E2D8;
}
html, body, [class*="css"] {
  font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--ink);
}
.stApp {
  background: linear-gradient(145deg, #F5F2EB 0%, #F8F7F3 48%, #EEF3F1 100%);
}
[data-testid="stSidebar"] {
  background: var(--navy);
  border-right: 1px solid #344563;
}
[data-testid="stSidebar"] * { color: var(--cream); }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small { color: #F5F2EB !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
  background: #FFFEFB !important;
  border: 1px solid #D8D2C6 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input {
  color: #263247 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background: var(--gold);
}
.hero {
  background: linear-gradient(112deg, #1B2A4A 0%, #263B60 68%, #2A817A 120%);
  border: 1px solid rgba(245,242,235,.18);
  border-radius: 18px;
  padding: 27px 32px;
  margin: 4px 0 24px 0;
  box-shadow: 0 10px 26px rgba(27,42,74,.13);
}
.hero-kicker {
  color: #D9C59F;
  font-size: 11px;
  letter-spacing: 2.1px;
  font-weight: 700;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.hero h1 {
  color: #F5F2EB !important;
  font-size: 31px;
  line-height: 1.18;
