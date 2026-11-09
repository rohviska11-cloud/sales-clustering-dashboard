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
  font-weight: 650;
  margin: 0 0 9px 0;
}
.hero p {
  color: #E8E5DC !important;
  font-size: 15px;
  margin: 0;
  max-width: 850px;
}
.hero-mark {
  float: right;
  color: #D9C59F;
  font-size: 38px;
  opacity: .95;
  padding: 8px 10px 0 20px;
}
.page-intro {
  color: var(--muted);
  font-size: 14px;
  margin-top: -10px;
  margin-bottom: 18px;
}
.section-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 15px;
  padding: 17px 19px;
  margin: 9px 0 15px 0;
  box-shadow: 0 3px 12px rgba(27,42,74,.035);
}
div[data-testid="stMetric"] {
  background: #FFFEFB;
  border: 1px solid #E4DED2;
  border-top: 3px solid #C49A52;
  border-radius: 12px;
  padding: 13px 15px;
  box-shadow: 0 3px 10px rgba(27,42,74,.035);
}
div[data-testid="stMetricLabel"] { color: #687386 !important; }
div[data-testid="stMetricValue"] { color: #1B2A4A !important; }
h1, h2, h3 { color: #1B2A4A; letter-spacing: -.35px; }
h2 { font-size: 25px !important; font-weight: 650 !important; }
h3 { font-size: 19px !important; font-weight: 650 !important; }
div[data-testid="stDataFrame"] { border: 1px solid #E7E2D8; border-radius: 10px; }
.stButton > button, .stDownloadButton > button {
  background: #1B2A4A;
  color: #F5F2EB;
  border: 1px solid #1B2A4A;
  border-radius: 9px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #2A817A; color: white; border-color: #2A817A;
}
div[data-testid="stAlert"] { border-radius: 11px; }
hr { border-color: #E7E2D8; }
.caption-note { color: #687386; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. PATHS, DATA, AND MATRIX LOADING
# =============================================================================
BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE / "Sales transaction data.xlsx",
    BASE / "Sales transaction data.csv",
    BASE / "sales_transaction_data.xlsx",
    BASE / "sales_transaction_data.csv",
]
MATRIX_URL = (
    "https://github.com/rohviska11-cloud/sales-clustering-dashboard/"
    "releases/download/v1.0.0/all_distance_matrices.npz"
)

@st.cache_data(show_spinner=False)
def load_sales(file_bytes, suffix):
    bio = BytesIO(file_bytes)
    df = pd.read_excel(bio) if suffix in (".xlsx", ".xls") else pd.read_csv(bio)
    weekly = [f"W{i}" for i in range(52)]
    if "Product_Code" not in df.columns:
        raise ValueError("Kolom Product_Code tidak ditemukan.")
    missing = [c for c in weekly if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom mingguan belum lengkap: {missing}")
    df = df.drop_duplicates("Product_Code").copy()
    df[weekly] = df[weekly].apply(pd.to_numeric, errors="coerce")
    df[weekly] = df[weekly].replace([np.inf, -np.inf], np.nan)
    df[weekly] = df[weekly].fillna(df[weekly].median()).fillna(0)
    return df, weekly

@st.cache_resource(show_spinner="Mengambil matriks jarak dari GitHub Releases (sekali saat cache kosong)...")
def load_matrices_from_url(url):
    # allow_pickle=True digunakan karena arsip eksperimen juga menyimpan product_code
    # sebagai object array. Gunakan hanya untuk file milik sendiri dari GitHub Release.
    response = requests.get(url, timeout=(30, 600))
    response.raise_for_status()
    archive = np.load(BytesIO(response.content), allow_pickle=True)
    result = {key: archive[key] for key in archive.files if key != "product_code"}
    if "product_code" in archive.files:
        result["__product_code__"] = archive["product_code"].astype(str)
    return result

@st.cache_data(show_spinner=False)
def make_mds(D):
    model = MDS(
        n_components=5, metric=True, dissimilarity="precomputed",
        random_state=42, n_init=1, max_iter=250
    )
    return model.fit_transform(D)

def row_minmax(A):
    lo = np.nanmin(A, axis=1, keepdims=True)
    hi = np.nanmax(A, axis=1, keepdims=True)
    span = hi - lo
    return np.divide(A - lo, span, out=np.zeros_like(A), where=span != 0)

def derivative_ts(A):
    A = np.asarray(A, dtype=float)
    out = np.zeros_like(A)
    out[:, 0] = A[:, 1] - A[:, 0]
    for t in range(1, A.shape[1] - 1):
        out[:, t] = 0.5 * (A[:, t] - A[:, t-1]) + 0.25 * (A[:, t+1] - A[:, t-1])
    out[:, -1] = A[:, -1] - A[:, -2]
    return out

def matrix_key(rep_key, distance, radius=None, g=None):
    if distance == "DTW":
        return f"DTW_{rep_key}"
    if distance == "CDTW":
        return f"CDTW_{rep_key}_R{radius}"
    if distance == "DDTW":
        return f"DDTW_{rep_key}"
    return f"WDTW_{rep_key}_g{g}"

def balance_score(labels):
    """Normalized entropy across non-noise clusters; 1 means equal-sized clusters."""
    y = np.asarray(labels)
    y = y[y != -1]
    vals, counts = np.unique(y, return_counts=True)
    k = len(vals)
    if k <= 1:
        return 0.0, {}
    p = counts / counts.sum()
    contrib = -(p * np.log(p)) / np.log(k)
    return float(contrib.sum()), {int(v): float(c) for v, c in zip(vals, contrib)}

def kmedoids_pam(D, k, seed=42, max_iter=100):
    """PAM sederhana dengan matriks jarak precomputed."""
    D = np.asarray(D, dtype=float)
    n = len(D)
    if k >= n:
        raise ValueError("K harus lebih kecil dari jumlah produk.")
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(n))]
    while len(medoids) < k:
        nearest = np.min(D[:, medoids], axis=1)
        nearest[medoids] = 0
        weights = nearest ** 2
        weights[medoids] = 0
        if weights.sum() <= 0:
            medoids.append(int(rng.choice(np.setdiff1d(np.arange(n), medoids))))
        else:
            nxt = int(rng.choice(n, p=weights / weights.sum()))
            if nxt in medoids:
                nxt = int(rng.choice(np.setdiff1d(np.arange(n), medoids)))
            medoids.append(nxt)
    medoids = np.asarray(medoids, dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members):
                block = D[np.ix_(members, members)]
                new[c] = members[np.argmin(block.sum(axis=1))]
        if np.array_equal(new, medoids):
            break
        medoids = new
    return np.argmin(D[:, medoids], axis=1), medoids

def run_cluster(algo, D, X, X_deriv, k, linkage_method, eps, min_samples, key):
    t0 = time.perf_counter()
    centers = None
    medoid_indices = None
    if algo == "K-Means DBA":
        if key.startswith("WDTW"):
            raise ValueError("K-Means DBA untuk WDTW belum tersedia di tslearn.")
        Xfit = X_deriv if key.startswith("DDTW") else X
        params = {}
        if key.startswith("CDTW"):
            params = {
                "global_constraint": "sakoe_chiba",
                "sakoe_chiba_radius": int(key.rsplit("R", 1)[1])
            }
        model = TimeSeriesKMeans(
            n_clusters=k, metric="dtw", metric_params=params,
            max_iter=10, max_iter_barycenter=10, n_init=2,
            random_state=42, n_jobs=-1
        )
        labels = model.fit_predict(Xfit[:, :, None])
        centers = model.cluster_centers_[:, :, 0]
        center_type = "Centroid DBA"
    elif algo == "K-Medoids PAM":
        labels, medoid_indices = kmedoids_pam(D, k)
        centers = X[medoid_indices]
        center_type = "Medoid aktual"
    elif algo == "Hierarchical":
        Z = linkage(squareform(D, checks=False), method=linkage_method)
        labels = cut_tree(Z, n_clusters=[k]).ravel()
        center_type = "Median profil anggota"
    else:
        labels = DBSCAN(
            eps=float(eps), min_samples=int(min_samples),
            metric="precomputed"
        ).fit_predict(D)
        center_type = "Median profil anggota"
    return labels, centers, medoid_indices, center_type, time.perf_counter() - t0

def evaluate_labels(D, coords, labels):
    labels = np.asarray(labels)
    keep = labels != -1
    y = labels[keep]
    unique = np.unique(y)
    si = dbi = chi = np.nan
    if 2 <= len(unique) < len(y):
        si = silhouette_score(D[np.ix_(keep, keep)], y, metric="precomputed")
        dbi = davies_bouldin_score(coords[keep], y)
        chi = calinski_harabasz_score(coords[keep], y)
    bs, contributions = balance_score(labels)
    return si, dbi, chi, bs, len(unique), float((labels == -1).mean() * 100), contributions

def representative_index(D, member_idx):
    if len(member_idx) == 1:
        return int(member_idx[0])
    block = D[np.ix_(member_idx, member_idx)]
    return int(member_idx[np.argmin(block.sum(axis=1))])

# =============================================================================
# 3. SIDEBAR AND NAVIGATION
# =============================================================================
st.sidebar.markdown(
    "<div style='font-size:11px;letter-spacing:1.8px;color:#D9C59F;font-weight:700;'>SALES ANALYTICS</div>"
    "<div style='font-size:22px;font-weight:700;color:#F5F2EB;margin:5px 0 3px;'>Weekly Transactions</div>"
    "<div style='font-size:12px;color:#D9DFE8;margin-bottom:18px;'>Exploration & time-series clustering</div>",
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "NAVIGASI",
    ["Eksplorasi Data", "Hasil Klasterisasi", "Rekomendasi Metode", "Panduan Metode"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("Pengaturan model digunakan pada halaman klasterisasi dan rekomendasi.")
rep_label = st.sidebar.selectbox("Representasi data", ["Data asli", "MinMax", "Scaled"])
rep_key = {"Data asli": "Original", "MinMax": "MinMax", "Scaled": "Scaled"}[rep_label]
distance = st.sidebar.selectbox("Metode jarak", ["DTW", "CDTW", "DDTW", "WDTW"])
radius = st.sidebar.selectbox("Radius CDTW", [2, 3, 5]) if distance == "CDTW" else None
g = st.sidebar.selectbox("Parameter g — WDTW", [0.05, 0.1, 0.5, 1.0]) if distance == "WDTW" else None
algorithm = st.sidebar.selectbox(
    "Algoritma klaster",
    ["K-Means DBA", "K-Medoids PAM", "Hierarchical", "DBSCAN"]
)
k = st.sidebar.slider("Jumlah cluster (K)", 2, 10, 3) if algorithm != "DBSCAN" else None
linkage_method = st.sidebar.selectbox("Linkage", ["average", "complete"]) if algorithm == "Hierarchical" else "average"
eps = st.sidebar.number_input("DBSCAN — eps", min_value=0.000001, value=1.0, step=0.1, format="%.6f") if algorithm == "DBSCAN" else None
min_samples = st.sidebar.select_slider("DBSCAN — min_samples", options=[5, 10, 15, 20], value=10) if algorithm == "DBSCAN" else None

uploaded = st.sidebar.file_uploader("Dataset alternatif (.xlsx/.csv)", type=["xlsx", "xls", "csv"])

# =============================================================================
# 4. LOAD DATASET
# =============================================================================
source = uploaded
if source is None:
    source = next((p for p in DATA_CANDIDATES if p.exists()), None)
if source is None:
    st.error("Dataset tidak ditemukan. Pastikan Sales transaction data.xlsx berada di root repository atau unggah melalui sidebar.")
    st.stop()
suffix = Path(source.name).suffix.lower() if uploaded else source.suffix.lower()
raw = source.getvalue() if uploaded else source.read_bytes()
try:
    data, weeks = load_sales(raw, suffix)
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

codes = data["Product_Code"].astype(str).to_numpy()
X_raw = data[weeks].to_numpy(dtype=float)
X_raw = np.nan_to_num(X_raw, nan=0, posinf=0, neginf=0)

if rep_key == "Original":
    X = X_raw.copy()
elif rep_key == "MinMax":
    norm_cols = [f"Normalized {i}" for i in range(52)]
    if all(col in data.columns for col in norm_cols):
        X = data[norm_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    else:
        X = row_minmax(X_raw)
else:
    X = StandardScaler().fit_transform(X_raw)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_deriv = derivative_ts(X)

# =============================================================================
# 5. HERO
# =============================================================================
st.markdown("""
<div class="hero">
  <div class="hero-mark">▥</div>
  <div class="hero-kicker">DATA INTELLIGENCE · WEEKLY SALES</div>
  <h1>Sales Time Series Analytics</h1>
  <p>Eksplorasi karakteristik transaksi mingguan dan segmentasi produk berdasarkan kemiripan pola penjualan.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE 1: EXPLORATORY ANALYSIS
# =============================================================================
if page == "Eksplorasi Data":
    st.header("Eksplorasi Awal Data")
    st.markdown('<div class="page-intro">Memahami skala transaksi, dinamika mingguan, dan karakteristik produk sebelum membentuk cluster.</div>', unsafe_allow_html=True)

    n_products = len(data)
    total = float(X_raw.sum())
    avg_product = float(X_raw.mean())
    weekly_total = X_raw.sum(axis=0)
    weekly_mean = X_raw.mean(axis=0)
    peak = int(np.argmax(weekly_total))
    low = int(np.argmin(weekly_total))
    product_total = X_raw.sum(axis=1)
    product_mean = X_raw.mean(axis=1)
    product_sd = X_raw.std(axis=1)
    zero_share = float((X_raw == 0).mean() * 100)
    cv_products = np.divide(product_sd, product_mean, out=np.zeros_like(product_sd), where=product_mean != 0)

    a, b, c, d = st.columns(4)
    a.metric("Jumlah produk", f"{n_products:,}")
    b.metric("Total transaksi", f"{total:,.0f}")
    c.metric("Rata-rata transaksi", f"{avg_product:.2f}", "per produk per minggu")
    d.metric("Proporsi nilai nol", f"{zero_share:.1f}%", "dari seluruh sel produk-minggu")

    st.subheader("Dinamika transaksi mingguan")
    trend_df = pd.DataFrame({
        "Minggu": weeks,
        "Total transaksi": weekly_total,
        "Rata-rata per produk": weekly_mean
    })
    mode = st.radio("Tampilkan ukuran", ["Total transaksi", "Rata-rata per produk"], horizontal=True)
    fig = px.line(
        trend_df, x="Minggu", y=mode, markers=True,
        color_discrete_sequence=["#2A817A"]
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig.update_layout(template="plotly_white", height=360, hovermode="x unified",
                      margin=dict(l=15, r=15, t=20, b=15))
    st.plotly_chart(fig, use_container_width=True)
    q1, q2, q3 = st.columns(3)
    q1.metric("Minggu tertinggi", f"W{peak}", f"{weekly_total[peak]:,.0f} transaksi")
    q2.metric("Minggu terendah", f"W{low}", f"{weekly_total[low]:,.0f} transaksi")
    q3.metric("Rentang total mingguan", f"{weekly_total.max()-weekly_total.min():,.0f}")

    st.divider()
    st.subheader("Karakteristik produk")
    left, right = st.columns(2)
    top_order = np.argsort(product_total)[::-1]
    with left:
        st.markdown("**10 produk dengan total transaksi tertinggi**")
        top = pd.DataFrame({"Product_Code": codes[top_order[:10]],
                            "Total transaksi": product_total[top_order[:10]].astype(int)})
        st.dataframe(top, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**10 produk dengan total transaksi terendah**")
        bottom_idx = np.argsort(product_total)[:10]
        bottom = pd.DataFrame({"Product_Code": codes[bottom_idx],
                               "Total transaksi": product_total[bottom_idx].astype(int)})
        st.dataframe(bottom, use_container_width=True, hide_index=True)

    dist_col, vol_col = st.columns(2)
    with dist_col:
        st.markdown("**Distribusi total transaksi per produk**")
        hist = px.histogram(pd.DataFrame({"Total transaksi": product_total}),
                            x="Total transaksi", nbins=35,
                            color_discrete_sequence=["#C49A52"])
        hist.update_layout(template="plotly_white", height=330)
        st.plotly_chart(hist, use_container_width=True)
    with vol_col:
        st.markdown("**Produk berdasarkan tingkat volatilitas**")
        vol_df = pd.DataFrame({"Koefisien variasi": cv_products})
        vol_df = vol_df.replace([np.inf, -np.inf], np.nan).dropna()
        vol = px.histogram(vol_df, x="Koefisien variasi", nbins=35,
                           color_discrete_sequence=["#477B9B"])
        vol.update_layout(template="plotly_white", height=330)
        st.plotly_chart(vol, use_container_width=True)

    st.divider()
    st.subheader("Temuan awal yang dapat ditelusuri")
    high_var = int(np.sum(cv_products >= np.nanquantile(cv_products, .90)))
    no_sales = int(np.sum(product_total == 0))
    steady = int(np.sum(cv_products <= np.nanquantile(cv_products, .25)))
    f1, f2, f3 = st.columns(3)
    f1.metric("Produk tanpa transaksi", f"{no_sales:,}")
    f2.metric("Produk dengan volatilitas 10% tertinggi", f"{high_var:,}")
    f3.metric("Produk relatif stabil (kuartil bawah CV)", f"{steady:,}")
    st.caption(
        f"Ringkasan otomatis: total transaksi tertinggi terjadi pada W{peak}, sedangkan terendah pada W{low}. "
        f"Sebanyak {zero_share:.1f}% observasi produk-minggu bernilai nol. "
        "Nilai nol dapat menunjukkan tidak adanya transaksi pada minggu tersebut, bukan otomatis missing value."
    )

    st.markdown("**Pola beberapa produk (pilih produk untuk ditelusuri)**")
    selected_products = st.multiselect(
        "Product_Code", options=codes.tolist(),
        default=codes[top_order[:3]].tolist(),
        max_selections=8
    )
    if selected_products:
        fig_prod = go.Figure()
        for code in selected_products:
            idx = int(np.where(codes == code)[0][0])
            fig_prod.add_trace(go.Scatter(
                x=weeks, y=X_raw[idx], mode="lines+markers",
                name=code, line=dict(width=2),
                hovertemplate=f"{code}<br>%{{x}}: %{{y}}<extra></extra>"
            ))
        fig_prod.update_layout(template="plotly_white", height=390,
                               xaxis_title="Minggu", yaxis_title="Transaksi",
                               hovermode="x unified")
        st.plotly_chart(fig_prod, use_container_width=True)

    with st.expander("Lihat statistik deskriptif seluruh variabel mingguan"):
        desc = pd.DataFrame(X_raw, columns=weeks).describe().T
        st.dataframe(desc, use_container_width=True)

# =============================================================================
# LOAD MATRIX ONLY FOR CLUSTERING PAGES
# =============================================================================
elif page in ["Hasil Klasterisasi", "Rekomendasi Metode"]:
    try:
        matrices = load_matrices_from_url(MATRIX_URL)
    except Exception as exc:
        st.error(f"Gagal memuat matriks dari GitHub Releases: {exc}")
        st.info("Pastikan Release v1.0.0 bersifat publik dan asset all_distance_matrices.npz tersedia.")
        st.stop()

    saved_codes = matrices.get("__product_code__")
    if saved_codes is not None and not np.array_equal(saved_codes.astype(str), codes.astype(str)):
        st.error("Daftar atau urutan Product_Code pada dataset berbeda dari dataset pembentuk matriks. Gunakan dataset yang sama dengan eksperimen.")
        st.stop()

    # -------------------------------------------------------------------------
    # PAGE 2: CLUSTERING RESULT
    # -------------------------------------------------------------------------
    if page == "Hasil Klasterisasi":
        st.header("Hasil Klasterisasi")
        st.markdown('<div class="page-intro">Pilih kombinasi representasi data, metode jarak, dan algoritma untuk membentuk segmentasi produk.</div>', unsafe_allow_html=True)
        key = matrix_key(rep_key, distance, radius, g)
        D = matrices.get(key)
        if D is None:
            st.error(f"Matriks {key} tidak ditemukan dalam arsip NPZ.")
            st.stop()
        if D.shape != (len(data), len(data)):
            st.error(f"Dimensi matriks {D.shape} tidak sesuai dengan jumlah produk {len(data)}.")
            st.stop()
        if algorithm == "K-Means DBA" and distance == "WDTW":
            st.warning("K-Means DBA tidak tersedia untuk WDTW karena tslearn tidak menyediakan WDTW barycenter bawaan. Pilih K-Medoids, Hierarchical, atau DBSCAN.")
            st.stop()

        try:
            with st.spinner("Membentuk cluster dan menghitung metrik evaluasi..."):
                labels, centers, medoids, center_type, runtime = run_cluster(
                    algorithm, D, X, X_deriv, k, linkage_method, eps, min_samples, key
                )
                coords = make_mds(D)
                si, dbi, chi, bs, ncl, noise_pct, balance_contrib = evaluate_labels(D, coords, labels)
        except Exception as exc:
            st.error(f"Proses clustering gagal: {exc}")
            st.stop()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Silhouette", "—" if np.isnan(si) else f"{si:.4f}")
        m2.metric("DBI · MDS", "—" if np.isnan(dbi) else f"{dbi:.4f}")
        m3.metric("CHI · MDS", "—" if np.isnan(chi) else f"{chi:,.1f}")
        m4.metric("Balance Score", f"{bs:.4f}")
        m5.metric("Jumlah cluster", ncl, f"Noise {noise_pct:.2f}%")
        st.caption(f"Model: {algorithm}  |  Matriks: {key}  |  Waktu komputasi: {runtime:.2f} detik")
        if np.isnan(si):
            st.warning("Metrik pemisahan tidak dapat dihitung: jumlah cluster valid kurang dari dua atau hanya satu anggota non-noise.")

        # Remap labels to display labels, keep -1 as noise.
        cluster_ids = sorted(int(x) for x in np.unique(labels) if x != -1)
        display_names = {lab: f"Cluster {i+1}" for i, lab in enumerate(cluster_ids)}
        display_labels = np.array([display_names.get(int(v), "Noise") for v in labels])

        # Summary table: balance contribution is additive per cluster.
        rows = []
        rep_indices = {}
        for lab in cluster_ids:
            members = np.where(labels == lab)[0]
            rep_idx = representative_index(D, members)
            rep_indices[lab] = rep_idx
            p = len(members) / max(np.sum(labels != -1), 1)
            contribution = balance_contrib.get(lab, 0.0)
            rows.append({
                "Cluster": display_names[lab],
                "Jumlah produk": len(members),
                "Proporsi anggota (%)": 100 * len(members) / len(labels),
                "Kontribusi Balance Score": contribution,
                "Produk representatif": codes[rep_idx],
                "Total transaksi representatif": int(X_raw[rep_idx].sum())
            })
        if np.any(labels == -1):
            rows.append({
                "Cluster": "Noise",
                "Jumlah produk": int(np.sum(labels == -1)),
                "Proporsi anggota (%)": float(np.mean(labels == -1) * 100),
                "Kontribusi Balance Score": np.nan,
                "Produk representatif": "—",
                "Total transaksi representatif": np.nan
            })
        summary = pd.DataFrame(rows)

        st.subheader("Ringkasan cluster dan produk representatif")
        st.dataframe(
            summary.style.format({
                "Proporsi anggota (%)": "{:.2f}",
                "Kontribusi Balance Score": "{:.4f}",
                "Total transaksi representatif": "{:,.0f}"
            }, na_rep="—"),
            use_container_width=True, hide_index=True
        )
        st.caption(
            "Produk representatif adalah anggota cluster dengan jumlah jarak terkecil terhadap anggota lain (medoid deskriptif). "
            "Untuk K-Means DBA, ini bukan centroid; centroid DBA ditampilkan terpisah pada grafik. "
            "Kontribusi Balance Score per cluster dijumlahkan menjadi Balance Score keseluruhan."
        )

        # Pie below the table, as requested.
        st.markdown("**Komposisi anggota cluster**")
        pie_df = summary[summary["Cluster"] != "Noise"]
        pie = px.pie(pie_df, names="Cluster", values="Jumlah produk",
                     hole=.48, color_discrete_sequence=["#2A817A", "#C49A52", "#6D7892", "#A6B7A8", "#B77B5D"])
        pie.update_layout(template="plotly_white", height=360,
                          legend_title="Kelompok produk")
        st.plotly_chart(pie, use_container_width=True)

        st.divider()
        st.subheader("Profil deret waktu setiap cluster")
        # Ringkasan seluruh profil pusat cluster dalam satu grafik.
        fig_all_centers = go.Figure()
        for lab in cluster_ids:
            ids = np.where(labels == lab)[0]
            if centers is not None and algorithm == "K-Means DBA":
                prof = centers[lab]
                prof_name = "Centroid DBA"
            elif algorithm == "K-Medoids PAM":
                ridx = rep_indices[lab]
                prof = X[ridx]
                prof_name = f"Medoid {codes[ridx]}"
            else:
                prof = np.median(X[ids], axis=0)
                prof_name = "Median profil"
            fig_all_centers.add_trace(go.Scatter(
                x=weeks, y=prof, mode="lines",
                name=f"{display_names[lab]} · {prof_name}",
                line=dict(width=3)
            ))
        fig_all_centers.update_layout(
            template="plotly_white", height=390,
            title="Perbandingan profil pusat seluruh cluster",
            xaxis_title="Minggu", yaxis_title=f"Nilai transaksi ({rep_label})",
            hovermode="x unified"
        )
        st.plotly_chart(fig_all_centers, use_container_width=True)

        selected_cluster_name = st.selectbox("Pilih cluster untuk ditelusuri", [display_names[x] for x in cluster_ids])
        selected_lab = cluster_ids[[display_names[x] for x in cluster_ids].index(selected_cluster_name)]
        member_idx = np.where(labels == selected_lab)[0]
        rep_idx = rep_indices[selected_lab]

        # Profile center: DBA centroid if available, otherwise median member profile.
        if centers is not None and algorithm in ["K-Means DBA", "K-Medoids PAM"]:
            if algorithm == "K-Means DBA":
                center_profile = centers[selected_lab]
            else:
                center_profile = X[rep_idx]
            center_label = "Centroid DBA" if algorithm == "K-Means DBA" else "Medoid produk"
        else:
            center_profile = np.median(X[member_idx], axis=0)
            center_label = "Median profil anggota"

        rng = np.random.default_rng(2026 + selected_lab)
        sample_idx = rng.choice(member_idx, size=min(35, len(member_idx)), replace=False)
        fig_cluster = go.Figure()
        for idx in sample_idx:
            fig_cluster.add_trace(go.Scatter(
                x=weeks, y=X[idx], mode="lines",
                line=dict(width=1, color="rgba(116,126,140,0.22)"),
                showlegend=False,
                hovertemplate=f"Produk {codes[idx]}<br>%{{x}}: %{{y:.3f}}<extra></extra>"
            ))
        fig_cluster.add_trace(go.Scatter(
            x=weeks, y=center_profile, mode="lines+markers",
            name=center_label, line=dict(color="#B17C35", width=4),
            marker=dict(size=5),
            hovertemplate=f"{center_label}<br>%{{x}}: %{{y:.3f}}<extra></extra>"
        ))
        fig_cluster.update_layout(
            template="plotly_white", height=430,
            title=f"{selected_cluster_name} · {len(member_idx)} produk",
            xaxis_title="Minggu", yaxis_title=f"Nilai transaksi ({rep_label})",
            hovermode="x unified", legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Interactive member list and selected product profile.
        st.subheader(f"Anggota {selected_cluster_name}")
        member_df = pd.DataFrame({
            "Product_Code": codes[member_idx],
            "Total transaksi": X_raw[member_idx].sum(axis=1),
            "Rata-rata mingguan": X_raw[member_idx].mean(axis=1),
            "Standar deviasi": X_raw[member_idx].std(axis=1),
            "Koefisien variasi": np.divide(
                X_raw[member_idx].std(axis=1), X_raw[member_idx].mean(axis=1),
                out=np.zeros(len(member_idx)), where=X_raw[member_idx].mean(axis=1) != 0
            )
        }).sort_values("Total transaksi", ascending=False)
        search_code = st.text_input("Cari Product_Code dalam cluster", placeholder="Ketik kode produk...")
        if search_code.strip():
            member_df = member_df[member_df["Product_Code"].str.contains(search_code.strip(), case=False, na=False)]
        st.dataframe(member_df, use_container_width=True, hide_index=True, height=300)
        st.caption(f"Menampilkan {len(member_df):,} produk yang sesuai pada {selected_cluster_name}.")
        if len(member_df):
            product_choice = st.selectbox("Lihat pola satu produk", member_df["Product_Code"].tolist())
            product_idx = int(np.where(codes == product_choice)[0][0])
            fig_one = px.line(
                x=weeks, y=X_raw[product_idx], markers=True,
                labels={"x": "Minggu", "y": "Transaksi", "y": "Jumlah transaksi"},
                title=f"Pola transaksi {product_choice}"
            )
            fig_one.update_traces(line=dict(color="#2A817A", width=3))
            fig_one.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig_one, use_container_width=True)

        # Per-cluster interpretation and actionable notes.
        st.divider()
        st.subheader("Interpretasi dan rekomendasi cluster")
        for lab in cluster_ids:
            idxs = np.where(labels == lab)[0]
            profile = X_raw[idxs].mean(axis=0)
            early = float(profile[:13].mean())
            late = float(profile[-13:].mean())
            pct = ((late - early) / abs(early) * 100) if early != 0 else np.nan
            cv = float(np.std(profile) / (np.mean(profile) + 1e-9))
            if np.isfinite(pct) and pct > 10:
                pattern, action = "Menguat", "Tinjau kenaikan kebutuhan dan pertimbangkan penyesuaian safety stock secara bertahap."
            elif np.isfinite(pct) and pct < -10:
                pattern, action = "Melemah", "Evaluasi reorder point dan hindari penambahan stok sebelum tren penurunan terkonfirmasi."
            else:
                pattern, action = "Relatif stabil", "Gunakan replenishment berkala dengan safety stock sesuai variasi historis."
            if cv > .75:
                action += " Variasi cukup tinggi; pantau produk lebih sering dan evaluasi kebutuhan buffer."
            with st.container(border=True):
                st.markdown(f"**{display_names[lab]} · {pattern}**")
                st.write(f"{len(idxs)} produk · perubahan rata-rata 13 minggu awal ke 13 minggu akhir: " +
                         (f"{pct:.1f}%" if np.isfinite(pct) else "tidak terdefinisi"))
                st.write(action)

        export_df = pd.DataFrame({
            "Product_Code": codes,
            "Cluster": display_labels,
            "Cluster_ID": labels
        })
        st.download_button(
            "Unduh label cluster (CSV)",
            export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cluster_labels_{key}_{algorithm.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    # -------------------------------------------------------------------------
    # PAGE 3: METHOD RECOMMENDATION PER DISTANCE
    # -------------------------------------------------------------------------
    else:
        st.header("Rekomendasi Metode Klaster")
        st.markdown('<div class="page-intro">Membandingkan algoritma pada matriks jarak yang dipilih. Hasil rekomendasi didasarkan pada evaluasi empiris, bukan klaim bahwa satu algoritma selalu terbaik.</div>', unsafe_allow_html=True)
        rec_rep_label = st.selectbox("Representasi untuk evaluasi metode", ["Data asli", "MinMax", "Scaled"], key="rec_rep")
        rec_rep = {"Data asli": "Original", "MinMax": "MinMax", "Scaled": "Scaled"}[rec_rep_label]
        rec_dist = st.selectbox("Pilih metode jarak untuk dibandingkan", ["DTW", "CDTW", "DDTW", "WDTW"], key="rec_dist")
        rec_radius = st.selectbox("Radius CDTW", [2, 3, 5], key="rec_radius") if rec_dist == "CDTW" else None
        rec_g = st.selectbox("Parameter g WDTW", [0.05, 0.1, 0.5, 1.0], key="rec_g") if rec_dist == "WDTW" else None
        rec_k = st.slider("Jumlah cluster pembanding", 2, 10, 3, key="rec_k")
        rec_key = matrix_key(rec_rep, rec_dist, rec_radius, rec_g)
        RD = matrices.get(rec_key)
        if RD is None:
            st.error(f"Matriks {rec_key} tidak ditemukan.")
            st.stop()
        if RD.shape != (len(data), len(data)):
            st.error("Dimensi matriks tidak cocok dengan dataset.")
            st.stop()
        if st.button("Jalankan perbandingan metode", type="primary"):
            results = []
            with st.spinner("Mengevaluasi beberapa algoritma pada matriks terpilih..."):
                rcoords = make_mds(RD)
                # K-Means DBA is not available for WDTW.
                candidates = [
                    "K-Medoids PAM", "Hierarchical Average",
                    "Hierarchical Complete", "DBSCAN (eps otomatis)"
                ]
                if rec_dist != "WDTW":
                    candidates.insert(0, "K-Means DBA")
                for candidate in candidates:
                    t0 = time.perf_counter()
                    try:
                        if candidate == "K-Means DBA":
                            params = {}
                            if rec_dist == "CDTW":
                                params = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": rec_radius}
                            fit_x = derivative_ts(X) if rec_dist == "DDTW" else X
                            km = TimeSeriesKMeans(
                                n_clusters=rec_k, metric="dtw", metric_params=params,
                                max_iter=8, max_iter_barycenter=8, n_init=1,
                                random_state=42, n_jobs=-1
                            )
                            lab = km.fit_predict(fit_x[:, :, None])
                        elif candidate == "K-Medoids PAM":
                            lab, _ = kmedoids_pam(RD, rec_k, max_iter=70)
                        elif candidate == "DBSCAN (eps otomatis)":
                            min_s = 10
                            kth = np.sort(RD, axis=1)[:, min(min_s, RD.shape[0]-1)]
                            eps_auto = float(np.quantile(kth, 0.85))
                            lab = DBSCAN(
                                eps=max(eps_auto, 1e-12), min_samples=min_s,
                                metric="precomputed"
                            ).fit_predict(RD)
                        else:
                            method = "average" if candidate.endswith("Average") else "complete"
                            zz = linkage(squareform(RD, checks=False), method=method)
                            lab = cut_tree(zz, n_clusters=[rec_k]).ravel()
                        si2, dbi2, chi2, bs2, ncl2, noise2, _ = evaluate_labels(RD, rcoords, lab)
                        results.append({
                            "Metode": candidate, "Silhouette": si2, "DBI (MDS)": dbi2,
                            "CHI (MDS)": chi2, "Balance Score": bs2,
                            "Jumlah cluster": ncl2, "Noise (%)": noise2,
                            "Waktu (detik)": time.perf_counter() - t0
                        })
                    except Exception as exc:
                        results.append({"Metode": candidate, "Silhouette": np.nan,
                                        "DBI (MDS)": np.nan, "CHI (MDS)": np.nan,
                                        "Balance Score": np.nan, "Jumlah cluster": 0,
                                        "Noise (%)": np.nan, "Waktu (detik)": np.nan,
                                        "Catatan": str(exc)})
                result_df = pd.DataFrame(results)
                valid = result_df.dropna(subset=["Silhouette", "DBI (MDS)", "CHI (MDS)"]).copy()
                if len(valid):
                    valid["Rank_SI"] = valid["Silhouette"].rank(ascending=False, method="average")
                    valid["Rank_DBI"] = valid["DBI (MDS)"].rank(ascending=True, method="average")
                    valid["Rank_CHI"] = valid["CHI (MDS)"].rank(ascending=False, method="average")
                    valid["Skor ranking"] = valid[["Rank_SI", "Rank_DBI", "Rank_CHI"]].mean(axis=1)
                    valid = valid.sort_values("Skor ranking")
                    st.session_state["method_comparison"] = valid
                    st.session_state["method_comparison_key"] = rec_key
                else:
                    st.session_state["method_comparison"] = result_df
                    st.session_state["method_comparison_key"] = rec_key

        if "method_comparison" in st.session_state and st.session_state.get("method_comparison_key") == rec_key:
            result_df = st.session_state["method_comparison"]
            if "Skor ranking" in result_df.columns and result_df["Skor ranking"].notna().any():
                st.success(f"Metode dengan peringkat terbaik pada kombinasi {rec_key}: **{result_df.iloc[0]['Metode']}**.")
                st.caption("Peringkat dihitung dari rata-rata ranking Silhouette (besar lebih baik), DBI (kecil lebih baik), dan CHI (besar lebih baik). Bobot ketiga metrik sama.")
            else:
                st.warning("Belum ada model dengan metrik evaluasi lengkap yang valid.")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Unduh perbandingan metode (CSV)",
                result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"method_comparison_{rec_key}.csv",
                mime="text/csv"
            )
        st.info("Catatan: DBSCAN pada bagian ini menggunakan eps otomatis dari persentil ke-85 jarak tetangga ke-10 (min_samples=10). Karena DBSCAN tidak menetapkan K secara langsung, jumlah cluster dan noise dapat berbeda dari algoritma lain. Evaluasi DBI dan CHI tetap dihitung pada embedding MDS 5 dimensi.")

# =============================================================================
# PAGE 4: METHOD GUIDE
# =============================================================================
elif page == "Panduan Metode":
    st.header("Panduan dan Dokumentasi")
    st.markdown('<div class="page-intro">Definisi singkat untuk membantu pembaca memahami pilihan model dan interpretasi hasil dashboard.</div>', unsafe_allow_html=True)
    with st.expander("Representasi data", expanded=True):
        st.markdown("""
        - **Data asli:** mempertahankan besaran transaksi sebagaimana tercatat.
        - **MinMax:** menormalkan rentang setiap produk ke skala 0–1; berguna untuk menekankan bentuk pola relatif dan mengurangi pengaruh skala volume.
        - **Scaled:** standardisasi per minggu/kolom sehingga posisi relatif produk pada setiap minggu dibandingkan terhadap produk lain.
        """)
    with st.expander("Metode jarak"):
        st.markdown("""
        - **DTW:** mengukur kemiripan dua deret dengan penyelarasan waktu yang fleksibel.
        - **CDTW:** DTW dengan batas Sakoe–Chiba. Radius membatasi seberapa jauh pergeseran waktu yang diperbolehkan.
        - **DDTW:** DTW pada representasi turunan, sehingga lebih menekankan pola perubahan dan arah pergerakan.
        - **WDTW:** memberi bobot pada pasangan waktu berdasarkan jarak indeks waktunya; parameter *g* mengatur bentuk pembobotan.
        """)
    with st.expander("Algoritma clustering"):
        st.markdown("""
        - **K-Means DBA:** membentuk centroid deret waktu dengan Dynamic Time Warping Barycenter Averaging.
        - **K-Medoids PAM:** memilih produk aktual sebagai pusat cluster berdasarkan total jarak ke anggota.
        - **Hierarchical:** membentuk cluster bertingkat; average memakai rata-rata jarak antaranggotanya, complete memakai jarak maksimum.
        - **DBSCAN:** mengelompokkan objek berdasarkan kepadatan; label -1 menunjukkan noise.
        """)
    with st.expander("Metrik evaluasi"):
        st.markdown("""
        - **Silhouette:** semakin tinggi semakin baik; mengukur kekompakan cluster dan pemisahan antarklaster. Dihitung langsung dari matriks jarak.
        - **Davies–Bouldin Index (DBI):** semakin rendah semakin baik. Pada dashboard ini dihitung pada embedding MDS 5 dimensi.
        - **Calinski–Harabasz Index (CHI):** semakin tinggi semakin baik. Pada dashboard ini dihitung pada embedding MDS 5 dimensi.
        - **Balance Score:** normalized entropy ukuran cluster. Nilai mendekati 1 menunjukkan ukuran cluster lebih merata. Ini bukan ukuran kualitas pemisahan cluster.
        """)
    st.info("Kontribusi Balance Score per cluster adalah komponen entropy masing-masing cluster. Jumlah seluruh kontribusi sama dengan Balance Score keseluruhan. Untuk DBSCAN, noise tidak dimasukkan dalam perhitungan.")
    st.markdown("**Catatan interpretasi bisnis**")
    st.write("Pola historis dapat membantu mengidentifikasi produk yang meningkat, menurun, stabil, atau fluktuatif. Rekomendasi stok pada dashboard bersifat indikatif dan sebaiknya dipadukan dengan lead time, biaya penyimpanan, kebijakan layanan, serta informasi persediaan aktual.")
