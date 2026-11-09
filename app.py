# app.py
# Sales Analytics Dashboard — Weekly Transactions & Time-Series Clustering
# Dataset: Sales transaction data.xlsx (root repository)
# Distance matrices: GitHub Release v1.0.0 / all_distance_matrices.npz
#
# Deployment:
# 1) Simpan app.py, requirements.txt, dan Sales transaction data.xlsx di root repository.
# 2) Simpan all_distance_matrices.npz sebagai asset pada GitHub Release v1.0.0.
# 3) Deploy app.py melalui Streamlit Community Cloud.
#
# Security note: allow_pickle=True hanya digunakan untuk NPZ eksperimen milik sendiri
# yang diunduh dari GitHub Release repository ini.

from __future__ import annotations

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
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import squareform
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw_path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sales Analytics | Weekly Transactions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# 1. THEME
# ---------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    :root {
      --navy:#1B2A4A; --navy2:#263B60; --cream:#F5F2EB;
      --paper:#FFFEFB; --teal:#2A817A; --gold:#C49A52;
      --terracotta:#B86F52; --ink:#263247; --muted:#687386;
      --line:#E7E2D8;
    }
    html, body, [class*="css"] {
      font-family:'Plus Jakarta Sans',sans-serif !important;
      color:var(--ink);
    }
    .stApp {background:linear-gradient(145deg,#F5F2EB 0%,#FAF9F5 52%,#EEF3F1 100%);}
    [data-testid="stSidebar"] {background:var(--navy);border-right:1px solid #344563;}
    [data-testid="stSidebar"] * {color:var(--cream);}
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] p {color:var(--cream)!important;}
    [data-testid="stSidebar"] [data-baseweb="select"]>div {
      background:#FFFEFB!important;border:1px solid #D8D2C6!important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {color:#263247!important;}
    .hero {
      background:linear-gradient(112deg,#1B2A4A 0%,#263B60 68%,#2A817A 120%);
      border:1px solid rgba(245,242,235,.2);border-radius:20px;
      padding:30px 34px;margin:4px 0 24px;
      box-shadow:0 12px 30px rgba(27,42,74,.14);
      animation:fadeIn .45s ease-out;
    }
    .hero-kicker {color:#D9C59F;font-size:11px;letter-spacing:2.2px;font-weight:800;margin-bottom:9px;}
    .hero h1 {color:#F5F2EB!important;font-size:36px!important;line-height:1.15;font-weight:800;margin:0 0 7px;}
    .hero h2 {color:#F5F2EB!important;font-size:20px!important;font-weight:600;margin:0 0 10px;}
    .hero p {color:#E8E5DC!important;font-size:14px;margin:0;max-width:900px;}
    .hero-mark {float:right;color:#D9C59F;font-size:48px;padding:7px 10px 0 18px;}
    .page-intro {color:#687386;font-size:13px;margin-top:-8px;margin-bottom:18px;}
    div[data-testid="stMetric"] {
      background:#FFFEFB;border:1px solid #E4DED2;border-top:3px solid #C49A52;
      border-radius:14px;padding:13px 15px;box-shadow:0 3px 10px rgba(27,42,74,.04);
      transition:all .3s ease;
    }
    div[data-testid="stMetric"]:hover {transform:translateY(-3px);box-shadow:0 8px 18px rgba(27,42,74,.10);}
    div[data-testid="stMetricLabel"] {color:#687386!important;}
    div[data-testid="stMetricValue"] {color:#1B2A4A!important;}
    h1,h2,h3 {color:#1B2A4A;letter-spacing:-.35px;font-weight:750!important;}
    div[data-testid="stDataFrame"] {border:1px solid #E7E2D8;border-radius:11px;}
    .stButton>button,.stDownloadButton>button {
      background:#1B2A4A;color:#F5F2EB;border:1px solid #1B2A4A;border-radius:10px;
      transition:all .25s ease;
    }
    .stButton>button:hover,.stDownloadButton>button:hover {background:#2A817A;color:white;border-color:#2A817A;transform:translateY(-1px);}
    .exec-card {background:#FFFEFB;border:1px solid #E7E2D8;border-left:4px solid #2A817A;border-radius:14px;padding:16px 18px;margin:8px 0;box-shadow:0 4px 13px rgba(27,42,74,.045);}
    .soft-note {background:#F0F3F1;border:1px solid #DCE6E1;border-radius:12px;padding:12px 15px;color:#405365;}
    @keyframes fadeIn {from{opacity:0;transform:translateY(5px);}to{opacity:1;transform:translateY(0);}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 2. PATHS / CONSTANTS
# ---------------------------------------------------------------------
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
WEEKS = [f"W{i}" for i in range(52)]
PALETTE = ["#2A817A", "#C49A52", "#477B9B", "#B86F52", "#7A6595",
           "#78936F", "#C17B9B", "#6D7892", "#D18B36", "#4F8C91"]

# ---------------------------------------------------------------------
# 3. DATA / DISTANCE HELPERS
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sales(file_bytes: bytes, suffix: str):
    bio = BytesIO(file_bytes)
    frame = pd.read_excel(bio) if suffix in (".xlsx", ".xls") else pd.read_csv(bio)
    if "Product_Code" not in frame.columns:
        raise ValueError("Kolom Product_Code tidak ditemukan.")
    missing = [c for c in WEEKS if c not in frame.columns]
    if missing:
        raise ValueError(f"Kolom mingguan belum lengkap: {missing}")
    frame = frame.drop_duplicates("Product_Code").copy()
    frame[WEEKS] = frame[WEEKS].apply(pd.to_numeric, errors="coerce")
    frame[WEEKS] = frame[WEEKS].replace([np.inf, -np.inf], np.nan)
    frame[WEEKS] = frame[WEEKS].fillna(frame[WEEKS].median()).fillna(0)
    return frame, WEEKS


@st.cache_resource(show_spinner="Mengunduh matriks jarak dari GitHub Releases...")
def load_matrices_from_url(url: str):
    response = requests.get(url, timeout=(30, 600))
    response.raise_for_status()
    archive = np.load(BytesIO(response.content), allow_pickle=True)
    result = {key: archive[key] for key in archive.files if key != "product_code"}
    if "product_code" in archive.files:
        result["__product_code__"] = archive["product_code"].astype(str)
    archive.close()
    return result


def row_minmax(array):
    lo = np.nanmin(array, axis=1, keepdims=True)
    hi = np.nanmax(array, axis=1, keepdims=True)
    span = hi - lo
    return np.divide(array - lo, span, out=np.zeros_like(array), where=span != 0)


def derivative_ts(array):
    array = np.asarray(array, dtype=float)
    result = np.zeros_like(array)
    result[:, 0] = array[:, 1] - array[:, 0]
    for t in range(1, array.shape[1] - 1):
        result[:, t] = (
            0.5 * (array[:, t] - array[:, t - 1])
            + 0.25 * (array[:, t + 1] - array[:, t - 1])
        )
    result[:, -1] = array[:, -1] - array[:, -2]
    return result


def matrix_key(rep_key, distance, radius=None, g=None):
    if distance == "DTW":
        return f"DTW_{rep_key}"
    if distance == "CDTW":
        return f"CDTW_{rep_key}_R{radius}"
    if distance == "DDTW":
        return f"DDTW_{rep_key}"
    return f"WDTW_{rep_key}_g{g}"


def balance_score(labels):
    values = np.asarray(labels)
    values = values[values != -1]
    clusters, counts = np.unique(values, return_counts=True)
    n_cluster = len(clusters)
    if n_cluster <= 1:
        return 0.0, {int(c): 0.0 for c in clusters}
    proportions = counts / counts.sum()
    contributions = -(proportions * np.log(proportions)) / np.log(n_cluster)
    return float(contributions.sum()), {
        int(cluster): float(value)
        for cluster, value in zip(clusters, contributions)
    }


def representative_index(distance_matrix, member_indices):
    if len(member_indices) == 1:
        return int(member_indices[0])
    block = distance_matrix[np.ix_(member_indices, member_indices)]
    return int(member_indices[np.argmin(block.sum(axis=1))])


def kmedoids_pam(distance_matrix, k, seed=42, max_iter=100):
    """PAM sederhana untuk matriks jarak precomputed; pusat adalah produk aktual."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(distance_matrix)
    if not 1 <= k < n:
        raise ValueError("Jumlah cluster K harus lebih kecil dari jumlah produk.")
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(n))]
    while len(medoids) < k:
        nearest = np.min(distance_matrix[:, medoids], axis=1)
        nearest[medoids] = 0
        weights = nearest ** 2
        weights[medoids] = 0
        available = np.setdiff1d(np.arange(n), medoids)
        if weights.sum() <= 0:
            next_medoid = int(rng.choice(available))
        else:
            probabilities = weights / weights.sum()
            next_medoid = int(rng.choice(n, p=probabilities))
            if next_medoid in medoids:
                next_medoid = int(rng.choice(available))
        medoids.append(next_medoid)
    medoids = np.asarray(medoids, dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(distance_matrix[:, medoids], axis=1)
        updated = medoids.copy()
        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if len(members):
                block = distance_matrix[np.ix_(members, members)]
                updated[cluster_id] = members[np.argmin(block.sum(axis=1))]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return np.argmin(distance_matrix[:, medoids], axis=1), medoids


@st.cache_data(show_spinner=False)
def make_mds(distance_matrix):
    model = MDS(
        n_components=5,
        metric=True,
        dissimilarity="precomputed",
        random_state=42,
        n_init=1,
        max_iter=250,
    )
    return model.fit_transform(distance_matrix)


def evaluate_labels(distance_matrix, coordinates, labels):
    labels = np.asarray(labels)
    keep = labels != -1
    valid_labels = labels[keep]
    unique = np.unique(valid_labels)
    silhouette = dbi = chi = np.nan
    if 2 <= len(unique) < len(valid_labels):
        sub_distance = distance_matrix[np.ix_(keep, keep)]
        silhouette = silhouette_score(sub_distance, valid_labels, metric="precomputed")
        dbi = davies_bouldin_score(coordinates[keep], valid_labels)
        chi = calinski_harabasz_score(coordinates[keep], valid_labels)
    balance, contributions = balance_score(labels)
    noise_count = int(np.sum(labels == -1))
    noise_pct = float(noise_count / len(labels) * 100) if len(labels) else 0.0
    return silhouette, dbi, chi, balance, len(unique), noise_count, noise_pct, contributions


def fit_clustering(algo, distance_matrix, series, series_derivative, k,
                   linkage_method, eps, min_samples, key):
    started = time.perf_counter()
    centers = None
    medoids = None
    hierarchy_tree = None
    if algo == "K-Means DBA":
        if key.startswith("WDTW"):
            raise ValueError(
                "K-Means DBA untuk WDTW belum tersedia sebagai implementasi native tslearn. "
                "Gunakan K-Medoids, Hierarchical, atau DBSCAN."
            )
        fit_series = series_derivative if key.startswith("DDTW") else series
        metric_params = {}
        if key.startswith("CDTW"):
            metric_params = {
                "global_constraint": "sakoe_chiba",
                "sakoe_chiba_radius": int(key.rsplit("R", 1)[1]),
            }
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            metric_params=metric_params,
            max_iter=10,
            max_iter_barycenter=10,
            n_init=2,
            random_state=42,
            n_jobs=-1,
        )
        labels = model.fit_predict(fit_series[:, :, None])
        centers = model.cluster_centers_[:, :, 0]
        center_kind = "Centroid DBA"
    elif algo == "K-Medoids PAM":
        labels, medoids = kmedoids_pam(distance_matrix, k)
        centers = series[medoids]
        center_kind = "Medoid aktual"
    elif algo == "Hierarchical":
        hierarchy_tree = linkage(
            squareform(distance_matrix, checks=False),
            method=linkage_method,
        )
        labels = cut_tree(hierarchy_tree, n_clusters=[k]).ravel()
        center_kind = "Median profil"
    else:
        labels = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
            metric="precomputed",
        ).fit_predict(distance_matrix)
        center_kind = "Median profil"
    elapsed = time.perf_counter() - started
    return labels, centers, medoids, hierarchy_tree, center_kind, elapsed


def silhouette_by_k(distance_matrix, k_values):
    rows = []
    for candidate_k in k_values:
        try:
            labels, _ = kmedoids_pam(distance_matrix, candidate_k, seed=42, max_iter=70)
            value = silhouette_score(distance_matrix, labels, metric="precomputed")
            rows.append({"K": candidate_k, "Silhouette": value})
        except Exception:
            rows.append({"K": candidate_k, "Silhouette": np.nan})
    return pd.DataFrame(rows)


def get_profile(series, labels, cluster_id, centers, medoids, algorithm, distance_matrix):
    member_idx = np.where(labels == cluster_id)[0]
    if algorithm == "K-Means DBA" and centers is not None:
        return centers[cluster_id], "Centroid DBA", member_idx
    if algorithm == "K-Medoids PAM" and medoids is not None:
        medoid_idx = int(medoids[cluster_id])
        return series[medoid_idx], f"Medoid produk #{medoid_idx}", member_idx
    return np.median(series[member_idx], axis=0), "Median profil anggota", member_idx


def silhouette_quality(value):
    if not np.isfinite(value):
        return "Belum dapat dievaluasi"
    if value >= 0.71:
        return "Struktur Klaster Kuat (Strong structure)"
    if value >= 0.51:
        return "Struktur Klaster Baik (Good/medium structure)"
    if value >= 0.26:
        return "Struktur Klaster Lemah (Weak structure)"
    return "Tidak Ada Struktur Jelas (No structure / poor)"


def make_accumulated_cost(a, b, radius=None):
    """Matriks akumulasi squared Euclidean untuk ilustrasi warping path."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    local = (a[:, None] - b[None, :]) ** 2
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start, j_end = 1, m
        if radius is not None:
            j_start = max(1, i - radius)
            j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            accumulated[i, j] = local[i - 1, j - 1] + min(
                accumulated[i - 1, j],
                accumulated[i, j - 1],
                accumulated[i - 1, j - 1],
            )
    return accumulated[1:, 1:], local


# ---------------------------------------------------------------------
# 4. LOAD DATASET
# ---------------------------------------------------------------------
source = next((path for path in DATA_CANDIDATES if path.exists()), None)
if source is None:
    st.error(
        "Dataset tidak ditemukan. Pastikan `Sales transaction data.xlsx` "
        "berada di root repository GitHub."
    )
    st.stop()

try:
    data, weeks = load_sales(source.read_bytes(), source.suffix.lower())
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

codes = data["Product_Code"].astype(str).to_numpy()
x_raw = data[weeks].to_numpy(dtype=float)
x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------------------------------------------------------------
# 5. NAVIGATION (sidebar contains navigation only)
# ---------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="font-size:11px;letter-spacing:2px;color:#D9C59F;font-weight:800;">SALES ANALYTICS</div>
    <div style="font-size:22px;font-weight:800;color:#F5F2EB;margin:5px 0 3px;">Weekly Transactions</div>
    <div style="font-size:12px;color:#D9DFE8;margin-bottom:18px;">Exploration · Distance · Clustering</div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "NAVIGASI UTAMA",
    [
        "Eksplorasi Data",
        "Hasil Klasterisasi",
        "Eksplorasi Jarak DTW",
        "Rekomendasi Metode",
        "Panduan Metode",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Sales Transactions · 52 weekly observations")

# ---------------------------------------------------------------------
# 6. HERO
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-mark">▥</div>
      <div class="hero-kicker">DATA INTELLIGENCE · WEEKLY SALES</div>
      <h1>SALES ANALYTICS</h1>
      <h2>Weekly Transactions</h2>
      <p>Memahami dinamika transaksi mingguan dan mengelompokkan produk berdasarkan kemiripan pola penjualan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# PAGE 1 — EXPLORATORY DATA
# ---------------------------------------------------------------------
if page == "Eksplorasi Data":
    st.header("Eksplorasi Awal Data")
    st.markdown(
        '<div class="page-intro">Mengenali skala transaksi, tren mingguan, dan variasi perilaku produk sebelum clustering.</div>',
        unsafe_allow_html=True,
    )

    n_products = len(data)
    total_sales = float(x_raw.sum())
    weekly_total = x_raw.sum(axis=0)
    weekly_mean = x_raw.mean(axis=0)
    product_total = x_raw.sum(axis=1)
    product_mean = x_raw.mean(axis=1)
    product_sd = x_raw.std(axis=1)
    product_cv = np.divide(
        product_sd, product_mean,
        out=np.zeros_like(product_sd), where=product_mean != 0,
    )
    zero_share = float((x_raw == 0).mean() * 100)
    peak_week = int(np.argmax(weekly_total))
    low_week = int(np.argmin(weekly_total))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah produk", f"{n_products:,}")
    c2.metric("Total transaksi", f"{total_sales:,.0f}")
    c3.metric("Rata-rata per produk-minggu", f"{x_raw.mean():.2f}")
    c4.metric("Proporsi nilai nol", f"{zero_share:.1f}%")

    st.subheader("Dinamika transaksi mingguan")
    trend = pd.DataFrame({
        "Minggu": weeks,
        "Total transaksi": weekly_total,
        "Rata-rata per produk": weekly_mean,
    })
    metric_choice = st.radio(
        "Ukuran transaksi",
        ["Total transaksi", "Rata-rata per produk"],
        horizontal=True,
    )
    fig = px.line(
        trend, x="Minggu", y=metric_choice, markers=True,
        color_discrete_sequence=["#2A817A"],
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig.update_layout(template="plotly_white", height=360, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Minggu tertinggi", f"W{peak_week}", f"{weekly_total[peak_week]:,.0f}")
    p2.metric("Minggu terendah", f"W{low_week}", f"{weekly_total[low_week]:,.0f}")
    p3.metric("Rentang transaksi mingguan", f"{weekly_total.max()-weekly_total.min():,.0f}")

    st.divider()
    st.subheader("Produk dengan volume transaksi ekstrem")
    top_idx = np.argsort(product_total)[::-1][:10]
    bottom_idx = np.argsort(product_total)[:10]
    left, right = st.columns(2)
    with left:
        st.markdown("**10 produk dengan total transaksi tertinggi**")
        st.dataframe(
            pd.DataFrame({
                "Product_Code": codes[top_idx],
                "Total transaksi": product_total[top_idx].astype(int),
                "Rata-rata mingguan": product_mean[top_idx],
            }),
            use_container_width=True, hide_index=True,
        )
    with right:
        st.markdown("**10 produk dengan total transaksi terendah**")
        st.dataframe(
            pd.DataFrame({
                "Product_Code": codes[bottom_idx],
                "Total transaksi": product_total[bottom_idx].astype(int),
                "Rata-rata mingguan": product_mean[bottom_idx],
            }),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Distribusi dan volatilitas")
    d1, d2 = st.columns(2)
    with d1:
        fig_hist = px.histogram(
            pd.DataFrame({"Total transaksi": product_total}),
            x="Total transaksi", nbins=35,
            color_discrete_sequence=["#C49A52"],
        )
        fig_hist.update_layout(template="plotly_white", height=330)
        st.plotly_chart(fig_hist, use_container_width=True)
    with d2:
        finite_cv = product_cv[np.isfinite(product_cv)]
        fig_cv = px.histogram(
            pd.DataFrame({"Koefisien variasi": finite_cv}),
            x="Koefisien variasi", nbins=35,
            color_discrete_sequence=["#477B9B"],
        )
        fig_cv.update_layout(template="plotly_white", height=330)
        st.plotly_chart(fig_cv, use_container_width=True)

    st.subheader("Temuan awal otomatis")
    zero_products = int(np.sum(product_total == 0))
    high_cv = int(np.sum(product_cv >= np.quantile(product_cv, .90)))
    stable_cv = int(np.sum(product_cv <= np.quantile(product_cv, .25)))
    f1, f2, f3 = st.columns(3)
    f1.metric("Produk tanpa transaksi", f"{zero_products:,}")
    f2.metric("Produk pada 10% CV tertinggi", f"{high_cv:,}")
    f3.metric("Produk pada 25% CV terendah", f"{stable_cv:,}")
    st.markdown(
        f'<div class="soft-note">Transaksi agregat tertinggi tercatat pada W{peak_week}, '
        f'sedangkan terendah pada W{low_week}. Proporsi sel produk-minggu bernilai nol '
        f'mencapai {zero_share:.1f}%. Nilai nol diperlakukan sebagai tidak adanya transaksi '
        'pada minggu tersebut, bukan otomatis sebagai data hilang.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Telusuri pola produk")
    default_codes = codes[top_idx[:3]].tolist()
    selected_codes = st.multiselect(
        "Pilih Product_Code (maksimal 8)",
        options=codes.tolist(),
        default=default_codes,
        max_selections=8,
    )
    if selected_codes:
        product_fig = go.Figure()
        for code in selected_codes:
            idx = int(np.where(codes == code)[0][0])
            product_fig.add_trace(go.Scatter(
                x=weeks, y=x_raw[idx], mode="lines+markers",
                name=code, line=dict(width=2),
            ))
        product_fig.update_layout(
            template="plotly_white", height=390,
            xaxis_title="Minggu", yaxis_title="Jumlah transaksi",
            hovermode="x unified",
        )
        st.plotly_chart(product_fig, use_container_width=True)

    with st.expander("Statistik deskriptif seluruh minggu"):
        st.dataframe(pd.DataFrame(x_raw, columns=weeks).describe().T, use_container_width=True)

# ---------------------------------------------------------------------
# LOAD MATRICES FOR DISTANCE / CLUSTER / RECOMMENDATION PAGES
# ---------------------------------------------------------------------
elif page in [
    "Hasil Klasterisasi",
    "Eksplorasi Jarak DTW",
    "Rekomendasi Metode",
]:
    try:
        matrices = load_matrices_from_url(MATRIX_URL)
    except Exception as exc:
        st.error(f"Gagal memuat matriks dari GitHub Releases: {exc}")
        st.info("Pastikan Release v1.0.0 publik dan asset all_distance_matrices.npz tersedia.")
        st.stop()

    saved_codes = matrices.get("__product_code__")
    if saved_codes is not None and not np.array_equal(saved_codes.astype(str), codes.astype(str)):
        st.error(
            "Urutan Product_Code pada dataset tidak sama dengan dataset pembentuk matriks. "
            "Gunakan file dataset yang sama dengan eksperimen."
        )
        st.stop()

    # -----------------------------------------------------------------
    # PAGE 2 — CLUSTERING RESULTS
    # -----------------------------------------------------------------
    if page == "Hasil Klasterisasi":
        st.header("Hasil Klasterisasi")
        st.markdown(
            '<div class="page-intro">Atur skenario eksperimen, evaluasi hasil, lalu telusuri profil dan anggota seluruh cluster.</div>',
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ Kontrol Parameter Klasterisasi", expanded=True):
            a, b, c = st.columns(3)
            with a:
                rep_label = st.selectbox("Representasi data", ["Data asli", "MinMax", "Scaled"])
                rep_key = {"Data asli": "Original", "MinMax": "MinMax", "Scaled": "Scaled"}[rep_label]
            with b:
                distance = st.selectbox("Metode jarak", ["DTW", "CDTW", "DDTW", "WDTW"])
                radius = st.selectbox("Radius Sakoe–Chiba", [2, 3, 5]) if distance == "CDTW" else None
                g = st.selectbox("Parameter g", [0.05, 0.1, 0.5, 1.0]) if distance == "WDTW" else None
            with c:
                algorithm = st.selectbox(
                    "Algoritma",
                    ["K-Means DBA", "K-Medoids PAM", "Hierarchical", "DBSCAN"],
                )
                if algorithm == "DBSCAN":
                    eps = st.number_input("DBSCAN eps", min_value=0.000001, value=1.0, step=0.1, format="%.6f")
                    min_samples = st.select_slider("min_samples", [5, 10, 15, 20], value=10)
                    k = None
                    linkage_method = "average"
                else:
                    k = st.slider("Jumlah cluster (K)", 2, 10, 3)
                    linkage_method = st.selectbox("Linkage", ["average", "complete"]) if algorithm == "Hierarchical" else "average"
                    eps, min_samples = None, None

        key = matrix_key(rep_key, distance, radius, g)
        distance_matrix = matrices.get(key)
        if distance_matrix is None:
            st.error(f"Matriks {key} tidak ditemukan dalam file NPZ.")
            st.stop()
        if distance_matrix.shape != (len(data), len(data)):
            st.error(f"Dimensi matriks {distance_matrix.shape} tidak cocok dengan {len(data)} produk.")
            st.stop()
        if algorithm == "K-Means DBA" and distance == "WDTW":
            st.warning(
                "K-Means DBA untuk WDTW belum tersedia sebagai implementasi native tslearn. "
                "Silakan pilih K-Medoids PAM, Hierarchical, atau DBSCAN."
            )
            st.stop()

        try:
            with st.spinner("Membentuk cluster dan menghitung evaluasi..."):
                series = row_minmax(x_raw) if rep_key == "MinMax" else (
                    StandardScaler().fit_transform(x_raw) if rep_key == "Scaled" else x_raw.copy()
                )
                series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
                series_derivative = derivative_ts(series)
                (
                    labels, centers, medoids, tree, center_kind, runtime
                ) = fit_clustering(
                    algorithm, distance_matrix, series, series_derivative,
                    k, linkage_method, eps, min_samples, key,
                )
                mds_coords = make_mds(distance_matrix)
                (
                    silhouette, dbi, chi, balance, n_clusters,
                    noise_count, noise_pct, contributions,
                ) = evaluate_labels(distance_matrix, mds_coords, labels)
        except Exception as exc:
            st.error(f"Proses clustering gagal: {exc}")
            st.stop()

        # Metrics: 5 main cards; DBSCAN noise is shown separately.
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Silhouette Score", "—" if not np.isfinite(silhouette) else f"{silhouette:.4f}")
        m2.metric("DBI (MDS)", "—" if not np.isfinite(dbi) else f"{dbi:.4f}")
        m3.metric("CHI (MDS)", "—" if not np.isfinite(chi) else f"{chi:,.2f}")
        m4.metric("Balance Score", f"{balance:.4f}")
        m5.metric("Jumlah cluster", n_clusters)
        st.caption(
            f"Model: {algorithm} | Matriks: {key} | Waktu: {runtime:.2f} detik"
        )
        st.markdown(
            f'<div class="soft-note"><b>Interpretasi Silhouette:</b> {silhouette_quality(silhouette)}. '
            'Rentang baku yang digunakan: 0.71–1.00 kuat; 0.51–0.70 baik/menengah; '
            '0.26–0.50 lemah; ≤0.25 tidak menunjukkan struktur yang jelas.</div>',
            unsafe_allow_html=True,
        )
        if algorithm == "DBSCAN":
            n_noise_col, p_noise_col = st.columns(2)
            n_noise_col.metric("Jumlah produk noise", f"{noise_count:,}")
            p_noise_col.metric("Persentase produk noise", f"{noise_pct:.2f}%")
        if not np.isfinite(silhouette):
            st.warning("Silhouette, DBI, dan CHI tidak dapat dihitung untuk jumlah cluster valid saat ini.")

        # Cluster display mapping
        cluster_ids = sorted(int(value) for value in np.unique(labels) if value != -1)
        if not cluster_ids:
            st.warning("DBSCAN belum membentuk cluster. Coba naikkan eps atau ubah min_samples.")
            st.stop()
        display_names = {lab: f"Cluster {i+1}" for i, lab in enumerate(cluster_ids)}
        summary_rows = []
        profile_rows = []
        rep_indices = {}
        profile_by_cluster = {}

        for lab in cluster_ids:
            member_idx = np.where(labels == lab)[0]
            rep_idx = representative_index(distance_matrix, member_idx)
            rep_indices[lab] = rep_idx
            profile, profile_label, _ = get_profile(
                series, labels, lab, centers, medoids, algorithm, distance_matrix
            )
            profile_by_cluster[lab] = (profile, profile_label, member_idx)
            member_total = x_raw[member_idx].sum(axis=1)
            member_week_mean = x_raw[member_idx].mean(axis=1)
            cluster_week_profile = x_raw[member_idx].mean(axis=0)
            cluster_mean = float(cluster_week_profile.mean())
            cluster_sd = float(cluster_week_profile.std())
            cluster_cv = cluster_sd / cluster_mean if cluster_mean else np.nan
            summary_rows.append({
                "Cluster": display_names[lab],
                "Jumlah produk": len(member_idx),
                "Proporsi anggota (%)": len(member_idx) / len(labels) * 100,
                "Kontribusi Balance Score": contributions.get(lab, 0.0),
                "Produk representatif": codes[rep_idx],
                "Total transaksi representatif": int(x_raw[rep_idx].sum()),
            })
            profile_rows.append({
                "Cluster": display_names[lab],
                "Rata-rata penjualan mingguan": cluster_mean,
                "Total penjualan cluster": float(member_total.sum()),
                "Volatilitas (CV)": cluster_cv,
                "Proporsi anggota (%)": len(member_idx) / len(labels) * 100,
            })

        if noise_count:
            summary_rows.append({
                "Cluster": "Noise",
                "Jumlah produk": noise_count,
                "Proporsi anggota (%)": noise_pct,
                "Kontribusi Balance Score": np.nan,
                "Produk representatif": "—",
                "Total transaksi representatif": np.nan,
            })

        st.subheader("Ringkasan anggota dan produk representatif")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.format({
                "Proporsi anggota (%)": "{:.2f}",
                "Kontribusi Balance Score": "{:.4f}",
                "Total transaksi representatif": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        st.subheader("Profilisasi deskriptif cluster")
        profile_df = pd.DataFrame(profile_rows)
        st.dataframe(
            profile_df.style.format({
                "Rata-rata penjualan mingguan": "{:,.3f}",
                "Total penjualan cluster": "{:,.0f}",
                "Volatilitas (CV)": "{:.3f}",
                "Proporsi anggota (%)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Produk representatif merupakan medoid deskriptif: anggota dengan total jarak "
            "terkecil terhadap anggota lain. Pada K-Means DBA, pusat matematisnya tetap centroid DBA."
        )

        if algorithm == "Hierarchical" and tree is not None:
            st.subheader("Dendrogram Hierarchical Clustering")
            fig_den, ax_den = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(12, 4))
            dendrogram(tree, truncate_mode="lastp", p=min(30, len(data)), leaf_rotation=90, ax=ax_den)
            ax_den.set_xlabel("Kelompok / observasi yang digabung")
            ax_den.set_ylabel("Jarak penggabungan")
            ax_den.set_title(f"Dendrogram ({linkage_method} linkage)")
            fig_den.tight_layout()
            st.pyplot(fig_den, clear_figure=True)

        st.subheader("Silhouette Score menurut jumlah cluster (K)")
        if algorithm != "DBSCAN":
            with st.spinner("Menghitung Silhouette K=2 sampai 8 menggunakan K-Medoids PAM..."):
                scree_df = silhouette_by_k(distance_matrix, range(2, 9))
            scree = px.line(
                scree_df, x="K", y="Silhouette", markers=True,
                color_discrete_sequence=["#2A817A"],
                title="Silhouette per K — K-Medoids PAM sebagai pembanding berbasis matriks jarak",
            )
            scree.add_vline(x=k, line_dash="dash", line_color="#B86F52", annotation_text=f"K terpilih = {k}")
            scree.update_layout(template="plotly_white", height=330, yaxis_title="Silhouette Score")
            st.plotly_chart(scree, use_container_width=True)
            st.caption(
                "Grafik ini merupakan scree pembanding K berbasis K-Medoids PAM pada matriks jarak terpilih; "
                "bukan hasil fit ulang algoritma yang sedang dipilih."
            )
        else:
            st.info("Grafik Silhouette per K tidak ditampilkan untuk DBSCAN karena DBSCAN tidak menggunakan parameter K.")

        st.divider()
        st.subheader("Profil deret waktu seluruh cluster")
        st.caption(
            "Seluruh cluster ditampilkan berdampingan. Garis tipis menunjukkan sampel anggota; "
            "garis tebal berwarna gold menunjukkan pusat/profil cluster."
        )
        display_series = series_derivative if key.startswith("DDTW") else series
        cluster_columns = st.columns(max(1, len(cluster_ids)))
        for col, lab in zip(cluster_columns, cluster_ids):
            profile, profile_label, member_idx = profile_by_cluster[lab]
            with col:
                st.markdown(f"**{display_names[lab]}**")
                st.caption(f"{len(member_idx):,} produk")
                chart = go.Figure()
                rng = np.random.default_rng(100 + lab)
                sample_idx = rng.choice(
                    member_idx, size=min(30, len(member_idx)), replace=False
                )
                for idx in sample_idx:
                    chart.add_trace(go.Scatter(
                        x=weeks, y=display_series[idx], mode="lines",
                        line=dict(color="rgba(100,112,128,0.22)", width=1),
                        showlegend=False,
                        hovertemplate=f"{codes[idx]}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
                    ))
                chart.add_trace(go.Scatter(
                    x=weeks, y=profile, mode="lines",
                    name=profile_label,
                    line=dict(color="#C49A52", width=4),
                    hovertemplate=f"{profile_label}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
                ))
                chart.update_layout(
                    template="plotly_white", height=360,
                    margin=dict(l=5, r=5, t=25, b=5),
                    xaxis_title="Minggu", yaxis_title=rep_label,
                    showlegend=False,
                )
                st.plotly_chart(chart, use_container_width=True, key=f"cluster_profile_{lab}_{key}_{algorithm}")

                member_table = pd.DataFrame({
                    "Product_Code": codes[member_idx],
                    "Total transaksi": x_raw[member_idx].sum(axis=1),
                    "Rata-rata mingguan": x_raw[member_idx].mean(axis=1),
                    "CV": np.divide(
                        x_raw[member_idx].std(axis=1),
                        x_raw[member_idx].mean(axis=1),
                        out=np.zeros(len(member_idx)),
                        where=x_raw[member_idx].mean(axis=1) != 0,
                    ),
                }).sort_values("Total transaksi", ascending=False)
                st.markdown("**Produk representatif**")
                st.code(codes[rep_indices[lab]])
                st.markdown("**Anggota cluster**")
                st.dataframe(member_table, use_container_width=True, hide_index=True, height=240)

        st.divider()
        st.subheader("Interpretasi dan rekomendasi bisnis")
        for lab in cluster_ids:
            member_idx = np.where(labels == lab)[0]
            weekly_profile = x_raw[member_idx].mean(axis=0)
            first_period = float(weekly_profile[:13].mean())
            last_period = float(weekly_profile[-13:].mean())
            change_pct = (
                (last_period - first_period) / abs(first_period) * 100
                if first_period != 0 else np.nan
            )
            cv = float(np.std(weekly_profile) / (np.mean(weekly_profile) + 1e-9))
            if np.isfinite(change_pct) and change_pct > 10:
                pattern = "Tren menguat"
                action = "Tinjau kenaikan kebutuhan dan pertimbangkan penyesuaian safety stock secara bertahap."
            elif np.isfinite(change_pct) and change_pct < -10:
                pattern = "Tren melemah"
                action = "Evaluasi reorder point dan hindari penambahan stok sebelum penurunan terkonfirmasi."
            else:
                pattern = "Relatif stabil"
                action = "Gunakan replenishment berkala dengan safety stock sesuai variasi historis."
            if cv > .75:
                action += " Volatilitas tinggi; pantau produk lebih sering dan evaluasi kebutuhan buffer."
            st.markdown(
                f'<div class="exec-card"><b>{display_names[lab]} · {pattern}</b><br>'
                f'<span style="color:#687386">{len(member_idx)} produk · perubahan rerata 13 minggu awal '
                f'ke 13 minggu akhir: {change_pct:.1f}% · CV profil: {cv:.3f}</span><br>{action}</div>',
                unsafe_allow_html=True,
            )

        export = pd.DataFrame({"Product_Code": codes, "Cluster_ID": labels})
        export["Cluster"] = [
            "Noise" if value == -1 else display_names[int(value)] for value in labels
        ]
        st.download_button(
            "Unduh label cluster (CSV)",
            export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cluster_labels_{key}_{algorithm.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # -----------------------------------------------------------------
    # PAGE 3 — DTW DISTANCE EXPLORER
    # -----------------------------------------------------------------
    elif page == "Eksplorasi Jarak DTW":
        st.header("Eksplorasi Jarak DTW")
        st.markdown(
            '<div class="page-intro">Bandingkan dua produk, lihat nilai jarak, dan amati bagaimana DTW menyelaraskan minggu yang tidak selalu berpasangan satu-ke-satu.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            product_a = st.selectbox("Produk A", codes.tolist(), index=0)
        with c2:
            product_b = st.selectbox("Produk B", codes.tolist(), index=min(1, len(codes)-1))
        with c3:
            alignment_radius = st.selectbox("Radius CDTW", [2, 3, 5], index=1)
        idx_a = int(np.where(codes == product_a)[0][0])
        idx_b = int(np.where(codes == product_b)[0][0])
        a_series = x_raw[idx_a].astype(float)
        b_series = x_raw[idx_b].astype(float)
        euclidean = float(np.linalg.norm(a_series - b_series))
        path_dtw, dtw_value = dtw_path(a_series, b_series)
        path_cdtw, cdtw_value = dtw_path(
            a_series, b_series,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=int(alignment_radius),
        )
        v1, v2, v3 = st.columns(3)
        v1.metric("Euclidean Distance", f"{euclidean:.3f}")
        v2.metric("DTW Unconstrained", f"{float(dtw_value):.3f}")
        v3.metric(f"CDTW · Radius {alignment_radius}", f"{float(cdtw_value):.3f}")
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=weeks, y=a_series, mode="lines+markers", name=product_a,
                                      line=dict(color="#2A817A", width=2.5)))
        line_fig.add_trace(go.Scatter(x=weeks, y=b_series, mode="lines+markers", name=product_b,
                                      line=dict(color="#B86F52", width=2.5)))
        line_fig.update_layout(template="plotly_white", height=350, title="Pola transaksi dua produk",
                               xaxis_title="Minggu", yaxis_title="Jumlah transaksi", hovermode="x unified")
        st.plotly_chart(line_fig, use_container_width=True)

        st.subheader("Week Alignment — DTW Unconstrained")
        align_fig = go.Figure()
        for i, j in path_dtw[::max(1, len(path_dtw)//70)]:
            align_fig.add_trace(go.Scatter(
                x=[weeks[i], weeks[j]], y=[a_series[i], b_series[j]],
                mode="lines", line=dict(color="rgba(130,140,150,0.28)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
        align_fig.add_trace(go.Scatter(x=weeks, y=a_series, mode="lines+markers", name=product_a,
                                       line=dict(color="#2A817A", width=2.5)))
        align_fig.add_trace(go.Scatter(x=weeks, y=b_series, mode="lines+markers", name=product_b,
                                       line=dict(color="#B86F52", width=2.5)))
        align_fig.update_layout(template="plotly_white", height=400, title="Garis penghubung pasangan waktu yang disejajarkan",
                                xaxis_title="Minggu", yaxis_title="Jumlah transaksi")
        st.plotly_chart(align_fig, use_container_width=True)

        st.subheader("Warping Path Grid dan Matriks Akumulasi Jarak")
        accumulated, local_cost = make_accumulated_cost(a_series, b_series)
        path_rows, path_cols = zip(*path_dtw)
        heat = go.Figure(data=go.Heatmap(
            z=accumulated,
            x=weeks,
            y=weeks,
            colorscale="YlGnBu",
            colorbar=dict(title="Akumulasi"),
        ))
        heat.add_trace(go.Scatter(
            x=[weeks[j] for i, j in path_dtw],
            y=[weeks[i] for i, j in path_dtw],
            mode="lines+markers",
            line=dict(color="#B86F52", width=3),
            marker=dict(size=4, color="#B86F52"),
            name="Warping path",
        ))
        heat.update_layout(
            template="plotly_white", height=600,
            title="Matriks akumulasi biaya DTW dengan warping path",
            xaxis_title=f"Minggu {product_b}",
            yaxis_title=f"Minggu {product_a}",
        )
        st.plotly_chart(heat, use_container_width=True)
        st.caption(
            "Heatmap menampilkan akumulasi squared difference pada grid waktu. Garis terracotta "
            "adalah jalur optimal DTW. Nilai jarak DTW dari tslearn menggunakan definisi normanya sendiri."
        )

    # -----------------------------------------------------------------
    # PAGE 4 — METHOD RECOMMENDATION
    # -----------------------------------------------------------------
    elif page == "Rekomendasi Metode":
        st.header("Rekomendasi Metode Klaster")
        st.markdown(
            '<div class="page-intro">Bandingkan beberapa algoritma pada matriks jarak yang dipilih. Rekomendasi berasal dari evaluasi empiris dataset ini.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Parameter perbandingan", expanded=True):
            r1, r2, r3 = st.columns(3)
            with r1:
                rec_rep_label = st.selectbox("Representasi", ["Data asli", "MinMax", "Scaled"], key="rec_rep")
                rec_rep = {"Data asli": "Original", "MinMax": "MinMax", "Scaled": "Scaled"}[rec_rep_label]
            with r2:
                rec_distance = st.selectbox("Metode jarak", ["DTW", "CDTW", "DDTW", "WDTW"], key="rec_distance")
                rec_radius = st.selectbox("Radius CDTW", [2, 3, 5], key="rec_radius") if rec_distance == "CDTW" else None
                rec_g = st.selectbox("Parameter g WDTW", [0.05, 0.1, 0.5, 1.0], key="rec_g") if rec_distance == "WDTW" else None
            with r3:
                rec_k = st.slider("Jumlah cluster pembanding", 2, 10, 3, key="rec_k")
                rec_linkage = st.selectbox("Linkage", ["average", "complete"], key="rec_linkage")
        rec_key = matrix_key(rec_rep, rec_distance, rec_radius, rec_g)
        rec_matrix = matrices.get(rec_key)
        if rec_matrix is None:
            st.error(f"Matriks {rec_key} tidak ditemukan.")
            st.stop()
        if st.button("Jalankan perbandingan metode", type="primary"):
            results = []
            coordinates = make_mds(rec_matrix)
            candidates = ["K-Medoids PAM", "Hierarchical Average", "Hierarchical Complete", "DBSCAN (eps otomatis)"]
            if rec_distance != "WDTW":
                candidates.insert(0, "K-Means DBA")
            series = row_minmax(x_raw) if rec_rep == "MinMax" else (
                StandardScaler().fit_transform(x_raw) if rec_rep == "Scaled" else x_raw.copy()
            )
            for candidate in candidates:
                started = time.perf_counter()
                try:
                    if candidate == "K-Means DBA":
                        params = {}
                        if rec_distance == "CDTW":
                            params = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": rec_radius}
                        fit_series = derivative_ts(series) if rec_distance == "DDTW" else series
                        model = TimeSeriesKMeans(
                            n_clusters=rec_k, metric="dtw", metric_params=params,
                            max_iter=8, max_iter_barycenter=8, n_init=1,
                            random_state=42, n_jobs=-1,
                        )
                        candidate_labels = model.fit_predict(fit_series[:, :, None])
                    elif candidate == "K-Medoids PAM":
                        candidate_labels, _ = kmedoids_pam(rec_matrix, rec_k, max_iter=70)
                    elif candidate.startswith("Hierarchical"):
                        method = "average" if candidate.endswith("Average") else "complete"
                        tree = linkage(squareform(rec_matrix, checks=False), method=method)
                        candidate_labels = cut_tree(tree, n_clusters=[rec_k]).ravel()
                    else:
                        min_s = 10
                        kth = np.sort(rec_matrix, axis=1)[:, min(min_s, rec_matrix.shape[0]-1)]
                        eps_auto = max(float(np.quantile(kth, .85)), 1e-12)
                        candidate_labels = DBSCAN(
                            eps=eps_auto, min_samples=min_s, metric="precomputed"
                        ).fit_predict(rec_matrix)
                    si, dbi, chi, bs, ncl, noise_n, noise_p, _ = evaluate_labels(
                        rec_matrix, coordinates, candidate_labels
                    )
                    results.append({
                        "Metode": candidate, "Silhouette": si, "DBI (MDS)": dbi,
                        "CHI (MDS)": chi, "Balance Score": bs,
                        "Jumlah cluster": ncl, "Noise (produk)": noise_n,
                        "Noise (%)": noise_p, "Waktu (detik)": time.perf_counter()-started,
                    })
                except Exception as exc:
                    results.append({
                        "Metode": candidate, "Silhouette": np.nan, "DBI (MDS)": np.nan,
                        "CHI (MDS)": np.nan, "Balance Score": np.nan,
                        "Jumlah cluster": 0, "Noise (produk)": np.nan,
                        "Noise (%)": np.nan, "Waktu (detik)": np.nan,
                        "Catatan": str(exc),
                    })
            result_frame = pd.DataFrame(results)
            valid = result_frame.dropna(subset=["Silhouette", "DBI (MDS)", "CHI (MDS)"]).copy()
            if len(valid):
                valid["Rank SI"] = valid["Silhouette"].rank(ascending=False, method="average")
                valid["Rank DBI"] = valid["DBI (MDS)"].rank(ascending=True, method="average")
                valid["Rank CHI"] = valid["CHI (MDS)"].rank(ascending=False, method="average")
                valid["Skor ranking"] = valid[["Rank SI", "Rank DBI", "Rank CHI"]].mean(axis=1)
                valid = valid.sort_values("Skor ranking")
                st.session_state["recommendation_result"] = valid
            else:
                st.session_state["recommendation_result"] = result_frame
            st.session_state["recommendation_key"] = rec_key

        if (
            st.session_state.get("recommendation_key") == rec_key
            and "recommendation_result" in st.session_state
        ):
            result_frame = st.session_state["recommendation_result"]
            if "Skor ranking" in result_frame.columns and result_frame["Skor ranking"].notna().any():
                best = result_frame.iloc[0]
                st.success(
                    f"Rekomendasi empiris untuk {rec_key}: **{best['Metode']}** "
                    f"(skor ranking {best['Skor ranking']:.2f})."
                )
                st.caption(
                    "Ranking memakai rata-rata peringkat Silhouette (tinggi lebih baik), "
                    "DBI (rendah lebih baik), dan CHI (tinggi lebih baik), dengan bobot sama."
                )
            else:
                st.warning("Belum ada metode dengan ketiga metrik evaluasi yang valid.")
            st.dataframe(result_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Unduh hasil perbandingan (CSV)",
                result_frame.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"recommendation_{rec_key}.csv",
                mime="text/csv",
            )
        st.info(
            "DBSCAN menggunakan eps otomatis dari persentil ke-85 jarak tetangga ke-10 "
            "dengan min_samples=10. DBSCAN tidak menetapkan K; jumlah cluster dan noise "
            "dapat berbeda. DBI dan CHI dihitung pada embedding MDS 5 dimensi."
        )

# ---------------------------------------------------------------------
# PAGE 5 — METHOD GUIDE
# ---------------------------------------------------------------------
elif page == "Panduan Metode":
    st.header("Panduan Metode dan Dokumentasi")
    st.markdown(
        '<div class="page-intro">Definisi metode dan panduan membaca hasil analisis.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Representasi data", expanded=True):
        st.markdown(
            """
            - **Data asli:** mempertahankan skala transaksi sebagaimana tercatat.
            - **MinMax:** menormalkan rentang setiap produk ke 0–1; fokus pada bentuk pola relatif.
            - **Scaled:** standardisasi per minggu/kolom, sehingga posisi relatif produk pada tiap minggu dibandingkan terhadap produk lain.
            """
        )
    with st.expander("Metode jarak"):
        st.markdown(
            """
            - **DTW:** mengukur kemiripan dua deret melalui penyelarasan waktu yang fleksibel.
            - **CDTW:** DTW dengan batas Sakoe–Chiba; radius membatasi pergeseran indeks waktu.
            - **DDTW:** DTW pada representasi turunan, lebih menekankan pola perubahan.
            - **WDTW:** memberi bobot pada pasangan waktu berdasarkan selisih indeks; parameter *g* mengatur bentuk pembobotan.
            """
        )
    with st.expander("Algoritma clustering"):
        st.markdown(
            """
            - **K-Means DBA:** membentuk centroid deret waktu menggunakan DTW Barycenter Averaging.
            - **K-Medoids PAM:** memilih produk aktual sebagai pusat cluster berdasarkan total jarak.
            - **Hierarchical:** membentuk cluster bertingkat; average memakai rata-rata jarak antaranggotanya, complete memakai jarak maksimum.
            - **DBSCAN:** mengelompokkan objek berdasarkan kepadatan; label -1 adalah noise.
            """
        )
    with st.expander("Metrik evaluasi"):
        st.markdown(
            """
            - **Silhouette:** semakin tinggi semakin baik; dihitung langsung dari matriks jarak.
            - **DBI:** semakin rendah semakin baik; pada dashboard ini dihitung pada embedding MDS 5 dimensi.
            - **CHI:** semakin tinggi semakin baik; pada dashboard ini dihitung pada embedding MDS 5 dimensi.
            - **Balance Score:** normalized entropy ukuran cluster. Nilai mendekati 1 berarti ukuran cluster lebih merata, bukan otomatis berarti pemisahan cluster lebih baik.
            """
        )
    with st.expander("Rentang interpretasi Silhouette"):
        st.markdown(
            """
            | Rentang | Interpretasi |
            |---|---|
            | 0.71–1.00 | Struktur Klaster Kuat (*Strong structure*) |
            | 0.51–0.70 | Struktur Klaster Baik (*Good/medium structure*) |
            | 0.26–0.50 | Struktur Klaster Lemah (*Weak structure*) |
            | ≤ 0.25 | Tidak Ada Struktur Jelas (*No structure / poor*) |
            """
        )
    st.markdown(
        '<div class="soft-note">Rekomendasi persediaan yang ditampilkan bersifat indikatif. '
        'Keputusan operasional perlu mempertimbangkan lead time, biaya penyimpanan, service level, '
        'stok aktual, dan faktor eksternal lainnya.</div>',
        unsafe_allow_html=True,
    )
