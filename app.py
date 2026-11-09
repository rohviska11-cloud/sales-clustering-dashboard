# app.py
# Dashboard Streamlit untuk clustering deret waktu transaksi mingguan.
# Letakkan app.py, Sales transaction data.xlsx, dan hasil_clustering_dtw/all_distance_matrices.npz
# dalam satu repository GitHub. Deploy melalui Streamlit Community Cloud dengan entrypoint app.py.

from pathlib import Path
import re
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
from tslearn.clustering import TimeSeriesKMeans

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Sales Time Series Clustering", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------- STYLE -----------------------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f5f7ff 0%,#eef7f7 55%,#fff7ed 100%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#14213d,#263f67); color:white;}
[data-testid="stSidebar"] * {color:white !important;}
.hero {padding:25px 28px;border-radius:20px;color:white;
       background:linear-gradient(110deg,#14213d,#315c91 58%,#16a085);
       box-shadow:0 8px 25px rgba(20,33,61,.18);margin-bottom:18px;}
.hero h1 {font-size:32px;margin:0 0 8px 0;color:white;}
.hero p {font-size:15px;margin:0;color:#e8f1ff;}
.section {padding:14px 18px;border-radius:15px;background:white;
          border:1px solid #e6eaf2;box-shadow:0 3px 14px rgba(30,50,80,.06);
          margin:8px 0 14px 0;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#fff,#f0f6ff);
  border:1px solid #dce7f5;padding:14px;border-radius:14px;}
.small-note {color:#65748b;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# --------------------------- DATA LOADING ----------------------------------
BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE / "Sales transaction data.xlsx",
    BASE / "Sales transaction data.csv",
    BASE / "sales_transaction_data.xlsx",
    BASE / "sales_transaction_data.csv",
    BASE / "streamlit_assets" / "Sales transaction data.xlsx",
    BASE / "streamlit_assets" / "Sales transaction data.csv",
]
MATRIX_PATH = BASE / "hasil_clustering_dtw" / "all_distance_matrices.npz"
if not MATRIX_PATH.exists():
    MATRIX_PATH = BASE / "all_distance_matrices.npz"
if not MATRIX_PATH.exists():
    MATRIX_PATH = BASE / "streamlit_assets" / "all_distance_matrices.npz"

@st.cache_data(show_spinner=False)
def load_sales(file_bytes, suffix):
    from io import BytesIO
    bio = BytesIO(file_bytes)
    data = pd.read_excel(bio) if suffix in [".xlsx", ".xls"] else pd.read_csv(bio)
    weekly = [f"W{i}" for i in range(52)]
    if "Product_Code" not in data.columns:
        raise ValueError("Kolom Product_Code tidak ditemukan.")
    missing = [c for c in weekly if c not in data.columns]
    if missing:
        raise ValueError(f"Kolom mingguan tidak lengkap: {missing}")
    data = data.drop_duplicates("Product_Code").copy()
    data[weekly] = data[weekly].apply(pd.to_numeric, errors="coerce")
    # Imputasi per minggu menggunakan median; jika satu minggu seluruhnya kosong -> 0.
    data[weekly] = data[weekly].replace([np.inf, -np.inf], np.nan)
    data[weekly] = data[weekly].fillna(data[weekly].median()).fillna(0)
    return data, weekly

@st.cache_resource(show_spinner=False)
def load_matrices(path, mtime):
    if not Path(path).exists():
        return {}
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "product_code"}
    if "product_code" in z.files:
        out["__product_code__"] = z["product_code"].astype(str)
    return out

@st.cache_data(show_spinner=False)
def make_mds(D):
    model = MDS(n_components=5, metric=True, dissimilarity="precomputed",
                random_state=42, n_init=1, max_iter=200)
    return model.fit_transform(D)

def row_minmax(X):
    lo = np.nanmin(X, axis=1, keepdims=True)
    hi = np.nanmax(X, axis=1, keepdims=True)
    span = hi - lo
    return np.divide(X-lo, span, out=np.zeros_like(X), where=span != 0)

def get_matrix_name(representation, distance, radius=None, g=None):
    if distance == "DTW":
        return f"DTW_{representation}"
    if distance == "CDTW":
        return f"CDTW_{representation}_R{radius}"
    if distance == "DDTW":
        return f"DDTW_{representation}"
    return f"WDTW_{representation}_g{g}"

def cluster_data(algo, D, X, X_deriv, k, linkage_method, eps, min_samples,
                 matrix_name):
    start = time.perf_counter()
    model = None
    if algo == "K-Means":
        if matrix_name.startswith("WDTW"):
            raise ValueError("K-Means DBA untuk WDTW belum didukung tslearn. Pilih K-Medoids, Hierarchical, atau DBSCAN.")
        use_x = X_deriv if matrix_name.startswith("DDTW") else X
        params = {}
        if matrix_name.startswith("CDTW"):
            radius = int(matrix_name.rsplit("R", 1)[1])
            params = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": radius}
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", metric_params=params,
             max_iter=10, max_iter_barycenter=10, n_init=2,
             random_state=42, n_jobs=-1)
        labels = model.fit_predict(use_x[:, :, None])
        centers = model.cluster_centers_[:, :, 0]
        center_space = "centroid DBA"
    elif algo == "K-Medoids":
        model = KMedoids(n_clusters=k, metric="precomputed", method="pam",
                         init="k-medoids++", random_state=42)
        labels = model.fit_predict(D)
        centers = X[model.medoid_indices_]
        center_space = "medoid (produk representatif)"
    elif algo == "Hierarchical":
        Z = linkage(squareform(D, checks=False), method=linkage_method)
        labels = cut_tree(Z, n_clusters=[k]).ravel()
        centers = np.vstack([np.median(X[labels == c], axis=0) for c in sorted(np.unique(labels))])
        center_space = "median profil cluster (deskriptif)"
    else:
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed")
        labels = model.fit_predict(D)
        clusters = sorted(c for c in np.unique(labels) if c != -1)
        centers = np.vstack([np.median(X[labels == c], axis=0) for c in clusters]) if clusters else np.empty((0, X.shape[1]))
        center_space = "median profil cluster (deskriptif)"
    runtime = time.perf_counter() - start
    return labels, centers, center_space, runtime

def score_clusters(D, coords, labels):
    mask = labels != -1
    y = labels[mask]
    ncl = len(np.unique(y))
    n = len(y)
    si = dbi = chi = np.nan
    if 2 <= ncl < n:
        si = silhouette_score(D[np.ix_(mask, mask)], y, metric="precomputed")
        dbi = davies_bouldin_score(coords[mask], y)
        chi = calinski_harabasz_score(coords[mask], y)
    return si, dbi, chi, int(ncl), float((labels == -1).mean()*100)

# --------------------------- SIDEBAR ---------------------------------------
st.sidebar.title("🎛️ Panel Eksperimen")
st.sidebar.caption("Pilih skenario clustering untuk dieksplorasi.")
representation = st.sidebar.selectbox("Representasi data", ["Original","MinMax","Scaled"])
distance = st.sidebar.selectbox("Metode jarak", ["DTW","CDTW","DDTW","WDTW"])
radius = st.sidebar.selectbox("CDTW — Sakoe-Chiba radius", [2,3,5]) if distance == "CDTW" else None
g = st.sidebar.selectbox("WDTW — parameter g", [0.05,0.1,0.5,1.0]) if distance == "WDTW" else None
algorithm = st.sidebar.selectbox("Algoritma klaster", ["K-Means","K-Medoids","Hierarchical","DBSCAN"])
k = st.sidebar.slider("Jumlah klaster (K)", 2, 10, 3) if algorithm != "DBSCAN" else None
linkage_method = st.sidebar.selectbox("Linkage", ["average","complete"]) if algorithm == "Hierarchical" else "average"
eps = st.sidebar.number_input("DBSCAN — eps", min_value=0.000001, value=1.0, step=0.1, format="%.6f") if algorithm == "DBSCAN" else None
min_samples = st.sidebar.select_slider("DBSCAN — min_samples", options=[5,10,15,20], value=10) if algorithm == "DBSCAN" else None
sample_lines = st.sidebar.slider("Maksimum garis produk per klaster", 5, 100, 25, 5)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi", ["Dashboard","About & Dokumentasi"])
uploaded = st.sidebar.file_uploader("Unggah dataset alternatif (.xlsx/.csv)", type=["xlsx","xls","csv"])

# Resolve dataset
source_file = uploaded
if source_file is None:
    source_file = next((p for p in DATA_CANDIDATES if p.exists()), None)
if source_file is None:
    st.error("Dataset tidak ditemukan. Letakkan Sales transaction data.xlsx/csv di folder aplikasi atau unggah melalui sidebar.")
    st.stop()
suffix = Path(source_file.name).suffix.lower() if uploaded else source_file.suffix.lower()
raw_bytes = source_file.getvalue() if uploaded else source_file.read_bytes()
try:
    data, weekly_cols = load_sales(raw_bytes, suffix)
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

X_original = data[weekly_cols].to_numpy(float)
if representation == "Original":
    X = X_original.copy()
elif representation == "MinMax":
    norm_cols = [f"Normalized {i}" for i in range(52)]
    if all(c in data.columns for c in norm_cols):
        X = data[norm_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    else:
        X = row_minmax(X_original)
else:
    # Standarisasi per minggu/kolom, konsisten dengan notebook eksperimen.
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X_original)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
def derivative_ts_app(A):
    out = np.zeros_like(A, dtype=float)
    out[:, 0] = A[:, 1] - A[:, 0]
    for t in range(1, A.shape[1]-1):
        out[:, t] = 0.5*(A[:, t]-A[:, t-1]) + 0.25*(A[:, t+1]-A[:, t-1])
    out[:, -1] = A[:, -1] - A[:, -2]
    return out
X_deriv = derivative_ts_app(X)
codes = data["Product_Code"].astype(str).to_numpy()

# Matrices
matrices = load_matrices(str(MATRIX_PATH), MATRIX_PATH.stat().st_mtime if MATRIX_PATH.exists() else 0)
matrix_name = get_matrix_name(representation, distance, radius, g)
saved_codes = matrices.get("__product_code__")
if saved_codes is not None and not np.array_equal(saved_codes.astype(str), codes.astype(str)):
    st.error("Urutan atau daftar Product_Code pada dataset tidak sama dengan dataset saat matriks jarak dihitung. Gunakan dataset asli yang sama dengan notebook eksperimen.")
    st.stop()
D = matrices.get(matrix_name)
if D is None:
    st.error(f"Matriks `{matrix_name}` tidak tersedia. Jalankan notebook eksperimen hingga ekspor all_distance_matrices.npz, lalu letakkan file tersebut di {MATRIX_PATH}.")
    st.stop()
if D.shape != (len(data), len(data)):
    st.error(f"Ukuran matriks {matrix_name} ({D.shape}) tidak cocok dengan dataset ({len(data)}, {len(data)}). Pastikan urutan dan isi produk sama dengan saat eksperimen.")
    st.stop()

# --------------------------- HEADER ----------------------------------------
st.markdown("""<div class="hero"><h1>📈 Sales Time Series Clustering</h1>
<p>Eksplorasi pola transaksi mingguan, evaluasi kualitas cluster, dan rekomendasi persediaan berbasis perilaku produk.</p></div>""", unsafe_allow_html=True)

if page == "About & Dokumentasi":
    st.title("📚 About & Dokumentasi")
    st.subheader("Metode jarak")
    st.markdown("""
    - **DTW (Dynamic Time Warping):** mengukur kemiripan dua deret dengan mengizinkan penyelarasan waktu yang fleksibel.
    - **CDTW:** DTW dengan batas Sakoe–Chiba agar penyelarasan tidak bergeser terlalu jauh; radius 2, 3, atau 5 membatasi lebar jalur.
    - **DDTW:** DTW diterapkan pada representasi turunan sehingga fokus pada perubahan/pola naik-turun, bukan hanya level transaksi.
    - **WDTW:** memberi bobot berbeda pada pasangan waktu yang berjauhan; parameter *g* mengatur bentuk pembobotan.
    """)
    st.subheader("Algoritma")
    st.markdown("""
    - **K-Means DBA:** membentuk cluster dan memperbarui centroid deret waktu dengan Dynamic Time Warping Barycenter Averaging.
    - **K-Medoids PAM:** memilih produk aktual sebagai medoid; menerima matriks jarak precomputed.
    - **Hierarchical:** menggabungkan cluster bertahap. *Average* memakai rata-rata jarak antaranggotanya, *complete* memakai jarak maksimum.
    - **DBSCAN:** membentuk cluster berdasarkan kepadatan; label `-1` berarti noise/tidak teralokasi.
    """)
    st.subheader("Metrik evaluasi")
    st.markdown("""
    - **Silhouette:** semakin mendekati 1 semakin baik pemisahan dan kekompakan; nilai negatif mengindikasikan potensi salah penempatan.
    - **DBI:** semakin kecil semakin baik; membandingkan dispersi dalam cluster dengan jarak antarklaster.
    - **CHI:** semakin besar semakin baik; membandingkan variasi antarcluster terhadap variasi dalam cluster.

    Catatan metodologis: Silhouette dihitung langsung dari matriks jarak yang dipilih. DBI dan CHI dihitung pada embedding MDS 5 dimensi, sehingga merupakan metrik pendukung pada ruang MDS, bukan ukuran langsung pada ruang DTW.
    """)
    st.info("Untuk WDTW, K-Means DBA tidak ditawarkan karena tslearn tidak menyediakan WDTW barycenter bawaan. Jangan mengganti metode ini diam-diam dengan K-Means pada koordinat MDS.")
    st.stop()

# --------------------------- SECTION 1 -------------------------------------
st.header("1. Executive Summary & Descriptive Dashboard")
total_products = len(data)
total_sales = float(X_original.sum())
avg_weekly = X_original.mean(axis=0)
peak_week = int(np.argmax(avg_weekly))
growth = (avg_weekly[-1] - avg_weekly[0]) / abs(avg_weekly[0]) * 100 if avg_weekly[0] != 0 else np.nan
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total produk", f"{total_products:,}")
c2.metric("Total transaksi tercatat", f"{total_sales:,.0f}")
c3.metric("Rata-rata per produk/minggu", f"{X_original.mean():,.2f}")
c4.metric("Minggu rata-rata tertinggi", f"W{peak_week}", f"{avg_weekly[peak_week]:,.2f}")
trend = pd.DataFrame({"Minggu":weekly_cols,"Rata-rata transaksi":avg_weekly,
                      "Total transaksi":X_original.sum(axis=0)})
fig_trend = px.line(trend, x="Minggu", y="Rata-rata transaksi", markers=True,
                    title="Tren rata-rata transaksi seluruh produk")
fig_trend.update_traces(line=dict(width=3,color="#087f8c"))
fig_trend.update_layout(template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)
top_idx = np.argsort(X_original.sum(axis=1))[::-1][:10]
top_df = pd.DataFrame({"Product_Code":codes[top_idx],
                       "Total transaksi":X_original.sum(axis=1)[top_idx]})
left,right = st.columns([1,1])
with left:
    st.subheader("🏆 Top 10 produk berdasarkan total transaksi")
    st.dataframe(top_df, use_container_width=True, hide_index=True)
with right:
    st.subheader("Distribusi total transaksi produk")
    hist = px.histogram(pd.DataFrame({"Total transaksi":X_original.sum(axis=1)}),
                        x="Total transaksi", nbins=35, color_discrete_sequence=["#f4a261"])
    hist.update_layout(template="plotly_white")
    st.plotly_chart(hist, use_container_width=True)

# --------------------------- CLUSTERING ------------------------------------
st.header("2. Hasil Klaster & Evaluasi Metrik")
if algorithm == "K-Means" and distance == "WDTW":
    st.warning("K-Means DBA + WDTW tidak tersedia. Silakan pilih algoritma lain.")
    st.stop()
try:
    with st.spinner("Menjalankan clustering dan menghitung evaluasi..."):
        labels, centers, center_space, runtime = cluster_data(
            algorithm,D,X,X_deriv,k,linkage_method,eps,min_samples,matrix_name)
        coords = make_mds(D)
        si,dbi,chi,n_clusters,noise_pct = score_clusters(D,coords,labels)
except Exception as e:
    st.error(f"Clustering gagal: {e}")
    st.stop()

m1,m2,m3,m4 = st.columns(4)
m1.metric("Silhouette", "—" if np.isnan(si) else f"{si:.4f}",
          help="Dihitung langsung dari matriks jarak terpilih.")
m2.metric("Davies-Bouldin (MDS)", "—" if np.isnan(dbi) else f"{dbi:.4f}",
          help="Dihitung pada embedding MDS 5 dimensi.")
m3.metric("Calinski-Harabasz (MDS)", "—" if np.isnan(chi) else f"{chi:,.2f}",
          help="Dihitung pada embedding MDS 5 dimensi.")
m4.metric("Jumlah cluster", n_clusters, f"Noise {noise_pct:.2f}%")
st.caption(f"Model: **{algorithm}** | Matriks: **{matrix_name}** | Waktu komputasi: **{runtime:.2f} detik**")

unique_labels = sorted(np.unique(labels))
dist_rows=[]
for lab in unique_labels:
    mask=labels==lab
    dist_rows.append({"Cluster":"Noise (-1)" if lab==-1 else f"Cluster {lab+1}",
                      "Jumlah produk":int(mask.sum()),
                      "Proporsi (%)":float(mask.mean()*100)})
dist_df=pd.DataFrame(dist_rows)
st.subheader("Distribusi anggota cluster")
col_a,col_b=st.columns([1,1])
with col_a:
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
with col_b:
    fig_dist=px.pie(dist_df,names="Cluster",values="Jumlah produk",hole=.48,
                    color_discrete_sequence=px.colors.qualitative.Bold)
    fig_dist.update_layout(template="plotly_white")
    st.plotly_chart(fig_dist,use_container_width=True)

# --------------------------- SECTION 3 -------------------------------------
st.header("3. Visualisasi Klaster Time Series")
st.caption(f"Profil pusat: {center_space}. Untuk Hierarchical/DBSCAN, garis pusat adalah median profil anggota cluster, bukan centroid DTW.")
fig=go.Figure()
cluster_ids=[c for c in unique_labels if c!=-1]
# Plot centers / representative profiles
for j,lab in enumerate(cluster_ids):
    center = centers[j] if j < len(centers) else np.median(X[labels==lab],axis=0)
    fig.add_trace(go.Scatter(x=weekly_cols,y=center,mode="lines+markers",
        name=f"Pusat Cluster {lab+1}",line=dict(width=4)))
    members=np.where(labels==lab)[0]
    if len(members):
        rng=np.random.default_rng(42+int(lab))
        chosen=rng.choice(members,size=min(sample_lines,len(members)),replace=False)
        for idx in chosen:
            fig.add_trace(go.Scatter(x=weekly_cols,y=X[idx],mode="lines",
                line=dict(width=1,color="rgba(120,130,150,0.20)"),
                opacity=.55,showlegend=False,
                hovertemplate=f"Produk {codes[idx]}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
# Noise as a separate trace group, if any
if np.any(labels==-1):
    members=np.where(labels==-1)[0]
    rng=np.random.default_rng(999)
    for idx in rng.choice(members,size=min(sample_lines,len(members)),replace=False):
        fig.add_trace(go.Scatter(x=weekly_cols,y=X[idx],mode="lines",
            name="Noise (contoh)" if idx==members[0] else None,
            line=dict(width=1,dash="dot",color="rgba(120,120,120,.35)"),
            showlegend=(idx==members[0])))
fig.update_layout(template="plotly_white",height=560,
    title="Pola transaksi mingguan per cluster",
    xaxis_title="Minggu",yaxis_title=f"Nilai transaksi ({representation})",
    hovermode="x unified",legend_title="Cluster / profil")
st.plotly_chart(fig,use_container_width=True)

# --------------------------- SECTION 4 -------------------------------------
st.header("4. Business Insights & Actionable Recommendations")
recommendations=[]
for lab in cluster_ids:
    member_idx=np.where(labels==lab)[0]
    profile=np.mean(X_original[member_idx],axis=0)
    first=max(float(np.mean(profile[:13])),0)
    last=float(np.mean(profile[-13:]))
    change=(last-first)/first*100 if first>0 else np.nan
    cv=float(np.std(profile)/(np.mean(profile)+1e-9))
    if np.isfinite(change) and change>10:
        trend_label="cenderung meningkat"
        action="Pertimbangkan peningkatan safety stock secara bertahap, pantau stock-out, dan validasi kenaikan dengan data periode terbaru."
    elif np.isfinite(change) and change<-10:
        trend_label="cenderung menurun"
        action="Tahan penambahan stok, evaluasi reorder point, dan susun promosi atau pengurangan persediaan secara bertahap."
    else:
        trend_label="relatif stabil"
        action="Gunakan replenishment rutin berbasis rata-rata kebutuhan; sesuaikan safety stock dengan variasi permintaan."
    if cv>0.75:
        action += " Variasi pola relatif tinggi, sehingga gunakan buffer stok dan pemantauan lebih sering."
    recommendations.append({"Cluster":f"Cluster {lab+1}","Produk":len(member_idx),
        "Perubahan rata-rata (13 minggu awal ke akhir)":change,
        "Karakteristik":trend_label,"Rekomendasi":action})
for rec in recommendations:
    with st.container(border=True):
        st.subheader(f"💡 {rec['Cluster']} — {rec['Karakteristik']}")
        st.write(f"**Jumlah produk:** {rec['Produk']} | **Perubahan profil:** " +
                 (f"{rec['Perubahan rata-rata (13 minggu awal ke akhir)']:.1f}%" if np.isfinite(rec["Perubahan rata-rata (13 minggu awal ke akhir)"]) else "tidak terdefinisi"))
        st.write(rec["Rekomendasi"])
if np.any(labels==-1):
    st.info(f"{int((labels==-1).sum())} produk ({noise_pct:.2f}%) berstatus noise pada DBSCAN. Periksa produk ini secara terpisah sebelum keputusan stok.")
st.caption("Rekomendasi bersifat indikatif dari pola historis, bukan forecast permintaan atau keputusan pengadaan otomatis.")

# --------------------------- EXPORT ----------------------------------------
with st.expander("Unduh hasil clustering yang sedang ditampilkan"):
    export = pd.DataFrame({"Product_Code":codes,"Cluster":labels})
    st.download_button("⬇️ Download label cluster (CSV)",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"labels_{matrix_name}_{algorithm.replace('-','')}.csv",
        mime="text/csv")
    st.download_button("⬇️ Download distribusi cluster (CSV)",
        dist_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="distribusi_cluster.csv",mime="text/csv")
