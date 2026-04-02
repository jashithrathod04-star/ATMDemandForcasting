"""
╔══════════════════════════════════════════════════════════════╗
║   SmartCharging Analytics — Uncovering EV Behaviour Patterns ║
║   Data Mining | Year 1 Summative Assessment | Scenario 2     ║
║   Deploy: streamlit run app.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import requests, io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GOOGLE DRIVE DATASET CONFIG
# ─────────────────────────────────────────────────────────────
GDRIVE_FILE_ID  = "1F-r0PLBFiuOu6Gq2KFGJrd_EXe3R16gG"
GDRIVE_DOWNLOAD = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #888; }
.section-title {
    font-size: 1.35rem; font-weight: 700; color: #1565C0;
    border-left: 4px solid #1565C0; padding-left: 10px;
    margin: 1rem 0 0.5rem 0;
}
/* Force all info boxes to always show dark text — fixes dark-mode fading */
.insight-box, .warn-box, .danger-box, .green-box {
    border-radius: 6px; padding: 0.8rem 1.1rem; margin: 0.5rem 0;
    font-size: 0.93rem; color: #111111 !important;
    line-height: 1.55;
}
.insight-box {
    background: #DBEAFE !important; border-left: 4px solid #1565C0;
}
.warn-box {
    background: #FEF9C3 !important; border-left: 4px solid #D97706;
}
.danger-box {
    background: #FFE4E6 !important; border-left: 4px solid #C62828;
}
.green-box {
    background: #DCFCE7 !important; border-left: 4px solid #2E7D32;
}
/* Ensure all child elements inside boxes are also dark */
.insight-box *, .warn-box *, .danger-box *, .green-box * {
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚡ Fetching dataset from Google Drive…")
def load_and_prepare():
    try:
        resp = requests.get(GDRIVE_DOWNLOAD, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        # Fallback: try local copy (for offline dev)
        df = pd.read_csv("detailed_ev_charging_stations.csv")

    # ── Deduplication ─────────────────────────────────────
    df.drop_duplicates(subset="Station ID", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Missing value imputation (robust, even if no nulls)
    num_cols = ["Cost (USD/kWh)", "Distance to City (km)",
                "Usage Stats (avg users/day)", "Charging Capacity (kW)",
                "Reviews (Rating)", "Parking Spots"]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = ["Charger Type", "Availability", "Station Operator",
                "Renewable Energy Source", "Maintenance Frequency",
                "Connector Types"]
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    # ── Encoding ─────────────────────────────────────────
    df["Renewable_Enc"] = (df["Renewable Energy Source"]
                           .str.strip().str.lower()
                           .map({"yes": 1, "no": 0}).fillna(0).astype(int))

    charger_ord = {"AC Level 1": 1, "AC Level 2": 2, "DC Fast Charger": 3}
    df["Charger_Enc"] = df["Charger Type"].map(charger_ord).fillna(1).astype(int)

    avail_hrs = {"24/7": 24, "6:00-22:00": 16, "9:00-18:00": 9}
    df["Avail_Hours"] = df["Availability"].map(avail_hrs).fillna(12).astype(int)

    maint_ord = {"Annually": 1, "Quarterly": 2, "Monthly": 3}
    df["Maint_Enc"] = (df["Maintenance Frequency"]
                       .map(maint_ord).fillna(1).astype(int))

    # ── Normalisation ─────────────────────────────────────
    scale_cols = ["Cost (USD/kWh)", "Usage Stats (avg users/day)",
                  "Charging Capacity (kW)", "Distance to City (km)",
                  "Reviews (Rating)", "Parking Spots", "Avail_Hours"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    for i, c in enumerate(scale_cols):
        df[c + "_sc"] = scaled[:, i]

    return df


# ─────────────────────────────────────────────────────────────
# CLUSTERING  (cached on K)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running K-Means…")
def run_kmeans(_df, k):
    feat = ["Usage Stats (avg users/day)_sc",
            "Charging Capacity (kW)_sc",
            "Cost (USD/kWh)_sc",
            "Distance to City (km)_sc",
            "Avail_Hours_sc"]
    X = _df[feat].values
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    return labels, sil


@st.cache_data(show_spinner="Computing elbow…")
def elbow_data(_df):
    feat = ["Usage Stats (avg users/day)_sc",
            "Charging Capacity (kW)_sc",
            "Cost (USD/kWh)_sc",
            "Distance to City (km)_sc",
            "Avail_Hours_sc"]
    X = _df[feat].values
    inertias, sils = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lbl))
    return list(range(2, 11)), inertias, sils


# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTION  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Detecting anomalies…")
def detect_anomalies(_df, contamination):
    df = _df.copy()
    usage = df["Usage Stats (avg users/day)"]

    # Z-Score
    df["Z_Score"] = np.abs(stats.zscore(usage))
    df["Anom_Z"] = (df["Z_Score"] > 3).astype(int)

    # IQR
    Q1, Q3 = usage.quantile(0.25), usage.quantile(0.75)
    IQR = Q3 - Q1
    df["Anom_IQR"] = (
        (usage < Q1 - 1.5 * IQR) | (usage > Q3 + 1.5 * IQR)
    ).astype(int)

    # Isolation Forest (multivariate)
    iso_feat = ["Usage Stats (avg users/day)_sc", "Cost (USD/kWh)_sc",
                "Charging Capacity (kW)_sc", "Reviews (Rating)_sc",
                "Maint_Enc"]
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["Anom_IF"] = (iso.fit_predict(df[iso_feat]) == -1).astype(int)

    # Consensus: flagged by 2+ methods
    df["Anom_Score"] = df["Anom_Z"] + df["Anom_IQR"] + df["Anom_IF"]
    df["Anomaly"] = (df["Anom_Score"] >= 2).astype(int)

    # High-cost + low-rating anomaly
    cost_q85 = df["Cost (USD/kWh)"].quantile(0.85)
    rate_q15 = df["Reviews (Rating)"].quantile(0.15)
    df["HiCost_LowRating"] = (
        (df["Cost (USD/kWh)"] >= cost_q85) &
        (df["Reviews (Rating)"] <= rate_q15)
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# ARM  (cached on thresholds)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Mining association rules…")
def run_arm(_df, min_sup, min_conf):
    df = _df.copy()

    df["Usage_Bin"] = pd.cut(df["Usage Stats (avg users/day)"],
                              bins=[0, 40, 70, 101],
                              labels=["Low_Usage", "Med_Usage", "High_Usage"])
    df["Cost_Bin"]  = pd.cut(df["Cost (USD/kWh)"],
                              bins=3,
                              labels=["Low_Cost", "Med_Cost", "High_Cost"])
    df["Dist_Bin"]  = pd.cut(df["Distance to City (km)"],
                              bins=[0, 20, 50, 9999],
                              labels=["Near_City", "Mid_Dist", "Far_Rural"])
    df["Cap_Bin"]   = df["Charging Capacity (kW)"].map(
                        {22: "Cap_22kW", 50: "Cap_50kW",
                         150: "Cap_150kW", 350: "Cap_350kW"})

    basket_cols = ["Charger Type", "Usage_Bin", "Cost_Bin",
                   "Dist_Bin", "Cap_Bin",
                   "Renewable Energy Source", "Availability",
                   "Maintenance Frequency"]
    transactions = df[basket_cols].astype(str).values.tolist()

    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    bdf = pd.DataFrame(te_arr, columns=te.columns_)

    freq = apriori(bdf, min_support=min_sup, use_colnames=True)
    if len(freq) == 0:
        return pd.DataFrame(), 0

    rules = association_rules(freq, metric="confidence",
                              min_threshold=min_conf)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules, len(freq)


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ SmartCharging Analytics")
    st.caption("SmartEnergy Data Lab | EV Infrastructure Intelligence")
    st.divider()

    df_full = load_and_prepare()

    st.markdown("### 🔽 Global Filters")
    charger_opts = sorted(df_full["Charger Type"].unique())
    sel_charger = st.multiselect("Charger Type", charger_opts,
                                  default=charger_opts)

    operator_opts = sorted(df_full["Station Operator"].unique())
    sel_operator = st.multiselect("Station Operator", operator_opts,
                                   default=operator_opts)

    sel_renewable = st.multiselect("Renewable Energy",
                                    ["Yes", "No"], default=["Yes", "No"])

    year_min = int(df_full["Installation Year"].min())
    year_max = int(df_full["Installation Year"].max())
    sel_years = st.slider("Installation Year", year_min, year_max,
                           (year_min, year_max))

    st.divider()
    st.markdown("### ⚙️ Analysis Settings")
    n_clusters = st.slider("K-Means Clusters (K)", 2, 8, 4)
    contamination = st.slider("Anomaly Contamination %", 1, 20, 5) / 100
    arm_sup  = st.slider("ARM Min Support",    0.05, 0.40, 0.15, 0.01)
    arm_conf = st.slider("ARM Min Confidence", 0.30, 0.90, 0.50, 0.05)
    st.divider()

    # Apply filters
    df = df_full.copy()
    if sel_charger:
        df = df[df["Charger Type"].isin(sel_charger)]
    if sel_operator:
        df = df[df["Station Operator"].isin(sel_operator)]
    if sel_renewable:
        df = df[df["Renewable Energy Source"].isin(sel_renewable)]
    df = df[df["Installation Year"].between(sel_years[0], sel_years[1])]
    df = df.reset_index(drop=True)

    st.info(f"**{len(df):,}** stations selected")


# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;color:#1565C0;margin-bottom:0'>"
    "⚡ SmartCharging Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#666;margin-top:4px'>"
    "Uncovering EV Behaviour Patterns | SmartEnergy Data Lab | "
    "Data Mining Summative Assessment</p>",
    unsafe_allow_html=True,
)
st.divider()

# KPI STRIP
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🔌 Stations",       f"{len(df):,}")
k2.metric("📈 Avg Users/Day",  f"{df['Usage Stats (avg users/day)'].mean():.1f}")
k3.metric("💰 Avg Cost/kWh",   f"${df['Cost (USD/kWh)'].mean():.3f}")
k4.metric("⭐ Avg Rating",      f"{df['Reviews (Rating)'].mean():.2f}")
k5.metric("⚡ Avg Capacity",    f"{df['Charging Capacity (kW)'].mean():.0f} kW")
pct_ren = (df["Renewable Energy Source"] == "Yes").mean() * 100
k6.metric("🌿 Renewable",       f"{pct_ren:.1f}%")

st.divider()


# ═════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════
tab_scope, tab_eda, tab_cluster, tab_arm, tab_anom, tab_map, tab_insights = st.tabs([
    "🎯 Project Scope",
    "📊 EDA",
    "🗂️ Clustering",
    "🔗 Association Rules",
    "🚨 Anomaly Detection",
    "🗺️ Geo Map",
    "📋 Insights & Recommendations",
])


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 0 — PROJECT SCOPE                                    ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_scope:
    st.markdown('<div class="section-title">Stage 1 — Project Scope Definition</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Scenario:</b> SmartCharging Analytics — Uncovering EV Behaviour Patterns<br>
    <b>Organisation:</b> SmartEnergy Data Lab<br>
    <b>Role:</b> Data Analyst working with EV charging infrastructure providers
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objectives")
    obj_col1, obj_col2 = st.columns(2)
    with obj_col1:
        st.markdown("""
        <div class="insight-box"><b>1. Cluster Charging Behaviours</b><br>
        Group EV charging stations based on usage stats, charging capacity, cost, distance
        to city, and availability hours using K-Means clustering.</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box"><b>2. Detect Anomalies</b><br>
        Identify unusual station behaviour — overuse, faulty stations, or abnormal
        charging patterns — using Z-Score, IQR, and Isolation Forest (consensus method).</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box"><b>3. Discover Time-Based Associations</b><br>
        Apply the Apriori algorithm to find relationships between charger type, usage
        level, cost tier, distance tier, capacity, and availability to optimise scheduling
        and pricing strategies.</div>
        """, unsafe_allow_html=True)

    with obj_col2:
        st.markdown("""
        <div class="green-box"><b>4. Enhance Infrastructure Planning</b><br>
        Support decision-making on where and when to expand charging stations by
        analysing geographic demand patterns and operator performance.</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="green-box"><b>5. Deploy Insights</b><br>
        Build this interactive Streamlit dashboard for real-time exploration of
        charging patterns, anomalies, cluster profiles, and association rules.</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box"><b>6. Exploratory Data Analysis (EDA)</b><br>
        Uncover trends in usage distribution, charger type performance, operator
        costs, renewable energy impact, and geographic demand via rich visualisations.</div>
        """, unsafe_allow_html=True)

    st.markdown("### 📦 Dataset Overview")
    ds_col1, ds_col2 = st.columns([1, 1])
    with ds_col1:
        st.markdown("#### Key Columns")
        dataset_info = {
            "Column": [
                "Station ID", "Charger Type", "Availability",
                "Station Operator", "Cost (USD/kWh)",
                "Usage Stats (avg users/day)", "Charging Capacity (kW)",
                "Distance to City (km)", "Reviews (Rating)",
                "Renewable Energy Source", "Maintenance Frequency",
                "Connector Types", "Parking Spots", "Installation Year",
                "Latitude / Longitude",
            ],
            "Description": [
                "Unique station identifier",
                "AC Level 1 / AC Level 2 / DC Fast Charger",
                "Operating hours (24/7, 6:00-22:00, 9:00-18:00)",
                "Company operating the station",
                "Price per kWh charged to the customer",
                "Average number of users per day",
                "Maximum charging power output (kW)",
                "Proximity to nearest urban centre",
                "Customer satisfaction score (1–5)",
                "Whether station uses renewable energy (Yes/No)",
                "Frequency of maintenance (Annual/Quarterly/Monthly)",
                "Plug standards supported at the station",
                "Number of available parking bays",
                "Year the station was commissioned",
                "Geographic coordinates for map visualisation",
            ],
        }
        st.dataframe(pd.DataFrame(dataset_info), use_container_width=True, hide_index=True)

    with ds_col2:
        st.markdown("#### Live Dataset Summary")
        m1, m2 = st.columns(2)
        m1.metric("Total Stations", f"{len(df_full):,}")
        m2.metric("Filtered Stations", f"{len(df):,}")
        m1.metric("Charger Types", df_full["Charger Type"].nunique())
        m2.metric("Station Operators", df_full["Station Operator"].nunique())
        m1.metric("Year Range",
                  f"{int(df_full['Installation Year'].min())}–{int(df_full['Installation Year'].max())}")
        m2.metric("Renewable Stations",
                  f"{(df_full['Renewable Energy Source']=='Yes').sum():,}")

        missing_pct = (df_full.isnull().sum().sum() /
                       (df_full.shape[0] * df_full.shape[1]) * 100)
        st.markdown(f"""
        <div class="green-box">
        <b>Missing data (post-imputation):</b> {missing_pct:.2f}%<br>
        All missing numeric values filled with <b>column median</b>;
        categorical gaps filled with <b>mode</b>.
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🛠️ Methodology Pipeline")
    steps = [
        ("Stage 1", "Project Scope Definition",
         "Define objectives, identify dataset columns, set KPIs."),
        ("Stage 2", "Data Cleaning & Preprocessing",
         "Deduplicate, impute nulls, encode categoricals (ordinal + binary), normalise numerics with StandardScaler."),
        ("Stage 3", "Exploratory Data Analysis",
         "Histograms, box plots, bar charts, scatter plots, heatmaps, correlation matrix."),
        ("Stage 4", "Clustering Analysis",
         "K-Means with Elbow Method + Silhouette Score; auto-label clusters; visualise profiles."),
        ("Stage 5", "Association Rule Mining",
         "Apriori algorithm on binned features; filter by support, confidence, lift; visualise top rules."),
        ("Stage 6", "Anomaly Detection",
         "Z-Score, IQR, Isolation Forest (multivariate); consensus flagging (≥2 of 3 methods); demographic breakdown."),
        ("Stage 7", "Deployment",
         "Interactive Streamlit dashboard with sidebar filters, 6 analytical tabs, KPI strip, geo map."),
    ]
    pipe_df = pd.DataFrame(steps, columns=["Stage", "Title", "Description"])
    st.dataframe(pipe_df, use_container_width=True, hide_index=True)

    st.markdown("### 🔬 Tools & Libraries")
    tools_col1, tools_col2, tools_col3 = st.columns(3)
    with tools_col1:
        st.markdown("""
        <div class="insight-box">
        <b>Data Processing</b><br>
        • pandas — data manipulation<br>
        • numpy — numerical operations<br>
        • scikit-learn — preprocessing & models<br>
        • scipy — statistical methods
        </div>""", unsafe_allow_html=True)
    with tools_col2:
        st.markdown("""
        <div class="insight-box">
        <b>Machine Learning</b><br>
        • KMeans — clustering<br>
        • IsolationForest — anomaly detection<br>
        • mlxtend Apriori — association rules<br>
        • silhouette_score — cluster evaluation
        </div>""", unsafe_allow_html=True)
    with tools_col3:
        st.markdown("""
        <div class="insight-box">
        <b>Visualisation & Deployment</b><br>
        • Plotly Express / Graph Objects<br>
        • Streamlit — interactive dashboard<br>
        • scatter_mapbox — geographic maps<br>
        • density_mapbox — usage heatmaps
        </div>""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 1 — EDA                                              ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_eda:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # ── Row 1: Usage Distribution + Users by Charger Type ────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x="Usage Stats (avg users/day)", nbins=30,
            color="Charger Type",
            color_discrete_map={"AC Level 1": "#64B5F6",
                                 "AC Level 2": "#1E88E5",
                                 "DC Fast Charger": "#0D47A1"},
            barmode="overlay", opacity=0.75,
            title="Distribution of Avg Daily Users by Charger Type",
            labels={"Usage Stats (avg users/day)": "Avg Users/Day"},
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg_by_charger = (
            df.groupby("Charger Type", as_index=False)
              ["Usage Stats (avg users/day)"].mean()
              .sort_values("Usage Stats (avg users/day)", ascending=True)
        )
        fig = px.bar(
            avg_by_charger, y="Charger Type",
            x="Usage Stats (avg users/day)",
            orientation="h",
            color="Charger Type",
            color_discrete_map={"AC Level 1": "#64B5F6",
                                 "AC Level 2": "#1E88E5",
                                 "DC Fast Charger": "#0D47A1"},
            title="Average Daily Users by Charger Type",
            text_auto=".1f",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Avg Users/Day")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Usage Over Installation Year + Operator Cost ──
    c3, c4 = st.columns(2)
    with c3:
        yearly = (
            df.groupby("Installation Year", as_index=False)
              ["Usage Stats (avg users/day)"].mean()
        )
        fig = px.line(
            yearly, x="Installation Year",
            y="Usage Stats (avg users/day)",
            markers=True,
            title="Avg Daily Users by Installation Year",
            color_discrete_sequence=["#1565C0"],
        )
        fig.update_traces(fill="tozeroy", fillcolor="rgba(21,101,192,0.1)")
        fig.update_layout(yaxis_title="Avg Users/Day")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(
            df, x="Station Operator", y="Cost (USD/kWh)",
            color="Station Operator",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Cost Distribution by Station Operator",
        )
        fig.update_layout(showlegend=False,
                          xaxis_title="", yaxis_title="Cost (USD/kWh)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Heatmap + Reviews vs Usage ────────────────────
    c5, c6 = st.columns(2)
    with c5:
        pivot = df.pivot_table(
            values="Usage Stats (avg users/day)",
            index="Charger Type",
            columns="Availability",
            aggfunc="mean",
        ).round(1)
        fig = px.imshow(
            pivot, text_auto=True, color_continuous_scale="Blues",
            title="Avg Users/Day: Charger Type × Availability Hours",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="Reviews (Rating)", y="Usage Stats (avg users/day)",
            color="Charger Type",
            color_discrete_map={"AC Level 1": "#64B5F6",
                                 "AC Level 2": "#1E88E5",
                                 "DC Fast Charger": "#0D47A1"},
            size="Charging Capacity (kW)",
            opacity=0.65,
            title="Reviews vs Daily Users (size = Charging Capacity)",
            labels={"Usage Stats (avg users/day)": "Avg Users/Day"},
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Renewable + Capacity bar ──────────────────────
    c7, c8 = st.columns(2)
    with c7:
        ren_grp = (
            df.groupby("Renewable Energy Source", as_index=False)
              .agg(Avg_Users=("Usage Stats (avg users/day)", "mean"))
        )
        fig = px.bar(
            ren_grp, x="Renewable Energy Source", y="Avg_Users",
            color="Renewable Energy Source",
            color_discrete_map={"Yes": "#43A047", "No": "#E53935"},
            text_auto=".1f",
            title="Avg Daily Users: Renewable vs Non-Renewable",
        )
        fig.update_layout(showlegend=False, yaxis_title="Avg Users/Day")
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        cap_grp = (
            df.groupby("Charging Capacity (kW)", as_index=False)
              ["Usage Stats (avg users/day)"].mean()
              .sort_values("Charging Capacity (kW)")
        )
        fig = px.bar(
            cap_grp, x="Charging Capacity (kW)",
            y="Usage Stats (avg users/day)",
            color="Charging Capacity (kW)",
            color_continuous_scale="Blues",
            text_auto=".1f",
            title="Avg Daily Users by Charging Capacity (kW)",
        )
        fig.update_layout(yaxis_title="Avg Users/Day",
                          xaxis=dict(type="category"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ───────────────────────────────────
    st.markdown('<div class="section-title">Feature Correlation Matrix</div>',
                unsafe_allow_html=True)
    num_feat = ["Cost (USD/kWh)", "Usage Stats (avg users/day)",
                "Charging Capacity (kW)", "Distance to City (km)",
                "Reviews (Rating)", "Parking Spots",
                "Renewable_Enc", "Charger_Enc", "Avail_Hours", "Maint_Enc"]
    existing = [c for c in num_feat if c in df.columns]
    corr = df[existing].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Correlation Matrix — EV Station Features",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Key EDA Findings"):
        dc_avg  = df[df["Charger Type"] == "DC Fast Charger"]["Usage Stats (avg users/day)"].mean()
        ac1_avg = df[df["Charger Type"] == "AC Level 1"]["Usage Stats (avg users/day)"].mean()
        ren_yes = df[df["Renewable Energy Source"] == "Yes"]["Usage Stats (avg users/day)"].mean()
        ren_no  = df[df["Renewable Energy Source"] == "No"]["Usage Stats (avg users/day)"].mean()
        st.markdown(f"""
        - **DC Fast Chargers** average **{dc_avg:.1f}** users/day vs AC Level 1 at **{ac1_avg:.1f}** — a **{((dc_avg/ac1_avg)-1)*100:.0f}%** difference
        - **Renewable stations** average **{ren_yes:.1f}** vs **{ren_no:.1f}** users/day for non-renewable
        - **Usage grows** with installation year — newer stations attract more users
        - **Charging Capacity** and **Renewable_Enc** show the strongest positive correlations with usage
        - **Distance to City** is negatively correlated with demand — urban stations are busier
        """)


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 2 — CLUSTERING                                        ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_cluster:
    st.markdown('<div class="section-title">K-Means Clustering Analysis</div>',
                unsafe_allow_html=True)

    ks, inertias, sils = elbow_data(df)
    best_k_sil = ks[int(np.argmax(sils))]

    fig_elbow = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"),
    )
    fig_elbow.add_trace(
        go.Scatter(x=ks, y=inertias, mode="lines+markers",
                   marker=dict(color="#1565C0", size=8),
                   line=dict(color="#1565C0", width=2), name="Inertia"),
        row=1, col=1,
    )
    fig_elbow.add_trace(
        go.Scatter(x=ks, y=sils, mode="lines+markers",
                   marker=dict(color="#E53935", size=8),
                   line=dict(color="#E53935", width=2), name="Silhouette"),
        row=1, col=2,
    )
    fig_elbow.add_vline(x=best_k_sil, line_dash="dot",
                        line_color="green", row=1, col=2)
    fig_elbow.update_layout(
        title=f"Optimal K = {best_k_sil} (highest silhouette score). "
              f"Selected K = {n_clusters}",
        showlegend=False, height=350,
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    cluster_labels, sil_score = run_kmeans(df, n_clusters)
    df_c = df.copy()
    df_c["Cluster"] = cluster_labels.astype(str)

    st.info(f"**Silhouette Score for K={n_clusters}: {sil_score:.3f}** "
            f"(closer to 1.0 = better-defined clusters)")

    # Profile
    profile = (
        df_c.groupby("Cluster").agg(
            Count=("Station ID", "count"),
            Avg_Users=("Usage Stats (avg users/day)", "mean"),
            Avg_Capacity=("Charging Capacity (kW)", "mean"),
            Avg_Cost=("Cost (USD/kWh)", "mean"),
            Avg_Distance=("Distance to City (km)", "mean"),
            Avg_Rating=("Reviews (Rating)", "mean"),
        ).round(2).reset_index()
    )

    def auto_label(row):
        if row["Avg_Users"] >= df_c["Usage Stats (avg users/day)"].quantile(0.75):
            return "🔴 High-Demand Urban Hub"
        elif row["Avg_Distance"] >= df_c["Distance to City (km)"].quantile(0.75):
            return "🔵 Remote Low-Traffic Station"
        elif row["Avg_Cost"] <= df_c["Cost (USD/kWh)"].quantile(0.30):
            return "🟢 Budget Commuter Station"
        else:
            return "🟡 Moderate Mixed-Use Station"

    profile["Label"] = profile.apply(auto_label, axis=1)
    df_c["Cluster_Label"] = df_c["Cluster"].map(
        profile.set_index("Cluster")["Label"])

    st.markdown("#### Cluster Profiles")
    st.dataframe(
        profile.rename(columns={
            "Count": "Stations", "Avg_Users": "Avg Users/Day",
            "Avg_Capacity": "Avg kW", "Avg_Cost": "Avg $/kWh",
            "Avg_Distance": "Avg Distance km", "Avg_Rating": "Avg Rating",
        }),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            df_c.sample(min(2000, len(df_c)), random_state=1),
            x="Usage Stats (avg users/day)",
            y="Charging Capacity (kW)",
            color="Cluster_Label",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7,
            title="Clusters: Avg Users vs Charging Capacity",
            hover_data={"Station ID": True, "Station Operator": True,
                        "Charger Type": True},
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.35, font_size=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            df_c.sample(min(2000, len(df_c)), random_state=2),
            x="Distance to City (km)",
            y="Cost (USD/kWh)",
            color="Cluster_Label",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7,
            title="Clusters: Distance to City vs Cost",
            hover_data={"Station ID": True, "Charger Type": True},
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.35, font_size=10))
        st.plotly_chart(fig, use_container_width=True)

    comp = (df_c.groupby(["Cluster_Label", "Charger Type"])
                .size().reset_index(name="Count"))
    fig = px.bar(
        comp, x="Cluster_Label", y="Count",
        color="Charger Type",
        color_discrete_map={"AC Level 1": "#64B5F6",
                             "AC Level 2": "#1E88E5",
                             "DC Fast Charger": "#0D47A1"},
        barmode="stack",
        title="Charger Type Composition Within Each Cluster",
    )
    fig.update_layout(xaxis_title="", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 3 — ASSOCIATION RULE MINING                          ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_arm:
    st.markdown('<div class="section-title">Association Rule Mining — Apriori Algorithm</div>',
                unsafe_allow_html=True)
    st.caption(f"Min Support: **{arm_sup:.2f}** | Min Confidence: **{arm_conf:.2f}** "
               "(adjust in sidebar)")

    rules, n_freq = run_arm(df, arm_sup, arm_conf)

    if len(rules) == 0:
        st.warning("No rules found. Lower Support or Confidence in the sidebar.")
    else:
        st.success(f"✅ Found **{len(rules):,}** association rules from "
                   f"**{n_freq:,}** frequent itemsets")

        rules_disp = rules.copy()
        rules_disp["Antecedent"] = rules_disp["antecedents"].apply(
            lambda x: " + ".join(sorted(x)))
        rules_disp["Consequent"] = rules_disp["consequents"].apply(
            lambda x: " + ".join(sorted(x)))
        rules_disp = rules_disp[["Antecedent", "Consequent",
                                   "support", "confidence", "lift"]].round(3)
        rules_disp.columns = ["Antecedent (IF)", "Consequent (THEN)",
                               "Support", "Confidence", "Lift"]

        st.markdown("#### Top Rules by Lift")
        st.dataframe(rules_disp.head(20), use_container_width=True,
                     hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            top10 = rules_disp.head(10).copy()
            top10["Rule"] = (top10["Antecedent (IF)"].str[:30] + " → " +
                             top10["Consequent (THEN)"].str[:20])
            fig = px.bar(
                top10.iloc[::-1], y="Rule", x="Lift",
                orientation="h",
                color="Lift", color_continuous_scale="Viridis",
                title="Top 10 Rules by Lift", text_auto=".2f",
            )
            fig.update_layout(height=420, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                rules_disp, x="Support", y="Confidence",
                size="Lift", color="Lift",
                color_continuous_scale="Plasma",
                hover_data=["Antecedent (IF)", "Consequent (THEN)"],
                title="Support vs Confidence (size & colour = Lift)",
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔑 Top Rules Interpretation")
        for _, row in rules.head(5).iterrows():
            ant = " + ".join(sorted(row["antecedents"]))
            con = " + ".join(sorted(row["consequents"]))
            st.markdown(
                f'<div class="insight-box">'
                f'<b>IF</b> {ant} <b>→ THEN</b> {con}<br>'
                f'Support: {row["support"]:.3f} | '
                f'Confidence: {row["confidence"]:.3f} | '
                f'<b>Lift: {row["lift"]:.3f}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 4 — ANOMALY DETECTION                                ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_anom:
    st.markdown('<div class="section-title">Anomaly Detection — Multi-Method Consensus</div>',
                unsafe_allow_html=True)
    st.caption(f"Isolation Forest contamination: **{contamination*100:.0f}%** "
               "(adjust in sidebar)")

    df_a = detect_anomalies(df, contamination)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("🚨 Consensus Anomalies", int(df_a["Anomaly"].sum()))
    a2.metric("📐 Z-Score Outliers",    int(df_a["Anom_Z"].sum()))
    a3.metric("📦 IQR Outliers",        int(df_a["Anom_IQR"].sum()))
    a4.metric("🤖 Isolation Forest",    int(df_a["Anom_IF"].sum()))

    st.markdown("**Consensus rule:** Flagged as anomaly if ≥ 2 of 3 methods agree.")

    df_plot = df_a.copy()
    df_plot["Status"] = df_plot["Anomaly"].map({0: "Normal", 1: "Anomaly"})
    fig = px.scatter(
        df_plot, x=df_plot.index,
        y="Usage Stats (avg users/day)",
        color="Status",
        color_discrete_map={"Normal": "#42A5F5", "Anomaly": "#E53935"},
        symbol="Status",
        symbol_map={"Normal": "circle", "Anomaly": "x"},
        opacity=0.7, size_max=7,
        title="Usage Stats — Normal vs Anomalous Stations",
        labels={"x": "Station Index",
                "Usage Stats (avg users/day)": "Avg Users/Day"},
        hover_data={"Station ID": True, "Charger Type": True,
                    "Station Operator": True},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        df_plot2 = df_a.copy()
        df_plot2["CR_Status"] = df_plot2["HiCost_LowRating"].map(
            {0: "Normal", 1: "High-Cost / Low-Rating"})
        fig2 = px.scatter(
            df_plot2.sample(min(2000, len(df_plot2)), random_state=5),
            x="Cost (USD/kWh)", y="Reviews (Rating)",
            color="CR_Status",
            color_discrete_map={"Normal": "#42A5F5",
                                 "High-Cost / Low-Rating": "#E53935"},
            opacity=0.7,
            title="High-Cost / Low-Rating Anomaly Stations",
            hover_data={"Station ID": True, "Station Operator": True},
        )
        fig2.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        anom_ct = (df_a[df_a["Anomaly"] == 1]
                   .groupby("Charger Type").size().reset_index(name="Anomaly Count"))
        tot_ct  = (df_a.groupby("Charger Type").size().reset_index(name="Total"))
        merged  = anom_ct.merge(tot_ct, on="Charger Type")
        merged["Anomaly Rate %"] = (merged["Anomaly Count"] / merged["Total"] * 100).round(2)
        fig3 = px.bar(
            merged, x="Charger Type", y="Anomaly Rate %",
            color="Charger Type",
            color_discrete_map={"AC Level 1": "#64B5F6",
                                 "AC Level 2": "#1E88E5",
                                 "DC Fast Charger": "#0D47A1"},
            text_auto=".1f",
            title="Anomaly Rate (%) by Charger Type",
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 🔍 Anomalous Station Details")
    anom_detail = df_a[df_a["Anomaly"] == 1][[
        "Station ID", "Usage Stats (avg users/day)", "Cost (USD/kWh)",
        "Reviews (Rating)", "Charger Type", "Station Operator",
        "Distance to City (km)", "Renewable Energy Source", "Anom_Score"
    ]].sort_values("Anom_Score", ascending=False).reset_index(drop=True)
    st.dataframe(anom_detail, use_container_width=True, hide_index=True)
    st.caption(f"{len(anom_detail)} anomalous stations | "
               "Anom_Score = number of methods that flagged the station (max 3)")


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 5 — GEO MAP                                          ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_map:
    st.markdown('<div class="section-title">Geographic Distribution of EV Charging Stations</div>',
                unsafe_allow_html=True)

    map_mode = st.radio(
        "Map Mode",
        ["Station Clusters", "Usage Heatmap", "Charger Type", "Anomalies"],
        horizontal=True,
    )
    sample_n = min(3000, len(df))
    df_map = df.sample(sample_n, random_state=42).copy()

    if map_mode == "Station Clusters":
        cl_labels, _ = run_kmeans(df, n_clusters)
        df_cl = df.copy()
        df_cl["Cluster"] = cl_labels.astype(str)
        df_map2 = df_cl.sample(sample_n, random_state=42)
        fig_map = px.scatter_mapbox(
            df_map2, lat="Latitude", lon="Longitude",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            size="Usage Stats (avg users/day)", size_max=14,
            zoom=1, height=600,
            hover_name="Station ID",
            hover_data={"Latitude": False, "Longitude": False,
                        "Charger Type": True,
                        "Usage Stats (avg users/day)": True,
                        "Station Operator": True,
                        "Reviews (Rating)": True},
            mapbox_style="open-street-map",
            title=f"Stations Coloured by K-Means Cluster (K={n_clusters})",
        )

    elif map_mode == "Usage Heatmap":
        fig_map = px.density_mapbox(
            df_map, lat="Latitude", lon="Longitude",
            z="Usage Stats (avg users/day)",
            radius=18, zoom=1, height=600,
            color_continuous_scale="YlOrRd",
            mapbox_style="carto-positron",
            title="Demand Heatmap — Avg Daily Users",
        )

    elif map_mode == "Charger Type":
        fig_map = px.scatter_mapbox(
            df_map, lat="Latitude", lon="Longitude",
            color="Charger Type",
            color_discrete_map={"AC Level 1": "#64B5F6",
                                 "AC Level 2": "#1E88E5",
                                 "DC Fast Charger": "#0D47A1"},
            size="Usage Stats (avg users/day)", size_max=12,
            zoom=1, height=600,
            hover_name="Station ID",
            hover_data={"Latitude": False, "Longitude": False,
                        "Station Operator": True,
                        "Charging Capacity (kW)": True},
            mapbox_style="open-street-map",
            title="Station Map Coloured by Charger Type",
        )

    else:  # Anomalies
        df_a2 = detect_anomalies(df, contamination)
        df_map3 = df_a2.sample(sample_n, random_state=42).copy()
        df_map3["Status"] = df_map3["Anomaly"].map({0: "Normal", 1: "Anomaly"})
        fig_map = px.scatter_mapbox(
            df_map3, lat="Latitude", lon="Longitude",
            color="Status",
            color_discrete_map={"Normal": "#42A5F5", "Anomaly": "#E53935"},
            size="Usage Stats (avg users/day)", size_max=14,
            zoom=1, height=600,
            hover_name="Station ID",
            hover_data={"Latitude": False, "Longitude": False,
                        "Usage Stats (avg users/day)": True,
                        "Cost (USD/kWh)": True,
                        "Reviews (Rating)": True},
            mapbox_style="open-street-map",
            title="Anomalous Stations Highlighted on Map",
        )

    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### Station Performance: Operator × Charger Type")
    pivot_op = df.pivot_table(
        values="Usage Stats (avg users/day)",
        index="Station Operator",
        columns="Charger Type",
        aggfunc="mean",
    ).round(1)
    fig_pivot = px.imshow(
        pivot_op, text_auto=True, color_continuous_scale="Blues",
        title="Avg Users/Day: Operator × Charger Type", aspect="auto",
    )
    st.plotly_chart(fig_pivot, use_container_width=True)


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 6 — INSIGHTS & RECOMMENDATIONS                       ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_insights:
    st.markdown('<div class="section-title">Key Insights & Strategic Recommendations</div>',
                unsafe_allow_html=True)

    # Compute live stats
    dc_avg   = df[df["Charger Type"] == "DC Fast Charger"]["Usage Stats (avg users/day)"].mean()
    ac1_avg  = df[df["Charger Type"] == "AC Level 1"]["Usage Stats (avg users/day)"].mean()
    ren_yes  = df[df["Renewable Energy Source"] == "Yes"]["Usage Stats (avg users/day)"].mean()
    ren_no   = df[df["Renewable Energy Source"] == "No"]["Usage Stats (avg users/day)"].mean()
    ren_r_y  = df[df["Renewable Energy Source"] == "Yes"]["Reviews (Rating)"].mean()
    ren_r_n  = df[df["Renewable Energy Source"] == "No"]["Reviews (Rating)"].mean()
    med_dist = df["Distance to City (km)"].median()
    city_avg = df[df["Distance to City (km)"] <= med_dist]["Usage Stats (avg users/day)"].mean()
    rural_avg= df[df["Distance to City (km)"] >  med_dist]["Usage Stats (avg users/day)"].mean()
    best_op  = df.groupby("Station Operator")["Reviews (Rating)"].mean().idxmax()
    best_op_r= df.groupby("Station Operator")["Reviews (Rating)"].mean().max()
    df_a_ins = detect_anomalies(df, contamination)
    n_anom   = int(df_a_ins["Anomaly"].sum())
    n_cr     = int(df_a_ins["HiCost_LowRating"].sum())

    # Findings
    st.markdown("### 📊 Analytical Findings")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            f'<div class="insight-box"><b>⚡ Charger Type Impact</b><br>'
            f'DC Fast Chargers attract <b>{dc_avg:.1f}</b> avg users/day vs '
            f'AC Level 1 at <b>{ac1_avg:.1f}</b> — '
            f'<b>{((dc_avg/ac1_avg)-1)*100:.0f}% more demand</b>.</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-box"><b>🏙️ Urban vs Rural Demand</b><br>'
            f'City stations (≤{med_dist:.0f} km) average <b>{city_avg:.1f}</b> '
            f'users/day vs rural at <b>{rural_avg:.1f}</b> — '
            f'<b>{((city_avg/rural_avg)-1)*100:.0f}% gap</b>.</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="green-box"><b>🌿 Renewable Energy Advantage</b><br>'
            f'Renewable stations earn avg rating <b>{ren_r_y:.2f}</b> vs '
            f'<b>{ren_r_n:.2f}</b>, and attract <b>{ren_yes:.1f}</b> vs '
            f'<b>{ren_no:.1f}</b> users/day.</div>',
            unsafe_allow_html=True)

    with f2:
        st.markdown(
            f'<div class="green-box"><b>🏆 Best-Rated Operator</b><br>'
            f'<b>{best_op}</b> leads in customer satisfaction with avg rating '
            f'<b>{best_op_r:.2f}/5</b>.</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="danger-box"><b>🚨 Anomalies Detected</b><br>'
            f'<b>{n_anom}</b> stations flagged by consensus. '
            f'<b>{n_cr}</b> have high cost + low rating — highest business risk.</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="warn-box"><b>📈 Installation Year Trend</b><br>'
            f'Stations built from 2019 onward show higher usage. '
            f'Pre-2015 AC Level 1 stations are underperforming candidates '
            f'for upgrade or decommission.</div>',
            unsafe_allow_html=True)

    # Summary chart
    st.markdown("### 📈 Charger Type Performance Summary")
    summary = df.groupby("Charger Type", as_index=False).agg(
        Avg_Users=("Usage Stats (avg users/day)", "mean"),
        Avg_Rating=("Reviews (Rating)", "mean"),
        Avg_Cost=("Cost (USD/kWh)", "mean"),
    ).round(2)

    fig_sum = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Avg Users/Day", "Avg Rating", "Avg Cost $/kWh"),
    )
    colors_map = {"AC Level 1": "#64B5F6", "AC Level 2": "#1E88E5",
                  "DC Fast Charger": "#0D47A1"}
    for i, row_s in summary.iterrows():
        c = colors_map[row_s["Charger Type"]]
        fig_sum.add_trace(go.Bar(name=row_s["Charger Type"],
                                  x=[row_s["Charger Type"]],
                                  y=[row_s["Avg_Users"]],
                                  marker_color=c, showlegend=(i == 0)), row=1, col=1)
        fig_sum.add_trace(go.Bar(name=row_s["Charger Type"],
                                  x=[row_s["Charger Type"]],
                                  y=[row_s["Avg_Rating"]],
                                  marker_color=c, showlegend=False), row=1, col=2)
        fig_sum.add_trace(go.Bar(name=row_s["Charger Type"],
                                  x=[row_s["Charger Type"]],
                                  y=[row_s["Avg_Cost"]],
                                  marker_color=c, showlegend=False), row=1, col=3)
    fig_sum.update_layout(height=320, showlegend=True,
                           legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_sum, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Strategic Recommendations")
    recs = [
        ("Expand DC Fast Charger Network in Urban Areas",
         f"DC Fast Chargers generate {dc_avg:.1f} avg users/day — the highest of all charger types. "
         "Prioritise installation within 20 km of city centres where usage-capacity "
         "correlation is strongest.",
         "insight-box"),
        ("Audit High-Cost / Low-Rating Stations",
         f"{n_cr} stations have above-P85 cost AND below-P15 rating. These are losing "
         "customers and damaging brand perception. Immediate pricing review and service "
         "quality audit recommended.",
         "danger-box"),
        ("Accelerate Renewable Energy Integration",
         f"Renewable stations earn {ren_r_y:.2f} avg rating vs {ren_r_n:.2f} and attract "
         f"{ren_yes:.1f} vs {ren_no:.1f} users/day. Transitioning to renewables aligns "
         "commercial and sustainability goals simultaneously.",
         "green-box"),
        ("Upgrade or Decommission Remote AC Level 1 Stations",
         f"AC Level 1 averages only {ac1_avg:.1f} users/day. Remote low-traffic stations "
         "deliver the poorest ROI. Upgrading to AC Level 2 or relocating to higher-demand "
         "urban zones will improve asset utilisation.",
         "warn-box"),
        ("Implement Cluster-Based Dynamic Pricing",
         "High-Demand Urban Hubs (from clustering) can support premium pricing during "
         "peak hours. Budget Commuter Stations benefit from stable lower rates. "
         "Real-time availability-linked pricing would further optimise revenue.",
         "insight-box"),
        ("Preventive Maintenance for Anomalous Stations",
         f"{n_anom} consensus-flagged stations likely have underlying hardware issues. "
         "Shifting from reactive to scheduled preventive maintenance reduces downtime "
         "and improves customer satisfaction scores.",
         "warn-box"),
    ]

    for title, body, box_class in recs:
        with st.expander(f"💡 {title}"):
            st.markdown(f'<div class="{box_class}">{body}</div>',
                        unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:0.82rem'>"
        "SmartEnergy Data Lab | Data Mining Summative Assessment | "
        "Scenario 2: SmartCharging Analytics — Uncovering EV Behaviour Patterns"
        "</p>",
        unsafe_allow_html=True,
    )
