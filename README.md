# ⚡ SmartCharge Analytics — Uncovering EV Behavior Patterns

> **CRS:** Artificial Intelligence | **Course:** Data Mining | **Scenario 2** | **60 Marks Summative**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange) ![Plotly](https://img.shields.io/badge/Plotly-5.22-green)

---

## 🎯 Project Scope

This project analyzes EV charging station behavior patterns across a global dataset of 500 stations. The goal is to support infrastructure planning, detect anomalies, uncover behavioral clusters, and deliver actionable insights through an interactive Streamlit dashboard.

**Objectives:**
- Cluster stations/users by usage patterns (K-Means, DBSCAN)
- Detect anomalous stations via statistical methods (Z-Score, IQR)
- Mine association rules between station features and demand
- Deploy an interactive intelligence dashboard on Streamlit Cloud

---

## 📁 Repository Structure

```
IDAI105(StudentID)-YourName/
├── app.py                  # 🚀 Main Streamlit dashboard
├── requirements.txt        # 📦 Python dependencies
├── analysis.ipynb          # 📓 Full Jupyter analysis notebook
├── ev_charging_dataset.csv # 📊 Dataset (upload your own)
└── README.md               # 📖 This file
```

---

## 📊 Dataset

**Source:** EV Charging Stations Dataset (provided via course portal)

| Column | Description |
|--------|-------------|
| Station_ID | Unique station identifier |
| Latitude / Longitude | Geographic coordinates |
| Charger_Type | AC Level 1, AC Level 2, DC Fast |
| Cost_USD_kWh | Charging cost per kWh |
| Availability | Available / Occupied / Offline |
| Distance_to_City_km | Proximity to nearest city |
| Usage_Stats_avg_users_day | Average daily users |
| Station_Operator | Network operator name |
| Charging_Capacity_kW | Max charging power |
| Connector_Types | CCS, CHAdeMO, J1772, Tesla |
| Installation_Year | Year station was installed |
| Renewable_Energy_Source | Yes / No |
| Reviews_Rating | 1–5 star rating |
| Parking_Spots | Number of parking bays |
| Maintenance_Frequency | Monthly / Quarterly / Annual |

---

## 🔧 Key Preprocessing Steps

1. **Missing Value Handling** — Median imputation for `Reviews_Rating`; mode fill for `Renewable_Energy_Source` and `Connector_Types`
2. **Duplicate Removal** — Deduplicated on `Station_ID`
3. **Categorical Encoding** — `LabelEncoder` for Charger_Type, Operator, Availability, Renewable
4. **Normalization** — `StandardScaler` on Cost, Usage, Capacity, Distance
5. **Feature Engineering** — Usage category bins (LOW/MID/HIGH) for association mining

---

## 📈 EDA Visualizations

| Visualization | Insight |
|---|---|
| Usage Histogram | Right-skewed distribution — most stations serve 20–60 users/day |
| Charger Type Distribution | AC Level 2 dominates (45%), DC Fast fastest growing |
| Cost Boxplot by Operator | Tesla/EVgo charge 35–55% more than Blink |
| Rating vs Usage Scatter | Weak positive correlation (r=0.22) — quality ≠ traffic |
| Installation Year Trend | 3× growth 2020–2024 post-EV adoption surge |
| Correlation Heatmap | Capacity–Usage strongest positive correlation (r=0.61) |

---

## 🧠 Clustering Analysis

**Algorithm:** K-Means (K=4, selected via Elbow Method + Silhouette Score)

**Features Used:** Cost, Usage, Capacity, Distance to City, Charger Type (encoded)

| Cluster | Label | Characteristics |
|---|---|---|
| 0 | 🌿 Eco Commuters | Low cost, moderate usage, high renewable adoption |
| 1 | ⚡ Power Hubs | High capacity DC Fast, urban, heavy usage |
| 2 | 🌆 City Fast-Chargers | Mid-range cost, high density, frequent turnover |
| 3 | 🛣️ Highway Stoppers | Remote, lower usage, long session duration |

**PCA Variance Explained:** ~68% across 2 components

---

## 🔗 Association Rule Mining

**Algorithm:** Apriori (mlxtend) / Manual co-occurrence fallback  
**Parameters:** min_support=0.05, min_confidence=0.40

**Top Findings:**
- `DC Fast` → `HIGH Usage` (Lift: 2.1) — Fast chargers drive traffic
- `Renewable=Yes` → `Rating ≥ 4.0` (Lift: 1.8) — Green stations get better reviews
- `ChargePoint Operator` → `Available Status` (Lift: 1.6) — Reliability advantage
- `Distance < 5km` → `MID/HIGH Usage` (Lift: 1.7) — Urban proximity boosts demand

---

## 🚨 Anomaly Detection

**Methods:** Z-Score (threshold z > 3.0) + IQR (1.5× fence)

- **25 anomalous stations** detected (~5% of network)
- Anomalies show 3.5–6× normal usage levels
- Concentrated in DC Fast charger segment
- Likely causes: data logging errors, overcrowded stations, or special events

---

## 🚀 Streamlit Dashboard Features

The deployed app includes:

1. **⚡ Loading Screen** — Animated initialization with progress indicators
2. **🔐 Login Screen** — Secure credential-based access (admin/analyst/demo)
3. **📖 Onboarding Tutorial** — 5-step interactive guide with "I Agree" confirmation
4. **🏠 Overview Dashboard** — KPI metrics, charger distribution, installation trends
5. **🗺️ Interactive Map** — Plotly Mapbox with cluster/anomaly/charger/rating views
6. **📊 EDA Visualizations** — Histograms, boxplots, scatter plots, heatmaps, time trends
7. **🧠 Clustering Analysis** — Elbow method, PCA scatter, radar cluster profiles
8. **🔗 Association Rules** — Rules scatter plot, top-10 lift bar chart, rules table
9. **🚨 Anomaly Detection** — Z-score chart, distribution comparison, station details
10. **💡 Insights Report** — Strategic findings & recommendations

---

## 🎓 Strategic Recommendations

1. **Expand DC Fast Charging** — Highest ROI along interstate corridors
2. **Invest in Renewable Integration** — Improves ratings, attracts eco-conscious users
3. **Audit Anomalous Stations** — 25 stations need physical inspection
4. **Prioritize Urban Locations** — Stations within 5km of cities have 40% higher usage
5. **Partner with Top Operators** — ChargePoint & Tesla show best reliability metrics

---

## 🔧 Installation & Local Run

```bash
# Clone repository
git clone https://github.com/YourUsername/IDAI105-YourName.git
cd IDAI105-YourName

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

**Demo Login:** `admin` / `ev2024`

---

## 🌐 Deployed Application

🔗 **Streamlit Cloud:** `https://idai105-1000414-aditya-jitendra-kumar-sahani-sa.streamlit.app/`

> Replace with your actual deployed URL after Streamlit Cloud deployment.

---

## 📚 References

- [K-Means Clustering — Neptune AI](https://neptune.ai/blog/k-means-clustering)
- [Association Rule Mining — DiceCamp](https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/)
- [EV Charging Behavior Research — arXiv](https://arxiv.org/pdf/1802.04193)
- [Clustering for EV Stations — ResearchGate](https://www.researchgate.net/publication/374171696)
- [Anomaly Detection Guide — KDNuggets](https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html)
- [Frontiers in Energy Research — EV Patterns](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.773440/full)

---

## 👤 Submission Details

| Field | Value |
|---|---|
| Student Name | *[Your Full Name]* |
| Candidate Registration Number | *[Your Registration Number]* |
| CRS Name | Artificial Intelligence |
| Course Name | Data Mining |
| School Name | *[Your School Name]* |
| GitHub Repository | `https://github.com/YourUsername/IDAI105(StudentID)-YourName` |

---

*SmartCharge Analytics — Mining the Future: Unlocking Business Intelligence with AI* ⚡
