import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")

# -----------------------------
# Feature Engineering
# -----------------------------
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("RFM Created Successfully")
print(rfm.head())

# -----------------------------
# Standardize Data
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

print("Data Standardized")

# -----------------------------
# Elbow Method
# -----------------------------
inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(2,11), inertia, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.grid(True)

plt.tight_layout()

plt.savefig("elbow_curve.png")

plt.close()

print("Elbow Curve Saved")

# -----------------------------
# Silhouette Scores
# -----------------------------
print("\nSilhouette Scores\n")

best_score = -1
best_k = 2

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(rfm_scaled)

    score = silhouette_score(rfm_scaled, labels)

    print(f"K = {k}   Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K according to Silhouette Score =", best_k)

# =====================================================
# FINAL MODEL (USE 4 CLUSTERS FOR PROJECT)
# =====================================================

print("\nTraining Final Model with 4 Clusters...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("\nCluster Counts")
print(rfm["Cluster"].value_counts())

print("\nCluster Summary")
cluster_summary = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
print(cluster_summary)

# -----------------------------
# Customer Segment Labels
# -----------------------------
print("\nCustomer Segment Labels")

for cluster in cluster_summary.index:

    r = cluster_summary.loc[cluster, "Recency"]
    f = cluster_summary.loc[cluster, "Frequency"]
    m = cluster_summary.loc[cluster, "Monetary"]

    if r < 50 and f > 10 and m > 5000:
        label = "High Value"

    elif r < 100 and f > 5:
        label = "Regular"

    elif r > 200:
        label = "At Risk"

    else:
        label = "Occasional"

    print(f"Cluster {cluster} : {label}")

# -----------------------------
# Save Models
# -----------------------------
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels Saved Successfully")

# -----------------------------
# Customer Cluster Plot
# -----------------------------
plt.figure(figsize=(10,6))

scatter = plt.scatter(
    rfm["Recency"],
    rfm["Monetary"],
    c=rfm["Cluster"]
)

plt.title("Customer Segmentation")
plt.xlabel("Recency")
plt.ylabel("Monetary")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig("customer_clusters.png")

plt.close()

print("Customer Cluster Plot Saved")

# -----------------------------
# Save RFM Table
# -----------------------------
rfm.to_csv("rfm_customers.csv", index=True)

print("RFM CSV Saved")