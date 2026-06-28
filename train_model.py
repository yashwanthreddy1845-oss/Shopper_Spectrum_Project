import pandas as pd

# -----------------------------
# Matplotlib Backend
# -----------------------------
import matplotlib
matplotlib.use("Agg")   # Save charts without opening a window

import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")

# -----------------------------
# Display Dataset Information
# -----------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

# Remove rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid Quantity and UnitPrice
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")

# -----------------------------
# Cleaned Dataset Information
# -----------------------------
print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFirst 5 Rows After Cleaning:")
print(df.head())

# -----------------------------
# Create TotalPrice
# -----------------------------
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\nFirst 5 TotalPrice Values:")
print(df[["Quantity", "UnitPrice", "TotalPrice"]].head())

# -----------------------------
# Create Snapshot Date
# -----------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

print("\nSnapshot Date:")
print(snapshot_date)

# -----------------------------
# Create RFM Table
# -----------------------------
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nFirst 5 Rows of RFM Table:")
print(rfm.head())

print("\nRFM Shape:")
print(rfm.shape)

# -----------------------------
# EDA 1 - Transaction Volume by Country
# -----------------------------
country_sales = df["Country"].value_counts().head(10)

plt.figure(figsize=(10, 6))
country_sales.plot(kind="bar")

plt.title("Top 10 Countries by Number of Transactions")
plt.xlabel("Country")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("country_transactions.png")

plt.close()

print("\nChart saved successfully as 'country_transactions.png'")
# -----------------------------
# EDA 2 - Top 10 Selling Products
# -----------------------------

top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
top_products.plot(kind="bar")

plt.title("Top 10 Selling Products")
plt.xlabel("Product")
plt.ylabel("Total Quantity Sold")

plt.xticks(rotation=75)

plt.tight_layout()

plt.savefig("top_selling_products.png")

plt.close()

print("Chart saved successfully as 'top_selling_products.png'")
# -----------------------------
# EDA 3 - Purchase Trend Over Time
# -----------------------------
df["Date"] = df["InvoiceDate"].dt.date

daily_sales = df.groupby("Date")["InvoiceNo"].count()

plt.figure(figsize=(12,6))
daily_sales.plot()

plt.title("Purchase Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Transactions")

plt.tight_layout()
plt.savefig("purchase_trend.png")
plt.close()

print("purchase_trend.png saved")


# -----------------------------
# EDA 4 - Monetary Distribution
# -----------------------------
plt.figure(figsize=(8,6))

plt.hist(df["TotalPrice"], bins=50)

plt.title("Monetary Distribution")
plt.xlabel("Total Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("monetary_distribution.png")
plt.close()

print("monetary_distribution.png saved")


# -----------------------------
# EDA 5 - RFM Distributions
# -----------------------------

# Recency
plt.figure(figsize=(8,6))
plt.hist(rfm["Recency"], bins=30)

plt.title("Recency Distribution")
plt.xlabel("Recency")
plt.ylabel("Customers")

plt.tight_layout()
plt.savefig("rfm_recency.png")
plt.close()

# Frequency
plt.figure(figsize=(8,6))
plt.hist(rfm["Frequency"], bins=30)

plt.title("Frequency Distribution")
plt.xlabel("Frequency")
plt.ylabel("Customers")

plt.tight_layout()
plt.savefig("rfm_frequency.png")
plt.close()

# Monetary
plt.figure(figsize=(8,6))
plt.hist(rfm["Monetary"], bins=30)

plt.title("Monetary Distribution")
plt.xlabel("Monetary")
plt.ylabel("Customers")

plt.tight_layout()
plt.savefig("rfm_monetary.png")
plt.close()

print("RFM charts saved")