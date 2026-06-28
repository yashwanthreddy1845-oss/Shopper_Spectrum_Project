import pandas as pd

# -----------------------------
# Matplotlib Backend
# -----------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
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

# -----------------------------
# Customer-Product Matrix
# -----------------------------
customer_product_matrix = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("Customer-Product Matrix Created")
print(customer_product_matrix.shape)

# -----------------------------
# Product Similarity Matrix
# -----------------------------
product_similarity = cosine_similarity(customer_product_matrix.T)

product_similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)

print("Product Similarity Matrix Created")

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_products(product_name, top_n=5):

    if product_name not in product_similarity_df.index:
        print("Product not found!")
        return

    recommendations = (
        product_similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    print(f"\nTop {top_n} Recommendations for '{product_name}':")
    print(recommendations)


# -----------------------------
# Example Recommendation
# -----------------------------
sample_product = customer_product_matrix.columns[0]

recommend_products(sample_product)

# -----------------------------
# Save Recommendation Model
# -----------------------------
joblib.dump(product_similarity_df, "models/product_similarity.pkl")

print("\nProduct Similarity Model Saved Successfully")

# -----------------------------
# Product Similarity Heatmap
# -----------------------------
top_products = product_similarity_df.iloc[:20, :20]

plt.figure(figsize=(12,10))

sns.heatmap(
    top_products,
    cmap="viridis",
    xticklabels=False,
    yticklabels=False
)

plt.title("Product Similarity Heatmap")

plt.tight_layout()

plt.savefig("product_similarity_heatmap.png", dpi=300)

plt.close()

print("Product Similarity Heatmap Saved Successfully")