import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(BASE, "data", "online_retail.csv.gz"),
        compression="gzip",
        encoding="ISO-8859-1"
    )

df = load_data()

@st.cache_resource
def load_models():
    return (
        joblib.load(os.path.join(BASE, "models", "kmeans_model.pkl")),
        joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    )

try:
    kmeans, scaler = load_models()
except Exception:
    kmeans = scaler = None

@st.cache_resource
def create_similarity():
    customer_product = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity = cosine_similarity(customer_product.T)

    return pd.DataFrame(
        similarity,
        index=customer_product.columns,
        columns=customer_product.columns
    )

similarity = create_similarity()

st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 EDA Dashboard",
        "👥 Customer Segmentation",
        "🛍 Product Recommendation",
        "ℹ️ About"
    ]
)

if page=="🏠 Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation and Product Recommendation")

    c1,c2,c3=st.columns(3)
    c1.metric("Transactions",len(df))
    c2.metric("Customers",df["CustomerID"].nunique())
    c3.metric("Products",df["Description"].nunique())

    st.markdown("---")
    st.info("RFM Analysis • K-Means Clustering • Collaborative Filtering")

elif page=="📊 Dataset":
    st.title("📊 Dataset")
    st.dataframe(df.head(20),use_container_width=True)

    a,b=st.columns(2)
    a.metric("Rows",df.shape[0])
    b.metric("Columns",df.shape[1])

    st.subheader("Missing Values")
    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={"index":"Column",0:"Missing"}
        ),
        use_container_width=True
    )

    st.subheader("Statistics")
    st.dataframe(df.describe(),use_container_width=True)

elif page=="📈 EDA Dashboard":

    st.title("📈 EDA Dashboard")

    charts=[
        ("Transaction Volume","country_transactions.png"),
        ("Top Products","top_selling_products.png"),
        ("Purchase Trend","purchase_trend.png"),
        ("Monetary Distribution","monetary_distribution.png"),
        ("Recency","rfm_recency.png"),
        ("Frequency","rfm_frequency.png"),
        ("Customer Monetary","rfm_monetary.png"),
        ("Elbow Curve","elbow_curve.png"),
        ("Customer Clusters","customer_clusters.png"),
        ("Similarity Heatmap","product_similarity_heatmap.png")
    ]

    cols=st.columns(2)

    for i,(title,file) in enumerate(charts):

        with cols[i%2]:

            st.subheader(title)

            path=os.path.join(BASE,file)

            if os.path.exists(path):
                st.image(path,use_container_width=True)
            else:
                st.warning(f"{file} not found")

elif page=="👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    if kmeans is None or scaler is None:
        st.error("Model files not found.")
    else:

        c1,c2,c3=st.columns(3)

        r=c1.number_input("Recency",0,5000,30)
        f=c2.number_input("Frequency",1,1000,5)
        m=c3.number_input("Monetary",0.0,100000000.0,1000.0)

        if st.button("Predict Cluster",type="primary"):

            cluster=int(
                kmeans.predict(
                    scaler.transform(np.array([[r,f,m]]))
                )[0]
            )

            labels={
                0:"🟡 Occasional Customer",
                1:"🔴 At Risk Customer",
                2:"🟢 High Value Customer",
                3:"🔵 Regular Customer"
            }

            st.success(labels.get(cluster,"Unknown"))

elif page=="🛍 Product Recommendation":

    st.title("🛍 Product Recommendation")

    product=st.selectbox(
        "Select Product",
        sorted(similarity.index)
    )

    if st.button("Get Recommendations",type="primary"):

        recs=(
            similarity[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.subheader("Recommended Products")

        for i,item in enumerate(recs.index,1):
            st.success(f"{i}. {item}")

elif page=="ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("""
### Shopper Spectrum

**Domain:** E-Commerce & Retail Analytics

**Algorithms**
- RFM Analysis
- K-Means Clustering
- Collaborative Filtering
- Cosine Similarity

**Technologies**
- Python
- Pandas
- Streamlit
- Scikit-Learn
- Matplotlib
- Seaborn
""")
