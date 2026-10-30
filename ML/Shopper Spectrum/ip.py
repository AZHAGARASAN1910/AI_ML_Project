import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load Models
# --------------------------

@st.cache_resource
def load_models():
    kmeans = joblib.load("rfm_kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    sim_df = joblib.load("similarity_df.pkl")  # product similarity matrix
    return kmeans, scaler, sim_df

kmeans_model, scaler, similarity_df = load_models()

# Cluster label mapping
cluster_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}

# --------------------------
# Functions
# --------------------------

def recommend(product_name, top_n=5):
    """Returns top N similar products using cosine similarity matrix."""
    if product_name not in similarity_df.columns:
        return ["❌ Product not found in database"]

    scores = similarity_df[product_name].sort_values(ascending=False)
    recommended = list(scores.iloc[1:top_n+1].index)
    return recommended


def predict_segment(r, f, m):
    """Predict RFM Cluster."""
    scaled = scaler.transform([[r, f, m]])
    cluster = kmeans_model.predict(scaled)[0]
    return cluster_labels[cluster]


# --------------------------
# Streamlit App UI
# --------------------------

st.set_page_config(page_title="Customer Analytics App", layout="wide")

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Home", "Clustering", "Recommendation"]
)

# --------------------------
# Home
# --------------------------

if page == "Home":
    st.title("🏪 Customer Analytics Application")
    st.write("""
Welcome to the Customer Analytics App!

### Modules:
✔ **Customer Segmentation** – Predict customer cluster using RFM  
✔ **Product Recommendation** – Get top 5 similar products  

Upload your models and start exploring insights!
""")

# --------------------------
# Customer Segmentation
# --------------------------

elif page == "Clustering":
    st.title("👤 Customer Segmentation")

    st.write("Enter RFM values:")

    r = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    f = st.number_input("Frequency (number of purchases)", min_value=0, value=1)
    m = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0)

    if st.button("Predict Segment"):
        segment = predict_segment(r, f, m)
        st.success(f"🟢 This customer belongs to: **{segment}**")


# --------------------------
# Product Recommendation
# --------------------------

elif page == "Recommendation":
    st.title("🛒 Product Recommender")

    product_input = st.text_input("Enter Product Name")

    if st.button("Recommend"):
        results = recommend(product_input)
        st.subheader("Recommended Products:")

        for item in results:
            st.write("✔", item)

product_name = st.text_input("Enter product name:")

if product_name:
    product_name = product_name.strip().upper()

    if product_name not in similarity_df.index:
        st.error("❌ Product not found in database. Try exact product name.")
    else:
        # recommend top 5
        scores = similarity_df[product_name].sort_values(ascending=False)[1:6]
        st.success("Recommended Products:")

        for rec in scores.index:
            st.write("✔", rec)

