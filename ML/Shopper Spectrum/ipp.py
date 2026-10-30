import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("rfm_kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    sim_df = joblib.load("similarity_df.pkl")  # similarity matrix
    return kmeans, scaler, sim_df

kmeans_model, scaler, similarity_df = load_models()

# Cluster Labels
cluster_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}

# ----------------------------------------------------
# Recommendation Function
# ----------------------------------------------------
def recommend(product_name, top_n=5):
    product_name = product_name.strip().upper()

    # Check if product exists
    if product_name not in similarity_df.index:
        return ["❌ Product not found in database"]

    # Get sorted similarity scores
    scores = similarity_df.loc[product_name].sort_values(ascending=False)

    # Top N excluding the same product (index 0 is itself)
    recommended = list(scores.index[1:top_n+1])
    return recommended

# ----------------------------------------------------
# Segment Prediction
# ----------------------------------------------------
def predict_segment(r, f, m):
    scaled = scaler.transform([[r, f, m]])
    cluster = kmeans_model.predict(scaled)[0]
    return cluster_labels[cluster]

# ----------------------------------------------------
# Streamlit App Layout
# ----------------------------------------------------
st.set_page_config(page_title="Customer Analytics App", layout="wide")

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Home", "Clustering", "Recommendation"]
)

# ================================
# HOME PAGE
# ================================
if page == "Home":
    st.title("🏪 Customer Analytics Application")
    st.write("""
Welcome to the Customer Analytics App!

### Features:
- ✔ **Customer Segmentation** using RFM  
- ✔ **Product Recommendation** using similarity matrix  

Upload the required model files and explore insights!
""")

# ================================
# SEGMENTATION PAGE
# ================================
elif page == "Clustering":
    st.title("👤 Customer Segmentation")

    r = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    f = st.number_input("Frequency (number of purchases)", min_value=0, value=1)
    m = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0)

    if st.button("Predict Segment"):
        segment = predict_segment(r, f, m)
        st.success(f"🟢 This customer belongs to: **{segment}**")

# ================================
# PRODUCT RECOMMENDATION PAGE
# ================================
elif page == "Recommendation":
    st.title("🛒 Product Recommender")

    product_input = st.text_input("Enter Product Name")

    if st.button("Recommend"):
        st.subheader("Recommended Products:")
        results = recommend(product_input)

        for item in results:
            st.write("✔", item)
