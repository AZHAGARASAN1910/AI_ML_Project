import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Diamond Price Prediction", layout="centered")

# ----------------------------------
# Load Models (Cached)
# ----------------------------------
@st.cache_resource
def load_models():
    price_model = joblib.load("diamond_price_model.pkl")
    cluster_model = joblib.load("diamond_cluster_model.pkl")
    scaler = joblib.load("scaler.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    # IMPORTANT: avoid keras.metrics.mse error
    ann_model = load_model("ann_diamond_model.h5", compile=False)

    return price_model, cluster_model, scaler, cluster_scaler, encoder, ann_model


price_model, cluster_model, scaler, cluster_scaler, encoder, ann_model = load_models()

# ----------------------------------
# UI
# ----------------------------------
st.title("💎 Diamond Price Prediction & Market Segmentation")

st.sidebar.header("Enter Diamond Details")

carat = st.sidebar.number_input("Carat", 0.1, 5.0, 0.5)
depth = st.sidebar.number_input("Depth", 40.0, 80.0, 61.5)
table = st.sidebar.number_input("Table", 40.0, 80.0, 55.0)
x = st.sidebar.number_input("X (length)", 0.0, 10.0, 5.0)
y = st.sidebar.number_input("Y (width)", 0.0, 10.0, 5.0)
z = st.sidebar.number_input("Z (height)", 0.0, 10.0, 3.0)

cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# ----------------------------------
# Prediction Button
# ----------------------------------
if st.button("Predict Price"):
    try:
        # ----------------------------
        # Feature Engineering
        # ----------------------------
        num_features = np.array([[carat, depth, table, x, y, z]])
        cat_features = encoder.transform([[cut, color, clarity]])

        X = np.hstack((num_features, cat_features))

        # 🔍 DEBUG INFO
        st.write("Input features:", X.shape[1])
        st.write("Scaler expects:", scaler.n_features_in_)

        # ----------------------------
        # FIX feature mismatch
        # ----------------------------
        if X.shape[1] < scaler.n_features_in_:
            diff = scaler.n_features_in_ - X.shape[1]
            X = np.hstack((X, np.zeros((1, diff))))

        # ----------------------------
        # Scaling
        # ----------------------------
        X_scaled = scaler.transform(X)

        # ----------------------------
        # Predictions
        # ----------------------------
        ml_price = price_model.predict(X_scaled)[0]
        ann_price = ann_model.predict(X_scaled)[0][0]

        X_cluster = cluster_scaler.transform(X_scaled)
        cluster = cluster_model.predict(X_cluster)[0]

        # ----------------------------
        # Output
        # ----------------------------
        st.success("✅ Prediction Successful")

        st.write(f"💰 **ML Model Price:** ₹ {ml_price:,.2f}")
        st.write(f"🤖 **ANN Model Price:** ₹ {ann_price:,.2f}")
        st.write(f"📊 **Market Segment (Cluster):** {cluster}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
