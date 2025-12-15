import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Load Model
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load("model/scaler.pkl")
    kmeans = joblib.load("model/kmeans.pkl")
    logreg = joblib.load("model/logreg.pkl")
    features = joblib.load("model/features.pkl")
    return scaler, kmeans, logreg, features

scaler, kmeans, logreg, features = load_models()

# =========================
# UI
# =========================
st.set_page_config(page_title="Credit Card Clustering", layout="wide")

st.title("💳 Credit Card Clustering & Analysis")
st.write("K-Means Clustering + Logistic Regression")

# =========================
# Sidebar Input
# =========================
st.sidebar.header("🔢 Input Data Nasabah")

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature, value=0.0
    )

input_df = pd.DataFrame([input_data])

# =========================
# Prediction
# =========================
if st.sidebar.button("🔍 Prediksi Cluster"):
    # Scaling
    scaled_input = scaler.transform(input_df)

    # Clustering
    cluster = kmeans.predict(scaled_input)[0]

    # Probabilitas Logistic Regression
    prob = logreg.predict_proba(scaled_input)[0]

    st.subheader("📌 Hasil Analisis")
    st.success(f"Nasabah termasuk ke **Cluster {cluster}**")

    st.write("### 📊 Probabilitas Tiap Cluster")
    prob_df = pd.DataFrame(
        prob.reshape(1, -1),
        columns=[f"Cluster {i}" for i in logreg.classes_]
    )
    st.dataframe(prob_df)

# =========================
# Koefisien Model
# =========================
st.subheader("📈 Pengaruh Fitur (Logistic Regression)")
coef_df = pd.DataFrame(
    logreg.coef_,
    columns=features,
    index=[f"Cluster {i}" for i in logreg.classes_]
)

st.dataframe(coef_df)

# =========================
# Visualisasi Koefisien
# =========================
cluster_select = st.selectbox(
    "Pilih Cluster",
    coef_df.index
)

fig, ax = plt.subplots(figsize=(8,4))
coef_df.loc[cluster_select].plot(kind="bar", ax=ax)
ax.set_title(f"Koefisien Logistic Regression - {cluster_select}")
ax.set_ylabel("Nilai Koefisien")
st.pyplot(fig)
