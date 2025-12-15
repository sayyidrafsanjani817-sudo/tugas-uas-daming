import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Credit Card Clustering (Random Forest)",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load("model/scaler.pkl")
    kmeans = joblib.load("model/kmeans.pkl")
    rf = joblib.load("model/random_forest.pkl")
    features = joblib.load("model/features.pkl")
    return scaler, kmeans, rf, features

scaler, kmeans, rf, features = load_models()

# =========================
# TITLE
# =========================
st.title("💳 Credit Card Clustering")
st.write("Metode: **K-Means + Random Forest (Model Tunggal)**")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("🔢 Input Data Nasabah")

input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature,
        value=0.0
    )

input_df = pd.DataFrame([input_data])

# =========================
# PREDIKSI
# =========================
if st.sidebar.button("🔍 Prediksi Cluster"):
    scaled_input = scaler.transform(input_df)

    cluster = rf.predict(scaled_input)[0]
    prob = rf.predict_proba(scaled_input)[0]

    st.subheader("📌 Hasil Analisis")
    st.success(f"Nasabah termasuk ke **Cluster {cluster}**")

    prob_df = pd.DataFrame(
        [prob],
        columns=[f"Cluster {i}" for i in rf.classes_]
    )
    st.dataframe(prob_df)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📈 Feature Importance (Random Forest)")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

# =========================
# VISUALISASI FEATURE IMPORTANCE
# =========================
fig, ax = plt.subplots(figsize=(10,4))
importance_df.set_index("Feature").plot(kind="bar", ax=ax)
ax.set_title("Feature Importance - Random Forest")
ax.set_ylabel("Nilai Importance")
st.pyplot(fig)
