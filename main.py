import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# Page Config
# =========================
st.set_page_config(layout="wide")
st.title("💳 Credit Card Clustering (RF + Logistic Regression)")

# =========================
# Load Models
# =========================
scaler = joblib.load("model/scaler.pkl")
kmeans = joblib.load("model/kmeans.pkl")
rf = joblib.load("model/random_forest.pkl")
logreg = joblib.load("model/logreg.pkl")
features = joblib.load("model/features.pkl")

# =========================
# Upload Data
# =========================
uploaded = st.file_uploader("Upload CC_GENERAL.csv", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)

# =========================
# Preprocessing
# =========================
df = df.drop(columns=["CUST_ID"], errors="ignore")
df = df.fillna(df.median(numeric_only=True))

# =========================
# Scaling
# =========================
X = df[features]
X_scaled = scaler.transform(X)

# =========================
# Clustering
# =========================
df["Cluster"] = kmeans.predict(X_scaled)

st.subheader("📊 Data + Cluster")
st.dataframe(df.head())

# =========================
# PCA Visualization
# =========================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
ax.set_title("Visualisasi Cluster (PCA)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

st.pyplot(fig)

# =========================
# Random Forest Prediction
# =========================
st.subheader("🌲 Random Forest Prediction")
rf_pred = rf.predict(X_scaled)
st.bar_chart(pd.Series(rf_pred).value_counts())

# =========================
# Logistic Regression Prediction
# =========================
st.subheader("📐 Logistic Regression Prediction")
lr_pred = logreg.predict(X_scaled)
st.bar_chart(pd.Series(lr_pred).value_counts())
