import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Credit Card Clustering & Classification",
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
    logreg = joblib.load("model/logreg.pkl")
    features = joblib.load("model/features.pkl")
    return scaler, kmeans, rf, logreg, features

scaler, kmeans, rf, logreg, features = load_models()

# =========================
# TITLE
# =========================
st.title("💳 Credit Card Clustering")
st.write("K-Means + Random Forest & Logistic Regression")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("🔢 Input Data Nasabah")

input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature, value=0.0
    )

input_df = pd.DataFrame([input_data])

# =========================
# PREDIKSI
# =========================
if st.sidebar.button("🔍 Prediksi Cluster"):
    scaled_input = scaler.transform(input_df)

    rf_cluster = rf.predict(scaled_input)[0]
    lr_cluster = logreg.predict(scaled_input)[0]

    rf_prob = rf.predict_proba(scaled_input)[0]
    lr_prob = logreg.predict_proba(scaled_input)[0]

    st.subheader("📌 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Random Forest → Cluster {rf_cluster}")
        rf_df = pd.DataFrame(
            [rf_prob],
            columns=[f"Cluster {i}" for i in rf.classes_]
        )
        st.dataframe(rf_df)

    with col2:
        st.info(f"Logistic Regression → Cluster {lr_cluster}")
        lr_df = pd.DataFrame(
            [lr_prob],
            columns=[f"Cluster {i}" for i in logreg.classes_]
        )
        st.dataframe(lr_df)

# =========================
# INTERPRETASI MODEL
# =========================
st.subheader("📈 Interpretasi Model")

tab1, tab2 = st.tabs(["Random Forest", "Logistic Regression"])

with tab1:
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df)

    fig, ax = plt.subplots(figsize=(10,4))
    importance_df.set_index("Feature").plot(kind="bar", ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    st.pyplot(fig)

with tab2:
    coef_df = pd.DataFrame(
        logreg.coef_,
        columns=features,
        index=[f"Cluster {i}" for i in logreg.classes_]
    )

    st.dataframe(coef_df)

    cluster_select = st.selectbox(
        "Pilih Cluster",
        coef_df.index
    )

    fig, ax = plt.subplots(figsize=(10,4))
    coef_df.loc[cluster_select].plot(kind="bar", ax=ax)
    ax.set_title(f"Koefisien Logistic Regression - {cluster_select}")
    st.pyplot(fig)
