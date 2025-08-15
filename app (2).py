
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Wholesale Customers Clustering", page_icon="🛒")
st.title("Wholesale Customers – KMeans Clustering App")
st.markdown(
    "This app loads your **StandardScaler** and **KMeans** artifacts "
    "and predicts the cluster for new inputs. It also shows cluster centers in original units."
)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")
    return scaler, kmeans

feature_names = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

# Try to load artifacts and handle errors gracefully
artifacts_loaded = True
try:
    scaler, kmeans = load_artifacts()
except Exception as e:
    artifacts_loaded = False
    st.error(
        "Couldn't load artifacts. Make sure **scaler.pkl** and **kmeans.pkl** are in the same folder as this app.\n\n"
        f"Error: {e}"
    )

tab1, tab2 = st.tabs(["Single input", "Batch CSV"])

with tab1:
    st.subheader("Single sample prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        Fresh = st.number_input("Fresh", min_value=0, value=12000, step=100)
        Grocery = st.number_input("Grocery", min_value=0, value=8000, step=100)
    with col2:
        Milk = st.number_input("Milk", min_value=0, value=5000, step=100)
        Frozen = st.number_input("Frozen", min_value=0, value=1000, step=50)
    with col3:
        Detergents_Paper = st.number_input("Detergents_Paper", min_value=0, value=2000, step=50)
        Delicassen = st.number_input("Delicassen", min_value=0, value=1000, step=50)

    if st.button("Predict cluster", type="primary", disabled=not artifacts_loaded):
        X = pd.DataFrame([[Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen]], columns=feature_names)
        # The original notebook uses log1p before scaling. If your scaler was fit on log-transformed data,
        # you must apply the SAME transform here. Uncomment the next line if you trained on log1p(data).
        # X = np.log1p(X)
        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])
        st.success(f"Predicted cluster: **{cluster}**")

with tab2:
    st.subheader("Batch prediction from CSV")
    st.write("CSV must contain columns:", feature_names)
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None and artifacts_loaded:
        try:
            df = pd.read_csv(up)
            # Uncomment if you fit scaler on log1p-transformed data:
            # df = np.log1p(df[feature_names])
            X_scaled = scaler.transform(df[feature_names])
            clusters = kmeans.predict(X_scaled)
            out = df.copy()
            out["Cluster"] = clusters
            st.write("Preview:", out.head())
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv, "clusters.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.divider()

if artifacts_loaded:
    st.subheader("Cluster centers (back-transformed to original units)")
    try:
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_original, columns=feature_names)
        centers_df.index.name = "Cluster"
        st.dataframe(centers_df)
    except Exception as e:
        st.info("If you trained on **log1p(data)** then inverse-transforming the centers gives **exp(center) - 1**. "
                "Adjust accordingly in your notebook or here.")
        st.text(f"Details: {e}")
