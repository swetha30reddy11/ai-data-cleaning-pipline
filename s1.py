import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 AI-Powered Data Cleaning Pipeline (Enhanced)")
st.markdown(
    """
    Upload a CSV file to automatically profile, clean, and prepare your dataset
    using **AI/ML-powered techniques** with full user control.
    """
)

# -------------------- Sidebar Controls --------------------
with st.sidebar:
    st.header("⚙️ Cleaning Settings")
    outlier_contamination = st.slider(
        "Outlier Sensitivity (Isolation Forest)", 0.01, 0.3, 0.05
    )

    scaling_method = st.selectbox(
        "Feature Scaling Method",
        ["None", "StandardScaler", "MinMaxScaler"]
    )

    show_ai_suggestions = st.checkbox("Show AI Cleaning Suggestions", value=True)

# -------------------- Upload CSV --------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

if uploaded_file is not None:
    original_df = pd.read_csv(uploaded_file)
    df = original_df.copy()

    # -------------------- Data Quality Report (Before) --------------------
    st.subheader("📊 Data Quality Report (Before Cleaning)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    st.dataframe(df)
    st.markdown("---")

    if st.button("🚀 Run AI Cleaning Pipeline"):
        progress = st.progress(0)

        # -------------------- Identify Column Types --------------------
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        # -------------------- AI Suggestions --------------------
        if show_ai_suggestions:
            st.subheader("🤖 AI Cleaning Suggestions")
            for col in df.columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 20:
                    st.warning(
                        f"Column '{col}' has {missing_pct:.1f}% missing values – consider dropping or advanced imputation."
                    )
        progress.progress(20)

        # -------------------- Handle Missing Values --------------------
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                train_df = df[df[col].notna()]
                test_df = df[df[col].isna()]
                features = [c for c in numeric_cols if c != col]

                if len(features) == 0:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(train_df[features], train_df[col])
                    df.loc[df[col].isna(), col] = model.predict(test_df[features])

        for col in cat_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

        progress.progress(50)

        # -------------------- Remove Duplicates --------------------
        before_rows = df.shape[0]
        df = df.drop_duplicates()
        removed_duplicates = before_rows - df.shape[0]

        # -------------------- Outlier Detection --------------------
        removed_outliers = 0
        if len(numeric_cols) > 0:
            iso = IsolationForest(
                contamination=outlier_contamination, random_state=42
            )
            df["_outlier"] = iso.fit_predict(df[numeric_cols])
            removed_outliers = (df["_outlier"] == -1).sum()
            df = df[df["_outlier"] == 1].drop(columns=["_outlier"])

        progress.progress(70)

        # -------------------- Feature Scaling --------------------
        if scaling_method != "None" and len(numeric_cols) > 0:
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        progress.progress(90)

        # -------------------- Data Quality Report (After) --------------------
        st.subheader("📊 Data Quality Report (After Cleaning)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Rows", df.shape[0])
        col2.metric("Duplicates Removed", removed_duplicates)
        col3.metric("Outliers Removed", removed_outliers)

        st.subheader("✅ Cleaned Dataset")
        st.dataframe(df)

        # -------------------- Download Cleaned Data --------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned CSV",
            csv,
            "cleaned_data.csv",
            "text/csv"
        )

        st.success("🎉 AI data cleaning pipeline completed successfully!")
        progress.progress(100)

else:
    st.info("👆 Please upload a CSV file to begin.")
