import streamlit as st
import pandas as pd
import numpy as np

# =====================================
# Page configuration
# =====================================
st.set_page_config(
    page_title="Diabetes Prediction Project",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# Project title
# =====================================
st.title("AI System for Early Detection of Diabetes")

st.write(
    """
    This application presents the main steps of a machine learning project
    for diabetes prediction: data exploration, data cleaning, and preparation
    for model training.
    """
)

# =====================================
# Load dataset
# =====================================
df = pd.read_csv("diabetes.csv")

# Keep a copy of the original data
df_raw = df.copy()

# =====================================
# Dataset overview
# =====================================
st.header("1. Dataset Overview")

st.subheader("Dataset Preview")
rows = st.slider("Number of rows to display", 5, len(df), 10)
st.dataframe(df.head(rows))

st.subheader("Dataset Shape")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.subheader("Columns")
st.write(list(df.columns))

# =====================================
# Data problems detection
# =====================================
st.header("2. Data Problems Detection")

# Identifier column
if "Person" in df.columns:
    st.subheader("Identifier Column")
    st.write("The column 'Person' is an identifier and will not be used for analysis or model training.")

# Missing values
st.subheader("Missing Values")
missing = df.isnull().sum().reset_index()
missing.columns = ["Column", "Missing"]

# keep only columns with missing values
missing = missing[missing["Missing"] > 0]

# reset index so the number column disappears
missing = missing.reset_index(drop=True)

st.dataframe(missing, width=300)

# Zero values in columns where zero is not realistic
st.subheader("Impossible Zero Values")
zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_columns:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        st.write(f"{col}: {zero_count} zero values")

# =====================================
# Data cleaning
# =====================================
st.header("3. Data Cleaning")

df_clean = df.copy()

# Remove identifier column
if "Person" in df_clean.columns:
    df_clean = df_clean.drop("Person", axis=1)

# Replace impossible zeros with NaN
for col in zero_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(0, np.nan)

# Fill missing numeric values with median
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

st.write("Cleaning steps applied:")
st.write("- Removed identifier column: Person")
st.write("- Replaced impossible zeros with missing values")
st.write("- Filled missing numeric values using the median")

# =====================================
# Clean dataset preview
# =====================================
st.header("4. Clean Dataset")

st.subheader("Remaining Missing Values")
remaining_missing = df_clean.isnull().sum().reset_index()
remaining_missing.columns = ["Column", "Missing"]

remaining_missing = remaining_missing[remaining_missing["Missing"] > 0]
remaining_missing = remaining_missing.reset_index(drop=True)

st.dataframe(remaining_missing, width=300)

st.subheader("Cleaned Dataset Preview")

rows_to_show = st.slider(
    "Number of rows to display",
    min_value=5,
    max_value=len(df_clean),
    value=10,
    step=5
)

preview_df = df_clean.copy()

# add Person column back for display
preview_df.insert(0, "Person", df["Person"])

st.dataframe(preview_df.head(rows_to_show))




# =====================================
# Clean dataset statistics
# =====================================
st.header("5. Clean Dataset Statistics")
st.write(df_clean.describe())

# =====================================
# Target distribution
# =====================================
if "Outcome" in df_clean.columns:
    st.header("6. Outcome Distribution")
    st.bar_chart(df_clean["Outcome"].value_counts())

# =====================================
# Final note
# =====================================
st.info("Next step: feature preparation, train/test split, model training, and evaluation.")