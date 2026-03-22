import streamlit as st
import pandas as pd

def show_cleaning(df, df_clean):
    st.header("2. Data Cleaning")

    # Missing values
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing"]
    missing = missing[missing["Missing"] > 0]
    missing = missing.reset_index(drop=True)

    st.subheader("Missing Values")
    st.dataframe(missing, width=300)

    # Clean dataset
    st.subheader("Clean Dataset Preview")

    rows = st.slider("Rows", 5, len(df_clean), 10)
    st.dataframe(df_clean.head(rows))