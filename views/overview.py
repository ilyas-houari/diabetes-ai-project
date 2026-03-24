import streamlit as st

def show_overview(df):
    st.header("1. Dataset Overview")

    rows = st.slider("Number of rows to display", 5, len(df), 10)
    st.dataframe(df.head(rows))

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(list(df.columns))

    # Add this 👇
    if "Person" in df.columns:
        st.info("The column 'Person' is an identifier and will not be used for model training.")