import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_cleaning(df, df_clean):
    st.header("2. Data Cleaning")

    # ========================
    # Missing values (original)
    # ========================
    st.subheader("Missing Values (Before Cleaning)")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing"]
    missing = missing[missing["Missing"] > 0]
    missing = missing.reset_index(drop=True)

    st.dataframe(missing, width=300)

    # ========================
    # Impossible zero values
    # ========================
    st.subheader("Impossible Zero Values")

    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    zero_data = []
    for col in zero_columns:
        if col in df.columns:
            count = (df[col] == 0).sum()
            zero_data.append({"Column": col, "Zero Values": count})

    st.dataframe(pd.DataFrame(zero_data), width=350)

    # ========================
    # Cleaning explanation
    # ========================
    st.subheader("Cleaning Steps Applied")

    st.write("""
    - Removed identifier column: Person  
    - Replaced impossible zeros with missing values  
    - Filled missing values using the median  
    """)

    # ========================
    # Remaining missing values
    # ========================
    st.subheader("Remaining Missing Values")

    remaining = df_clean.isnull().sum().reset_index()
    remaining.columns = ["Column", "Missing"]
    remaining = remaining[remaining["Missing"] > 0]
    remaining = remaining.reset_index(drop=True)

    if remaining.empty:
        st.success("No missing values remaining ✅")
    else:
        st.dataframe(remaining, width=300)

    # ========================
    # Clean dataset preview
    # ========================
    st.subheader("Clean Dataset Preview")

    rows = st.slider("Rows", 5, len(df_clean), 10)
    st.dataframe(df_clean.head(rows))

    # ========================
    # Statistics
    # ========================
    st.subheader("Dataset Statistics")

    st.dataframe(df_clean.describe())

    # ========================
    # Outcome distribution
    # ========================
    if "Outcome" in df_clean.columns:
        
        st.subheader("Outcome Distribution")

        counts = df_clean["Outcome"].value_counts()

        fig, ax = plt.subplots(figsize=(3,2))

        labels = ["No Diabetes", "Diabetes"]
        bars = ax.bar(labels, counts.values)

        ax.set_title("Outcome", fontsize=10)
        ax.set_ylabel("Patients", fontsize=8)

        # Y axis ticks
        ax.set_yticks(np.arange(0, counts.max()+100, 100))

        # smaller ticks
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

        # 🔥 ADD VALUES + PERCENTAGE ON BARS
        total = counts.sum()

        for bar in bars:
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{int(height)}\n({percentage:.1f}%)",
                ha='center',
                va='bottom',
                fontsize=7
            )

        # clean borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)