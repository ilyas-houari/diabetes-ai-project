import streamlit as st

from utils.load_data import load_dataset
from utils.cleaning import clean_data

from views.overview import show_overview
from views.data_cleaning import show_cleaning
from views.model_training import show_model
# =====================================
# Page configuration
# =====================================
st.set_page_config(
    page_title="Diabetes Prediction Project",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# Main title
# =====================================
st.title("AI System for Diabetes Prediction")

st.write(
    """
    This application presents the main steps of a machine learning project
    for diabetes prediction: dataset exploration, data cleaning, and preparation
    for model training.
    """
)

# =====================================
# Load and clean data
# =====================================
df = load_dataset()
df_clean = clean_data(df)

# =====================================
# Sidebar navigation
# =====================================
section = st.sidebar.selectbox(
    "Choose Section",
    ["Overview", "Data Cleaning","Model Training"]
)

# =====================================
# Display selected section
# =====================================
if section == "Overview":
    show_overview(df)

elif section == "Data Cleaning":
    show_cleaning(df, df_clean)
    
elif section == "Model Training":
    show_model(df_clean)