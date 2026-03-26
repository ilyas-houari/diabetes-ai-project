import streamlit as st

from utils.load_data import load_dataset
from utils.cleaning import clean_data

from views.overview import show_overview
from views.data_cleaning import show_cleaning
from views.model_training import show_model, train_model
from views.prediction import show_prediction

# =====================================
# Page configuration
# =====================================
st.set_page_config(
    page_title="Diabetes Prediction Project",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# Sidebar style
# =====================================
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    text-align: left;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #2a2f3a;
    background-color: #111827;
    color: white;
    font-weight: 500;
    margin-bottom: 8px;
}
div.stButton > button:hover {
    border: 1px solid #4b5563;
    background-color: #1f2937;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# Load and clean data
# =====================================
df = load_dataset()
df_clean = clean_data(df)

# =====================================
# Sidebar navigation
# =====================================
if "section" not in st.session_state:
    st.session_state.section = "Overview"

st.sidebar.markdown("## Choose Section")

if st.sidebar.button("Overview", use_container_width=True):
    st.session_state.section = "Overview"

if st.sidebar.button("Data Cleaning", use_container_width=True):
    st.session_state.section = "Data Cleaning"

if st.sidebar.button("Model Training", use_container_width=True):
    st.session_state.section = "Model Training"

if st.sidebar.button("Prediction", use_container_width=True):
    st.session_state.section = "Prediction"

section = st.session_state.section

# =====================================
# Display selected section
# =====================================
if section == "Overview":
    st.title("AI System for Diabetes Prediction")

    st.write(
        """
        This application presents the main steps of a machine learning project
        for diabetes prediction: dataset exploration, data cleaning, and preparation
        for model training.
        """
    )
    show_overview(df)

elif section == "Data Cleaning":
    show_cleaning(df, df_clean)

elif section == "Model Training":
    show_model(df_clean)

elif section == "Prediction":
    results = train_model(df_clean)
    model = results["rf_model"]
    features = results["X"].columns
    show_prediction(model, features)