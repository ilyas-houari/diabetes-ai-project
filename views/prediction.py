import streamlit as st
import numpy as np
import pandas as pd

def show_prediction(model, feature_names):
    st.header("4. Make Prediction")
    st.subheader("Enter Patient Data")

    inputs = []

    for feature in feature_names:
        if feature == "Pregnancies":
            value = st.number_input(feature, min_value=0, max_value=20, value=0, step=1)
        elif feature == "Age":
            value = st.number_input(feature, min_value=1, max_value=120, value=21, step=1)
        elif feature == "Glucose":
            value = st.number_input(feature, min_value=0.0, max_value=300.0, value=100.0)
        elif feature == "BloodPressure":
            value = st.number_input(feature, min_value=0.0, max_value=200.0, value=70.0)
        elif feature == "SkinThickness":
            value = st.number_input(feature, min_value=0.0, max_value=100.0, value=20.0)
        elif feature == "Insulin":
            value = st.number_input(feature, min_value=0.0, max_value=900.0, value=80.0)
        elif feature == "BMI":
            value = st.number_input(feature, min_value=0.0, max_value=70.0, value=25.0)
        elif feature == "DiabetesPedigreeFunction":
            value = st.number_input(feature, min_value=0.0, max_value=3.0, value=0.5)
        else:
            value = st.number_input(feature, value=0.0)

        inputs.append(value)

    if st.button("Predict"):
        data = np.array(inputs).reshape(1, -1)

        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0]

        if prediction == 1:
            st.error(f"High risk of Diabetes ({proba[1] * 100:.1f}%) ⚠️")
        else:
            st.success(f"Low risk of Diabetes ({proba[0] * 100:.1f}%) ✅")

        st.write(f"Diabetes probability: {proba[1] * 100:.2f}%")
        st.write(f"No Diabetes probability: {proba[0] * 100:.2f}%")

        if "Glucose" in feature_names:
            glucose_index = list(feature_names).index("Glucose")
            if inputs[glucose_index] > 140:
                st.warning("High glucose level detected")

        if "BMI" in feature_names:
            bmi_index = list(feature_names).index("BMI")
            if inputs[bmi_index] > 30:
                st.warning("High BMI detected")

        st.subheader("Patient Data Summary")
        df_input = pd.DataFrame([inputs], columns=feature_names)
        st.dataframe(df_input, use_container_width=True)