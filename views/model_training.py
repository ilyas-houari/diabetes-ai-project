import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def show_model(df_clean):
    st.header("3. Model Training")

    # Features & target
    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    # -------------------------
    # Features
    # -------------------------
    st.subheader("Features Used for Training")
    st.write(", ".join(X.columns))

    # -------------------------
    # Target
    # -------------------------
    st.subheader("Target Variable")
    st.write("Outcome (0 = No Diabetes, 1 = Diabetes)")

    # -------------------------
    # Train / Test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    st.subheader("Train / Test Split")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Rows", X_train.shape[0])
    with col2:
        st.metric("Test Rows", X_test.shape[0])

    # -------------------------
    # Train model
    # -------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    st.success("Model trained successfully ✅")

    # -------------------------
    # Predictions
    # -------------------------
    y_pred = model.predict(X_test)

    st.subheader("Prediction Sample")

    results = pd.DataFrame({
        "Real": y_test.values[:10],
        "Predicted": y_pred[:10]
    })

    st.dataframe(results, width=300)
    
    # =========================
    # Evaluation Metrics
    # =========================
    st.subheader("Model Evaluation")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")

    with col2:
        st.metric("Precision", f"{precision:.2f}")

    with col3:
        st.metric("Recall", f"{recall:.2f}")

    with col4:
        st.metric("F1-score", f"{f1:.2f}")


    # =========================
    # Confusion Matrix
    # =========================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(3,2))

    ax.imshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="black", fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)