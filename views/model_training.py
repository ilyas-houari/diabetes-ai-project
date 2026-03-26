import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


    # =========================
    # Train Logistic Regression
    # =========================
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)

    st.success("Logistic Regression trained ✅")

    # =========================
    # Train Random Forest
    # =========================
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    st.success("Random Forest trained ✅")

    st.subheader("Prediction Sample")

    results = pd.DataFrame({
        "Real": y_test.values[:10],
        "Logistic Regression": y_pred_lr[:10],
        "Random Forest": y_pred_rf[:10]
    })

    st.dataframe(results, width=300)
    
    # =========================
    # Model Evaluation
    # =========================
    st.subheader("Model Comparison")

    # Logistic Regression metrics
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)

    # Random Forest metrics
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Logistic Regression": [acc_lr, prec_lr, rec_lr, f1_lr],
        "Random Forest": [acc_rf, prec_rf, rec_rf, f1_rf]
    })

    st.dataframe(metrics_df, width=500)


    # =========================
    # Confusion Matrix
    # =========================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred_rf)

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