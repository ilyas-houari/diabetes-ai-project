We will follow this structure in the web app:
1. Project Introduction
2. Dataset Overview
3. Dataset Exploration (EDA)
4. Data Problems Detection
5. Data Cleaning
6. Feature Preparation
7. Model Training
8. Model Evaluation
9. Prediction Interface

------------------

app.py → main web app

diabetes.csv → dataset

dataset_description.txt → explanation of columns

model/ → saved model later

utils/ → helper functions

assets/ → images / style files later

---------------

streamlit → build the web interface

pandas → handle the dataset

numpy → numerical calculations

scikit-learn → machine learning models

-------------

Run the app like this
python -m streamlit run app.py