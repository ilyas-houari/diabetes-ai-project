import numpy as np

def clean_data(df):
    df_clean = df.copy()

    # remove identifier
    if "Person" in df_clean.columns:
        df_clean = df_clean.drop("Person", axis=1)

    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # replace zeros
    for col in zero_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)

    # fill missing
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    return df_clean