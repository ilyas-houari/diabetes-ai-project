import pandas as pd

def load_dataset():
    df = pd.read_csv("diabetes.csv")
    return df