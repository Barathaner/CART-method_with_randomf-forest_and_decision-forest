import pandas as pd

def preprocess_dataframe(df):
    df.dropna(inplace=True)
    return df