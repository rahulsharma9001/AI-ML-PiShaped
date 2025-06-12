import pandas as pd
import os

def load_churn_data(filename):
    # Get path to classification/data/<filename>
    base_dir = os.path.dirname(__file__)  # classification/
    file_path = os.path.join(base_dir, "data", filename)  # classification/data/<filename>

    df = pd.read_csv(file_path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df