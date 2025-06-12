import pandas as pd
import os

def load_emission_data(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "data", filename)
    df = pd.read_csv(file_path)
    return df
