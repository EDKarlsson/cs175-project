import pandas as pd
import os

def load_data():
    DATA_DIR ="../../data"
    ARTICLES = os.listdir(DATA_DIR)
    return [pd.read_csv(DATA_DIR + "/" + f) for f in ARTICLES if '.csv' in f]


data = load_data()
