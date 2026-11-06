import os
import requests
import pandas as pd

DATA_DIR = "./data"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"

START, END = "2024-01", "2026-09"

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for date in pd.date_range(START, END, freq="MS"):
        file = f"yellow_tripdata_{date.strftime("%Y-%m")}.parquet"
        url = DATA_URL + file

        response = requests.get(url)
        response.raise_for_status()

        with open(os.path.join(DATA_DIR, file), "wb") as f:
            f.write(response.content)

        print(f"Downloaded {file}")