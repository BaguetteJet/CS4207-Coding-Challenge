import requests
import os

DATA_DIR = "./data"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"

os.makedirs(DATA_DIR, exist_ok=True)

for month in [1, 2, 3, 4]:
    file = f"yellow_tripdata_2026-{month:02d}.parquet"
    url = DATA_URL + file

    response = requests.get(url)
    response.raise_for_status()

    with open(f"{DATA_DIR}/{file}", "wb") as f:
        f.write(response.content)

    print(f"Downloaded {file}")