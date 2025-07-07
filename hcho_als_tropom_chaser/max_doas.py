# %%
import pandas as pd
from datetime import datetime
from glob import glob

BASE_MAXDOAS_DIR = "/mnt/dg3/ngoc/max-doas/unzip"
MAXDOAS_FILES = glob(f"{BASE_MAXDOAS_DIR}/*.dat")

MAX_DOAS_COORDS = {
    "Chiba": {"lat": 35.63, "lon": 140.1},
    "Phimai": {"lat": 15.18, "lon": 102.56},
    "Tsukuba": {"lat": 36.06, "lon": 140.13},
}


def read_maxdoas_csv(file_path, skiprows=26):
    sel_cols = ["hcho"]
    try:
        df = pd.read_csv(file_path, skiprows=skiprows, delimiter=" ")
        df["time"] = df.apply(
            lambda x: datetime(
                year=int(x["Year"]),
                month=int(x["Month"]),
                day=int(x["Day"]),
            ),
            axis=1,
        )
        df = df.groupby("time").mean()
        df = df[sel_cols]
        return df

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def read_all_maxdoas_csv(file_list=MAXDOAS_FILES):
    all_data = {station: [] for station in MAX_DOAS_COORDS.keys()}
    for station in MAX_DOAS_COORDS.keys():
        station_files = [f for f in file_list if station.lower() in f]
        for file_path in station_files:
            df = read_maxdoas_csv(file_path)
            all_data[station].append(df)
        all_data[station] = pd.concat(all_data[station])
    return all_data


# %%
