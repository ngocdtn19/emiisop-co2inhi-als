# %%
import glob
import os
import math
import geopandas as gpd
import regionmask
import random


DIM_TIME = "time"
DIM_LAT = "lat"
DIM_LON = "lon"

VAR_AREA = "areacella"
VAR_ISOP = "emiisop"
VAR_SFTLF = "sftlf"
VAR_MASK = "mask"

VAR_MONTH_RATE = f"{VAR_ISOP}_month"
VAR_ISOP_AREA = f"{VAR_ISOP}_{VAR_AREA}"

ISOP_2_C = 60 / 68
DAY_RATE = 60 * 60 * 24

KG_2_G = 1e3
KG_2_TG = 1e-9
KG_2_PG = 1e-12
K_2_C = 273.15

MG_2_G = 1e-6  # convert from micro gram to gram
MG_2_TG = 1e-18  # convert from micro gram to Teragram

REGION = regionmask.defined_regions.srex
# LIST_REGION = [REGION.regions[i].abbrev for i in REGION.regions.keys()]

LIST_SREX_REGION = [
    # "AMZ",
    # "ENA",
    #    "SAF",
    #    "MED",
    #    "CEU",
    # "EAS",
    #    "SAS",
    # "SEA",
    "NAU",
]

hoque_reg_coords = {
    "Amazonia": {
        "lat": [5, -20],  # 5°N to 20°S
        "lon": [-75, -40],  # 40-75°W
    },
    "S-E US": {
        "lat": [36, 30],  # 30-36°N
        "lon": [-95, -78],  # 75-100°W
    },
    "Mato Grosso": {
        "lat": [-10, -16],  # 10°S to 16°S
        "lon": [-60, -50],  # 60°W to 50°W
    },
    "Indonesia": {
        "lat": [6, -10],  # 6°N to 10°S (descending]
        "lon": [95, 142],  # 95°E to 142°E
    },
    "South China": {
        "lat": [28, 22],  # 28°N to 22°N
        "lon": [100, 112],  # 100°E to 112°E
    },
    "C_Africa": {
        "lat": [5, -4],  # 5°N to 4°S
        "lon": [10, 40],  # 10°E to 40°E
    },
    "N_Africa": {
        "lat": [15, 5],  # 15°N to 5°N
        "lon": [-10, 30],  # 10°W to 30°E → -10 to 30
    },
    "S_Africa": {
        "lat": [-5, -15],  # 5°S to 15°S
        "lon": [10, 30],  # 10°E to 30°E
    },
}

HOQUE_REGIONS = list(hoque_reg_coords.keys())
LIST_REGION = HOQUE_REGIONS + LIST_SREX_REGION

LIST_COLOR = [
    "#ff5005",
    "#ffe100",
    "#ffff80",
    "#990000",
    "#740aff",
    "#e0ff66",
    "#00998f",
    "#5ef1f2",
    "#ff0010",
    "#426600",
    "#ffa8bb",
    "#ffa405",
    "#003380",
    "#c20088",
    "#9dcc00",
    "#8f7c00",
    "#94ffb5",
    "#808080",
    "#ffcc99",
    "#2bce48",
    "#005c31",
    "#191919",
    "#4c005c",
    "#993f00",
    "#0075dc",
    "#f0a3ff",
]


ROI_COLORS = {roi: color for roi, color in zip(LIST_REGION, LIST_COLOR)}

VIZ_OPT = {
    "emiisop": {
        "map_unit": "[gC m$^{-2}$ yr$^{-1}$]",
        "map_vmin": 0,
        "map_vmax": 40,
        "map_levels": 17,
        "line_bar_unit": "[TgC yr$^{-1}$]",
        "line_ylim": [350, 650],
        "bar_ylim": [0, 670],
    },
}

WORLD_SHP = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# %%
