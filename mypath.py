import glob
import os
from const import *

"""
Directory structure
{data_dir}/data 
    /original
        /var
        /axl
    /processed_org_data 
        /annual_per_area_unit
        /mk_trends_map
    /visit_latlon
"""

DATA_SERVER = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/"
DATA_LOCAL = "../data/"


DATA_DIR = DATA_LOCAL
if os.path.exists(DATA_SERVER):
    DATA_DIR = DATA_SERVER

VAR_DIR = os.path.join(DATA_DIR, "original/var")
AXL_DIR = os.path.join(DATA_DIR, "original/axl")

LIST_ATTR = [attr.split("\\")[-1] for attr in glob.glob(os.path.join(VAR_DIR, "*"))]

ISOP_LIST = glob.glob(os.path.join(VAR_DIR, "emiisop", "*.nc"))

AREA_LIST = glob.glob(os.path.join(AXL_DIR, VAR_AREA, "*.nc"))
SFLTF_LIST = glob.glob(os.path.join(AXL_DIR, VAR_SFTLF, "*.nc"))
MASK_LIST = glob.glob(os.path.join(AXL_DIR, VAR_MASK, "*.nc"))


VISIT_LAT_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_lat.npy")
VISIT_LONG_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_long.npy")

RAW_CAMS_FILE = os.path.join(
    DATA_DIR,
    "CAMS_GLOB_BIO",
    "CAMS-GLOB-BIO_Glb_0.25x0.25_bio_isoprene_v3.1_monthly.nc",
)
PREP_CAMS_FILE = os.path.join(
    DATA_DIR,
    "original/var/emiisop",
    "emiisop_AERmon_CAMS-GLOB-BIO_historical_r1i1p1f1_gn_200001-202301.nc",
)

RAW_ALBERI_LIST = glob.glob(os.path.join(DATA_DIR, "ALBERI_ISOPRENEv2021", "*.nc"))
PREP_ALBERI_FILE = os.path.join(
    DATA_DIR,
    "original/var/emiisop",
    "emiisop_AERmon_ALBERI-v2021_historical_r1i1p1f1_gn_200001-201812.nc",
)
