# %%
import xarray as xr
import cftime
import numpy as np
import copy

from datetime import datetime, timedelta
from MultiVar import *

from utils import *


def cams_preprocess():
    org_ds = xr.open_dataset(RAW_CAMS_FILE)
    # rename variable
    org_ds = org_ds.rename({"emiss_bio": "emiisop"})
    # interpolate to lat, lon cordination of VISIT
    interp_ds = interpolate(org_ds)
    interp_ds.to_netcdf(PREP_CAMS_FILE)


def alberi_preprocess():
    ds = []
    for f in reversed(RAW_ALBERI_LIST):
        org_ds = xr.open_dataset(f)
        year = int(f.split("_")[-1].split(".")[0])
        org_ds.coords["month"] = [cftime.datetime(year, m, 1) for m in np.arange(1, 13)]
        org_ds = org_ds.rename({"month": "time", "isop_flux": "emiisop"})
        ds.append(org_ds)
        print(year)

    ds = xr.concat(ds, dim="time")
    ds = interpolate(ds)
    ds["emiisop"] = ds["emiisop"].transpose("time", "lat", "lon")
    ds.to_netcdf(PREP_ALBERI_FILE)

    return ds


def tropomi_l3_extract():
    HCHO_DATA_DIR = f"/mnt/dg3/ngoc/obs_data/mon_TROPOMI_HCHO_L3/"
    RAW_HCHO_LIST = glob.glob(os.path.join(HCHO_DATA_DIR, "ORG", "*.nc"))
    vars = ["tropospheric_HCHO_column_number_density_clear", "qa_L3", "land_water_mask"]
    var_name = "hcho"
    m_name = "BIRA-TROPOMI-L3"
    time = "20180601-20240701"
    final_ds = []
    for f in RAW_HCHO_LIST:
        ds = xr.open_dataset(f)
        ds = ds[vars]
        final_ds.append(ds)
    final_ds = xr.concat(final_ds, dim="time")
    final_ds = final_ds.sortby("time")
    final_ds = final_ds.rename(
        {"tropospheric_HCHO_column_number_density_clear": "hcho_vcd"}
    )
    final_ds.to_netcdf(
        f"{HCHO_DATA_DIR}/EXTRACT/{var_name}_AERmon_{m_name}_historical_gn_{time}.nc"
    )


# %%
