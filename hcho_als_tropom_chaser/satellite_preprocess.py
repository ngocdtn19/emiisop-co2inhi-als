# %%
import os
import xarray as xr
import glob

import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import pygtool


def prep_hcho_tropomi(ds):

    geogrid = pygtool.readgrid()
    chaser_lon, chaser_lat = geogrid.getlonlat()
    # filter qa value
    ds = ds.where(ds["qa_L3"] == 1)
    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = ds.interp(lat=chaser_lat, lon=sorted(((chaser_lon + 180) % 360) - 180))

    return ds


def satellite_HchoL3_extract(product="tropomi"):
    var_name = "hcho"
    data_dir = f"/mnt/dg3/ngoc/obs_data"

    # OMI data
    m_name = "mon_BIRA_OMI_HCHO_L3"
    org_hcho_files = glob.glob(os.path.join(data_dir, m_name, "ORG", "*/*.nc"))
    ak_col = "HCHO_averaging_kernel"
    time = "20050101-20231201"

    # TROPOOMI data
    if product == "tropomi":
        m_name = "mon_TROPOMI_HCHO_L3"
        org_hcho_files = glob.glob(os.path.join(data_dir, m_name, "ORG", "*.nc"))
        time = "20180601-20240701"

    vars = [
        "tropospheric_HCHO_column_number_density_clear",
        "tropospheric_HCHO_column_number_density_apriori",
        "HCHO_volume_mixing_ratio_dry_air_apriori",
        "qa_L3",
        "land_water_mask",
        "surface_pressure",
        "ctm_sigma_a",
        "ctm_sigma_b",
        ak_col,
    ]
    final_ds = []
    for f in org_hcho_files:
        print(f)
        ds = xr.open_dataset(f)
        ds = ds[vars]
        ds = prep_hcho_tropomi(ds)
        final_ds.append(ds)

    final_ds = xr.concat(final_ds, dim="time")
    final_ds = final_ds.sortby("time")
    final_ds = final_ds.rename(
        {
            "tropospheric_HCHO_column_number_density_clear": "tcolhcho",
            "tropospheric_HCHO_column_number_density_apriori": "tcolhcho_apriori",
            "HCHO_volume_mixing_ratio_dry_air_apriori": "hcho_vertical_profile_apriori",
            ak_col: "AK",
        }
    )
    out_dir = f"{data_dir}/{m_name}/EXTRACT"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = f"{var_name}_AERmon_{m_name}_historical_gn_{time}.nc"
    final_ds.to_netcdf(f"{out_dir}/{filename}")
    return final_ds


# %%
