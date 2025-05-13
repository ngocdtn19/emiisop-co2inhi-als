# %%
import os
import pandas as pd
import numpy as np
import xarray as xr

import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import os
import pygtool_core
import pygtool


geogrid = pygtool.readgrid()
Clon, Clat = geogrid.getlonlat()
# === Config ===
ak_dir = "/mnt/dg2/hoque/TROPOMI_model_analysis/2019/analysis/"
output_file = (
    "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/TROPOMI_AK_2019_combined.nc"
)

months = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


def load_month_data():
    list_df = [
        pd.read_csv(
            f"{ak_dir}/AK_{m}.txt",
            delimiter="\t",
            header=None,
            names=["time", "lat", "lon", "AK", "pressure", "level"],
        )
        for m in months
    ]
    merge_df = pd.concat(list_df, ignore_index=True)
    merge_df["time"] = pd.to_datetime(merge_df["time"], format="%Y/%m/%d")

    dates = merge_df.time.unique()

    list_ds = []
    for t in dates:
        print(t)
        df = merge_df[merge_df["time"] == t]
        ds = df.groupby(["time", "lat", "lon", "level"]).mean().to_xarray()
        ds = ds.interp(
            lat=Clat,
            lon=sorted(((Clon + 180) % 360) - 180),
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
        list_ds.append(ds)

    merged_ds = xr.concat(list_ds, dim="time")
    merged_ds.to_netcdf(output_file)

    return df


# %%
# %%
# %%
import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import pygtool_core
import pygtool


def convert_longitude_to_0_360(longitude):
    """Convert longitude from -180 to 180 range to 0 to 360 range."""
    return np.where(longitude < 0, longitude + 360, longitude)


# === Config ===
ak_dir = "/mnt/dg2/hoque/TROPOMI_model_analysis/2019/analysis/"
output_file = (
    "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/TROPOMI_AK_2019_combined_test.nc"
)
plot_dir = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/plots/"

# Create plot directory if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)

months = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

# Get target grid from pygtool
geogrid = pygtool.readgrid()
target_lon, target_lat = geogrid.getlonlat()
# Ensure target_lon is in 0-360 range for consistent interpolation
target_lon_0_360 = convert_longitude_to_0_360(target_lon)


def plot_comparison(
    original_ds, regridded_ds, variable, time_idx=0, level_idx=0, date_str=""
):
    """Plot comparison between original and regridded data for visual verification"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot original data
    orig_data = original_ds[variable].isel(time=time_idx, level=level_idx).values
    im1 = ax1.pcolormesh(original_ds.lon, original_ds.lat, orig_data, shading="auto")
    ax1.set_title(f"Original Data - {date_str}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    plt.colorbar(im1, ax=ax1)

    # Plot regridded data
    regrid_data = regridded_ds[variable].isel(time=time_idx, level=level_idx).values
    im2 = ax2.pcolormesh(
        regridded_ds.lon, regridded_ds.lat, regrid_data, shading="auto"
    )
    ax2.set_title(f"Regridded Data - {date_str}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{variable}_{date_str}.png")


def regrid_data(ds, target_lat, target_lon):
    """
    Regrid data from original coordinates to target coordinates.

    Args:
        ds (xarray.Dataset): Original dataset
        target_lat (numpy.ndarray): Target latitude coordinates
        target_lon (numpy.ndarray): Target longitude coordinates (0-360 range)

    Returns:
        xarray.Dataset: Regridded dataset
    """
    # Create output dataset with the same dimensions as input but with new coordinates
    regridded_ds = xr.Dataset(
        coords={
            "time": ds.time,
            "lat": target_lat,
            "lon": target_lon,
            "level": ds.level,
        }
    )

    # Convert original longitudes to 0-360 range
    orig_lon_0_360 = convert_longitude_to_0_360(ds.lon.values)

    # Sort the longitudes and corresponding data to ensure monotonically increasing
    sort_idx = np.argsort(orig_lon_0_360)
    sorted_lon = orig_lon_0_360[sort_idx]

    # Regrid each variable in the dataset
    for var_name, var in ds.data_vars.items():
        print(f"Regridding variable: {var_name}")

        # Create empty array for regridded data
        regridded_data = np.zeros(
            (len(ds.time), len(target_lat), len(target_lon), len(ds.level))
        )

        # Process each time and level combination
        for t in range(len(ds.time)):
            for l in range(len(ds.level)):
                # Extract the 2D slice and reorder according to sorted longitudes
                slice_data = var[t, :, :, l].values
                sorted_slice = slice_data[:, sort_idx]

                # Create interpolator
                interpolator = RegularGridInterpolator(
                    (ds.lat.values, sorted_lon),
                    sorted_slice,
                    bounds_error=False,
                    fill_value=None,
                )

                # Create a grid of target coordinates
                lat_grid, lon_grid = np.meshgrid(
                    target_lat, target_lon_0_360, indexing="ij"
                )
                points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

                # Interpolate data
                regridded_slice = interpolator(points).reshape(lat_grid.shape)
                regridded_data[t, :, :, l] = regridded_slice

        # Add the regridded variable to the output dataset
        regridded_ds[var_name] = xr.DataArray(
            data=regridded_data,
            dims=["time", "lat", "lon", "level"],
            coords={
                "time": ds.time,
                "lat": target_lat,
                "lon": target_lon,
                "level": ds.level,
            },
            attrs=var.attrs,
        )

    return regridded_ds


def load_and_process_data():
    """Load data for all months, convert to xarray, and regrid to target coordinates"""
    print("Loading data from all months...")
    # Load data from all months into a single dataframe
    list_df = [
        pd.read_csv(
            f"{ak_dir}/AK_{m}.txt",
            delimiter="\t",
            header=None,
            names=["time", "lat", "lon", "AK", "pressure", "level"],
        )
        for m in months
    ]
    merge_df = pd.concat(list_df, ignore_index=True)
    merge_df["time"] = pd.to_datetime(merge_df["time"], format="%Y/%m/%d")

    # Get unique dates
    dates = sorted(merge_df.time.unique())

    # Process data for each date
    all_datasets = []
    for i, date in enumerate(dates):
        print(f"Processing data for {date}")
        # Filter data for current date
        df_date = merge_df[merge_df["time"] == date]

        # Create a regular grid from the irregular points
        unique_lats = sorted(df_date.lat.unique())
        unique_lons = sorted(df_date.lon.unique())
        unique_levels = sorted(df_date.level.unique())

        # Create empty arrays for the gridded data
        grid_shape = (1, len(unique_lats), len(unique_lons), len(unique_levels))
        ak_array = np.full(grid_shape, np.nan)
        pressure_array = np.full(grid_shape, np.nan)

        # Fill the arrays with data
        for _, row in df_date.iterrows():
            lat_idx = unique_lats.index(row.lat)
            lon_idx = unique_lons.index(row.lon)
            level_idx = unique_levels.index(row.level)
            ak_array[0, lat_idx, lon_idx, level_idx] = row.AK
            pressure_array[0, lat_idx, lon_idx, level_idx] = row.pressure

        # Create xarray dataset
        ds = xr.Dataset(
            data_vars={
                "AK": (["time", "lat", "lon", "level"], ak_array),
                "pressure": (["time", "lat", "lon", "level"], pressure_array),
            },
            coords={
                "time": [date],
                "lat": unique_lats,
                "lon": unique_lons,
                "level": unique_levels,
            },
        )

        # Regrid the data to target coordinates
        try:
            regridded_ds = regrid_data(ds, target_lat, target_lon)
            all_datasets.append(regridded_ds)

            # Create verification plot for first few dates
            if i < 3:
                date_str = pd.to_datetime(date).strftime("%Y%m%d")
                plot_comparison(
                    ds, regridded_ds, "AK", time_idx=0, level_idx=0, date_str=date_str
                )

        except Exception as e:
            print(f"Error processing date {date}: {e}")

    # Combine all regridded datasets
    if all_datasets:
        print("Combining all datasets...")
        combined_ds = xr.concat(all_datasets, dim="time")
        print(f"Saving combined dataset to {output_file}")
        combined_ds.to_netcdf(output_file)
        print("Processing completed successfully!")
        return combined_ds
    else:
        print("No datasets were successfully processed.")
        return None


if __name__ == "__main__":
    result_ds = load_and_process_data()

# %%
