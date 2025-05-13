# %%

"""
CH2O Text Files to xarray Dataset Conversion Script
---------------------------------------------------
This script reads 12 monthly CH2O text files with averaging kernels applied,
combines them into a single xarray Dataset, and saves the result as a NetCDF file.

The script handles data with the following structure:
- Date, Hour, Latitude, Longitude, Partial_col, Concen, Level

Created: April 2025
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import glob
from tqdm import tqdm  # For progress bars
import multiprocessing as mp

# Define paths
data_dir = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/ngoc_sim_AKapplied"
output_file = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/ch2o_ngoc_sim_AKapplied_combined_2019.nc"

# List of months to process
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


def read_ch2o_file(file_path):
    """
    Read a CH2O file into a pandas DataFrame - optimized version based on actual data format
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        # Read the data with optimized settings and fixed-width format
        # Based on the sample data pattern
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,  # Handle variable spacing
            skiprows=1,  # Skip header row
            names=[
                "Date",
                "Hour",
                "Latitude",
                "Longitude",
                "Partial_col",
                "Concen",
                "Level",
            ],
            dtype={  # Pre-specify dtypes for faster loading
                "Date": str,
                "Hour": int,  # Hour appears to be an integer
                "Latitude": float,
                "Longitude": float,
                "Partial_col": float,
                "Concen": float,
                "Level": float,
            },
            engine="c",  # Use the faster C engine
        )

        # Format Hour to ensure it's a string with leading zeros if needed
        df["Hour"] = df["Hour"].astype(str).str.zfill(2)

        # Create datetime column with direct parsing to datetime64[ns]
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Hour"] + ":00:00", format="%Y/%m/%d %H:%M:%S"
        )

        # Add day of year for potential seasonal analysis
        df["doy"] = df["datetime"].dt.dayofyear

        # Explicitly convert Level to category - making sure it works
        df["Level"] = df["Level"].astype("category")

        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_month(month):
    """Process a single month file"""
    file_path = os.path.join(data_dir, f"AK_applied_ch2o_{month}_2019_new.txt")
    print(f"Processing {month} data...")
    df = read_ch2o_file(file_path)
    if df is not None and not df.empty:
        df["month"] = month  # Add month info
        return df
    return None


def main():
    print("Starting CH2O data conversion...")
    start_time = datetime.now()

    # Use multiprocessing for faster file reading
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} CPU cores for parallel processing")

    # Process files in parallel
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_month, months)
        all_dataframes = [df for df in results if df is not None]

    if not all_dataframes:
        print("No valid data found in any of the monthly files.")
        return

    # Combine all dataframes
    print(f"Successfully processed {len(all_dataframes)} month(s) of data")
    print("Merging all monthly data...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")

    # Display some basic statistics
    print("\nData summary:")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"Number of unique timestamps: {combined_df['datetime'].nunique()}")
    print(f"Number of unique latitudes: {combined_df['Latitude'].nunique()}")
    print(f"Number of unique longitudes: {combined_df['Longitude'].nunique()}")
    print(f"Number of unique levels: {combined_df['Level'].nunique()}")

    # Free memory by removing the individual dataframes
    del all_dataframes

    # Get unique values for each dimension
    print("Creating coordinate arrays...")

    # Convert datetime to string, then back to datetime to ensure consistency
    # This is a simpler fix than trying to floor timestamps
    combined_df["datetime_str"] = combined_df["datetime"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    combined_df["datetime"] = pd.to_datetime(combined_df["datetime_str"])
    combined_df.drop("datetime_str", axis=1, inplace=True)

    unique_times = pd.DatetimeIndex(
        np.sort(combined_df["datetime"].unique())
    )  # Ensure DatetimeIndex
    unique_lats = np.sort(combined_df["Latitude"].unique())
    unique_lons = np.sort(combined_df["Longitude"].unique())

    # FIX: Handle Level as a regular float column instead of a categorical
    # Check if Level is actually categorical, if not, handle it as a regular column
    if pd.api.types.is_categorical_dtype(combined_df["Level"].dtype):
        unique_levels = np.sort(combined_df["Level"].cat.categories.to_numpy())
        level_is_categorical = True
    else:
        print(
            "Warning: Level column is not categorical. Processing as a regular float column."
        )
        unique_levels = np.sort(combined_df["Level"].unique())
        level_is_categorical = False

    print(
        f"Dataset dimensions: time={len(unique_times)}, lat={len(unique_lats)}, "
        f"lon={len(unique_lons)}, level={len(unique_levels)}"
    )

    # Create lookup dictionaries for faster indexing
    # Use dictionary comprehension to create the mappings
    # The key here is to convert datetime64 to timestamp strings to avoid precision issues
    time_to_idx = {}
    for i, t in enumerate(unique_times):
        # Convert datetime to standard string format to ensure consistent keys
        time_str = pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        time_to_idx[time_str] = i

    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    level_to_idx = {level: i for i, level in enumerate(unique_levels)}

    # Initialize data arrays with NaN
    print("Creating data arrays...")
    ch2o = np.full(
        (len(unique_times), len(unique_levels), len(unique_lats), len(unique_lons)),
        np.nan,
        dtype=np.float32,
    )  # Use float32 to save memory
    partial_col = np.full(
        (len(unique_times), len(unique_levels), len(unique_lats), len(unique_lons)),
        np.nan,
        dtype=np.float32,
    )

    # Vectorized approach to filling the arrays (much faster than row-by-row)
    print("Filling data arrays (optimized approach)...")

    # Extract arrays from the DataFrame for faster access
    # Convert datetime values to strings to match the dictionary keys
    df_times = combined_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").values
    df_lats = combined_df["Latitude"].values
    df_lons = combined_df["Longitude"].values
    df_concen = combined_df["Concen"].values
    df_partial = combined_df["Partial_col"].values

    # FIX: Extract levels based on whether they are categorical or not
    if level_is_categorical:
        df_levels = combined_df["Level"].cat.codes.values  # Use category codes directly
    else:
        # Map level values to their indices in unique_levels
        df_level_values = combined_df["Level"].values
        df_levels = np.zeros(len(df_level_values), dtype=np.int32)
        for i, level in enumerate(df_level_values):
            df_levels[i] = level_to_idx[level]

    # Process in batches to avoid memory issues
    batch_size = 2500000  # Adjust based on your available RAM
    num_batches = (len(combined_df) + batch_size - 1) // batch_size

    # Track skipped entries
    skipped_entries = 0

    for batch in tqdm(range(num_batches), desc="Processing data batches"):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(combined_df))

        # Get indices for this batch
        batch_indices = slice(start_idx, end_idx)
        batch_size_actual = end_idx - start_idx

        # Pre-allocate arrays for indices
        t_indices = np.zeros(batch_size_actual, dtype=np.int32)
        lat_indices = np.zeros(batch_size_actual, dtype=np.int32)
        lon_indices = np.zeros(batch_size_actual, dtype=np.int32)
        valid_indices = np.ones(batch_size_actual, dtype=bool)  # Track valid entries

        # Fill index arrays with robust error handling
        for i in range(batch_size_actual):
            idx = i + start_idx

            # Handle time indices with explicit try/except to avoid KeyError
            try:
                t_indices[i] = time_to_idx[df_times[idx]]
            except KeyError:
                # If key not found, mark as invalid and skip
                valid_indices[i] = False
                skipped_entries += 1
                continue

            # Handle lat indices with explicit try/except
            try:
                lat_indices[i] = lat_to_idx[df_lats[idx]]
            except KeyError:
                valid_indices[i] = False
                skipped_entries += 1
                continue

            # Handle lon indices with explicit try/except
            try:
                lon_indices[i] = lon_to_idx[df_lons[idx]]
            except KeyError:
                valid_indices[i] = False
                skipped_entries += 1
                continue

        # Get level indices for this batch
        level_indices = df_levels[batch_indices]

        # Fill the arrays for this batch, but only for valid indices
        for i in range(batch_size_actual):
            if not valid_indices[i]:
                continue

            idx = i + start_idx
            ch2o[t_indices[i], level_indices[i], lat_indices[i], lon_indices[i]] = (
                df_concen[idx]
            )
            partial_col[
                t_indices[i], level_indices[i], lat_indices[i], lon_indices[i]
            ] = df_partial[idx]

    if skipped_entries > 0:
        print(
            f"Warning: Skipped {skipped_entries} entries due to coordinate mismatches"
        )

    # Free memory
    del combined_df

    # Create the xarray dataset
    print("Creating xarray dataset...")
    ds = xr.Dataset(
        data_vars={
            "ch2o": (
                ("time", "level", "latitude", "longitude"),
                ch2o,
                {
                    "long_name": "Formaldehyde Concentration",
                    "units": "ppb",
                    "description": "HCHO concentration with averaging kernels applied",
                },
            ),
            "partial_column": (
                ("time", "level", "latitude", "longitude"),
                partial_col,
                {
                    "long_name": "Partial Column Formaldehyde",
                    "description": "Partial column HCHO with averaging kernels applied",
                    "units": "molecules/cm^2",
                },
            ),
        },
        coords={
            "time": unique_times,
            "level": unique_levels,
            "latitude": unique_lats,
            "longitude": unique_lons,
        },
        attrs={
            "description": "CHASER CH2O data with averaging kernels applied for 2019",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Converted from monthly text files",
        },
    )

    # Add dimension coordinates attributes
    ds.latitude.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.longitude.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds.level.attrs = {"standard_name": "model_level_number"}
    ds.time.attrs = {"standard_name": "time"}

    # Ensure time is datetime64
    ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))

    # Save to netCDF file - using compression for smaller file size
    print(f"Saving dataset to {output_file}...")
    comp = dict(zlib=True, complevel=5)  # Compression settings
    encoding = {var: comp for var in ds.data_vars}

    # Add special encoding for time to ensure proper datetime64 format
    encoding["time"] = {"units": "days since 2019-01-01", "calendar": "standard"}

    ds.to_netcdf(output_file, format="NETCDF4", encoding=encoding)

    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    print(f"Conversion complete! Total time: {elapsed_time}")

    # Display summary of the created dataset
    print("\nDataset Summary:")
    print(ds)


if __name__ == "__main__":
    main()

# %%
