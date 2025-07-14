# %%
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import pygtool_core
import pygtool
import xarray as xr
import numpy as np
import regionmask
import pandas as pd
import mk
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import pearsonr
from datetime import datetime
from mypath import *
from pathlib import Path


geogrid = pygtool.readgrid()
chaser_lon, chaser_lat = geogrid.getlonlat()


def load_sigma():
    file = "/home/ngoc/nc2gtool/pygtool3/pygtool3/GTAXDIR/GTAXLOC.HETA36"
    data = open(file, "br")
    summ = str(int(36))
    dt = np.dtype(
        [
            ("f_header", ">i"),
            ("header", ">64S16"),
            ("1f_tail", "i"),
            ("2f_header", ">i"),
            ("arr", ">" + summ + "f"),
            ("2f_tail", ">i"),
        ]
    )
    heta = np.fromfile(data, dtype=dt)
    sigma = heta[0][4]
    return sigma


def prep_chaser_for_ak(ds):
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    ds = ds.sortby("lon")

    if "sigma" in list(ds.dims):
        v = list(ds.data_vars)[0]
        ds[v] = ds[v].transpose("time", "lat", "lon", "sigma")
        ds = ds.rename({"sigma": "layer"})
    return ds


def prep_org_hcho_chaser(all_gtool):

    list_ds = []
    for f in all_gtool:
        year = int(f.split("/")[-2])
        print(year)
        list_time = [datetime(year, month, 15) for month in range(1, 13)]

        hcho_chaser_gtool = pygtool_core.Gtool2d(f, count=len(list_time))
        hcho_chaser_gtool.set_datetimeindex(list_time)
        hcho_chaser_ds = hcho_chaser_gtool.to_xarray(
            lat=chaser_lat, lon=chaser_lon, na_values=np.nan
        )
        list_ds.append(hcho_chaser_ds)
    ts_ds = xr.concat(list_ds, dim="time")
    ts_ds = ts_ds.assign_coords(lon=((ts_ds.lon + 180) % 360) - 180)
    ts_ds = ts_ds.sortby("lon")
    ts_ds = ts_ds.rename({list(ts_ds.data_vars.keys())[0]: "tcolhcho"})
    return ts_ds


def prep_hcho_tropomi(chaser_lat=chaser_lat, chaser_lon=chaser_lon):
    ds = xr.open_dataset(hcho_tropomi_path)
    # filter qa value
    ds = ds.where(ds["qa_L3"] == 1)
    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = ds.interp(lat=chaser_lat, lon=sorted(((chaser_lon + 180) % 360) - 180))
    ds = ds.rename({list(ds.data_vars.keys())[0]: "hcho"})

    return ds


def mask_chaser_by_sat_hcho(chaser_ds, sat_ds, var_name="tcolhcho"):

    sat_ds = sat_ds.sel(time=(sat_ds.time.isin(chaser_ds.time.values)))

    if var_name not in sat_ds:
        raise ValueError(f"Variable '{var_name}' not found in sat_ds.")

    # Create mask: 1 where sat_ds[var_name] is not NaN, NaN where it is NaN
    mask = xr.where(~sat_ds[var_name].isnull(), 1.0, np.nan)

    assert (
        chaser_ds.shape == mask.shape
    ), f"Shape mismatch: chaser_ds.hcho {chaser_ds.shape} and mask {mask.shape}"

    masked_chaser = chaser_ds * mask

    return masked_chaser


def cal_mk_map(ds, product_name):
    file_mk_org = os.path.join("./plt_data/mk", f"{product_name}.nc")
    if not os.path.exists(file_mk_org):

        y = xr.DataArray(
            np.arange(len(ds["year"])) + 1,
            dims="year",
            coords={"year": ds["year"]},
        )
        slope = xr.Dataset({})
        slope["tcolhcho_ann_trend"] = mk.kendall_correlation(ds, y, "year")
        slope.to_netcdf(file_mk_org)
    else:
        slope = xr.open_dataset(file_mk_org)
    return slope


def compute_rmse(ds1, ds2):

    squared_diff = (ds1 - ds2) ** 2
    mean_squared_error = squared_diff.mean(dim="year", skipna=True)
    rmse = np.sqrt(mean_squared_error)

    mean_ref = ds2.mean(dim="year", skipna=True)
    rmse_percent = (rmse / mean_ref) * 100
    return rmse_percent


def map_corr_by_time(model_ds, sat_ds):
    def corr_and_p(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            return np.nan, np.nan
        r, p = pearsonr(x[mask], y[mask])
        return r, p

    sat_ds = sat_ds.sel(year=(sat_ds.year.isin(model_ds.year.values)))
    corr, pval = xr.apply_ufunc(
        corr_and_p,
        model_ds,
        sat_ds,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )

    significant = pval < 0.05

    return corr, significant


def create_visit_mask():
    visit_mask = "/mnt/dg3/ngoc/cmip6_bvoc_als/data/original/axl/mask/mask_fx_VISIT-S3(G1997)_historical_r1i1p1f1_gn.nc"
    mask = xr.open_dataset(visit_mask)
    mask = mask.interp(
        lat=chaser_lat, lon=sorted(((chaser_lon + 180) % 360) - 180), method="nearest"
    )
    mask.to_netcdf(
        "./visit_land_mask/mask.nc",
    )
    return mask


def rmv_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed file: {path}")
    else:
        print(f"File not found: {path}")


def min_max_normalize(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def check_ak_by_plt(model_ak, sat_ak, model_pressure, sat_pressure):
    time_steps = model_ak.shape[0]
    n_model_layer = model_ak.shape[-1]
    n_sat_layer = sat_pressure.shape[-1]

    # Compute mean over lat/lon
    data_list = [
        (
            np.nanmean(model_ak, axis=(1, 2)),
            n_model_layer,
            "AK Value",
            "Model AK",
        ),
        (
            np.nanmean(sat_ak, axis=(1, 2)),
            n_sat_layer,
            "AK Value",
            "Sattellite AK",
        ),
        (
            np.nanmean(model_pressure, axis=(1, 2)),
            n_model_layer,
            "Pressure",
            "Model Pressure",
        ),
        (
            np.nanmean(sat_pressure, axis=(1, 2)),
            n_sat_layer,
            "Pressure",
            "Satellite Pressure",
        ),
    ]

    colors = cm.viridis(np.linspace(0, 1, time_steps))

    # Create subplots
    fig, axs = plt.subplots(
        1, 4, figsize=(24, 6), sharey=False, constrained_layout=True
    )

    # Loop over subplots
    for i, (data, n_layer, xlabel, title) in enumerate(data_list):
        for t in range(time_steps):
            axs[i].plot(data[t], np.arange(n_layer), color=colors[t])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel("Layer Index")
        axs[i].set_title(title)
        axs[i].grid(True)
        axs[i].set_ylim(0, 40)  # Reverse y-axis for layers

    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(0, time_steps - 1))
    fig.colorbar(sm, ax=axs[-1], label="Time Index")

    plt.show()


def get_sat_file(sat_name, sat_ver="v2"):
    sat_dir = f"/mnt/dg3/ngoc/obs_data"

    if sat_ver == "v1":
        time_omi = "20050101-20231201"
        time_tropo = "20180601-20240701"
    else:
        time_omi = "20050101-20221231"
        time_tropo = "20180507-20231231"

    m_name_omi = f"mon_BIRA_OMI_HCHO_L3_{sat_ver}"
    m_name_tropo = f"mon_TROPOMI_HCHO_L3_{sat_ver}"

    time = time_omi
    m_name = m_name_omi
    if sat_name == "tropo":
        time = time_tropo
        m_name = m_name_tropo

    return f"{sat_dir}/{m_name}/EXTRACT/hcho_AERmon_{m_name}_historical_gn_{time}.nc"


def cal_ss_by_lat(ds):
    ds = ds[["hcho"]]

    min_y = ds["time"].dt.year.min().item()
    max_y = ds["time"].dt.year.max().item()
    if len(ds.sel(time=ds["time"].dt.year == min_y)) < 12:
        # If the first year has less than 12 months, we assume it is incomplete
        min_y += 1

    valid_years = np.arange(min_y, max_y + 1)
    print(f"Valid years for ss mean by lat: {valid_years}")
    # Create masks
    lat = ds["lat"]
    mask_north = lat > 0
    mask_tropics = (lat <= 0) & (lat >= -30)
    mask_south = lat < -30

    # Define month ranges
    months_north = [6, 7, 8]  # JJA
    months_tropics = [2, 3, 4]  # FMA
    months_south = [12, 1, 2]  # DJF

    # North: JJA
    ds_north = ds.sel(time=ds["time"].dt.month.isin(months_north))
    north_mean = (
        ds_north.where(mask_north, drop=True).groupby("time.year").mean(dim="time")
    )
    north_mean = north_mean.sel(year=north_mean.year.isin(valid_years))

    # Tropics: FMA
    ds_tropics = ds.sel(time=ds["time"].dt.month.isin(months_tropics))
    tropics_mean = (
        ds_tropics.where(mask_tropics, drop=True).groupby("time.year").mean(dim="time")
    )
    tropics_mean = tropics_mean.sel(year=tropics_mean.year.isin(valid_years))

    # South: DJF (requires shifting across years)
    ds_south = ds.sel(time=ds["time"].dt.month.isin(months_south))
    year_adj = ds_south["time"].dt.year.where(
        ds_south["time"].dt.month != 12, ds_south["time"].dt.year + 1
    )
    ds_south = ds_south.assign_coords(season_year=year_adj)
    south_mean = (
        ds_south.where(mask_south, drop=True).groupby("season_year").mean(dim="time")
    )
    south_mean = south_mean.rename({"season_year": "year"})
    south_mean = south_mean.sel(year=south_mean.year.isin(valid_years))

    # Combine the three means using where mask
    combined = xr.combine_by_coords([north_mean, tropics_mean, south_mean])

    # Final mean with lat-dependent seasonality
    return combined


class HCHO:
    regs = {
        # "REMOTE_PACIFIC": {
        #     "lat": [32, -28],  # 32°N to 28°S
        #     "lon": [-177, -117],  # -177° to -117°
        # },
        # Muller et al. (2024) regions (Amazonia, S-E US)
        # Opacka et al. (2021) regions (Indonesia, Mato Grosso, South China)
        # Hoque et al. (2024) regions (C_Africa, N_Africa, S_Africa)
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
            "lat": [6, -6],  # 6°N to 6°S (descending]
            "lon": [95, 112],  # 95°E to 112°E
        },
        "South China": {
            "lat": [28, 22],  # 28°N to 22°N
            "lon": [100, 112],  # 100°E to 112°E
        },
        "C_Africa": {
            "lat": [6, -6],  # 6°N to 6°S
            "lon": [10, 35],  # 10°E to 35°E
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

    def __init__(self, file_path, hcho_var="tcolhcho", layer_used=14, sat_filter=None):

        if isinstance(file_path, str):
            self.ds = xr.open_dataset(file_path)
        else:
            self.ds = prep_org_hcho_chaser(file_path)

        if hcho_var in list(self.ds.data_vars):
            self.ds = self.ds.rename({hcho_var: "hcho"})
        if "hcho" not in list(self.ds.data_vars):
            self.ds = self.ds.rename({list(self.ds.data_vars.keys())[0]: "hcho"})

        if sat_filter is not None:
            # Calculate the total column HCHO
            self.ds = self.ds.sel(layer=(self.ds.layer.isin(np.arange(layer_used))))
            print("No. of Layers:", self.ds.layer.shape)
            self.ds = self.ds.sum("layer")

            # filtering by satellite valid pixels
            print("Filtering by satellite valid pixels")
            sat_ds = xr.open_dataset(get_sat_file(sat_filter))
            self.ds["hcho"] = mask_chaser_by_sat_hcho(self.ds["hcho"], sat_ds)

        self.ds = self.ds.fillna(0)
        self.hcho = self.ds["hcho"] * 1e-15
        self.hcho = self.hcho.sel(time=(self.hcho.time.dt.year < 2024))

        self.ds_ss_by_lat = cal_ss_by_lat(self.ds)
        self.hcho_sslat = self.ds_ss_by_lat.hcho * 1e-15

        self.mask_land()
        self.cal_weights()

        (_, self.reg_ann, self.reg_ss, _) = HCHO.ann_ss_reg(self.hcho, self.weights)
        self.reg_ann_sslat = HCHO.ann_ss_reg(self.hcho_sslat, self.weights, sslat=True)

        # MK trend
        c = Path(file_path).parents[1].name
        if sat_filter is not None:
            c = f"{c}_{sat_filter}"
        print(f"Calculating MK trend for {c}...")
        self.hcho_ann_mk = cal_mk_map(
            self.hcho.groupby(self.hcho.time.dt.year).mean("time", skipna=True), c
        )
        self.hcho_ann_sslat_mk = cal_mk_map(self.hcho_sslat, f"{c}_sslat")

        # self.tropmomi_inhi_chaser_ss_corr = map_corr_by_time(
        #     self.inhi_chaser_hcho, self.tropomi_hcho, "tropo_chaser", mode="ss"
        # )
        # self.tropmomi_inhi_chaser_ann_corr = map_corr_by_time(
        #     self.inhi_chaser_hcho, self.tropomi_hcho, "tropo_chaser", mode="ann"
        # )

    def mask_land(self):
        visit_mask = xr.open_dataset("./visit_land_mask/mask.nc")
        # mask = xr.where(~visit_mask["mask"].isnull(), 1.0, np.nan)
        mask = visit_mask.mask.values
        self.hcho = self.hcho * mask
        self.hcho_sslat = self.hcho_sslat * mask

    def cal_weights(self):
        ds = self.ds.isel(time=0)
        lat = [d for d in list(ds.dims) if "lat" in d][0]
        self.weights = np.cos(np.deg2rad(ds[lat]))
        self.weights.name = "weights"

    @staticmethod
    def ann_ss_reg(ds, ws, sslat=False):

        hoque_regions = list(HCHO.regs.keys())
        list_regs = regionmask.defined_regions.srex.abbrevs + hoque_regions
        lls = ["lat", "lon"]

        if sslat:
            mask_3D = regionmask.defined_regions.srex.mask_3D(ds.isel(year=0))
            reg_ann = {**{r: [] for r in list_regs}, "year": ds.year.values}

            for r in list_regs:
                if r not in hoque_regions:
                    mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                    region_ds = ds.weighted(mask_r * ws).mean(lls)
                else:
                    lat, lon = HCHO.regs[r]["lat"], HCHO.regs[r]["lon"]
                    region_ds = ds.sel(lat=slice(*lat), lon=slice(*lon)).mean(lls)
                reg_ann[r] = (
                    region_ds.groupby(region_ds.year)
                    .mean(..., skipna=True)
                    .to_dataframe()["hcho"]
                    .values
                )
            return pd.DataFrame.from_dict(reg_ann)
        else:
            mask_3D = regionmask.defined_regions.srex.mask_3D(ds.isel(time=0))
            years = np.unique(ds.time.dt.year.values)

            glob_hcho_ann = {"year": years, "avg_glob_ann": []}
            reg_hcho_ann = {**{r: [] for r in list_regs}, "year": years}
            reg_hcho_ss = {**{r: [] for r in list_regs}, "month": np.arange(1, 13)}

            # cal annual hcho for glob and reg
            for y in glob_hcho_ann["year"]:
                ds_y = ds.sel(time=(ds.time.dt.year == y))
                hcho_reg = {r: [] for r in list_regs}
                hcho_glob = []

                for m in range(1, 13):
                    ds_month = ds_y.sel(time=(ds_y.time.dt.month == m))
                    # cal global hcho
                    hcho_m_w = ds_month.weighted(ws).mean(skipna=True).item()
                    hcho_glob.append(hcho_m_w)
                    # cal regional hcho
                    for r in list_regs:
                        if r not in hoque_regions:
                            mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                            hcho_m_w_r = (
                                ds_month.weighted(mask_r * ws).mean(skipna=True).item()
                            )
                        else:
                            lat, lon = (
                                HCHO.regs[r]["lat"],
                                HCHO.regs[r]["lon"],
                            )
                            hcho_m_w_r = (
                                ds_month.sel(lat=slice(*lat), lon=slice(*lon))
                                .mean(skipna=True)
                                .item()
                            )
                        hcho_reg[r].append(hcho_m_w_r)

                for r in list_regs:
                    data_r = np.array(hcho_reg[r])
                    valid_r = data_r[(data_r != 0) & ~np.isnan(data_r)]
                    reg_hcho_ann[r].append(np.mean(valid_r))

                data_glob = np.array(hcho_glob)
                valid_glob = data_glob[(data_glob != 0) & ~np.isnan(data_glob)]
                glob_hcho_ann["avg_glob_ann"].append(np.mean(valid_glob))

            # cal ss hcho for reg
            for r in list_regs:
                if r not in hoque_regions:
                    mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                    region_ds = ds.weighted(mask_r * ws).mean(lls)
                else:
                    lat, lon = HCHO.regs[r]["lat"], HCHO.regs[r]["lon"]
                    region_ds = ds.sel(lat=slice(*lat), lon=slice(*lon)).mean(lls)
                reg_hcho_ss[r] = (
                    region_ds.groupby(ds.time.dt.month)
                    .mean(..., skipna=True)
                    .to_dataframe()["hcho"]
                    .values
                )

            glob_hcho_ann = pd.DataFrame.from_dict(glob_hcho_ann)
            reg_hcho_ann = pd.DataFrame.from_dict(reg_hcho_ann)
            reg_hcho_ss = pd.DataFrame.from_dict(reg_hcho_ss)

            hcho_lat_mean = (
                ds.mean("time")
                .weighted(ws)
                .mean("lon")
                .to_dataframe()
                .reset_index()
                .drop("time", axis=1)
            )

            return (glob_hcho_ann, reg_hcho_ann, reg_hcho_ss, hcho_lat_mean)


class HCHO_maxdoas:
    def __init__(self, file_path, max_doas_coords, layer_used=15):
        print(file_path)

        self.max_doas_coords = max_doas_coords
        self.ds = xr.open_dataset(file_path)

        if "hcho" not in list(self.ds.data_vars):
            self.ds = self.ds.rename({list(self.ds.data_vars.keys())[0]: "hcho"})

        self.ds = self.ds.sel(layer=(self.ds.layer.isin(np.arange(layer_used))))
        print("No. of Layers:", self.ds.layer.shape)
        self.ds = self.ds.sum("layer")

        self.hcho_at_maxdoas = {}
        self.extract()

    def extract(self):
        for station in self.max_doas_coords.keys():
            lat = self.max_doas_coords[station]["lat"]
            lon = self.max_doas_coords[station]["lon"]

            # self.hcho_at_maxdoas[station] = self.ds.sel(
            #     lat=lat, lon=lon, method="nearest"
            # ).to_dataframe()
            self.hcho_at_maxdoas[station] = (
                self.ds.sel(
                    lat=slice(lat + 2.8, lat - 2.8), lon=slice(lon - 2.8, lon + 2.8)
                )
                .to_dataframe()
                .groupby("time")
                .mean()
            )


class HCHO_hoque(HCHO):
    def __init__(self, ds):
        self.ds = ds
        self.hcho = ds.hcho
        self.cal_weights()
        (_, self.reg_ann, self.reg_ss, _) = HCHO.ann_ss_reg(self.hcho, self.weights)


def notused_load_hoque_aked_nc():
    def process_nc(nc_path):
        ds = xr.open_dataset(nc_path)
        hcho = (
            ds["partial_column"]
            .sum("level")
            .resample(time="1M")
            .mean()
            .to_dataset(name="hcho")
        )
        hcho = hcho.sortby("latitude", ascending=False)
        hcho["hcho"] = hcho["hcho"] * 1e-15
        return hcho.rename({"latitude": "lat", "longitude": "lon"})

    base_dir = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test"
    hoque = f"{base_dir}/ch2o_hoque_sim_AKapplied_combined_2019.nc"
    ngoc = f"{base_dir}/ch2o_ngoc_sim_AKapplied_combined_2019.nc"

    prep_hoque = HCHO_hoque(process_nc(hoque))
    prep_ngoc = HCHO_hoque(process_nc(ngoc))
    return prep_hoque, prep_ngoc


# chaser_hcho = prep_hcho_chaser(all_hcho_chaser_paths)
# tropomi_hcho = prep_hcho_tropomi()
