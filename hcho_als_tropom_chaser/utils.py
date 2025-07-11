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

from datetime import datetime
from mypath import *


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
        ds = ds.groupby(ds.time.dt.year).mean(skipna=True)
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


def map_corr_by_time(ds1, ds2, mode, start_date=None, end_date=None):
    if len(ds1.lat) != len(ds2.lat):
        ds1 = ds2.interp(lat=ds2.lat, lon=ds2.lon, method="nearest")

    if start_date and end_date:
        ds1 = ds1.sel(time=slice(start_date, end_date))
        ds2 = ds2.sel(time=slice(start_date, end_date))

    if mode == "ss":
        ds1_ss = ds1.groupby(ds1.time.dt.month).mean(skipna=True)
        ds2_ss = ds2.groupby(ds2.time.dt.month).mean(skipna=True)
        dim = "month"
    else:
        ds1_ss = ds1.groupby(ds1.time.dt.year).mean(skipna=True)
        ds2_ss = ds2.groupby(ds2.time.dt.year).mean(skipna=True)
        dim = "year"

    c = xr.corr(ds1_ss, ds2_ss, dim=dim)
    return c


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


class HCHO:
    hoque_reg_coords = {
        "REMOTE_PACIFIC": {
            "lat": [32, -28],  # 32°N to 28°S
            "lon": [-177, -117],  # -177° to -117°
        },
        "Indonesia": {
            "lat": [6, -10],  # 6°N to 10°S (descending]
            "lon": [95, 142],  # 95°E to 142°E
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

    def __init__(self, file_path, hcho_var="tcolhcho"):
        # self.ds = xr.open_dataset(file_path).fillna(0)
        if isinstance(file_path, str):
            self.ds = xr.open_dataset(file_path)
        else:
            self.ds = prep_org_hcho_chaser(file_path)
        if hcho_var in list(self.ds.data_vars):
            self.ds = self.ds.rename({hcho_var: "hcho"})
        if "hcho" not in list(self.ds.data_vars):
            self.ds = self.ds.rename({list(self.ds.data_vars.keys())[0]: "hcho"})

        self.ds = self.ds.fillna(0)
        self.hcho = self.ds["hcho"] * 1e-15
        self.hcho = self.hcho.sel(time=(self.hcho.time.dt.year < 2024))

        self.mask_land()

        self.cal_weights()
        (
            self.glob_ann,
            self.reg_ann,
            self.reg_ann_summer,
            self.lat_mean,
            self.reg_ss,
        ) = HCHO.cal_glob_reg_hcho(self.hcho, self.weights)

        # self.chaser_hcho_map_mk = cal_mk_map(self.chaser_hcho, "chaser_hcho")

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

    def cal_weights(self):
        ds = self.ds.isel(time=0)
        lat = [d for d in list(ds.dims) if "lat" in d][0]
        self.weights = np.cos(np.deg2rad(ds[lat]))
        self.weights.name = "weights"

    @staticmethod
    def cal_glob_reg_hcho(ds, weights):

        hoque_regions = list(HCHO.hoque_reg_coords.keys())
        list_srex_regs = regionmask.defined_regions.srex.abbrevs + hoque_regions
        mask_3D = regionmask.defined_regions.srex.mask_3D(ds.isel(time=0))

        # annual global hcho
        glob_hcho_ann = {"year": np.unique(ds.time.dt.year.values)}
        glob_hcho_ann["avg_glob_ann"] = []

        # annual regional hcho
        reg_hcho_ann = {r: [] for r in list_srex_regs}
        reg_hcho_ann["year"] = np.unique(ds.time.dt.year.values)

        reg_hcho_ann_summer = copy.deepcopy(reg_hcho_ann)

        # seasonal regional hcho
        reg_hcho_ss = {r: [] for r in list_srex_regs}
        reg_hcho_ss["month"] = np.arange(1, 13)

        # cal annual hcho for glob and reg
        for y in glob_hcho_ann["year"]:
            ds_y = ds.sel(time=(ds.time.dt.year == y))
            hcho_reg = {r: [] for r in list_srex_regs}
            hcho_glob = []

            for m in range(1, 13):
                ds_month = ds_y.sel(time=(ds_y.time.dt.month == m))
                # cal global hcho
                hcho_m_w = ds_month.weighted(weights).mean(skipna=True).item()
                hcho_glob.append(hcho_m_w)
                # cal regional hcho
                for r in list_srex_regs:
                    if r not in hoque_regions:
                        mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                        hcho_m_w_r = (
                            ds_month.weighted(mask_r * weights).mean(skipna=True).item()
                        )
                    else:
                        lat = HCHO.hoque_reg_coords[r]["lat"]
                        lon = HCHO.hoque_reg_coords[r]["lon"]
                        hcho_m_w_r = (
                            ds_month.sel(
                                lat=slice(lat[0], lat[1]),
                                lon=slice(lon[0], lon[1]),
                            )
                            .mean(skipna=True)
                            .item()
                        )
                    hcho_reg[r].append(hcho_m_w_r)

            for r in list_srex_regs:
                data_r = np.array(hcho_reg[r])
                valid_r = data_r[(data_r != 0) & ~np.isnan(data_r)]
                reg_hcho_ann[r].append(np.mean(valid_r))

                data_summer = data_r[5:8]  # June, July, August
                valid_summer = data_summer[(data_summer != 0) & ~np.isnan(data_summer)]
                reg_hcho_ann_summer[r].append(np.mean(valid_summer))

            data_glob = np.array(hcho_glob)
            valid_glob = data_glob[(data_glob != 0) & ~np.isnan(data_glob)]
            glob_hcho_ann["avg_glob_ann"].append(np.mean(valid_glob))

        # cal reg hcho for reg
        for r in list_srex_regs:
            if r not in hoque_regions:
                mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                reg_hcho_ss[r] = (
                    ds.weighted(mask_r * weights)
                    .mean(["lat", "lon"])
                    .groupby(ds.time.dt.month)
                    .mean(skipna=True)
                    .to_dataframe()["hcho"]
                    .values
                )
            else:
                lat = HCHO.hoque_reg_coords[r]["lat"]
                lon = HCHO.hoque_reg_coords[r]["lon"]
                reg_hcho_ss[r] = (
                    ds.sel(
                        lat=slice(lat[0], lat[1]),
                        lon=slice(lon[0], lon[1]),
                    )
                    .mean(["lat", "lon"])
                    .groupby(ds.time.dt.month)
                    .mean(skipna=True)
                    .to_dataframe()["hcho"]
                    .values
                )
                print(len(reg_hcho_ss[r]), r)
            assert len(reg_hcho_ss[r]) == 12

        print("1", glob_hcho_ann)
        glob_hcho_ann = pd.DataFrame.from_dict(glob_hcho_ann)
        print("2", reg_hcho_ann)
        reg_hcho_ann = pd.DataFrame.from_dict(reg_hcho_ann)
        print("reg_hcho_ann_summer", reg_hcho_ann_summer)
        reg_hcho_ann_summer = pd.DataFrame.from_dict(reg_hcho_ann_summer)
        reg_hcho_ss = pd.DataFrame.from_dict(reg_hcho_ss)

        hcho_lat_mean = (
            ds.mean("time")
            .weighted(weights)
            .mean("lon")
            .to_dataframe()
            .reset_index()
            .drop("time", axis=1)
        )

        # glob_hcho_ann_path = f"./plt_data/glob_hcho_ann/{product_name}.csv"
        # reg_hcho_ann_path = f"./plt_data/reg_hcho_ann/{product_name}.csv"
        # hcho_lat_mean_path = f"./plt_data/lat_mean_hcho/{product_name}.csv"
        # reg_hcho_ss_path = f"./plt_data/reg_hcho_ss/{product_name}.csv"

        # if not os.path.exists(glob_hcho_ann_path):
        # glob_hcho_ann.to_csv(glob_hcho_ann_path)
        # if not os.path.exists(reg_hcho_ann_path):
        #     reg_hcho_ann.to_csv(reg_hcho_ann_path)
        # if not os.path.exists(hcho_lat_mean_path):
        #     hcho_lat_mean.to_csv(hcho_lat_mean_path)
        # if not os.path.exists(reg_hcho_ss_path):
        #     reg_hcho_ss.to_csv(reg_hcho_ss_path)

        return (
            glob_hcho_ann,
            reg_hcho_ann,
            reg_hcho_ann_summer,
            hcho_lat_mean,
            reg_hcho_ss,
        )


class HCHO_hoque(HCHO):
    def __init__(self, ds):
        self.ds = ds
        self.hcho = ds.hcho
        self.cal_weights()
        (
            self.glob_ann,
            self.reg_ann,
            self.lat_mean,
            self.reg_ss,
        ) = HCHO.cal_glob_reg_hcho(self.hcho, self.weights)


class HCHO_maxdoas:
    def __init__(self, file_path, max_doas_coords):
        print(file_path)

        self.max_doas_coords = max_doas_coords
        self.ds = xr.open_dataset(file_path)

        if "hcho" not in list(self.ds.data_vars):
            self.ds = self.ds.rename({list(self.ds.data_vars.keys())[0]: "hcho"})

        # self.ds = self.ds.sel(layer=(self.ds.layer.isin(np.arange(5))))
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
# %%
