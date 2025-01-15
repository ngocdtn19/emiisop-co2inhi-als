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


def prep_hcho_chaser(all_gtool):

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
    ds = ds.rename({list(ds.data_vars.keys())[0]: "tcolhcho"})

    return ds


def cal_mk_map(ds, product_name):
    file_mk_org = os.path.join("./plt_data/mk", f"{product_name}.nc")
    if not os.path.exists(file_mk_org):
        ds = ds.groupby(ds.time.dt.year).mean()
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


def map_corr_by_time(
    ds1, ds2, case_name, mode, start_date="2018-06-01", end_date="2023-12-01"
):
    if len(ds1.lat) != len(ds2.lat):
        ds1 = ds2.interp(lat=ds2.lat, lon=ds2.lon, method="nearest")

    ds1_ss = ds1.sel(time=slice(start_date, end_date))
    ds2_ss = ds2.sel(time=slice(start_date, end_date))
    if mode == "ss":
        ds1_ss = ds1.groupby(ds1.time.dt.month).mean()
        ds2_ss = ds2.groupby(ds2.time.dt.month).mean()
        dim = "month"
    else:
        ds1_ss = ds1.groupby(ds1.time.dt.year).mean()
        ds2_ss = ds2.groupby(ds2.time.dt.year).mean()
        dim = "year"

    c = xr.corr(ds1_ss, ds2_ss, dim=dim)

    c.to_netcdf(f"./plt_data/map_corr/{case_name}_{mode}.nc")
    return c


class HCHO:
    def __init__(self):
        self.inhi_chaser_hcho = (
            prep_hcho_chaser(hcho_chaser_inhi_paths)["tcolhcho"] * 1e-15
        )
        self.no_inhi_chaser_hcho = (
            prep_hcho_chaser(hcho_chaser_noInhi_paths)["tcolhcho"] * 1e-15
        )
        self.tropomi_hcho = prep_hcho_tropomi()["tcolhcho"] * 1e-15

        self.no_inhi_chaser_hcho = self.no_inhi_chaser_hcho.sel(
            time=(self.no_inhi_chaser_hcho.time.dt.year > 2018)
        )

        self.inhi_chaser_hcho = self.inhi_chaser_hcho.sel(
            time=(self.inhi_chaser_hcho.time.dt.year > 2018)
        )
        self.tropomi_hcho = self.tropomi_hcho.sel(
            time=(self.tropomi_hcho.time.dt.year.isin([np.arange(2019, 2024)]))
        )

        self.cal_weights()
        (
            self.inhi_chaser_glob,
            self.inhi_chaser_reg_ann,
            self.inhi_chaser_lat_mean,
            self.inhi_chaser_reg_ss,
        ) = HCHO.cal_glob_reg_hcho(self.inhi_chaser_hcho, self.weights, "chaser")
        (
            self.no_inhi_chaser_glob,
            self.no_inhi_chaser_reg_ann,
            self.no_inhi_chaser_lat_mean,
            self.no_inhi_chaser_reg_ss,
        ) = HCHO.cal_glob_reg_hcho(self.no_inhi_chaser_hcho, self.weights, "chaser")
        (
            self.tropomi_glob,
            self.tropomi_reg_ann,
            self.tropomi_lat_mean,
            self.tropomi_reg_ss,
        ) = HCHO.cal_glob_reg_hcho(self.tropomi_hcho, self.weights, "tropomi")

        # self.chaser_hcho_map_mk = cal_mk_map(self.chaser_hcho, "chaser_hcho")
        # self.tropomi_hcho_map_mk = cal_mk_map(self.tropomi_hcho, "tropomi_hcho")
        self.tropmomi_inhi_chaser_ss_corr = map_corr_by_time(
            self.inhi_chaser_hcho, self.tropomi_hcho, "tropo_chaser", mode="ss"
        )
        self.tropmomi_inhi_chaser_ann_corr = map_corr_by_time(
            self.inhi_chaser_hcho, self.tropomi_hcho, "tropo_chaser", mode="ann"
        )
        self.tropmomi_no_inhi_chaser_ss_corr = map_corr_by_time(
            self.no_inhi_chaser_hcho, self.tropomi_hcho, "tropo_chaser", mode="ss"
        )
        self.tropmomi_no_inhi_chaser_ann_corr = map_corr_by_time(
            self.no_inhi_chaser_hcho, self.tropomi_hcho, "tropo_chaser", mode="ann"
        )

        # self.cal_isop_corr()

    def cal_weights(self):
        ds = self.inhi_chaser_hcho.isel(time=0)
        lat = [d for d in list(ds.dims) if "lat" in d][0]
        self.weights = np.cos(np.deg2rad(ds[lat]))
        self.weights.name = "weights"

    def cal_isop_corr(self):
        list_isop_files = glob(
            "/mnt/dg3/ngoc/nc2gtool/gtool_in/regrided_nc/wCO2inhi/1901-2023/*.nc"
        )
        list_isop_files.append(
            "/mnt/dg3/ngoc/nc2gtool/gtool_in/regrided_nc/woCO2inhi/1900-2023/BFLXC5H8.nc"
        )
        isop_tropo_ann_corr = {}
        isop_tropo_ss_corr = {}
        for f in list_isop_files:
            var = f.split("/")[-1].split("_")[-1].split(".")[0]
            var = var if var != "BFLXC5H8" else "noInhi"
            isop_ds = xr.open_dataset(f)["BFLXC5H8"]
            isop_ds = isop_ds.assign_coords(lon=((isop_ds.lon + 180) % 360) - 180)
            isop_ds = isop_ds.sortby("lon")

            isop_tropo_ann_corr[var] = map_corr_by_time(
                self.tropomi_hcho, isop_ds, "isop_tropo", mode="ann"
            )
            isop_tropo_ss_corr[var] = map_corr_by_time(
                self.tropomi_hcho, isop_ds, "isop_tropo", mode="ss"
            )

        self.isop_tropo_ann_corr = isop_tropo_ann_corr
        self.isop_tropo_ss_corr = isop_tropo_ss_corr

    @staticmethod
    def cal_glob_reg_hcho(ds, weights, product_name):

        list_srex_regs = regionmask.defined_regions.srex.abbrevs
        mask_3D = regionmask.defined_regions.srex.mask_3D(ds.isel(time=0))

        # annual global hcho
        glob_hcho_ann = {"year": np.unique(ds.time.dt.year.values)}
        glob_hcho_ann["avg_glob_ann"] = []

        # annual regional hcho
        reg_hcho_ann = {r: [] for r in list_srex_regs}
        reg_hcho_ann["year"] = np.unique(ds.time.dt.year.values)

        # seasonal regional hcho
        reg_hcho_ss = {r: [] for r in list_srex_regs}
        reg_hcho_ss["month"] = np.arange(1, 13)

        # cal annual hcho for glob and reg
        for y in glob_hcho_ann["year"]:
            ds_y = ds.sel(time=(ds.time.dt.year == y))
            hcho_reg = {r: 0 for r in list_srex_regs}
            hcho_glob = 0

            for m in range(1, 13):
                ds_month = ds_y.sel(time=(ds_y.time.dt.month == m))
                # cal global hcho
                hcho_m_w = ds_month.weighted(weights).mean().item()
                hcho_glob += hcho_m_w / 12
                # cal regional hcho
                for r in list_srex_regs:
                    mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                    hcho_m_w_r = ds_month.weighted(mask_r * weights).mean().item()
                    hcho_reg[r] += hcho_m_w_r / 12

            for r in list_srex_regs:
                reg_hcho_ann[r].append(hcho_reg[r])

            glob_hcho_ann["avg_glob_ann"].append(hcho_glob)

        # cal reg hcho for reg
        for r in list_srex_regs:
            mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
            reg_hcho_ss[r] = (
                ds.weighted(mask_r * weights)
                .mean(["lat", "lon"])
                .groupby(ds.time.dt.month)
                .mean()
                .to_dataframe()["tcolhcho"]
                .values
            )
            assert len(reg_hcho_ss[r]) == 12

        glob_hcho_ann = pd.DataFrame.from_dict(glob_hcho_ann)
        reg_hcho_ann = pd.DataFrame.from_dict(reg_hcho_ann)
        reg_hcho_ss = pd.DataFrame.from_dict(reg_hcho_ss)

        hcho_lat_mean = (
            ds.mean("time")
            .weighted(weights)
            .mean("lon")
            .to_dataframe()
            .reset_index()
            .drop("time", axis=1)
        )

        glob_hcho_ann_path = f"./plt_data/glob_hcho_ann/{product_name}.csv"
        reg_hcho_ann_path = f"./plt_data/reg_hcho_ann/{product_name}.csv"
        hcho_lat_mean_path = f"./plt_data/lat_mean_hcho/{product_name}.csv"
        reg_hcho_ss_path = f"./plt_data/reg_hcho_ss/{product_name}.csv"

        # if not os.path.exists(glob_hcho_ann_path):
        # glob_hcho_ann.to_csv(glob_hcho_ann_path)
        # if not os.path.exists(reg_hcho_ann_path):
        #     reg_hcho_ann.to_csv(reg_hcho_ann_path)
        # if not os.path.exists(hcho_lat_mean_path):
        #     hcho_lat_mean.to_csv(hcho_lat_mean_path)
        # if not os.path.exists(reg_hcho_ss_path):
        #     reg_hcho_ss.to_csv(reg_hcho_ss_path)

        return glob_hcho_ann, reg_hcho_ann, hcho_lat_mean, reg_hcho_ss


# chaser_hcho = prep_hcho_chaser(all_hcho_chaser_paths)
# tropomi_hcho = prep_hcho_tropomi()
# %%
