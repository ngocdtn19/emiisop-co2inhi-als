# %%
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import os
import pygtool_core
import pygtool
import xarray as xr
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from utils import *
from plt import *

geogrid = pygtool.readgrid()
Clon, Clat = geogrid.getlonlat()

BASE_DIR = "/mnt/dg3/ngoc/CHASER_output"
CASES = [
    # "VISIT20172023_no_nudg",
    "VISITst20012023_nudg",
    # "hoque_sim",
    # "XhalfVISITst20012023",
    "UKpft20012023_nudg",
    "MEGANst20012023_nudg",
    "MEGANpft20012023_nudg",
    "UKst20012023_nudg",
    "MIXpft20012023_nudg",
]

SIGMA = load_sigma()


def sampling_12to14h_hoque():
    def process_ps(ds):
        ps = ds["PS"].values
        ps_layers = [ps * SIGMA[l] for l in range(len(SIGMA))]
        ps_layers = np.stack(ps_layers, axis=-1)
        return xr.Dataset(
            {
                "ps": (
                    ("time", "lat", "lon", "layer"),
                    ps_layers,
                )
            },
            coords={
                "time": ds.time.values,
                "lat": ds.lat.values,
                "lon": ds.lon.values,
                "layer": np.arange(len(SIGMA)),
            },
        )

    vars = ["ch2o", "t", "ps"]
    years = np.arange(2019, 2021)
    # years = np.arange(2001, 2024)
    sigma = np.arange(1, 37)

    case_dir = "/mnt/dg2/hoque/TROPOMI_model_analysis"
    for var in vars:

        out_dir = f"{BASE_DIR}/sample_12to14h/hoque_sim"
        out_file = f"{out_dir}/{var}.nc"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(out_file):

            list_ds = []
            for y in years:
                if var == "ch2o":
                    chaser_var_path = (
                        f"{case_dir}/New_emission_chaser_2019_2020/{y}/{var}"
                    )
                else:
                    chaser_var_path = f"{case_dir}/{y}/{var}"
                time = pd.date_range(
                    f"{y}-01-01 02:00", f"{y+1}-01-01 00:00", freq="2H"
                )
                if var != "ps":
                    gtool = pygtool_core.Gtool3d(chaser_var_path, count=len(time))
                    gtool.set_datetimeindex(time)
                    var_ds = gtool.to_xarray(
                        lat=Clat, lon=Clon, sigma=sigma, na_values=np.nan
                    )
                else:
                    gtool = pygtool_core.Gtool2d(chaser_var_path, count=len(time))
                    gtool.set_datetimeindex(time)
                    var_ds = gtool.to_xarray(lat=Clat, lon=Clon, na_values=np.nan)
                sampled_ds = var_ds.sel(time=(var_ds.time.dt.hour.isin([12, 14])))
                sampled_ds = sampled_ds.groupby(sampled_ds.time.dt.date).mean()
                sampled_ds = sampled_ds.rename({"date": "time"})
                sampled_ds = process_ps(sampled_ds) if var == "ps" else sampled_ds

                list_ds.append(sampled_ds)

            concat_ds = xr.concat(list_ds, dim="time")
            concat_ds["time"] = np.array(concat_ds.time.values, dtype="datetime64[D]")

            concat_ds.to_netcdf(out_file)
        print(var)


def hcho_adj_by_ps(t_ds, ps_ds, ch2o_ds):

    def adj(p, p_next, t, t_next, hcho):
        def p2h(p, t):
            return (-1) * ((np.log(p * 100 / 101325)) * 8.314 * t) / (0.028 * 9.8)

        H = abs(p2h(p, t) - p2h(p_next, t_next))
        return (hcho * 1e-9 * H * p * 100 * 1e-4) / (1.38e-23 * t)

    c_temp = t_ds["T"].values  # (time, lat, lon, layer)
    c_ch2o = ch2o_ds["CH2O"].values  # (time, lat, lon, layer)
    c_ps = ps_ds["PS"].values  # (time, lat, lon, layer)

    c_time = t_ds.time.values
    hcho_layers = []
    for l in range(34):
        p = c_ps[:, :, :, l]
        p_next = c_ps[:, :, :, l + 1]

        t = c_temp[:, :, :, l]
        t_next = c_temp[:, :, :, l + 1]
        hcho = c_ch2o[:, :, :, l]

        ak_hcho = adj(p, p_next, t, t_next, hcho)

        hcho_layers.append(ak_hcho)

    assert len(hcho_layers) == 34

    hcho_ds = xr.Dataset(
        {
            "hcho": (
                ("layer", "time", "lat", "lon"),
                np.stack(hcho_layers, axis=0),
            )
        },
        coords={
            "layer": np.arange(len(hcho_layers)),
            "time": c_time,
            "lat": t_ds.lat.values,
            "lon": t_ds.lon.values,
        },
    )

    return hcho_ds["hcho"].sum("layer", skipna=True)


def chaser_tcol_cal():

    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for _case in CASES:
        print(_case)
        case_dir = f"{BASE_DIR}/sample_12to14h/{_case}"

        t_ds = xr.open_dataset(f"{case_dir}/t.nc").resample(time="M").mean()
        ch2o_ds = xr.open_dataset(f"{case_dir}/ch2o.nc").resample(time="M").mean()
        ps_ds = xr.open_dataset(f"{case_dir}/ps.nc").resample(time="M").mean()

        t_ds = prep_chaser_for_ak(t_ds)
        ch2o_ds = prep_chaser_for_ak(ch2o_ds)
        ps_ds = prep_chaser_for_ak(ps_ds)

        mtype = "datetime64[M]"
        nstype = "datetime64[ns]"

        t_ds["time"] = t_ds.time.values.astype(mtype).astype(nstype)
        ch2o_ds["time"] = ch2o_ds.time.values.astype(mtype).astype(nstype)
        ps_ds["time"] = ps_ds.time.values.astype(mtype).astype(nstype)

        ps_ds = ps_ds.rename({list(ps_ds.data_vars)[0]: "PS"})

        adj_hcho = hcho_adj_by_ps(t_ds, ps_ds, ch2o_ds)
        adj_hcho.to_netcdf(f"{out_dir}/{_case}.nc")


def load_data():
    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    hcho = {c: HCHO(f"{out_dir}/{c}.nc") for c in CASES}

    # this var is before correction
    # hcho["TROPOMI"] = HCHO(TROPO_FILE, hcho_var="tcolhcho_apriori")
    # hcho["OMI"] = HCHO(OMI_FILE, hcho_var="tcolhcho_apriori")

    return hcho


def plt_reg(hcho):
    list_regions = [
        "AMZ",
        "ENA",
        "EAS",
        "SEA",
        "NAU",
        "Indonesia",
        "C_Africa",
        "N_Africa",
        "S_Africa",
        # "REMOTE_PACIFIC",
    ]
    colors = [
        "#D55E00",  # vermillion
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#CC79A7",  # reddish purple
        "#999933",  # olive green
        "#882255",
    ]

    ylim_dict = {
        "AMZ": (0, 20),
        "ENA": (0, 15),
        # "SAF": (0, 15),
        # "MED": (0, 15),
        # "CEU": (0, 15),
        "EAS": (0, 15),
        # "SAS": (0, 15),
        "SEA": (0, 20),
        "NAU": (0, 15),
        "Indonesia": (0, 20),
        "C_Africa": (0, 20),
        "N_Africa": (0, 20),
        "S_Africa": (0, 20),
        "REMOTE_PACIFIC": (0, 4),
    }

    tits = ["Seasonal", "Inter-annual Variability"]
    for k, mode in enumerate(["ss", "ann"]):
        index = "month" if mode == "ss" else "year"
        fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

        for i, r in enumerate(list_regions):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]
            for j, c in enumerate(list(hcho.keys())[::-1]):
                ds = hcho[c].reg_ss if mode == "ss" else hcho[c].reg_ann
                reg_df = ds[[index, r]].set_index(index).rename(columns={r: c})
                sns.lineplot(
                    reg_df,
                    ax=ax,
                    palette=[colors[j]],
                    markers=True,
                    lw=2,
                )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel("Year")
            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            if ri < 2:
                ax.set_xlabel("")
            ax.set_ylabel("(\u00d710$^{15}$ molec.cm$^{-2}$)")
            ax.set_title(f"{r}")
            ax.set_xlim(2005, 2023)
            if r in ylim_dict:
                ax.set_ylim(ylim_dict[r])

        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )
        plt.suptitle(tits[k], fontsize=16, fontweight="bold")


# %%
