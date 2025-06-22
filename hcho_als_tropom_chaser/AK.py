# %%
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import os
import pygtool_core
import pygtool
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from utils import *

geogrid = pygtool.readgrid()
Clon, Clat = geogrid.getlonlat()
BASE_DIR = "/mnt/dg3/ngoc/CHASER_output"
CASES = [f.split("/")[-1] for f in glob(f"{BASE_DIR}/*20012023_nudg")]
# CASES = ["BVOCoff20012023_nudg"]
SIGMA = load_sigma()

# step 1
# - input var: Temp, HCHO - ch2o, Pressure - output from CHASER simulation
# - time: sampling from 12 to 14h
# - calculation:
#       1: Pressure = pressure * sigma from "/home/onishi/GTAXDIR/GTAXLOC.HETA36"
#           not used => no need to do?
#  - note: no use of CHASER pressure


def sampling_12to14h(years=np.arange(2005, 2024), cases=CASES):
    def process_ps(ds):
        ps = ds["PS"].values
        ps_layers = [ps * SIGMA[l] for l in range(len(SIGMA))]
        ps_layers = np.stack(ps_layers, axis=-1)
        return xr.Dataset(
            {
                "PS": (
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
    # vars = ["ps"]
    sigma = np.arange(1, 37)

    for case in cases:
        case_dir = f"{BASE_DIR}/{case}"
        for var in vars:

            out_dir = f"{BASE_DIR}/sample_12to14h/{case}"
            out_file = f"{out_dir}/{var}.nc"

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # if not os.path.exists(out_file):

            list_ds = []
            for y in years:
                chaser_var_path = f"{case_dir}/{y}/2hr/{var}"
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

                sampled_ds["time"] = pd.to_datetime(sampled_ds["time"].values)

                sampled_ds = process_ps(sampled_ds) if var == "ps" else sampled_ds

                list_ds.append(sampled_ds)
                # print(sampled_ds.time.values)

            concat_ds = xr.concat(list_ds, dim="time")
            # concat_ds["time"] = np.array(
            #     concat_ds.time.values, dtype="datetime64[D]"
            # )

            concat_ds.to_netcdf(out_file)
            print(case, var)
            # concat_ds.to_netcdf(
            #     out_file,
            #     encoding={
            #         "time": {"dtype": "int64", "units": "hours since 1970-01-01"}
            #     },
            # )


# step 2
# - input var:
#       - Temp, ch2o sampled from 12 to 14h - output from CHASER simulation
#       - AK + Pressure data - from TROPOMI data
# - calculation:
#       1: caluate the layer pressure  = ctm_sigma_a + surface_pressure*ctm_sigma_b
#           Satellite data is from /mnt/dg3/ngoc/obs_data/**L3 HCHO TROPOMI
#       2: Following equation form Hoque-san to recalculate the column HCHO


def extract_layer_sat_ps(ds):
    ds = ds.where(np.isfinite(ds), np.nan)

    sigma_a = ds["ctm_sigma_a"].values
    sigma_b = ds["ctm_sigma_b"].values
    surf_p = ds["surface_pressure"].values

    center_ps_layers = []
    border_ps_layers = []

    nlayers = len(sigma_a)
    # print(nlayers)

    # center_ps_layers.append(surf_p)
    border_ps_layers.append(surf_p)

    # cal center ps
    for l in range(nlayers):
        center_ps = surf_p * sigma_b[l] + sigma_a[l]
        center_ps_layers.append(center_ps)

    # cal border ps

    for l in range(nlayers - 1):
        border_ps = np.exp(
            np.log(center_ps_layers[l])
            - (np.log(center_ps_layers[l]) - np.log(center_ps_layers[l + 1])) / 2
        )

        border_ps_layers.append(border_ps)
    # cal border ps for last layer
    l_last = nlayers - 1

    border_ps_last = np.exp(
        np.log(border_ps_layers[l_last])
        - 2 * (np.log(border_ps_layers[l_last]) - np.log(center_ps_layers[l_last]))
    )

    border_ps_layers.append(border_ps_last)

    center_ps_layers = np.stack(center_ps_layers, axis=-1)
    border_ps_layers = np.stack(border_ps_layers, axis=-1)
    # print(center_ps_layers.shape)
    # print(border_ps_layers.shape)

    center_ps_ds = xr.Dataset(
        {
            "Sat_center_ps": (
                ("time", "lat", "lon", "sat_layer"),
                center_ps_layers,
            )
        },
        coords={
            "time": ds.time.values,
            "lat": ds.lat.values,
            "lon": ds.lon.values,
            "sat_layer": np.arange(center_ps_layers.shape[-1]),
        },
    )
    border_ps_ds = xr.Dataset(
        {
            "Sat_border_ps": (
                ("time", "lat", "lon", "sat_layer"),
                border_ps_layers,
            )
        },
        coords={
            "time": ds.time.values,
            "lat": ds.lat.values,
            "lon": ds.lon.values,
            "sat_layer": np.arange(border_ps_layers.shape[-1]),
        },
    )
    return center_ps_ds, border_ps_ds


def interpolate_to_sat_ps(c_t, c_ch2o, c_ps, sat_center, sat_border, case):

    def interp_t_ch2o(temp, hcho, old_p, new_p):
        """
        Interpolate temperature and hcho to new pressure levels for each point.
        """
        # Interpolate temperature
        fv = "extrapolate"
        interp_func_temp = interp1d(
            old_p, temp, axis=-1, bounds_error=False, fill_value=fv
        )
        new_temp = interp_func_temp(new_p)

        # Interpolate hcho
        interp_func_hcho = interp1d(
            old_p, hcho, axis=-1, bounds_error=False, fill_value=fv
        )
        new_hcho = interp_func_hcho(new_p)

        return new_temp, new_hcho

    ds = xr.Dataset(
        {
            "T": (("time", "lat", "lon", "layer"), c_t["T"].values),
            "CH2O": (("time", "lat", "lon", "layer"), c_ch2o["CH2O"].values),
            "PS": (("time", "lat", "lon", "layer"), c_ps["PS"].values),
        },
        coords={
            "time": c_t.time.values,
            "lat": c_t.lat.values,
            "lon": c_t.lon.values,
            "layer": np.arange(len(c_t.layer.values)),
        },
    )
    # sat_intep_temp, sat_interp_hcho = reg_t_ch2o(ds, sat_ps)
    sat_intep_temp, sat_interp_hcho = xr.apply_ufunc(
        interp_t_ch2o,
        ds["T"],
        ds["CH2O"],
        ds["PS"],
        sat_border["Sat_border_ps"],
        input_core_dims=[["layer"], ["layer"], ["layer"], ["sat_layer"]],
        output_core_dims=[["sat_layer"], ["sat_layer"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    sat_intep_temp = sat_intep_temp.where(np.isfinite(sat_intep_temp), np.nan)
    sat_interp_hcho = sat_interp_hcho.where(np.isfinite(sat_interp_hcho), np.nan)

    return sat_intep_temp, sat_interp_hcho


def ak_apply(p, p_next, t, t_next, ak, hcho):
    def p2h(p, t):
        return (-1) * ((np.log(p * 100 / 101325)) * 8.314 * t) / (0.028 * 9.8)

    H = abs(p2h(p, t) - p2h(p_next, t_next))
    return ((hcho * 1e-9 * H * p * 100 * 1e-4) / (1.38e-23 * t)) * ak


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


def ak_apply_chaser(
    sat_ds, chaser_t, chaser_ch2o, chaser_ps, sat_pressure, case, sat_interp
):
    # S1: Matching time between CHASER and SATELLITE
    sat_time = sat_ds.time.values

    Ctemp_filtered = chaser_t.sel(time=(chaser_t.time.isin(sat_time)))
    Cch2o_filtered = chaser_ch2o.sel(time=(chaser_ch2o.time.isin(sat_time)))
    Cps_filtered = chaser_ps.sel(time=(chaser_ps.time.isin(sat_time)))

    c_time = Ctemp_filtered.time.values
    sat_filtered = sat_ds.sel(time=(sat_ds.time.isin(c_time)))
    sat_filtered = sat_filtered.where(np.isfinite(sat_filtered), np.nan)

    sat_center_ps, sat_border_ps = sat_pressure

    sat_center_ps = sat_center_ps.where(np.isfinite(sat_center_ps), np.nan)
    center_ps_filtered = sat_center_ps.sel(time=(sat_center_ps.time.isin(c_time)))

    sat_border_ps = sat_border_ps.where(np.isfinite(sat_border_ps), np.nan)
    border_ps_filtered = sat_border_ps.sel(time=(sat_border_ps.time.isin(c_time)))

    assert len(sat_filtered.time.values) == len(Cch2o_filtered.time.values)

    # S2: Interpolate HCHO and Temp from CHASER to SAT's Pressure
    if sat_interp:
        sat_intep_temp, sat_interp_hcho = interpolate_to_sat_ps(
            Ctemp_filtered,
            Cch2o_filtered,
            Cps_filtered,
            center_ps_filtered,
            border_ps_filtered,
            case,
        )

    # # S3: do AK application
    sat_ak = sat_filtered["AK"].values  # (time, lat, lon, layer)
    sat_ps_val = border_ps_filtered["Sat_border_ps"].values  # (time, lat, lon, layer)

    c_temp = Ctemp_filtered["T"].values  # (time, lat, lon, layer)
    c_ch2o = Cch2o_filtered["CH2O"].values  # (time, lat, lon, layer)
    c_ps = Cps_filtered["PS"].values  # (time, lat, lon, layer)

    if sat_interp:
        c_temp = sat_intep_temp.values  # (time, lat, lon, layer)
        c_ch2o = sat_interp_hcho.values  # (time, lat, lon, layer)

        # plot checking
        # fig, axis = plt.subplots(1, 3, figsize=(4 * 3, 4.2), layout="constrained")
        # sat_intep_temp.mean(["lat", "lon", "sat_layer"]).to_dataframe(
        #     name="Sat_interp_temp"
        # ).plot.line(ax=axis[0], color="red")
        # Ctemp_filtered.rename({"T": "Chaser_temp"}).mean(
        #     ["lat", "lon", "layer"]
        # ).to_dataframe().plot.line(ax=axis[0], color="blue")

        # sat_interp_hcho.mean(["lat", "lon", "sat_layer"]).to_dataframe(
        #     name="Sat_interp_hcho"
        # ).plot.line(ax=axis[1], color="red")
        # Cch2o_filtered.rename({"CH2O": "Chaser_hcho"}).mean(
        #     ["lat", "lon", "layer"]
        # ).to_dataframe().plot.line(ax=axis[1], color="blue")

        # center_ps_filtered.mean(["lat", "lon", "sat_layer"]).to_dataframe().plot.line(
        #     ax=axis[2], color="red"
        # )
        # border_ps_filtered.mean(["lat", "lon", "sat_layer"]).to_dataframe().plot.line(
        #     ax=axis[2], color="green"
        # )

        # Cps_filtered.rename({"PS": "Chaser_pressure"}).mean(
        #     ["lat", "lon", "layer"]
        # ).to_dataframe().plot.line(ax=axis[2], color="blue")
        # plt.suptitle(case, fontsize=16, fontweight="bold")
        # print(sat_pressure.time.values)

    hcho_layers = []
    no_layers = 36 if sat_ak.shape[-1] > 36 else sat_ak.shape[-1]
    for l in range(no_layers - 1):
        # sattelite pressure
        l_ps = l + 1
        p = sat_ps_val[:, :, :, l_ps]
        p_next = sat_ps_val[:, :, :, l_ps + 1]

        # chaser pressure
        # p = c_ps[:, :, :, l]
        # p_next = c_ps[:, :, :, l + 1]

        ak = sat_ak[:, :, :, l]

        t = c_temp[:, :, :, l]
        t_next = c_temp[:, :, :, l + 1]
        hcho = c_ch2o[:, :, :, l]

        ak_hcho = ak_apply(p, p_next, t, t_next, ak, hcho)

        hcho_layers.append(ak_hcho)

    assert len(hcho_layers) == no_layers - 1
    print("No. layers: ", len(hcho_layers))

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
            "lat": Ctemp_filtered.lat.values,
            "lon": Ctemp_filtered.lon.values,
        },
    )

    hcho_ds = hcho_ds["hcho"].sum("layer", skipna=True)
    print("masking")
    hcho_ds_filter = mask_chaser_by_sat_hcho(hcho_ds, sat_filtered)
    return hcho_ds_filter


def ak_apply_to_chaser_do(sat_version="v1"):

    sat_dir = f"/mnt/dg3/ngoc/obs_data"
    if sat_version == "v1":
        time_omi = "20050101-20231201"
        time_tropo = "20180601-20240701"
    else:
        time_omi = "20050101-20221231"
        time_tropo = "20180507-20231231"

    m_name_omi = f"mon_BIRA_OMI_HCHO_L3_{sat_version}"
    m_name_tropo = f"mon_TROPOMI_HCHO_L3_{sat_version}"

    omi_file = f"{sat_dir}/{m_name_omi}/EXTRACT/hcho_AERmon_{m_name_omi}_historical_gn_{time_omi}.nc"
    tropo_file = f"{sat_dir}/{m_name_tropo}/EXTRACT/hcho_AERmon_{m_name_tropo}_historical_gn_{time_tropo}.nc"

    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"

    ds_omi = xr.open_dataset(omi_file)
    omi_pressure = extract_layer_sat_ps(ds_omi)

    # ds_tropo = xr.open_dataset(tropo_file)
    # tropo_pressure = extract_layer_sat_ps(ds_tropo)

    for sat_interp in [True, False]:
        for case in CASES:

            chaser_case_dir = f"{BASE_DIR}/sample_12to14h/{case}"
            t_ds = xr.open_dataset(f"{chaser_case_dir}/t.nc")
            t_ds = t_ds.resample(time="M").mean()

            ch2o_ds = xr.open_dataset(f"{chaser_case_dir}/ch2o.nc")
            ch2o_ds = ch2o_ds.resample(time="M").mean()

            ps_ds = xr.open_dataset(f"{chaser_case_dir}/ps.nc")
            ps_ds = ps_ds.resample(time="M").mean()

            t_ds = prep_chaser_for_ak(t_ds)
            ch2o_ds = prep_chaser_for_ak(ch2o_ds)
            ps_ds = prep_chaser_for_ak(ps_ds)

            dtM = "datetime64[M]"
            dtns = "datetime64[ns]"

            t_ds["time"] = t_ds.time.values.astype(dtM).astype(dtns)
            ch2o_ds["time"] = ch2o_ds.time.values.astype(dtM).astype(dtns)
            ps_ds["time"] = ps_ds.time.values.astype(dtM).astype(dtns)

            ps_ds = ps_ds.rename({list(ps_ds.data_vars)[0]: "PS"})

            # ch2o_tropo_ak = ak_apply_chaser(
            #     ds_tropo,
            #     t_ds,
            #     ch2o_ds,
            #     ps_ds,
            #     tropo_pressure,
            #     f"{case}_TROPO",
            #     sat_interp,
            # )
            ch2o_omi_ak = ak_apply_chaser(
                ds_omi, t_ds, ch2o_ds, ps_ds, omi_pressure, f"{case}_OMI", sat_interp
            )

            interp_folder = "no_sat_interp"
            if sat_interp:
                interp_folder = "sat_interp"

            out_dir_case = f"{out_dir}/{case}/{interp_folder}"
            if not os.path.exists(out_dir_case):
                os.makedirs(out_dir_case)

            print(case, interp_folder)

            # ch2o_tropo_ak.to_netcdf(f"{out_dir_case}/tropo_ak_hcho_{sat_version}.nc")
            ch2o_omi_ak.to_netcdf(f"{out_dir_case}/omi_ak_hcho_{sat_version}.nc")


def clean_before_ak(keywords, base_dir=None):
    if base_dir is None:
        base_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file = os.path.join(root, file)
            for case in CASES:
                if case in file:
                    if keywords in file:
                        rmv_file(file)


def ak_do_hoque_ch2o_tropo_l2():
    # data
    # /mnt/dg2/hoque/TROPOMI_model_analysis/New_emission_chaser_2019_2020/2019/ch2o
    # /mnt/dg2/hoque/TROPOMI_model_analysis/2019/t

    # sampling_12to14h(years=[2019], cases=["hoque_sim_2019"])
    sat_ak_ps = xr.open_dataset(
        "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/TROPOMI_AK_2019_combined.nc"
    )
    sat_ak_ps = sat_ak_ps.rename({"level": "sat_layer"})

    chaser_case_dir = f"{BASE_DIR}/sample_12to14h/hoque_sim_2019"
    chaser_t = xr.open_dataset(f"{chaser_case_dir}/t.nc")
    chaser_ch2o = xr.open_dataset(f"{chaser_case_dir}/ch2o.nc")

    chaser_t = prep_chaser_for_ak(chaser_t)
    chaser_ch2o = prep_chaser_for_ak(chaser_ch2o)

    dtD = "datetime64[D]"
    dtns = "datetime64[ns]"

    list_ds = []
    for hour in [12, 14]:
        chaser_t_h = chaser_t.sel(time=(chaser_t.time.dt.hour.isin([hour])))
        chaser_ch2o_h = chaser_ch2o.sel(time=(chaser_ch2o.time.dt.hour.isin([hour])))

        chaser_t_h["time"] = chaser_t_h.time.values.astype(dtD).astype(dtns)
        chaser_ch2o_h["time"] = chaser_ch2o_h.time.values.astype(dtD).astype(dtns)

        sat_time = sat_ak_ps.time.values

        Ctemp_filtered = chaser_t_h.sel(time=(chaser_t_h.time.isin(sat_time)))
        Cch2o_filtered = chaser_ch2o_h.sel(time=(chaser_ch2o_h.time.isin(sat_time)))

        c_time = Ctemp_filtered.time.values
        sat_filtered = sat_ak_ps.sel(time=(sat_ak_ps.time.isin(c_time)))
        sat_filtered = sat_filtered.where(np.isfinite(sat_filtered), np.nan)

        assert len(sat_filtered.time.values) == len(Cch2o_filtered.time.values)

        sat_ak = sat_filtered["AK"].values  # (time, lat, lon, layer)
        sat_ps_val = sat_filtered["pressure"].values * 1e-2  # (time, lat, lon, layer)

        c_temp = Ctemp_filtered["T"].values  # (time, lat, lon, layer)
        c_ch2o = Cch2o_filtered["CH2O"].values  # (time, lat, lon, layer)

        hcho_layers = []
        for l in range(sat_ak.shape[-1] - 1):

            p = sat_ps_val[:, :, :, l]
            p_next = sat_ps_val[:, :, :, l + 1]

            ak = sat_ak[:, :, :, l]

            t = c_temp[:, :, :, l]
            t_next = c_temp[:, :, :, l + 1]
            hcho = c_ch2o[:, :, :, l]

            ak_hcho = ak_apply(p, p_next, t, t_next, ak, hcho)

            hcho_layers.append(ak_hcho)

        assert len(hcho_layers) == 33

        hcho_ds = xr.Dataset(
            {
                "hcho": (
                    ("layer", "time", "lat", "lon"),
                    np.stack(hcho_layers, axis=0),
                )
            },
            coords={
                "layer": np.arange(len(hcho_layers)),
                "time": c_time.astype("datetime64[ns]") + np.timedelta64(hour, "h"),
                "lat": Ctemp_filtered.lat.values,
                "lon": Ctemp_filtered.lon.values,
            },
        )

        hcho_ds = hcho_ds.sel(layer=~hcho_ds.layer.isin([5, 10, 15, 20, 25, 30]))
        assert len(hcho_ds.layer.values) == 27
        final_ds = hcho_ds["hcho"].sum("layer", skipna=True)
        list_ds.append(final_ds)
    final_ds = xr.concat(list_ds, dim="time")

    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"
    case = "hoque_sim_2019"
    interp_folder = "no_sat_interp"
    out_dir_case = f"{out_dir}/{case}/{interp_folder}"
    if not os.path.exists(out_dir_case):
        os.makedirs(out_dir_case)

    final_ds.to_netcdf(f"{out_dir_case}/tropo_ak_hcho.nc")


# %%
