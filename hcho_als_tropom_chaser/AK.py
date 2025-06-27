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
# CASES = ["VISITst20012023_nudg"]
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


def get_chaser_data(case):
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

    ds = xr.merge([ps_ds, t_ds, ch2o_ds])
    return ds


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
    ds["layer"] = np.arange(len(ds["layer"]))

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

    center_ps_da = xr.DataArray(
        center_ps_layers,
        dims=("time", "lat", "lon", "layer"),
        coords={
            "time": ds.time,
            "lat": ds.lat,
            "lon": ds.lon,
            "layer": np.arange(center_ps_layers.shape[-1]),
        },
        name="center_ps",
    )

    border_ps_da = xr.DataArray(
        border_ps_layers,
        dims=("time", "lat", "lon", "layer"),
        coords={
            "time": ds.time,
            "lat": ds.lat,
            "lon": ds.lon,
            "layer": np.arange(border_ps_layers.shape[-1]),
        },
        name="border_ps",
    )
    # Assign to original dataset
    ds["center_ps"] = center_ps_da
    ds["border_ps"] = border_ps_da
    ds = ds.rename({"layer": "sat_layer"})
    return ds


def interp_to_sat(chaser_ds, sat_ds):

    def interp_fn(temp, hcho, old_p, new_p):
        fv = "extrapolate"

        # Step 1: Normalize each variable (min-max)
        def normalize(x):
            return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

        def denormalize(x_norm, x_orig):
            return x_norm * (np.nanmax(x_orig) - np.nanmin(x_orig)) + np.nanmin(x_orig)

        temp_norm = normalize(temp)
        hcho_norm = normalize(hcho)
        old_p_norm = normalize(old_p)
        new_p_norm = (new_p - np.nanmin(old_p)) / (np.nanmax(old_p) - np.nanmin(old_p))

        # Step 2: Interpolate in normalized space
        temp_fn = interp1d(
            old_p_norm, temp_norm, axis=-1, bounds_error=False, fill_value=fv
        )
        hcho_fn = interp1d(
            old_p_norm, hcho_norm, axis=-1, bounds_error=False, fill_value=fv
        )

        temp_interp_norm = temp_fn(new_p_norm)
        hcho_interp_norm = hcho_fn(new_p_norm)

        # Step 3: Un-normalize the result
        temp_interp = denormalize(temp_interp_norm, temp)
        hcho_interp = denormalize(hcho_interp_norm, hcho)

        return temp_interp, hcho_interp

    sat_intep_temp, sat_interp_hcho = xr.apply_ufunc(
        interp_fn,
        chaser_ds["T"],
        chaser_ds["CH2O"],
        np.log(chaser_ds["PS"]),
        np.log(sat_ds["border_ps"]),
        input_core_dims=[["layer"], ["layer"], ["layer"], ["sat_layer"]],
        output_core_dims=[["sat_layer"], ["sat_layer"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    sat_intep_temp = sat_intep_temp.where(np.isfinite(sat_intep_temp), np.nan)
    sat_interp_hcho = sat_interp_hcho.where(np.isfinite(sat_interp_hcho), np.nan)

    return sat_intep_temp, sat_interp_hcho


def interp_to_chaser(chaser_ds, sat_ds):
    def interp_fn(ak, old_p, new_p):
        fv = "extrapolate"
        ak_fn = interp1d(old_p, ak, axis=-1, bounds_error=False, fill_value=fv)
        return ak_fn(new_p)

    interp_ak = xr.apply_ufunc(
        interp_fn,
        sat_ds["AK"],
        sat_ds["border_ps"],
        chaser_ds["PS"],
        input_core_dims=[["sat_layer"], ["sat_layer"], ["layer"]],
        output_core_dims=[["layer"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return interp_ak.where(np.isfinite(interp_ak), np.nan)


def assign_ak_to_model(model_pressure, sat_pressure, sat_ak):
    print("Assigning AK to model pressure layers...")
    time, lat, lon, model_layers = model_pressure.shape
    _, _, _, sat_layers = sat_pressure.shape

    p_model = model_pressure.values
    p_sat = sat_pressure.values
    ak_sat = sat_ak.values

    n_layers = min(model_layers, sat_layers)

    mapped_ak = np.full_like(p_model, np.nan)

    for t in range(time):
        for y in range(lat):
            for x in range(lon):

                p_trop = p_sat[t, y, x, :]
                if np.any(np.isnan(p_trop)):
                    continue

                for m in range(n_layers):
                    p_mod = p_model[t, y, x, m]

                    if np.isnan(p_mod):
                        continue
                    idx = np.where((p_trop[:-1] >= p_mod) & (p_trop[1:] <= p_mod))[0]

                    if idx.size > 0:
                        i = idx[0]
                        mapped_ak[t, y, x, m] = ak_sat[t, y, x, i]
                    else:
                        continue
    # mapped_ak = np.nan_to_num(mapped_ak)

    print(mapped_ak.shape)
    return mapped_ak


def ak_apply(p, p_next, t, t_next, ak, hcho):
    def p2h(p, t):
        return (-1) * ((np.log(p * 100 / 101325)) * 8.314 * t) / (0.028 * 9.8)

    H = abs(p2h(p, t) - p2h(p_next, t_next))
    return ((hcho * 1e-9 * H * p * 100 * 1e-4) / (1.38e-23 * t)) * ak


def ak_apply_chaser(sat_ds, chaser_ds, interp_to):
    # S1: Matching time between CHASER and SATELLITE
    chaser_filtered = chaser_ds.sel(time=(chaser_ds.time.isin(sat_ds.time.values)))
    sat_filtered = sat_ds.sel(time=(sat_ds.time.isin(chaser_filtered.time.values)))
    sat_filtered = sat_filtered.where(np.isfinite(sat_filtered), np.nan)

    assert len(sat_filtered.time.values) == len(chaser_filtered.time.values)

    # # S2: do AK application
    used_ak = sat_filtered["AK"].values  # (time, lat, lon, layer)
    sat_ps_val = sat_filtered["border_ps"].values  # (time, lat, lon, layer)

    c_temp = chaser_filtered["T"].values  # (time, lat, lon, layer)
    c_ch2o = chaser_filtered["CH2O"].values  # (time, lat, lon, layer)
    c_ps = chaser_filtered["PS"].values  # (time, lat, lon, layer)

    if interp_to == "sat":
        sat_intep_temp, sat_interp_hcho = interp_to_sat(chaser_filtered, sat_filtered)
        c_temp = sat_intep_temp.values  # (time, lat, lon, layer)
        c_ch2o = sat_interp_hcho.values  # (time, lat, lon, layer)
    elif interp_to == "chaser":
        chaser_ak = interp_to_chaser(chaser_filtered, sat_filtered)
        used_ak = chaser_ak.values  # (time, lat, lon, layer)
    elif interp_to == "assign":
        used_ak = assign_ak_to_model(
            chaser_filtered["PS"], sat_filtered["center_ps"], sat_filtered["AK"]
        )

    hcho_layers = []
    # no_layers = 36 if used_ak.shape[-1] > 36 else used_ak.shape[-1]
    no_layers = 16
    for l in range(no_layers - 1):
        p = sat_ps_val[:, :, :, l]
        p_next = sat_ps_val[:, :, :, l + 1]
        # sattelite pressure
        if interp_to == "sat":
            # l_ps = l + 1
            p = sat_ps_val[:, :, :, l]
            p_next = sat_ps_val[:, :, :, l + 1]

        elif interp_to in ["chaser", "assign"]:
            p = c_ps[:, :, :, l]
            p_next = c_ps[:, :, :, l + 1]

        ak = used_ak[:, :, :, l]

        t = c_temp[:, :, :, l]
        t_next = c_temp[:, :, :, l + 1]
        hcho = c_ch2o[:, :, :, l]

        ak_hcho = ak_apply(p, p_next, t, t_next, ak, hcho)

        hcho_layers.append(ak_hcho)

    # assert len(hcho_layers) == no_layers - 1
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
            "time": chaser_filtered.time.values,
            "lat": chaser_ds.lat.values,
            "lon": chaser_ds.lon.values,
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

    ds_omi = extract_layer_sat_ps(xr.open_dataset(omi_file))
    ds_tropo = extract_layer_sat_ps(xr.open_dataset(tropo_file))

    for sat_interp in ["assign"]:
        for case in CASES:

            chaser_ds = get_chaser_data(case)

            ch2o_tropo_aked = ak_apply_chaser(ds_tropo, chaser_ds, sat_interp)
            ch2o_omi_aked = ak_apply_chaser(ds_omi, chaser_ds, sat_interp)

            interp_folder = "no_sat_interp"
            if sat_interp:
                interp_folder = "sat_interp"

            out_dir_case = f"{out_dir}/{case}/{interp_folder}"
            if not os.path.exists(out_dir_case):
                os.makedirs(out_dir_case)

            print(case, interp_folder)

            ch2o_tropo_aked.to_netcdf(
                f"{out_dir_case}/tropo_ak_hcho_{sat_version}_ak_assign.nc"
            )
            ch2o_omi_aked.to_netcdf(
                f"{out_dir_case}/omi_ak_hcho_{sat_version}_ak_assign.nc"
            )


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
