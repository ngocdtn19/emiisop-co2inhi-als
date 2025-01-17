# %%
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import os
import pygtool_core
import pygtool
import xarray as xr
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from utils import *

geogrid = pygtool.readgrid()
Clon, Clat = geogrid.getlonlat()
BASE_DIR = "/mnt/dg3/ngoc/CHASER_output"
# CASES = [f.split("/")[-1] for f in glob(f"{BASE_DIR}/*20012023_nudg")]
CASES = ["VISIT20172023_no_nudg", "UKpft20172023_no_nudg"]
SIGMA = load_sigma()

# step 1
# - input var: Temp, HCHO - ch2o, Pressure - output from CHASER simulation
# - time: sampling from 12 to 14h
# - calculation:
#       1: Pressure = pressure * sigma from "/home/onishi/GTAXDIR/GTAXLOC.HETA36"
#           not used => no need to do?
#  - note: no use of CHASER pressure


def sampling_12to14h():
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
    years = np.arange(2017, 2024)
    # years = np.arange(2001, 2024)
    sigma = np.arange(1, 37)

    for case in CASES:
        case_dir = f"{BASE_DIR}/{case}"
        for var in vars:

            out_dir = f"{BASE_DIR}/sample_12to14h/{case}"
            out_file = f"{out_dir}/{var}.nc"

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if not os.path.exists(out_file):

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
                    sampled_ds = process_ps(sampled_ds) if var == "ps" else sampled_ds

                    list_ds.append(sampled_ds)

                concat_ds = xr.concat(list_ds, dim="time")
                concat_ds["time"] = np.array(
                    concat_ds.time.values, dtype="datetime64[D]"
                )

                concat_ds.to_netcdf(out_file)
            print(case, var)


# step 2
# - input var:
#       - Temp, ch2o sampled from 12 to 14h - output from CHASER simulation
#       - AK + Pressure data - from TROPOMI data
# - calculation:
#       1: caluate the layer pressure  = ctm_sigma_a + surface_pressure*ctm_sigma_b
#           Satellite data is from /mnt/dg3/ngoc/obs_data/**L3 HCHO TROPOMI
#       2: Following equation form Hoque-san to recalculate the column HCHO


def extract_layer_sat_ps(ds):
    layers = []
    surf_p = ds["surface_pressure"].values
    layers.append(surf_p)

    sigma_a = ds["ctm_sigma_a"].values
    sigma_b = ds["ctm_sigma_b"].values
    for l in range(len(sigma_a)):
        l_p = surf_p * sigma_b[l] + sigma_a[l]
        layers.append(l_p)
    layer_p = np.stack(layers, axis=-1)
    print(layer_p.shape)

    return xr.Dataset(
        {
            "layer_pressure": (
                ("time", "lat", "lon", "sat_layer"),
                layer_p,
            )
        },
        coords={
            "time": ds.time.values,
            "lat": ds.lat.values,
            "lon": ds.lon.values,
            "sat_layer": np.arange(layer_p.shape[-1]),
        },
    )


def interpolate_to_sat_ps(c_t, c_ch2o, c_ps, sat_ps, case):

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

    def plot_interp(interp_t_df, interp_hcho_df, sat_ps_df, chaser_df, case):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        tits = ["Temp", "HCHO", "Pressure"]
        sat_cols = ["interp_t", "interp_hcho", "layer_pressure"]
        sat_labels = ["Interp", "Interp", "Sat"]
        c_cols = ["T", "CH2O", "PS"]
        for i, df in enumerate([interp_t_df, interp_hcho_df, sat_ps_df]):
            ax = axes[i]
            ax.plot(
                df.index, df[sat_cols[i]], label=sat_labels[i], color="blue", marker="o"
            )

            ax.plot(
                chaser_df.index,
                chaser_df[c_cols[i]],
                label="Chaser",
                color="red",
                marker="s",
            )
            ax.set_title(f"Line Plot of {tits[i]} by Layer")
            ax.legend()
            ax.set_xlabel("Layer")
            ax.set_ylabel("Value")
            ax.grid(True)

        fig.suptitle(case, fontsize=16, fontweight="bold")

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
    sat_intep_temp, sat_interp_hcho = xr.apply_ufunc(
        interp_t_ch2o,
        ds["T"],
        ds["CH2O"],
        ds["PS"],
        sat_ps["layer_pressure"],
        input_core_dims=[["layer"], ["layer"], ["layer"], ["sat_layer"]],
        output_core_dims=[["sat_layer"], ["sat_layer"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    sat_intep_temp = sat_intep_temp.where(np.isfinite(sat_intep_temp), np.nan)
    sat_interp_hcho = sat_interp_hcho.where(np.isfinite(sat_interp_hcho), np.nan)

    # global mean plot checking
    sat_levels = sat_ps["layer_pressure"].values.shape[-1]
    sat_intep_temp["sat_layer"] = np.arange(sat_levels)
    sat_interp_hcho["sat_layer"] = np.arange(sat_levels)

    dims = ["time", "lat", "lon"]

    interp_t_df = sat_intep_temp.mean(dims).to_dataframe(name="interp_t")
    interp_hcho_df = sat_interp_hcho.mean(dims).to_dataframe(name="interp_hcho")
    sat_ps_df = sat_ps.mean(dims).to_dataframe()

    chaser_df = ds.mean(dims).to_dataframe()

    plot_interp(interp_t_df, interp_hcho_df, sat_ps_df, chaser_df, case)
    return sat_intep_temp, sat_interp_hcho


def ak_apply_chaser(
    sat_ds, chaser_t, chaser_ch2o, chaser_ps, sat_pressure, case, sat_interp
):
    def ak_apply(p, p_next, t, t_next, ak, hcho):
        def p2h(p, t):
            return (-1) * ((np.log(p / 101325)) * 8.314 * t) / (0.028 * 9.8)

        H = abs(p2h(p, t) - p2h(p_next, t_next))
        return ((hcho * 1e-9 * H * p * 100 * 1e-4) / (1.38e-23 * t)) * ak

    # S1: Matching time between CHASER and SATELLITE
    sat_time = sat_ds.time.values

    Ctemp_filtered = chaser_t.sel(time=(chaser_t.time.isin(sat_time)))
    Cch2o_filtered = chaser_ch2o.sel(time=(chaser_ch2o.time.isin(sat_time)))
    Cps_filtered = chaser_ps.sel(time=(chaser_ps.time.isin(sat_time)))

    c_time = Ctemp_filtered.time.values
    sat_filtered = sat_ds.sel(time=(sat_ds.time.isin(c_time)))
    sat_pressure = sat_pressure.where(np.isfinite(sat_pressure), np.nan)
    sat_ps_filtered = sat_pressure.sel(time=(sat_pressure.time.isin(c_time)))

    assert len(sat_filtered.time.values) == len(Cch2o_filtered.time.values)

    # S2: Interpolate HCHO and Temp from CHASER to SAT's Pressure
    if sat_interp:
        sat_intep_temp, sat_interp_hcho = interpolate_to_sat_ps(
            Ctemp_filtered, Cch2o_filtered, Cps_filtered, sat_ps_filtered, case
        )

    # # S3: do AK application
    sat_ak = sat_filtered["AK"].values  # (time, lat, lon, layer)
    sat_ps_val = sat_ps_filtered["layer_pressure"].values  # (time, lat, lon, layer)

    c_temp = Ctemp_filtered["T"].values  # (time, lat, lon, layer)
    c_ch2o = Cch2o_filtered["CH2O"].values  # (time, lat, lon, layer)

    if sat_interp:
        c_temp = sat_intep_temp.values  # (time, lat, lon, layer)
        c_ch2o = sat_interp_hcho.values  # (time, lat, lon, layer)

    hcho_layers = []
    for l in range(sat_ak.shape[-1]):
        p = sat_ps_val[:, :, :, l]
        p_next = sat_ps_val[:, :, :, l + 1]

        ak = sat_ak[:, :, :, l]

        t = c_temp[:, :, :, l]
        t_next = c_temp[:, :, :, l + 1]
        hcho = c_ch2o[:, :, :, l]

        ak_hcho = ak_apply(p, p_next, t, t_next, ak, hcho)

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
            "lat": Ctemp_filtered.lat.values,
            "lon": Ctemp_filtered.lon.values,
        },
    )

    return hcho_ds["hcho"].sum("layer")


def ak_apply_to_chaser_do():

    sat_dir = f"/mnt/dg3/ngoc/obs_data"
    time_omi = "20050101-20231201"
    time_tropo = "20180601-20240701"

    m_name_omi = "mon_BIRA_OMI_HCHO_L3"
    m_name_tropo = "mon_TROPOMI_HCHO_L3"
    omi_file = f"{sat_dir}/{m_name_omi}/EXTRACT/hcho_AERmon_{m_name_omi}_historical_gn_{time_omi}.nc"
    tropo_file = f"{sat_dir}/{m_name_tropo}/EXTRACT/hcho_AERmon_{m_name_tropo}_historical_gn_{time_tropo}.nc"

    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"

    ds_omi = xr.open_dataset(omi_file)
    omi_pressure = extract_layer_sat_ps(ds_omi)

    ds_tropo = xr.open_dataset(tropo_file)
    tropo_pressure = extract_layer_sat_ps(ds_tropo)

    for sat_interp in [True, False]:
        for case in CASES:
            # case = "MIXpft20012023_nudg"

            chaser_case_dir = f"{BASE_DIR}/sample_12to14h/{case}"
            t_ds = xr.open_dataset(f"{chaser_case_dir}/t.nc").resample(time="M").mean()
            ch2o_ds = (
                xr.open_dataset(f"{chaser_case_dir}/ch2o.nc").resample(time="M").mean()
            )
            ps_ds = (
                xr.open_dataset(f"{chaser_case_dir}/ps.nc").resample(time="M").mean()
            )

            t_ds = prep_chaser_for_ak(t_ds)
            ch2o_ds = prep_chaser_for_ak(ch2o_ds)
            ps_ds = prep_chaser_for_ak(ps_ds)

            t_ds["time"] = t_ds.time.values.astype("datetime64[M]").astype(
                "datetime64[ns]"
            )
            ch2o_ds["time"] = ch2o_ds.time.values.astype("datetime64[M]").astype(
                "datetime64[ns]"
            )
            ps_ds["time"] = ps_ds.time.values.astype("datetime64[M]").astype(
                "datetime64[ns]"
            )
            ps_ds = ps_ds.rename({list(ps_ds.data_vars)[0]: "PS"})

            ch2o_tropo_ak = ak_apply_chaser(
                ds_tropo,
                t_ds,
                ch2o_ds,
                ps_ds,
                tropo_pressure,
                f"{case}_TROPO",
                sat_interp,
            )
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

            ch2o_tropo_ak.to_netcdf(f"{out_dir_case}/tropo_ak_hcho.nc")
            ch2o_omi_ak.to_netcdf(f"{out_dir_case}/omi_ak_hcho.nc")


# %%
