# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import numpy as np
import seaborn as sns
import regionmask
import cartopy.crs as ccrs
import pymannkendall as pymk
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from pathlib import Path

from utils import *
from max_doas import *

# AKED_CHASER_HCHO SETTING
ORG_CHASER_DIR = "/mnt/dg3/ngoc/CHASER_output"
AKED_DIR = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"

SAT_NAMES = ["TROPO", "OMI"]
# CASES = [f.split("/")[-1] for f in glob(f"{ORG_CHASER_DIR}/*2023_*")]
CASES = [
    "VISITst20012023_nudg",
    "UKpft20012023_nudg",
    "MEGANst20012023_nudg",
    "MEGANpft20012023_nudg",
    "UKst20012023_nudg",
    "MIXpft20012023_nudg",
    "BVOCoff20012023_nudg",
    "OBS",
]
BASE_CASE = "VISITst20012023_nudg"
OFF_CASE = "BVOCoff20012023_nudg"
SAT_CASE = ["OMI", "TROPOMI"]

ROIS = [
    # "AMZ",
    # "ENA",
    # "SAF",
    # "MED",
    # "CEU",
    # "EAS",
    # "SAS",
    # "SEA",
    # "REMOTE_PACIFIC",
    "Amazonia",
    "S-E US",
    "Mato Grosso",
    "Indonesia",
    "South China",
    "C_Africa",
    "N_Africa",
    "S_Africa",
    "NAU",
]

colors = [
    "#D55E00",  # vermillion
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    # "#CC79A7",  # reddish purple
    "red",
    # "#999933",  # olive green
    "#882255",
    "#AA4499",  # deep magenta
    "#44AA99",  # teal green
    "#332288",  # dark navy
]


def list_all_files(base_dir, sat_ver, cases=CASES, config_name="assign_aked_35L"):
    print(f"Searching in {base_dir}")
    print(f"for {sat_ver}")
    print(f"with config {config_name}")
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file = os.path.join(root, file)
            for case in cases:
                if case in file:
                    if sat_ver in file and config_name in file:
                        all_files.append(file)
    return all_files


def load_hcho(sat_name, sat_ver, layer_used):

    files = list_all_files(AKED_DIR, sat_ver, config_name="assign_aked_35L")
    print(files)

    obs = HCHO(get_sat_file(sat_name, sat_ver), sat_filter=None)

    ak_files = [f for f in files if sat_name in f]
    interp_files = [f for f in ak_files if "/sat_interp/" in f]

    hcho_interp = {
        Path(f).parents[1].name: HCHO(f, sat_filter=sat_name, layer_used=layer_used)
        for f in interp_files
    }
    obs_case = "OMI"
    if sat_name == "tropo":
        obs_case = "TROPOMI"
    hcho_interp[obs_case] = obs

    return hcho_interp


def load_hcho_maxdoas(sat_name, layer_used):

    files = list_all_files(AKED_DIR, "v2", config_name="assign_aked_35L")
    files_by_sat = [f for f in files if sat_name in f]
    hcho = {
        Path(f)
        .parents[1]
        .name: HCHO_maxdoas(f, MAX_DOAS_COORDS, layer_used=layer_used)
        .hcho_at_maxdoas
        for f in files_by_sat
    }
    # hcho[sat_name] = obs
    hcho["MAX_DOAS"] = read_all_maxdoas_csv()

    return hcho


def get_sat_case(list_cases):
    for c in list_cases:
        if "OMI" in c:
            return c
    return None


# interp_omi_v2 = load_hcho("omi", "v2", layer_used=14)
# interp_tropo_v2 = load_hcho("tropo", "v2", layer_used=14)


# %%
def plt_maxdoas(sat_name="omi", layer_used=15, norm=True, summer_only=False):
    hcho = load_hcho_maxdoas(sat_name, layer_used)

    f_annual, ax_annual = plt.subplots(3, 1, figsize=(4, 8), layout="constrained")
    f_ss, ax_ss = plt.subplots(3, 1, figsize=(4, 8), layout="constrained")

    for i, c in enumerate(hcho.keys()):
        df_case = hcho[c]

        for j, station in enumerate(MAX_DOAS_COORDS.keys()):
            df_station = df_case[station]
            df_ss = df_station.groupby(df_station.index.month).mean()
            df_annual = df_station.groupby(df_station.index.year).mean()
            if summer_only:
                df_annual = df_station[df_station.index.month.isin([6, 7, 8])]
                df_annual = df_annual.groupby(df_annual.index.year).mean()

            ss_hcho = df_ss.hcho
            ann_hcho = df_annual.hcho
            if norm:
                ss_hcho = min_max_normalize(ss_hcho)
                ann_hcho = min_max_normalize(ann_hcho)

            ax_ss[j].plot(
                df_ss.index, ss_hcho, label=f"{c}", color=colors[i], marker="o"
            )
            ax_ss[j].set_title(f"{station} - (Monthly Mean)")

            ax_annual[j].plot(
                df_annual.index, ann_hcho, label=f"{c}", color=colors[i], marker="o"
            )
            ax_annual[j].set_title(f"{station} - (Annual Mean)")

            ax_annual[j].set_xlim(2012, 2020)

    for fig in [f_ss, f_annual]:
        handles, labels = ax_annual[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
        )


def plt_reg(hcho, norm=False, unit=None, sslat=False):

    interested_case = list(hcho.keys())
    interested_case = [c for c in interested_case if c != "BVOCoff20012023_nudg"]

    ylim_dict = {
        # "SAF": (0, 15),
        # "MED": (0, 15),
        # "CEU": (0, 15),
        # "SAS": (0, 15),
        "AMZ": (0, 25),
        "ENA": (0, 20),
        "EAS": (0, 20),
        "SEA": (0, 15),
        "NAU": (0, 15),
        "Indonesia": (0, 15),
        "C_Africa": (0, 25),
        "N_Africa": (0, 25),
        "S_Africa": (0, 25),
        "REMOTE_PACIFIC": (0, 4),
    }

    for mode in ["ss", "ann"]:
        index = "month" if mode == "ss" else "year"

        fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

        for i, r in enumerate(ROIS):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]
            for j, c in enumerate(interested_case):
                df = hcho[c].reg_ss
                if mode == "ann":
                    df = hcho[c].reg_ann
                    if sslat:
                        df = hcho[c].reg_ann_sslat
                reg_df = (
                    df[[index, r]]
                    .set_index(index)
                    .rename(columns={r: c.replace("20012023_nudg", "")})
                )
                if norm:
                    reg_df[c] = min_max_normalize(reg_df[c])
                sns.lineplot(reg_df, ax=ax, palette=[colors[j]], markers=True, lw=2)
            # Set y-axis limit
            # if r in ylim_dict:
            #     ax.set_ylim(ylim_dict[r])

            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel("Year")
            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            if ri < 2:
                ax.set_xlabel("")

            if unit is None:
                unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"
            ax.set_ylabel(unit)

            ax.set_title(f"{r}")

        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )


# Plot regional contributions of BVOC emissions to HCHO
def plt_reg_bvoc_contri(hcho, norm=True, unit=None, sslat=True):
    list_regions = [
        "Amazonia",
        "S-E US",
        "Mato Grosso",
        "Indonesia",
        "South China",
        "C_Africa",
        "N_Africa",
        "S_Africa",
        "NAU",
    ]
    interested_case = list(hcho.keys())

    for mode in ["ss", "ann"]:
        index = "month" if mode == "ss" else "year"
        fig, axis = plt.subplots(3, 3, figsize=(9, 9), layout="constrained")

        for i, r in enumerate(list_regions):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]

            for j, case in enumerate(interested_case):
                if case == OFF_CASE:
                    continue
                # Select time period
                ds = (
                    hcho[case].reg_ss
                    if mode == "ss"
                    else (hcho[case].reg_ann_sslat if sslat else hcho[case].reg_ann)
                )
                reg_df = ds[[index, r]].set_index(index).rename(columns={r: case})

                # Subtract BVOCoff (if applicable)
                if case not in SAT_CASE:
                    ds_off = (
                        hcho[OFF_CASE].reg_ss
                        if mode == "ss"
                        else (
                            hcho[OFF_CASE].reg_ann_sslat
                            if sslat
                            else hcho[OFF_CASE].reg_ann
                        )
                    )
                    reg_off = (
                        ds_off[[index, r]].set_index(index).rename(columns={r: case})
                    )
                    reg_df[case] = reg_df[case] - reg_off[case]

                if norm:
                    reg_df[case] = min_max_normalize(reg_df[case])

                sns.lineplot(
                    x=reg_df.index,
                    y=reg_df[case],
                    ax=ax,
                    label=case.replace("20012023_nudg", ""),
                    color=colors[j],
                    lw=2,
                    marker="o",
                )

            # Axis settings
            if unit is None:
                unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"
            ax.set_ylabel(unit)
            ax.set_title(f"{r}")
            ax.get_legend().remove()

            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            else:
                ax.set_xlabel("Year")

            if ri < 2:
                ax.set_xlabel("")

            # Optional: set y-axis limits if needed
            # if r in ylim_dict:
            #     ax.set_ylim(ylim_dict[r])

        # Global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )


def plt_metric_reg(hcho_dict, bvoc_contri=False):

    list_case = list(hcho_dict.keys())
    obs_c = get_sat_case(list_case)

    list_reg = ROIS
    R = {"model": [c for c in list_case if c not in [obs_c, OFF_CASE]]}
    rmse = {"model": [c for c in list_case if c not in [obs_c, OFF_CASE]]}
    trend = {"model": [c for c in list_case if c not in [OFF_CASE]]}

    for reg in list_reg:
        R[reg] = []
        rmse[reg] = []
        for c in list_case:
            if c not in [obs_c, OFF_CASE]:
                model_reg = hcho_dict[c].reg_ann[reg]
                obs_reg = hcho_dict[obs_c].reg_ann[reg]
                if bvoc_contri:
                    model_reg = model_reg - hcho_dict[OFF_CASE].reg_ann[reg]

                R[reg].append(pearsonr(model_reg, obs_reg)[0])
                rmse[reg].append(np.sqrt(mse(obs_reg, model_reg)))

        trend[reg] = []
        trend[f"{reg}_sig"] = []
        for c in list_case:
            if c != OFF_CASE:
                hcho_reg = hcho_dict[c].reg_ann[reg]
                if bvoc_contri:
                    hcho_reg = hcho_reg - hcho_dict[OFF_CASE].reg_ann[reg]

                trend_dict = pymk.original_test(hcho_reg, alpha=0.05)
                trend[reg].append(trend_dict.slope)
                trend[f"{reg}_sig"].append(0 if not trend_dict.h else 1)

    rmse = pd.DataFrame.from_dict(rmse).round(2)
    R = pd.DataFrame.from_dict(R).round(2)
    trend = pd.DataFrame.from_dict(trend).round(2)

    tits = ["RMSE", "R", "Linear Trend"]
    for i, ds in enumerate([rmse, R, trend]):
        ds["model"] = ds["model"].apply(lambda x: x.replace("20012023_nudg", ""))
        ds.set_index("model", inplace=True)
        ds.index.name = "Model"
        ds = ds[[col for col in ds.columns if "sig" not in col]]

        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(ds.T, annot=True, cmap="YlGnBu")
        plt.title(tits[i], fontsize=16)
        plt.xticks(rotation=15)

    return rmse, R, trend


def plt_mean_ann(hcho, sslat=False):

    cases = list(hcho.keys())
    obs_c = get_sat_case(cases)

    rows, cols = 4, 2
    fig, axis = plt.subplots(
        rows,
        cols,
        figsize=(8 * 2, 5 * 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"

    cases.remove(obs_c)
    cases.remove(OFF_CASE)
    cases.insert(0, obs_c)
    cases.insert(2, "VISITst20012023_nudg")

    print(cases)

    mappable_group1 = None  # for j < 2
    mappable_group2 = None  # for j > 1

    obs_ds = hcho[obs_c].hcho.mean("time", skipna=True)
    if sslat:
        obs_ds = hcho[obs_c].hcho_sslat.mean("year", skipna=True)

    for j, c in enumerate(cases):

        ds = hcho[c].hcho.mean("time", skipna=True)
        if sslat:
            ds = hcho[c].hcho_sslat.mean("year", skipna=True)

        if j > 1:
            ds = (ds - obs_ds) * 100 / obs_ds

        ri, ci = j // cols, j % cols

        ax = axis[ri, ci]

        im = ds.plot(
            ax=ax,
            cmap="Spectral_r" if j < 2 else "bwr",
            add_colorbar=False,
            vmin=0 if j < 2 else -100,
            vmax=15 if j < 2 else 100,
        )
        if j < 2 and mappable_group1 is None:
            mappable_group1 = im
        if j > 1 and mappable_group2 is None:
            mappable_group2 = im

        replace_str = "" if j < 2 else " (OBS subtracted)"
        tit = c.replace("20012023_nudg", replace_str)
        ax.set_title(tit, fontsize=18)
        ax.coastlines()
        ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
    # Turn off unused axes
    cbar_ax1 = fig.add_axes([0.25, 0.77, 0.5, 0.01])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(mappable_group1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label(unit, fontsize=16)
    cbar1.ax.tick_params(labelsize=14)

    # Add colorbar for j > 1 (anomalies)
    cbar_ax2 = fig.add_axes([0.25, 0.02, 0.5, 0.01])
    cbar2 = fig.colorbar(mappable_group2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label("% Difference from OBS", fontsize=16)
    cbar2.ax.tick_params(labelsize=14)


def plt_map(hcho_dict, score="corr", sslat=False):
    fig, axis = plt.subplots(
        3,
        2,
        figsize=(8 * 2, 5 * 3),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    interested_case = list(hcho_dict.keys())
    obs_c = get_sat_case(interested_case)
    obs_ds = hcho_dict[obs_c].hcho
    obs_ds = obs_ds.groupby(obs_ds.time.dt.year).mean("time")

    if sslat:
        obs_ds = hcho_dict[obs_c].hcho_sslat

    for j, c in enumerate(interested_case):
        if c not in [obs_c, "BVOCoff20012023_nudg"]:
            ri, ci = j // 2, j % 2
            ax = axis[ri, ci]
            add_colorbar = False

            model_ds = hcho_dict[c].hcho
            model_ds = model_ds.groupby(model_ds.time.dt.year).mean("time")
            if sslat:
                model_ds = hcho_dict[c].hcho_sslat

            if score == "corr":
                corr, sig = map_corr_by_time(model_ds, obs_ds)

                cb = corr.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    add_colorbar=add_colorbar,
                )

                # Stippling
                lat, lon = np.meshgrid(corr["lat"], corr["lon"], indexing="ij")
                ax.plot(
                    lon[sig.values],
                    lat[sig.values],
                    "k.",
                    markersize=0.5,
                    transform=ccrs.PlateCarree(),
                )

            elif score == "rmse":
                rmse = compute_rmse(model_ds, obs_ds)

                cb = rmse.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap="rainbow",
                    vmin=0,
                    vmax=100,
                    add_colorbar=add_colorbar,
                )

            # Colorbar
            ax.coastlines()
            ax.set_title(c.replace("20012023_nudg", ""), fontsize=18)
    # Add a single shared colorbar at bottom center
    label = "Pearson R" if score == "corr" else "Normalized RMSE (%)"
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.01])  # [left, bottom, width, height]
    cbar = fig.colorbar(cb, cax=cbar_ax, orientation="horizontal", label=label)
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)


def plt_mk(hcho, sslat=False):
    fig, axis = plt.subplots(
        4,
        2,
        figsize=(8 * 2, 5 * 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    interested_case = list(hcho.keys())

    for j, c in enumerate(interested_case):
        if c != "BVOCoff20012023_nudg":

            mk_ds = hcho[c].hcho_ann_mk.tcolhcho_ann_trend
            if sslat:
                mk_ds = hcho[c].hcho_ann_sslat_mk.tcolhcho_ann_trend

            ri, ci = j // 2, j % 2
            ax = axis[ri, ci]

            cb = mk_ds.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                vmin=-0.2,
                vmax=0.2,
                add_colorbar=False,
            )
            # Colorbar
            ax.coastlines()
            ax.set_title(c.replace("20012023_nudg", ""), fontsize=18)

    label = "MK trend"
    cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.01])  # [left, bottom, width, height]
    cbar = fig.colorbar(cb, cax=cbar_ax, orientation="horizontal", label=label)
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    axes = axis.ravel() if hasattr(axis, "ravel") else [axis]

    for ax in axes:
        if not ax.has_data():  # Checks if anything was plotted
            ax.set_visible(False)


def plt_ann_glob():
    fig, axis = plt.subplots(1, 1, figsize=(4.5, 4), layout="constrained")
    dfs = [hcho.inhi_chaser_glob, hcho.tropomi_glob, hcho.no_inhi_chaser_glob]
    names = ["Inhi-CHASER", "TROPOMI", "No-Inhi-CHASER"]

    for i, df in enumerate(dfs):
        df = df.set_index("year")
        rename_df = df.rename(columns={"avg_glob_ann": names[i]})
        sns.lineplot(
            rename_df,
            ax=axis,
            palette=[colors[i]],
            markers=True,
        )
    axis.set_ylim(3.5, 5.5)
    axis.set_ylabel("(\u00d710$^{15}$ molec.cm$^{-2}$)")

    hcho_inhi_chaser = hcho.inhi_chaser_glob["avg_glob_ann"].values
    hcho_no_inhi_chaser = hcho.no_inhi_chaser_glob["avg_glob_ann"].values

    hcho_tropomi = hcho.tropomi_glob["avg_glob_ann"].values
    res_tropomi = pymk.original_test(hcho_tropomi, alpha=0.05)
    trend_tropomi = round(res_tropomi.slope, 2)
    trend_tropomi = trend_tropomi if not res_tropomi.h else f"{trend_tropomi}*"

    print(f"trend_tropomi:{trend_tropomi:.2f}")
    notes = ["inhi", "noinhi"]
    for i, hcho_chaser in enumerate([hcho_inhi_chaser, hcho_no_inhi_chaser]):

        res_chaser = pymk.original_test(hcho_chaser, alpha=0.05)

        trend_chaser = round(res_chaser.slope, 2)

        trend_chaser = trend_chaser if not res_chaser.h else f"{trend_chaser}*"

        # axis.text(2020, 3.7, trend_chaser, fontsize=12, color=colors[0])
        # axis.text(2021, 3.7, trend_tropomi, fontsize=12, color=colors[1])

        pearson_r, _ = pearsonr(hcho_chaser, hcho_tropomi)
        rmse = np.sqrt(mse(hcho_chaser, hcho_tropomi))

        axis.set_title(f" Global Mean Annual HCHO")
        print(notes[i])
        print(f"R:{pearson_r:.2f}")
        print(f"RMSE:{rmse:.2f}")
        print(f"trend chaser:{trend_chaser:.2f}")


def plt_check_AK():
    sat_dir = f"/mnt/dg3/ngoc/obs_data"
    time_omi = "20050101-20231201"
    time_tropo = "20180601-20240701"

    m_name_omi = "mon_BIRA_OMI_HCHO_L3"
    m_name_tropo = "mon_TROPOMI_HCHO_L3"
    omi_file = f"{sat_dir}/{m_name_omi}/EXTRACT/hcho_AERmon_{m_name_omi}_historical_gn_{time_omi}.nc"
    tropo_file = f"{sat_dir}/{m_name_tropo}/EXTRACT/hcho_AERmon_{m_name_tropo}_historical_gn_{time_tropo}.nc"

    tropo_ds = xr.open_dataset(tropo_file)
    tropo_ds = tropo_ds.sel(time=(tropo_ds.time.dt.year == 2019))
    tropo_pressure = extract_layer_sat_ps(tropo_ds)

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
    # list_df = []
    for i, m in enumerate(months):
        ak_hoque_df = pd.read_csv(
            f"/mnt/dg2/hoque/TROPOMI_model_analysis/2019/analysis/AK_{m}.txt",
            delimiter="\t",
            names=["Date", "Lat", "Lon", "AK", "Pressure_grid", "Level"],
            header=None,
        )

        fig, axis = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3), layout="constrained")
        month = i + 1
        tropo_ds.Sat_center_ps.sel(time=(tropo_ds.time.dt.month == month)).plot(
            ax=axis[0]
        )
        ak_hoque_df["Pressure_grid"].plot.hist(ax=axis[1])


def plt_check_interpolated_AK():
    ak_dir = "/mnt/dg2/hoque/TROPOMI_model_analysis/2019/analysis/"
    m = "jul"
    df_jul = pd.read_csv(
        f"{ak_dir}/AK_{m}.txt",
        delimiter="\t",
        header=None,
        names=["time", "lat", "lon", "AK", "pressure", "level"],
    )
    times = df_jul.time.unique()
    levels = df_jul.level.unique()
    df_jul = df_jul[df_jul["time"] == times[0]]
    df_jul = df_jul[df_jul["level"] == levels[0]]
    print(times[0], levels[0])

    lat_hoque = df_jul.lat.values
    lon_hoque = df_jul.lon.values
    ak_hoque = df_jul.AK.values
    ps_hoque = df_jul.pressure.values
    # hoque_ak_ps = df_jul.groupby(["time", "lat", "lon", "level"]).mean().to_xarray()
    pa_ak_ps = xr.open_dataset(
        "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/TROPOMI_AK_2019_combined.nc"
    )

    pa_ak_ps_jul = pa_ak_ps.sel(time="2019-07-01", level=0)
    # pa_ak_ps_jul = pa_ak_ps_jul.sortby("lat")

    # hoque code test
    clat = pa_ak_ps.lat.values
    clon = pa_ak_ps.lon.values
    xmax1 = np.linspace(
        clat.min(),
        clat.max(),
        int((clat.max() - clat.min()) // 2.75),
    )
    ymax1 = np.linspace(
        clon.min(),
        clon.max(),
        int((clon.max() - clon.min()) // 2.75),
    )
    # Yi, Xi = np.meshgrid(ymax1, xmax1)
    Yi, Xi = np.meshgrid(clon, clat)
    points = []
    for m in range(len(lat_hoque)):
        points.append([lon_hoque[m], lat_hoque[m]])

    interp_ak = scipy.interpolate.griddata(points, ak_hoque, (Yi, Xi), method="nearest")
    interp_ps = scipy.interpolate.griddata(points, ps_hoque, (Yi, Xi), method="nearest")
    hoque_ak_ps_jul = xr.Dataset(
        {
            "AK": (("lat", "lon"), interp_ak),
            "pressure": (("lat", "lon"), interp_ps),
        },
        coords={
            "lat": (("lat"), clat),
            "lon": (("lon"), clon),
        },
    )

    diff_ak = (hoque_ak_ps_jul.AK - pa_ak_ps_jul.AK) * 100 / hoque_ak_ps_jul.AK
    diff_ps = (
        (hoque_ak_ps_jul.pressure - pa_ak_ps_jul.pressure)
        * 100
        / hoque_ak_ps_jul.pressure
    )


def plt_check_glob_AK_OMI(mode="profile"):
    SAT_DIR = f"/mnt/dg3/ngoc/obs_data"
    TIME_OMI = "20050101-20231201"
    TIME_TROPO = "20180601-20240701"
    M_NAME_OMI = "mon_BIRA_OMI_HCHO_L3"
    M_NAME_TROPO = "mon_TROPOMI_HCHO_L3"

    OMI_FILE = f"{SAT_DIR}/{M_NAME_OMI}/EXTRACT/hcho_AERmon_{M_NAME_OMI}_historical_gn_{TIME_OMI}.nc"
    ds_omi = xr.open_dataset(OMI_FILE)
    ak = ds_omi["AK"]

    if mode != "profile":
        # Step 1: Compute the global mean over lat, lon, and layer
        ak_global_mean = ak.mean(dim=["lat", "lon", "layer"])

        # Step 2: Group by year and take the mean across each year
        ak_annual_mean = ak_global_mean.groupby("time.year").mean(dim="time")

        # Extract data for plotting
        years = ak_annual_mean.year.values.astype(int)
        values = ak_annual_mean.values

        # Step 3: Plot with integer x-axis
        plt.figure(figsize=(8, 5))
        plt.plot(years, values, marker="o", linewidth=2)
        plt.title("Global Mean of AK by Year - OMI")
        plt.xlabel("Year")
        plt.ylabel("AK (global mean)")
        plt.xticks(ticks=years, labels=years, rotation=45)
    else:
        ak_global_mean_by_layers = ak.mean(dim=["lat", "lon"])
        ak_annual_mean_by_layer = ak_global_mean_by_layers.groupby("time.year").mean(
            dim="time"
        )
        years = ak_annual_mean_by_layer.year.values
        n_years = len(years)
        colors = cm.viridis(
            np.linspace(0, 1, n_years)
        )  # choose colormap (e.g., viridis)

        plt.figure(figsize=(8, 7))
        for i, y in enumerate(years):
            plt.plot(
                ak_annual_mean_by_layer.sel(year=y),
                ak_annual_mean_by_layer.layer,
                color=colors[i],
                linewidth=2,
            )

        plt.xlabel("AK")
        plt.ylabel("Layer")
        plt.title("AK Vertical Profile of OMI (2005–2023)")

        # Add colorbar to show which color corresponds to which year
        sm = plt.cm.ScalarMappable(
            cmap=cm.viridis, norm=plt.Normalize(vmin=years.min(), vmax=years.max())
        )
        cbar = plt.colorbar(sm, label="Year")
        plt.grid(True)


def plt_ch4():
    obj = {"VISITst20012023_nudg": CH4()}
    plt_reg(obj, obj)


# %%
