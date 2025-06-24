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

# AKED_CHASER_HCHO SETTING
ORG_CHASER_DIR = "/mnt/dg3/ngoc/CHASER_output"
AKED_DIR = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"

SAT_NAMES = ["TROPO", "OMI"]
# CASES = [f.split("/")[-1] for f in glob(f"{ORG_CHASER_DIR}/*2023_*")]
CASES = [
    "VISITst20012023_nudg",
    # "UKpft20012023_nudg",
    # "MEGANst20012023_nudg",
    # "MEGANpft20012023_nudg",
    # "UKst20012023_nudg",
    # "MIXpft20012023_nudg",
    # "BVOCoff20012023_nudg",
    "OBS",
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


def list_all_files(base_dir, sat_ver):
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file = os.path.join(root, file)
            for case in CASES:
                if case in file:
                    if sat_ver in file and "ak_assign" in file:
                        all_files.append(file)
    return all_files


def load_hcho(sat_name, sat_ver):

    # SATELLITE SETTING
    SAT_DIR = f"/mnt/dg3/ngoc/obs_data"
    if sat_ver == "v1":
        time_omi = "20050101-20231201"
        time_tropo = "20180601-20240701"
    else:
        time_omi = "20050101-20221231"
        time_tropo = "20180507-20231231"

    M_NAME_OMI = f"mon_BIRA_OMI_HCHO_L3_{sat_ver}"
    M_NAME_TROPO = f"mon_TROPOMI_HCHO_L3_{sat_ver}"

    OMI_FILE = f"{SAT_DIR}/{M_NAME_OMI}/EXTRACT/hcho_AERmon_{M_NAME_OMI}_historical_gn_{time_omi}.nc"
    TROPO_FILE = f"{SAT_DIR}/{M_NAME_TROPO}/EXTRACT/hcho_AERmon_{M_NAME_TROPO}_historical_gn_{time_tropo}.nc"

    base_dir = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied/"
    files = list_all_files(base_dir, sat_ver)
    print(files)

    obs = HCHO(TROPO_FILE) if sat_name == "tropo" else HCHO(OMI_FILE)

    ak_files = [f for f in files if sat_name in f]
    interp_files = [f for f in ak_files if "/sat_interp/" in f]
    nointerp_files = [f for f in ak_files if "/no_sat_interp/" in f]

    hcho_interp = {Path(f).parents[1].name: HCHO(f) for f in interp_files}
    hcho_nointerp = {Path(f).parents[1].name: HCHO(f) for f in nointerp_files}
    hcho_interp["OBS"] = obs
    hcho_nointerp["OBS"] = obs

    return hcho_interp, hcho_nointerp


# interp_omi_v1, nointerp_omi_v1 = load_hcho("omi", "v1")
# interp_omi_v2, nointerp_omi_v2 = load_hcho("omi", "v2")
# interp_tropo_v1, nointerp_tropo_v1 = load_hcho("tropo", "v1")
# interp_tropo_v2, nointerp_tropo_v2 = load_hcho("tropo", "v2")


# %%
def plt_reg(hcho_interp, hcho_nointerp, tits=None):

    list_regions = [
        "AMZ",
        "ENA",
        # "SAF",
        # "MED",
        # "CEU",
        "EAS",
        # "SAS",
        "SEA",
        "NAU",
        "Indonesia",
        "C_Africa",
        "N_Africa",
        "S_Africa",
        # "REMOTE_PACIFIC",
    ]
    if tits is None:
        tits = ["Temp/HCHO Interpolated by Satellite Pressure", "No Interpolation"]

    # interested_case = list(hcho_interp.keys())[::-1]
    # interested_case = [
    #     "OBS",
    #     # "VISIT20172023_no_nudg",
    #     "VISITst20012023_nudg",
    #     "XhalfVISITst20012023",
    #     "XhalfUKpft20012023",
    # ]
    # interested_case = list(hcho_interp.keys())
    interested_case = CASES

    # ylim_dict = {
    #     "AMZ": (0, 20),
    #     "ENA": (0, 15),
    #     # "SAF": (0, 15),
    #     # "MED": (0, 15),
    #     # "CEU": (0, 15),
    #     "EAS": (0, 15),
    #     # "SAS": (0, 15),
    #     "SEA": (0, 15),
    #     "NAU": (0, 15),
    #     "Indonesia": (0, 15),
    #     "C_Africa": (0, 15),
    #     "N_Africa": (0, 15),
    #     "S_Africa": (0, 15),
    #     "REMOTE_PACIFIC": (0, 4),
    # }

    for mode in ["ss", "ann"]:
        index = "month" if mode == "ss" else "year"
        for k, hcho in enumerate([hcho_interp, hcho_nointerp]):
            fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

            for i, r in enumerate(list_regions):
                ri, ci = i // 3, i % 3
                ax = axis[ri, ci]
                # for j, c in enumerate(list(hcho_interp.keys())[::-1]):
                for j, c in enumerate(interested_case):
                    ds = hcho[c].reg_ss if mode == "ss" else hcho[c].reg_ann
                    reg_df = ds[[index, r]].set_index(index).rename(columns={r: c})
                    sns.lineplot(
                        reg_df,
                        ax=ax,
                        palette=[colors[j]],
                        markers=True,
                        lw=2,
                    )
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
                ax.set_ylabel("(\u00d710$^{15}$ molec.cm$^{-2}$)")
                ax.set_title(f"{r}")

            fig.legend(
                handles,
                labels,
                ncol=4,
                loc="center",
                bbox_to_anchor=(0.5, -0.06),
            )
            plt.suptitle(tits[k], fontsize=16, fontweight="bold")


def plt_metric_reg(hcho_dict):

    list_case = list(hcho_dict.keys())
    list_reg = list(hcho_dict[list_case[0]].reg_ann.keys())
    R = {"model": list_case[:-1]}
    rmse = {"model": list_case[:-1]}
    trend = {"model": list_case}
    obs_c = list_case[-1]

    for reg in list_reg:
        if reg != "REMOTE_PACIFIC":
            R[reg] = []
            rmse[reg] = []
            for c in list_case[:-1]:

                model_reg = hcho_dict[c].reg_ann[reg]
                obs_reg = hcho_dict[obs_c].reg_ann[reg]

                # print(reg, c)
                R[reg].append(pearsonr(model_reg, obs_reg)[0])
                rmse[reg].append(np.sqrt(mse(obs_reg, model_reg)))

            trend[reg] = []
            trend[f"{reg}_sig"] = []
            for c in list_case:
                hcho_reg = hcho_dict[c].reg_ann[reg]
                trend_dict = pymk.original_test(hcho_reg, alpha=0.05)
                trend[reg].append(trend_dict.slope)
                trend[f"{reg}_sig"].append(0 if not trend_dict.h else 1)

    rmse = pd.DataFrame.from_dict(rmse).round(2)
    R = pd.DataFrame.from_dict(R).round(2)
    trend = pd.DataFrame.from_dict(trend).round(2)
    return rmse, R, trend


def plt_map_mean_ss(hcho_interp, hcho_nointerp, sat_name):
    ss_months = [[12, 1, 2], [6, 7, 8]]
    seasons = ["DJF", "JJA"]
    # plotting vars
    cmaps = "bwr"
    vmins = -100
    vmaxs = 100
    unit = "[%]"
    levels = 7

    data_tits = ["Sat_Interp", "No_Interp"]
    for k, hcho in enumerate([hcho_interp, hcho_nointerp]):
        obs = hcho["OBS"].hcho

        for s, months in enumerate(ss_months):
            cases = list(hcho_interp.keys())[::-1][1:]
            rows, cols = len(cases) // 2, 2
            fig, axis = plt.subplots(
                rows,
                cols,
                figsize=(4 * 2, 2.5 * 3),
                layout="constrained",
                subplot_kw=dict(projection=ccrs.PlateCarree()),
            )

            ss_obs = obs.sel(time=(obs.time.dt.month.isin(months))).mean(
                "time", skipna=True
            )

            for j, c in enumerate(cases):
                ri, ci = j // cols, j % cols
                ax = axis[ri, ci]

                chaser = hcho[c].hcho
                ss_chaser = chaser.sel(time=(chaser.time.dt.month.isin(months))).mean(
                    "time", skipna=True
                )

                ss_diff = (ss_chaser - ss_obs) * 1e2 / ss_obs

                add_colorbar = True if ri > 1 else False
                cbar_kwargs = (
                    {
                        "orientation": "horizontal",
                        "shrink": 0.8,
                        "label": unit,
                    }
                    if add_colorbar
                    else {}
                )

                ss_diff.plot(
                    ax=ax,
                    cmap=cmaps,
                    levels=levels,
                    vmin=vmins,
                    vmax=vmaxs,
                    add_colorbar=add_colorbar,
                    cbar_kwargs=cbar_kwargs,
                )

                ax.set_title(c, fontsize=14)
                ax.coastlines()
                ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
                ax.set_xticks([])
                ax.set_xlabel("")
                ax.set_yticks([])
                ax.set_ylabel("")
            plt.suptitle(
                f"{data_tits[k]}_CHASER-{sat_name} ({seasons[s]})",
                fontsize=16,
                fontweight="bold",
            )


def plt_map_corr(hcho_interp, hcho_nointerp, sat_name):
    data_tits = ["Sat_Interp", "No_Interp"]
    modes = ["ss", "ann"]
    tits = ["Seasonal", "Interannual"]
    cmaps = "Blues"
    levels = 11
    vmins = 0
    vmaxs = 1
    unit = "Pearson R"
    for k, hcho in enumerate([hcho_interp, hcho_nointerp]):
        obs = hcho["OBS"].hcho
        cases = list(hcho_interp.keys())[::-1][1:]
        for i, m in enumerate(modes):
            rows, cols = len(cases) // 2, 2
            fig, axis = plt.subplots(
                rows,
                cols,
                figsize=(4 * 2, 2.5 * 3),
                layout="constrained",
                subplot_kw=dict(projection=ccrs.PlateCarree()),
            )
            for j, c in enumerate(cases):
                ri, ci = j // cols, j % cols
                ax = axis[ri, ci]

                chaser = hcho[c].hcho
                corr = map_corr_by_time(chaser, obs, m)
                add_colorbar = True if ri > 1 else False
                cbar_kwargs = (
                    {
                        "orientation": "horizontal",
                        "shrink": 0.8,
                        "label": unit,
                    }
                    if add_colorbar
                    else {}
                )
                corr.plot(
                    ax=ax,
                    cmap=cmaps,
                    levels=levels,
                    vmin=vmins,
                    vmax=vmaxs,
                    add_colorbar=add_colorbar,
                    cbar_kwargs=cbar_kwargs,
                )
                ax.set_title(c, fontsize=14)
                ax.coastlines()
                ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
                ax.set_xticks([])
                ax.set_xlabel("")
                ax.set_yticks([])
                ax.set_ylabel("")
            plt.suptitle(
                f"{tits[i]} Corr. {data_tits[k]}_CHASER-{sat_name}",
                fontsize=16,
                fontweight="bold",
            )


def old_plt_map_mean_year():
    fig, axis = plt.subplots(
        4,
        5,
        figsize=(3.75 * 5, 2 * 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    for j, year in enumerate(np.arange(2020, 2024)):

        tropomi_y = tropomi.sel(time=(tropomi.time.dt.year == year)).mean("time")
        inhi_chaser_y = inhi_chaser.sel(time=(inhi_chaser.time.dt.year == year)).mean(
            "time"
        )
        no_inhi_chaser_y = no_inhi_chaser.sel(
            time=(no_inhi_chaser.time.dt.year == year)
        ).mean("time")

        diff_inhi_year = (inhi_chaser_y - tropomi_y) * 1e2 / tropomi_y
        diff_no_inhi_year = (no_inhi_chaser_y - tropomi_y) * 1e2 / tropomi_y

        titles = [
            f"TROPOMI ({year})",
            f"Inhi-CHASER ({year})",
            f"NoInhi-CHASER ({year})",
            f"Inhi-CHASER-TROPOMI ({year})",
            f"NoInhi-CHASER-TROPOMI ({year})",
        ]
        cmaps = ["rainbow", "rainbow", "rainbow", "bwr", "bwr"]
        vmins = [0, 0, 0, -150, -150]
        vmaxs = [20, 20, 20, 150, 150]
        add_colorbar = True if j == 3 else False

        for i, ds in enumerate(
            [
                tropomi_y,
                inhi_chaser_y,
                no_inhi_chaser_y,
                diff_inhi_year,
                diff_no_inhi_year,
            ]
        ):

            ci = i
            ax = axis[j, ci]

            unit = (
                "HCHO total col. (\u00d710$^{15}$ molec.cm$^{-2}$)" if ci < 3 else "[%]"
            )
            cbar_kwargs = (
                {
                    "orientation": "horizontal",
                    "shrink": 0.6,
                    "label": unit,
                }
                if add_colorbar
                else {}
            )
            levels = 21 if i < 3 else 5

            ds.plot(
                ax=ax,
                cmap=cmaps[ci],
                levels=levels,
                vmin=vmins[ci],
                vmax=vmaxs[ci],
                add_colorbar=add_colorbar,
                cbar_kwargs=cbar_kwargs,
            )

            ax.set_title(titles[i], fontsize=14)
            ax.coastlines()
            ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.set_ylabel("")


def plt_chaser_tropo_hcho_corr():
    fig, axis = plt.subplots(
        2,
        2,
        figsize=(4 * 2, 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    cmap = "coolwarm"
    hcho.tropmomi_inhi_chaser_ss_corr.plot(
        ax=axis[0][0], cmap=cmap, vmin=-1, vmax=1, add_colorbar=False
    )
    hcho.tropmomi_inhi_chaser_ann_corr.plot(
        ax=axis[0][1], cmap=cmap, vmin=-1, vmax=1, add_colorbar=False
    )

    hcho.tropmomi_no_inhi_chaser_ss_corr.plot(
        ax=axis[1][0],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.6,
            "label": "Pearson R",
        },
    )
    hcho.tropmomi_no_inhi_chaser_ann_corr.plot(
        ax=axis[1][1],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.6,
            "label": "Pearson R",
        },
    )

    axis[0][0].set_title("Inhi Seasonal Corr.", fontsize=14)
    axis[0][1].set_title("Inhi Annual Corr.", fontsize=14)
    axis[1][0].set_title("NoInhi Seasonal Corr.", fontsize=14)
    axis[1][1].set_title("NoInhi Annual Corr.", fontsize=14)

    for i in [0, 1]:
        for j in [0, 1]:
            axis[i][j].coastlines()
            axis[i][j].set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
            axis[i][j].set_xticks([])
            axis[i][j].set_xlabel("")
            axis[i][j].set_yticks([])
            axis[i][j].set_ylabel("")


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


def old_plt_reg(hcho, ax, case_name, mode="ss"):
    list_regions = ["AMZ", "ENA", "SAF", "MED", "CEU", "EAS", "SAS", "SEA", "NAU"]

    if mode == "ann":
        chaser_reg = hcho.no_inhi_chaser_reg_ann
        tropomi_reg = hcho.tropomi_reg_ann
        index = "year"
    else:
        chaser_reg = hcho.inhi_chaser_reg_ss
        tropomi_reg = hcho.tropomi_reg_ss
        index = "month"

    fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")
    df_stats = {
        "reg": list_regions,
        "r_inhi": [],
        "r_no_inhi": [],
        "rmse_inhi": [],
        "rmse_no_inhi": [],
    }

    for i, reg in enumerate(list_regions):
        ri, ci = i // 3, i % 3
        ax = axis[ri, ci]

        reg_df_tropomi = (
            tropomi_reg[[index, reg]].set_index(index).rename(columns={reg: "TROPOMI"})
        )

        # legend = True if ri == 0 and ci == 2 else False
        # legend = False
        sns.lineplot(
            reg_df_tropomi,
            ax=ax,
            palette=[colors[1]],
            markers=True,
            lw=2,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        # ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.set_ylim(2, 12)
        if ri == 0 and ci == 0:
            ax.set_ylim(2, 20)

        ax.set_xlabel("Year")
        if mode == "ss":
            ax.set_xlabel("Month")
            ax.set_xticks(np.arange(1, 13))
        if ri < 2:
            ax.set_xlabel("")
        ax.set_ylabel("(\u00d710$^{15}$ molec.cm$^{-2}$)")

        hcho_inhi_chaser = reg_df_inhi["Inhi-CHASER"].values
        hcho_no_inhi_chaser = reg_df_no_inhi["NoInhi-CHASER"].values
        hcho_tropomi = reg_df_tropomi["TROPOMI"].values

        r_inhi, _ = pearsonr(hcho_inhi_chaser, hcho_tropomi)
        r_no_inhi, _ = pearsonr(hcho_no_inhi_chaser, hcho_tropomi)

        rmse_inhi = np.sqrt(mse(hcho_tropomi, hcho_inhi_chaser))
        rmse_no_inhi = np.sqrt(mse(hcho_tropomi, hcho_no_inhi_chaser))

        df_stats["r_inhi"].append(f"{r_inhi:.2f}")
        df_stats["r_no_inhi"].append(f"{r_no_inhi:.2f}")
        df_stats["rmse_inhi"].append(f"{rmse_inhi:.2f}")
        df_stats["rmse_no_inhi"].append(f"{rmse_no_inhi:.2f}")

        if mode == "ann":

            res_inhi_chaser = pymk.original_test(hcho_inhi_chaser, alpha=0.05)
            res_no_inhi_chaser = pymk.original_test(hcho_no_inhi_chaser, alpha=0.05)
            res_tropomi = pymk.original_test(hcho_tropomi, alpha=0.05)

            trend_inhi_chaser = round(res_inhi_chaser.slope, 2)
            trend_no_inhi_chaser = round(res_no_inhi_chaser.slope, 2)
            trend_tropomi = round(res_tropomi.slope, 2)

            trend_inhi_chaser = (
                trend_inhi_chaser if not res_inhi_chaser.h else f"{trend_inhi_chaser}*"
            )
            trend_no_inhi_chaser = (
                trend_no_inhi_chaser
                if not res_no_inhi_chaser.h
                else f"{trend_no_inhi_chaser}*"
            )
            trend_tropomi = trend_tropomi if not res_tropomi.h else f"{trend_tropomi}*"

            ax.text(2020, 3, trend_inhi_chaser, fontsize=12, color=colors[0])
            ax.text(2021, 3, trend_no_inhi_chaser, fontsize=12, color=colors[2])
            ax.text(2022, 3, trend_tropomi, fontsize=12, color=colors[1])

        ax.set_title(f"{reg}")

    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
    )

    return pd.DataFrame.from_dict(df_stats)


def plt_sudo(hcho):
    # cases = list(hcho.keys())[::-1][:-1]
    cases = list(hcho.keys())[::-1]
    # for i, m in enumerate(modes):
    rows, cols = len(cases) + 1 // 2, 2
    fig, axis = plt.subplots(
        rows,
        cols,
        figsize=(4 * 2, 2.5 * 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"
    for j, c in enumerate(cases):
        ri, ci = j // cols, j % cols
        ax = axis[ri, ci]
        add_colorbar = True if ri > 2 else False
        cbar_kwargs = (
            {
                "orientation": "horizontal",
                "shrink": 0.8,
                "label": unit,
            }
            if add_colorbar
            else {}
        )
        chaser = hcho[c].hcho
        chaser = chaser.sel(time=(chaser.time.dt.year == 2020)).mean("time")
        chaser.plot(
            ax=ax,
            cmap="Spectral_r",
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            vmin=0,
            vmax=15,
        )
        ax.set_title(c, fontsize=14)
        ax.coastlines()
        ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
    # Turn off unused axes
    for j in range(len(cases), rows * cols):
        ri, ci = j // cols, j % cols
        axis[ri, ci].set_visible(False)


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


# %%
