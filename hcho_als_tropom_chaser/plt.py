# %%
import matplotlib.pyplot as plt
import seaborn as sns
import regionmask
import cartopy.crs as ccrs
import pymannkendall as pymk
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr

from utils import *

hcho = HCHO()
inhi_chaser = hcho.inhi_chaser_hcho
no_inhi_chaser = hcho.no_inhi_chaser_hcho
tropomi = hcho.tropomi_hcho


# %%
colors = ["#66c2a5", "#fc8d62", "#8da0cb"]


def plt_map_mean_year():
    fig, axis = plt.subplots(
        4,
        5,
        figsize=(3.75 * 5, 2 * 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    for j, year in enumerate(np.arange(2020, 2024)):

        tropomi_y = tropomi.sel(time=(tropomi.time.dt.year == year)).mean(
            "time"
        )
        inhi_chaser_y = inhi_chaser.sel(time=(inhi_chaser.time.dt.year == year)).mean(
            "time"
        )
        no_inhi_chaser_y = no_inhi_chaser.sel(time=(no_inhi_chaser.time.dt.year == year)).mean(
            "time"
        )

        diff_inhi_year = (inhi_chaser_y - tropomi_y) * 1e2 / tropomi_y
        diff_no_inhi_year = (no_inhi_chaser_y - tropomi_y) * 1e2 / tropomi_y

        titles = [
            f"TROPOMI ({year})",
            f"Inhi-CHASER ({year})",
            f"NoInhi-CHASER ({year})",
            f"Inhi-CHASER-TROPOMI ({year})",
            f"NoInhi-CHASER-TROPOMI ({year})",
        ]
        cmaps = ["rainbow", "rainbow", "rainbow", "bwr","bwr"]
        vmins = [0, 0, 0, -150, -150]
        vmaxs = [20, 20, 20, 150, 150]
        add_colorbar = True if j == 3 else False

        for i, ds in enumerate([tropomi_y, inhi_chaser_y, no_inhi_chaser_y, diff_inhi_year, diff_no_inhi_year]):

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


def plt_map_mean_ss():
    fig, axis = plt.subplots(
        2,
        5,
        figsize=(3.75*5 , 4),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    winters = [12, 1, 2]
    summers = [6, 7, 8]

    djf_tropomi = tropomi.sel(
        time=(tropomi.time.dt.month.isin(winters))
    ).mean("time")
    jja_tropomi = tropomi.sel(
        time=(tropomi.time.dt.month.isin(summers))
    ).mean("time")
    
    djf_inhi_chaser = inhi_chaser.sel(time=(inhi_chaser.time.dt.month.isin(winters))).mean(
        "time", skipna=True
    )
    jja_inhi_chaser = inhi_chaser.sel(time=(inhi_chaser.time.dt.month.isin(summers))).mean(
        "time", skipna=True
    )

    djf_no_inhi_chaser = no_inhi_chaser.sel(time=(no_inhi_chaser.time.dt.month.isin(winters))).mean(
        "time", skipna=True
    )
    jja_no_inhi_chaser = no_inhi_chaser.sel(time=(no_inhi_chaser.time.dt.month.isin(summers))).mean(
        "time", skipna=True
    )

    djf_inhi_diff = (djf_inhi_chaser - djf_tropomi) * 1e2 / djf_tropomi

    jja_inhi_diff = (jja_inhi_chaser - jja_tropomi) * 1e2 / jja_tropomi
    
    djf_no_inhi_diff = (djf_no_inhi_chaser - djf_tropomi) * 1e2 / djf_tropomi

    jja_no_inhi_diff = (jja_no_inhi_chaser - jja_tropomi) * 1e2 / jja_tropomi

    titles = [
        [
            "TROPOMI (DJF)",
            "Inhi-CHASER (DJF)",
            "No-Inhi-CHASER (DJF)",
            "Inhi-CHASER-TROPOMI (DJF)",
            "No-Inhi-CHASER-TROPOMI (DJF)", 
        ],
        [
            "TROPOMI (JJA)",
            "Inhi-CHASER (JJA)",
            "No-Inhi-CHASER (JJA)",
            "Inhi-CHASER-TROPOMI (JJA)",
            "No-Inhi-CHASER-TROPOMI (JJA)",
        ]
    ]
    cmaps = ["rainbow", "rainbow", "rainbow", "bwr", "bwr"]
    vmins = [0, 0, 0, -150, -150]
    vmaxs = [20, 20, 20, 150, 150]

    dss = [
        [djf_tropomi, djf_inhi_chaser, djf_no_inhi_chaser, djf_inhi_diff, djf_no_inhi_diff],
        [jja_tropomi, jja_inhi_chaser, jja_no_inhi_chaser, jja_inhi_diff, jja_no_inhi_diff]
    ]

    for i, ds in enumerate(dss):
        for j, _ in enumerate(ds):
            ax = axis[i, j]

            add_colorbar = True if i > 0 else False
            unit = "HCHO total col. (\u00d710$^{15}$ molec.cm$^{-2}$)" if j < 3 else "[%]"
            cbar_kwargs = (
                {
                    "orientation": "horizontal",
                    "shrink": 0.6,
                    "label": unit,
                }
                if add_colorbar
                else {}
            )

            levels = 21 if j < 3 else 5

            dss[i][j].plot(
                ax=ax,
                cmap=cmaps[j],
                levels=levels,
                vmin=vmins[j],
                vmax=vmaxs[j],
                add_colorbar=add_colorbar,
                cbar_kwargs=cbar_kwargs,
            )

            ax.set_title(titles[i][j], fontsize=14)
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
        figsize=(4 * 2,4 ),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    cmap = "coolwarm"
    hcho.tropmomi_inhi_chaser_ss_corr.plot(
        ax=axis[0][0],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        add_colorbar = False
    )
    hcho.tropmomi_inhi_chaser_ann_corr.plot(
        ax=axis[0][1],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        add_colorbar = False
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
        for j in [0,1]:
            axis[i][j].coastlines()
            axis[i][j].set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
            axis[i][j].set_xticks([])
            axis[i][j].set_xlabel("")
            axis[i][j].set_yticks([])
            axis[i][j].set_ylabel("")


def plt_tropo_isop(mode="ss"):
    isop_ds = hcho.isop_tropo_ann_corr if mode == "ann" else hcho.isop_tropo_ss_corr

    fig, axis = plt.subplots(
        2,
        3,
        figsize=(5 * 2, 5.5),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    cmap = "coolwarm"
    for i, var in enumerate(list(isop_ds.keys())):
        ci = i % 3
        ax = axis[i // 3, ci]
        isop_ds[var].plot(
            ax=ax,
            cmap=cmap,
            levels=21,
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.6,
                "label": "Pearson R",
            },
        )

        ax.set_title(var, fontsize=14)
        ax.coastlines()
        ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_ylabel("")


def plt_ann_glob():
    fig, axis = plt.subplots(1, 1, figsize=(4.5, 4), layout="constrained")
    dfs = [ hcho.inhi_chaser_glob,hcho.tropomi_glob,hcho.no_inhi_chaser_glob]
    names = [ "Inhi-CHASER", "TROPOMI", "No-Inhi-CHASER"]

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

        axis.set_title(
            f" Global Mean Annual HCHO"
        )
        print(notes[i])
        print(f"R:{pearson_r:.2f}")
        print(f"RMSE:{rmse:.2f}")
        print(f"trend chaser:{trend_chaser:.2f}")


def plt_reg(mode="ss"):
    list_regions = ["AMZ", "ENA", "SAF", "MED", "CEU", "EAS", "SAS", "SEA", "NAU"]

    if mode == "ann":
        inhi_chaser_reg = hcho.inhi_chaser_reg_ann
        no_inhi_chaser_reg = hcho.no_inhi_chaser_reg_ann
        tropomi_reg = hcho.tropomi_reg_ann
        index = "year"
    else:
        inhi_chaser_reg = hcho.inhi_chaser_reg_ss
        no_inhi_chaser_reg = hcho.no_inhi_chaser_reg_ss
        tropomi_reg = hcho.tropomi_reg_ss
        index = "month"

    fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")
    df_stats = {
        "reg": list_regions,
        "r_inhi":[],
        "r_no_inhi":[],
        "rmse_inhi":[],
        "rmse_no_inhi":[]
    }

    for i, reg in enumerate(list_regions):
        ri, ci = i // 3, i % 3
        ax = axis[ri, ci]
        reg_df_inhi = (
            inhi_chaser_reg[[index, reg]].set_index(index).rename(columns={reg: "Inhi-CHASER"})
        )
        reg_df_no_inhi = (
            no_inhi_chaser_reg[[index, reg]].set_index(index).rename(columns={reg: "NoInhi-CHASER"})
        )
        reg_df_tropomi = (
            tropomi_reg[[index, reg]].set_index(index).rename(columns={reg: "TROPOMI"})
        )

        # legend = True if ri == 0 and ci == 2 else False
        # legend = False

        sns.lineplot(
            reg_df_inhi,
            ax=ax,
            palette=[colors[0]],
            markers=True,
            lw=2, 
        )
        sns.lineplot(
            reg_df_no_inhi,
            ax=ax,
            palette=[colors[2]],
            markers=True,
            lw=2, 
        )
        sns.lineplot(reg_df_tropomi, ax=ax, palette=[colors[1]], markers=True, lw=2, )
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

            trend_inhi_chaser = trend_inhi_chaser if not res_inhi_chaser.h else f"{trend_inhi_chaser}*"
            trend_no_inhi_chaser = trend_no_inhi_chaser if not res_no_inhi_chaser.h else f"{trend_no_inhi_chaser}*"
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


# %%
