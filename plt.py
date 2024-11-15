# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib as mpl
import xarray as xr
import pandas as pd
import pickle

from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from mypath import *
from MultiModel import *
import mk
import pymannkendall as pymk

# colors_dict = {
#     "co2": "#8da0cb",
#     "co2f": "#762a83",
#     "co2fi": "#9970ab",
#     "lulcc": "#b3de69",
#     "clim": "#fb8072",
#     "all": "#66c2a5",
#     "tas": "#e31a1c",
#     "rsds": "#fee08b",
#     "pr": "#386cb0",
# }

# linestyles_dict = {
#     "co2": "-",
#     "co2f": "-",
#     "co2fi": "-",
#     "lulcc": "-",
#     "clim": "-.",
#     "all": "--",
#     "tas": "-",
#     "rsds": "-",
#     "pr": "-",
# }

# map_colors_dict = {
#     "co2": "#beaed4",
#     "co2f": "#beaed4",
#     "co2fi": "#beaed4",
#     "co$_2$": "#beaed4",
#     "lulcc": "#b3de69",
#     "clim": "#fb9a99",
#     "tas": "#e31a1c",
#     "rsds": "#ffff99",
#     "pr": "#386cb0",
#     "nan": "lightgrey",
# }

# model_orders = []

title_sz = 16
legend_sz = 14
unit_sz = 12


# Plot Fig. 1a - Mean global isoprene emission for the near present day (2016-2021)
def plt_mean_glob_pd(emiisop):
    model_names = list(emiisop.multi_models.keys())
    df = pd.DataFrame()
    for name in model_names:
        df[name] = emiisop.multi_models[name].global_rate.sel(year=slice(2001, 2018))
    fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
    sns.barplot(
        df,
        ax=ax,
        palette=sns.color_palette(
            [
                "#94C973",
                "#fdbf6f",
                "#fb9a99",
                "#478C5C",
                "#9467BD",
                "#e41a1c",
            ]
        ),
        errorbar="sd",
        width=0.6,
    )
    ax.set_ylim([0, 600])
    # for x, y in enumerate(df.mean()):
    #     ax.annotate(np.round(y, decimals=1), (x, y + 15), ha="center")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("[TgC yr$^{-1}$]", fontsize=unit_sz)


# Plot Fig. 1b - regional contribution for the near present-day (2016-2021)
def plt_regional_contri(emiisop):
    l_roi = LIST_REGION
    model_names = list(emiisop.multi_models.keys())

    l_y = []
    for roi in l_roi:
        l_y.append(
            np.array(
                [
                    emiisop.multi_models[name]
                    .regional_rate[roi]
                    .sel(year=slice(2001, 2018))
                    .mean()
                    .item()
                    for name in model_names
                ]
            )
        )
    l_y = np.array(l_y)
    df = pd.DataFrame({roi: val for roi, val in zip(l_roi, l_y)}, index=model_names)
    ax = df.plot.bar(
        stacked=True,
        color=ROI_COLORS,
        rot=45,
    )
    for x, y in enumerate(df.sum(axis=1)):
        ax.annotate(np.round(y, decimals=0), (x, y + 5), ha="center")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="right",
        ncol=1,
        bbox_to_anchor=(0.5, 0.5, 0.7, 0.1),
        borderaxespad=0.0,
    )
    ax.set_ylim([0, 600])
    ax.set_ylabel(VIZ_OPT[emiisop.var_name]["line_bar_unit"], fontsize=unit_sz)


# Plot Fig. 2 - Difference map between each sen exp and VISIT-woCO2inhi (2016-2021)
def plt_glob_present_map(emiisop, cmap="bwr"):
    list_models = list(emiisop.multi_models.keys())
    vmin, vmax = -1, 1
    cmap = mpl.colormaps.get_cmap(cmap)
    # cmap.set_under("snow")

    rows = 3
    cols = 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    for i, m in enumerate(list_models):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        if m != "VISIT-woCO2inhi":
            data1 = (
                emiisop.multi_models[m]
                .annual_per_area_unit.sel(year=slice(2001, 2018))
                .mean("year")
            )
            data0 = (
                emiisop.multi_models["VISIT-woCO2inhi"]
                .annual_per_area_unit.sel(year=slice(2001, 2018))
                .mean("year")
            )
            data = (data1 - data0) * emiisop.multi_models[m].ds_mask["mask"]
            data = data.sel(lat=slice(82.75, -55.25))
            data.plot.pcolormesh(
                ax=ax,
                cmap=cmap,
                levels=11,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            ax.coastlines()
            ax.set_title(m)
        axes[-1, -1].set_axis_off()
    bounds = np.arange(vmin, vmax + vmax * 0.2, vmax * 0.2)
    norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, location="bottom")
    cbar.set_label("[gC m$^{-2}$ yr$^{-1}$]", size=unit_sz)


# Plot Fig. 3 - Mean annual isoprene emission by lat
def plt_pd_mean_by_lat(emiisop):
    l_m_names = list(emiisop.multi_models.keys())
    colors = [
        "#94C973",
        "#fdbf6f",
        "#fb9a99",
        "#478C5C",
        "#9467BD",
        "#e41a1c",
    ]
    colors_dict = {m_name: c for m_name, c in zip(l_m_names, colors[: len(l_m_names)])}
    lss = ["-", "-.", "-", "--", "-", "-"]
    ls_dict = {m_name: c for m_name, c in zip(l_m_names, lss[: len(l_m_names)])}
    fig, ax = plt.subplots(figsize=(9.5, 6.5), layout="constrained")
    axbox = ax.get_position()
    for m_name in l_m_names:
        org_ds = (
            emiisop.multi_models[m_name]
            .annual_per_area_unit.sel(year=slice(2000, 2021))
            .mean(dim="year")
        )
        ds = org_ds.mean(dim="lon")
        ds = ds.sel(lat=np.arange(-90, 90, 5), method="nearest")
        ax.plot(
            ds.lat,
            ds,
            label=m_name,
            linewidth=2.5,
            color=colors_dict[m_name],
            ls=ls_dict[m_name],
        )
        ax.set_xlabel("Latitude")
        ax.set_ylabel(VIZ_OPT[emiisop.var_name]["map_unit"], fontsize=14)
        plt.ylim([0, 5])
        ax.legend(
            loc="center",
            ncol=3,
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )


# Plot Fig. 4 - Interannual variations in global isoprene emission over 1901–2021
def plt_glob_annual_variation(emiisop):
    model_names = list(emiisop.multi_models.keys())
    colors = [
        "#94C973",
        "#fdbf6f",
        "#fb9a99",
        "#478C5C",
        "#9467BD",
        "#e41a1c",
    ]
    colors_dict = {
        m_name: c for m_name, c in zip(model_names, colors[: len(model_names)])
    }
    lss = ["-", "-.", "-", "--", "-", "-"]
    ls_dict = {m_name: c for m_name, c in zip(model_names, lss[: len(model_names)])}
    fig, ax = plt.subplots(figsize=(9.5, 6.5), layout="constrained")
    axbox = ax.get_position()
    for m_name in model_names:
        cmip6_obj = emiisop.multi_models[m_name]
        x, y = cmip6_obj.global_rate["year"], cmip6_obj.global_rate
        ax.plot(
            x,
            y,
            label=m_name,
            linewidth=2.5,
            color=colors_dict[m_name],
            ls=ls_dict[m_name],
        )
        res = pymk.original_test(y, alpha=0.05)
        print(m_name, res)

    ax.set_xlabel("Year")
    ax.set_ylabel(VIZ_OPT[emiisop.var_name]["line_bar_unit"], fontsize=14)
    ax.legend(
        loc="center",
        ncol=3,
        bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
    )


# Plot Fig. 5 - Interannual variations in regional isoprene emission over 1901-2021
def plt_reg_annual_variation(emiisop):
    model_names = emiisop.multi_models.keys()
    colors = [
        "#94C973",
        "#fdbf6f",
        "#fb9a99",
        "#478C5C",
        "#9467BD",
        "#e41a1c",
    ]
    colors_dict = {
        m_name: c for m_name, c in zip(model_names, colors[: len(model_names)])
    }
    lss = ["-", "-.", "-", "--", "-", "-"]
    ls_dict = {m_name: c for m_name, c in zip(model_names, lss[: len(model_names)])}
    roi_ds = {}
    for roi in LIST_REGION:
        fig, ax = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
        axbox = ax.get_position()
        roi_ds[roi] = []
        years = []
        for name in model_names:
            years.append(emiisop.multi_models[name].regional_rate[roi]["year"])
            roi_ds[roi].append(emiisop.multi_models[name].regional_rate[roi])
        for x, y, n in zip(years, roi_ds[roi], model_names):
            ax.plot(
                x,
                y,
                label=n,
                linewidth=2.5,
                color=colors_dict[n],
                ls=ls_dict[n],
            )
        ax.set_title(f"{roi}", fontsize=title_sz)
        # ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT["emiisop"]["line_bar_unit"])
        ax.legend(
            loc="center",
            ncol=3,
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
            fontsize=legend_sz,
        )


def cal_org_trends_map(var_obj, var_name, model_name):
    file_mk_org = os.path.join(
        DATA_DIR, "processed_org_data/mk_trends_map", f"{model_name}_{var_name}.nc"
    )
    if not os.path.exists(file_mk_org):
        annual_ds = var_obj.multi_models[model_name].annual_per_area_unit

        y = xr.DataArray(
            np.arange(len(annual_ds["year"])) + 1,
            dims="year",
            coords={"year": annual_ds["year"]},
        )
        slope = xr.Dataset({})
        slope[var_name] = mk.kendall_correlation(annual_ds, y, "year")
        slope.to_netcdf(file_mk_org)
    else:
        slope = xr.open_dataset(file_mk_org)
    return slope


# Plot Fig.5 - Spatial distribution of isoprene emission trends from 1901 to 2021
def plt_emiisop_trends_map(emiisop, cmap="bwr"):
    list_models = list(emiisop.multi_models.keys())

    cmap = mpl.colormaps.get_cmap(cmap)
    vmin, vmax = -50, 50

    rows = 3
    cols = 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    for i, m in enumerate(list_models):
        # calculate mk trends
        slope_ds = cal_org_trends_map(emiisop, "emiisop", m)

        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.coastlines()
        data = slope_ds[list(slope_ds.keys())[0]] * 1e3
        if "VISIT" in m:
            data = data.sel(lat=slice(82.75, -55.25))
        else:
            data = data.sel(lat=slice(-55.25, 82.75))
        data.plot.pcolormesh(
            ax=ax,
            cmap=cmap,
            levels=21,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_title(m)
    bounds = np.arange(vmin, vmax + vmax * 0.1, vmax * 0.1)
    norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.plasma.N, extend="both")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, location="bottom")
    cbar.set_label("[mgC m$^{-2}$ yr$^{-2}$]", size=unit_sz)

    # if fig_name:
    #     path_ = f"../figures/{fig_name}.tiff"
    #     fig.savefig(
    #         path_,
    #         format="tiff",
    #         dpi=300,
    #         bbox_inches="tight",
    #     )


# sup plt Fig. S3 - SREX regions
def sup_plt_srex_regions():
    text_kws = dict(color="#67000d", fontsize=9, bbox=dict(pad=0.2, color="w"))
    regionmask.defined_regions.srex.plot(label="abbrev", text_kws=text_kws)
    plt.tight_layout()


# %%
