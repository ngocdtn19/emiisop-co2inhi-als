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
        df[name] = emiisop.multi_models[name].global_rate.sel(year=slice(2018, 2023))
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
                    .sel(year=slice(2018, 2023))
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


# Plot Fig. 2 - Plot base map (VISIT-woCO2inhi) and Difference map between each sen exp and VISIT-woCO2inhi (2000-2023)
def plt_glob_present_diff_map(emiisop, cmap="bwr"):
    list_models = list(emiisop.multi_models.keys())
    # Make sure VISIT-woCO2inhi comes first
    list_models = ["VISIT-woCO2inhi"] + [
        m for m in list_models if m != "VISIT-woCO2inhi"
    ]

    # Difference colorbar settings
    vmin_diff, vmax_diff = -1, 1
    levels_diff = 21
    bounds_diff = np.linspace(vmin_diff, vmax_diff, levels_diff)
    cmap_diff = mpl.colormaps.get_cmap(cmap)
    norm_diff = mpl.colors.BoundaryNorm(bounds_diff, mpl.cm.plasma.N, extend="both")

    # Base map colorbar settings
    vmin_base, vmax_base = 0, 30
    levels_base = 15
    bounds_base = np.linspace(vmin_base, vmax_base, levels_base + 1)
    norm_base = mpl.colors.BoundaryNorm(bounds_base, mpl.cm.plasma.N, extend="max")
    cmap_base = mpl.colormaps.get_cmap("Spectral_r")

    unit_sz = 10

    # Grid layout
    n_models = len(list_models)
    rows, cols = 3, 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2.5 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    # Flatten axes for easier indexing
    axes = axes.flatten()

    for i, m in enumerate(list_models):
        ax = axes[i]

        if m == "VISIT-woCO2inhi":
            # Plot base map
            data = (
                emiisop.multi_models[m]
                .annual_per_area_unit.sel(year=slice(2000, 2023))
                .mean("year")
                * emiisop.multi_models[m].ds_mask["mask"]
            ).sel(lat=slice(82.75, -55.25))

            im_base = data.plot.pcolormesh(
                ax=ax,
                cmap=cmap_base,
                levels=levels_base,
                vmin=vmin_base,
                vmax=vmax_base,
                add_colorbar=False,
            )
            ax.set_title("VISIT-woCO2inhi")

        else:
            # Plot difference with base
            data1 = (
                emiisop.multi_models[m]
                .annual_per_area_unit.sel(year=slice(2000, 2023))
                .mean("year")
            )
            data0 = (
                emiisop.multi_models["VISIT-woCO2inhi"]
                .annual_per_area_unit.sel(year=slice(2000, 2023))
                .mean("year")
            )
            diff = (data1 - data0) * emiisop.multi_models[m].ds_mask["mask"]
            diff = diff.sel(lat=slice(82.75, -55.25))

            diff.plot.pcolormesh(
                ax=ax,
                cmap=cmap_diff,
                levels=levels_diff,
                add_colorbar=False,
            )
            ax.set_title(f"{m}")

        ax.coastlines()

    # Turn off any unused axes
    for j in range(n_models, len(axes)):
        axes[j].set_axis_off()

    # Add colorbar for base map (first subplot)
    sm_base = mpl.cm.ScalarMappable(cmap=cmap_base, norm=norm_base)
    sm_base.set_array([])
    cbar_base = fig.colorbar(
        sm_base, ax=axes[::2], orientation="horizontal", shrink=0.8, pad=0.08
    )
    cbar_base.set_label("Isoprene emission [gC m$^{-2}$ yr$^{-1}$]", size=unit_sz)

    # Add shared colorbar for difference maps (excluding the first one)
    sm_diff = mpl.cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_diff.set_array([])
    cbar_diff = fig.colorbar(
        sm_diff, ax=axes[1::2], orientation="horizontal", shrink=0.8, pad=0.08
    )
    cbar_diff.set_label("Difference [gC m$^{-2}$ yr$^{-1}$]", size=unit_sz)


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

    fig, axis = plt.subplots(3, 3, figsize=(3.5 * 3, 3.6 * 3), layout="constrained")

    for i, roi in enumerate(LIST_REGION):
        ri, ci = i // 3, i % 3
        ax = axis[ri, ci]

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
        # if ri == 2 and ci ==1:
        #     ax.legend(
        #         loc="center",
        #         ncol=3,
        #         bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        #         fontsize=legend_sz,
        #     )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.rcParams.update({"font.size": legend_sz})


def cal_org_trends_map(var_obj, var_name, model_name):
    file_mk_org = os.path.join(
        DATA_DIR,
        "processed_org_data/mk_trends_map/2000-2023/",
        f"{model_name}_{var_name}.nc",
    )
    if not os.path.exists(file_mk_org):
        # annual_ds = var_obj.multi_models[model_name].annual_per_area_unit.sel(
        #     year=slice(2000, 2023)
        # )
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
        # if "VISIT" in m:
        data = data * emiisop.multi_models[m].ds_mask["mask"]
        data = data.sel(lat=slice(82.75, -55.25))
        # else:
        #     data = data.sel(lat=slice(-55.25, 82.75))
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


# Plot Fig - Spatial distribution of isoprene emission trends from 2000 to 2023 in VISIT-woCO2inhi and difference maps with other cases
def plt_emiisop_trends_diff_map(emiisop):
    list_models = list(emiisop.multi_models.keys())
    list_models = ["VISIT-woCO2inhi"] + [
        m for m in list_models if m != "VISIT-woCO2inhi"
    ]

    # Base map color scale settings
    vmin_base, vmax_base = -2.5, 2.5
    levels_base = 11
    bounds_base = np.linspace(vmin_base, vmax_base, levels_base)
    norm_base = mpl.colors.BoundaryNorm(bounds_base, mpl.cm.plasma.N, extend="both")
    cmap_base = mpl.colormaps.get_cmap("bwr")

    # Difference map color scale settings
    vmin_diff, vmax_diff = -1, 1
    levels_diff = 21
    bounds_diff = np.linspace(vmin_diff, vmax_diff, levels_diff)
    norm_diff = mpl.colors.BoundaryNorm(bounds_diff, mpl.cm.plasma.N, extend="both")
    cmap_diff = mpl.colormaps.get_cmap("Spectral_r")

    rows, cols = 3, 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 2.5 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    axes = axes.flatten()

    # Compute VISIT-woCO2inhi trend (% per year)
    visit_slope_ds = cal_org_trends_map(emiisop, "emiisop", "VISIT-woCO2inhi")
    visit_slope = list(visit_slope_ds.values())[0]
    visit_mean = emiisop.multi_models["VISIT-woCO2inhi"].annual_per_area_unit.mean(
        "year"
    )
    visit_mask = emiisop.multi_models["VISIT-woCO2inhi"].ds_mask["mask"]
    visit_percent = (visit_slope / (visit_mean * visit_mask)) * 100
    visit_percent = visit_percent.where(np.isfinite(visit_percent)).sel(
        lat=slice(82.75, -55.25)
    )

    # Plot base map
    ax = axes[0]
    visit_percent.plot.pcolormesh(
        ax=ax,
        cmap=cmap_base,
        vmin=vmin_base,
        vmax=vmax_base,
        levels=levels_base,
        extend="both",
        add_colorbar=False,
    )
    ax.coastlines()
    ax.set_title("VISIT-woCO2inhi")

    # Plot difference maps (% per year)
    for i, m in enumerate(list_models[1:], start=1):
        ax = axes[i]

        slope_ds = cal_org_trends_map(emiisop, "emiisop", m)
        slope = list(slope_ds.values())[0]
        model_mean = emiisop.multi_models[m].annual_per_area_unit.mean("year")
        model_mask = emiisop.multi_models[m].ds_mask["mask"]
        model_percent = (slope / (model_mean * model_mask)) * 100

        diff_percent = (model_percent - visit_percent).where(np.isfinite(visit_percent))
        diff_percent = diff_percent.sel(lat=slice(82.75, -55.25))

        diff_percent.plot.pcolormesh(
            ax=ax,
            vmin=vmin_diff,
            vmax=vmax_diff,
            cmap=cmap_diff,
            levels=levels_diff,
            extend="both",
            add_colorbar=False,
        )
        ax.coastlines()
        ax.set_title(f"{m}")

    # Turn off unused axes
    for j in range(len(list_models), len(axes)):
        axes[j].set_axis_off()

    # Colorbar for base map (left column)
    sm_base = mpl.cm.ScalarMappable(cmap=cmap_base, norm=norm_base)
    sm_base.set_array([])
    cbar_base = fig.colorbar(
        sm_base,
        ax=axes[::2],  # left column
        orientation="horizontal",
        shrink=0.8,
        pad=0.08,
        extend="both",
    )
    cbar_base.set_label("Isoprene trend (% yr$^{-1}$)", size=10)

    # # Colorbar for difference maps (right column)
    sm_diff = mpl.cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_diff.set_array([])
    cbar_diff = fig.colorbar(
        sm_diff,
        ax=axes[1::2],  # right column
        orientation="horizontal",
        shrink=0.8,
        pad=0.08,
        extend="both",
    )
    cbar_diff.set_label("Difference in relative trends (% yr$^{-1}$)", size=10)


# sup plt Fig. S3 - SREX regions
def sup_plt_srex_regions():
    text_kws = dict(color="#67000d", fontsize=9, bbox=dict(pad=0.2, color="w"))
    regionmask.defined_regions.srex.plot(label="abbrev", text_kws=text_kws)
    plt.tight_layout()


# %%
