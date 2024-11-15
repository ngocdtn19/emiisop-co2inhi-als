# %%

from MultiVar import *
from const import *
from mypath import *
from scipy import stats
from visit_preprocess import *
from cartopy.util import add_cyclic_point
import mk
import xskillscore as xs

sns.set_style("ticks")


class ModelVar:
    def __init__(self):
        self.var_obj_dict = {
            "emiisop": EMIISOP,
        }
        self.var_obj_visit_dict = {
            "emiisop": VisitEMIISOP,
        }


class Var(ModelVar):
    """
    Intermodel comparison of variables from historical simulations among selected models (1 var - n models/datasets)
        List models:
            VISIT
            Other datasets:
                ALBERI-v2021
                CAMS-GLOB-BIO
                OMIv2-TOPDOWN
        List variables:
            emiisop: isoprene emission
    """

    processed_dir = os.path.join(DATA_DIR, "processed_org_data")

    def __init__(self, var_name):
        super().__init__()
        self.var_name = var_name
        self.obj_type = None
        self.multi_models = {}

        self.get_obj_type()
        self.get_multi_models()

    def get_obj_type(self):
        self.obj_type = (
            self.var_obj_dict[self.var_name]
            if self.var_name in self.var_obj_dict.keys()
            else None
        )
        self.obj_type_visit = (
            self.var_obj_visit_dict[self.var_name]
            if self.var_name in self.var_obj_visit_dict.keys()
            else None
        )

    def get_model_name(self, path):
        var_name = {
            "emiisop": "AERmon",
        }
        return (
            path.split("\\")[-1]
            .split(var_name[self.var_name])[-1]
            .split("historical")[0]
            .replace("_", "")
        )

    def get_multi_models(self):
        """
        Load and concatenate data for each model, then create a dictionary of timeseries xr.Dataset for multiple models.
        """
        all_files = sorted(glob.glob(os.path.join(VAR_DIR, self.var_name, "*.nc")))
        model_names = sorted(list(set([self.get_model_name(f) for f in all_files])))

        multi_models = {}
        for m_name in model_names:
            print(m_name)
            if "VISIT" not in m_name:
                l_model = []
                for f in all_files:
                    if m_name in f:
                        l_model.append(xr.load_dataset(f))

                multi_models[m_name] = self.obj_type(
                    m_name, xr.concat(l_model, dim=DIM_TIME), self.var_name
                )
            else:
                for f in all_files:
                    if m_name in f:
                        multi_models[m_name] = self.obj_type_visit(
                            m_name, visit_t2cft(f, self.var_name, m_name), self.var_name
                        )

        self.multi_models = multi_models

    def plot_regional_map(self):
        """Regional map visual examination"""
        rois = LIST_REGION
        l_m_name = list(self.multi_models.keys())
        ds = self.multi_models[l_m_name[0]]
        for i, r in enumerate(rois):
            fig = plt.figure(1 + i, figsize=(30, 13))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            data = ds.regional_ds[r].sel(year=2021)
            data.plot.pcolormesh(
                ax=ax,
                cmap="tab20c_r",
                levels=8,
                cbar_kwargs={"label": VIZ_OPT[self.var_name]["map_unit"]},
            )
            plt.title(f"{l_m_name[0]} - {r} ", fontsize=18)

    def save_2_nc(self):
        model_names = list(self.multi_models.keys())
        for name in model_names:
            annual_data = self.multi_models[name].annual_per_area_unit
            if self.var_name in ["emiotherbvocs", "emibvoc", "gpp", "npp"]:
                annual_ds = xr.Dataset(
                    data_vars=dict(
                        var_name=(["lat", "lon", "year"], annual_data.values)
                    ),
                    coords=dict(
                        lat=annual_data.lat, lon=annual_data.lon, year=annual_data.year
                    ),
                )
            else:
                annual_ds = xr.Dataset(
                    data_vars=dict(
                        var_name=(["year", "lat", "lon"], annual_data.values)
                    ),
                    coords=dict(
                        lat=annual_data.lat, lon=annual_data.lon, year=annual_data.year
                    ),
                )
            annual_ds = annual_ds.rename({"var_name": self.var_name})
            annual_ds.to_netcdf(
                os.path.join(
                    self.processed_dir,
                    "annual_per_area_unit",
                    f"{name}_{self.var_name}.nc",
                )
            )


# emiisop = Var("emiisop")
# %%
