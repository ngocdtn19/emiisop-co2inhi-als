# %%

import pandas as pd
import datetime
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.basemap import Basemap
import numpy as np
from scipy.interpolate import interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# from mpl_toolkits.basemap import maskoceans
import pandas as pd
import scipy
import xarray as xr
import re
import datetime as dt
from operator import attrgetter


dat = np.zeros(128)
summ = str(int(128))
file = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/GTAXLOC.GLON128"
data = open(file, "br")
dt = np.dtype(
    [
        ("f_header", ">i"),
        ("header", ">64S16"),
        ("1f_tail", "i"),
        ("2f_header", ">i"),
        ("arr", ">" + summ + "f"),
        ("2f_tail", ">i"),
    ]
)  # big endian
glon = np.fromfile(data, dtype=dt)
lon = glon[0][4]  # set reading-output lon (t42)
###
dat = np.zeros(64)
summ = str(int(64))
file = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/GTAXLOC.GLAT64"
data = open(file, "br")
dt = np.dtype(
    [
        ("f_header", ">i"),
        ("header", ">64S16"),
        ("1f_tail", "i"),
        ("2f_header", ">i"),
        ("arr", ">" + summ + "f"),
        ("2f_tail", ">i"),
    ]
)
glat = np.fromfile(data, dtype=dt)
lat = glat[0][4]  # set reading-output lat (t42)
Lon = []
for i in range(0, len(lon)):
    if lon[i] > 180:
        Lon.append(lon[i] - 360)
    else:
        Lon.append(lon[i])
Lon = np.array(Lon)
xmax1 = np.linspace(lat.min(), lat.max(), int((lat.max() - lat.min()) // 2.75))
ymax1 = np.linspace(Lon.min(), Lon.max(), int((Lon.max() - Lon.min()) // 2.75))


def a2x(arr, var, xmax1=xmax1, ymax1=ymax1):
    xr_ds = xr.Dataset(
        {
            var: (("lat", "lon"), arr.reshape(64, 129)),
        },
        coords={
            "lat": (("lat"), xmax1),
            "lon": (("lon"), ymax1),
        },
    )
    return xr_ds.sortby("lat", ascending=False)


# %%
## Sat ###
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
mon = "jul"
# time_h=['04','05','07','08','09','10','11','12']


# ff = open(
#     "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/ngoc_sim_AKapplied/AK_applied_ch2o_%s_2019_new.txt"
#     % item,
#     "a",
# )  ### the AK applied CHASER HCHO will be written in this file

# ff.write(
#     "Date \t Hour \t Latitude \t Longitude \t Partial_col  \t Concen \t Level\n "
# )
# xmax1=lat; ymax1=Lon
####.strip('\n')
x11 = []
x21 = []
x31 = []
x41 = []
x51 = []
x61 = []
x71 = []
x81 = []

item = "jul"
for chunk in pd.read_table(
    "/mnt/dg2/hoque/TROPOMI_model_analysis/New_emission_chaser_2019_2020/2019/Chaser_ch2o_%s_all_level_2019_new1.txt"
    % item,
    chunksize=2500000,
    # chunksize=64 * 128 * 35,
):
    df_model = chunk.reset_index()
    df_model.columns = [
        "index",
        "Dates",
        "Lat",
        "Lon",
        "HCHO",
        "Pressure_grid",
        "Temp",
        "Level",
        "Height",
    ]

    # df_model=(df_model.drop(df_model.index[[0]]))

    df_model["Lat"] = round(df_model["Lat"].astype(float), 2)
    df_model["Lon"] = round(df_model["Lon"].astype(float), 2)
    df_model["HCHO"] = df_model["HCHO"].astype(float)
    df_model["Date"] = (pd.to_datetime(df_model.Dates)).dt.strftime("%Y/%m/%d")
    df_model["Pressure_grid"] = df_model["Pressure_grid"].astype(float)
    df_model["Level"] = df_model["Level"].astype(float)
    df_model["Temp"] = df_model["Temp"].astype(float)
    df_model["Hour"] = (pd.to_datetime(df_model.Dates)).dt.strftime("%H")
    # df_model=df_model.drop(columns=['Date'])

    # with open ("Regridded_satelite_new_2.txt","r") as r:
    with open(
        "/mnt/dg2/hoque/TROPOMI_model_analysis/2019/analysis/AK_%s.txt" % item, "r"
    ) as r:  # Averaging kernel information from TROPOMI
        next(r)
        for line in r:
            j1 = re.split(r"\t", line)
            if len(j1) > 2:
                x1.append(j1[0])
                x2.append(j1[1])
                x3.append(j1[2])
                x4.append(j1[3])
                x5.append(j1[4])
                x6.append(j1[5].strip("\n"))

    df_sat = pd.DataFrame(
        {
            "Date": x1,
            "Lat": x2,
            "Lon": x3,
            "AK": x4,
            "Pressure_grid": x5,
            "Level": x6,
        }
    )

    # df_sat=df_sat.drop([24409808, 25766273, 41949124]) ### for 2020 AK file
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    df_sat["Lat"] = df_sat["Lat"].astype(float)
    df_sat["Lon"] = df_sat["Lon"].astype(float)
    df_sat["AK"] = df_sat["AK"].astype(float)
    df_sat["Dates"] = (pd.to_datetime(df_sat.Date)).dt.strftime("%Y/%m/%d")
    df_sat["Pressure_grid"] = df_sat["Pressure_grid"].astype(float)
    df_sat["Level"] = df_sat["Level"].astype(float)
    df_sat = df_sat.drop(columns=["Date"])

    # P_to_h1= (-1)*((np.log(np.array(df_model1.PressuP_to_h1= (-1)*((np.log(np.array(df_model1.Pressure_grid)/101325))*8.314*np.array(df_model1.Temp))/(0.028*9.8)re_grid)/101325))*8.314*np.array(df_model1.Temp))/(0.028*9.8)
    h3 = np.unique(df_model.Level)
    # h4=np.unique(df_sat_modified.Date)
    h2 = np.unique(df_model.Date)
    h4 = np.unique(df_model.Hour)
    # h4=np.delete(h4,0)

    for g1 in range(0, len(h2)):
        df_sat_1 = df_sat[(df_sat.Dates == h2[g1])]
        if len(df_sat_1) == 0:
            pass
        else:

            for t in range(0, len(h3) - 1):
                for z in range(0, len(h4)):

                    df_model_1 = df_model[
                        (df_model.Level == h3[t])
                        & (df_model.Date == h2[g1])
                        & (df_model.Hour == h4[z])
                    ]
                    df_model_2 = df_model[
                        (df_model.Level == h3[t + 1])
                        & (df_model.Date == h2[g1])
                        & (df_model.Hour == h4[z])
                    ]
                    df_sat_2 = df_sat_1[(df_sat_1.Level == h3[t])]
                    df_sat_3 = df_sat_1[(df_sat_1.Level == h3[t + 1])]
                    if len(df_model_1) != len(df_model_2):
                        pass
                    elif (
                        len(df_model_1) == 0
                        or len(df_model_2) == 0
                        or len(df_sat_2) == 0
                        or len(df_sat_3) == 0
                    ):
                        pass

                    else:

                        Lat1 = np.array(df_model_1.Lat)
                        Lon1 = np.array(df_model_1.Lon)
                        Pres_grid_model = np.array(df_model_1.Pressure_grid)
                        HCHO = np.array(df_model_1.HCHO)
                        Temp = np.array(df_model_1.Temp)
                        Lat4 = np.array(df_model_2.Lat)
                        Lon4 = np.array(df_model_2.Lon)
                        Temp2 = np.array(df_model_2.Temp)
                        Lat2 = np.array(df_sat_2.Lat)
                        Lon2 = np.array(df_sat_2.Lon)
                        Pres_grid_sat1 = np.array(df_sat_2.Pressure_grid)
                        AVK = np.array(df_sat_2.AK)
                        Lat3 = np.array(df_sat_3.Lat)
                        Lon3 = np.array(df_sat_3.Lon)
                        Pres_grid_sat2 = np.array(df_sat_3.Pressure_grid)
                        # xmax=np.linspace(np.min(Lat2),np.max(Lat2),64)
                        # ymax=np.linspace(np.min(Lon2),np.max(Lon2),128)
                        if len(Lat1) == 0 or len(Lat2) == 0:
                            pass
                        else:
                            xmax1 = np.linspace(
                                Lat1.min(),
                                Lat1.max(),
                                int((Lat1.max() - Lat1.min()) // 2.75),
                            )
                            ymax1 = np.linspace(
                                Lon1.min(),
                                Lon1.max(),
                                int((Lon1.max() - Lon1.min()) // 2.75),
                            )
                            Yi, Xi = np.meshgrid(ymax1, xmax1)
                            points = []
                            points1 = []
                            points2 = []
                            points3 = []
                            for m in range(len(Lat1)):
                                points.append([Lon1[m], Lat1[m]])
                            for m2 in range(len(Lat2)):
                                points1.append([Lon2[m2], Lat2[m2]])
                            for m3 in range(len(Lat3)):
                                points2.append([Lon3[m3], Lat3[m3]])
                            for m4 in range(len(Lat4)):
                                points3.append([Lon4[m4], Lat4[m4]])
                            new_pres_grid = scipy.interpolate.griddata(
                                points, Pres_grid_model, (Yi, Xi), method="nearest"
                            )
                            new_grd_HCHO = scipy.interpolate.griddata(
                                points, HCHO, (Yi, Xi), method="nearest"
                            )
                            new_tem = scipy.interpolate.griddata(
                                points, Temp, (Yi, Xi), method="nearest"
                            )
                            new_tem2 = scipy.interpolate.griddata(
                                points3, Temp2, (Yi, Xi), method="nearest"
                            )
                            # new_heg=scipy.interpolate.griddata(points,Height, (Yi,Xi),method='nearest')
                            new_pres_grid2 = scipy.interpolate.griddata(
                                points1, Pres_grid_sat1, (Yi, Xi), method="nearest"
                            )
                            new_AK_grid = scipy.interpolate.griddata(
                                points1, AVK, (Yi, Xi), method="nearest"
                            )
                            new_pres_grid3 = scipy.interpolate.griddata(
                                points2, Pres_grid_sat2, (Yi, Xi), method="nearest"
                            )
                            Lat_re_mod = []
                            Lon_re_mod = []
                            Pres_re_mod = []
                            HCHO_re_mod = []
                            Tem_re_mod = []
                            Height_re_mod = []
                            Tem_re_mod2 = []
                            AVK_re_mod = []
                            Pres_re_mod1 = []
                            Pres_re_mod2 = []
                            for i in range(len(Xi)):
                                for j in range(len(Xi[0])):
                                    Lat_re_mod.append(Xi[i][j])
                                    Lon_re_mod.append(Yi[i][j])
                                    Pres_re_mod.append(new_pres_grid[i][j])
                                    HCHO_re_mod.append(new_grd_HCHO[i][j])
                                    Tem_re_mod.append(new_tem[i][j])
                                    Tem_re_mod2.append(new_tem2[i][j])
                                    AVK_re_mod.append(new_AK_grid[i][j])
                                    Pres_re_mod1.append(new_pres_grid2[i][j])
                                    Pres_re_mod2.append(new_pres_grid3[i][j])

                            P_to_h1 = (
                                (-1)
                                * (
                                    (np.log(np.array(Pres_re_mod1) / 101325))
                                    * 8.314
                                    * np.array(Tem_re_mod)
                                )
                                / (0.028 * 9.8)
                            )
                            P_to_h2 = (
                                (-1)
                                * (
                                    (np.log(np.array(Pres_re_mod2) / 101325))
                                    * 8.314
                                    * np.array(Tem_re_mod2)
                                )
                                / (0.028 * 9.8)
                            )
                            Height_re_mod = abs(P_to_h2 - P_to_h1)
                            # L1=1/(np.exp((np.log(Pres_grid_sat/101325))/M))
                            # Pres_to_height= abs(T/LR*(1/L1 -1))
                            # Present_value=np.mean(Pres_to_height)
                            # Partial_col=((np.array(f2_upd)*1e-12*Height*Pres_grid_sat*1e-4)/(1.38e-23*np.array(Tem_re_mod).astype(float)))*AVK
                            Partial_col = (
                                (
                                    np.array(HCHO_re_mod)
                                    * 1e-9
                                    * Height_re_mod
                                    * Pres_re_mod1
                                    * 1e-4
                                )
                                / (1.38e-23 * np.array(Tem_re_mod).astype(float))
                            ) * AVK_re_mod
                            # Linearize=Height_re_mod*Pres_re_mod1*1e-4/(1.38e-23*np.array(Tem_re_mod))
                            # Previous_value=np.mean(Pres_to_height)
                            break
                break
            break
        break
    break
# %%
