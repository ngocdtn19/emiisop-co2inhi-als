# %%

"""
chaser_file_preparation_for_ak_apply.py

This script prepares CHASER model data for subsequent application of TROPOMI
averaging kernels. It reads model output fields, processes them, and writes
formatted data suitable for the AK_application.py script.
"""

import pandas as pd
import numpy as np
import xarray as xr
import os as os

import re

time_stamp = pd.date_range("2019-01-01 02:00", "2020-01-01 00:00", freq="2H")
dat = np.zeros((len(time_stamp), 36, 64, 128))
summ = str(int(36 * 64 * 128))
ann = np.zeros_like(dat)
dt = np.dtype(
    [
        ("f_header", ">i4"),
        ("header", ">64S16"),
        ("1f_tail", "i4"),
        ("2f_header", ">i4"),
        ("arr", ">" + summ + "f"),
        ("2f_tail", ">i4"),
    ]
)


def readgtool3D(ifile):
    dat = np.zeros((len(time_stamp), 36, 64, 128))
    summ = str(int(36 * 64 * 128))
    ann = np.zeros_like(dat)
    data = open(ifile, mode="br")
    for i in range(len(time_stamp)):
        dat = np.fromfile(data, dtype=dt, count=1)
        dat = np.reshape(dat[0][4], (36, 64, 128))
        ann[i, :, :, :] = dat[:, :, :]
    return ann


# Temperature = readgtool3D("/mnt/dg2/hoque/TROPOMI_model_analysis/2019/t")
# HCHO_conc = readgtool3D(
#     "/mnt/nobita/dg2/hoque/TROPOMI_model_analysis/New_emission_chaser_2019_2020/2019/ch2o"
# ) # Hoque-sim

Temperature = readgtool3D("/mnt/dg3/ngoc/CHASER_output/VISITst20012023_nudg/2019/2hr/t")
# %%
HCHO_conc = readgtool3D(
    "/mnt/dg3/ngoc/CHASER_output/VISITst20012023_nudg/2019/2hr/ch2o"
)

#####
dat = np.zeros((len(time_stamp), 64, 128))
summ = str(int(64 * 128))
ann = np.zeros_like(dat)
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


def readgtool2D(ifile):
    dat = np.zeros((len(time_stamp), 64, 128))
    ann = np.zeros_like(dat)
    data = open(ifile, mode="br")
    for i in range(len(time_stamp)):
        dat = np.fromfile(data, dtype=dt, count=1)
        dat = np.reshape(dat[0][4], (64, 128))
        ann[i, :, :] = dat[::]
    return ann


# Pressure = readgtool2D("/mnt/dg2/hoque/TROPOMI_model_analysis/2019/ps")
Pressure = readgtool2D("/mnt/dg3/ngoc/CHASER_output/VISITst20012023_nudg/2019/2hr/ps")

########################################################################################
# tt = pd.date_range('2014-01-01','2014-12-31',freq='MS') # set reading-output time
###
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
###
dat = np.zeros(36)
summ = str(int(36))
file = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/GTAXLOC.HETA36"
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
heta = np.fromfile(data, dtype=dt)
sigma = heta[0][4]  # set reading-output sigma (heta36)

################################################

# and pd.to_datetime(time_stamp[i]).strftime("%m")>='09' and pd.to_datetime(time_stamp[i]).strftime("%m")<='11' :

x11 = []
x21 = []
x31 = []
x41 = []
x51 = []
x61 = []
x71 = []
mon = [
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
# mon=['jan']
time_h = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
# time_h=['12','14']
for q in range(0, len(mon)):
    # ff2 = open(
    #     "/mnt/nobita/dg2/hoque/TROPOMI_model_analysis/New_emission_chaser_2019_2020/2019/Chaser_ch2o_%s_all_level_2019_new1.txt"
    #     % mon[q],
    #     "a",
    # )
    ff2 = open(
        "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hoque_test/ngoc_sim_sample12to14/Chaser_ch2o_%s_all_level_2019.txt"
        % mon[q],
        "a",
    )
    ff2.write(
        "Date \t Latitude \t Longitude \t HCHO \t Pressure_grid \t Temp \t level \t Height \n "
    )
    for l in range(0, 34):
        # for l in range(0,2):
        # ff1=open ('Model_hcho_modified_mean.txt','a')
        # ff1.write("Date \t Latitude \t Longitude \t Concentration  \t Pressure_grid \t Temp \n ")
        for i in range(0, len(time_stamp)):
            if (
                pd.to_datetime(time_stamp[i]).strftime("%m") == time_h[q]
                and pd.to_datetime(time_stamp[i]).strftime("%H") >= "12"
                and pd.to_datetime(time_stamp[i]).strftime("%H") <= "14"
            ):
                # if pd.to_datetime(time_stamp[i]).strftime("%H")=='14' :
                for j in range(0, len(lat)):
                    for k in range(0, len(lon)):
                        if lon[k] > 180:
                            ff2.write(
                                str(time_stamp[i])
                                + "\t"
                                + str(lat[j])
                                + "\t"
                                + str(lon[k] - 360)
                                + "\t"
                                + str(HCHO_conc[i][l][j][k])
                                + "\t"
                                + str((Pressure[i][j][k] * sigma[l]) * 100)
                                + "\t"
                                + str(Temperature[i][l][j][k])
                                + "\t"
                                + str(l)
                                + "\t"
                                + str(
                                    (-1)
                                    * (
                                        (
                                            np.log(
                                                (Pressure[i][j][k] * sigma[l])
                                                * 100
                                                / 101325
                                            )
                                        )
                                        * 8.314
                                        * (Temperature[i][l][j][k])
                                    )
                                    / (0.028 * 9.8)
                                )
                                + "\n"
                            )
                        else:
                            ff2.write(
                                str(time_stamp[i])
                                + "\t"
                                + str(lat[j])
                                + "\t"
                                + str(lon[k])
                                + "\t"
                                + str(HCHO_conc[i][l][j][k])
                                + "\t"
                                + str((Pressure[i][j][k] * sigma[l]) * 100)
                                + "\t"
                                + str(Temperature[i][l][j][k])
                                + "\t"
                                + str(l)
                                + "\t"
                                + str(
                                    (-1)
                                    * (
                                        (
                                            np.log(
                                                (Pressure[i][j][k] * sigma[l])
                                                * 100
                                                / 101325
                                            )
                                        )
                                        * 8.314
                                        * (Temperature[i][l][j][k])
                                    )
                                    / (0.028 * 9.8)
                                )
                                + "\n"
                            )
            else:
                continue

# %%
