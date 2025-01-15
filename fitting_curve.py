# %%
# fitting to Wikinson et al., 2009 functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, Ismax, h, cstar):
    return Ismax * (1 - (1 / (pow(cstar / x, h) + 1)))


fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
axbox = ax.get_position()

xdata_lorg = [0, 150, 240, 400, 1000, 1500]
ydata_lorg = [0, 1.45, 1.38, 1, 0.58, 0.5]
plt.plot(
    xdata_lorg, ydata_lorg, color="#EE6677", marker="o", label="L.formosana (measured)"
)
xdata_l = [150, 240, 400, 1000, 1500]
ydata_l = [1.45, 1.38, 1, 0.58, 0.5]
curve_fit(func, xdata_l, ydata_l)
popt_l, pcov_l = curve_fit(func, xdata_l, ydata_l)
plt.plot(
    xdata_l,
    func(xdata_l, *popt_l),
    color="#EE6677",
    ls="--",
    label="L.formosana (I$_{smax}$=%5.3f, h=%5.4f, C*=%5.0f)" % tuple(popt_l),
)

xdata_morg = [0, 240, 400, 1000, 1500]
ydata_morg = [np.nan, 1.05, 1, 0.7, 0.57]
plt.plot(
    xdata_morg, ydata_morg, color="#CCBB44", marker="o", label="M.indica (measured)"
)
xdata_m = [240, 400, 1000, 1500]
ydata_m = [1.05, 1, 0.7, 0.57]
curve_fit(func, xdata_m, ydata_m)
popt_m, pcov_m = curve_fit(func, xdata_m, ydata_m)
plt.plot(
    xdata_m,
    func(xdata_m, *popt_m),
    color="#CCBB44",
    ls="--",
    label="M.indica (I$_{smax}$=%5.3f, h=%5.4f, C*=%5.0f)" % tuple(popt_m),
)

xdata_porg = [0, 240, 400, 600, 1200, 1500]
ydata_porg = [np.nan, 1.1, 1, 0.79, 0.4, 0.36]
plt.plot(
    xdata_porg,
    ydata_porg,
    "k-",
    marker="o",
    label="P.tremula x P.tremuloides (measured)",
)

xdata_p = [240, 400, 600, 1200, 1500]
ydata_p = [1.1, 1, 0.79, 0.4, 0.36]
curve_fit(func, xdata_p, ydata_p)
popt_p, pcov_p = curve_fit(func, xdata_p, ydata_p)
plt.plot(
    xdata_p,
    func(xdata_p, *popt_p),
    "k--",
    label="P.tremula (I$_{smax}$=%5.3f, h=%5.4f, C*=%5.0f)" % tuple(popt_p),
)

xdata_qorg = [0, 240, 400, 1000, 1500]
ydata_qorg = [np.nan, 0.95, 1, 0.87, 0.81]
plt.plot(
    xdata_qorg, ydata_qorg, color="#4477AA", marker="o", label="Q.rubra (measured)"
)
xdata_q = [240, 400, 1000, 1500]
ydata_q = [0.95, 1, 0.87, 0.81]
curve_fit(func, xdata_q, ydata_q)
popt_q, pcov_q = curve_fit(func, xdata_q, ydata_q)
plt.plot(
    xdata_q,
    func(xdata_q, *popt_q),
    color="#4477AA",
    ls="--",
    label="Q.rubra (I$_{smax}$=%5.3f, h=%5.4f, C*=%5.0f)" % tuple(popt_q),
)

xdata_qorg = [0, 180, 280, 370, 400, 600]
ydata_qorg = [np.nan, 1.970854, 1.367418, 1.071849, 1.0, 0.690692]
plt.plot(
    xdata_qorg, ydata_qorg, color="#AA3377", marker="o", label="A.nigrescens (measured)"
)
xdata_q = [180, 280, 370, 400, 600]
ydata_q = [1.970854, 1.367418, 1.071849, 1.0, 0.690692]
curve_fit(func, xdata_q, ydata_q)
popt_q, pcov_q = curve_fit(func, xdata_q, ydata_q)
plt.plot(
    xdata_q,
    func(xdata_q, *popt_q),
    color="#AA3377",
    ls="--",
    lw=3,
    label="A.nigrescens (I$_{smax}$=%5.3f, h=%5.4f, C*=%5.0f)" % tuple(popt_q),
)

plt.ylim([0, 2.2])
plt.xlim([0, 1800])
plt.xlabel("Ca (ppm)")
plt.ylabel("I$_{x}$/I$_{400}$")
ax.legend(loc="center left", ncol=1, bbox_to_anchor=(1, 0.5))
plt.show()

# %%


# fitting to UKESM functions
from scipy.optimize import curve_fit


def func(x, cast):
    return cast / x


fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
axbox = ax.get_position()

xdata_lorg = [0, 150, 240, 400, 1000, 1500]
ydata_lorg = [0, 1.45, 1.38, 1, 0.58, 0.5]
plt.plot(
    xdata_lorg, ydata_lorg, color="#EE6677", marker="o", label="L.formosana (measured)"
)
xdata_l = [150, 240, 400, 1000, 1500]
ydata_l = [1.45, 1.38, 1, 0.58, 0.5]
curve_fit(func, xdata_l, ydata_l)
popt_l, pcov_l = curve_fit(func, xdata_l, ydata_l)
plt.plot(
    xdata_l,
    func(xdata_l, *popt_l),
    color="#EE6677",
    ls="--",
    label="L.formosana (Ca$_{st}$=%5.3f)" % tuple(popt_l),
)

xdata_morg = [0, 240, 400, 1000, 1500]
ydata_morg = [np.nan, 1.05, 1, 0.7, 0.57]
plt.plot(
    xdata_morg, ydata_morg, color="#CCBB44", marker="o", label="M.indica (measured)"
)
xdata_m = [240, 400, 1000, 1500]
ydata_m = [1.05, 1, 0.7, 0.57]
curve_fit(func, xdata_m, ydata_m)
popt_m, pcov_m = curve_fit(func, xdata_m, ydata_m)
plt.plot(
    xdata_m,
    func(xdata_m, *popt_m),
    color="#CCBB44",
    ls="--",
    label="M.indica (Ca$_{st}$=%5.3f)" % tuple(popt_m),
)

xdata_porg = [0, 240, 400, 600, 1200, 1500]
ydata_porg = [np.nan, 1.1, 1, 0.79, 0.4, 0.36]
plt.plot(
    xdata_porg,
    ydata_porg,
    "k-",
    marker="o",
    label="P.tremula x P.tremuloides (measured)",
)
xdata_p = [240, 400, 600, 1200, 1500]
ydata_p = [1.1, 1, 0.79, 0.4, 0.36]
curve_fit(func, xdata_p, ydata_p)
popt_p, pcov_p = curve_fit(func, xdata_p, ydata_p)
plt.plot(
    xdata_p,
    func(xdata_p, *popt_p),
    "k--",
    label="P.tremula x P.tremuloides (Ca$_{st}$=%5.3f)" % tuple(popt_p),
)

xdata_qorg = [0, 240, 400, 1000, 1500]
ydata_qorg = [np.nan, 0.95, 1, 0.87, 0.81]
plt.plot(
    xdata_qorg, ydata_qorg, color="#4477AA", marker="o", label="Q.rubra (measured)"
)
xdata_q = [240, 400, 1000, 1500]
ydata_q = [0.95, 1, 0.87, 0.81]
curve_fit(func, xdata_q, ydata_q)
popt_q, pcov_q = curve_fit(func, xdata_q, ydata_q)
plt.plot(
    xdata_q,
    func(xdata_q, *popt_q),
    color="#4477AA",
    ls="--",
    label="Q.rubra (Ca$_{st}$=%5.3f)" % tuple(popt_q),
)

xdata_qorg = [0, 180, 280, 370, 400, 600]
ydata_qorg = [np.nan, 1.970854, 1.367418, 1.071849, 1.0, 0.690692]
plt.plot(
    xdata_qorg,
    ydata_qorg,
    color="#AA3377",
    marker="o",
    label="A.nigrescens (measured)",
)
xdata_q = [180, 280, 370, 400, 600]
ydata_q = [1.970854, 1.367418, 1.071849, 1.0, 0.690692]
curve_fit(func, xdata_q, ydata_q)
popt_q, pcov_q = curve_fit(func, xdata_q, ydata_q)
plt.plot(
    xdata_q,
    func(xdata_q, *popt_q),
    color="#AA3377",
    ls="--",
    label="A.nigrescens (Ca$_{st}$=%5.3f)" % tuple(popt_q),
)

plt.ylim([0, 2.2])
plt.xlim([0, 1800])
plt.xlabel("Ca (ppm)")
plt.ylabel("I$_{x}$/I$_{400}$")
ax.legend(loc="center left", ncol=1, bbox_to_anchor=(1.1, 0.5))
plt.show()

# %%
