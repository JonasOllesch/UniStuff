from re import T
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

t0175, tab175, tauf175 = np.genfromtxt("content/tables/txt/V175.txt", unpack=True)
t0200, tab200, tauf200 = np.genfromtxt("content/tables/txt/V200.txt", unpack=True)
t0225, tab225, tauf225 = np.genfromtxt("content/tables/txt/V225.txt", unpack=True)
t0250, tab250, tauf250 = np.genfromtxt("content/tables/txt/V250.txt", unpack=True)
t0275, tab275, tauf275 = np.genfromtxt("content/tables/txt/V275.txt", unpack=True)

err = 0.2 * np.ones(15)

t0175 = unp.uarray(t0175, err)
tab175 = unp.uarray(tab175, err)
tauf175 = unp.uarray(tauf175, err)

t0200 = unp.uarray(t0200, err)
tab200 = unp.uarray(tab200, err)
tauf200 = unp.uarray(tauf200, err)

t0225 = unp.uarray(t0225, err)
tab225 = unp.uarray(tab225, err)
tauf225 = unp.uarray(tauf225, err)

t0250 = unp.uarray(t0250, err)
tab250 = unp.uarray(tab250, err)
tauf250 = unp.uarray(tauf250, err)

t0275 = unp.uarray(t0275, err)
tab275 = unp.uarray(tab275, err)
tauf275 = unp.uarray(tauf275, err)

print(t0175)

t0175 = t0175.reshape(5, 3)
tauf175 = tauf175.reshape(5, 3)
tab175 = tab175.reshape(5, 3)
t175 = np.array([t0175, tauf175, tab175])
# print(t175)
# print(np.mean(t175[0, 1]))
t0200 = t0200.reshape(5, 3)
tauf200 = tauf200.reshape(5, 3)
tab200 = tab200.reshape(5, 3)
t200 = np.array([t0200, tauf200, tab200])

t0225 = t0225.reshape(5, 3)
tauf225 = tauf225.reshape(5, 3)
tab225 = tab225.reshape(5, 3)
t225 = np.array([t0225, tauf225, tab225])

t0250 = t0250.reshape(5, 3)
tauf250 = tauf250.reshape(5, 3)
tab250 = tab250.reshape(5, 3)
t250 = np.array([t0250, tauf250, tab250])

t0275 = t0275.reshape(5, 3)
tauf275 = tauf275.reshape(5, 3)
tab275 = tab275.reshape(5, 3)
t275 = np.array([t0275, tauf275, tab275])

t = np.array([t175, t200, t225, t250, t275])
t = np.mean(t, axis=3)

s = 0.5 * 1e-3

v = s / t
# v *= 1e3  # mm/s

print("v = " + "\n")
print(str(v) + "\n" + "\n" + "\n")

# print(str(2 * v[:, 0]) + "\n")
# print(str(v[:, 2] - v[:, 1]) + "\n")
d = 2 * v[:, 0] - (v[:, 2] - v[:, 1])
print("Difference:" + "\n" + str(d) + "\n")
o = abs(np.mean(abs(v[:, :]), axis=1) - abs(d)) / np.mean(abs(v[:, :]), axis=1) > 0.92
print("Okay:" + "\n" + str(o) + "\n")
print(str(abs(np.mean(abs(v[:, :]), axis=1) - abs(d)) / np.mean(abs(v[:, :]), axis=1)))

dist = ufloat(7.625, 0.0051) * 1e-3
rho_O = 886  # kg/m^3
rho_L = 1.225  # kg/m^3

ar = np.ones(5)
eta = np.array([1.85 * ar, 1.865 * ar, 1.855 * ar, 1.8325 * ar, 1.87 * ar])
eta *= 1e-5


def rad(etaL, vab, vauf):
    return unp.sqrt(
        (4 * np.multiply(etaL, (vab - vauf))) / (9 * const.g * (rho_O - rho_L))
    )


def eta_corr(etaL, r):
    return etaL * (1 / (1 + (6.17e-5 * 133.33) * 1 / (101325 * r)))


def q(eta_corr, vab, vauf, U):
    return (
        3
        * np.pi
        * eta_corr
        * unp.sqrt(9 / 4 * eta_corr / const.g * (vab - vauf) / (rho_O - rho_L))
        * (vab + vauf)
        / (U / dist.n)
    )


U = unp.uarray(
    [175 * ar, 200 * ar, 225 * ar, 250 * ar, 275 * ar],
    [1 * ar, 1 * ar, 1 * ar, 1 * ar, 1 * ar],
)
r = rad(eta, v[:, 2, :], v[:, 1, :])
eta_c = eta_corr(eta, r)
q_m = q(eta_c, v[:, 2, :], v[:, 1, :], U)
print(str(eta_c) + "\n" + "\n")
print(str(r) + "\n" + "\n")
print(str(q_m) + "\n" + "\n")
q_t = q_m[o]
q_t *= 1e23
# q_t = np.round(q_t)
q_t *= 1e-23


# darkorange, crimson, deepskyblue, royalblue
# plt.plot(
#     x, y, color="deepskyblue", marker="", linestyle="-", label="Regression",
# )
plt.errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    unp.nominal_values(q_t),
    xerr=None,
    yerr=unp.std_devs(q_t),
    color="darkorange",
    ms=4,
    marker="x",
    linestyle="",
    label="Messwerte",
)
plt.xlabel(r"Messung")
plt.ylabel(r"$e_0~\text{in}~10^{-19}~\unit{\coulomb}$")
plt.legend(loc="best")
plt.grid(linestyle="dashed")

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig("build/e")
plt.clf()
print("q = " + "\n")
print(str(q_t) + "\n" + "\n")
q_t = np.sort(q_t)
print(str(q_t) + "\n" + "\n")
plt.errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    unp.nominal_values(q_t),
    xerr=None,
    yerr=unp.std_devs(q_t),
    color="darkorange",
    ms=4,
    marker="x",
    linestyle="",
    label="Messwerte",
)
plt.xlabel(r"Messung")
plt.ylabel(r"$e_0~\text{in}~10^{-19}~\unit{\coulomb}$")
plt.legend(loc="best")
plt.grid(linestyle="dashed")

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig("build/e_sortiert")
plt.clf()


def Euklid(q, max):
    gcd = q[0]
    for i in range(1, len(q)):
        n = 0
        while abs(gcd - q[i]) > 1e-19 and n <= max:
            if gcd > q[i]:
                gcd -= q[i]
            else:
                q[i] -= gcd
            n += 1
    return gcd


q_0 = Euklid(q_t, 1000)
print("q_0 = " + str(q_0))
# print(
#     "Automatic number of digits on the uncertainty, exponent notation: {:.uS}".format(
#         q_0
#     )
# )
nalit = 6.022141e23
qlit = 1.602176e-19

na = 96485.3321233100184 / q_0
print("na = " + str(na))

file = open("build/values.txt", "w")
file.write("Überschrift" + "\n")
file.write("NA Abweichung:" + str((nalit - na) / nalit * 100) + "\n")
file.write("q Abweichung:" + str((qlit - q_0) / qlit * 100) + "\n")
file.write("\n")
file.write("val_1_m = " + str(2) + "\n")
file.write("val_1_std = " + str(3) + "\n")
file.write("\n")
file.write("\n")
file.write("val_2_a = " + str(0) + "\n")
file.write("val_2_b = " + str(1) + "\n")
file.write("\n")
file.write("val_2_m = " + str(2) + "\n")
file.write("val_2_std = " + str(3) + "\n")
file.write("\n")
file.write("\n")
file.write("\n")
file.write("--------------------------------------------")
file.write("\n")
file.write("\n")
file.write("\n")
file.write("Überschrift 2")
file.close()

# Tabellen
np.savetxt(
    "build/v175.txt", np.column_stack(unp.nominal_values(v[0])), header="v0 vauf vab"
)
np.savetxt(
    "build/v200.txt", np.column_stack(unp.nominal_values(v[1])), header="v0 vauf vab"
)
np.savetxt(
    "build/v225.txt", np.column_stack(unp.nominal_values(v[2])), header="v0 vauf vab"
)
np.savetxt(
    "build/v250.txt", np.column_stack(unp.nominal_values(v[3])), header="v0 vauf vab"
)
np.savetxt(
    "build/v275.txt", np.column_stack(unp.nominal_values(v[4])), header="v0 vauf vab"
)

np.savetxt(
    "build/v_err175.txt", np.column_stack(unp.std_devs(v[0])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err200.txt", np.column_stack(unp.std_devs(v[1])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err225.txt", np.column_stack(unp.std_devs(v[2])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err250.txt", np.column_stack(unp.std_devs(v[3])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err275.txt", np.column_stack(unp.std_devs(v[4])), header="v0 vauf vab"
)

