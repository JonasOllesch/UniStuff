import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from re import T

output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")

def writeW(Wert,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    try:
        for i in Wert:
            my_file.write(str(i))
            my_file.write('\n')
    except:
        my_file.write(str(Wert))
        my_file.write('\n')

    return 0

tab150, tauf150 = np.genfromtxt("data/U157.txt", unpack=True)
tab175, tauf175 = np.genfromtxt("data/U175.txt", unpack=True)
tab200, tauf200 = np.genfromtxt("data/U200.txt", unpack=True)
tab225, tauf225 = np.genfromtxt("data/U225.txt", unpack=True)
tab250, tauf250 = np.genfromtxt("data/U250.txt", unpack=True)

err = 0.2 * np.ones(15)

tab150  = unp.uarray(tab150, err)
tauf150 = unp.uarray(tauf150, err)

tab175 = unp.uarray(tab175, err)
tauf175 = unp.uarray(tauf175, err)

tab200 = unp.uarray(tab200, err)
tauf200 = unp.uarray(tauf200, err)

tab225 = unp.uarray(tab225, err)
tauf225 = unp.uarray(tauf225, err)

tab250 = unp.uarray(tab250, err)
tauf250 = unp.uarray(tauf250, err)

tab150 = tab150.reshape(5, 3)
tauf150 = tauf150.reshape(5, 3)
t150 = np.array([tauf150, tab150])

tauf175 = tauf175.reshape(5, 3)
tab175 = tab175.reshape(5, 3)
t175 = np.array([tauf175, tab175])

tab200 = tab200.reshape(5, 3)
tauf200 = tauf200.reshape(5, 3)
t200 = np.array([tauf200, tab200])

tauf225 = tauf225.reshape(5, 3)
tab225 = tab225.reshape(5, 3)
t225 = np.array([tauf225, tab225])

tauf250 = tauf250.reshape(5, 3)
tab250 = tab250.reshape(5, 3)
t250 = np.array([tauf250, tab250])

t = np.array([t150, t175, t200, t225, t250])
print('t',t)
t = np.mean(t, axis=3)
print('t_mean',t)

s = 0.5 * 1e-3

v = s / t
print('v',v)
# v *= 1e3  # mm/s

#print("v = " + "\n")
#print(str(v) + "\n" + "\n" + "\n")

# print(str(2 * v[:, 0]) + "\n")
# print(str(v[:, 2] - v[:, 1]) + "\n")
d = 2 * v[0,:] - (v[1,:] - v[0,:])
print("Difference:" + "\n" + str(d) + "\n")
#o = abs(np.mean(abs(v[:, :]), axis=1) - abs(d)) / np.mean(abs(v[:, :]), axis=1) > 0.92
#print("Okay:" + "\n" + str(o) + "\n")
#print('Wo das' + '\n', str(abs(np.mean(abs(v[:, :]), axis=1) - abs(d)) / np.mean(abs(v[:, :]), axis=1)))

dist = ufloat(7.625, 0.0051) * 1e-3
rho_O = 886  # kg/m^3
rho_L = 1.225  # kg/m^3

ar = np.ones(5)
etaL = np.array([1.865 * ar,  1.8525 * ar])
etaL *= 1e-5


#def rad(etaL, vab, vauf):
#    return unp.sqrt(
#        (4 * np.multiply(etaL, abs(vab - vauf))) / (9 * const.g * (rho_O - rho_L))
#    )
def rad(etaL, vab, vauf):
    return unp.sqrt(
        (9 * etaL * (vab - vauf)) / (2 * const.g * (rho_O - rho_L))
    )

def eta_corr(etaL, r):
    return etaL * (1 / (1 + (6.17e-5 * 133.33) / (101325 * r)))


def q(eta_corr, vab, vauf, U):
    return (
        3
        * np.pi
        * eta_corr
        * unp.sqrt(9 / 4 * eta_corr / const.g * (vab - vauf) / (rho_O - rho_L))
        * (vab + vauf)
        / (U / dist.n)
    )

#print('eta =', eta)
#print('v_ab - v_auf = ', abs(v[:, 2, :] - v[:, 1, :]))

U = unp.uarray(
    [150 * ar, 175 * ar, 200 * ar, 225 * ar, 250 * ar],
    [1 * ar, 1 * ar, 1 * ar, 1 * ar, 1 * ar],
)
print('v[0,:]:',v[0,:])
print('v[1,:]:',v[1,:])

v_auf = v[0,:]
v_ab = v[1,:]
r = rad(etaL, v_ab, v_auf)
eta_c = eta_corr(etaL, r)
print('r',r)
print('eta_c',eta_c)
print('U', U)
q_m = q(eta_c, v_ab, v_auf, U)

print('Differenz v_ab v_auf', -v[0,:]+v[1, :])
print(str(eta_c) + "\n" + "\n")
print(str(r) + "\n" + "\n")
print(str(q_m) + "\n" + "\n")
q_t = q_m
q_t *= 1e23
# q_t = np.round(q_t)
q_t *= 1e-23

print('q_t = ',unp.nominal_values(q_t))

# darkorange, crimson, deepskyblue, royalblue
# plt.plot(
#     x, y, color="deepskyblue", marker="", linestyle="-", label="Regression",
# )
##################plt.errorbar(
##################    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15],
##################    unp.nominal_values(q_t),
##################    xerr=None,
##################    yerr=unp.std_devs(q_t),
##################    color="crimson",
##################    ms=4,
##################    marker="x",
##################    linestyle="",
##################    label="Messwerte",
##################)
##################plt.xlabel(r"Messung")
##################plt.ylabel(r"$e_0 \mathbin{/} 10^{-19} \unit{\coulomb}$")
##################plt.legend(loc="best")
##################plt.grid(linestyle="dashed")
##################
##################plt.tight_layout()
##################plt.savefig("build/e")
##################
##################plt.clf()
print("q = " + "\n")
print(str(q_t) + "\n" + "\n")
q_t = np.sort(q_t)
print('sorted', str(q_t) + "\n" + "\n")

#########################plt.errorbar(
#########################    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15],
#########################    unp.nominal_values(q_t),
#########################    xerr=None,
#########################    yerr=unp.std_devs(q_t),
#########################    color="crimson",
#########################    ms=4,
#########################    marker="x",
#########################    linestyle="",
#########################    label="Messwerte",
#########################)
#########################plt.xlabel(r"Messung")
#########################plt.ylabel(r"$e_0~\text{in}~10^{-19}~\unit{\coulomb}$")
#########################plt.legend(loc="best")
#########################plt.grid(linestyle="dashed")
#########################
########################## in matplotlibrc leider (noch) nicht möglich
#########################plt.tight_layout()
#########################plt.savefig("build/e_sortiert")
#########################plt.clf()


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
q_0 = np.mean(q_t)/(1+(6.17e-5 * 133.33)*(101325 * np.mean(r)))**(-2/3)
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
#file.write("\n")
#file.write("val_1_m = " + str(2) + "\n")
#file.write("val_1_std = " + str(3) + "\n")
#file.write("\n")
#file.write("\n")
#file.write("val_2_a = " + str(0) + "\n")
#file.write("val_2_b = " + str(1) + "\n")
#file.write("\n")
#file.write("val_2_m = " + str(2) + "\n")
#file.write("val_2_std = " + str(3) + "\n")
#file.write("\n")
#file.write("\n")
#file.write("\n")
#file.write("--------------------------------------------")
#file.write("\n")
#file.write("\n")
#file.write("\n")
#file.write("Überschrift 2")
file.close()

# Tabellen
np.savetxt(
    "build/v150.txt", np.column_stack(unp.nominal_values(v[0])), header="vauf vab"
)
np.savetxt(
    "build/v175.txt", np.column_stack(unp.nominal_values(v[1])), header="vauf vab"
)
np.savetxt(
    "build/v200.txt", np.column_stack(unp.nominal_values(v[2])), header="v0 vauf vab"
)
np.savetxt(
    "build/v225.txt", np.column_stack(unp.nominal_values(v[3])), header="v0 vauf vab"
)
np.savetxt(
    "build/v250.txt", np.column_stack(unp.nominal_values(v[4])), header="v0 vauf vab"
)


np.savetxt(
    "build/v_err150.txt", np.column_stack(unp.std_devs(v[0])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err175.txt", np.column_stack(unp.std_devs(v[1])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err200.txt", np.column_stack(unp.std_devs(v[2])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err225.txt", np.column_stack(unp.std_devs(v[3])), header="v0 vauf vab"
)
np.savetxt(
    "build/v_err250.txt", np.column_stack(unp.std_devs(v[4])), header="v0 vauf vab"
)
