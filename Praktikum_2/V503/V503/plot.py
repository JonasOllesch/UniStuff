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


def rad(etaL, vab, vauf):
    return unp.sqrt(
        (9 * etaL * abs(vab - vauf)) / (2 * const.g * (rho_O - rho_L))
    )

def eta_corr(etaL, r):
    return etaL * (1 / (1 + (6.17e-5 * 133.33) / (101325 * r)))


def q(eta_corr, vab, vauf, U):
    return (
        3
        * np.pi
        * eta_corr
        * unp.sqrt(9 / 4 * eta_corr / const.g * abs(vab - vauf) / (rho_O - rho_L))
        * (vab + vauf)
        / (U / dist.n)
    )

tab157, tauf157 = np.genfromtxt("data/U157.txt", unpack=True)
tab175, tauf175 = np.genfromtxt("data/U175.txt", unpack=True)
tab200, tauf200 = np.genfromtxt("data/U200.txt", unpack=True)
tab225, tauf225 = np.genfromtxt("data/U225.txt", unpack=True)
tab250, tauf250 = np.genfromtxt("data/U250.txt", unpack=True)

err = 0.2 * np.ones(15)

tab157  = unp.uarray(tab157, err)
tauf157 = unp.uarray(tauf157, err)

tab175 = unp.uarray(tab175, err)
tauf175 = unp.uarray(tauf175, err)

tab200 = unp.uarray(tab200, err)
tauf200 = unp.uarray(tauf200, err)

tab225 = unp.uarray(tab225, err)
tauf225 = unp.uarray(tauf225, err)

tab250 = unp.uarray(tab250, err)
tauf250 = unp.uarray(tauf250, err)

tab157 = tab157.reshape(5, 3)
tauf157 = tauf157.reshape(5, 3)
t157 = np.array([tauf157, tab157])

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

s = 0.5 * 1e-3

vauf157 = s / tauf157 
vab157 =  s /  tab157

print('vauf157',vauf157[0,:])

vauf175 = s / tauf175 
vab175 =  s / tab175

vauf200 = s / tauf200 
vab200 =  s / tab200

vauf225 = s / tauf225 
vab225 =  s / tab225

vauf250 = s / tauf250 
vab250 =  s / tab250

dist = ufloat(7.625, 0.0051) * 1e-3
rho_O = 886  # kg/m^3
rho_L = 1.225  # kg/m^3

etaL157 = 1.8475 * 1e-5
etaL175 = etaL157
etaL200  = etaL157
etaL225  = etaL157
etaL250  = 1.8525 *1e-5

rad157 = rad(etaL157, vauf157, vab157)
rad157 = rad157.mean(axis=1)

rad175 = rad(etaL175, vauf175, vab175)
rad175 = rad175.mean(axis=1)

rad200 = rad(etaL200, vauf200, vab200)
rad200 = rad200.mean(axis=1)

rad225 = rad(etaL225, vauf225, vab225)
rad225 = rad225.mean(axis=1)

rad250 = rad(etaL250, vauf250, vab250)
rad250 = rad250.mean(axis=1)

print('rad157',rad157)

etacorr157 = eta_corr(etaL157, rad157)
etacorr175 = eta_corr(etaL175, rad175)
etacorr200 = eta_corr(etaL200, rad200)
etacorr225 = eta_corr(etaL225, rad225)
etacorr250 = eta_corr(etaL250, rad250)
print('eta_corr157', etacorr157)
etacorr157 = np.mean(etacorr157) 
etacorr175 = np.mean(etacorr175)
etacorr200 = np.mean(etacorr200)
etacorr225 = np.mean(etacorr225)
etacorr250 = np.mean(etacorr250)

print(etacorr157)

q157 = q(etacorr157, vauf157, vab157, 157)
q175 = q(etacorr175, vauf175, vab175, 175)
q200 = q(etacorr200, vauf200, vab200, 200)
q225 = q(etacorr225, vauf225, vab225, 225)
q250 = q(etacorr250, vauf250, vab250, 250)

print('rad157.mean()',rad157.mean())

q157  = np.mean(q157,axis=1)
q175  = np.mean(q175,axis=1)
q200  = np.mean(q200,axis=1)
q225  = np.mean(q225,axis=1)
q250  = np.mean(q250,axis=1)
q_0157 = q157/(1+(6.17e-5 * 133.33)/(101325 * rad157))**(-2/3)
q_0175 = q175/(1+(6.17e-5 * 133.33)/(101325 * rad175))**(-2/3)
q_0200 = q200/(1+(6.17e-5 * 133.33)/(101325 * rad200))**(-2/3)
q_0225 = q225/(1+(6.17e-5 * 133.33)/(101325 * rad225))**(-2/3)
q_0250 = q250/(1+(6.17e-5 * 133.33)/(101325 * rad250))**(-2/3)

print(unp.nominal_values(q157))
print(unp.nominal_values(q175))

plt.errorbar(
    [1, 2, 3, 4, 5],
    unp.nominal_values(q157),
    xerr=None,
    yerr=unp.std_devs(q157),
    color="crimson",
    ms=4,
    marker="x",
    linestyle="",

)
plt.errorbar(
    [6, 7, 8, 9, 10],
    unp.nominal_values(q175),
    xerr=None,
    yerr=unp.std_devs(q175),
    color="crimson",
    ms=4,
    marker="x",
    linestyle="",

)
plt.errorbar(
    [11, 12, 13, 14, 15],
    unp.nominal_values(q200),
    xerr=None,
    yerr=unp.std_devs(q200),
    color="crimson",
    ms=4,
    marker="x",
    linestyle="",

)
plt.errorbar(
    [16,17,18,19,20],
    unp.nominal_values(q225),
    xerr=None,
    yerr=unp.std_devs(q225),
    color="crimson",
    ms=4,
    marker="x",
    linestyle="",
)
plt.errorbar(
    [21, 22, 23, 24, 25],
    unp.nominal_values(q250),
    xerr=None,
    yerr=unp.std_devs(q250),
    color="crimson",
    ms=4,
    marker="x",
    linestyle="",
)
plt.xlim(0,25.5)
plt.ylim(0,2*10**(-18))
plt.xlabel(r"Messung")
plt.ylabel(r"$e_0~\text{in}~10^{-19}~\unit{\coulomb}$")
plt.legend(loc="best")
plt.grid(linestyle="dashed")

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout()
plt.savefig("build/e")
plt.clf()

print('Ladung q157',q157)
print('Ladung q175',q175)
print('Ladung q200',q200)
print('Ladung q225',q225)
print('Ladung q250',q250)

q_0 = np.array([q_0157, q_0175,q_0200,q_0225,q_0250])
q_0 = q_0.reshape(1,25)
print('q_0',q_0)

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

e_0 = Euklid(q_0,1000)
print('e_0',np.mean(e_0))
N_a = 96485.3321233100184 / np.mean(e_0)

print('N_a',N_a)


for i in range (0,5):
    writeW(np.mean(vauf157[i,:]),'v_auf bei U = 157V für Teilchen i')

for i in range(0,5):
    writeW(np.mean(vab157[i,:]), 'v_ab bei U = 157V für Teilchen i')

writeW('nix','nächste Spannung')

for i in range (0,5):
    writeW(np.mean(vauf175[i,:]),'v_auf bei U = 175V für Teilchen i')

for i in range(0,5):
    writeW(np.mean(vab175[i,:]), 'v_ab bei U = 175V für Teilchen i')

for i in range (0,5):
    writeW(np.mean(vauf200[i,:]),'v_auf bei U = 200V für Teilchen i')

for i in range(0,5):
    writeW(np.mean(vab200[i,:]), 'v_ab bei U = 200V für Teilchen i')

for i in range (0,5):
    writeW(np.mean(vauf225[i,:]),'v_auf bei U = 225V für Teilchen i')

for i in range(0,5):
    writeW(np.mean(vab225[i,:]), 'v_ab bei U = 225V für Teilchen i')

for i in range (0,5):
    writeW(np.mean(vauf250[i,:]),'v_auf bei U = 250V für Teilchen i')

for i in range(0,5):
    writeW(np.mean(vab250[i,:]), 'v_ab bei U = 250V für Teilchen i')


for i in range (0,5):
    writeW(q_0157,'Ladung q_0 bei U = 157V')

for i in range (0,5):
    writeW(q_0175,'Ladung q_0 bei U = 175V')

for i in range (0,5):
    writeW(q_0200,'Ladung q_0 bei U = 200V')

for i in range (0,5):
    writeW(q_0225,'Ladung q_0 bei U = 225V')

for i in range (0,5):
    writeW(q_0250,'Ladung q_0 bei U = 250V')


nalit = 6.022141e23
qlit = 1.602176e-19


writeW((qlit - np.mean(q_0)) / qlit * 100, 'Abweichung der berechneten Elementarladung vom Litwert')
writeW((nalit - N_a) / nalit * 100, 'Abweichung der berechneten Avogadrokonstante vom Litwert')





