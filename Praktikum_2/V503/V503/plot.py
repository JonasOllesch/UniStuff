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

ar = np.ones(5)
etaL157 = 1.9525 * 1e-5
etaL175 = etaL157
etaL200  = etaL157
etaL225  = etaL157
etaL250  = 1.865 *1e-5

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

etacorr157 = eta_corr(etaL157,rad157)
etacorr175 = eta_corr(etaL175,rad175)
etacorr200 = eta_corr(etaL200,rad200)
etacorr225 = eta_corr(etaL225,rad225)
etacorr250 = eta_corr(etaL250,rad250)
print('eta_corr157', etacorr157)
etacorr157 = etacorr157.mean() 
etacorr175 = etacorr175.mean()
etacorr200 = etacorr200.mean()
etacorr225 = etacorr225.mean()
etacorr250 = etacorr250.mean()

print(etacorr157)

q157 = q(etacorr157, vauf157, vab157, 157)
q175 = q(etacorr175, vauf175, vab175, 175)
q200 = q(etacorr200, vauf200, vab200, 200)
q225 = q(etacorr225, vauf225, vab225, 225)
q250 = q(etacorr250, vauf250, vab250, 250)

q157 = q157.mean()

print(q157)


