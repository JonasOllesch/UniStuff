import numpy as np
import matplotlib.pyplot as plt
from uncertainties import correlated_values
import uncertainties.unumpy as unp
from scipy import constants
import copy
from scipy.optimize import curve_fit 

def pol1(x,a,b):
    return a*x+b

def rechne_ohm_zu_temperatur_celsius(Widerstand):
    return 0.00134*Widerstand**2+2.296*Widerstand -243.02

Kompressionsmodul = 123*10**9
Masse = 0.342  #Masse in kg 
tab1=np.array([[24.9430,   24.9310,   24.8930,   24.8310,   24.7450,   24.6340,   24.5000,   24.3430,   24.1630,   23.9610],
               [23.7390,   23.4970,   23.2360,   22.9560,   22.6600,   22.3480,   22.0210,   21.6800,   21.3270,   20.9630],
               [20.5880,   20.2050,   19.8140,   19.4160,   19.0120,   18.6040,   18.1920,   17.7780,   17.3630,   16.9470],
               [16.5310,   16.1170,   15.7040,   15.2940,   14.8870,   14.4840,   14.0860,   13.6930,   13.3050,   12.9230],
               [12.5480,   12.1790,   11.8170,   11.4620,   11.1150,   10.7750,   10.4440,   10.1190,   9.8030,    9.4950],
               [9.1950 ,   8.9030,    8.6190,    8.3420,    8.0740,    7.8140,    7.5610,    7.3160,    7.0780,    6.8480],
               [6.6250 ,   6.4090,    6.2000,    5.9980,    5.8030,    5.6140,    5.4310,    5.2550,    5.0840,    4.9195],
               [4.7606 ,   4.6071,    4.4590,    4.3160,    4.1781,    4.0450,    3.9166,    3.7927,    3.6732,    3.5580],
               [3.4468 ,   3.3396,    3.2362,    3.1365,    3.0403,    2.9476,    2.8581,    2.7718,    2.6886,    2.6083],
               [2.5309 ,   2.4562,    2.3841,    2.3146,    2.2475,    2.1828,    2.1203,    2.0599,    2.0017,    1.9455],
               [1.8912 ,   1.8388,    1.7882,    1.7393,    1.6920,    1.6464,    1.6022,    1.5596,    1.5184,    1.4785],
               [1.4400 ,   1.4027,    1.3667,    1.3318,    1.2980,    1.2654,    1.2337,    1.2031,    1.1735,    1.1448],
               [1.1170 ,   1.0900,    1.0639,    1.0386,    1.0141,    0.9903,    0.9672,    0.9449,    0.9232,    0.9021],
               [0.8817 ,   0.8618,    0.8426,    0.8239,    0.8058,    0.7881,    0.7710,    0.7544,    0.7382,    0.7225],
               [0.7072 ,   0.6923,    0.6779,    0.6638,    0.6502,    0.6368,    0.6239,    0.6113,    0.5990,    0.5871],
               [0.5755 ,   0.5641,    0.5531,    0.5424,    0.5319,    0.5210,    0.5117,    0.5020,    0.4926,    0.4834]])

#Widerstand in Ohm, Strom  in mA , Spannung in V, Zeit in hh, mm ,ss
Widerstand, Strom , Spannung, Stunden, Minuten, Sekunden = np.genfromtxt('Messdaten/Messdaten.txt',encoding='unicode-escape',dtype=None,unpack=True)
länge = len(Widerstand)
Strom  = Strom /1000
Zeit = Stunden*3600 + Minuten*60 + Sekunden
Zeit = unp.uarray(Zeit,0.5)
Widerstand = unp.uarray(Widerstand,0.5)
Spannung = unp.uarray(Spannung,Spannung/100)
Strom = unp.uarray(Strom,Strom/100)

Delta_t = copy.deepcopy(Zeit) # der Unterschied in der Zeit


for i in range (0,länge-1):
    Delta_t[i] = Zeit[i+1]-Zeit[i]

Delta_t[länge-1] = 0


Delta_E = Spannung*Strom*Delta_t
Temperatur = rechne_ohm_zu_temperatur_celsius(Widerstand)
Temperatur = Temperatur + constants.zero_Celsius
#print("Temperatur in C", Temperatur[:10])


Delta_T = copy.deepcopy(Widerstand) # der Unterschied in der Temperatur
for i in range(0,länge-1):
    Delta_T[i] = Temperatur[i+1]-Temperatur[i]

Delta_T[länge-1] = 0  #der letzte Eintrag kann nicht berechnet werden, da es keine nächste Temperatur gibt 
#Berechnung der Wärmekapazität C = (Delta E) / (Delta T)
MolareMasseCu =63.546/1000
C_p = copy.deepcopy(Widerstand)
C_p[:länge-1] = Delta_E[:länge-1]/Delta_T[:länge-1]*MolareMasseCu/Masse
C_p[länge-1] = 0

linAusKoef_T_tab = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300])
linAusKoef_tab = np.array([7.00, 8.50, 9.75, 10.70, 11.50, 12.10, 12.65, 13.15, 13.60, 13.90, 14.25, 14.50, 14.75, 14.95, 15.20, 15.40, 15.60, 15.75, 15.90, 16.10, 16.25, 16.35, 16.50, 16.65])
linAusKoef_tab = linAusKoef_tab*10**(-6)
linAusKoef = np.interp(x=unp.nominal_values(Temperatur),xp=linAusKoef_T_tab,fp=linAusKoef_tab)
#print(linAusKoef)
#C_p - C_V = 9*a**2*K*V_0*T
Molvolumen = 0.063546 /8933.0
C_v = C_p - 9*linAusKoef**2*Kompressionsmodul*Molvolumen*Temperatur




plt.errorbar(unp.nominal_values(Temperatur[:länge-1]),unp.nominal_values(C_p[:länge-1]),xerr=unp.std_devs(C_p[:länge-1]), yerr=unp.std_devs(Temperatur[:länge-1]),elinewidth=1,capthick=1,markersize=3,color = 'blue', fmt='x',label=r"$ C_\text{p}$")
plt.errorbar(unp.nominal_values(Temperatur[:länge-1]),unp.nominal_values(C_v[:länge-1]),xerr=unp.std_devs(C_v[:länge-1]), yerr=unp.std_devs(Temperatur[:länge-1]),elinewidth=1,capthick=1,markersize=3,color = 'red', fmt='x',label=r"$ C_\text{v}$")

plt.hlines(3*constants.R,xmin=65,xmax=280,color='green',linestyles='dashed',label='Dulong-Petit Gesetzt')
plt.ylabel(r"$ C \mathbin{/} \unit{\dfrac{\joule}{\kelvin \, \mol}} $")
plt.xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/heat_kapazität.pdf')
plt.clf()

def find_nearest(a0):
    idx = (np.abs(tab1 - a0)).argmin()
    w = tab1.shape[1]
    i = idx // w
    j = idx - i * w
    return float(f"{i}.{j}")
theta_dT = np.zeros(länge)

for i in range(länge):
    theta_dT[i] = find_nearest(C_v[i])
print(theta_dT[:10])#die Werte :10 liegen unter 170K wie in der Anleitung gefordert der 0. Wert wird herrausgenommen, da er zuweit außerhalb liegt


theta = theta_dT*Temperatur
print(theta[1:10])
print(np.mean(unp.nominal_values(theta[1:10])))
print(np.sqrt(np.sum(unp.std_devs(theta[1:10])**2)))

plt.scatter(unp.nominal_values(C_v[1:10]),theta_dT[1:10],label='Debye-Funktion')
plt.xlabel(r"$ C_{\text{p}} \mathbin{/} \unit{\dfrac{\joule}{\kelvin\mol}} $")
plt.ylabel(r"$\dfrac{\Theta_\text{d}}{T} $")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Debye_Funktion.pdf')
plt.clf()
