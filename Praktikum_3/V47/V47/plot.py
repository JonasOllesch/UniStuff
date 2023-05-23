import numpy as np
import matplotlib.pyplot as plt
from uncertainties import correlated_values
import uncertainties.unumpy as unp
from scipy import constants
import copy

def rechne_ohm_zu_temperatur_celsius(Widerstand):
    return 0.00134*Widerstand**2+2.296*Widerstand -243.02


Kompressionsmodul = 123*10**9
Masse = 0.342  #Masse in kg 
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
Delta_T = copy.deepcopy(Widerstand) # der Unterschied in der Temperatur
for i in range(0,länge-1):
    Delta_T[i] = Temperatur[i+1]-Temperatur[i]

Delta_T[länge-1] = 0  #der letzte Eintrag kann nicht berechnet werden, da es keine nächste Temperatur gibt 
#Berechnung der Wärmekapazität C = (Delta E) / (Delta T)
MolareMasseCu =63.546/1000
C_p = copy.deepcopy(Widerstand)
C_p[:länge-1] = Delta_E[:länge-1]/Delta_T[:länge-1]*MolareMasseCu/Masse
C_p[länge-1] = 0
#C_p - C_V = 9*a**2*K*V_0*T

plt.errorbar(unp.nominal_values(Temperatur[:länge-1]),unp.nominal_values(C_p[:länge-1]),xerr=unp.std_devs(C_p[:länge-1]), yerr=unp.std_devs(Temperatur[:länge-1]),elinewidth=1,capthick=1,markersize=3,color = 'blue', fmt='x',label=" isobare Wärmekapazität")
plt.ylabel(r"$ C_{\text{p}} \mathbin{/} \unit{\dfrac{\joule}{\kelvin\mol}} $")
plt.xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/C_p.pdf')
plt.clf()


linAusKoef_T_tab = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300])
linAusKoef_tab = np.array([7.00, 8.50, 9.75, 10.70, 11.50, 12.10, 12.65, 13.15, 13.60, 13.90, 14.25, 14.50, 14.75, 14.95, 15.20, 15.40, 15.60, 15.75, 15.90, 16.10, 16.25, 16.35, 16.50, 16.65])
linAusKoef_tab = linAusKoef_tab*10**(-6)
linAusKoef = np.interp(x=unp.nominal_values(Temperatur),xp=linAusKoef_T_tab,fp=linAusKoef_tab)
#print(linAusKoef)
#C_v = C_p - 9*linAusKoef**2*Kompressionsmodul*Molvolumen*Temperatur