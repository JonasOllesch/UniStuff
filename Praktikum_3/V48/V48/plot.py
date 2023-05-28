import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import copy
from scipy import constants

def methode_eins(T,c,W):
    return c - W/(constants.Boltzmann*T)

Zeit, Temperatur, Strom, b = np.genfromtxt('Messdaten/Not_my_data.txt',encoding='unicode-escape',unpack=True)
del b
Strom = Strom*(1e-12)
Daten_länge = len(Zeit)
b = np.zeros(Daten_länge)
for i in range(1,Daten_länge):
    b[i] = Temperatur[i]-Temperatur[i-1]

Temperatur = Temperatur+constants.zero_Celsius
popt_m_1, pcov_m_1 = curve_fit(methode_eins,Temperatur[Daten_länge-5:],np.log(Strom[Daten_länge-5:]))
para_m_1 = correlated_values(popt_m_1, pcov_m_1)
x_fit = np.linspace(-80,80,1000)
y_fit = methode_eins(x_fit,*para_m_1)


plt.scatter(Temperatur,Strom*(1e12),marker='x',s=3,label='Messdaten')
#plt.plot(unp.nominal_values(x_fit),unp.nominal_values(y_fit),color='red',label='Fit')
plt.xlabel(r"$ T  \mathbin{/}  \unit{\celsius} $")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere} $")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Messdaten.pdf')
plt.clf()
