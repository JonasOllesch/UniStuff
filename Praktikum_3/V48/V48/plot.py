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

plt.scatter(Temperatur,Strom*(1e12),marker='x',s=3,label='Messdaten')
#plt.plot(unp.nominal_values(x_fit),unp.nominal_values(y_fit),color='red',label='Fit')
plt.xlabel(r"$ T  \mathbin{/}  \unit{\celsius} $")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere} $")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Messdaten.pdf')
plt.clf()

Strom = np.log(Strom)
#methode_eins(T,c,W)
popt_m_1, pcov_m_1 = curve_fit(methode_eins,Temperatur[1:Daten_länge-4],Strom[1:Daten_länge-4],p0=[5,10])
para_m_1 = correlated_values(popt_m_1, pcov_m_1)
x_fit = np.linspace(200,325,500)
y_fit = methode_eins(x_fit,*para_m_1)
print(para_m_1)
y_fit_test = methode_eins(x_fit,-14.556,4.156)
#plt.plot(x_fit,unp.nominal_values(y_fit_test),color='green',label='parameter test')


plt.scatter(Temperatur,Strom,marker='x',s=3,label='Messdaten')
plt.plot(x_fit,unp.nominal_values(y_fit),color='red',label='Fit')
plt.xlabel(r"$ T  \mathbin{/}  \unit{\celsius} $")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere} $")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/log_Messdaten_m1.pdf')
plt.clf()