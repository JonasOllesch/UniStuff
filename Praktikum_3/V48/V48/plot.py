import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import copy
from scipy import constants
from scipy.integrate import simpson

def methode_eins(T,c,W):
    return c - W/(constants.Boltzmann*T)

#def Hintergrund_func(x,a,b,c):
#    return np.exp(a*x+b)+c
#
def Hintergrund_func(x,a,b):
    return a*x+b

def pol1(x,a,b):
    return a*x+b
#my_file = open("Messdaten/Messung1.txt", "a") 
#for i in range(0,68):
#    my_file.write(str(i))
#    my_file.write('\n')

Zeit1, Temperatur1, Strom1 = np.genfromtxt('Messdaten/Messung1.txt',encoding='unicode-escape',unpack=True)
Temperatur1 = Temperatur1+constants.zero_Celsius
Strom1 = Strom1*(1e-11)
Daten_länge = len(Zeit1)
Heizrate1 = np.zeros(Daten_länge)
for i in range(1,Daten_länge):
    Heizrate1[i] = Temperatur1[i]-Temperatur1[i-1]

mask1 = np.logical_or(Temperatur1 <= -35+ constants.zero_Celsius, np.logical_and(Temperatur1 >= 0+ constants.zero_Celsius, Temperatur1 <= 6+ constants.zero_Celsius))
background_Tem = Temperatur1[mask1]
background_Str = Strom1[mask1]

x1 = np.linspace(-50+ constants.zero_Celsius,50+ constants.zero_Celsius,1000)
popt, pcov = curve_fit(Hintergrund_func,background_Tem,background_Str,maxfev=800)
para = correlated_values(popt,pcov)
y1 = Hintergrund_func(x1,*para)
plt.plot(unp.nominal_values(x1),unp.nominal_values(y1)*1e12,color='red')

mask2 = np.logical_and(Temperatur1 >= -35+ constants.zero_Celsius, Temperatur1 <= 6+ constants.zero_Celsius)
signal_Tem = Temperatur1[mask2]
signal_Str = Strom1[mask2]
signal_Str = signal_Str-Hintergrund_func(signal_Tem,*para)

plt.scatter(Temperatur1,Strom1*1e12,s=6,label='Messwerte')
plt.scatter(background_Tem,background_Str*1e12,s=6,c='green',label='Hintergrund')


plt.xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")
plt.ylim(0,8)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Messung1.pdf')
plt.clf()

#print(signal_Tem)
plt.errorbar(signal_Tem,unp.nominal_values(signal_Str*1e12),yerr=unp.std_devs(signal_Str*1e12),fmt='x',c='darkorange',label='Bereinigte Messwerte')
plt.xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")
plt.ylim(0,8)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Messung1_bereinigt.pdf')
plt.clf()


#Ansatz über die Polarisation
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem[signal_Str>0]))*1e-18,np.log(unp.nominal_values(signal_Str[signal_Str>0])),marker='x',c='darkorange',label='Werte')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem[1:12]))*1e-18,np.log(unp.nominal_values(signal_Str[1:12])),marker='x',c='red',label='Gefittete Werte')
x_pol1 = np.linspace(1/(constants.k*unp.nominal_values(signal_Tem[11])),1/(constants.k*unp.nominal_values(signal_Tem[1])),2)
popt, pcov = curve_fit(pol1,xdata=1/(constants.k*unp.nominal_values(signal_Tem[1:12])),ydata=np.log(unp.nominal_values(signal_Str[1:12])))
para_pol1 = correlated_values(popt, pcov)

y_pol1 = pol1(x_pol1,*para_pol1)
plt.plot(x_pol1*1e-18,unp.nominal_values(y_pol1),color='red')
plt.ylabel(r"$ \text{ln} \left(\dfrac{I}{\unit{\ampere}} \right) $")
plt.xlabel(r"$ \dfrac{1}{k_B T} \mathbin{/} \dfrac{a}{\unit{\joule}}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/pol1.pdf')
plt.clf()

signal_Str_pos = unp.nominal_values(signal_Str[signal_Str>0])
signal_Tem_pos = unp.nominal_values(signal_Tem[signal_Str>0])

f1 = np.zeros(len(signal_Str_pos))
for i in range(0,len(signal_Str_pos)):
    f1[i] = np.log(simpson(signal_Str_pos[i:],signal_Tem_pos[i:]))-np.log(unp.nominal_values(signal_Str_pos[i]))


signal_Tem_pos = signal_Tem_pos[f1>0]
f1 =f1[f1>0]


popt, pcov = curve_fit(pol1,1/(constants.k*unp.nominal_values(signal_Tem_pos[0:20])),np.log(f1[0:20]))
para_int1 = correlated_values(popt, pcov)
x_int1 = np.linspace((1/(constants.k*np.min(unp.nominal_values(signal_Tem_pos[0:20])))),(1/(constants.k*np.max(unp.nominal_values(signal_Tem_pos[0:20])))),2)
y_int1 = pol1(x_int1,*para_int1)

plt.plot(x_int1*1e-18,unp.nominal_values(y_int1),color='red',label='lin. Fit')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem_pos[21:]))*1e-18,np.log(f1) ,color='darkorange',s=6,marker='x',label='nicht gefittete Messwerte')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem_pos[0:20]))*1e-18,np.log(f1[0:20]) ,color='red',s=6,marker='x',label='Gefittete Messwerte')
plt.ylabel(r"$ \text{ln} f\left( T \right) $")
plt.xlabel(r"$ \dfrac{1}{k_B T} \mathbin{/} \dfrac{a}{\unit{\joule}}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/int1.pdf')
plt.clf()