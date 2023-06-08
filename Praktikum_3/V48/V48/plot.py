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
#my_file = open("Messdaten/Messung1.txt", "a+") 
#for i in range(0,68):
#    my_file.write(str(i))
#    my_file.write('\n')

#die erste Messung 
Zeit1, Temperatur1, Strom1 = np.genfromtxt('Messdaten/Messung1.txt',encoding='unicode-escape',unpack=True)
Temperatur1 = Temperatur1+constants.zero_Celsius
Strom1 = Strom1*(1e-11)
Daten_länge = len(Zeit1)
Heizrate1 = np.zeros(Daten_länge)
for i in range(1,Daten_länge):
    Heizrate1[i] = Temperatur1[i]-Temperatur1[i-1]

print("Heizrate1 : ", Heizrate1)
print("Mean der Heizrate1 ", np.mean(Heizrate1), "+/-" , np.std(Heizrate1))

mask1 = np.logical_or(Temperatur1 <= -35+ constants.zero_Celsius, np.logical_and(Temperatur1 >= 0+ constants.zero_Celsius, Temperatur1 <= 6+ constants.zero_Celsius))
background_Tem = Temperatur1[mask1]
background_Str = Strom1[mask1]

x1 = np.linspace(-50+ constants.zero_Celsius,50+ constants.zero_Celsius,1000)
popt, pcov = curve_fit(Hintergrund_func,background_Tem,background_Str,maxfev=800)
para = correlated_values(popt,pcov)
print("Parameter Hintergrund 1", para)
y1 = Hintergrund_func(x1,*para)
plt.plot(unp.nominal_values(x1),unp.nominal_values(y1)*1e12,color='darkorange')

mask2 = np.logical_and(Temperatur1 >= -35+ constants.zero_Celsius, Temperatur1 <= 6+ constants.zero_Celsius)
signal_Tem = Temperatur1[mask2]
signal_Str = Strom1[mask2]
signal_Str = signal_Str-Hintergrund_func(signal_Tem,*para)

plt.scatter(Temperatur1,Strom1*1e12,s=6,label='Messwerte',marker='x')
plt.scatter(background_Tem,background_Str*1e12,s=6,c='darkorange',label='Hintergrund',marker='x')
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
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem[0:14]))*1e-18,np.log(unp.nominal_values(signal_Str[0:14])),marker='x',c='red',label='Gefittete Werte')
x_pol1 = np.linspace(1/(constants.k*np.min(unp.nominal_values(signal_Tem[0:14]))),1/(constants.k*np.max(unp.nominal_values(signal_Tem[0:14]))),2)
popt, pcov = curve_fit(pol1,xdata=1/(constants.k*unp.nominal_values(signal_Tem[0:14])),ydata=np.log(unp.nominal_values(signal_Str[0:14])))
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
print(para_int1)

plt.plot(x_int1*1e-18,unp.nominal_values(y_int1),color='red',label='lin. Fit')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem_pos[21:]))*1e-18,np.log(f1[21:]) ,color='darkorange',s=6,marker='x',label='nicht gefittete Messwerte')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem_pos[0:20]))*1e-18,np.log(f1[0:20]) ,color='red',s=6,marker='x',label='Gefittete Messwerte')
plt.ylabel(r"$ \text{ln} f\left( T \right) $")
plt.xlabel(r"$ \dfrac{1}{k_B T} \mathbin{/} \dfrac{a}{\unit{\joule}}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/int1.pdf')
plt.clf()

#die zweite Messung 
Zeit2, Temperatur2, Strom2 = np.genfromtxt('Messdaten/Messung2.txt',encoding='unicode-escape',unpack=True)
Temperatur2 = Temperatur2+constants.zero_Celsius
Strom2 = Strom2*(1e-11)
Daten_länge2 = len(Zeit2)
Heizrate2 = np.zeros(Daten_länge2)
for i in range(1,Daten_länge2):
    Heizrate2[i] = Temperatur2[i]-Temperatur2[i-1]



mask2 = np.logical_or(Temperatur2 <= -35+ constants.zero_Celsius, np.logical_and(Temperatur2 >= 1+ constants.zero_Celsius, Temperatur2 <= 9+ constants.zero_Celsius))
background_Tem2 = Temperatur2[mask2]
background_Str2 = Strom2[mask2]

x2 = np.linspace(-50+ constants.zero_Celsius,50+ constants.zero_Celsius,1000)
popt, pcov = curve_fit(Hintergrund_func,background_Tem2,background_Str2,maxfev=800)
para = correlated_values(popt,pcov)
print("parameter Hintergrund 2", para)
y2 = Hintergrund_func(x2,*para)
plt.plot(unp.nominal_values(x2),unp.nominal_values(y2)*1e12,color='darkorange')

mask2 = np.logical_and(Temperatur2 >= -35+ constants.zero_Celsius, Temperatur2 <= 6+ constants.zero_Celsius)
signal_Tem2 = Temperatur2[mask2]
signal_Str2 = Strom2[mask2]
signal_Str2 = signal_Str2-Hintergrund_func(signal_Tem2,*para)

plt.scatter(Temperatur2,Strom2*1e12,s=6,label='Messwerte',marker='x')
plt.scatter(background_Tem2,background_Str2*1e12,s=6,c='darkorange',label='Hintergrund',marker='x')
plt.xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")
#plt.ylim(0,8)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Messung2.pdf')
plt.clf()

#print(signal_Tem2)
plt.errorbar(signal_Tem2,unp.nominal_values(signal_Str2*1e12),yerr=unp.std_devs(signal_Str2*1e12),fmt='x',c='darkorange',label='Bereinigte Messwerte')
plt.xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
plt.ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")
#plt.ylim(0,8)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Messung2_bereinigt.pdf')
plt.clf()


#Ansatz über die Polarisation
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem2[signal_Str2>0]))*1e-18,np.log(unp.nominal_values(signal_Str2[signal_Str2>0])),marker='x',c='darkorange',label='Werte')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem2[0:12]))*1e-18,np.log(unp.nominal_values(signal_Str2[0:12])),marker='x',c='red',label='Gefittete Werte')
x_pol2 = np.linspace(1/(constants.k*unp.nominal_values(signal_Tem2[11])),1/(constants.k*unp.nominal_values(signal_Tem2[0])),2)
popt, pcov = curve_fit(pol1,xdata=1/(constants.k*unp.nominal_values(signal_Tem2[0:12])),ydata=np.log(unp.nominal_values(signal_Str2[0:12])))
para_pol2 = correlated_values(popt, pcov)

y_pol2 = pol1(x_pol2,*para_pol2)
plt.plot(x_pol2*1e-18,unp.nominal_values(y_pol2),color='red')
plt.ylabel(r"$ \text{ln} \left(\dfrac{I}{\unit{\ampere}} \right) $")
plt.xlabel(r"$ \dfrac{1}{k_B T} \mathbin{/} \dfrac{a}{\unit{\joule}}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/pol2.pdf')
plt.clf()

signal_Str2_pos = unp.nominal_values(signal_Str2[signal_Str2>0])
signal_Tem2_pos = unp.nominal_values(signal_Tem2[signal_Str2>0])

f1 = np.zeros(len(signal_Str2_pos))
for i in range(0,len(signal_Str2_pos)):
    f1[i] = np.log(simpson(signal_Str2_pos[i:],signal_Tem2_pos[i:]))-np.log(unp.nominal_values(signal_Str2_pos[i]))

signal_Tem2_pos = signal_Tem2_pos[f1>0]
f1 =f1[f1>0]

popt, pcov = curve_fit(pol1,1/(constants.k*unp.nominal_values(signal_Tem2_pos[0:15])),np.log(f1[0:15]))
para_int2 = correlated_values(popt, pcov)
x_int2 = np.linspace((1/(constants.k*np.min(unp.nominal_values(signal_Tem2_pos[0:15])))),(1/(constants.k*np.max(unp.nominal_values(signal_Tem2_pos[0:15])))),2)
y_int2 = pol1(x_int2,*para_int2)

plt.plot(x_int2*1e-18,unp.nominal_values(y_int2),color='red',label='lin. Fit')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem2_pos[15:]))*1e-18,np.log(f1[15:]) ,color='darkorange',s=6,marker='x',label='nicht gefittete Messwerte')
plt.scatter(1/(constants.k*unp.nominal_values(signal_Tem2_pos[0:15]))*1e-18,np.log(f1[0:15]) ,color='red',s=6,marker='x',label='Gefittete Messwerte')
plt.ylabel(r"$ \text{ln} f\left( T \right) $")
plt.xlabel(r"$ \dfrac{1}{k_B T} \mathbin{/} \dfrac{a}{\unit{\joule}}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/int2.pdf')
plt.clf()




print("Heizrate1 : ", Heizrate1)
print("Mean der Heizrate1 ", np.mean(Heizrate1), "+/-" , np.std(Heizrate1))
print("Heizrate2 : ", *Heizrate2, sep='\n')
print("Mean der Heizrate2 ", np.mean(Heizrate2), "+/-" , np.std(Heizrate2))
print("Parameter aus dem Integralfit1",para_int1)
print("Parameter aus dem Integralfit2",para_int2)
print("Aktivierungsenergei 1 in eV ",para_int1[0]/constants.elementary_charge)
print("Aktivierungsenergei 2 in eV ",para_int2[0]/constants.elementary_charge)
print("Relaxationszeit aus int 1",repr(unp.exp(para_int1[1])))
print("Relaxationszeit aus int 2",repr(unp.exp(para_int2[1])))

print("para_pol1", para_pol1)
print("para_pol2", para_pol2)
print("Aktivierungsenergei 1 in eV ",para_pol1[0]/constants.elementary_charge)
print("Aktivierungsenergei 2 in eV ",para_pol2[0]/constants.elementary_charge)
print("T_M 1", signal_Tem[np.argmax(signal_Str)])
print("T_M 2", signal_Tem2[np.argmax(signal_Str2)])

def tau0(TM,b,W):
    print(TM)
    print(b)
    print(W)
    return (constants.k*(TM**2))/(b*W)*unp.exp(-W/(constants.k*TM))

H1u = ufloat(np.mean(Heizrate1), np.std(Heizrate1))/60
H2u = ufloat(np.mean(Heizrate2), np.std(Heizrate2))/60
print("tau pol1,", repr(tau0(signal_Tem[np.argmax(signal_Str)]  ,H1u,-para_pol1[0])))
print("tau pol2,", repr(tau0(signal_Tem2[np.argmax(signal_Str2)],H2u,-para_pol2[0])))

def tau(T, tau0, W):
    print("tau0", tau0)
    return tau0*unp.exp(W/(constants.k*T))
T = np.linspace(270,300,2000)
T_M1 = signal_Tem[np.argmax(signal_Str)]
T_M2 = signal_Tem2[np.argmax(signal_Str2)]
tau_pol_1 = tau(T,1.99*1e-7,-para_pol1[0])
tau_pol_2 = tau(T,3.75*1e-7,-para_pol2[0])
print("para_pol1[0]",para_pol1[0])
print("para_pol2[0]",para_pol2[0])
plt.plot(T,unp.nominal_values(tau_pol_1),label=r"$\tau(T)$ 1. Messung",color='red')
plt.plot(T,unp.nominal_values(tau_pol_2),label=r"$\tau(T)$ 2. Messung",color='darkorange')
plt.ylabel(r"$ \ln ( \tau \mathbin{/} \unit{\second} )$")
plt.xlabel(r"$ T \mathbin{/} \unit{\kelvin}$")
plt.yscale('log')
#plt.ylim(0,100)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/taupol.pdf')
plt.clf()

tau_int_1 = tau(T,3.57*1e-18,para_int1[0])
tau_int_2 = tau(T,0.89*1e-15,para_int2[0])
print("para_int1[0]",para_int1[0])
print("para_int2[0]",para_int2[0])
plt.yscale('log')
plt.plot(T,unp.nominal_values(tau_int_1),label=r"$\tau(T)$ 1. Messung",color='red')
plt.plot(T,unp.nominal_values(tau_int_2),label=r"$\tau(T)$ 2. Messung",color='darkorange')
#plt.ylim(0,100)
plt.ylabel(r"$ \ln (\tau \mathbin{/} \unit{\second} )$")
plt.xlabel(r"$ T \mathbin{/} \unit{\kelvin}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/tauint.pdf')
plt.clf()