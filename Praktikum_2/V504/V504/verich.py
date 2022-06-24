import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib widget
import numpy as np
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray
from uncertainties import unumpy as unp 
import scipy.constants as const

Tabelle: Kennlinien

md1 = pd.read_csv('tables/md1.csv')
print(md1.to_latex(index = False, column_format= "c c c c c c", decimal=',')) 

Daten auswerten:

np.savetxt('tables/md1.txt', md1.values, fmt='%.3f')
U, k1, k2, k3, k4, k5 = np.genfromtxt('tables/md1.txt', unpack=True)

Plot 1:

plt.plot(U, k1, 'xr', label = "Kennlinie 1", zorder=2)

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)

plt.show()

Plot 2:

plt.plot(U, k2, 'xr', label = "Kennlinie 2")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)

Plot 3:

plt.plot(U, k3, 'xr', label = "Kennlinie 3")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)   

Plot 4:

plt.plot(U, k4, 'xr', label = "Kennlinie 4")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)   

Plot 5:

Plot 5: Berechnung der Parameter mithilfe des Raumladungsgesetzes

plt.plot(U, k5, 'xr', label = "Kennlinie 5", zorder=2, alpha = 0.7)

def lsr(u, a, b):
    return (4/9) * const.epsilon_0 * np.sqrt(2 * const.elementary_charge/ const.electron_mass) * u**(b)/a**2

para, pcov = curve_fit(lsr, U[:15], k5[:15])
a, b = para
fa, fb = np.sqrt(np.diag(pcov))

ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 280, 100)
plt.plot(xx, lsr(xx, noms(ua), noms(ub)), '-b', label = "Ausgleichsfunktion", linewidth = 1, zorder=1)

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)
plt.xlim(-10, 150)
plt.ylim(-0.1, 3)

ua = ua*10**3
print(1/ua**2)
print(ub)
print('a = (%.3f +- %.3f)*10-3' % (noms(ua), stds(ua)))
print('b = (%.3f +- %.3f)' % (noms(ub), stds(ub)))

abw = 1-noms(ub)/1.5
print('Abweichung von b = 1,5:', np.round(abw, 2), '%')

0.0033+/-0.0004
1.232+/-0.031
a = (-17.475 +- 1.148)*10-3
b = (1.232 +- 0.031)
Abweichung von b = 1,5: 0.18 %

Tabelle Anlaufstrom:

md2 = pd.read_csv('tables/md2.csv')
print(md2.to_latex(index = False, column_format= "c c", decimal=',')) 

np.savetxt('tables/md2.txt', md2.values, fmt='%.4f')
U, I = np.genfromtxt('tables/md2.txt', unpack=True)
U = U + I * 1e-9 * 1e6     # Korregierte Spannung wegen Widerstand von 1MOhm

plt.plot(U, I, 'xr', label = "Messdaten",  alpha = 0.7)

def f(u, a, b):
    return a * np.exp(-u / b)

para, pcov = curve_fit(f, U, I)
a, b = para
fa, fb = np.sqrt(np.diag(pcov))

ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(-0.5, 1.5, 100)
plt.plot(xx, f(xx, a, b), '-b', label = "Ausgleichsfunktion")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{nA}$')
plt.legend(loc="best")     
plt.grid(True)
plt.xlim(0, 1.05)
plt.ylim(-0.5, 9)

ub = const.elementary_charge * ub / const.Boltzmann # nach umformen im exponenten

print(a)
print(b)

print('a = (%.3f +- %.3f) nA' % (noms(ua), stds(ua)))
print('b = (%.3f +- %.3f) K' % (noms(ub), stds(ub)))

14.211735029260199
0.17708700028655838
a = (14.212 +- 0.252) nA
b = (2055.009 +- 62.069) K

Temperaturen berechnen: c)

md3 = pd.read_csv('tables/md3.csv')
np.savetxt('tables/md3.txt', md3.values, fmt='%.1f')
I, U = np.genfromtxt('tables/md3.txt', unpack=True)

def temp(u, i): # Berechnung der Temperatur mit U und I
    f = 0.32; eta = 0.28; sigma = 5.7e-12; Nwl =1
    return ( (u * i - Nwl) / (f*eta*sigma) )**(1/4)

T = temp(U, I)
T = np.round(T, 2)

z = {'U/V': U, 'I/A': I, 'T/K': T}
dz = pd.DataFrame(data=z)
print(dz.to_latex(index = False, column_format= "c c c", decimal=',')) 

Berechnung der Austrittsarbeit: e)

def a(t):   # Austrittsarbeit
    kb = const.Boltzmann
    e0 = const.elementary_charge
    m0 = const.electron_mass
    h = const.Planck
    Is = [0.12, 0.27, 0.6, 1.2, 2.1]
    Is = np.multiply(Is, 1000)    # Sättigungstrom in Ampere
    f = 0.32e-2                   # Fläche in m
    js = Is/f                     # Sättigungsstromdichte
    return - T * kb * np.log( js * h**3 / (4*np.pi*e0*m0*kb**2 * T**2) )

WJ = a(T)
WeV = a(T)/const.e
print('W/J =', WJ)
print('W/eV =', WeV)
print('Mittelwert und Fehler: (%.3f +- %.3f) eV' % (np.mean(WeV), np.std(WeV)), '\n')

K = ['Kennlinie 1', 'Kennlinie 2', 'Kennlinie 3', 'Kennlinie 4', 'Kennlinie 5']
z = {' ': K, 'Wa/eV': np.round(a(T)/const.e, 3), 'Wa/J*10**-19': np.round(a(T)*10**19, 3)}
dz = pd.DataFrame(data=z)
print(dz.to_latex(index = False, column_format= "c c c", decimal=',')) 

W/J = [5.04339135e-19 5.03374872e-19 4.91301452e-19 4.92454513e-19
 5.06636062e-19]
W/eV = [3.14783729 3.14181883 3.06646247 3.07365931 3.16217358]
Mittelwert und Fehler: (3.118 +- 0.040) eV 
