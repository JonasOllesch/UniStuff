import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
from multiprocessing import Process
import copy
from scipy import constants

"""
caption = 'meine Caption'
label = 'das label'
#header = 'Hier soll der Header stehen 4'
header = 'T \\mathbin{/} \\unit{\\second} & 1 & 2 & 3 & 4  \\\\'

make_table(data=data,caption=caption,label=label, header=header)
"""
def pol1(x,a,b):
    return a*x+b

def Magnetfeld_Spule(N,I,L):
    return (constants.mu_0*8*N*I)/(np.sqrt(125)*L)

Frequenz1, U_hor1, P1_V = np.genfromtxt('Messdaten/Messung1.txt',encoding='unicode-escape',unpack=True)
Frequenz1 *= 1000
U_hor1 *= (1e-3)
#Frequenz1 = unp.uarray(Frequenz1,0.1)
U_hor1 = unp.uarray(U_hor1,1e-4)
P1_V = unp.uarray(P1_V,0.01)
B1_ges = Magnetfeld_Spule(11,P1_V*0.1,0.1639) + Magnetfeld_Spule(154, U_hor1*2, 0.1579)

popt, pcov = curve_fit(pol1,unp.nominal_values(Frequenz1),unp.nominal_values(B1_ges))
para_Erdmagnetfeld1 = correlated_values(popt, pcov)

x_fit_1 = np.linspace(5*1e4,1.1*1e6,2)
y_fit_1 = pol1(x_fit_1,*para_Erdmagnetfeld1)


Frequenz2, U_hor2, P2_V = np.genfromtxt('Messdaten/Messung2.txt',encoding='unicode-escape',unpack=True)
Frequenz2 *= 1000
U_hor2 *= (1e-3)
#Frequenz1 = unp.uarray(Frequenz1,0.1)
U_hor2 = unp.uarray(U_hor2,1e-4)
P2_V = unp.uarray(P2_V,0.01)
B2_ges = Magnetfeld_Spule(11,P2_V*0.1,0.1639) + Magnetfeld_Spule(154, U_hor2*2, 0.1579)

popt, pcov = curve_fit(pol1,unp.nominal_values(Frequenz2),unp.nominal_values(B2_ges))
para_Erdmagnetfeld2 = correlated_values(popt, pcov)

x_fit_2 = np.linspace(5*1e4,1.1*1e6,2)
y_fit_2 = pol1(x_fit_2,*para_Erdmagnetfeld2)

print("para_Erdmagnetfeld1: ", para_Erdmagnetfeld1)
print("para_Erdmagnetfeld2: ", para_Erdmagnetfeld2)

print("B1_ges: ", B1_ges*1e6)
print("B2_ges: ", B2_ges*1e6)


plt.plot(x_fit_1/1e6, unp.nominal_values(y_fit_1)*1e6, c = 'darkorange', label = 'lin. Reg. 1. Peak')
plt.errorbar(unp.nominal_values(Frequenz1)/1e6,unp.nominal_values(B1_ges)*1e6,xerr=0, yerr=unp.std_devs(B1_ges),color = 'steelblue', fmt='x',label="1. Peak")


plt.plot(x_fit_2/1e6, unp.nominal_values(y_fit_2)*1e6, c = 'firebrick', label = 'lin. Reg. 2. Peak')
plt.errorbar(unp.nominal_values(Frequenz2)/1e6,unp.nominal_values(B2_ges)*1e6,xerr=0, yerr=unp.std_devs(B2_ges),color = 'darkolivegreen', fmt='x',label="2. Peak")


plt.ylabel(r"$ B \mathbin{/} \unit{\micro\tesla} $")
plt.xlabel(r"$f \mathbin{/} \si{\mega\hertz}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Erdmagnetfeld.pdf')
plt.clf()


gF1 = constants.Planck/(9.2740100783e-24*para_Erdmagnetfeld1[0])
gF2 = constants.Planck/(9.2740100783e-24*para_Erdmagnetfeld2[0])
print("gF1: ", gF1)
print("gF2: ", gF2)

KernS1 = 1/gF1 -1/2
KernS2 = 1/gF2 -1/2
print("KernS1: ", KernS1)
print("KernS2: ", KernS2)

def brechneQuadZee(gf, B, mf, DeltaHyper):
    return (gf*9.2740100783e-24*B) + (gf*9.2740100783e-24*B)**2*((1-2*mf)/DeltaHyper)

print("QuadZee 1:", brechneQuadZee(gF1, B1_ges[-1], 2, 4.53*1e-24 ))
print("QuadZee 2:", brechneQuadZee(gF2, B2_ges[-1], 3, 2.01*1e-24 ))
