import numpy as np
import matplotlib.pyplot as plt
from uncertainties import correlated_values
import uncertainties.unumpy as unp
from scipy import constants

def rechne_ohm_zu_temperatur_celsius(R):
    return 0.00134*R**2+2.296*R- 243.02#

#Widerstand in Ohm, Strom  in mA , Spannung in V, Zeit in hh, mm ,ss
Widerstand, Strom , Spannung, Stunden, Minuten, Sekunden = np.genfromtxt('Messdaten/Messdaten.txt',encoding='unicode-escape',dtype=None,unpack=True)
Strom  = Strom /1000
Zeit = Stunden*3600 + Minuten*60 + Sekunden
Zeit = unp.uarray(Zeit,1)
Widerstand = unp.uarray(Widerstand,1)
Spannung = unp.uarray(Spannung,Spannung/100)
Strom = unp.uarray(Strom,Strom/100)

Delta_t = np.zeros((len(Widerstand)))


for i in range (0,len(Widerstand)-1):
    Delta_t[i] = Zeit[i+1]-Zeit[i]

Energie = Spannung*Strom*Delta_t

#KK10u = unp.uarray(KK10[:,2],np.sqrt(KK10[:,2]))