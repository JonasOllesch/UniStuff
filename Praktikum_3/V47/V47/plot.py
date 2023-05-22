import numpy as np
import matplotlib.pyplot as plt
from uncertainties import correlated_values
import uncertainties.unumpy as unp
from scipy import constants
import copy

def rechne_ohm_zu_temperatur_celsius(R):
    return 0.00134*R**2+2.296*R- 243.02#

m = 0.342  #Masse in kg 
#Widerstand in Ohm, Strom  in mA , Spannung in V, Zeit in hh, mm ,ss
Widerstand, Strom , Spannung, Stunden, Minuten, Sekunden = np.genfromtxt('Messdaten/Messdaten.txt',encoding='unicode-escape',dtype=None,unpack=True)
länge = len(Widerstand)
Strom  = Strom /1000
Zeit = Stunden*3600 + Minuten*60 + Sekunden
Zeit = unp.uarray(Zeit,1)
Widerstand = unp.uarray(Widerstand,1)
Spannung = unp.uarray(Spannung,Spannung/100)
Strom = unp.uarray(Strom,Strom/100)

Delta_t = copy.deepcopy(Zeit) # der Unterschied in der Zeit
print(type(Zeit[0]))
print(type(Delta_t[0]))


for i in range (0,länge-1):
    Delta_t[i+1] = Zeit[i+1]-Zeit[i]
print(Delta_t)
Energie = Spannung*Strom*Delta_t
print(Energie)
Temperatur = rechne_ohm_zu_temperatur_celsius(Widerstand)

Delta_T = copy.deepcopy(Widerstand) # der Unterschied in der Temperatur
for i in range(0,länge-1):
    Delta_T[i] = Temperatur[i+1]-Temperatur[i]

#Berechnung der Wärmekapazität C = \Delta E / \Delta T
Wärmekapazität = copy.deepcopy(Widerstand)


#plt.errorbar(Zeit,gefüllte_Kanäle,yerr=np.sqrt(gefüllte_Kanäle),elinewidth=0.2,capthick=1,markersize=2,color = 'green', fmt='x')
#plt.plot(x_zeit,y_zeit,color='red',label="gefittete Funktion")
#plt.fill_between(Zeit,gefüllte_Kanäle,step="pre",alpha=1,label="Histogram der Myonenzerfälle")
#plt.ylabel("Zählung")
#plt.xlabel(r"$t \mathbin{/} \unit{\micro\second}$")
#plt.grid(linestyle = ":")
#plt.tight_layout()
#plt.legend()
#plt.savefig('build/Wärmekapazität.pdf')
#plt.clf()