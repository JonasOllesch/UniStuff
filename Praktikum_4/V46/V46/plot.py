import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

#from multiprocessing  import Process


Messreihe1 = np.genfromtxt('Messdaten/P1.txt', encoding='unicode-escape')
Messreihe2 = np.genfromtxt('Messdaten/P2.txt', encoding='unicode-escape')
Messreihe3 = np.genfromtxt('Messdaten/P3.txt', encoding='unicode-escape')

Magnetfeld = np.genfromtxt('Messdaten/BFeld.txt', encoding='unicode-escape')
Magnetfeld = Magnetfeld/1000

Winkel1 = np.zeros((len(Messreihe1[:,0]),2))
Winkel2 = np.zeros((len(Messreihe1[:,0]),2))
Winkel3 = np.zeros((len(Messreihe1[:,0]),2))

Winkel1[:,0] = Messreihe1[:,1]+ Messreihe1[:,2]/60
Winkel1[:,1] = Messreihe1[:,3]+ Messreihe1[:,4]/60


Winkel2[:,0] = Messreihe2[:,1]+ Messreihe2[:,2]/60
Winkel2[:,1] = Messreihe2[:,3]+ Messreihe2[:,4]/60

Winkel3[:,0] = Messreihe3[:,1]+ Messreihe3[:,2]/60
Winkel3[:,1] = Messreihe3[:,3]+ Messreihe3[:,4]/60


Winkel1 = Winkel1 * np.pi/180
Winkel2 = Winkel2 * np.pi/180
Winkel3 = Winkel3 * np.pi/180

print(Winkel1)
Wellenlänge = Messreihe1[:,0]/(10**6)


d = np.array([1.296, 1.36,   5.11])
d = d/1000
N = np.array([2.8*10**18,1.2*10**18, 0])
N = N*10**3

diffWinkel = np.zeros((len(Wellenlänge),3))


diffWinkel[:,0] = 1/(2*d[0]) *(Winkel1[:,1] -Winkel1[:,0])
diffWinkel[:,1] = 1/(2*d[1]) *(Winkel2[:,1] -Winkel2[:,0])
diffWinkel[:,2] = 1/(2*d[2]) *(Winkel3[:,1] -Winkel3[:,0])

print(diffWinkel*180/np.pi)
   

plt.scatter(Magnetfeld[:,0]*1000, Magnetfeld[:,1]*1000,label = 'Flussdichte',  c='navy', marker = 'x', s = 20)
plt.hlines(405,xmin=85, xmax=112, colors='navy', linestyles='dotted')
plt.xlim(85,112)
plt.xlabel(r"$ z \mathbin{/} \unit{\milli\meter} $")
plt.ylabel(r"$B \mathbin{/} \unit{\milli\tesla}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig('build/BFeld.pdf')
plt.clf()
