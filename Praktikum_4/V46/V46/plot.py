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

d = np.array([1.296, 1.36,   5.11])
d = d/1000
N = np.array([2.8*10**18,1.2*10**18, 0])
N = N*10**3

   

plt.scatter(Magnetfeld[:,0]*1000, Magnetfeld[:,1]*1000,label = 'Flussdichte',  c='navy', marker = 'x', s = 20)
plt.hlines(405,xmin=85, xmax=112, colors='navy', linestyles='dotted')
plt.legend(loc='upper left')
plt.xlim(85,112)
plt.xlabel(r"$ z \mathbin{/} \unit{\milli\meter} $")
plt.ylabel(r"$B \mathbin{/} \unit{\milli\tesla}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/BFeld.pdf')
plt.clf()
