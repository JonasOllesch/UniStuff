import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

from multiprocessing  import Process


def Gaus(x, a, mu, sigma):
    return a/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2) )


Detektorscann = np.genfromtxt('Messdaten/GaussScan.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape') 


def plotte_Detektorscan(Detektorscann):
    plt.scatter(Detektorscann[:,0], Detektorscann[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    print("--------------")
    plt.savefig('build/Detectorscan.pdf')
    plt.clf()

p1 = Process(target=plotte_Detektorscan, args=([Detektorscann]))
#p2 = Process(target=plote_n_vs_p, args=(Messreihe_Druck[:,0], Brechungsindex_Luft_arr))
#p2.start()
p1.start()
#p2.join()
p1.join()