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
ZScann1 = np.genfromtxt('Messdaten/Z1Scan.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
RockingCurve1 = np.genfromtxt('Messdaten/RockingCurve2_1.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
Omega2Theta = np.genfromtxt('Messdaten/Omega2Theta.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
Diffus = np.genfromtxt('Messdaten/Diffus.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')



def plotte_Z1Scan(Z1Scan):
    plt.scatter(Z1Scan[:,0], Z1Scan[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.xlabel(r"$z \mathbin{/} \unit{\milli\meter}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Z1Scann.pdf')
    plt.clf()


def plotte_Detektorscan(Detektorscann):
    plt.scatter(Detektorscann[:,0], Detektorscann[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Detectorscan.pdf')
    plt.clf()

def plotte_RockingCurve(RockingCurve):
    plt.scatter(RockingCurve[:,0], RockingCurve[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Rockingcurve.pdf')
    plt.clf()

def plotte_Omega2Theta(Omega2Theta):
#    plt.scatter(Omega2Theta[:,0], Omega2Theta[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.plot(Omega2Theta[:,0], Omega2Theta[:,1], label = "Data", c = "midnightblue")
    plt.yscale('log')
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Omega2Theta.pdf')
    plt.clf()

def plotte_Diffus(Diffus):
#    plt.scatter(Diffus[:,0], Diffus[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.plot(Diffus[:,0], Diffus[:,1], label = "Data", c = "midnightblue")

    plt.yscale('log')
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Diffus.pdf')
    plt.clf()


Processe = []
Processe.append(Process(target=plotte_Detektorscan, args=([Detektorscann])))
Processe.append(Process(target=plotte_Z1Scan, args=([ZScann1])))
Processe.append(Process(target=plotte_RockingCurve, args=([RockingCurve1])))
Processe.append(Process(target=plotte_Omega2Theta, args=([Omega2Theta])))
Processe.append(Process(target=plotte_Diffus, args=([Diffus])))


#p1 = Process(target=plotte_Detektorscan, args=([Detektorscann]))
#p2 = Process(target=plotte_Detektorscan, args=([ZScann1]))

for p in Processe:
    p.start()

for p in Processe:
    p.join()


#p1 = Process(target=plotte_Detektorscan, args=([Detektorscann]))
##p2 = Process(target=plote_n_vs_p, args=(Messreihe_Druck[:,0], Brechungsindex_Luft_arr))
#p2.start()
#p1.start()
##p2.join()
#p1.join()