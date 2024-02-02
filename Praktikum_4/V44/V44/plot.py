import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import peak_widths
import scipy.constants as constants

from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp

from multiprocessing  import Process


def Gaus(x, a, mu, sigma, b):
    return a/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2) ) + b


def uGaus(x, a, mu, sigma, b):
    return a/(sigma * unp.sqrt(2 * np.pi)) * unp.exp( - (x - mu)**2 / (2 * sigma**2) ) + b



Detektorscann = np.genfromtxt('Messdaten/GaussScan.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape') 
ZScann1 = np.genfromtxt('Messdaten/Z2raw.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
RockingCurve = np.genfromtxt('Messdaten/RockingCurve1raw.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
Omega2Theta = np.genfromtxt('Messdaten/Omega2Theta.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
Diffus = np.genfromtxt('Messdaten/Diffus.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')

#Detektorscan
#Fit der Gauskurve
x_fit_Dek = np.linspace(-0.5,0.5, 10000)

popt, pcov = curve_fit(Gaus, Detektorscann[:,0], Detektorscann[:,1], absolute_sigma = True ,p0=[1.5e05, 0, 0.05,  200])

para_Detektorscan = correlated_values(popt, pcov)


#Brechnung von FWHM
y_fit_Dek = Gaus(x_fit_Dek, *popt)


Gaus_Peak_idx = np.array([np.argmax(unp.nominal_values(y_fit_Dek))])

Gaus_Peak_FWHM_in_idx, tmp, tmp1, tmp2 = peak_widths(unp.nominal_values(y_fit_Dek), Gaus_Peak_idx, rel_height=0.5)
Gaus_Peak_FWHM = x_fit_Dek[int(Gaus_Peak_FWHM_in_idx)] - x_fit_Dek[0]



print(f'Die Parameter des Gaus Detektorscan a, mu, sigma, b {para_Detektorscan}')
print(f'Die FWHM des Detektorscan {Gaus_Peak_FWHM} in Grad')
print(f'Das Maximum des Fits: {uGaus(x_fit_Dek[Gaus_Peak_idx], *para_Detektorscan)}')


#der Z-Scan
lineOffset = 56
SampleStart = 87
SampleEnd = 97
Z_Scan_width = ZScann1[SampleEnd-lineOffset, 0] - ZScann1[SampleStart-lineOffset, 0]
Z_Scan_width = ufloat(Z_Scan_width/1000, 0.02/1000)

def plotte_Z1Scan(Z1Scan1, SampleStart, SampleEnd):
    lineOffset = 56
    plt.scatter(Z1Scan1[:,0], Z1Scan1[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)

    plt.scatter(Z1Scan1[SampleStart-lineOffset, 0], Z1Scan1[SampleStart-lineOffset ,1], c = "firebrick", marker='x', s = 10)
    plt.scatter(Z1Scan1[SampleEnd-lineOffset, 0], Z1Scan1[SampleEnd-lineOffset ,1], c = "firebrick", marker='x', s = 10)


    print(f'Die Breite des Beams in z-scan {Z1Scan1[SampleEnd-lineOffset, 0] - Z1Scan1[SampleStart-lineOffset, 0]} +- 0.02 / mm ')


    plt.vlines(Z1Scan1[SampleStart-lineOffset, 0], 0, 1.25e6, colors='firebrick', label = "Beam boundaries", linestyles='dashed')
    plt.vlines(Z1Scan1[SampleEnd-lineOffset, 0], 0, 1.25e6, colors='firebrick', linestyles='dashed')



    plt.xlabel(r"$z \mathbin{/} \unit{\milli\meter}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.savefig('build/Z1Scann.pdf')
    plt.clf()

def plotte_Detektorscan(Detektorscann, x_fit_Dek, y_fit_Dek, Gaus_Peak_idx, Gaus_Peak_FWHM_in_idx):
    tmp_x = np.array([(Gaus_Peak_idx-Gaus_Peak_FWHM_in_idx/2), (Gaus_Peak_idx+Gaus_Peak_FWHM_in_idx/2)])
    tmp_x = tmp_x.astype(int)
    tmp_y = y_fit_Dek[tmp_x]
    
    plt.plot(x_fit_Dek[tmp_x], tmp_y, label = 'FWHM', color = 'black')
    plt.plot(x_fit_Dek, y_fit_Dek, color = 'firebrick' , label = 'Gaussfit')

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
Processe.append(Process(target=plotte_Detektorscan, args=([Detektorscann, x_fit_Dek, y_fit_Dek, Gaus_Peak_idx, Gaus_Peak_FWHM_in_idx])))
Processe.append(Process(target=plotte_Z1Scan, args=([ZScann1, SampleStart, SampleEnd])))
Processe.append(Process(target=plotte_RockingCurve, args=([RockingCurve])))
Processe.append(Process(target=plotte_Omega2Theta, args=([Omega2Theta])))
Processe.append(Process(target=plotte_Diffus, args=([Diffus])))


#p1 = Process(target=plotte_Detektorscan, args=([Detektorscann]))
#p2 = Process(target=plotte_Detektorscan, args=([ZScann1]))

for p in Processe:
    p.start()

for p in Processe:
    p.join()