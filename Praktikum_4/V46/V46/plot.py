import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

#from multiprocessing  import Process


def berechne_freien_Winkel(dotiert, undotiert):
    return np.abs(dotiert-undotiert)


def linReg1(Wellenlänge, a, b):
    n = 3.354
    N = 2.8*10**24
    B = 405/1000 *1.02
    return a*Wellenlänge**2*N*B/n+b*N*B/n

def linReg2(Wellenlänge, a, b):
    n = 3.354
    N = 1.2*10**24
    B = 405/1000 *1.02
    return a*Wellenlänge**2*N*B/n+b*N*B/n

def berechneEffektiveMasse(a):
    a = ufloat(np.abs(unp.nominal_values(a)), unp.std_devs(a))
    u = 8 * np.pi * constants.epsilon_0*constants.c**3*a
    o = constants.e**3

    return unp.sqrt(o/u)


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
N = N*10**6

diffWinkel_nom = np.zeros((len(Wellenlänge),3))


diffWinkel_nom[:,0] = 1/(2*d[0]) *(Winkel1[:,1] -Winkel1[:,0])
diffWinkel_nom[:,1] = 1/(2*d[1]) *(Winkel2[:,1] -Winkel2[:,0])
diffWinkel_nom[:,2] = 1/(2*d[2]) *(Winkel3[:,1] -Winkel3[:,0])

popt1, pcov1 = curve_fit(linReg1, Wellenlänge, diffWinkel_nom[:,0], absolute_sigma=True)
print(popt1)
print(pcov1)
para1 = correlated_values(popt1, pcov1)


popt2, pcov2 = curve_fit(linReg2, Wellenlänge, diffWinkel_nom[:,1], absolute_sigma=True)
para2 = correlated_values(popt2, pcov2)


print(f'Parameter aus dem 1. Fit {para1}')
print(f'Parameter aus dem 2. Fit {para2}')


m_eff1 = berechneEffektiveMasse(para1[0])
m_eff2 = berechneEffektiveMasse(para2[0])
print(f' 1. Die effektiven Massen der Elektronen {repr(berechneEffektiveMasse(para1[0]))}')
print(f' 2. Die effektiven Massen der Elektronen {repr(berechneEffektiveMasse(para2[0]))}')

print(repr(m_eff1))
print(repr(m_eff2))
me = constants.m_e
print(f'1.m_eff/me {repr(m_eff1/me)}')
print(f'2.m_eff/me {repr(m_eff2/me)}')

print(f'relative Abweichungen in Prozent {repr((m_eff1/me-0.063)/0.063*100)}')
print(f'relative Abweichungen in Prozent {repr((m_eff2/me-0.063)/0.063*100)}')

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

#plt.scatter(Wellenlänge**2, diffWinkel_nom[:,0], label =" 1. Dotieres", c = 'navy', marker='x' s = 20)
plt.scatter((Wellenlänge**2)*10**12, diffWinkel_nom[:,0], label ="1. Probe", c = 'darkorange', marker='x' ,s = 20)
plt.scatter((Wellenlänge**2)*10**12, diffWinkel_nom[:,1], label ="2. Probe", c = 'firebrick', marker='x' ,s = 20)
plt.scatter((Wellenlänge**2)*10**12, diffWinkel_nom[:,2], label ="3. Probe", c = 'navy', marker='x'  ,s = 20)

#print(Wellenlänge)
x1 = np.linspace(1*10**(-6),2.65*10**(-6),2)
y1 = linReg1(x1, a=unp.nominal_values(para1[0]), b=unp.nominal_values(para1[1]))

x2 = np.linspace(1*10**(-6),2.65*10**(-6),2)
y2 = linReg2(x2, a=unp.nominal_values(para2[0]), b=unp.nominal_values(para2[1]))

plt.plot(x1**2*10**12,y1, c = 'darkorange')
plt.plot(x2**2*10**12,y2, c = 'firebrick')


plt.xlabel(r"$ \lambda \mathbin{/} \unit{\micro\meter²} $")
plt.ylabel(r"$\theta_{\text{nom}} \mathbin{/} \dfrac{1}{\unit{\meter}}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig('build/winkelnom.pdf')
plt.clf()
