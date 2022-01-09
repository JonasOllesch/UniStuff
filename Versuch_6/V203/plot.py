import matplotlib.pyplot as pyplot
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
from numpy import arange
import scipy.constants

def writeW(Werte,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    my_file.write(str(Werte))
    my_file.write('\n')
    return 0

def func_pol1(x, a, b):
    return a * x +b

def func_pol3(x, a, b , c ,d):
    return a*x**3   +  b*x**2    +   c*x    +   d

def func_pol3d(x, a, b, c):
    return 3*a*x**2 + 2*b*x     +   c

Messung_a = np.array(np.genfromtxt('Messung_a.txt'))
Messung_a[:,0] = Messung_a[:,0]+273.15
Messung_a[:,1] = Messung_a[:,1]*(10**5)
Messung_a[:,1] = Messung_a[:,1]*(10**-3)

Messung_b = np.array(np.genfromtxt('Messung_b.txt'))
Messung_b[:,0] = Messung_b[:,0]*(10**5)
Messung_b[:,1] = Messung_b[:,1]+273.15
p0 = 1016*100

xdata =  [0]*83
for i in range(0,83):
    xdata[i] = Messung_a[i][0]**-1
ydata = [0]*83
for i in range(0,83):
    ydata[i]= np.log(Messung_a[i][1]/p0)

pyplot.scatter(xdata, ydata,color ='blue', label="Messdaten",s=10,marker='x')


popt, pcov = curve_fit(func_pol1, xdata, ydata)
x = np.linspace(0.00265,0.0035)
y = func_pol1(x, popt[0], popt[1])
pyplot.plot(x, y,label='lineare Regression')


pyplot.legend()
pyplot.grid()
pyplot.ylabel(r'$\log{\frac{p}{p_0}}$')
pyplot.xlabel(r'$\dfrac{1}{T}  in \unit{\kelvin}⁻1$')
pyplot.savefig('build/Messung_a')
pyplot.clf()

a =ufloat(popt[0],np.sqrt(pcov[0][0]))      #im Vergleich zum Altprotokoll haben wir ca einen Faktor von 2
b =ufloat(popt[1],np.sqrt(pcov[1][1]))      #im Vergleich zum Altprotokoll haben wir ca einen Faktor von 2
L = -a*scipy.constants.R                    

La = scipy.constants.R *373
Li = L -La
Li_in_eV = Li/ (scipy.constants.N_A) / scipy.constants.elementary_charge




x2data=  [0]*15
for i in range(0,15):
    x2data[i] = Messung_b[i][1]
y2data = [0]*15
for i in range(0,15):
    y2data[i]= Messung_b[i][0]

pyplot.scatter(x2data, y2data,color ='green', label="Messdaten",s=10,marker='x')
popt, pcov = curve_fit(func_pol3, x2data, y2data)
T = np.linspace(394.15,465.15)
p = func_pol3(T, popt[0], popt[1], popt[2], popt[3])

a2 =ufloat(popt[0],np.sqrt(pcov[0][0]))         #die Parameter haben teilweise große Unterschiede zu denen aus dem Altprotokoll, aber der Fit sieht ziemlich gut aus
b2 =ufloat(popt[1],np.sqrt(pcov[1][1]))
c2 =ufloat(popt[2],np.sqrt(pcov[2][2]))     
d2 =ufloat(popt[3],np.sqrt(pcov[3][3]))
pyplot.plot(T, p,label='pol 3')


pyplot.legend()
pyplot.grid()
#pyplot.xlabel(r'$a² \mathbin{/}\unit{\meter^2}$')
pyplot.xlabel(r'$T \mathbin{/}\unit{\kelvin}$')
pyplot.ylabel(r'$p \mathbin{/}\unit{\pascal}$')

pyplot.savefig('build/Messung_b')
pyplot.clf()

dpdt = func_pol3d(T, a2.nominal_value, b2.nominal_value, c2.nominal_value)

wurzel = np.sqrt((scipy.constants.R*T/2)**2 - 0.9*p)
tmp = (scipy.constants.R*T)/2 
tmp2 =  T/p
L2p = tmp2*(tmp + wurzel)*dpdt
L2m = tmp2*(tmp - wurzel)*dpdt



pyplot.scatter(T, L2p, color ='red', label="Messdaten p",s=10,marker='x')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$T \mathbin{/}\unit{\kelvin}$')
pyplot.ylabel(r'$L \mathbin{/}  \dfrac{\unit{\mol}}{\unit{\joule}} $')
pyplot.tight_layout()
pyplot.savefig('build/L_positiv')
pyplot.clf()


pyplot.scatter(T, L2m,color ='red', label="Messdaten m",s=10,marker='x')
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$T \mathbin{/}\unit{\kelvin}$')
pyplot.ylabel(r'$L \mathbin{/}  \dfrac{\unit{\mol}}{\unit{\joule}} $')
pyplot.tight_layout()
pyplot.savefig('build/L_negativ')
pyplot.clf()
#------------------------------------------------------------------------------------
output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")
writeW(a, "Parameter a")
writeW(b, "Parameter b")
writeW(L, "Verdampfungswärme")
writeW(La, "La")
writeW(Li, "Li")
writeW(Li_in_eV, "Li in eV")

writeW(a2, "a2")            
writeW(b2, "b2")
writeW(c2, "c2")
writeW(d2, "d2")