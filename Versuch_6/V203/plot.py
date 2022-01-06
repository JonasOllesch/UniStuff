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


Messung_a = np.array(np.genfromtxt('Messung_a.txt'))
Messung_a[:,0] = Messung_a[:,0]+273.15
Messung_a[:,1] = Messung_a[:,1]*(10**5)
Messung_a[:,1] = Messung_a[:,1]*(10**-3)

Messung_b = np.array(np.genfromtxt('Messung_b.txt'))
Messung_b[:,0] = Messung_b[:,0]*(10**5)
Messung_b[:,1] = Messung_b[:,1]+273.15
p0 = 1016*100

xdata=  [0]*83
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

pyplot.scatter(x2data, y2data,color ='blue', label="Messdaten",s=10,marker='x')
popt, pcov = curve_fit(func_pol3, x2data, y2data)
x = np.linspace(394.15,465.15)
y = func_pol3(x, popt[0], popt[1], popt[2], popt[3])

a2 =ufloat(popt[0],np.sqrt(pcov[0][0]))
b2 =ufloat(popt[1],np.sqrt(pcov[1][1]))
c2 =ufloat(popt[2],np.sqrt(pcov[2][2]))     
d2 =ufloat(popt[3],np.sqrt(pcov[3][3]))
pyplot.plot(x, y,label='lineare Regression')

#------------------------------------------------------------------------------------
output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")
writeW(a, "Parameter a")
writeW(b, "Parameter b")
writeW(L, "Verdampfungswärme")
writeW(La, "La")
writeW(Li, "Li")
writeW(Li_in_eV, "Li in eV")