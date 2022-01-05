import matplotlib.pyplot as pyplot
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
from numpy import arange

def writeW(Werte,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    my_file.write(str(Werte))
    my_file.write('\n')
    return 0


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

def func(x, a, b):
    return a * x +b

popt, pcov = curve_fit(func, xdata, ydata)
x = np.linspace(0.00265,0.0035)
y = func(x, popt[0], popt[1])
pyplot.plot(x, y,label='lineare Regression')


pyplot.legend()
pyplot.grid()
pyplot.ylabel(r'$\log{\frac{p}{p_0}}$')
pyplot.xlabel(r'$\dfrac{1}{T}  in \unit{\kelvin}⁻1$')
pyplot.savefig('build/Messung_a')
pyplot.clf()

a =ufloat(popt[0],np.sqrt(pcov[0][0]))      #im Vergleich zum Altprotokoll haben wir ca einen Faktor von 2
b =ufloat(popt[1],np.sqrt(pcov[1][1]))      #im Vergleich zum Altprotokoll haben wir ca einen Faktor von 2
L = -a*8.31446261815324                     #im Vergleich zum Altprotokoll haben wir ca einen Faktor von 2
#------------------------------------------------------------------------------------
output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")
writeW(a, "Parameter a")
writeW(b, "Parameter b")
writeW(L, "Verdampfungswärme")

 
