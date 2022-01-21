import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")
def writeW(Wert,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    try:
        for i in Wert:
            my_file.write(str(i))
            my_file.write('\n')
    except:
        my_file.write(str(Wert))
        my_file.write('\n')

    return 0

def func_pol1(x, a, b):
    return a * x +b



Messung_a = np.array(np.genfromtxt('Messung_a.txt'))
Messung_a[:,0]  = Messung_a[:,0]/1000
Messung_a[:,1]  = Messung_a[:,1]/1000
Messung_b = np.array(np.genfromtxt('Messung_b.txt'))

xdata = Messung_a[:,0]
ydata = Messung_a[:,1]
ydata = np.log(Messung_a[:,1]/Messung_a[0][1])
pyplot.scatter(xdata, ydata,s=8, c='red',marker='x',label="Messwerte")

popt, pcov = curve_fit(func_pol1, xdata, ydata)

xausgleich = np.linspace(-0.001,0.01)
yausgleich = func_pol1(xausgleich, popt[0], popt[1])
pyplot.plot(xausgleich, yausgleich,label='lineare Regression')
a1 = ufloat(popt[0],np.sqrt(pcov[0][0]))
b1 = ufloat(popt[1],np.sqrt(pcov[1][1]))
print(a1)
print(b1)
t = 1/a1
writeW(a1, "Erste Messung a1")
writeW(b1, "Erste Messung b1")
writeW(t, "RC")
pyplot.ylabel(r'$\ln{(\frac{U_c}{U_0})}$')
pyplot.xlabel(r'${T}  \mathbin{/} \unit{\second}$')

pyplot.xlim(-0.001,0.01)
pyplot.ylim(-5,0.8)
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph_a.pdf')
pyplot.clf()

#-------------------------------------------------
x2data = Messung_b[:,0]
y2data = Messung_b[:,1]


pyplot.scatter(x2data, y2data,s=8, c='red',marker='x',label="Messwerte")
#pyplot.xlim(-0.001,0.01)
#pyplot.ylim(-5,0.8)
pyplot.xscale('log')
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph_b.pdf')
pyplot.clf()
