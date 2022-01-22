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

def func_2(f,a):
    return 1/np.sqrt(1+(f*a)**2)


Messung_a = np.array(np.genfromtxt('Messung_a.txt'))
Messung_a[:,0]  = Messung_a[:,0]/1000
Messung_a[:,1]  = Messung_a[:,1]/1000
Messung_b = np.array(np.genfromtxt('Messung_b.txt'))

xdata = Messung_a[:,0]
ydata = Messung_a[:,1]
ydata = np.log(Messung_a[:,1]/Messung_a[0][1])
pyplot.scatter(xdata, ydata,s=8, c='red',marker='x',label="Messwerte")

popt_a, pcov_a = curve_fit(func_pol1, xdata, ydata)

x_ausgleich_a = np.linspace(-0.001,0.01)
y_ausgleich_a = func_pol1(x_ausgleich_a, popt_a[0], popt_a[1])
pyplot.plot(x_ausgleich_a, y_ausgleich_a,label='lineare Regression')
a1 = ufloat(popt_a[0],np.sqrt(pcov_a[0][0]))
b1 = ufloat(popt_a[1],np.sqrt(pcov_a[1][1]))
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
popt_b, pcov_b = curve_fit(func_2, x2data, y2data)
x_ausgleich_b = np.logspace(1,5)
y_ausgleich_b = func_2(x_ausgleich_b, popt_b[0])
pyplot.scatter(x2data, y2data,s=8, c='red',marker='x',label="Messwerte")
pyplot.plot(x_ausgleich_b, y_ausgleich_b,label='lineare Regression')
#pyplot.xlim(-0.001,0.01)
#pyplot.ylim(-5,0.8)
pyplot.xscale('log')
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph_b.pdf')
pyplot.clf()

