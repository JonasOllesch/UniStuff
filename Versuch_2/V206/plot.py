import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import scipy.constants as const
import sympy
from scipy.stats import linregress
import uncertainties.unumpy as unp
from pandas import read_csv
from scipy.optimize import curve_fit

# import pandas as pd 
#daten = pd.read_csv("daten", sep=',', index_col=0)
#print(daten.to_latex())
#output = ("Werte")    
#my_file = open(output + '.txt', "w") 
#for i in range(0, 32):
#    my_file.write(str(i))
#    my_file.write("\n")


Werte = np.array(np.genfromtxt('Werte.txt'))
#Zeit T1    T2     pb      pa      W
print(Werte)




# fit a straight line to the economic data
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot

# define the true objective function
def objective(x, a, b, c):
	return a* (x**2) + b*x + c
x = Werte[:,0]
y = Werte[:,1]
# load the dataset
# choose the input and output variables
# curve fit
popt, _ = curve_fit(objective, x, y)
print(_, "unterstrich")
# summarize the parameter values
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 0.5)
print(x_line)
# calculate the output for the range
y_line = objective(x_line, a, b,c)
print(y_line)
# create a line plot for the mapping function
plt.xlabel(r'$t \mathbin{/} \unit{\minute}$')
plt.ylabel(r'$T_1 \mathbin{/}\unit{\celsius}$')
plt.show()
plt.savefig('build/plot.pdf')

#plt.xlim(1,10.05)
#plt.ylim(1.6,1.8)
#plt.xticks(np.arange(1,10.05, step=1))
#plt.yticks(np.arange(1.6,1.81, step=0.05))
#plt.xlabel('Anzahl der Messungen')
#plt.ylabel(r'$T_i$/\,\si{s}')
#plt.legend(loc='best')

#print(Werte[0,:])  von links nach rechts
#print(Werte[1,:]) von links nach rechts

#print(Werte[:,0]) von oben nach unten
#print(Werte[:,1]) von oben nach unten

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
