import matplotlib.pyplot as pyplot
import numpy as np
from uncertainties import ufloat
import scipy.constants as const
import sympy
from scipy.stats import linregress
import uncertainties.unumpy as unp
from pandas import read_csv
import scipy
import math

from scipy.optimize import curve_fit
from numpy import arange
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
Werte[:,3] = Werte[:,3]+ 1
Werte[:,4] = Werte[:,4]+ 1
print('\n')
print(Werte)

#für plot 	1	2	3	4	5
#			a	a	a	a	a
#			b	b	b	b	b
#			c	c	c	c	c
FehlerderKurven = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
prams = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]



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
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a,b,c))
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
print(x_line)
# calculate the output for the range
y_line = objective(x_line, a, b,c)
print(y_line)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
#pyplot.ylabel(r'$T_1 \mathbin{/}\unit{\celsius}$')
#pyplot.show()
pyplot.savefig('build/plot_1.pdf')
for i in range (0,3):
	FehlerderKurven[i][0]= _[i][i]

prams[0][0]= a
prams[1][0]= b
prams[2][0]= c


print(_, " _<- da steht ein Unterstrich")
print(FehlerderKurven, " Fehler-Matrix")

pyplot.clf()
#der zweite Plot
y = Werte[:,2]
popt, _ = curve_fit(objective, x, y)
print(_, "unterstrich")
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
pyplot.scatter(x, y)
x_line = arange(min(x), max(x), 1)
print(x_line)
y_line = objective(x_line, a, b,c)
print(y_line)
pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
#pyplot.ylabel(r'$T_1 \mathbin{/}\unit{\celsius}$')
#pyplot.show()
pyplot.savefig('build/plot_2.pdf')
for i in range (0,3):
	FehlerderKurven[i][1]= _[i][i]

prams[0][1]= a
prams[1][1]= b
prams[2][1]= c


pyplot.clf()
#der dritte Plot
y = Werte[:,3]

popt, _ = curve_fit(objective, x, y)
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
pyplot.scatter(x, y)
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b,c)
pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
#pyplot.ylabel(r'$P_b \mathbin{/}\unit{\bar}$')
#pyplot.show()
pyplot.savefig('build/plot_3.pdf')
for i in range (0,3):
	FehlerderKurven[i][2]= _[i][i]

prams[0][2]= a
prams[1][2]= b
prams[2][2]= c
pyplot.clf()
#der vierte Plot

y = Werte[:,4]

popt, _ = curve_fit(objective, x, y)
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
pyplot.scatter(x, y)
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b,c)
pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
#pyplot.ylabel(r'$P_a \mathbin{/}\unit{\bar}$')
#pyplot.show()
pyplot.savefig('build/plot_4.pdf')
for i in range (0,3):
	FehlerderKurven[i][3]= _[i][i]


prams[0][3]= a
prams[1][3]= b
prams[2][3]= c	
pyplot.clf()
#der fünfte Plot

y = Werte[:,5]

popt, _ = curve_fit(objective, x, y)
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
pyplot.scatter(x, y)
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b,c)
pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
#pyplot.ylabel(r'$W \mathbin{/}\unit{\watt}$')
#pyplot.show()
pyplot.savefig('build/plot_5.pdf')
for i in range (0,3):
	FehlerderKurven[i][4]= _[i][i]


prams[0][4]= a
prams[1][4]= b
prams[2][4]= c

print(prams, " <- Prarmeter")
#print(FehlerderKurven, " Fehlerkurven")
for j in range(0,3):
	for i in range(0,5):
		FehlerderKurven[j][i] = np.sqrt(FehlerderKurven[j][i])

print(FehlerderKurven, " das sind auch Fehler")

x = sympy.var('x')
# Aufgabe b
T1 = prams[0][0]* (x**2) +prams[1][0]*x + prams[2][0]
T1_ = T1.diff(x)

T2 = prams[0][1]* (x**2) +prams[1][1]*x + prams[2][1]
T2_ = T2.diff(x)

w = prams[0][4]* (x**2) +prams[1][4]*x + prams[2][4]
W =	w.integrate(x)
print(W)
 #integrate(expr,(x,0,oo) )
#Aufgabe c
Auswertung=[[1,2,3,4],[1,2,3,4]]
for i in range(0,4):
		Auswertung[0][i]=	T1.evalf(subs={x:((i*10)+5)})
		Auswertung[1][i]=	T2.evalf(subs={x:((i*10)+5)})