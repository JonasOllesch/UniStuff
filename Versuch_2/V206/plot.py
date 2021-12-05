#Werte die noch verwendet werden gerundet in eine txt schreiben
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

#print(Werte)
Werte[:,3] = Werte[:,3]+ 1
Werte[:,4] = Werte[:,4]+ 1
#print('\n')
#print(Werte)




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
print('y = %.5f a * (x ** 2) + %.5f *b* x + %.5f c' % (a,b,c))
# plot input vs output
pyplot.scatter(x, y, label=r'$T_1$', s=15)
pyplot.legend()
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x)+1, 1)
print(x_line)
# calculate the output for the range
y_line = objective(x_line, a, b,c)
print(y_line)
# create a line plot for the mapping function

pyplot.plot(x_line, y_line, '--', color='red',label= r"$Reg T_1$ " r'$T_1$')

#der zweite Plot
y = Werte[:,2]
popt, _ = curve_fit(objective, x, y)
#print(_, "unterstrich")
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b* x + %.5f c' % (a, b,c))
pyplot.scatter(x, y, color='green', label=r'$T_2$', s= 15)

x_line = arange(min(x), max(x)+1, 1)
print(x_line)
y_line = objective(x_line, a, b,c)
print(y_line)
pyplot.plot(x_line, y_line, '--', color='gray', label=r"$Reg T_2$ " r'$T_2$' )
pyplot.grid()
pyplot.xticks(np.arange(0,31,step=5))
pyplot.yticks(np.arange(-5,51,step=5))
pyplot.xlim(0,31)
pyplot.ylim(-5,51)
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$T \mathbin{/}\unit{\celsius}$')
pyplot.legend()
pyplot.savefig('build/plot_1.pdf')


pyplot.clf()
#der dritte Plot
y = Werte[:,3]

popt, _ = curve_fit(objective, x, y)
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b* x  + %.5f c' % (a, b,c))
pyplot.scatter(x, y, s= 15, label=r'$p_b$')
pyplot.legend()
x_line = arange(min(x), max(x)+1, 1)
y_line = objective(x_line, a, b,c)
pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.show()


#pyplot.clf()
#der vierte Plot

y = Werte[:,4]

popt, _ = curve_fit(objective, x, y)
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b * x + %.5f c' % (a, b,c))
pyplot.scatter(x, y, color='green',s=15, label=r'$p_a$')
pyplot.legend()
x_line = arange(min(x), max(x)+1, 1)
y_line = objective(x_line, a, b,c)
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.grid()
pyplot.xticks(np.arange(0,31,step=5))
pyplot.yticks(np.arange(0,13,step=1))
pyplot.xlim(0,31)
pyplot.ylim(0,13)
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$p \mathbin{/}\unit{\bar}$')
pyplot.savefig('build/plot_2.pdf')
#pyplot.show()
#pyplot.savefig('build/plot_4.pdf')
	
pyplot.clf()
#der fünfte Plot

y = Werte[:,5]
popt, _ = curve_fit(objective, x, y)
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
pyplot.scatter(x, y, s=15)
x_line = arange(min(x), max(x)+1, 1)
y_line = objective(x_line, a, b,c)
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.grid()
pyplot.xticks(np.arange(0,31, step = 5))
pyplot.yticks(np.arange(0,130,step = 20))
pyplot.xlim(0,31.5)
pyplot.ylim(0,130)
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$N \mathbin{/}\unit{\watt}$')
pyplot.savefig('build/plot_5.pdf')





Werte[:,0] = Werte[:,0]*60

Werte[:,1]= Werte[:,1]+273.15
Werte[:,2]= Werte[:,2]+273.15


Werte[:,3] = Werte[:,3]*(10**5)
Werte[:,4] = Werte[:,4]*(10**5)


print('\n')
print(Werte, " Werte umgerechnet")



Fehler = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
prams = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

x = Werte[:,0]

for i in range (0,5):
	y = Werte[:,i+1]
	popt, _ = curve_fit(objective, x, y)
	a, b, c = popt
	print('y = %.5f a * (x ** 2) + %.5f + b * x + %.5f c' % (a, b,c))
	prams[0][i] = a
	prams[1][i] = b
	prams[2][i] = c
	
	Fehler[0][i] =np.sqrt(_[0][0])
	Fehler[1][i] =np.sqrt(_[1][1])
	Fehler[2][i] =np.sqrt(_[2][2])

#print(Fehler, " das sind Fehler")
#print(prams, " Parameter")

#Aufgabe c
#die vier Zeipunkt wählen  t = i*380
Ap=[1,2,3,4]
for i in range (0,4):
	Ap[i] = (i+1)*380
#die Diffentialquotienten
Diffentialquotienten=[[1,2,3,4],[1,2,3,4]] # warum eigentlich für beide Temperaturen aber nevermind

for j in range (0,2):
	for i in range(0,4):
		Diffentialquotienten[j][i] = ufloat(prams[0][j],Fehler[0][j])*2*Ap[i]+ ufloat(prams[1][j],Fehler[1][j])

#for i in range (0,4):	
#print(ufloat(prams[0][1],Fehler[0][1])*Ap[i]**2+ ufloat(prams[1][1],Fehler[1][1])*Ap[i] +ufloat(prams[2][1],Fehler[2][1]))


#print(Diffentialquotienten, " Diffentialquotienten")


#Aufgabe d
Gueteziffer = [1,2,3,4]
for i in range(0,4):
	Gueteziffer[i]=	(3*4180+750)*Diffentialquotienten[0][i]/(ufloat(prams[0][4],Fehler[0][4])*Ap[i]**2+ ufloat(prams[1][4],Fehler[1][4])*Ap[i] +ufloat(prams[2][4],Fehler[2][4]))

#Tmp = [1,2,3,4]
#for i in range(0,4):
#	Tmp[i]= (ufloat(prams[0][4],Fehler[0][4])*Ap[i]**2+ ufloat(prams[1][4],Fehler[1][4])*Ap[i] +ufloat(prams[2][4],Fehler[2][4]))
#print(Tmp, " tmp")
#print(Gueteziffer, "Gueteziffer")
IdGueteziffer = [1,2,3,4]
T1 = [1,2,3,4]
T2 = [1,2,3,4]

for i in range(0,4):
	
	T1[i]= (ufloat(prams[0][0],Fehler[0][0])*Ap[i]**2+ ufloat(prams[1][0],Fehler[1][0])*Ap[i] +ufloat(prams[2][0],Fehler[2][0]))
	T2[i]= (ufloat(prams[0][1],Fehler[0][1])*Ap[i]**2+ ufloat(prams[1][1],Fehler[1][1])*Ap[i] +ufloat(prams[2][1],Fehler[2][1]))
	IdGueteziffer[i]= T1[i]/(T1[i]-T2[i])

#print(IdGueteziffer, " die ideale Güteziffer")
ra = [1,2,3,4]
for i in range(0,4):
	ra[i] = ((Gueteziffer[i]-IdGueteziffer[i])/IdGueteziffer[i])*100

print(ra)


#Aufgabe e



x = Werte[:,1]
y = Werte[:,3]

def objective(x, l):
	return 5.51*np.exp(-l/((8.31446261815324 * x)))
popt, _ = curve_fit(objective, x, y)
_[0][0] =np.sqrt(_[0][0])
l = popt
#print(l, _, " Die Konstante L")
L = ufloat(l,_)


#popt, _ = curve_fit(lambda t, l: 5.51 * np.exp(-l/(8.31446261815324 * t)), x, y) #da kommit zum Glück genau das gleich raus
#_[0][0] =np.sqrt(_[0][0])


x_line = arange(min(x), max(x)+1, 1)
y_line = objective(x_line,l)
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.scatter(x, y, s=15)
#pyplot.xlabel(r'T_1 \mathbin{/} \unit{\pascal}$')
#pyplot.ylabel(r'$p_b \mathbin{/}\unit{\kelvin}$')
#pyplot.show

#pyplot.savefig('build/Test.pdf')

Mdurchsatz=[1,2,3,4]

for i in range(0,4):
	Mdurchsatz[i]=((3*4180+750)*Diffentialquotienten[1][i])/L

print(" Massendurchsatz in kg pro sek")
print(Mdurchsatz)


#Aufgabe f
#pb und pa an Ap[i] auswerten
pb=[1,2,3,4]
pa=[1,2,3,4]
#T2=[1,2,3,4]
for i in range(0,4):
	pb[i]= (ufloat(prams[0][2],Fehler[0][2])*Ap[i]**2+ ufloat(prams[1][2],Fehler[1][2])*Ap[i] +ufloat(prams[2][2],Fehler[2][2]))
	pa[i]= (ufloat(prams[0][3],Fehler[0][3])*Ap[i]**2+ ufloat(prams[1][3],Fehler[1][3])*Ap[i] +ufloat(prams[2][3],Fehler[2][3]))
#	T2[i]= (ufloat(prams[0][1],Fehler[0][1])*Ap[i]**2+ ufloat(prams[1][1],Fehler[1][1])*Ap[i] +ufloat(prams[2][1],Fehler[2][1]))



#print(pb)
#print(pa)
#print(T2)
MArbeit=[1,2,3,4]
for i in range(0,4):
	MArbeit[i]= (((1/0.14)*(pb[i]*((pa[i]/pb[i])**(1/1.14))-pa[i]))*(T2[i]*10000)*Mdurchsatz[i])/(5.51*273.15*pa[i])
	print((T2[i]*10000)/(5.51*273.15*pa[i]), " das rho fur " , i)

print(MArbeit, " die mechanische Leistung")

#ist ziemlich wenig aber egalxD
