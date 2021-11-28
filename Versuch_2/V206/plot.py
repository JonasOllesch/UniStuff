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
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$T_1 \mathbin{/}\unit{\celsius}$')
#pyplot.show()
pyplot.savefig('build/plot_1.pdf')




pyplot.clf()
#der zweite Plot
y = Werte[:,2]
popt, _ = curve_fit(objective, x, y)
#print(_, "unterstrich")
a, b, c = popt
print('y = %.5f a * (x ** 2) + %.5f *b + %.5f c' % (a, b,c))
pyplot.scatter(x, y)
x_line = arange(min(x), max(x), 1)
print(x_line)
y_line = objective(x_line, a, b,c)
print(y_line)
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$T_2 \mathbin{/}\unit{\celsius}$')
#pyplot.show()
pyplot.savefig('build/plot_2.pdf')


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
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$p_b \mathbin{/}\unit{\bar}$')
#pyplot.show()
pyplot.savefig('build/plot_3.pdf')

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
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$p_a \mathbin{/}\unit{\bar}$')
#pyplot.show()
pyplot.savefig('build/plot_4.pdf')
	
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
pyplot.xlabel(r'$t \mathbin{/} \unit{\minute}$')
pyplot.ylabel(r'$W \mathbin{/}\unit{\watt}$')
#pyplot.show()
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

print(Fehler, " das sind Fehler")
print(prams, " Parameter")

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
#	print(ufloat(prams[0][1],Fehler[0][1])*Ap[i]**2+ ufloat(prams[1][1],Fehler[1][1])*Ap[i] +ufloat(prams[2][1],Fehler[2][1]))


#print(Diffentialquotienten, " Diffentialquotienten")


#Aufgabe d
Gueteziffer = [1,2,3,4]
for i in range(0,4):
	Gueteziffer[i]=	(3*4180+750)*Diffentialquotienten[0][i]/(ufloat(prams[0][4],Fehler[0][4])*Ap[i]**2+ ufloat(prams[1][4],Fehler[1][4])*Ap[i] +ufloat(prams[2][4],Fehler[2][4]))

#Tmp = [1,2,3,4]
#for i in range(0,4):
#	Tmp[i]= (ufloat(prams[0][4],Fehler[0][4])*Ap[i]**2+ ufloat(prams[1][4],Fehler[1][4])*Ap[i] +ufloat(prams[2][4],Fehler[2][4]))
#print(Tmp, " tmp")
print(Gueteziffer, "Gueteziffer")


#Aufgabe e

x = Werte[:,1]
y = Werte[:,3]
def objective(x, l):
	return 5.51*np.exp(-l/(8.314*x))
popt, _ = curve_fit(objective, x, y)
l = popt
print(l, _, " Die Konstante L")
L = ufloat(l,_)

popt, _ = curve_fit(lambda t, l: 5.51 * np.exp(-l/(8.314 * t)), x, y) #da kommit zum Glück genau das gleich raus
print(popt, _)

Mdurchsatz=[1,2,3,4]

for i in range(0,4):
	Mdurchsatz[i]=((3*4180+750)*Diffentialquotienten[1][i])/L



print(Mdurchsatz, "<- Massendruchsatz")


#Aufgabe f
#pb und pa an Ap[i] auswerten
pb=[1,2,3,4]
pa=[1,2,3,4]
T2=[1,2,3,4]
for i in range(0,4):
	pb[i]= (ufloat(prams[0][2],Fehler[0][2])*Ap[i]**2+ ufloat(prams[1][2],Fehler[1][2])*Ap[i] +ufloat(prams[2][2],Fehler[2][2]))
	pa[i]= (ufloat(prams[0][3],Fehler[0][3])*Ap[i]**2+ ufloat(prams[1][3],Fehler[1][3])*Ap[i] +ufloat(prams[2][3],Fehler[2][3]))
	T2[i]= (ufloat(prams[0][1],Fehler[0][1])*Ap[i]**2+ ufloat(prams[1][1],Fehler[1][1])*Ap[i] +ufloat(prams[2][1],Fehler[2][1]))



print(pb)
print(pa)
print(T2)
MArbeit=[1,2,3,4]
for i in range(0,4):
	MArbeit[i]= (((1/0.14)*(pb[i]*(pa[i]/pb[i]**(1/1.14)))-pa[i])*(T2[i]*1)*Mdurchsatz[i])/(5.51*273.15*pa[i])
print(MArbeit)

#ist ziemlich wenig aber egalxD
"""
#---------------------------------

#	f von	T1  T2  pb 	pa  W
#für plot 	1	2	3	4	5
#			a	a	a	a	a
#			b	b	b	b	b
#			c	c	c	c	c
FehlerderKurven = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
prams = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]


Werte[:,0] = Werte[:,0]*60

#Vielleicht Druck von Pasal in bar umrechnen aber dann kommen doofe Werte raus
#Werte[:,2] = Werte[:,0]*(10**5)
#Werte[:,3] = Werte[:,0]*(10**5)


x = Werte[:,0]

for i in range (1,6):
	y = Werte[:,i]
	popt, _ = curve_fit(objective, x, y)
	a, b, c = popt
	print('y = %.5f a * (x ** 2) + %.5f + b * x + %.5f c' % (a, b,c))
	prams[0][i-1] = a
	prams[1][i-1] = b
	prams[2][i-1] = c
	
	FehlerderKurven[0][i-1] =np.sqrt(_[0][0])
	FehlerderKurven[1][i-1] =np.sqrt(_[1][1])
	FehlerderKurven[2][i-1] =np.sqrt(_[2][2])
	
print(FehlerderKurven, " das sind auch Fehler")

x = sympy.var('x')
# Aufgabe b
T1 = prams[0][0]* (x**2) +prams[1][0]*x + prams[2][0]
T1_ = T1.diff(x)

T2 = prams[0][1]* (x**2) +prams[1][1]*x + prams[2][1]
T2_ = T2.diff(x)

pb = prams[0][2]* (x**2) +prams[1][2]*x + prams[2][2]



w = prams[0][4]* (x**2) +prams[1][4]*x + prams[2][4]


#W =	w.integrate(x)
#print(W)
 #integrate(expr,(x,0,oo) )
#Aufgabe c
Auswertung=[[1,2,3,4],[1,2,3,4]]
Ap=[1,2,3,4]
for i in range(0,4):
		Ap[i] = (i*480)+120
		Auswertung[0][i]=	T1.evalf(subs={x:Ap[i]})
		Auswertung[1][i]=	T2.evalf(subs={x:Ap[i]})

Gueteziffer = [[1,2,3,4],[1,2,3,4]]
#Aufgabe d
#v = (m1*cw+ mc*ck)/N *T1
IdGueteziffer = [[1,2,3,4],[1,2,3,4]]
for i in range (0,4):
	IdGueteziffer[0][i]=((T1.evalf(subs={x:Ap[i]}))/(T1.evalf(subs={x:Ap[i]})-T2.evalf(subs={x:Ap[i]})))

for i in range (0,4):
	Gueteziffer[0][i] = ((3*4180+750)*T1_.evalf(subs={x:Ap[i]}))/w.evalf(subs={x:Ap[i]})
	#Gueteziffer[1][i] = ((3*4180+750)*T2_.evalf(subs={x:Ap[i]}))/w.evalf(subs={x:Ap[i]})

print(IdGueteziffer)
print(Gueteziffer)

for i in range(0,4):
	print(((Gueteziffer[0][i]-IdGueteziffer[0][i])/IdGueteziffer[0][i])*100, " Abweichungen der Güteziffer")
#Aufgabe e

#im Vergleich zu Altprotokoll haben wir ungefähr einen Faktor von 1000 
# Wasser hat einen Wert von 45,054 	bei T = 0°c also ist unser wahrscheinlich besser :)
LWert = [1,2,3,4]
for i in range(0,4):
	LWert[i]= -(math.log(pb.evalf(subs={x:Ap[i]})/5,51))*8.314462*T1.evalf(subs={x:Ap[i]})
Lmittel = np.mean(LWert[:])
print(LWert)
print(Lmittel)
print(T1_)
m_ = ((3*4180+750)/Lmittel)*T1_
print(m_)

Massendurchsatz = [1,2,3,4]
for i in range (0,4):
	Massendurchsatz[i]= m_.evalf(subs={x:Ap[i]})

print(Massendurchsatz)
#Aufgabe f
"""