#das ist die Zusatzaufgabe in Mathe \ die Plots sehen eigenlich ganz gut aus \ die Suche nach den Nullstellen ist rudimentär aber funktioniert ganz okay
import matplotlib.pyplot as pyplot
import numpy as np
import scipy
import scipy.special
gen = 10000

def func(x):
    return  2 / x

xdata = np.linspace(1, 20, gen, endpoint=False)# bei großen z asymtotisch gegen null

ydata =  scipy.special.jv(1,xdata)

pyplot.plot(xdata, ydata,color ='green', label="1.Ordnung")


for i in range(0,gen-1):
    if abs(scipy.special.jv(1,xdata[i])) < abs(scipy.special.jv(1,xdata[i+1])):
        if abs(scipy.special.jv(1,xdata[i])) < abs(scipy.special.jv(1,xdata[i-1])):
            print(" ",xdata[i], " i")
            print('\n')
            i= i+2

#y2data = scipy.special.jv(2,xdata)
#pyplot.plot(xdata, y2data,color ='black', label="2.Ordnung K")


y3data = (func(xdata))*scipy.special.jv(1,xdata)-scipy.special.jv(0,xdata)
pyplot.plot(xdata, y3data,color ='red',  linestyle='dashed' , label= '2.Ordnung R')


pyplot.legend()
pyplot.grid()
pyplot.xlabel("x-Achse")
pyplot.ylabel("y-Achse")
pyplot.savefig('Bessel.pdf')
pyplot.clf()
