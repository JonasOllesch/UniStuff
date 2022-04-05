import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
#To do list Brewster Winkel, Brechungsindex, Vergleich mit der Theorie, Theoriekurven,

#output = ("Messung_1")    
#my_file = open(output + '.txt', "w")
#
#for i in range(0,21):
#    my_file.write(str(i*5))
#    my_file.write('\n')

Messung_1 = np.array(np.genfromtxt('Messung_1.txt'))
Messung_2 = np.array(np.genfromtxt('Messung_2.txt'))
print(Messung_1)
Messung_1[:,0] = Messung_1[:,0] * (np.pi/180)
Messung_1[:,1] = Messung_1[:,1]* 1e-6

Messung_2[:,0] = Messung_2[:,0] * (np.pi/180)
Messung_2[:,1] = Messung_2[:,1] *1e-6

pyplot.scatter(Messung_1[:,0]*(180/np.pi),Messung_1[:,1],s=8, c='red',marker='x',label="senkrechte Polarisation")
pyplot.scatter(Messung_2[:,0]*(180/np.pi),Messung_2[:,1],s=8, c='blue',marker='+',label="parallele Polarisation")

pyplot.ylabel(r'$I \mathbin{/} \unit{\ampere} $')
pyplot.xlabel(r'$ \phi \mathbin{/} \unit{°} $')
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph.pdf')
pyplot.clf()
#Nullmessung phi = 0 ,offensichtlich A = 195 my bei parallel und senkrechtem Licht
I_0 = 195 * 1e-6
pyplot.scatter(Messung_1[:,0]*(180/np.pi),Messung_1[:,1]/I_0,s=8, c='red',marker='x',label="senkrechte Polarisation 1")
pyplot.scatter(Messung_2[:,0]*(180/np.pi),Messung_2[:,1]/I_0,s=8, c='blue',marker='+',label="parallele Polarisation")

#pyplot.ylabel(r'$I \mathbin{/} \unit{\ampere} $')
pyplot.xlabel(r'$ \phi \mathbin{/} \unit{°} $')
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph_a.pdf')
pyplot.clf()
