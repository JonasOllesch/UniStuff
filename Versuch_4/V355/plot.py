import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from pandas import read_csv
import math
from scipy.optimize import curve_fit
from numpy import arange


#to do list
#Auswertung:
#   a "ns" berechnen und mit den Theoriewerten berechnen
#   b ν- und ν+ mit den Theoriewerten vergleichen
#   c aus t_1 und t_2 mit dem theoretischen ν- und ν+ vergleichen und vielleicht mit denen aus b wenn man noch Lust hat

# Werte einlesen und in Si Einheiten konvertieren
Messung_a = np.array(np.genfromtxt('a.txt'))
Messung_a[:,0]= Messung_a[:,0]*(10**-9)

Messung_b = np.array(np.genfromtxt('b.txt'))
Messung_b[:,0]= Messung_b[:,0]*(10**-9)
Messung_b[:,1]= Messung_b[:,1]*(10**3)
Messung_b[:,2]= Messung_b[:,2]*(10**3)

Messung_c = np.array(np.genfromtxt('c.txt'))
Messung_c[:,0]= Messung_c[:,0]*(10**-9)
Messung_c[:,1]= Messung_c[:,1]*(5*10**(-6))
Messung_c[:,2]= Messung_c[:,2]*(5*10**(-6))
#theoretische Werte für ν- und ν+ berechnen

C_k = [12,9.99,8.18,6.86,4.74,2.86,2.19,0.997]
for i in range(0,len(C_k)):
    C_k[i] = C_k[i]*(10**-9)

L = ufloat(23.95*(10**-3),0.05*(10**-3))
C_sp = ufloat(0.028*(10**-9),0.028*(10**-9)*(1/20)) #die Fehler habe ich jetzt einfach auf 5% geschätzt wenn du andere haben möchtest einfach ändern :)
C = ufloat (0.7935*(10**-9),0.7935*(10**-9)*(1/20))

#ν-  mit Formel 4.14 aus dem Altprotokoll
dieKlammer = [0] * 8
v_mt = [0]*8
for i in range(0,len(v_mt)):
    dieKlammer[i] = (1/C + 2/C_k[i])**-1 +C_sp
    v_mt[i] = (2*np.pi*(unp.sqrt(L*dieKlammer[i])))**-1
#for i in range(0,len(v_mt)):
#    print(v_mt[i]/1000)

#mit Formel 4.13 aus dem Altprotokoll
dieWurzel = [0]*8
v_pt = [0]*8
for i in range(0,len(v_pt)):
    dieWurzel[i] = unp.sqrt(L*(C+C_sp))
    v_pt[i]      = (2*np.pi*dieWurzel[i])**-1
del dieWurzel
#for i in range(0,len(v_pt)):
#    print(v_pt[i]/1000)

# das theoretische Verhältnis zwischen Schwingung und Schwebugn
n_t =[0]*8
for i in range(0,len(n_t)):
    n_t[i] = (v_pt[i]+v_mt[i])/(2*(v_mt[i]-v_pt[i]))
print(n_t, " n_t")
Arel = [0]*8
for i in range(0,len(Arel)):
    Arel[i] = unp.nominal_values(abs((Messung_a[i][1]-n_t[i])/n_t[i]))
#print(Arel)

#die Aufgabe b
Brel_vp= [0]*8
Brel_vm= [0]*8

for i in range(0,len(Brel_vp)):
    Brel_vp[i] = unp.nominal_values(abs((Messung_b[i][1]-v_pt[i])/v_pt[i]))*100
    Brel_vm[i] = unp.nominal_values(abs((Messung_b[i][2]-v_mt[i])/v_mt[i]))*100

print(Brel_vm)  #irgendwie steht hier noch ein "array im array lol"
print('\n')
print(Brel_vp)  #irgendwie steht hier noch ein "array im array lol"
#die Aufgabe C
Messung_t = [[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]] # die Messung der Zeit
for i in range(0,8):
    Messung_t[i][0] = Messung_c[i][1]
    Messung_t[i][1] = Messung_c[i][2]

    Messung_t[i][0] = 1/Messung_t[i][0]
    Messung_t[i][1] = 1/Messung_t[i][1]

for i in range(0,8):
    print(Messung_t[i][0])

print("pause")

for i in range(0,8):
    print(Messung_t[i][1])


output = ("Werte")    
my_file = open(output + '.txt', "w") 
#for i in range(0, 8):
#    my_file.write(str("{"))
#    my_file.write(" ")
#    my_file.write(str(Messung_a[i][0]*10**9))
#    my_file.write(" ")
#    my_file.write(str("&"))
#    my_file.write(" ")
#    my_file.write(str(v_mt[i]*10**-3))
#    my_file.write(str("&"))
#    my_file.write(" ")
#    my_file.write(str(Messung_b[i][2]*10**-3))
#    my_file.write(str("&"))
#    my_file.write(" ")
#
#
#
#    my_file.write("\n")


for i in range (0,8):
    my_file.write(str(v_pt[i]*10**-3))
    my_file.write("\n")
for i in range (0,8):
    my_file.write(str(Messung_b[i][1]*10**-3))
    my_file.write("\n")
for i in range(0,8):
    my_file.write(str(Brel_vm[i]))
    my_file.write(str("\n"))



#for i in range(0,8):
#    print(np.round(Messung_b[i][0]*(10**-3),2))
#    print("\n")

