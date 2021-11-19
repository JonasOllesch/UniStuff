import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

# do list
# w+
# w-
# kopplungskonstante k                                      (Haken)
# Schwebungsdauer                                            (Haken)
#Schwebungsdauer mit den gemessenen verleichen
#Fehlerfortpflanzung für irgendwie alles xD

#erstesfreiesPendel = np.genfromtxt('l070T15f.txt', unpack =True) #x Zahl der Messung, erstesfreiesPendel Periodendauer x5 des Pendels 
#erstesfreiesPendel *= (1/5) #Werte duruch 5 teilen
#MeanerstesfreiesPendel = np.mean(erstesfreiesPendel)
#print(np.mean(erstesfreiesPendel))
#print("erstesfreiesPendel")
#stdMean = np.std(erstesfreiesPendel) 
#print("stdMeanerstesfreiesPendel")
#print(stdMean)         
#Dauer von   1. erstes freies Pendel                2. zweites freies Pendel        3.gekoppelte T plus             4.gekoppelte T minus            5.gekoppelte Pendel Schwingung  6.gekoppelte Pendel Schwebung
Werte = np.array([[np.genfromtxt('l070T15f.txt')/5, np.genfromtxt('l070T25f.txt')/5, np.genfromtxt('l070Tp5g.txt')/5,np.genfromtxt('l070Tm5g.txt')/5,np.genfromtxt('l070T_5g.txt')/5,np.genfromtxt('l070Ts_g.txt')],
              [np.genfromtxt('l100T15f.txt')/5, np.genfromtxt('l100T25f.txt')/5, np.genfromtxt('l100Tp5g.txt')/5,np.genfromtxt('l100Tm5g.txt')/5,np.genfromtxt('l100T_5g.txt')/5,np.genfromtxt('l100Ts_g.txt')]])
print(Werte)  
Mittelwerte = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]

for j in range(0,2):

    for i in range(0, 6):
        Mittelwerte[j][i]= np.mean(Werte[j][i])

print(Mittelwerte)

Standardabweichung = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
for j in range(0,2):

    for i in range(0, 6):
        Standardabweichung[j][i]= np.std(Werte[j][i])
#print(Standardabweichung)

StandardabweichungMittelwert = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
for j in range(0,2):

    for i in range(0, 6):

        StandardabweichungMittelwert[j][i] = (Standardabweichung[j][i])/np.sqrt(10)
#print(StandardabweichungMittelwert)
#Kopplungskonstante aus Tp und Tm vom Pendel und langen Pendel berechnen 
k = (Mittelwerte[0][2]**2 - Mittelwerte[0][3]**2) / (Mittelwerte[0][2]**2 + Mittelwerte[0][3]**2)
print(k, " k")
langk = (Mittelwerte[1][2]**2 - Mittelwerte[1][3]**2) / (Mittelwerte[1][2]**2 + Mittelwerte[1][3]**2)
print(langk, " langk")

#Schwebungsdauer Ts aus Tp und Tm berechnen
Ts = (Mittelwerte[0][2]* Mittelwerte[0][3]) / (Mittelwerte[0][2]- Mittelwerte[0][3])
print(Ts, " Ts")
langTs =Ts = (Mittelwerte[1][2]* Mittelwerte[1][3]) / (Mittelwerte[1][2]- Mittelwerte[1][3])
print(Ts, " langTs")

# in der Form gemessen wp wm ws theoretische KREIS-Frequenzen wp wm ws
# in der zweiten Spalte das gleich für das lange Pendel 
Schwingungsfrequenzen = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
for i in range (0,2):
    for j in range(0,2):
          Schwingungsfrequenzen[j][i] = 1/Mittelwerte[j][i]
for j in range(0,2):
    Schwingungsfrequenzen[j][2] = 1/Mittelwerte[j][5]
print(Schwingungsfrequenzen)


plt.subplot(1, 2, 1)
#plt.plot(x, erstesfreiesPendel, label='Die Periodendauer eins freien Fadenpendes')
plt.xlabel(r'')
plt.ylabel(r'')
plt.legend(loc='best')

#plt.subplot(1, 2, 2)
#plt.plot(x, z, label='Plot 2')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
