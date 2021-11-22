import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import scipy.constants as const
import sympy
from scipy.stats import linregress
import uncertainties.unumpy as unp

# do list
#Schwebungsdauer mit den gemessenen vergleichen
#Fehlerfortpflanzung für irgendwie alles xD

#Dauer von   1. erstes freies Pendel                2. zweites freies Pendel        3.gekoppelte T plus             4.gekoppelte T minus            5.gekoppelte Pendel Schwingung  6.gekoppelte Pendel Schwebung
Werte = np.array([[np.genfromtxt('l070T15f.txt')/5, np.genfromtxt('l070T25f.txt')/5, np.genfromtxt('l070Tp5g.txt')/5,np.genfromtxt('l070Tm5g.txt')/5,np.genfromtxt('l070T_5g.txt')/5,np.genfromtxt('l070Ts_g.txt')],
                    [np.genfromtxt('l100T15f.txt')/5, np.genfromtxt('l100T25f.txt')/5, np.genfromtxt('l100Tp5g.txt')/5,np.genfromtxt('l100Tm5g.txt')/5,np.genfromtxt('l100T_5g.txt')/5,np.genfromtxt('l100Ts_g.txt')]])
print(Werte)  
Mittelwerte = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
Mittelwerte_ = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]




for j in range(0, 2):
    for i in range(0, 6):
        Mittelwerte_[j][i] = ufloat(np.mean(Werte[j][i]), np.std(Werte[j][i]))
#print(Mittelwerte_, "Mittelwerte_ mit Standartabweichung")

k_ = [[1],[1]]
for j in range (0,2):
    k_[j][0] = (Mittelwerte_[j][2]**2 - Mittelwerte_[j][3]**2)/(Mittelwerte_[j][2]**2 + Mittelwerte_[j][3]**2)

print(k_ , " k_")

Ts_= [[1],[1]]
for j in range(0,2):
    Ts_[j][0]= Mittelwerte_[j][2]*Mittelwerte_[j][3]/(Mittelwerte_[j][2]-Mittelwerte_[j][3])
print(Ts_ , " Ts_")
    

Frequenzen_ = [[1, 2, 3], [1, 2, 3]]

for j in range(0,2):
        Frequenzen_[j][0] = (2*np.pi)/Mittelwerte_[j][2]  #wp
        Frequenzen_[j][1] = (2*np.pi)/Mittelwerte_[j][3]  #wm
        Frequenzen_[j][2] = (2*np.pi)/Mittelwerte_[j][5]  #ws

print(Frequenzen_, " frequenzen_")


for j in range(0,2):

    for i in range(0, 6):
        Mittelwerte[j][i]= np.mean(Werte[j][i])

#print(Mittelwerte)

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

#in der Form gemessen wp wm ws 
#n der zweiten Spalte das gleich für das lange Pendel 
Frequenzen = [[1, 2, 3], [1, 2, 3]]

for j in range(0,2):
        Frequenzen[j][0] = (2*np.pi)/Mittelwerte[j][2]  #wp
        Frequenzen[j][1] = (2*np.pi)/Mittelwerte[j][3]  #wm
        Frequenzen[j][2] = (2*np.pi)/Mittelwerte[j][5]  #ws

print(Frequenzen, " frequenzen")
kausw = [[1], [1]]
for j in range(0,2):
    kausw[j][0] = (Frequenzen[j][1]**2-Frequenzen[j][0]**2) / (Frequenzen[j][1]**2 + Frequenzen[j][0]**2)

print(kausw, " kausw")

Eigenfrequenzen = [[1,2,3],[1,2,3]]     #theoretische Frequenzen der Schwingungen

Eigenfrequenzen[0][0] = np.sqrt(9.81/0.7)   #wp
Eigenfrequenzen[1][0] = np.sqrt(9.81/1)     #wp lang

Eigenfrequenzen[0][1] = np.sqrt((9.81/0.7) + (2*k/0.7))     #wm
Eigenfrequenzen[1][1] = np.sqrt((9.81/0.7) + (2*langk/0.7)) #wm lang

for j in range (0,2):
    Eigenfrequenzen[j][2] = Eigenfrequenzen[j][0]-Eigenfrequenzen[j][1] 

print(Eigenfrequenzen, " Eigenfrequenzen") #WARUM die Schwebungsfrequenz negativ ?!?

#Fehlerfortpflanzung for k
Tp , Tm = sympy.var('Tp ,Tm')

fk = (Tp**2 - Tm**2 )/ (Tp**2 +Tm**2)
fk_Tp = fk.diff(Tp)
fk_Tm = fk.diff(Tm)




FehlerausFortpflanzung = [[1,2], [1,2]]    
for j in range(0,2):
    FehlerausFortpflanzung[j][0]= (fk_Tp.evalf(subs={Tp:Mittelwerte[j][2], Tm:Mittelwerte[j][3]})*Standardabweichung[j][2])**2+ (fk_Tm.evalf(subs={Tp:Mittelwerte[j][2], Tm:Mittelwerte[j][3]})*Standardabweichung[j][3])**2
    FehlerausFortpflanzung[j][0]= sympy.sqrt(FehlerausFortpflanzung[j][0])

fTs = (Tp*Tm)/(Tp-Tm) #für Ts
fTs_Tp = fTs.diff(Tp)
fTs_Tm = fTs.diff(Tm)



for j in range(0,2):
    FehlerausFortpflanzung[j][1] = (fTs_Tp.evalf(subs={Tp:Mittelwerte[j][2], Tm:Mittelwerte[j][3]})*Standardabweichung[j][2])**2+ (fTs_Tm.evalf(subs={Tp:Mittelwerte[j][2], Tm:Mittelwerte[j][3]})*Standardabweichung[j][3])**2
    FehlerausFortpflanzung[j][1]= sympy.sqrt(FehlerausFortpflanzung[j][1])

print(FehlerausFortpflanzung, " Fehlerfortpflanzung")

#print(fk)

#erster Plot für das erste freie Pendel, zumindest versuche ich das zu plotten :)
x= np.linspace(1,10,10)
y = np.genfromtxt('l070T1f.txt', unpack=True)

#Fehlerbalken und Mittelwert, obwohl du den schon hast lel

errY= np.std(y)/np.sqrt(len(y))
plt.errorbar(x, y, xerr=0, yerr=errY, fmt='o', label='erstes Pendel', markersize=3)

#Lineare Regression erstes freies Pendel

b,a,r,p,std= linregress(x,y)
plt.plot(x, b*x+a, 'g')

#Für das zweite Pendel
y = np.genfromtxt('l070T2f.txt', unpack=True)

errY= np.std(y)/np.sqrt(len(y))
plt.errorbar(x, y, xerr=0, yerr=errY, fmt='o', label='zweites Pendel', markersize=3)

#Lineare Regression zweites freies Pendel

b,a,r,p,std= linregress(x,y)
plt.plot(x, b*x+a, 'r')

#Achsenbeschriftungen und Legende

plt.xlim(1,10.05)
plt.ylim(1.6,1.8)
plt.xticks(np.arange(1,10.05, step=1))
plt.yticks(np.arange(1.6,1.81, step=0.05))
plt.xlabel('Anzahl der Messungen')
plt.ylabel(r'$T_i$/\,\si{s}')
plt.legend(loc='best')
#klappt :)


#np.savetxt('3.txt', np.column_stack([data_x, data_y]), header='x y')

#print(f'Das Ergebnis ist {result:.2f}')
#print('{:.1u}, {:.3uf}, {:.2uL}, {:0.2ue}'.format(a,a,a,a))
# 0.024626+/-0.000003, 0.02462631+/-0.00000335, 0.0246263 \pm 0.0000034, (2.46263+/-0.00034)e-02
for j in range(0,2):
    for i in range(0,6):
        print('{:.2uL}'.format(Mittelwerte_[j][i]) , "[j],[i]=",  j ," ", i)


for j in range(0,2):
    for i in range(0,6):
        Mittelwerte_[j][i] = '{:.2uL}'.format(Mittelwerte_[j][i])


print(Mittelwerte_)
np.savetxt('build/Mittelwerte.txt',np.column_stack([Mittelwerte]), header='1. erstes freies Pendel  2. zweites freies Pendel 3.gekoppelte T plus 4.gekoppelte T minus  5.gekoppelte Pendel Schwingung  6.gekoppelte Pendel Schwebung')
np.savetxt('build/Standardabweichungen.txt',np.column_stack([Standardabweichung]), header='1. erstes freies Pendel  2. zweites freies Pendel 3.gekoppelte T plus 4.gekoppelte T minus  5.gekoppelte Pendel Schwingung  6.gekoppelte Pendel Schwebung')
np.savetxt('build/StandardabweichungMittelwert.txt',np.column_stack([StandardabweichungMittelwert]), header='1. erstes freies Pendel  2. zweites freies Pendel 3.gekoppelte T plus 4.gekoppelte T minus  5.gekoppelte Pendel Schwingung  6.gekoppelte Pendel Schwebung')

np.savetxt('build/Frequenzen.txt',np.column_stack([Frequenzen]),header='Frequenzen  wp wm ws ')

#plt.subplot(1, 2, 2)
#plt.plot(x, z, label='Plot 2')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
