import matplotlib.pyplot as pyplot
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import math
from scipy.optimize import curve_fit
from numpy import arange

#wie haben jetzt einzelne Funktionen yay
#Trägheitsmoment eines Zylinder durch seine Symmetrieachse
def Trä_Mo_Zy_pSa(Radius,Masse):
    return (Masse*Radius**2)/2
#Trägheitsmoment eines Zylinder senkrecht zur Symmetrieachse
def Trä_Mo_Zy_sSa(Radius,Länge,Masse):
    return Masse*((Radius**2)/4 + (Länge**2)/12)
def Steiner(Trägheitmoment,Masse,Verschiebung):
    return Trägheitmoment+ Masse*Verschiebung**2

Messung_a = np.array(np.genfromtxt('Messung_a.txt'))
#print(math.sin(math.pi))# <-- das ist anscheinend nicht null aber math arbeitet in Bogenmaß
#print(np.sin(np.pi))      #    np wohl auch
Messung_a[:,0] = ((np.pi)/180)*Messung_a[:,0]# von Grad in Bogenmaß
#und alle andere Umrechnungen in Si Einheiten

Messung_b = np.array(np.genfromtxt('Messung_b.txt'))
Messung_b[:,0]  = Messung_b[:,0]/100
Messung_b[:,0]  = Messung_b[:,0]+(13.55/1000)
Messung_b[:,1]  = Messung_b[:,1]/3

Messung_c = np.array(np.genfromtxt('Messung_c.txt'))
Messung_c[:,1] = Messung_c[:,1]/3

Messung_d = np.array(np.genfromtxt('Messung_d.txt'))
Messung_d[:,1] = Messung_d[:,1]/3

Messung_e = np.array(np.genfromtxt('Messung_e.txt'))
Messung_e[:,1] = Messung_e[:,1]/1000
Messung_e[:,2] = Messung_e[:,2]/1000
Messung_e[:,3] = Messung_e[:,3]/1000
Messung_e[:,4] = Messung_e[:,4]/1000

Messung_f = np.array(np.genfromtxt('Messung_f.txt'))
Messung_f[:,1] = Messung_f[:,1]/3
Messung_f[:,2] = Messung_f[:,2]/3

Messung_g = np.array(np.genfromtxt('Messung_g.txt'))
Messung_g[:,1] = Messung_g[:,1]/3
Messung_g[:,2] = Messung_g[:,2]/3

#Bestimmung der Winkelrichtgröße
WinRgtmp = [0,1,2,3,4,5,6,7,8,9]
for i in range(0,len(WinRgtmp)):
    WinRgtmp[i] = (Messung_a[i][1]*0.2)/Messung_a[i][0]
Winkelrichtgröße = ufloat(np.mean(WinRgtmp),np.std(WinRgtmp))
del WinRgtmp
#Eigenträgheitsmoment bestimmen
print(Messung_b)
#T² gegen a² auftragen
xdata = [0,1,2,3,4,5,6,7,8,9]
ydata = [0,1,2,3,4,5,6,7,8,9]
y2data = [0,1,2,3,4,5,6,7,8,9]
for i in range(0,10):
    xdata[i]= Messung_b[i][0]**2
    ydata[i]= Messung_b[i][1]**2
print(xdata)
print(ydata)
def func(x, a, b):

    return a * x +b

#xdata = Messung_b[:,0]
#y = Messung_b[:,1]
#ydata = y
popt, pcov = curve_fit(func, xdata, ydata)
print( "popt:\n")
print( popt )
print( "pcov:\n")
print( pcov )
#pyplot.plot(xdata, func(xdata,popt[0],popt[1] ), label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
#pyplot.plot(xdata, ydata,color ='green')
for i in range(0,10):
    y2data[i]= func( xdata[i], popt[0], popt[1])

pyplot.plot(xdata, y2data,color ='red', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

print(popt)

#B = ufloat(popt[1][1],np.sqrt(pcov[1][1]))
#A = ufloat(popt[0][0],np.sqrt(pcov[0][0]))
#print(A)
#print(B)


pyplot.scatter(xdata, ydata, color='blue',s=15, label="T² gegen a²")
pyplot.legend()
pyplot.grid()
pyplot.xlabel(r'$a² \mathbin{/}\unit{\meter^2}$')
pyplot.ylabel(r'$T² \mathbin{/}\unit{\second^2}$')
pyplot.savefig('build/T^2ga^2')
pyplot.clf()

#I_eigen = (B*Winkelrichtgröße)

#Trägheitmoment der Kugel
T_k = ufloat(np.mean(Messung_c[:,1]),np.std(Messung_c[:,1]))
I_Kugel_e = (T_k**2*Winkelrichtgröße)/((2*np.pi)**2) 
I_Kugel_t = (2/5)*0.8113*((12.75/2)*10**-2)**2 # 4% ist doch gut

#print(I_Kugel_e," I_Kugel_e")
#print(I_Kugel_t," I_Kugel_t")
#print(abs((I_Kugel_e-I_Kugel_t)/I_Kugel_t)*100, " real Abw Kugel") #22 auch nicht schlecht

#Trägheitmoment des Zylinders
T_z = ufloat(np.mean(Messung_d[:,1]),np.std(Messung_d[:,1]))
I_Zylinder_e = (T_z**2*Winkelrichtgröße)/((2*np.pi)**2)
I_Zylinder_t = (0.3677/2)*(((97.6/2)*10**-3)**2)


#print(I_Zylinder_e, " I_Zylinder_e")
#print(I_Zylinder_t, " I_Zylinder_t")
#print(abs((I_Zylinder_e-I_Zylinder_t)/I_Zylinder_t)*100, " real Abw Zyl") 

#die Puppe
#die Messwerte mitteln
Kopf_r = ufloat(np.mean(Messung_e[:,1]/2), np.std(Messung_e[:,1]))
Arm_r = ufloat(np.mean(Messung_e[:,2]/2), np.std(Messung_e[:,2]))
Torso_r = ufloat(np.mean(Messung_e[:,3]/2), np.std(Messung_e[:,3]))
Bein_r = ufloat(np.mean(Messung_e[:,4]/2), np.std(Messung_e[:,4]))

Kopf_l =ufloat(43.1*10**-3, 0.0001)                     #eine kleine Abweichung, damit es eine gibt
Arm_l = ufloat((128.8+ 129.6)/2*10**-3,0.0001)
Torso_l = ufloat(87.4*10**-3,0.0001)
Bein_l = ufloat((146+146.6)/2*10**-3,0.0001)
#Kopf, Arm, Torso, Bein
#           Radius, Länge, Volumen, Masse, Träg-mo Tuppe, Träg-mo Skuppe
#Wichtig Werte für ein Arm/Bein
Puppenmaße =[[Kopf_r,Kopf_l,0,      0,      0,            0],[Arm_r,Arm_l,0,0,0,0],[Torso_r,Torso_l,0,0,0,0],[Bein_r,Bein_l,0,0,0,0]]
for i in range(0,4):
    Puppenmaße[i][2] = Puppenmaße[i][1]*Puppenmaße[i][0]**2*np.pi

G_Volumen = Puppenmaße[0][2]+Puppenmaße[1][2]*2+Puppenmaße[2][2]+Puppenmaße[3][2]*2# Wichtig das Ding hat zweit Arme und Beine
#print(G_Volumen, " das Gesamtvolumen")
#Masse berechnen
G_Masse = ufloat(0.1674,0.00001)
for i in range(0,4):
    Puppenmaße[i][3]=Puppenmaße[i][2]*G_Masse/G_Volumen
#G_Masse_tmp =   Puppenmaße[0][3]+Puppenmaße[1][3]*2+Puppenmaße[2][3]+Puppenmaße[3][3]*2

Puppenmaße[0][4] = Trä_Mo_Zy_pSa(Puppenmaße[0][0], Puppenmaße[0][3])
Puppenmaße[0][5] = Trä_Mo_Zy_pSa(Puppenmaße[0][0], Puppenmaße[0][3])

Puppenmaße[2][4] = Trä_Mo_Zy_pSa(Puppenmaße[2][0], Puppenmaße[2][3])
Puppenmaße[2][5] = Trä_Mo_Zy_pSa(Puppenmaße[2][0], Puppenmaße[2][3])

TmArm_tmp = Trä_Mo_Zy_sSa(Puppenmaße[1][0],Puppenmaße[1][1], Puppenmaße[1][3])
Puppenmaße[1][4]= Steiner(TmArm_tmp, Puppenmaße[1][3], Puppenmaße[2][0]) #EIN Arm wird um den Radius des Torsos verschoben
Puppenmaße[1][5] = Puppenmaße[1][4]                                      #Tuppe und Skuppe sind gleich
del TmArm_tmp

TmBei_tmp = Trä_Mo_Zy_pSa(Puppenmaße[3][0], Puppenmaße[3][3])
Puppenmaße[3][4] = Steiner(TmBei_tmp, Puppenmaße[3][3], Puppenmaße[3][0])# Verschibung um den Radius eines Beins
del TmBei_tmp

TmBei_tmp2 = Trä_Mo_Zy_sSa(Puppenmaße[3][0], Puppenmaße[3][1], Puppenmaße[3][3])
Puppenmaße[3][5] = Steiner(TmBei_tmp2, Puppenmaße[3][3], Puppenmaße[3][1]+Puppenmaße[2][0])

del TmBei_tmp2
#for j in range(0,6):
#    print('\n')
#    for i in range(0,4):
#        print(Puppenmaße[i][j],i, " ", j)

Tm_T_Puppe_t = Puppenmaße[0][4]+Puppenmaße[1][4]*2+Puppenmaße[2][4]+Puppenmaße[3][4]*2
Tm_S_Puppe_t = Puppenmaße[0][5]+Puppenmaße[1][5]*2+Puppenmaße[2][5]+Puppenmaße[3][5]*2

#Mittelwerte und Standartabweichung der Puppe
#T_Puppe, S_Puppe
#T für 90, T für 120, I_90, I_120
Trägheit_e = [[0,1,2,3],[4,5,6,7]]

for i in range(0,2):
    Trägheit_e[0][i]= ufloat(np.mean(Messung_f[:,i+1]),np.std(Messung_f[:,i+1]))
    Trägheit_e[1][i]= ufloat(np.mean(Messung_g[:,i+1]),np.std(Messung_g[:,i+1]))

for j in range(0,2):
    for i in range(0,2):
        Trägheit_e[j][i+2]= Trägheit_e[j][i]**2*Winkelrichtgröße/(4*np.pi**2)

rel_Verhältnis_90 = Trägheit_e[0][2]/Trägheit_e[1][2]#Verhältnis von der T-Puppe und der S-Puppe
rel_Verhältnis_120 = Trägheit_e[0][3]/Trägheit_e[1][3]

rel_Verhältnis_theorie = Tm_T_Puppe_t/Tm_S_Puppe_t

rel_Abw_Ver_90 = (rel_Verhältnis_90-rel_Verhältnis_theorie)/rel_Verhältnis_theorie
rel_Abw_Ver_120 = (rel_Verhältnis_120-rel_Verhältnis_theorie)/rel_Verhältnis_theorie

#Woooooooooooooooooooooooooooooooow