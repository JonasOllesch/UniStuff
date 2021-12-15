import matplotlib.pyplot as pyplot
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import math
from scipy.optimize import curve_fit
from numpy import arange

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
Messung_g[:,1] = Messung_g[:,1]/3

#Bestimmung der Winkelrichtgröße
WinRgtmp = [0,1,2,3,4,5,6,7,8,9]
for i in range(0,len(WinRgtmp)):
    WinRgtmp[i] = (Messung_a[i][1]*0.2)/Messung_a[i][0]
Winkelrichtgröße = ufloat(np.mean(WinRgtmp),np.std(WinRgtmp))
del WinRgtmp

#T² gegen a² auftragen
x = [0,1,2,3,4,5,6,7,8,9]
y = [0,1,2,3,4,5,6,7,8,9]


#for i in range(0,10):
#    x[i]= Messung_b[i][0]**2
#    y[i]= Messung_b[i][1]**2
#
#pyplot.scatter(x, y, color='blue',s=15, label="T² gegen a²")
#pyplot.legend()
#pyplot.grid()
#pyplot.xlabel(r'$a² \mathbin{/}\unit{\meter^2}$')
#pyplot.ylabel(r'$T² \mathbin{/}\unit{\second^2}$')
##pyplot.show()
#pyplot.savefig('build/T^2ga^2')
#pyplot.clf()


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
Kopf_b = ufloat(np.mean(Messung_e[:,1]), np.std(Messung_e[:,1]))
Arm_b = ufloat(np.mean(Messung_e[:,2]), np.std(Messung_e[:,2]))
Torso_b = ufloat(np.mean(Messung_e[:,3]), np.std(Messung_e[:,3]))
Bein_b = ufloat(np.mean(Messung_e[:,4]), np.std(Messung_e[:,4]))

Kopf_l =ufloat(43.1*10**-3, 0.0001)                     #eine kleine Abweichung damit es eine gibt
Arm_l = ufloat((128.8+ 129.6)/2*10**-3,0.0001)
Torso_l = ufloat(87.4*10**-3,0.0001)
Bein_l = ufloat((146+146.6)/2*10**-3,0.0001)

print(T_k)
#print(Kopf_b)
#print(Arm_b)
#print(Torso_b)
#print(Bein_b)
#print('\n')
#print(Kopf_l)
#print(Arm_l)
#print(Torso_l)
#print(Bein_l)
