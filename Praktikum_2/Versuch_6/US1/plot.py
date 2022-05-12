import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
output = ("build/Auswertung")    
my_file = open(output + '.txt', "w")
def writeW(Wert,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    try:
        for i in Wert:
            my_file.write(str(i))
            my_file.write('\n')
    except:
        my_file.write(str(Wert))
        my_file.write('\n')

    return 0

def pol_1(a,b,x):
    return a*x+b

Messung_1 = np.array(np.genfromtxt('Messung_1.txt'))
Messung_1[:,0] = Messung_1[:,0]*10**(-6)
Messung_1[:,1] = Messung_1[:,1]*10**(-3)

Messung_2 = np.array(np.genfromtxt('Messung_2.txt'))
Messung_2[:,0] = Messung_2[:,0]*10**(-6)
Messung_2[:,1] = Messung_2[:,1]*10**(-3)




D_Acrylplatte_theo = 0.1
T_1_2 = 7.43 *10**(-6)
T_2_3 = 7.41 *10**(-6)
T_3_4 = 7.30 *10**(-6)

#1. 
#mit der Schieblehre gemessene Zylinder
h_z1_l = ufloat(28.1*10**(-3),0.05*10**(-3) )
h_z2_l = ufloat(37.5*10**(-3) ,0.05*10**(-3))
h_z3_l = ufloat(77.5*10**(-3) ,0.05*10**(-3))
h_z4_l = ufloat(99.1*10**(-3) ,0.05*10**(-3))
h_z5_l = ufloat(117.8*10**(-3),0.05*10**(-3))

plt.scatter(Messung_1[:,0],2*Messung_1[:,1],s=8, c='blue',label="Messdaten")
x_a = np.linspace(22*10**(-6),90*10**(-6),1000)
popt_a, pcov_a = curve_fit(pol_1, Messung_1[:,0], 2*Messung_1[:,1])
y_a = pol_1(popt_a[1], popt_a[0], x_a)
plt.plot(x_a,y_a,c='red',label="Ausgleichsfunktion")
plt.ylabel(r'$2 \cdot l \mathbin{/} \unit{\meter}$')
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.xlim(22*10**(-6),90*10**(-6))
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_a.pdf')
plt.clf()

writeW(popt_a[1], "popt_a m")
writeW(popt_a[0], "popt_a b")
writeW(np.sqrt(pcov_a[1][1]), "sqrt(pcov_a[1][1]))")
writeW(np.sqrt(pcov_a[0][0]), "sqrt(pcov_a[0][0]))")

#2.
plt.scatter(Messung_2[:,0],Messung_2[:,1],s=8, c='blue',label="Messdaten")
x_b = np.linspace(22*10**(-6),61*10**(-6),1000)
popt_b, pcov_b = curve_fit(pol_1, Messung_2[:,0], Messung_2[:,1])
y_b = pol_1(popt_b[1], popt_b[0], x_b)
plt.plot(x_b,y_b,c='red',label="Ausgleichsfunktion")
plt.ylabel(r'$l \mathbin{/} \unit{\meter}$')
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.xlim(22*10**(-6),61*10**(-6))
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_b.pdf')#dieser Plot sieht nicht gut aus also wenn du ne Idee hast lemmino
plt.clf()                       #Wahrscheinlich ist es sinvoll die sinnlosen Messwerte nicht in die regression einzubinden

writeW(popt_b[1], "popt_b m")
writeW(popt_b[0], "popt_b b")
writeW(np.sqrt(pcov_b[1][1]), "sqrt(pcov_b[1][1]))")
writeW(np.sqrt(pcov_b[0][0]), "sqrt(pcov_b[0][0]))")

#Auge
T_Iris = 9.06*10**(-6)
T_Retina = 16.13*10**(-6)