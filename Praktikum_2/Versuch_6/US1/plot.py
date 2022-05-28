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

def intensitaet(a,l):
    return -2*a*l

Messung_1 = np.array(np.genfromtxt('Messung_1.txt'))
Messung_1[:,0] = Messung_1[:,0]*10**(-6)
Messung_1[:,1] = Messung_1[:,1]*10**(-3)

Messung_2 = np.array(np.genfromtxt('Messung_2.txt'))
Messung_2[:,0] = Messung_2[:,0]*10**(-6)
Messung_2[:,1] = Messung_2[:,1]*10**(-3)

Messung_3 = np.array(np.genfromtxt('Messung_3.txt'))
Messung_3[:,2] = Messung_3[:,2]*10**(-3)
lnIdI0  = np.log(Messung_3[:,1]/Messung_3[:,0])


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
plt.scatter(Messung_2[:,0]/2,Messung_2[:,1],s=8, c='blue',label="Messdaten")
x_b = np.linspace(22*10**(-6)/2,61*10**(-6),1000)
popt_b, pcov_b = curve_fit(pol_1, Messung_2[:,0]/2, Messung_2[:,1])
y_b = pol_1(popt_b[1], popt_b[0], x_b)
plt.plot(x_b,y_b,c='red',label="Ausgleichsfunktion")
plt.ylabel(r'$l \mathbin{/} \unit{\meter}$')
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.xlim(22*10**(-6)/2,61*10**(-6)/2)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_b.pdf')#dieser Plot sieht nicht gut aus also wenn du ne Idee hast lemmino
plt.clf()                       #Wahrscheinlich ist es sinvoll die sinnlosen Messwerte nicht in die regression einzubinden

writeW(popt_b[1], "popt_b m")
writeW(popt_b[0], "popt_b b")
writeW(np.sqrt(pcov_b[1][1]), "sqrt(pcov_b[1][1]))")
writeW(np.sqrt(pcov_b[0][0]), "sqrt(pcov_b[0][0]))")

#3. Hier muss die doppelte Länge verwendet werden, weil der Impuls hin un zurück gehen muss
plt.scatter(Messung_3[:,2]*2,lnIdI0,s=8, c='blue',label="Messdaten")
x_c = np.linspace(0,0.24,1000)
popt_c, pcov_c = curve_fit(intensitaet,Messung_3[:,2]*2,lnIdI0)
y_c = intensitaet(popt_c[0], x_c)
plt.plot(x_c,y_c,c='red',label="Ausgleichsfunktion")
plt.ylabel(r'$ \ln \left( \frac{I}{I_0} \right) $')
plt.xlabel(r'$l \mathbin{/} \unit{\meter}$')
plt.xlim(0,0.24)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_c.pdf')
plt.clf()
writeW(popt_c, "popt_c")
writeW(np.sqrt(pcov_c), "sqrt(popt_c)")

#N plots ohne Ausreißer
Messung_1_oA = np.array(np.genfromtxt('Messung_1_oA.txt'))
Messung_1_oA[:,0] = Messung_1_oA[:,0]*10**(-6)
Messung_1_oA[:,1] = Messung_1_oA[:,1]*10**(-3)

Messung_2_oA = np.array(np.genfromtxt('Messung_2_oA.txt'))
Messung_2_oA[:,0] = Messung_2_oA[:,0]*10**(-6)
Messung_2_oA[:,1] = Messung_2_oA[:,1]*10**(-3)

Messung_3_oA = np.array(np.genfromtxt('Messung_3_oA.txt'))
Messung_3_oA[:,2] = Messung_3_oA[:,2]*10**(-3)
lnIdI0_oA  = np.log(Messung_3_oA[:,1]/Messung_3_oA[:,0])


#1.
plt.scatter(Messung_1_oA[:,0],2*Messung_1_oA[:,1],s=8, c='blue',label="Messdaten ohne Ausreißer")
x_a_oA = np.linspace(22*10**(-6),90*10**(-6),1000)
popt_a_oA, pcov_a_oA = curve_fit(pol_1, Messung_1_oA[:,0], 2*Messung_1_oA[:,1])
y_a_oA = pol_1(popt_a_oA[1], popt_a_oA[0], x_a_oA)
plt.plot(x_a_oA,y_a_oA,c='red',label="Ausgleichsfunktion ohne Ausreißer")
plt.ylabel(r'$2 \cdot l \mathbin{/} \unit{\meter}$')
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.xlim(22*10**(-6),90*10**(-6))
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_a_oA.pdf')
plt.clf()

writeW(popt_a_oA[1], "popt_a_oA m")
writeW(popt_a_oA[0], "popt_a_oA b")
writeW(np.sqrt(pcov_a_oA[1][1]), "sqrt(pcov_a_oA[1][1]))")
writeW(np.sqrt(pcov_a_oA[0][0]), "sqrt(pcov_a_oA[0][0]))")

#2.Durchschallung
plt.scatter(Messung_2_oA[:,0]/2,Messung_2_oA[:,1],s=8, c='blue',label="Messdaten ohne Ausreißer")
x_b_oA = np.linspace(22*10**(-6)/2,61*10**(-6),1000)
popt_b_oA, pcov_b_oA = curve_fit(pol_1, Messung_2_oA[:,0]/2, Messung_2_oA[:,1])
y_b_oA = pol_1(popt_b_oA[1], popt_b_oA[0], x_b_oA)
plt.plot(x_b_oA,y_b_oA,c='red',label="Ausgleichsfunktion ohne Ausreißer")
plt.ylabel(r'$l \mathbin{/} \unit{\meter}$')
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.xlim(22*10**(-6)/2,61*10**(-6)/2)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_b_oA.pdf')
plt.clf()                       

writeW(popt_b_oA[1], "popt_b_oA m")
writeW(popt_b_oA[0], "popt_b_oA b")
writeW(np.sqrt(pcov_b_oA[1][1]), "sqrt(pcov_b_oA[1][1]))")
writeW(np.sqrt(pcov_b_oA[0][0]), "sqrt(pcov_b_oA[0][0]))")


plt.scatter(Messung_3_oA[:,2]*2,lnIdI0_oA,s=8, c='blue',label="Messdaten ohne Ausreißer")
x_c_oA = np.linspace(0,0.24,1000)
popt_c_oA, pcov_c_oA = curve_fit(intensitaet,Messung_3_oA[:,2]*2,lnIdI0_oA)
y_c_oA = intensitaet(popt_c_oA[0], x_c_oA)
plt.plot(x_c_oA,y_c_oA,c='red',label="Ausgleichsfunktion ohne Ausreißer")
plt.ylabel(r'$ \ln \left( \frac{I}{I_0} \right) $')
plt.xlabel(r'$l \mathbin{/} \unit{\meter}$')
plt.xlim(0,0.24)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_c_oA.pdf')
plt.clf()
writeW(popt_c_oA, "popt_c_oA")
writeW(np.sqrt(pcov_c_oA), "sqrt(popt_c_oA)")
#----------------------
#Auge bin mir nicht sicher ob man das so brechnet. In Altprotokollen haben die 4 Messwerte, aber mit unseren 2 sollte das auch gehen.
T_Iris = 9.06*10**(-6)
L_Iris = 1/2 * 2500 * T_Iris
writeW(L_Iris, "L_Iris")
T_Retina = 16.13*10**(-6)
L_Retina = 2*L_Iris + 1/2*1410*(T_Retina-T_Iris)
writeW(L_Retina, "L_Retina")