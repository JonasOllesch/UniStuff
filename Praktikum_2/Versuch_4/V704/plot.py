import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties as unp


output = ("Auswertung")   
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


def func_e(x, a, b):
    return np.exp(x*a+b)


Aktivität_a0 = 845/900
Aktivität_a0_abw = np.sqrt(845)/900
Aktivität_b0 = 845/900
Aktivität_b0_abw = np.sqrt(845)/900
Aktivität_20 = 579/900
Aktivität_20_abw = np.sqrt(845)/900




Messung_1_a = np.array(np.genfromtxt('Messung_1_a.txt'))
Messung_1_a[:,0] = Messung_1_a[:,0]*(10**-3)

Messung_1_b = np.array(np.genfromtxt('Messung_1_b.txt'))
Messung_1_b[:,0] = Messung_1_b[:,0]*(10**-3)


Messung_2 = np.array(np.genfromtxt('Messung_2.txt'))
Messung_2[:,0] = Messung_2[:,0]*(10**-6)
Messung_2[:,3] = Messung_2[:,3]*(10**-6)

writeW(np.sqrt(Messung_2), "Messung_2 Abweichung")


Zählrate_1_a_poissonabw = np.sqrt(Messung_1_a[:,1])

Aktivität_1_a =Messung_1_a[:,1]/Messung_1_a[:,2]
Aktivität_1_a_poissonabw = Zählrate_1_a_poissonabw[:]/Messung_1_a[:,2]
Aktivität_1_a_min_a0 = Aktivität_1_a -Aktivität_a0
Aktivität_1_a_min_a0_poissonabw =Aktivität_1_a_poissonabw 

#print(Aktivität_1_a)
#print(Aktivität_1_a_poissonabw)
#print(Aktivität_1_a_min_a0)

x_Ausgleich_1_a = np.linspace(1e-4,0.021,10000)
popt, pcov = curve_fit(func_e, Messung_1_a[:,0], Aktivität_1_a_min_a0)
y_Ausgleich_1_a = func_e(x_Ausgleich_1_a,popt[0] , popt[1])
writeW(popt[0],"Apsorptionskoeffizien 1 a")
writeW(popt[1],"Null Aktivität 1 a")
writeW(np.sqrt(pcov[0][0]),"Apsorptionskoeffizien 1 a Abweichung")
writeW(np.sqrt(pcov[1][1]),"Null Aktivität 1 a Abweichung")

plt.plot(x_Ausgleich_1_a,y_Ausgleich_1_a,label="Ausgleichsgerade",c='r')
plt.errorbar(Messung_1_a[:,0], Aktivität_1_a_min_a0[:], xerr=0.02*(10**-3), yerr=Aktivität_1_a_min_a0_poissonabw[:], fmt='o',marker='x',label="Messdaten")
plt.xlabel(r'$ \text{Dicke D} \, \mathbin{/} \unit{\meter}$')
plt.ylabel(r'$ \left(\text{A} - \text{A}_0 \right) \mathbin{/} \unit{\frac{1}{\second}}  $')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.grid()
plt.ylim(0,120)
plt.xlim(1e-4,0.021)
plt.savefig('build/Graph_a.pdf')# Der Plot sieht ja so kacke aus aber da kann man nichts machen. Irgendwer hat halt dumme Messwerte augenommen.(Ich war es nicht ;))
plt.clf()






Zählrate_1_b_poissonabw = np.sqrt(Messung_1_b[:,1])
Aktivität_1_b =Messung_1_b[:,1]/Messung_1_b[:,2]
Aktivität_1_b_poissonabw = Zählrate_1_b_poissonabw[:]/Messung_1_b[:,2]
Aktivität_1_b_min_b0 = Aktivität_1_b -Aktivität_b0
Aktivität_1_b_min_b0_poissonabw =Aktivität_1_b_poissonabw 
#print(Aktivität_1_b)
#print(Aktivität_1_b_poissonabw)
#print(Aktivität_1_b_min_b0)
x_Ausgleich_1_b = np.linspace(0,0.022,10000)
popt, pcov = curve_fit(func_e, Messung_1_b[:,0], Aktivität_1_b_min_b0)
y_Ausgleich_1_b = func_e(x_Ausgleich_1_b,popt[0] , popt[1])

writeW(popt[0],"Apsorptionskoeffizient 1 b")
writeW(np.sqrt(pcov[0][0]),"Apsorptionskoeffizien 1 b Abweichung")
writeW(popt[1],"Null Aktivität 1 b")
writeW(np.sqrt(pcov[1][1]),"Null Aktivität 1 b Abweichung")
plt.plot(x_Ausgleich_1_b,y_Ausgleich_1_b,label="Ausgleichsgerade",c='r')
plt.errorbar(Messung_1_b[:,0], Aktivität_1_b_min_b0[:], xerr=0.02*(10**-3), yerr=Aktivität_1_b_min_b0_poissonabw[:], fmt='o',marker='x',label="Messdaten")


plt.xlabel(r'$ \text{Dicke D} \, \mathbin{/} \unit{\meter}$')
plt.ylabel(r'$ \left( \text{A} - \text{A}_0 \right) \cdot \unit{\second}  $')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.grid()
plt.ylim(45,200)
plt.xlim(0,0.022)
plt.savefig('build/Graph_b.pdf')# Der Plot sieht ja so kacke aus aber da kann man nichts machen. Irgendwer hat halt dumme Messwerte augenommen.(Ich war es nicht ;))
plt.clf()#Dieser Plot sieht noch dümmer aus





Zählrate_2_poissonabw = np.sqrt(Messung_2[:,1])
Aktivität_2 =Messung_2[:,1]/Messung_2[:,2]
Aktivität_2_poissonabw = Zählrate_2_poissonabw[:]/Messung_2[:,2]
Aktivität_2_min_a0 = Aktivität_2 - Aktivität_20
Aktivität_2_min_a0_poissonabw = Aktivität_2_poissonabw - Aktivität_20_abw

plt.errorbar(Messung_2[:,0]*2.6989*1000, Aktivität_2[:], xerr=Messung_2[:,3], yerr=Aktivität_2_poissonabw[:], fmt='o',marker='x',label="Messdaten")



x_Ausgleich_2_1 = np.linspace(0,1.4,1000)
popt_21, pcov_21 = curve_fit(func_e, Messung_2[6:,0]*2.6989*1000, Aktivität_2[6:])
y_Ausgleich_2_1 = func_e(x_Ausgleich_2_1,popt_21[0] , popt_21[1])
plt.plot(x_Ausgleich_2_1,y_Ausgleich_2_1,label="durchgehende Strahlungsintensität",c='r')

x_Ausgleich_2_2 = np.linspace(0,1.4,1000)
popt_22, pcov_22 = curve_fit(func_e, Messung_2[:6,0]*2.6989*1000, Aktivität_2[:6])
y_Ausgleich_2_2 = func_e(x_Ausgleich_2_2,popt_22[0] , popt_22[1])
plt.plot(x_Ausgleich_2_2,y_Ausgleich_2_2,label="Untergrundstrahlung",c='b')#*2.6989*1000 muss da noch irgendwie rein

writeW(popt_21, 'Ausgleichgerade 1')
writeW(popt_22, 'Ausgleichgerade 2')

R_max= (ufloat(popt_22[1],np.sqrt(pcov_22[1][1]))-ufloat(popt_21[1],np.sqrt(pcov_21[1][1])))/(ufloat(popt_21[0],np.sqrt(pcov_21[0][0]))-ufloat(popt_22[0],np.sqrt(pcov_22[0][0])))
writeW(R_max, "R_max in g/m^2")

E_max = 1.92*(R_max**2 + 0.22*R_max)**(1/2)
writeW(E_max, "E_max in MeV")
#writeW(E_max/(1.6*1e-19), "E_max in eV") 
#writeW(E_max/6.242e+18, "E_max in eV") #Was soll das denn du willst doch in evolt umrechnen oder ? 


plt.xlabel(r'$ \text{Massenbelegung R} \, \mathbin{/} \dfrac{\unit{\kilo\gram}}{\unit{\meter²}}$')
plt.ylabel(r'$ \text{A} \cdot \unit{\second}  $')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.grid()
plt.ylim(0.1,100)
plt.xlim(0.2,1.4)
plt.savefig('build/Graph_c.pdf')
plt.clf()

writeW( Aktivität_a0," Aktivität_a0") 
writeW (Aktivität_a0_abw,"Aktivität_a0_abw")

writeW(Aktivität_1_a_min_a0, "Aktivität_1_a_min_a0")
writeW(Aktivität_1_a_min_a0_poissonabw, "Aktivität_1_a_min_a0_poissonabw")


popt_21[1] = np.exp(popt_21[1])
popt_22[1] = np.exp(popt_22[1])


writeW(popt_21, "popt_21")
writeW(popt_22, "popt_22")
b_21 = ufloat(popt_21[1],np.sqrt(pcov_21[1][1]))
b_22 = ufloat(popt_22[1],np.sqrt(pcov_22[1][1]))

m_21 = ufloat(popt_21[0],np.sqrt(pcov_21[0][0]))
m_22 = ufloat(popt_22[0],np.sqrt(pcov_22[0][0]))



writeW(Aktivität_1_b_min_b0, "Aktivität_1_b_min_a0")
writeW(Aktivität_1_b_min_b0_poissonabw, "Aktivität_1_b_min_a0_poissonabw")

writeW(Aktivität_2_min_a0, "Aktivität_2_min_a0")
writeW(Aktivität_2_min_a0_poissonabw, "Aktivität_2_min_a0_poissonabw")
