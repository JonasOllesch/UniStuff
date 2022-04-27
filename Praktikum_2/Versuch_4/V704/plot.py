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


plt.plot(x_Ausgleich_1_a,y_Ausgleich_1_a,label="Ausgleichsgerade",c='r')
plt.errorbar(Messung_1_a[:,0], Aktivität_1_a_min_a0[:], xerr=0.02*(10**-3), yerr=Aktivität_1_a_min_a0_poissonabw[:], fmt='o',marker='x',label="Messdaten")
plt.xlabel(r'$ \text{Dicke D} \, \mathbin{/} \unit{\meter}$')
plt.ylabel(r'$ \left( \text{A} - \text{A}_0 \right) \cdot \unit{\second}  $')
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
writeW(popt[0],"Apsorptionskoeffizien 1 b")
writeW(popt[1],"Null Aktivität 1 b")
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
plt.errorbar(Messung_2[:,0], Aktivität_2[:], xerr=Messung_2[:,3], yerr=Aktivität_2_poissonabw[:], fmt='o',marker='x',label="Messdaten")


writeW(Messung_2[:6,0], "Test")
writeW(Messung_2[6:,0], "Test")

x_Ausgleich_2_1 = np.linspace(0,0.0005,10000)
popt_21, pcov_21 = curve_fit(func_e, Messung_2[6:,0], Aktivität_2[6:])
y_Ausgleich_2_1 = func_e(x_Ausgleich_2_1,popt_21[0] , popt_21[1])
plt.plot(x_Ausgleich_2_1,y_Ausgleich_2_1,label="durchgehende Strahlungsintensität",c='r')



x_Ausgleich_2_2 = np.linspace(0,0.0005,10000)
popt_22, pcov_22 = curve_fit(func_e, Messung_2[:6,0], Aktivität_2[:6])
y_Ausgleich_2_2 = func_e(x_Ausgleich_2_2,popt_22[0] , popt_22[1])
plt.plot(x_Ausgleich_2_2,y_Ausgleich_2_2,label="Untergrundstrahlung",c='b')

R_max= (ufloat(popt_22[1],np.sqrt(pcov_22[1][1]))-ufloat(popt_21[1],np.sqrt(pcov_21[1][1])))/(ufloat(popt_21[0],np.sqrt(pcov_21[0][0]))-ufloat(popt_22[0],np.sqrt(pcov_22[0][0])))
writeW(R_max, "R_max in J")
writeW(R_max/(1.602176634*10**-9), "R_max in eV")


plt.xlabel(r'$ \text{Dicke D} \, \mathbin{/} \unit{\meter}$')
plt.ylabel(r'$ \text{A} \cdot \unit{\second}  $')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.grid()
plt.ylim(0.1,100)
plt.xlim(0,0.0005)
plt.savefig('build/Graph_c.pdf')
plt.clf()

