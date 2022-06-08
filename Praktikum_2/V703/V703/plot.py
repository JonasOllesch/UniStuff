import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy


output = ("Auswertung")   
my_file = open(output + '.txt', "w") 

def pol_1(a,b,x):
    return a*x+b

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

Messung_1 = np.array(np.genfromtxt('Messdaten/Messung_1&4.txt'))
Messung_1[:,1] = Messung_1[:,1]*1e-6

Messung_2 = np.array(np.genfromtxt('Messdaten/2-quellen-methode.txt'))



#Aufgabe a der Plateaubereich geht von 350 bis 550. Das habe ich jetzt so bestimmt. Du kannst das aber gerne ändern, wenn du möchtest. 

plt.scatter(Messung_1[:,0],Messung_1[:,1],s=8, c='blue',label="Spannung - Stromstärke")

plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I\mathbin{/} \unit{\ampere}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_a.pdf')
plt.clf()


plt.scatter(Messung_1[:,0],Messung_1[:,2],s=8, c='red',label="Spannung - Teilchenzahl")
plt.xlabel(r'$ V\mathbin{/} \unit{\volt} $')
plt.ylabel(r'$ \text{Teilchenzahl} $')
x_plateau = np.linspace(330,700,1000)

popt, pcov = curve_fit(pol_1, Messung_1[:22,0], Messung_1[:22,2])
writeW(popt, "curve fit Parameter")
writeW(np.sqrt(pcov[0][0]), "curve fit Wurzel der Unsicherheit pcov[0][0]")
writeW(np.sqrt(pcov[1][1]), "curve fit Wurzel der Unsicherheit pcov[1][1]")




koeffizienten_plateau = np.polyfit(Messung_1[:22,0],Messung_1[:22,2],1)
y_plateau = np.polyval(koeffizienten_plateau,x_plateau)
plt.plot(x_plateau,y_plateau,c='blue',label="Ausgleichsgerade")
plt.plot([400,400],[14400,15600], color='green') # Das sind die Senkrechten, die Carl gerne haben wollte.
plt.plot([600,600],[14400,15600], color='green') # Das sind die Senkrechten, die Carl gerne haben wollte.
plt.xlim(350,700)
plt.ylim(14400,15800)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_b.pdf')
plt.clf()

writeW(koeffizienten_plateau, "die Koeffizienten der Ausgleichsfunktion der Plateaufunktion")
writeW(Messung_1[:22,0], "Beschreibung")

#Aufgabe c
#zwei Quellen Methoden

Messung_2_N_1 = ufloat(Messung_2[0]/120,np.sqrt(Messung_2[0])/120)
Messung_2_N_2 = ufloat(Messung_2[2]/120,np.sqrt(Messung_2[2])/120)
Messung_2_N_2p1 = ufloat(Messung_2[1]/120,np.sqrt(Messung_2[1])/120)

Totzeit = (Messung_2_N_1+Messung_2_N_2-Messung_2_N_2p1)/(2*Messung_2_N_1*Messung_2_N_2)  

writeW(Totzeit, "Näherung der Totzeit")
writeW(Messung_2_N_1, "Messung_2_N_1")
writeW(Messung_2_N_2, "Messung_2_N_2")
writeW(Messung_2_N_2p1, "Messung_2_N_2p1")




#Aufgabe d
delta_Q = unumpy.uarray(Messung_1[:,2],np.sqrt(Messung_1[:,2]))
delta_I = unumpy.uarray(Messung_1[:,1],0.2*10**-7)

for i in range(0,36):
    delta_Q[i] = 120*delta_I[i]/delta_Q[i]

writeW(delta_Q, "delta_Q")

delta_Q_in_elementarladung = np.zeros(36,dtype="object")
for i in range(0,36):
    delta_Q_in_elementarladung[i] = delta_Q[i]/(1.602176634e-19)


writeW(delta_Q_in_elementarladung, "delta_Q_in_elementarladung")


for i in range(0,36):
    plt.errorbar(Messung_1[i][0], delta_Q[i].nominal_value, xerr=0, yerr=delta_Q[i].std_dev, fmt='o',marker='x',color ='b')

delta_Q_tmp = np.zeros(36)
for i in range(0,36):
    delta_Q_tmp[i] = delta_Q[i].nominal_value

plt.errorbar(Messung_1[0][0], delta_Q[0].nominal_value, xerr=0, yerr=delta_Q[0].std_dev, fmt='o',marker='x',color ='b',label="Messdaten")
x_c = np.linspace(340,710,1000)
popt_c, pcov_c = curve_fit(pol_1, Messung_1[:,0],delta_Q_tmp )
print(curve_fit(pol_1, Messung_1[:,0],delta_Q_tmp ))
y_c = pol_1(popt_c[1], popt_c[0], x_c)
plt.plot(x_c,y_c,label ="Ausgleichsgrade",c='r')

writeW(popt_c, "popt_c")
writeW(np.sqrt(pcov), "pcov_c")

plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$\Delta Q \mathbin{/} \unit{\coulomb}$')
plt.xlim(340,710)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_c.pdf')
plt.clf()



#Abweichung der Zahlrate
writeW(np.sqrt(Messung_1[:,2]), "Messung_1 Fehlerabweichung")


writeW(Messung_2, " 2 Zählrate Quellen Methode")
writeW(np.sqrt(Messung_2), "Abweichung 2 Quellen Methode")
