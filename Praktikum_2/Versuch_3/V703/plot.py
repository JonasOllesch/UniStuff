import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

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

Messung_1 = np.array(np.genfromtxt('Messdaten/Messung_1&4.txt'))
Messung_1[:,1] = Messung_1[:,1]*1e-6


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
plt.xlabel(r'$ I\mathbin{/} \unit{\ampere} $')
plt.ylabel(r'$ \text{Teilchenzahl} $')
x_plateau = np.linspace(330,700,1000)
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
Totzeit = (22084+18864-40352)/(2*22084*18864) 
writeW(Totzeit, "Näherung der Totzeit")


#Aufgabe d
delta_Q = np.zeros(36)
delta_Q[:] = 120*Messung_1[:,1]/Messung_1[:,2]
writeW(delta_Q, "delta_q")
delta_Q_in_elementarladung = np.zeros(36)
delta_Q_in_elementarladung[:] = delta_Q[:]/(1.602176634e-19)
writeW(delta_Q_in_elementarladung, "delta_Q_in_elementarladung")