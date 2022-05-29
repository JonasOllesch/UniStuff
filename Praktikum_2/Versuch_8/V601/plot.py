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

def berechnesaettigung(T):
    return (5.5*10**7)*np.exp(-6876/T)
Messung_1= np.array(np.genfromtxt('data/Messung1_25_7C.txt'))
Messung_2= np.array(np.genfromtxt('data/Messung1_142C.txt'))


saettigung = np.zeros(4)
saettigung[0]= berechnesaettigung(25.7+273.15)
saettigung[1]= berechnesaettigung(142+273.15)
saettigung[2]= berechnesaettigung(173+273.15)
saettigung[3]= berechnesaettigung(191.2+273.15)
writeW(saettigung, "Sättigung")

mittlerefreieWeglänge = np.zeros(4)
mittlerefreieWeglänge[:] = (0.0029/saettigung[:])/100





plt.scatter(Messung_1[:,0],Messung_1[:,1],s=8, c='blue',label="25,7°C",marker='x')
plt.scatter(Messung_2[:,0],Messung_2[:,1],s=8,c='#F9AF1F',label="142°C",marker='x')
plt.tick_params(left = False, labelleft = False)
plt.ylabel(r'$I_A$')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_a.pdf')
plt.clf()

stuekweiserdiffquo25_7 = np.zeros(26)
stuekweiserdiffquo142 = np.zeros(26)


for i in range(0,24):
    stuekweiserdiffquo25_7[i]=(Messung_1[i+1][1]-Messung_1[i][1])/(Messung_1[i+1][0]-Messung_1[i][0])
    stuekweiserdiffquo142[i]= (Messung_2[i+1][1]-Messung_2[i][1])/(Messung_2[i+1][0]-Messung_2[i][0])



plt.scatter(Messung_1[:,0],stuekweiserdiffquo25_7[:],s=8, c='blue',label="25,7°C",marker='x')
plt.scatter(Messung_2[:,0],stuekweiserdiffquo142[:],s=8,c='#F9AF1F',label="142°C",marker='x')
plt.plot(Messung_1[:,0],stuekweiserdiffquo25_7[:], c='blue')
plt.plot(Messung_2[:,0],stuekweiserdiffquo142[:],c='#F9AF1F')
plt.tick_params(left = False, labelleft = False)
plt.ylabel(r'$I_{A}  \,  \' $')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_b.pdf')
plt.clf()
