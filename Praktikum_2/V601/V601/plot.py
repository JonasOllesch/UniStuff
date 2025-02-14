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
Messung_3= np.array(np.genfromtxt('data/Messung2_173_5C.txt'))
Messung_3max = np.array([[16.5,23],[22.8,40],[30.6,50],[37.6,54],[46.6,64]])
deltaUmax1 = np.zeros(4)
deltaUmax1[0]= 6.3
deltaUmax1[1]= 7.8
deltaUmax1[2]= 7.0
deltaUmax1[3]=9
#print(Messung_3max[0][1])

Messung_4max = np.array([[13.5,19],[19.5,33],[25.25,43],[31.25,49],[37.5,51],[43.5,53],[49.75,59]]) 
deltaUmax2 = np.zeros(6)
deltaUmax2[0]=6
deltaUmax2[1]=5.75
deltaUmax2[2]=6
deltaUmax2[3]=6.25
deltaUmax2[4]=6
deltaUmax2[5]=6.25

Messung_4= np.array(np.genfromtxt('data/Messung2_191_2C.txt'))





saettigung = np.zeros(4)
saettigung[0]= berechnesaettigung(25.7+273.15)
saettigung[1]= berechnesaettigung(142+273.15)
saettigung[2]= berechnesaettigung(173+273.15)
saettigung[3]= berechnesaettigung(191.2+273.15)
writeW(saettigung, "Sättigung")

mittlerefreieWeglänge = np.zeros(4)
mittlerefreieWeglänge[:] = (0.0029/saettigung[:])

writeW(mittlerefreieWeglänge, "Mittlere freie Weglänge")

a = 1

komischerFaktor =  np.zeros(4)
komischerFaktor[:] = (a/mittlerefreieWeglänge[:])

writeW(komischerFaktor, 'Faktor a/bar{w}')



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

deltaUmax1mean = np.mean(deltaUmax1)
deltaUmax1std = np.std(deltaUmax1)
writeW(deltaUmax1mean, "deltaUmax1mean")
writeW(deltaUmax1std, "deltaUmax1std")
plt.scatter(Messung_3[:,0],Messung_3[:,1],s=6, c='blue',label="173,5°C",marker='x')
plt.scatter(Messung_3max[:,0], Messung_3max[:,1],s=6, c='red',label="Maxima",marker='x')
plt.tick_params(left = False, labelleft = False)
plt.ylabel(r'$I_A$')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_c.pdf')
plt.clf()




plt.scatter(Messung_4[:,0],Messung_4[:,1],s=6, c='blue',label="191,25°C",marker='x')
plt.scatter(Messung_4max[:,0], Messung_4max[:,1],s=6, c='red',label="Maxima",marker='x')
plt.tick_params(left = False, labelleft = False)
plt.ylabel(r'$I_A$')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_d.pdf')
plt.clf()

deltaUmax2mean = np.mean(deltaUmax2)
deltaUmax2std = np.std(deltaUmax2)
writeW(deltaUmax2mean, "deltaUmax2mean")
writeW(deltaUmax2std, "deltaUmax2std")
deltaUmax1ufloat=ufloat(deltaUmax1mean,deltaUmax1std)
deltaUmax2ufloat=ufloat(deltaUmax2mean,deltaUmax2std)
wellenleange1 = (6.62607015*10**(-34))*(3*10**8)/(deltaUmax1ufloat*1.6*10**(-19))
wellenleange2 = (6.62607015*10**(-34))*(3*10**8)/(deltaUmax2ufloat*1.6*10**(-19))
writeW(wellenleange1, "wellenleange1")
writeW(wellenleange2, "wellenleange2")
