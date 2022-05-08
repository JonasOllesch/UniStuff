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

def Stroe_formle(delta_frequenz,Winkel):
    return  delta_frequenz*1800/(2*2*(10**6)*np.cos(Winkel*np.pi/180))

#c = 1800
#v_0 = 2*10**6

Messung_1_a = np.array(np.genfromtxt('Messung_1_a.txt'))
Messung_1_b = np.array(np.genfromtxt('Messung_1_b.txt'))
Messung_1_c = np.array(np.genfromtxt('Messung_1_c.txt'))

Messung_2_a = np.array(np.genfromtxt('Messung_2_a.txt'))
Messung_2_a[:,0] = Messung_2_a[:,0]*10**(-3)    
Messung_2_b = np.array(np.genfromtxt('Messung_2_b.txt'))
Messung_2_b[:,0] = Messung_2_b[:,0]*10**(-3)


#Messreihe  A

Pumpleistunginpro = np.zeros(5)
Pumpleistunginpro[:] = (Messung_1_a[:,0]/8600 )*100

delta_v = np.zeros((5,3)) # 15 30 45
delta_v[:,0]= Messung_1_a[:,1]-Messung_1_a[:,2]
delta_v[:,1]= Messung_1_b[:,1]-Messung_1_b[:,2]
delta_v[:,2]= Messung_1_c[:,1]-Messung_1_c[:,2]



Stroemungsgeschwindigkeit = np.zeros((5,3))
Stroemungsgeschwindigkeit[:,0] = Stroe_formle(delta_v[:,0], 15)
Stroemungsgeschwindigkeit[:,1] = Stroe_formle(delta_v[:,1], 30)
Stroemungsgeschwindigkeit[:,2] = Stroe_formle(delta_v[:,2], 45)


#print(delta_v)
#print(Stroemungsgeschwindigkeit)
#print(Pumpleistunginpro)

#writeW(delta_v, "delta_v")
#writeW(Stroemungsgeschwindigkeit, "Stroemungsgeschwindigkeit")
#writeW(Pumpleistunginpro, "Pumpleistunginpro")



plt.scatter(Stroemungsgeschwindigkeit[:,0],delta_v[:,0]/np.cos(15*np.pi/180),s=8,c='b',label="Messdaten")
plt.xlabel(r'$ \frac{\Delta \nu}{\cos\alpha}$')
plt.ylabel(r'$ v \mathbin{/} \unit{\dfrac{\meter}{\second}}  $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph1a.pdf')
plt.clf()


plt.scatter(Stroemungsgeschwindigkeit[:,1],delta_v[:,1]/np.cos(30*np.pi/180),s=8,c='b',label="Messdaten")
plt.xlabel(r'$ \frac{\Delta \nu}{\cos\alpha}$')
plt.ylabel(r'$ v \mathbin{/} \unit{\dfrac{\meter}{\second}}  $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph1b.pdf')
plt.clf()

plt.scatter(Stroemungsgeschwindigkeit[:,2],delta_v[:,2]/np.cos(45*np.pi/180),s=8,c='b',label="Messdaten")
plt.xlabel(r'$ \frac{\Delta \nu}{\cos\alpha}$')
plt.ylabel(r'$ v \mathbin{/} \unit{\dfrac{\meter}{\second}}  $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph1c.pdf')
plt.clf()


#Messreihe B

Messtiefe_ains = np.zeros(16)
Messtiefe_ains[:] = Messung_2_a[:,0]/1800
Messtiefe_bins = np.zeros(16)
Messtiefe_bins[:] = Messung_2_b[:,0]/1800


plt.scatter(Messung_2_a[:,0],Messung_2_a[:,2],s=8,c='b',label="Streuintensität")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$I \mathbin{/} 1000 \cdot \unit{\dfrac{\volt^2}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2a1.pdf')
plt.clf()

plt.scatter(Messung_2_a[:,0],Messung_2_a[:,1],s=8,c='b',label="Momentangeschwindigkeit")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$I \mathbin{/} \unit{\dfrac{\meter}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2a2.pdf')
plt.clf()



plt.scatter(Messung_2_b[:,0],Messung_2_b[:,2],s=8,c='b',label="Streuintensität")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$I \mathbin{/} 1000 \cdot \unit{\dfrac{\volt^2}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2b1.pdf')
plt.clf()

plt.scatter(Messung_2_b[1:,0],Messung_2_b[1:,1],s=8,c='b',label="Momentangeschwindigkeit")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$I \mathbin{/} \unit{\dfrac{\meter}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2b2.pdf')
plt.clf()
#writeW(Messung_2_a[:,0]*10**3, "Messung_2_a Messtiefe in mm")
#writeW(Messung_2_a[:,1], "Messung_2_a Momentangeschwindigkeit in m/s")
#writeW(Messtiefe_ains, "Messtiefe_a in s")
#writeW(Messung_2_a[:,2], "Messung_2_a Intensität in V**2/s")

writeW(Messung_2_b[:,0]*10**3, "Messung_2_b Messtiefe in mm")
writeW(Messung_2_b[:,1], "Messung_2_b Momentangeschwindigkeit in m/s")
writeW(Messtiefe_bins, "Messtiefe_b in s")
writeW(Messung_2_b[:,2], "Messung_2_b Intensität in V**2/s")

