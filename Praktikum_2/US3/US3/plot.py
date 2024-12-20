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
Messung_2_a[:,0] = Messung_2_a[:,0]*10**(-6)    
Messung_2_b = np.array(np.genfromtxt('Messung_2_b.txt'))
Messung_2_b[:,0] = Messung_2_b[:,0]*10**(-6)


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
plt.xlabel(r'$ v \mathbin{/} \unit{\dfrac{\meter}{\second}}  $')
plt.ylabel(r'$ \frac{\Delta \nu}{\cos\alpha}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph1a.pdf')
plt.clf()


plt.scatter(Stroemungsgeschwindigkeit[:,1],delta_v[:,1]/np.cos(30*np.pi/180),s=8,c='b',label="Messdaten")
plt.xlabel(r'$ v \mathbin{/} \unit{\dfrac{\meter}{\second}}  $')
plt.ylabel(r'$ \frac{\Delta \nu}{\cos\alpha}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph1b.pdf')
plt.clf()


plt.scatter(Stroemungsgeschwindigkeit[:,2],delta_v[:,2]/np.cos(45*np.pi/180),s=8,c='b',label="Messdaten")
plt.xlabel(r'$ v \mathbin{/} \unit{\dfrac{\meter}{\second}}  $')
plt.ylabel(r'$ \frac{\Delta \nu}{\cos\alpha}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph1c.pdf')
plt.clf()


#Messreihe B
l = 30.7*10**(-3)
Messtiefe_ainm = np.zeros(16)
Messtiefe_ainm[:] = l+ 1800*(Messung_2_a[:,0]-l/2700)
Messtiefe_binm = np.zeros(16)
Messtiefe_binm[:] = l+ 1800*(Messung_2_b[:,0]-l/2700)
writeW(Messtiefe_ainm, "Messtiefe_ainm")
writeW(Messtiefe_binm, "Messtiefe_binm")
del l


plt.scatter(Messung_2_a[:,0]*10**3,Messung_2_a[:,2],s=8,c='b',label="Streuintensität")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$I \mathbin{/} 1000 \cdot \unit{\dfrac{\volt^2}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2a1.pdf')
plt.clf()

plt.scatter(Messung_2_a[:,0]*10**3,Messung_2_a[:,1],s=8,c='b',label="Momentangeschwindigkeit")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$I \mathbin{/} \unit{\dfrac{\meter}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2a2.pdf')
plt.clf()



plt.scatter(Messung_2_b[:,0]*10**3,Messung_2_b[:,2],s=8,c='b',label="Streuintensität")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$v \mathbin{/} 1000 \cdot \unit{\dfrac{\volt^2}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2b1.pdf')
plt.clf()

plt.scatter(Messung_2_b[1:,0]*10**3,Messung_2_b[1:,1],s=8,c='b',label="Momentangeschwindigkeit")
plt.xlabel(r'Messtiefe $\mathbin{/} \unit{\milli\meter}$')
plt.ylabel(r'$v \mathbin{/} \unit{\dfrac{\meter}{\second}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph2b2.pdf')
plt.clf()
#writeW(Messung_2_a[:,0]*10**3, "Messung_2_a Messtiefe in mm")
#writeW(Messung_2_a[:,1], "Messung_2_a Momentangeschwindigkeit in m/s")
#writeW(Messtiefe_ainm, "Messtiefe_a in s")
#writeW(Messung_2_a[:,2], "Messung_2_a Intensität in V**2/s")

writeW(Messung_2_b[:,0]*10**3, "Messung_2_b Messtiefe in mm")
writeW(Messung_2_b[:,1], "Messung_2_b Momentangeschwindigkeit in m/s")
writeW(Messtiefe_binm, "Messtiefe_b in s")
writeW(Messung_2_b[:,2], "Messung_2_b Intensität in V**2/s")

#### Test Test

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(r'Messtiefe $\mathbin{/} \unit{\second}$')
ax1.set_ylabel(r'$I \mathbin{/} 1000 \cdot \unit{\dfrac{\volt^2}{\second}}$')
ax1.scatter(Messung_2_a[:,0],Messung_2_a[:,2],s=8,c=color,label="Streuintensität")
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc='best')
plt.grid()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$v \mathbin{/} \unit{\dfrac{\meter}{\second}}$')
ax2.scatter(Messung_2_a[:,0],Messung_2_a[:,1],s=8,c=color,label="Momentangeschwindigkeit")
ax2.tick_params(axis='y', labelcolor=color)
plt.legend(loc='best')

fig.tight_layout() 
plt.savefig('build/Graph2a.pdf')


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(r'Messtiefe $\mathbin{/} \unit{\second}$')
ax1.set_ylabel(r'$I \mathbin{/} 1000 \cdot \unit{\dfrac{\volt^2}{\second}}$')
ax1.scatter(Messung_2_b[:,0],Messung_2_b[:,2],s=8,c=color,label="Streuintensität")
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc='best')
plt.grid()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$v \mathbin{/} \unit{\dfrac{\meter}{\second}}$')
ax2.scatter(Messung_2_b[1:,0],Messung_2_b[1:,1],s=8,c=color,label="Momentangeschwindigkeit")
ax2.tick_params(axis='y', labelcolor=color)
plt.legend(loc='best')

fig.tight_layout() 
plt.savefig('build/Graph2b.pdf')
