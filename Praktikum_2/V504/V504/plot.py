import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

# Die Kennlinien bitte in 'kennlinien.pdf' speichern, sonst muss der Code in auswertung.tex noch geändert werden '^^
# Den Fit (Aufgabenteil b)) für Messreihe 5 als 'linregraumladung.pdf' speichern (oder halt in der Auswertung ändern :D)
# Der Fit zum Anlaufstromgebiet sollte irgendwie 'reganlaufstrom.pdf' heißen :D

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

def pol1(m,b,x):
    return m*x+b

Messung1= np.array(np.genfromtxt('Messdaten/Messung_1.txt'))
Messung1[:,1]= Messung1[:,1]*10**(-3)
Messung2= np.array(np.genfromtxt('Messdaten/Messung_2.txt'))
Messung2[:,1]= Messung2[:,1]*10**(-3)
Messung3= np.array(np.genfromtxt('Messdaten/Messung_3.txt'))
Messung3[:,1]= Messung3[:,1]*10**(-3)
Messung4= np.array(np.genfromtxt('Messdaten/Messung_4.txt'))
Messung4[:,1]= Messung4[:,1]*10**(-3)
Messung5= np.array(np.genfromtxt('Messdaten/Messung_5.txt'))
Messung5[:,1]= Messung5[:,1]*10**(-3)

Messung6= np.array(np.genfromtxt('Messdaten/Messung_6.txt'))
Messung6[:,1]= Messung6[:,1]*10**(-9)


#plt.scatter(Messung1[:,0], Messung1[:,1]*1000,color='blue',s=8,label="Messdaten")
#plt.xlabel(r'$U \, \mathbin{/} \unit{\volt}$')
#plt.ylabel(r'$ I \mathbin{/} \unit{\milli\ampere}  $')
#plt.tight_layout()
#plt.legend()
#plt.grid()
#plt.savefig('build/Graph_a.pdf')
#plt.clf()
#
#plt.scatter(Messung2[:,0], Messung2[:,1]*1000,color='blue',s=8,label="Messdaten")
#plt.xlabel(r'$U \, \mathbin{/} \unit{\volt}$')
#plt.ylabel(r'$ I \mathbin{/} \unit{\milli\ampere}  $')
#plt.tight_layout()
#plt.legend()
#plt.grid()
#plt.savefig('build/Graph_b.pdf')
#plt.clf()
#
#plt.scatter(Messung3[:,0], Messung3[:,1]*1000,color='blue',s=8,label="Messdaten")
#plt.xlabel(r'$U \, \mathbin{/} \unit{\volt}$')
#plt.ylabel(r'$ I \mathbin{/} \unit{\milli\ampere}  $')
#plt.tight_layout()
#plt.legend()
#plt.grid()
#plt.savefig('build/Graph_c.pdf')
#plt.clf()
#
#plt.scatter(Messung4[:,0], Messung4[:,1]*1000,color='blue',s=8,label="Messdaten")
#plt.xlabel(r'$U \, \mathbin{/} \unit{\volt}$')
#plt.ylabel(r'$ I \mathbin{/} \unit{\milli\ampere}  $')
#plt.tight_layout()
#plt.legend()
#plt.grid()
#plt.savefig('build/Graph_d.pdf')
#plt.clf()
#
#plt.scatter(Messung5[:,0], Messung5[:,1]*1000,color='blue',s=8,label="Messdaten")
#plt.xlabel(r'$U \, \mathbin{/} \unit{\volt}$')
#plt.ylabel(r'$ I \mathbin{/} \unit{\milli\ampere}  $')
#plt.tight_layout()
#plt.legend()
#plt.grid()
#plt.savefig('build/Graph_e.pdf')
#plt.clf()



plt.scatter(Messung1[:,0], Messung1[:,1]*1000,s=4,label=r"$U_{H} = 3,2 \,\unit{\volt} $")
plt.scatter(Messung2[:,0], Messung2[:,1]*1000,s=4,label=r"$U_{H} = 3,5 \,\unit{\volt} $")
plt.scatter(Messung3[:,0], Messung3[:,1]*1000,s=4,label=r"$U_{H} = 4,0 \,\unit{\volt} $")
plt.scatter(Messung4[:,0], Messung4[:,1]*1000,s=4,label=r"$U_{H} = 4,2 \,\unit{\volt} $")
plt.scatter(Messung5[:,0], Messung5[:,1]*1000,s=4,label=r"$U_{H} = 5,0 \,\unit{\volt} $")
plt.xlabel(r'$U \, \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$ I \mathbin{/} \unit{\milli\ampere}  $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/kennlinien.pdf')
plt.clf()

plt.scatter(np.log(Messung5[:,0]), np.log(Messung5[:,1]),s=8,label="Messdaten")
plt.xlabel(r'$ \log{(U)}$')
plt.ylabel(r'$ \log{(I)}  $')
popt_b,pcov_b = curve_fit(pol1, np.log(Messung5[:,0]), np.log(Messung5[:,1]))
x = np.linspace(1,6,1000)
y = pol1(popt_b[1], popt_b[0], x)
plt.plot(x,y,color='red', label='lineare Regression')
plt.xlim(1,6)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/linregraumladung.pdf')
plt.clf()


writeW(popt_b, "popt von Messung 5")
writeW(np.sqrt(pcov_b), "sqrt(pcov) von Messung 5")


Ukorrigiert = np.copy(Messung6[:,0])
Ukorrigiert[:]= Messung6[:,0]+(1*10**6)*Messung6[:,1]
writeW(np.round(Ukorrigiert,5), "Ukorrigiert")


plt.scatter(Ukorrigiert[:], np.log(Messung6[:,1]),s=8,label="Messdaten")
popt_c, pcov_c = curve_fit(pol1, Ukorrigiert[:], np.log(Messung6[:,1]))
plt.xlabel(r'$ U \mathbin{/} \unit{\volt} $')
plt.ylabel(r'$ \log{(I)}  $')


x = np.linspace(0,1,1000)
y = pol1(popt_c[1], popt_c[0], x)
plt.plot(x,y,color='red', label='lineare Regression')
plt.xlim(0,1)

plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/reganlaufstrom.pdf')
plt.clf()

writeW(popt_c, "popt von Messung 5")
writeW(np.sqrt(pcov_c), "sqrt(pcov) von Messung 5")


tmp = ufloat(popt_c[1],np.sqrt(pcov_c[1][1]))
writeW(-(1.602176*10**-19)/((1.380649*10**-23)*tmp), "Kathodentemperatur")

Heitzstrom = np.array( [[3.2,1.95],[3.5,2], [4,2.1], [4,2.2],[5,2.4]] )
print(Heitzstrom[:,0])
print(Heitzstrom[:,1])
Temperatur = np.zeros(5)
Temperatur[:] =((Heitzstrom[:,0]*Heitzstrom[:,1]-0.9)/(0.32*5.7*10**(-12)*0.28))**(1/4) 
writeW(Temperatur, "Temperatur")

saettigungstrom = np.array([0.074,0.117,0.256,0.546,1.996])
tmp1 = np.zeros(5)
tmp2 = np.zeros(5)
tmp3 = np.zeros(5)
ephi = np.zeros(5)
tmp1[:] = -1.380649*10**(-23)*Temperatur[:]
print(tmp1)
tmp2[:] = saettigungstrom[:]*(6.62607015*10**(-34))*(6.62607015*10**(-34))*(6.62607015*10**(-34))
tmp3[:] = (4*np.pi*(1.602176*10**(-19))*((1.380649*10**(-23))**2)*(9.1093837015*10**(-31))*(0.32*10**(-4))*(Temperatur[:])**2)
ephi[:] = tmp1[:]*np.log(tmp2[:]/tmp3[:])/((1.602176*10**(-19)))
writeW(ephi, "Austrittsarbeit")

writeW(np.mean(Temperatur), "np.mean(Temperatur)")
writeW(np.std(Temperatur), "np.std(Temperatur)")

writeW(np.mean(ephi), "np.mean(ephi)")
writeW(np.std(ephi), "np.std(ephi)")
