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

def lorentzkurve(a,v_0,g_a,x):
    return a/((x**2-v_0)**2+(g_a)*(v_0))



Messung_1 = np.array(np.genfromtxt('Messung_1.txt'))
Messung_1[:,0] = Messung_1[:,0]*1000

#x = np.linspace(10,40,1000)
#y  =lorentzkurve(350000, 22**2, 10**2, x)
#popt, pcov = curve_fit(lorentzkurve, Messung_1[:,0]/1000, Messung_1[:,1]) der curve fit funktioniert so ungefähr gar weil, die Parameter nicht konvergieren
#y = lorentzkurve(popt[0], popt[1], popt[2], x)
#writeW(popt, "popt")
#writeW(popt, "Parameter der lorentzkurve")
#plt.plot(x,y,color='r',label="regression")

plt.scatter(Messung_1[:,0]/1000, Messung_1[:,1],s=8,label="Messdaten")
plt.xlabel(r'$ f \, \mathbin{/} \unit{\hertz}$')
plt.ylabel(r'$ U \mathbin{/} \unit{\volt}  $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_a.pdf')
plt.clf()

Q_real_Dy = 0.01438/(0.155*7810)
Q_real_Nd = 0.0766/(0.16*7240)
Q_real_Gd = 0.1020/(0.159*7410)
writeW(Q_real_Dy, "Q_real_Dy2O3")
writeW(Q_real_Nd, "Q_real_Nd2O3")
writeW(Q_real_Gd, "Q_real_Gd2O3")