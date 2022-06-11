import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes


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

def pol1(a,b,x):
    return a* x +b

# Plot mit Photostrom gegen Bremsspannung sollte 'photostrom.pdf' heißen, wenn du da noch einen besseren Namen findest (nicht 'graph1.pdf' ;)), kannst du das ja sonst auch in der Auswertung und hier anders nennen.
# Plot der Gegenspannungen gegen die Frequenzen bitte in 'gegspanngegfreq.pdf' :)
# Plot für gelbe Kennlinie bitte unter 'kennliniegelb.pdf' :D
Messung_a = np.array(np.genfromtxt('Messung1.txt'))
Messung_a[:,1]= Messung_a[:,1]*10**(-9)

Messung_b = np.array(np.genfromtxt('Messung2.txt'))
Messung_b[:,1]= Messung_b[:,1]*10**(-9)

Messung_c = np.array(np.genfromtxt('Messung3.txt'))
Messung_c[:,1]= Messung_c[:,1]*10**(-9)

Messung_d = np.array(np.genfromtxt('Messung4.txt'))
Messung_d[:,1]= Messung_d[:,1]*10**(-9)

Messung_e = np.array(np.genfromtxt('Messung5.txt'))
Messung_e[:,1]= Messung_e[:,1]*10**(-9)
#
x = np.linspace(-0.05,1.45,200)
plt.scatter(Messung_a[:,0],np.sqrt(Messung_a[:,1]),      s=8,c='red',label='rotes Licht')
popt_a, pcov_a = curve_fit(pol1, Messung_a[:,0], np.sqrt(Messung_a[:,1]))
y_a = pol1(popt_a[1], popt_a[0], x)
plt.plot(x,y_a,color='red')

plt.scatter(Messung_b[:,0],np.sqrt(Messung_b[:,1]),      s=8,c='green',label='grünes Licht')
popt_b, pcov_b = curve_fit(pol1, Messung_b[:,0], np.sqrt(Messung_b[:,1]))
y_b = pol1(popt_b[1], popt_b[0], x)
plt.plot(x,y_b,color='green')

plt.scatter(Messung_c[:,0],np.sqrt(Messung_c[:,1]),      s=8,c='blue',label='blaues Licht')
popt_c, pcov_c = curve_fit(pol1,Messung_c[:,0], np.sqrt(Messung_c[:,1]))
y_c = pol1(popt_c[1], popt_c[0], x)
plt.plot(x,y_c,color='blue')

plt.scatter(Messung_d[:,0],np.sqrt(Messung_d[:,1]),      s=8,c='#8B008B',label='violettes Licht')
popt_d, pcov_d = curve_fit(pol1,Messung_d[:,0], np.sqrt(Messung_d[:,1]))
y_d = pol1(popt_d[1], popt_d[0], x)
plt.plot(x,y_d,color='#8B008B')


plt.scatter(-1*Messung_e[:7,0],np.sqrt(Messung_e[:7,1]), s=8,c='#FFA500',label='oranges Licht')
popt_e, pcov_e = curve_fit(pol1,-1*Messung_e[:7,0], np.sqrt(Messung_e[:7,1]))
y_e = pol1(popt_e[1], popt_e[0], x)
plt.plot(x,y_e,color='#FFA500')

plt.xlabel(r'$ U \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$ \sqrt{I \mathbin{/} \unit{\ampere}}$')
plt.xlim(-0.05,1.45)
plt.ylim(-0.05*10**-5,4*10**-5)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/photostrom.pdf')
plt.clf()


writeW(popt_a, "Koeffizenten b , a rot")
writeW(popt_b, "Koeffizenten b , a grün")
writeW(popt_c, "Koeffizenten b , a blau")
writeW(popt_d, "Koeffizenten b , a violett")
writeW(popt_e, "Koeffizenten b , a orange")

writeW(-1*popt_a[0]/popt_a[1], "Nullstell von rot")
writeW(-1*popt_b[0]/popt_b[1], "Nullstell von grün")
writeW(-1*popt_c[0]/popt_c[1], "Nullstell von blau")
writeW(-1*popt_d[0]/popt_d[1], "Nullstell von violett")
writeW(-1*popt_e[0]/popt_e[1], "Nullstell von orange")

spektralline = np.array([623, 546 ,435 ,365, 587])
spektralline = spektralline*10**(-9)
gegenspannungen = np.array([-1*popt_a[0]/popt_a[1],-1*popt_b[0]/popt_b[1],-1*popt_c[0]/popt_c[1],-1*popt_d[0]/popt_d[1],-1*popt_e[0]/popt_e[1]])


plt.scatter((3*10**8)/spektralline[:],gegenspannungen[:],c='blue',s=8,label = "Messdaten")
x_g = np.linspace(4*10**14,9.5*10**14)
popt_g,pcov_g = curve_fit(pol1, (3*10**8)/spektralline[:], gegenspannungen[:])
y_g = pol1(popt_g[1], popt_g[0], x_g)
plt.xlim(4*10**14,9.5*10**14)
plt.plot(x_g,y_g,label="lineare Regression",color='red')
plt.ylabel(r'$ U_{g} \mathbin{/} \unit{\volt}$')
plt.xlabel(r'$ v \mathbin{/} \unit{\hertz}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/gegspanngegfreq.pdf')
plt.clf()

writeW(popt_g, "Koeffizenten b , a Gegenfeld")
writeW(np.sqrt(pcov_g), "Unsicherheit der Gegenfeldregression")

plt.scatter(Messung_e[:,0],Messung_e[:,1],c='blue',s=6,label = "Messdaten")
plt.ylabel(r'$ I \mathbin{/} \unit{\nano\ampere}$')
plt.xlabel(r'$ U \mathbin{/} \unit{\volt}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/kennliniegelb.pdf')
plt.clf()