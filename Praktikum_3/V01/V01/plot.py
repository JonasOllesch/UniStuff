import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


Channel_Kalibrierung = np.genfromtxt('Messdaten/Channel_Kalibrierung.txt',encoding='unicode-escape')





def pol1(a,b,x):
    return a*x +b 

popt, pcov = curve_fit(pol1,Channel_Kalibrierung[:,0],Channel_Kalibrierung[:,1])
x = np.linspace(0,450,10)
y = pol1(popt[1],popt[0],x)


plt.scatter(Channel_Kalibrierung[:,0],Channel_Kalibrierung[:,1],color = 'blue', marker='x',label='Messdaten')
plt.plot(x,y,label="linearer Fit: (" + str(round(popt[1],2)) + r" \pm \," + str(round(np.sqrt(pcov[1][1]))) + ") "+r" \unit{\second} "+ r"$ \cdot x \,$" + r"$ + ($"  +str(round(popt[0],2))+ r" \pm \," +str(round(np.sqrt(pcov[0][0]))) + ")"          ,color="red")


plt.ylabel(r'$\text{Zeit} \, \backslash \, \unit{\micro\second}')
plt.xlabel("Kanalnummer")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Channel_Kalibrierung')
plt.clf()