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
plt.plot(x,y,label="linearer Fit: (" + str(round(popt[1],2)) + r" \pm \," + str(round(np.sqrt(pcov[1][1]),2)) + ") "+r" \unit{\second} "+ r"$ \cdot x \,$" + r"$ + ($"+str(round(popt[0],2))+ r" \pm \," +str(round(np.sqrt(pcov[0][0]),2)) + ")",color="red")

plt.ylabel(r'$\text{Zeit} \, \backslash \, \unit{\micro\second}')
plt.xlabel("Kanalnummer")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Channel_Kalibrierung')
plt.clf()

Messung = np.genfromtxt('Messdaten/v01.Spe',skip_header=12,skip_footer=15,encoding='unicode-escape')        
Channel= np.arange(len(Messung))
#plt.bar(Channel,Messung,width=1.0,fill=True,label="Histogram der Myonenzerfälle")
#plt.step(Channel,Messung,fill=True,step='pre',label="Histogram der Myonenzerfälle")
gefüllte_Kanäle = Messung[4:431]
#plt.fill_between(Channel[4:431],Messung[4:431],step="pre",alpha=1,label="Histogram der Myonenzerfälle")
#plt.ylabel("Zählung")
#plt.xlabel("Kanalnummer")
#plt.xlim(4,450)
#plt.ylim(0,60)
#plt.grid(linestyle = ":")
#plt.tight_layout()
#plt.legend()
#plt.savefig('build/Myonenzerfall')
#plt.clf()
