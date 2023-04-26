import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

Channel_Kalibrierung = np.genfromtxt('Messdaten/Channel_Kalibrierung.txt',encoding='unicode-escape')
def pol1(a,b,x):
    return a*x +b 

def exponential(N0,lam,y0,x):
    return (N0*np.exp(-lam*x))+y0


popt_channel, pcov_cannel = curve_fit(pol1,Channel_Kalibrierung[:,0],Channel_Kalibrierung[:,1])
#a_channelu = ufloat(popt_channel[1],np.sqrt(popt_channel[1][1]))
#b_channelu = ufloat(popt_channel[0],np.sqrt(popt_channel[0][0]))


x = np.linspace(0,450,10)
y = pol1(popt_channel[1],popt_channel[0],x)

plt.scatter(Channel_Kalibrierung[:,0],Channel_Kalibrierung[:,1],color = 'blue', marker='x',label='Messdaten')
plt.plot(x,y,label="linearer Fit: (" + str(round(popt_channel[1],2)) + r" \pm \," + str(round(np.sqrt(pcov_cannel[1][1]),2)) + ") "+r" \unit{\second} "+ r"$ \cdot x \,$" + r"$ + ($"+str(round(popt_channel[0],2))+ r" \pm \," +str(round(np.sqrt(pcov_cannel[0][0]),2)) + ")",color="red")
plt.ylabel(r'$\text{Zeit} \, \backslash \, \unit{\micro\second}')
plt.xlabel("Kanalnummer")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Channel_Kalibrierung')
plt.clf()

Messung = np.genfromtxt('Messdaten/v01.Spe',skip_header=12,skip_footer=15,encoding='unicode-escape')        
gefüllte_Kanäle = Messung[4:437]
Channel= np.arange(len(gefüllte_Kanäle))
Zeit = (popt_channel[1]*Channel+popt_channel[0])#*(1e-6)

#print("summe der events " ,np.sum(gefüllte_Kanäle))
#popt_zeit, pcov_zeit = curve_fit(exponential,Channel,gefüllte_Kanäle,p0=[400,0.5,4])

plt.fill_between(Channel,gefüllte_Kanäle,step="pre",alpha=1,label="Histogram der Myonenzerfälle")
plt.ylabel("Zählung")
plt.xlabel("Kanalnummer")
plt.xlim(0,len(gefüllte_Kanäle))
plt.ylim(0,60)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Myonenzerfall_Kanalnummer')
plt.clf()

popt_zeit, pcov_zeit = curve_fit(exponential,Zeit,gefüllte_Kanäle)
x_fit_zeit=np.linspace(np.min(Zeit),np.max(Zeit),10000)
y_fit_zeit=exponential(*popt_zeit,x_fit_zeit)
plt.plot(x_fit_zeit,y_fit_zeit,color='red')
print(popt_zeit)
print(pcov_zeit)
plt.fill_between(Zeit,gefüllte_Kanäle,step="pre",alpha=1,label="Histogram der Myonenzerfälle")
plt.ylabel("Zählung")
plt.xlabel("Zeit")
plt.xlim(0,10)
plt.ylim(0,60)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Myonenzerfall_Zeit')
plt.clf()