import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

#from iminuit import cost
#from iminuit import Minuit

def pdf(x,N,lam,y):
    return (N*np.exp(-lam*x))+y

Channel_Kalibrierung = np.genfromtxt('Messdaten/Channel_Kalibrierung.txt',encoding='unicode-escape')
def pol1(a,b,x):
    return a*x +b 

def exponential(x,N,lam,y):
    return (N*np.exp(-lam*x))+y


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
gefüllte_Kanäle = Messung[:]
Channel= np.arange(len(gefüllte_Kanäle))
Zeit = (popt_channel[1]*Channel+popt_channel[0])#*(1e-6)

#print("summe der events " ,np.sum(gefüllte_Kanäle))
popt_zeit, pcov_zeit = curve_fit(exponential,Channel,gefüllte_Kanäle,p0=[39,0.014566,0.9])
print(popt_zeit)
#print(popt_zeit)
#print(Channel)
xmin = np.append(Channel,gefüllte_Kanäle)
#c = cost.UnbinnedNLL(xmin,pdf)
#c = cost.LeastSquares(Channel, gefüllte_Kanäle,0.1,pdf)
#m = Minuit(c, N=39, lam=0.014566, y=0.9,limit_N=(20,100),limit_lam=(0.0001,0.05),limit_y=(0.5,2))
#m.migrad()
#print(m.migrad())
#print(m.values)
x_test = np.linspace(0,400,1000)
y_test = pdf(x_test,36.24,0.01256,1.217)

plt.plot(x_test,y_test,color='red')
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

#popt_zeit, pcov_zeit = curve_fit(exponential,Zeit,gefüllte_Kanäle)
#x_fit_zeit=np.linspace(np.min(Zeit),np.max(Zeit),10000)
#y_fit_zeit=exponential(*popt_zeit,x_fit_zeit)






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