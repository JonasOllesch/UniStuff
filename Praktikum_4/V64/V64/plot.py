import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp


from multiprocessing  import Process


def berechne_Kontrast(Imax,Imin):
    return (Imax-Imin)/(Imax+Imin)
    

def pol_fit(x,a,b,c,d):
    return np.absolute(a*np.sin(b*x + c)) + d

Messreihe_Druck = np.genfromtxt('Messdaten/Druck.txt', encoding='unicode-escape')
Messreihe_Druck[:,0] = Messreihe_Druck[:,0]/1000 #von mbar in bar
#print(Messreihe_Druck[0,:])

Messreihe_Glas  = np.genfromtxt('Messdaten/Glas.txt', encoding='unicode-escape')
Messreihe_Pol   = np.genfromtxt('Messdaten/Pol.txt', encoding='unicode-escape')
Messreihe_Pol[:,0] = Messreihe_Pol[:,0]*(2*np.pi)/180


Kontraste = np.zeros((len(Messreihe_Pol[:,0]),3))
for i in range(0,3):
    Kontraste[:,i] = berechne_Kontrast(Messreihe_Pol[:,i+1],Messreihe_Pol[:,i+3])


#Pol_max_mean = np.zeros_like(Messreihe_Pol[:,0])
#Pol_max_std = np.zeros_like(Messreihe_Pol[:,0])
#
#Pol_min_mean = np.zeros_like(Messreihe_Pol[:,0])
#Pol_min_std = np.zeros_like(Messreihe_Pol[:,0])
#
#for i in range(0,len(Messreihe_Pol[:,0])):
#    Pol_max_mean[i] = np.mean(Messreihe_Pol[i,1:4])
#    Pol_max_std[i] = np.std(Messreihe_Pol[i,1:4])
#
#    Pol_min_mean[i] = np.mean(Messreihe_Pol[i,4:7])
#    Pol_min_std[i] = np.std(Messreihe_Pol[i,4:7])
#
#Pol_max = unp.uarray(Pol_max_mean,Pol_max_std)
#Pol_min = unp.uarray(Pol_min_mean,Pol_min_std)
#
#Kontrast = berechne_Kontrast(Pol_max,Pol_min)
#
#
#popt, pcov = curve_fit(pol_fit, xdata=Messreihe_Pol[:,0],ydata=unp.nominal_values(Kontrast),sigma =unp.std_devs(Kontrast), absolute_sigma=True)
#para_pol = correlated_value(popt, pcov)
#s(popt,pcov)



def plote_Kontrast(Winkel, Kontraste):
    x = np.linspace(0,2*np*pi,20000)
    y = np.absolute(np.sin(x))


    plt.errorbar(Winkel,unp.nominal_values(Kontraste[:,0]),yerr=unp.std_devs(Kontraste[:,0]), fmt ='x', color='darkorange', label='1. Data')
    plt.errorbar(Winkel,unp.nominal_values(Kontraste[:,1]),yerr=unp.std_devs(Kontraste[:,1]), fmt ='x', color='forestgreen', label='1. Data')
    plt.errorbar(Winkel,unp.nominal_values(Kontraste[:,2]),yerr=unp.std_devs(Kontraste[:,2]), fmt ='x', color='navy', label='1. Data')
    
    plt.plot(x, y, color='navy', label = 'Theorie')
    #plt.plot(x,pol_fit(x,*para),color='forestgreen',label='Fit')
    plt.xlabel(r"$ \theta \mathbin{/} \unit{\degree} $")
    plt.ylabel(r"$K$")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Kontrast.pdf')
    plt.clf()



p = Process(target=plote_Kontrast, args=(Messreihe_Pol[:,0], Kontraste))
p.start()
p.join()