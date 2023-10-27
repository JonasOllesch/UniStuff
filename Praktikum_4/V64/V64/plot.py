import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)
from multiprocessing  import Process

def berechne_Brechungsindex_Glas(Maxima):
    a = (632.816*10**-9)*Maxima
    b = 2*(10**-3)*(10*np.pi/180)**2
    return (1-a/b)**(-1)
    #return  ((1 - (632.816*10**-9)*Maxima)/(2*(1*10**3)*(10*np.pi/180)**2))**(-1)

def berechne_Kontrast(Imax,Imin):
    return (Imax-Imin)/(Imax+Imin)
    

def pol_fit(x,a,b,c,d):
    return np.absolute(a*np.sin(b*x + c)) + d

def upol_fit(x,a,b,c,d):
    return np.absolute(a*unp.sin(b*x + c)) + d

def brechne_n_Luft(Counts):
    return 1 + (Counts*632.816*10**-9)/(ufloat(0.1,10**(-5)))

#def uLorentzLorenz(p, T, alpha):
#    c = 1/(constants.epsilon_0*constants.k)
#    return unp.sqrt(1 + p*alpha/(c*T))

def LorentzLorenz_NaecherungT(p,  alpha):
    T = 22.0+constants.zero_Celsius
    c = 2*constants.epsilon_0*constants.k
    return 1 + p*alpha/(c*T)

def LorentzLorenz_Naecherung(p,  alpha, T):
    c = 2*constants.epsilon_0*constants.k
    return 1 + p*alpha/(c*T)



Messreihe_Druck = np.genfromtxt('Messdaten/Druck.txt', encoding='unicode-escape')
Messreihe_Druck[:,0] = Messreihe_Druck[:,0]/1000 #von mbar in bar
Temperatur_Druck = 22.0 + constants.zero_Celsius

Messreihe_Glas  = np.genfromtxt('Messdaten/Glas.txt', encoding='unicode-escape')
Messreihe_Pol   = np.genfromtxt('Messdaten/Pol.txt', encoding='unicode-escape')
Messreihe_Pol[:,0] = Messreihe_Pol[:,0]*(np.pi)/180


Kontraste = np.zeros((len(Messreihe_Pol[:,0]),3))
print(Messreihe_Pol[:,1])
print(Messreihe_Pol[:,4])

for i in range(0,3):
    Kontraste[:,i] = berechne_Kontrast(Messreihe_Pol[:,i+1], Messreihe_Pol[:,i+4])
print(f"Kontraste:{Kontraste}")


Brechungsindex_Glas_arr = berechne_Brechungsindex_Glas(Messreihe_Glas[:,1])
print("Der Brechungsindex von Glas: ",Brechungsindex_Glas_arr)
Brechungsindex_Glas = ufloat(np.mean(Brechungsindex_Glas_arr), np.std(Brechungsindex_Glas_arr))
print(f"Brechungsindex von Glas {Brechungsindex_Glas:.2u}")

print("Winkel der maximalen Kontraste ")
for i in range(0,3):
    print(np.argmax(Kontraste[:,i]))
    print(Messreihe_Pol[np.argmax(Kontraste[:,i]),0]*180/np.pi)

print(Kontraste)
for i in range(0, len(Kontraste[:,0])):
    print(f'avr Kontraste {np.mean(Kontraste[i,:])} \pm {np.std(Kontraste[i,:])}')

#Brechungsindex von Luft
#Brechungsindex_Luft = np.zeros((len(Messreihe_Druck[:,0]),3))

Brechungsindex_Luft_arr = h.uzeros(shape=(len(Messreihe_Druck[:,0]),4))

for i in range(0,4):
    Brechungsindex_Luft_arr[:,i] = brechne_n_Luft(Messreihe_Druck[:,i+1])

para_Luft = h.uzeros((4,1))
popt_Luft = np.zeros(4)
pcov_Luft = np.zeros(4)


for i in range(0,4):
    #print(Brechungsindex_Luft_arr[:,0])
    #print(curve_fit(LorentzLorenz_NaecherungT, xdata = Messreihe_Druck[:,0], ydata = unp.nominal_values(Brechungsindex_Luft_arr[:,i])))
    popt, pcov = curve_fit(LorentzLorenz_NaecherungT, xdata = Messreihe_Druck[:,0], ydata = unp.nominal_values(Brechungsindex_Luft_arr[:,i]))
    para_Luft[i] = correlated_values(popt, pcov)


print("Parameter der Luftnäherung:", para_Luft)
#para_Luft_mean = ufloat(np.mean(unp.nominal_values(para_Luft)),np.std(unp.nominal_values(para_Luft)))
#Brechungsindex_Raum = LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft[0], T = 15+constants.zero_Celsius)

#print(f'Mittelwert der Fitparameter {repr(para_Luft_mean)}')
print(f"Brechungsindices der Standardatmosphäre 1.: {repr(LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft[0], T = 15+constants.zero_Celsius))}")
print(f"Brechungsindices der Standardatmosphäre 2.: {repr(LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft[1], T = 15+constants.zero_Celsius))}")
print(f"Brechungsindices der Standardatmosphäre 3.: {repr(LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft[2], T = 15+constants.zero_Celsius))}")
print(f"Brechungsindices der Standardatmosphäre 4.: {repr(LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft[3], T = 15+constants.zero_Celsius))}")

Brechungsindex_Raum_arr = h.uzeros((4,1))
for i in range(0,4):
    Brechungsindex_Raum_arr[i] = LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft[i], T = 15+constants.zero_Celsius)

print(f'Brechungsindex bei Raumbeingungen {np.mean(unp.nominal_values(Brechungsindex_Raum_arr))} \pm {np.std(unp.nominal_values(Brechungsindex_Raum_arr))}')
Brechungsindex_Raum = ufloat(np.mean(unp.nominal_values(Brechungsindex_Raum_arr)),np.std(unp.nominal_values(Brechungsindex_Raum_arr)))

#Plotten
def plote_n_vs_p(Druck, Brechungsindex_Luft_arr):
    x = np.linspace(0,1,1000)
    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,0]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,0]), fmt ='x', color='navy', label='1. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[0])), color='navy', linestyle = ':')
 
    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,1]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,1]), fmt ='x', color='forestgreen', label='2. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[1])), color='forestgreen', linestyle = ':')
 
    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,2]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,2]), fmt ='x', color='darkorange', label='3. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[2])), color='darkorange', linestyle = ':')
    

    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,3]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,3]), fmt ='x', color='firebrick', label='4. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[3])), color='firebrick', linestyle = ':')


    plt.xlabel(r"$ p \mathbin{/} \unit{\bar} $")
    plt.ylabel(r"$n$")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Brechungsindex.pdf')
    plt.clf()

    return 0

def plote_Kontrast(Winkel, Kontraste):
    x = np.linspace(0,np.pi,20000)
    y = np.absolute(np.sin(2*x))

   
    plt.errorbar(Winkel*180/np.pi,unp.nominal_values(Kontraste[:,0]),yerr=unp.std_devs(Kontraste[:,0]), fmt ='x', color='darkorange', label='1. Data')
    plt.errorbar(Winkel*180/np.pi,unp.nominal_values(Kontraste[:,1]),yerr=unp.std_devs(Kontraste[:,1]), fmt ='x', color='forestgreen', label='2. Data')
    plt.errorbar(Winkel*180/np.pi,unp.nominal_values(Kontraste[:,2]),yerr=unp.std_devs(Kontraste[:,2]), fmt ='x', color='navy', label='3. Data')
    
    
    plt.plot(x*180/np.pi, y, color='firebrick', label = 'Theorie')
    
    plt.legend(loc='upper left')
    plt.xlabel(r"$ \theta \mathbin{/} \unit{\degree} $")
    plt.ylabel(r"$K$")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend(loc = 'upper right')
    plt.savefig('build/Kontrast.pdf')
    plt.clf()
    return 0

#
p1 = Process(target=plote_Kontrast, args=(Messreihe_Pol[:,0], Kontraste))
p2 = Process(target=plote_n_vs_p, args=(Messreihe_Druck[:,0], Brechungsindex_Luft_arr))

p2.start()
p1.start()
p2.join()
p1.join()


#Tabellen erstellen
#print(type(Brechungsindex_Glas_arr[0]))
#list = [Messreihe_Glas[:,1],np.round(Brechungsindex_Glas_arr, decimals = 2)]

Brechungsindex_Glas_arr = ["{:.2f}".format(x) for x in Brechungsindex_Glas_arr]
list = [Messreihe_Glas[:,1].astype(int),Brechungsindex_Glas_arr]
h.save_latex_table_to_file(list, header="Zero passes & Refractive index ", caption="The zero passes and the calculated refractive index of glass", label="glas")
#h.save_latex_table_to_file(np.array(Messreihe_Pol[:,1],Brechungsindex_Glas_arr), header="Zero passes & Refractive index", caption="The zero passes and the calculated refractive index of glass", label="tab:glas")

quartz = 1.457
lime = 1.522
Air = 1.00027654

print(f'Quartz rel deviation{(quartz-Brechungsindex_Glas)/Brechungsindex_Glas*100}')
print(f'Lime rel deviation {(lime-Brechungsindex_Glas)/Brechungsindex_Glas*100}')
print(f'Raum rel deviation {(Brechungsindex_Raum-Air)/Air*100}')