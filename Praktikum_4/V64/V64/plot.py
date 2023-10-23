import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants


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
    T = 22.0*constants.zero_Celsius
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
Messreihe_Pol[:,0] = Messreihe_Pol[:,0]*(2*np.pi)/180


Kontraste = np.zeros((len(Messreihe_Pol[:,0]),3))
print(Messreihe_Pol[:,1])
print(Messreihe_Pol[:,4])

for i in range(0,3):
    Kontraste[:,i] = berechne_Kontrast(Messreihe_Pol[:,i+1], Messreihe_Pol[:,i+4])
print(Kontraste)


Brechungsindex_Glas_arr = berechne_Brechungsindex_Glas(Messreihe_Glas[:,1])
print("Der Brechungsindex von Glas: ",Brechungsindex_Glas_arr)
Brechungsindex_Glas = ufloat(np.mean(Brechungsindex_Glas_arr), np.std(Brechungsindex_Glas_arr))
print(f"Brechungsindex von Glas {Brechungsindex_Glas:.2u}")

#Brechungsindex von Luft
#Brechungsindex_Luft = np.zeros((len(Messreihe_Druck[:,0]),3))

Brechungsindex_Luft_arr = h.uzeros(shape=(len(Messreihe_Druck[:,0]),4))

for i in range(0,4):
    Brechungsindex_Luft_arr[:,i] = brechne_n_Luft(Messreihe_Druck[:,i+1])

para_Luft = h.uzeros((3,1))
popt_Luft = np.zeros(3)
pcov_Luft = np.zeros(3)


for i in range(0,3):
    #print(Brechungsindex_Luft_arr[:,0])
    #print(curve_fit(LorentzLorenz_NaecherungT, xdata = Messreihe_Druck[:,0], ydata = unp.nominal_values(Brechungsindex_Luft_arr[:,i])))
    popt, pcov = curve_fit(LorentzLorenz_NaecherungT, xdata = Messreihe_Druck[:,0], ydata = unp.nominal_values(Brechungsindex_Luft_arr[:,i]))
    para_Luft[i] = correlated_values(popt, pcov)


print("Parameter der Luftnäherung:", para_Luft)
Brechungsindex_Raum = LorentzLorenz_Naecherung(p=1.013, alpha=para_Luft, T = 15+constants.zero_Celsius)
print(f"Brechungsindices der Standardatmosphäre: {Brechungsindex_Raum}")
print(f"Brechungsindex mean der Standardatmosphäre: {np.mean(unp.nominal_values(Brechungsindex_Raum))} \pm {np.std(unp.nominal_values(Brechungsindex_Raum))}")


#Plotten
def plote_n_vs_p(Druck, Brechungsindex_Luft_arr):
    x = np.linspace(0,1,1000)
    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,0]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,0]), fmt ='x', color='navy', label='1. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[0])), color='navy', linestyle = ':')
 
    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,1]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,1]), fmt ='x', color='forestgreen', label='2. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[1])), color='forestgreen', linestyle = ':')
 
    plt.errorbar(Druck,unp.nominal_values(Brechungsindex_Luft_arr[:,2]), yerr=unp.std_devs(Brechungsindex_Luft_arr[:,2]), fmt ='x', color='darkorange', label='3. Series')
    plt.plot(x, unp.nominal_values(LorentzLorenz_NaecherungT(x, para_Luft[2])), color='darkorange', linestyle = ':')
    

    plt.xlabel(r"$ p \mathbin{/} \unit{\bar} $")
    plt.ylabel(r"$n$")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Brechungsindex.pdf')
    plt.clf()

    return 0

def plote_Kontrast(Winkel, Kontrast):
    x = np.linspace(0,2*np.pi,20000)
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
    return 0

#
p1 = Process(target=plote_Kontrast, args=(Messreihe_Pol[:,0], Kontraste))
p2 = Process(target=plote_n_vs_p, args=(Messreihe_Druck[:,0], Brechungsindex_Luft_arr))

p2.start()
p1.start()
p2.join()
p1.join()

#Tabellen erstellen
data = np.array([[1.23, 2.34, 3.45], [ufloat(4.56, 0.01), 5.67, 6.78], [7.89, 8.90, ufloat(9.01, 0.02)], [10, 11, 12]])
label = "my_table"
h.save_latex_table_to_file(data, header=r'a & b & $c \mathbin{/} \unit{\meter}$', caption='Das ist eine Tabelle', label=label)
print(f"LaTeX table saved as {label}.tex")
