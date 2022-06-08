import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as sc



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

Messung_2_a = np.array(np.genfromtxt('Messung_2_a.txt'))
Messung_2_a[:,0] = Messung_2_a[:,0]*5*(10**(-3))
Messung_2_a[:,1] = Messung_2_a[:,1]*0.1
Messung_2_a[:,2] = Messung_2_a[:,2]*5*(10**(-3))
Messung_2_a[:,3] = Messung_2_a[:,3]*0.1


Messung_2_b = np.array(np.genfromtxt('Messung_2_b.txt'))
Messung_2_b[:,0] = Messung_2_b[:,0]*5*(10**(-3))
Messung_2_b[:,1] = Messung_2_b[:,1]*0.1
Messung_2_b[:,2] = Messung_2_b[:,2]*5*(10**(-3))
Messung_2_b[:,3] = Messung_2_b[:,3]*0.1

Messung_2_c = np.array(np.genfromtxt('Messung_2_c.txt'))
Messung_2_c[:,0] = Messung_2_c[:,0]*5*(10**(-3))
Messung_2_c[:,1] = Messung_2_c[:,1]*0.1
Messung_2_c[:,2] = Messung_2_c[:,2]*5*(10**(-3))
Messung_2_c[:,3] = Messung_2_c[:,3]*0.1


#x = np.linspace(10,40,1000)
#y  =lorentzkurve(350000, 22**2, 10**2, x)
#popt, pcov = curve_fit(lorentzkurve, Messung_1[:,0]/1000, Messung_1[:,1]) der curve fit funktioniert so ungefähr gar weil, die Parameter nicht konvergieren
#y = lorentzkurve(popt[0], popt[1], popt[2], x)
#writeW(popt, "popt")
#writeW(popt, "Parameter der lorentzkurve")
#plt.plot(x,y,color='r',label="regression")

plt.scatter(Messung_1[:,0]/1000, Messung_1[:,1],s=8,label="Messdaten")
x_a = np.linspace(8,42,1000)
y_a = 8.1/np.sqrt(2)+x_a[:]-x_a[:]
plt.plot(x_a,y_a,color = 'green',label=r"$\frac{1}{\sqrt{2}} \cdot U_{A,max} $ ")
plt.scatter(20.75, 8.1/np.sqrt(2),color = 'black',s=8)
plt.scatter(23, 8.1/np.sqrt(2),color = 'black',s=8)
plt.xlim(8,42)
plt.xlabel(r'$ f \, \mathbin{/} \unit{\kilo\hertz}$')
plt.ylabel(r'$ U \mathbin{/} \unit{\volt}  $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/Graph_a.pdf')
plt.clf()

Q_real_Dy = 0.01438/(0.155*7800)
Q_real_Gd = 0.1020/(0.159*7400)
Q_real_Nd = 0.0766/(0.16*7240)

writeW(Q_real_Dy, "Q_real_Dy2O3")
writeW(Q_real_Gd, "Q_real_Gd2O3")
writeW(Q_real_Nd, "Q_real_Nd2O3")

suszep_aus_r = np.zeros((3,3),dtype=float)#Dy2O3  Gd2O3 Nd2O3
suszep_aus_r[:,0] = (2*abs(Messung_2_a[:,0]-Messung_2_a[:,2])*(86.6*10**(-6)))/(998*Q_real_Dy)
suszep_aus_r[:,1] = (2*abs(Messung_2_b[:,0]-Messung_2_b[:,2])*(86.6*10**(-6)))/(998*Q_real_Gd)
suszep_aus_r[:,2] = (2*abs(Messung_2_c[:,0]-Messung_2_c[:,2])*(86.6*10**(-6)))/(998*Q_real_Nd)
writeW(suszep_aus_r, "suszep_aus_r")

suszep_aus_u = np.zeros((3,3),dtype=float)
suszep_aus_u[:,0] = 4*(86.6*10**(-6))*abs(Messung_2_a[:,1]-Messung_2_a[:,3])/(Q_real_Dy)
suszep_aus_u[:,1] = 4*(86.6*10**(-6))*abs(Messung_2_b[:,1]-Messung_2_b[:,3])/(Q_real_Gd)
suszep_aus_u[:,2] = 4*(86.6*10**(-6))*abs(Messung_2_c[:,1]-Messung_2_c[:,3])/(Q_real_Nd)

suszep_mean_std = np.zeros((3,4),dtype = float)

suszep_aus_r_Dy203_mean = np.mean(suszep_aus_r[:,0])
suszep_aus_r_Dy203_std = np.std(suszep_aus_r[:,0])
suszep_aus_r_Gd203_mean = np.mean(suszep_aus_r[:,1])
suszep_aus_r_Gd203_std = np.std(suszep_aus_r[:,1])
suszep_aus_r_Nd203_mean = np.mean(suszep_aus_r[:,2])
suszep_aus_r_Nd203_std = np.std(suszep_aus_r[:,2])

suszep_aus_u_Dy203_mean = np.mean(suszep_aus_u[:,0])
suszep_aus_u_Dy203_std = np.std(suszep_aus_u[:,0])
suszep_aus_u_Gd203_mean = np.mean(suszep_aus_u[:,1])
suszep_aus_u_Gd203_std = np.std(suszep_aus_u[:,1])
suszep_aus_u_Nd203_mean = np.mean(suszep_aus_u[:,2])
suszep_aus_u_Nd203_std = np.std(suszep_aus_u[:,2])

writeW(suszep_aus_r_Dy203_mean, "suszep_aus_r_Dy203_mean")
writeW(suszep_aus_r_Dy203_std, "suszep_aus_r_Dy203_std")
writeW(suszep_aus_r_Gd203_mean, "suszep_aus_r_Gd203_mean")
writeW(suszep_aus_r_Gd203_std, "suszep_aus_r_Gd203_std")
writeW(suszep_aus_r_Nd203_mean, "suszep_aus_r_Nd203_mean")
writeW(suszep_aus_r_Nd203_std, "suszep_aus_r_Nd203_std")
writeW(suszep_aus_u_Dy203_mean, "suszep_aus_u_Dy203_mean")
writeW(suszep_aus_u_Dy203_std, "suszep_aus_u_Dy203_std")
writeW(suszep_aus_u_Gd203_mean, "suszep_aus_u_Gd203_mean")
writeW(suszep_aus_u_Gd203_std, "suszep_aus_u_Gd203_std")
writeW(suszep_aus_u_Nd203_mean, "suszep_aus_u_Nd203_mean")
writeW(suszep_aus_u_Nd203_std, "suszep_aus_u_Nd203_std")

#print(Messung_2_a)
#print(Messung_2_b)
#print(Messung_2_c)

#print(abs(Messung_2_a[:,0]-Messung_2_a[:,2]), " delta ohm")
#print(abs(Messung_2_b[:,0]-Messung_2_b[:,2]), " delta ohm")
#print(abs(Messung_2_c[:,0]-Messung_2_c[:,2]), " delta ohm")

writeW(abs(Messung_2_a[:,1]-Messung_2_a[:,3]), "delta U Dy2o3")
writeW(abs(Messung_2_b[:,1]-Messung_2_b[:,3]), "delta U Gd2o3")
writeW(abs(Messung_2_c[:,1]-Messung_2_c[:,3]), "delta U Nd2o3")

## theoretische Suszeptibilitäten

mu_0 = sc.mu_0
mu_B = sc.value('Bohr magneton')
k    = sc.k
T = 295.15

# Probe 1
g_J1 = 8/11
J_1 = 9/2
N_1 = 2.59*10**28

chi_1 = (mu_0*mu_B**2*g_J1**2*N_1*J_1*(J_1 + 1))/(3*k*T)

writeW(chi_1, 'Theoretische Suszeptibilität für Nd')

# Probe 2
g_J2 = 2
J_2 = 3.5
N_2 = 2.46*10**28

chi_2 = (mu_0*mu_B**2*g_J2**2*N_2*J_2*(J_2 + 1))/(3*k*T)

writeW(chi_2, 'Theoretische Suszeptibilität für Gd')

# Probe 3
g_J3 = 4/3
J_3 = 7.5
N_3 = 2.52*10**28

chi_3 = (mu_0*mu_B**2*g_J3**2*N_3*J_3*(J_3 + 1))/(3*k*T)

writeW(chi_3, 'Theoretische Suszeptibilität für Dy')
