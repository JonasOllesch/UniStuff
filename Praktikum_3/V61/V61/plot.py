import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp

def Gaus(x,I0,x0,w):
    return I0*np.exp(-2*((x-x0)/w)**2)

def uGaus(x,I0,x0,w):
    return I0*unp.exp(-2*((x-x0)/w)**2)

def PolStrom(x,I0,phi0):
    return I0*(np.cos(x+phi0))**2

def uPolStrom(x,I0,phi0):
    return I0*(unp.cos(x+phi0))**2
#def TEM_01_func(x,I0,x0,w,phi):
#    return (I0*((x*np.cos(phi)-x0)/w)**2)*np.exp(-2*((x*np.cos(phi)-x0)/w)**2)

def TEM_01_func(x,I0,x01,x02,x03,w,a):
    return I0*(((x-x01)/w)**2)*(np.exp(-2*((x-x02)/w)**2)+a*np.exp(-2*((x-x03)/w)**2))



TEM_00_pos,TEM_00_I,TEM_00_gro = np.genfromtxt('Messdaten/TEM_00.txt',encoding='unicode-escape',dtype=None,unpack=True)
for i in range(0,len(TEM_00_I)):
    if TEM_00_gro[i]=="m":
        TEM_00_I[i] = TEM_00_I[i]*(1e-6)
    else:
        TEM_00_I[i] = TEM_00_I[i]*(1e-9)
TEM_00_I = TEM_00_I-(6e-9)        
TEM_00 = np.vstack((TEM_00_pos, TEM_00_I))
TEM_00 = np.transpose(TEM_00)

popt_TEM_00, pcov_TEM_00 = curve_fit(Gaus,xdata=TEM_00[:,0],ydata=TEM_00[:,1],sigma=TEM_00[:,1]*0.1,)
TEM_00_para = correlated_values(popt_TEM_00,pcov_TEM_00)
x_fit_TEM_00 = np.linspace(-23,43,10000)

y_fit_TEM_00 = uGaus(x_fit_TEM_00,*unp.nominal_values(TEM_00_para))

plt.plot(x_fit_TEM_00,unp.nominal_values(y_fit_TEM_00),color='red')

plt.errorbar(TEM_00[:,0],TEM_00[:,1],yerr=TEM_00[:,1]*0.1,color = 'blue', fmt='x',label='Photostrom')
plt.ylabel("I in A")
plt.xlabel("Radius in mm")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/TEM_00.pdf')
plt.clf()



TEM_01_pos,TEM_01_I,TEM_01_gro = np.genfromtxt('Messdaten/TEM_01.txt',encoding='unicode-escape',dtype=None,unpack=True)
for i in range(0,len(TEM_01_I)):
    if TEM_01_gro[i]=="m":
        TEM_01_I[i] = TEM_01_I[i]*(1e-6)
    else:
        TEM_01_I[i] = TEM_01_I[i]*(1e-9)

TEM_01_I = TEM_01_I-(5.35e-9)   
TEM_01 = np.vstack((TEM_01_pos, TEM_01_I))
TEM_01 = np.transpose(TEM_01)

popt_TEM_01, pcov_TEM_01 = curve_fit(TEM_01_func,xdata=TEM_01[:,0],ydata=TEM_01[:,1],sigma=TEM_01[:,1]*0.1,maxfev=20000,p0=[1.89*1e-6,-39,-2,-12,10,1.2])
#print(popt_TEM_01)
#print(pcov_TEM_01)

TEM_01_para = correlated_values(popt_TEM_01,pcov_TEM_01)
x_fit_TEM_01 = np.linspace(-32,32,10000)
#print(TEM_01_para)
y_fit_TEM_01 = TEM_01_func(x_fit_TEM_01,*unp.nominal_values(TEM_01_para))

plt.plot(x_fit_TEM_01,unp.nominal_values(y_fit_TEM_01),color='red',label="theo. MEM 01 Mode")

plt.errorbar(TEM_01[:,0],TEM_01[:,1],yerr=TEM_01[:,1]*0.1,color = 'blue', fmt='x',label='Photostrom')
plt.ylabel("I in A")
plt.xlabel("Radius in mm")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/TEM_01.pdf')
plt.clf()



polarisation = np.genfromtxt('Messdaten/Polarisation.txt',encoding='unicode-escape')
polarisation[:,1] = polarisation[:,1]*1e-6 -5.35e-9#in Ampere minus Hintergrund 

popt_pol, pcov_pol = curve_fit(PolStrom,polarisation[:,0]*np.pi/180,polarisation[:,1],sigma=polarisation[:,1]*0.1,p0=[6e-5,1.21])
para_pol = correlated_values(popt_pol,pcov_pol)
print(para_pol)
x_fit_pol = np.linspace(-0.5,2*np.pi+0.5,100)
y_fit_pol = uPolStrom(x_fit_pol,*para_pol)

plt.errorbar(polarisation[:,0]*np.pi/180,polarisation[:,1],yerr=polarisation[:,1]*0.1,color = 'blue', fmt='x',label='Photostrom hinter dem Polfilter')
plt.plot(x_fit_pol,unp.nominal_values(y_fit_pol),label="theo. Verlauf",color='red')
plt.ylabel("I in A")
plt.xlabel("Winkel in °")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/polarisation.pdf')
plt.clf()


fig = plt.figure()
ax = plt.axes(polar=True)
r =  polarisation[:,1]
theta = np.pi/180 * polarisation[:,0]
ax.set_rmax(np.max(r))
ax.set_rlabel_position(0)
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position+10),ax.get_rmax()/2.,'Photostrom in A',
        rotation=label_position,ha='center',va='center')
ax.errorbar(theta, r, yerr=polarisation[:,1]*0.1, capsize=0,fmt='x',label="Photostrom hinter dem Polfilter")
ax.plot(x_fit_pol,unp.nominal_values(y_fit_pol),color='red',label="theo Verlauf")
plt.tight_layout()
plt.legend(loc='lower left')
plt.savefig('build/polarisation_pol.pdf')
plt.clf()

#output = ("Messdaten/Polarisation")   
#my_file = open(output + '.txt', "a") 
#for i in range(0,73):
#    my_file.write(str(i*5))
#    my_file.write('\n')