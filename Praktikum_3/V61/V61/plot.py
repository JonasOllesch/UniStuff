import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp

def Gaus(x,I0,x0,w):
    return I0*np.exp(-2*((x-x0)/w)**2)

def Gaus2(x,I0,x0,w):
    return I0*np.exp(-2*((x-x0)/w)**2)

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

y_fit_TEM_00 = Gaus(x_fit_TEM_00,*unp.nominal_values(TEM_00_para))

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
print(popt_TEM_01)
print(pcov_TEM_01)

TEM_01_para = correlated_values(popt_TEM_01,pcov_TEM_01)
x_fit_TEM_01 = np.linspace(-32,32,10000)
print(TEM_01_para)
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

#output = ("Messdaten/TEM_00")   
#my_file = open(output + '.txt', "a") 
#for i in range(-22,43):
#    my_file.write(str(i))
#    my_file.write('\n')