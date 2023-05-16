import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import copy
from scipy import constants

def Gaus(x,I0,x0,w):
    return I0*np.exp(-((x-x0)/w)**2)

def uGaus(x,I0,x0,w):
    return I0*unp.exp(-((x-x0)/w)**2)

def PolStrom(x,I0,phi0):
    return I0*(np.cos(x+phi0))**2

def uPolStrom(x,I0,phi0):
    return I0*(unp.cos(x+phi0))**2
#def TEM_01_func(x,I0,x0,w,phi):
#    return (I0*((x*np.cos(phi)-x0)/w)**2)*np.exp(-2*((x*np.cos(phi)-x0)/w)**2)

def TEM_01_func(x,I0,x01,x02,x03,w,w2):
    return I0*(((x-x01))**2)*(np.exp(-((x-x02)/w)**2)+np.exp(-((x-x03)/w2)**2))


def Bragg(n,g,theta):#die Braggbedingung
    return 2*g*np.sin(theta)/n

def berechne_Abstand_mittel(dd,Abstand):
    #print("dd: ",dd)
    for i in range (0,len(Abstand)):
        #print("dd[i],d[len(dd)-i-1]: ",dd[i],dd[len(dd)-i-1])
        Abstand[i] = (dd[i]+dd[len(dd)-i-1])/2

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
print("Parameter der TEM_00_para: ", TEM_00_para)
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

popt_TEM_01, pcov_TEM_01 = curve_fit(TEM_01_func,xdata=TEM_01[:,0],ydata=TEM_01[:,1],sigma=TEM_01[:,1]*0.1,maxfev=1000,p0=[1.89*1e-6,-1,-15,-10,10,1])
#print(popt_TEM_01)
#print(pcov_TEM_01)

TEM_01_para = correlated_values(popt_TEM_01,pcov_TEM_01)
print("Parameter der TEM_01: ",TEM_01_para)
x_fit_TEM_01 = np.linspace(-32,32,10000)
#print(TEM_01_para)
y_fit_TEM_01 = TEM_01_func(x_fit_TEM_01,*unp.nominal_values(TEM_01_para))

plt.plot(x_fit_TEM_01,unp.nominal_values(y_fit_TEM_01),color='red',label=r"Fit der TEM_{01}")

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
print("min polarisation: ", np.min(polarisation[:,1]))
print("max polarisation: ", np.max(polarisation[:,1]))
print("Verhältnis min/max",np.min(polarisation[:,1])/np.max(polarisation[:,1]))


popt_pol, pcov_pol = curve_fit(PolStrom,polarisation[:,0]*np.pi/180,polarisation[:,1],sigma=polarisation[:,1]*0.1,p0=[6e-5,1.21])
para_pol = correlated_values(popt_pol,pcov_pol)
#print(para_pol)
x_fit_pol = np.linspace(-0.5,2*np.pi+0.5,100)
y_fit_pol = uPolStrom(x_fit_pol,*para_pol)
print("parameter der Polarisation I_0 : ", repr(para_pol[0]))
print("parameter der Polarisation theta 0 : ", repr(para_pol[1]*180/np.pi))

plt.errorbar(polarisation[:,0],polarisation[:,1],yerr=polarisation[:,1]*0.1,color = 'blue', fmt='x',label='Photostrom')
plt.plot(x_fit_pol*180/np.pi,unp.nominal_values(y_fit_pol),label="Fit",color='red')
plt.xlim(-0.1,360.1)
plt.ylabel(r"$I \mathbin{/} \unit{\ampere}$")
plt.xlabel(r"$\text{Winkel} \mathbin{/} °$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend(loc="upper right")
plt.savefig('build/polarisation.pdf')
plt.clf()


fig = plt.figure()
ax = plt.axes(polar=True)
r =  polarisation[:,1]
theta = np.pi/180 * polarisation[:,0]
ax.set_rmax(np.max(r))
ax.set_rlabel_position(0)
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position+0),ax.get_rmax()/ (3/6),r'$ \text{Photostrom} \mathbin{/} \unit{\ampere}$',
        rotation=label_position,ha='center',va='center',fontsize=8)
ax.errorbar(theta, r, yerr=polarisation[:,1]*0.1, capsize=0,fmt='x',label="Photostrom")
ax.plot(x_fit_pol,unp.nominal_values(y_fit_pol),color='red',label="Fit")
plt.tight_layout()
plt.legend(loc='lower right')
plt.savefig('build/polarisation_pol.pdf')
plt.clf()


Beugung_80   = np.genfromtxt('Messdaten/Wellenlaenge_80.txt',encoding='unicode-escape')
Beugung_100  = np.genfromtxt('Messdaten/Wellenlaenge_100.txt',encoding='unicode-escape')
Beugung_600  = np.genfromtxt('Messdaten/Wellenlaenge_600.txt',encoding='unicode-escape')
Beugung_1200 = np.genfromtxt('Messdaten/Wellenlaenge_1200.txt',encoding='unicode-escape')

Beugung_80[:,1] = Beugung_80[:,1]/100
Beugung_100[:,1] = Beugung_100[:,1]/100
Beugung_600[:,1] = Beugung_600[:,1]/100
Beugung_1200[:,1] = Beugung_1200[:,1]/100


d = np.array([77.3,77.3,29.4,16,4])
d = d /100 # von cm in m
Gitter_const = np.array([1/80,1/100,1/600,1/1200])
Gitter_const = Gitter_const/1000 #von mm in m

Mittlere_Abstand_Beugung_80 = np.zeros(int((len(Beugung_80[:,1])-1)/2))
berechne_Abstand_mittel(Beugung_80[:,1],Mittlere_Abstand_Beugung_80)

Mittlere_Abstand_Beugung_100 = np.zeros(int((len(Beugung_100[:,1])-1)/2))
berechne_Abstand_mittel(Beugung_100[:,1],Mittlere_Abstand_Beugung_100)

Mittlere_Abstand_Beugung_600 = np.zeros(int((len(Beugung_600[:,1])-1)/2))
berechne_Abstand_mittel(Beugung_600[:,1],Mittlere_Abstand_Beugung_600)

Mittlere_Abstand_Beugung_1200 = np.zeros(int((len(Beugung_1200[:,1])-1)/2))
berechne_Abstand_mittel(Beugung_1200[:,1],Mittlere_Abstand_Beugung_1200)

#print("gemittelter Abstand: ",Mittlere_Abstand_Beugung_80)
theta_80 = np.zeros(len(Mittlere_Abstand_Beugung_80[:]))
theta_80 = np.arctan(Mittlere_Abstand_Beugung_80[:]/d[0])
#print("Winkel: ",theta_80*180/np.pi)
lam_80 = Gitter_const[0]*np.sin(theta_80)/Beugung_80[:8,0]
#print(lam_80*1e9)
print("lam 80 mean: ", np.mean(unp.nominal_values(lam_80*1e9)))
print("lam 80 mean: ", np.std(unp.nominal_values(lam_80*1e9)))


#print("gemittelter Abstand: ",Mittlere_Abstand_Beugung_100)
theta_100 = np.zeros(len(Mittlere_Abstand_Beugung_100[:]))
theta_100 = np.arctan(Mittlere_Abstand_Beugung_100[:]/d[1])
#print("Winkel: ",theta_100*180/np.pi)
lam_100 = (Gitter_const[1]*np.sin(theta_100[:]))/Beugung_100[:6,0]
#print(lam_100*1e9)
print("lam 100 mean: ", np.mean(unp.nominal_values(lam_100*1e9)))
print("lam 100 mean: ", np.std(unp.nominal_values (lam_100*1e9)))
#print("lam 100 mean: ",np.mean(lam_100*1e9))
#print("lam 100 std: ", np.std(lam_100*1e9))

#print("gemittelter Abstand: ",Mittlere_Abstand_Beugung_600)
theta_600 = np.zeros(len(Mittlere_Abstand_Beugung_600[:]))
theta_600 = np.arctan(Mittlere_Abstand_Beugung_600[:]/d[2])
#print("Winkel: ",theta_600*180/np.pi)
lam_600 = (Gitter_const[2]*np.sin(theta_600[:]))/Beugung_600[:2,0]
#print(lam_600*1e9)
print("lam 600 mean: ", np.mean(unp.nominal_values(lam_600*1e9)))
print("lam 600 mean: ", np.std(unp.nominal_values (lam_600*1e9)))

#print("gemittelter Abstand: ",Mittlere_Abstand_Beugung_1200)
theta_1200 = np.zeros(len(Mittlere_Abstand_Beugung_1200[:]))
theta_1200 = np.arctan(Mittlere_Abstand_Beugung_1200[:]/d[3])
#print("Winkel: ",theta_1200*180/np.pi)
lam_1200 = (Gitter_const[3]*unp.sin(ufloat(theta_1200,0.001)))/Beugung_1200[:1,0]
print("lambda 1200: ",repr(lam_1200*1e9))

x = np.linspace(0,2,1000)
stab_plan_kon_g1 = 1 
stab_plan_kon_g2 = 1 - x/(1400*(1e-3))#platzhalter

stab_kon_kon_g1 = 1 - x/(1400*(1e-3))#platzhalter
stab_kon_kon_g2 = 1 - x/(1400*(1e-3))#platzhalter




plt.plot(x,stab_plan_kon_g1*stab_plan_kon_g2,label=r"$r_{1} = \infty\, r_{2} = 1400, \, \unit{\milli\meter}$",color="blue")
plt.plot(x,stab_kon_kon_g1*stab_kon_kon_g2,label=r"$r_{1} =  1400 \unit{\milli\meter}, \, r_{2} = 1400 \, \unit{\milli\meter}$",color="red")
plt.xlabel(r"$\text{Resonatorlänge} \mathbin{/} \unit{\meter}$")
plt.ylabel(r"$g_1 g_2$")
plt.xlim(-0.25,2.25)
plt.ylim(-0.25,1.25)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Stabilität_theo.pdf')
plt.clf()

longitudinal_modes = []
for i in range(0,27):
    longitudinal_modes.append(np.genfromtxt('Messdaten/Stabilität_plan_konv.txt',encoding='unicode-escape', skip_header=i,max_rows=1))
del longitudinal_modes[0]
del longitudinal_modes[0]

test = copy.deepcopy(longitudinal_modes)
for j in range(0,len(test[:])):
    for i in reversed(range(1,len(test[j][:]))):
        test[j][i] = test[j][i]-test[j][i-1]
test2 = []
for i in range(0,len(test[:])):
    test2.append(test[i][2:])

means = []
std = []
for i in range(0,len(test[:])):
    means.append(np.mean(test2[i][:]))
    std.append(np.std(test2[i][:]))
print("mittlerer Abstand der longitudinal Moden: ", means)
print("Standartabweichung Abstand der longitudinal Moden: ", std)

#output = ("Messdaten/Wellenlaenge_80")   
#my_file = open(output + '.txt', "a") 
#for i in range(-8,9):
#    my_file.write(str(i))
#    my_file.write('\n')
#mean_ges =  np.sqrt(2*constants.Boltzmann*300/(20.18*constants.atomic_mass))
#print("durchschnittliche Geschwindigkeit der Atome: ", mean_ges)
#def doppler(v):
#    return  (constants.speed_of_light/632.8*(1e-9)) * np.sqrt(  (constants.speed_of_light - v) / (constants.speed_of_light + v)  )
#f_plus = doppler(mean_ges)
#f_min = doppler(-mean_ges)
#print("f_plus", f_plus)
#print("f_min", f_min)
#print("Gesamtbreite der Dopplerverschiebung: ", f_plus-f_min)