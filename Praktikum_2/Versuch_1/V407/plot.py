import matplotlib.pyplot as pyplot
import numpy as np
import uncertainties as unp
from scipy.optimize import curve_fit
from uncertainties import ufloat
output = ("build/Auswertung")    
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
#To do list  Vergleich mit der Theorie, Theoriekurven,
#absoluter Graph

Messung_1 = np.array(np.genfromtxt('Messung_1.txt'))
Messung_2 = np.array(np.genfromtxt('Messung_2.txt'))

Messung_1[:,0] = Messung_1[:,0] * (np.pi/180)
Messung_1[:,1] = Messung_1[:,1]* 1e-6

Messung_2[:,0] = Messung_2[:,0] * (np.pi/180)
Messung_2[:,1] = Messung_2[:,1] *1e-6

pyplot.scatter(Messung_1[:,0]*(180/np.pi),Messung_1[:,1],s=8, c='red',marker='x',label="senkrechte Polarisation")
pyplot.scatter(Messung_2[:,0]*(180/np.pi),Messung_2[:,1],s=8, c='blue',marker='+',label="parallele Polarisation")



pyplot.ylabel(r'$I\mathbin{/} \unit{\ampere}$')
pyplot.xlabel(r'$\varphi$')
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph.pdf')
pyplot.clf()

#Nullmessung phi = 0, offensichtlich A = 195 my bei parallel und senkrechtem Licht
#relativ Graph
I_0 = 195 * 1e-6
pyplot.scatter(Messung_1[:,0]*(180/np.pi),Messung_1[:,1]/I_0,s=8, c='red',marker='x',label="senkrechte Polarisation")
pyplot.scatter(Messung_2[:,0]*(180/np.pi),Messung_2[:,1]/I_0,s=8, c='blue',marker='+',label="parallele Polarisation")
#irgendwas ist hier noch falsch
x = np.linspace(0,np.pi/2,1000)
y = ((np.sqrt(3.7**2-(np.sin(x))**2)-np.cos(x))**4)/(3.7**2-1)**2
x = 180/np.pi *x
pyplot.plot(x, y,c ='green',label='senkrecht Theorie')


x = np.linspace(0,np.pi/2,1000)
y = ((3.7**2*np.cos(x)-np.sqrt(3.7**2-np.sqrt(3.7**2-(np.sin(x)**2))))/((3.7**2*np.cos(x)+np.sqrt(3.7**2-np.sqrt(3.7**2-(np.sin(x)**2))))))**2
x = 180/np.pi *x
pyplot.plot(x, y, c='orange',label='parallel Theorie')

pyplot.xlim(5,85)
pyplot.ylim(0,1)
pyplot.xlabel(r'$\varphi$')
pyplot.ylabel(r'$I \mathbin{/} \unit{\ampere} $')
pyplot.tight_layout()
pyplot.legend()
pyplot.grid()
pyplot.savefig('build/Graph_a.pdf')
pyplot.clf()

#Brewsterwinkel der Brewsterwinkel wird bei 74.5 angesetzt
n_b = np.tan(74.5*np.pi/180) #Brechungsindex aus Brewsterwinkel
a_b = 74.5*np.pi/180


E = [0]*17
n_s= [0]*17

#Aus senkrechtem Licht
for i in range(0,17):
    E[i] = np.sqrt(Messung_1[i][1]/I_0)
#n_s[:,0] = ufloat(np.sqrt(1+ (4*np.sqrt(Messung_1[:,1]/I_0))*(np.cos(Messung_1[:,0]))^2/((np.sqrt(Messung_1[:,1]/I_0))-1)^2),(0))
for i in range(0,17):
    n_s[i] = np.sqrt(1+ (4*E[i] * (np.cos(Messung_1[i][0]))**2)/np.abs((E[i]-1)**2))


n_s_ufloat = ufloat(np.mean(n_s[:]),np.std(n_s[:]))


#Brechungsindex aus parallelem Licht
E_p = [0]*25
n_p = [0]*25
for i in range(0,25):
    E_p[i] = np.sqrt(Messung_2[i][1])

E_0 = np.sqrt(I_0)

for i in range(0,25):
    tmp1 = (4*E_p[i]*(np.cos(Messung_2[i][0])**2))/(E_p[i]+E_0)
    tmp2 = (4*E_p[i]*(np.cos(Messung_2[i][0])**2))/(E_p[i]+E_0)**2
    n_p[i] = np.sqrt(1-tmp1 + tmp2)


n_p_ufloat = ufloat(np.mean(n_p),np.std(n_p))



Brech_ind_dur = (n_p_ufloat + n_s_ufloat + n_b)/3
print(n_p)
print(n_s)


print(n_b)
print(n_p_ufloat)
print(n_s_ufloat)
print(Brech_ind_dur)

