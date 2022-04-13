import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat


Messung_1 = np.array(np.genfromtxt('Messdaten/Messung_1.txt'))


Messung_2 = np.array(np.genfromtxt('Messdaten/Messung_2.txt'))
Messung_2 = Messung_2*np.pi/180

Messung_4 = np.array(np.genfromtxt('Messdaten/Messung_4.txt'))
Messung_4 = Messung_4*np.pi/180

Messung_5a = np.array(np.genfromtxt('Messdaten/Messung_5a.txt'))
Messung_5a[:,1:] = Messung_5a[:,1:]*np.pi/180


Messung_5b = np.array(np.genfromtxt('Messdaten/Messung_5b.txt'))
Messung_5b[:,1:] = Messung_5b[:,1:]*np.pi/180
print(Messung_5b)

Messung_5c = np.array(np.genfromtxt('Messdaten/Messung_5c.txt'))
Messung_5d = np.array(np.genfromtxt('Messdaten/Messung_5d.txt'))

#print(Messung_1)
#print(Messung_2)
#print(Messung_4)
#print(Messung_5a)
#print(Messung_5b)
#print(Messung_5c)
#print(Messung_5d)
#Aufgabe 1
winkelunterschied_a = abs(Messung_1[:,0]- Messung_1[:,1])
winkelunterschied_a_ufloat=  ufloat(np.mean(winkelunterschied_a),np.std(winkelunterschied_a))
#print(winkelunterschied_a)
#print(winkelunterschied_a_ufloat)

#Aufgabe 2
n_b = np.sin(Messung_2[1:,0])/np.sin(Messung_2[1:,1])
n_b_ufloat =ufloat(np.mean(n_b),np.std(n_b))
v_b = (3*1e8)/n_b_ufloat
#print(n_b)
#print(n_b_ufloat)
#print(v_b)
n_plexiglas = n_b_ufloat

#Aufgabe 3
s_1 = 0.0585*np.sin(Messung_2[1:,0]-Messung_2[1:,1])/np.cos(Messung_2[1:,1])
beta = np.arcsin(np.sin(Messung_2[1:,0])/n_plexiglas.nominal_value)
s_2 = 0.0585*np.sin(Messung_2[1:,0] -beta)/np.cos(beta) 
#print("beta")
#print(beta*180/np.pi)
#print(s_1)
#print(s_2)
s_1_ufloat= ufloat(np.mean(s_1),np.std(s_1))
s_2_ufloat= ufloat(np.mean(s_2),np.std(s_2))

#Aufgabe 4
#gruner Laser
delta_g_1 = Messung_4[:,0]+Messung_4[:,1]-np.pi/3
#print(delta_g_1)
beta1_g = np.arcsin(np.sin(Messung_4[:,0])/1.555)
beta2_g = np.arcsin(np.sin(Messung_4[:,1])/1.555)

delta_g_2 = (Messung_4[:,0]+Messung_4[:,1])-(beta1_g+beta2_g)
#print(delta_g_1)
#print(delta_g_2)

#roter Laser
delta_r_1 = Messung_4[:,2]+Messung_4[:,3]-np.pi/3

beta1_r = np.arcsin(np.sin(Messung_4[:,2])/1.555)
beta2_r = np.arcsin(np.sin(Messung_4[:,3])/1.555)

delta_r_2 = (Messung_4[:,2]+Messung_4[:,3])-(beta1_r+beta2_r)

#print(delta_r_1*180/np.pi)
#print(delta_r_2*180/np.pi)
#Aufgabe 5
#a

lambda_600 = [0]*2,[0]*2,[0]*2
for i in range(1,3):
    for j in range(0,3):
        if Messung_5a[j][0] != 0:
            lambda_600[j][i-1] = 1/600 * 1/1000 * np.sin(Messung_5a[j][i])/Messung_5a[j][0]

        
#print(lambda_600)
#b
lambda_300 = [[0]*2]*8
for j in range(0,7):
    for i in range(1,3):
        if Messung_5b[j][0] != 0:
            lambda_300[j][i-1] = 1/300 * 1/1000 * np.sin(Messung_5b[j][i])/Messung_5b[j][0]

print(lambda_300)