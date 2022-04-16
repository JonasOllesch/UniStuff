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
#print(Messung_5b)

Messung_5c = np.array(np.genfromtxt('Messdaten/Messung_5c.txt'))
Messung_5c[:,1] = Messung_5c[:,1]*np.pi/180
#print(Messung_5c)
Messung_5d = np.array(np.genfromtxt('Messdaten/Messung_5d.txt'))
Messung_5d[:,1] = Messung_5d[:,1]*np.pi/180
#print(Messung_5d)

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
print('Brechungsindex der Platte')
print(n_b_ufloat)
print('Lichtgeschwindigkeit in der Platte')
print(v_b)
n_plexiglas = n_b_ufloat

#Aufgabe 3
s_1 = 0.0585*np.sin(Messung_2[1:,0]-Messung_2[1:,1])/np.cos(Messung_2[1:,1])
beta = np.arcsin(np.sin(Messung_2[1:,0])/n_plexiglas.nominal_value)
s_2 = 0.0585*np.sin(Messung_2[1:,0] -beta)/np.cos(beta) 
#print("beta")
#print(beta*180/np.pi)
print('Strahlversätze s über Messdaten')
print(s_1)
print('Strahlversätze s über berechnetes β')
print(s_2)
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

lambda_600 = np.zeros((2, 2))
lambda_600[:,0] = 1/600 * 1/1000 * np.sin(Messung_5a[:,1])/Messung_5a[:,0]
lambda_600[:,1] = 1/600 * 1/1000 * np.sin(Messung_5a[:,2])/Messung_5a[:,0]



lambda_600_g_ufloat = ufloat(np.mean(lambda_600[:,0]),np.std(lambda_600[:,0]))
lambda_600_r_ufloat = ufloat(np.mean(lambda_600[:,1]),np.std(lambda_600[:,1]))

        
#print(lambda_600)
#b


lambda_300 = np.zeros((6, 2))
#print(lambda_300)

lambda_300[:,0] = 1/300 * 1/1000 * np.sin(Messung_5b[:,1])/Messung_5b[:,0]
lambda_300[:,1] = 1/300 * 1/1000 * np.sin(Messung_5b[:,2])/Messung_5b[:,0]

#print(lambda_300)
lambda_300_g_ufloat = ufloat(np.mean(lambda_300[:,0]),np.std(lambda_300[:,0]))
lambda_300_r_ufloat = ufloat(np.mean(lambda_300[:,1]),np.std(lambda_300[:,1]))
#c
lambda_100_g=np.zeros((19))
lambda_100_g[:] = 1/100 * 1/1000 * np.sin(Messung_5c[:,1])/Messung_5c[:,0]
#print(lambda_100_g)


#d
lambda_100_r=np.zeros((14))
lambda_100_r[:] = 1/100 * 1/1000 * np.sin(Messung_5d[:,1])/Messung_5d[:,0]
#print(lambda_100_r)

lambda_100_g_ufloat = ufloat(np.mean(lambda_100_g[:]),np.std(lambda_100_g[:]))
lambda_100_r_ufloat = ufloat(np.mean(lambda_100_r[:]),np.std(lambda_100_r[:]))

print(lambda_600_g_ufloat)
print(lambda_600_r_ufloat)
print(lambda_600)

#print(lambda_300_g_ufloat)
#print(lambda_300_r_ufloat)
#print(lambda_300)


#print(lambda_100_g_ufloat)
#print(lambda_100_r_ufloat)

#print(lambda_100_g)
#print(lambda_100_r)