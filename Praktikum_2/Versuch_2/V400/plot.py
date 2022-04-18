import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

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

Messung_1 = np.array(np.genfromtxt('Messdaten/Messung_1.txt'))


Messung_2 = np.array(np.genfromtxt('Messdaten/Messung_2.txt'))
Messung_2 = Messung_2*np.pi/180

Messung_4 = np.array(np.genfromtxt('Messdaten/Messung_4.txt'))
Messung_4 = Messung_4*np.pi/180

Messung_5a = np.array(np.genfromtxt('Messdaten/Messung_5a.txt'))
Messung_5a[:,1:] = Messung_5a[:,1:]*np.pi/180


Messung_5b = np.array(np.genfromtxt('Messdaten/Messung_5b.txt'))
Messung_5b[:,1:] = Messung_5b[:,1:]*np.pi/180

Messung_5c = np.array(np.genfromtxt('Messdaten/Messung_5c.txt'))
Messung_5c[:,1] = Messung_5c[:,1]*np.pi/180

Messung_5d = np.array(np.genfromtxt('Messdaten/Messung_5d.txt'))
Messung_5d[:,1] = Messung_5d[:,1]*np.pi/180

#Aufgabe 1
winkelunterschied_a = abs(Messung_1[:,0]- Messung_1[:,1])
winkelunterschied_a_ufloat=  ufloat(np.mean(winkelunterschied_a),np.std(winkelunterschied_a))

#Aufgabe 2
n_b = np.sin(Messung_2[1:,0])/np.sin(Messung_2[1:,1])
n_b_ufloat =ufloat(np.mean(n_b),np.std(n_b))
v_b = (3*1e8)/n_b_ufloat
#print(n_b)
n_plexiglas = n_b_ufloat

#Aufgabe 3
s_1 = 0.0585*np.sin(Messung_2[1:,0]-Messung_2[1:,1])/np.cos(Messung_2[1:,1])
beta = np.arcsin(np.sin(Messung_2[1:,0])/n_plexiglas.nominal_value)
s_2 = 0.0585*np.sin(Messung_2[1:,0] -beta)/np.cos(beta) 


s_1_ufloat= ufloat(np.mean(s_1),np.std(s_1))
s_2_ufloat= ufloat(np.mean(s_2),np.std(s_2))

#Aufgabe 4
#gruner Laser
delta_g_1 = Messung_4[:,0]+Messung_4[:,1]-np.pi/3

beta1_g = np.arcsin(np.sin(Messung_4[:,0])/1.555)
beta2_g = np.arcsin(np.sin(Messung_4[:,1])/1.555)

delta_g_2 = (Messung_4[:,0]+Messung_4[:,1])-(beta1_g+beta2_g)

#roter Laser
delta_r_1 = Messung_4[:,2]+Messung_4[:,3]-np.pi/3

beta1_r = np.arcsin(np.sin(Messung_4[:,2])/1.555)
beta2_r = np.arcsin(np.sin(Messung_4[:,3])/1.555)

delta_r_2 = (Messung_4[:,2]+Messung_4[:,3])-(beta1_r+beta2_r)

writeW(delta_r_2*180/np.pi, "delta_r_2")
writeW(delta_g_2*180/np.pi, "delta_g_2")

#Aufgabe 5
#a

lambda_600 = np.zeros((2, 2))
lambda_600[:,0] = 1/600 * 1/1000 * np.sin(Messung_5a[:,1])/Messung_5a[:,0]
lambda_600[:,1] = 1/600 * 1/1000 * np.sin(Messung_5a[:,2])/Messung_5a[:,0]

lambda_600_g_ufloat = ufloat(np.mean(lambda_600[:,0]),np.std(lambda_600[:,0]))
lambda_600_r_ufloat = ufloat(np.mean(lambda_600[:,1]),np.std(lambda_600[:,1]))
        
#b
lambda_300 = np.zeros((6, 2))

lambda_300[:,0] = 1/300 * 1/1000 * np.sin(Messung_5b[:,1])/Messung_5b[:,0]
lambda_300[:,1] = 1/300 * 1/1000 * np.sin(Messung_5b[:,2])/Messung_5b[:,0]

lambda_300_g_ufloat = ufloat(np.mean(lambda_300[:,0]),np.std(lambda_300[:,0]))
lambda_300_r_ufloat = ufloat(np.mean(lambda_300[:,1]),np.std(lambda_300[:,1]))
#c
lambda_100_g=np.zeros((19))
lambda_100_g[:] = 1/100 * 1/1000 * np.sin(Messung_5c[:,1])/Messung_5c[:,0]


#d
lambda_100_r=np.zeros((14))
lambda_100_r[:] = 1/100 * 1/1000 * np.sin(Messung_5d[:,1])/Messung_5d[:,0]

lambda_100_g_ufloat = ufloat(np.mean(lambda_100_g[:]),np.std(lambda_100_g[:]))
lambda_100_r_ufloat = ufloat(np.mean(lambda_100_r[:]),np.std(lambda_100_r[:]))

writeW(lambda_600*1e9, "Lambda 600")
writeW(lambda_600_g_ufloat*1e9, "lambda_600_g_ufloat")
writeW(lambda_600_r_ufloat*1e9, "lambda_600_r_ufloat")

writeW(lambda_300*1e9, "Lambda 300")
writeW(lambda_300_g_ufloat**1e9, "lambda_300_g_ufloat")
writeW(lambda_300_r_ufloat**1e9, "lambda_300_r_ufloat")

writeW(lambda_100_g*1e9, "labda_100_g")
writeW(lambda_100_g_ufloat*1e9, "lambda_100_g_ufloat")
writeW(lambda_100_r*1e9, "labda_100_r")
writeW(lambda_100_r_ufloat*1e9, "lambda_100_r_ufloat")