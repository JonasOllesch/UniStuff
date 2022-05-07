import matplotlib.pyplot as pyplot
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

def Stroe_formle(delta_frequenz,Winkel):
    return  delta_frequenz*1800/(2*2*(10**6)*np.cos(Winkel*np.pi/180))

#c = 1800
#v_0 = 2*10**6

Messung_1_a = np.array(np.genfromtxt('Messung_1_a.txt'))
Messung_1_b = np.array(np.genfromtxt('Messung_1_b.txt'))
Messung_1_c = np.array(np.genfromtxt('Messung_1_c.txt'))

Pumpleistunginpro = np.zeros(5)
Pumpleistunginpro[:] = (Messung_1_a[:,0]/8600 )*100

delta_v = np.zeros((5,3)) # 15 30 45
delta_v[:,0]= Messung_1_a[:,1]-Messung_1_a[:,2]
delta_v[:,1]= Messung_1_b[:,1]-Messung_1_b[:,2]
delta_v[:,2]= Messung_1_c[:,1]-Messung_1_c[:,2]



Stroemungsgeschwindigkeit = np.zeros((5,3))
Stroemungsgeschwindigkeit[:,0] = Stroe_formle(delta_v[:,0], 15)
Stroemungsgeschwindigkeit[:,1] = Stroe_formle(delta_v[:,1], 30)
Stroemungsgeschwindigkeit[:,2] = Stroe_formle(delta_v[:,2], 45)


print(delta_v)
print(Stroemungsgeschwindigkeit)
print(Pumpleistunginpro)

writeW(delta_v, "delta_v")
writeW(Stroemungsgeschwindigkeit, "Stroemungsgeschwindigkeit")
writeW(Pumpleistunginpro, "Pumpleistunginpro")