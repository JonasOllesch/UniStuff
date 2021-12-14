import matplotlib.pyplot as pyplot
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import math
from scipy.optimize import curve_fit
from numpy import arange

Messung_a = np.array(np.genfromtxt('Messung_a.txt'))
#print(math.sin(math.pi))# <-- das ist anscheinend nicht null aber math arbeitet in Bogenmaß
#print(np.sin(np.pi))      #    np wohl auch
Messung_a[:,0] = ((np.pi)/180)*Messung_a[:,0]# von Grad in Bogenmaß
#und alle andere Umrechnungen in Si Einheiten

Messung_b = np.array(np.genfromtxt('Messung_b.txt'))
Messung_b[:,0]  = Messung_b[:,0]/100
Messung_b[:,1]  = Messung_b[:,1]/3

Messung_c = np.array(np.genfromtxt('Messung_c.txt'))
Messung_c[:,1] = Messung_c[:,1]/3

Messung_d = np.array(np.genfromtxt('Messung_d.txt'))
Messung_d[:,1] = Messung_d[:,1]/3

Messung_e = np.array(np.genfromtxt('Messung_e.txt'))
Messung_e[:,1] = Messung_e[:,1]/1000
Messung_e[:,2] = Messung_e[:,2]/1000
Messung_e[:,3] = Messung_e[:,3]/1000
Messung_e[:,4] = Messung_e[:,4]/1000

Messung_f = np.array(np.genfromtxt('Messung_f.txt'))
Messung_f[:,1] = Messung_f[:,1]/3
Messung_f[:,2] = Messung_f[:,2]/3

Messung_g = np.array(np.genfromtxt('Messung_g.txt'))
Messung_g[:,1] = Messung_g[:,1]/3
Messung_g[:,1] = Messung_g[:,1]/3

#Bestimmung der Winkelrichtgröße
WinRgtmp = [0,1,2,3,4,5,6,7,8,9]
for i in range(0,len(WinRgtmp)):
    WinRgtmp[i] = (Messung_a[i][1]*0.2)/Messung_a[i][0]
Winkelrichtgröße = ufloat(np.mean(WinRgtmp),np.std(WinRgtmp))
del WinRgtmp