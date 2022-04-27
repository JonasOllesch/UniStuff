import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties as unp
Aktivität_a0 = 845/900
Aktivität_a0_abw = np.sqrt(845)/900

Messung_1_a = np.array(np.genfromtxt('Messung_1_a.txt'))

Messung_1_a[:,0] = Messung_1_a[:,0]*(10**-3)
Zählrate_1_a_poissonabw = np.sqrt(Messung_1_a[:,1])

Aktivität_1_a =Messung_1_a[:,1]/Messung_1_a[:,2]
Aktivität_1_a_poissonabw = Zählrate_1_a_poissonabw[:]/Messung_1_a[:,2]
Aktivität_1_a_min_a0 = Aktivität_1_a -Aktivität_a0
Aktivität_1_a_min_a0_poissonabw =Aktivität_1_a_poissonabw 

#print(Aktivität_1_a)
#print(Aktivität_1_a_min_a0)
#print(Aktivität_1_a_min_a0_poissonabw)
