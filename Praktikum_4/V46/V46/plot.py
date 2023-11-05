import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

#from multiprocessing  import Process


#Messreihe_Druck = np.genfromtxt('Messdaten/Druck.txt', encoding='unicode-escape')

d = np.array(1.296, 1.36,   5.11)
d = d/1000
N = np.array(2.8*10**18,1.2*10**18, 0)
N = N*10**3