import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from pandas import read_csv
import math
from scipy.optimize import curve_fit
from numpy import arange


#to do list
#Auswertung:
#   a "ns" berechnen und mit den Theoriewerten berechnen
#   b ν- und ν+ mit den Theoriewerten vergleichen
#   c aus t_1 und t_2 mit dem theoretischen ν- und ν+ vergleichen und vielleicht mit denen aus b wenn man noch Lust hat

# Werte einlesen und in Si Einheiten konvertieren
Messung_a = np.array(np.genfromtxt('a.txt'))
Messung_a[:,0]= Messung_a[:,0]*(10**-9)

Messung_b = np.array(np.genfromtxt('b.txt'))
Messung_b[:,0]= Messung_b[:,0]*(10**-9)
Messung_b[:,1]= Messung_b[:,1]*(10**3)
Messung_b[:,2]= Messung_b[:,2]*(10**3)

Messung_c = np.array(np.genfromtxt('c.txt'))
Messung_c[:,0]= Messung_c[:,0]*(10**-9)
Messung_c[:,1]= Messung_c[:,1]*(10**-3)
Messung_c[:,2]= Messung_c[:,2]*(10**-3)
