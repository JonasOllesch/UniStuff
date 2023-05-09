import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import pandas as pd

#Channel_Kalibrierung = np.genfromtxt('Messdaten/Channel_Kalibrierung.txt',encoding='unicode-escape')