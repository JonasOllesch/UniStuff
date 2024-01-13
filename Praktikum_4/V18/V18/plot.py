import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

#from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)
#from multiprocessing  import Process


Caesium = np.genfromtxt('Messdaten/Caesium.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
Cobalt60 = np.genfromtxt('Messdaten/Cobalt-60.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
Europium = np.genfromtxt('Messdaten/Europium.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
Uranophan = np.genfromtxt('Messdaten/Uranophan.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
Hintergrund = np.genfromtxt('Messdaten/Hintergrund.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 



print(f'Summe der Caesium-Signale: {np.sum(Caesium)}')
print(f'Summe der Cobalt60-Signale: {np.sum(Cobalt60)}')
print(f'Summe der Europium-Signale: {np.sum(Europium)}')
print(f'Summe der Uranophan-Signale: {np.sum(Uranophan)}')
print(f'Summe der Hintergrund-Signale: {np.sum(Hintergrund)}')


Bin = np.arange(0, len(Caesium))

plt.scatter(Bin, Caesium, label ="Caesium", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Count")
plt.ylabel("Bin")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Caesium.pdf')
plt.clf()


plt.scatter(Bin, Cobalt60, label ="Cobalt60", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Count")
plt.ylabel("Bin")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Cobalt60.pdf')
plt.clf()



plt.scatter(Bin, Europium, label ="Europium", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Count")
plt.ylabel("Bin")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Europium.pdf')
plt.clf()


plt.scatter(Bin, Uranophan, label ="Uranophan", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Count")
plt.ylabel("Bin")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Uranophan.pdf')
plt.clf()


plt.scatter(Bin, Hintergrund, label ="Hintergrund", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Count")
plt.ylabel("Bin")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Hintergrund.pdf')
plt.clf()


