import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import scipy.constants as const
import sympy


#erster Plot für das erste freie Pendel, zumindest versuche ich das zu plotten :)
x,y = np.genfromtxt('l070T15f.txt', unpack=True)

plt.subplot(1, 2, 1)
plt.plot(x, y, label='Kurve')
plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(x, y, label='Kurve')
plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
