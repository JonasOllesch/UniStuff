import matplotlib.pyplot as plt
import numpy as np

#x = np.linspace(0, 10, 1)
x ,werte = np.genfromtxt('l070T15f.txt', unpack =True)
werte *= (1/5)


plt.subplot(1, 2, 1)
plt.plot(x, werte, label='Die Periodendauer eins freien Fadenpendes')
plt.xlabel(r'')
plt.ylabel(r'')
plt.legend(loc='best')

#plt.subplot(1, 2, 2)
#plt.plot(x, z, label='Plot 2')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
