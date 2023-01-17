import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lombscargle

#0)
column_names = ["Date", "Time", "Measurement", "Temperature"]

tempdata = pd.read_csv("temperatures_dortmund.csv", names=column_names, sep=",", skiprows=1)

#b)
Measurement = tempdata["Measurement"].to_numpy()
Temperature = tempdata["Temperature"].to_numpy()

#we kill everything that is broken
mask = np.where(Measurement < 2009)
#Measurement_2 = Measurement(mask)
M_scarg_1 = Measurement[mask] /2000
T_scarg_1 = Temperature[mask]
#print(len(M_scarg_1))
#print(len(T_scarg_1))
mask2 = np.where(~np.isnan(T_scarg_1))
M_scarg_2 = M_scarg_1[mask2]
T_scarg_2 = T_scarg_1[mask2]

freq = np.linspace(0.00001, 365.25*1 +20, 500) 
lomb_scarg = lombscargle(np.array(M_scarg_2), np.array(T_scarg_2), freq, normalize=True)


#plt.figure(figsize=(7, 4))
plt.plot(freq/(365.25*2 +1), lomb_scarg, lw=1.0, c='paleturquoise')
#plt.stem(freq/(365.25*2 +1), lomb_scarg, label=r"$\mathrm{P}(N_{\mathrm{signal}})$")
#plt.axvline(1/11, 0, 1, label=r'$T\,=\,11\,$a', color='black')
plt.xlabel(r'Frequency $f\,/\,\mathrm{a}$')
plt.ylabel(r"$\mathrm{P}(2\pi f)$")
plt.legend(loc='best')
#plt.yscale('log')
#plt.xlim(0, 1)

plt.savefig('lomb_scarg.pdf')
plt.clf()