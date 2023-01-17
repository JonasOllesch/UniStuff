import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lombscargle
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq


#0)
column_names = ["Date", "Time", "Measurement", "Temperature"]

tempdata = pd.read_csv("temperatures_dortmund.csv", names=column_names, sep=",", skiprows=1)

#b)
Measurement = tempdata["Measurement"].to_numpy()
Temperature = tempdata["Temperature"].to_numpy()

#we kill everything that is broken
mask = np.where(Measurement < 2009)
#Measurement_2 = Measurement(mask)
M_scarg_1 = Measurement[mask] 
T_scarg_1 = Temperature[mask]
#print(len(M_scarg_1))
#print(len(T_scarg_1))
mask2 = np.where(~np.isnan(T_scarg_1))
M_scarg_2 = M_scarg_1[mask2]
T_scarg_2 = T_scarg_1[mask2]
freq_max = 400

freq = np.linspace(0.001, freq_max*2*np.pi, 10) 
lomb_scarg = lombscargle(np.array(M_scarg_2), np.array(T_scarg_2), freq, normalize=True)


#plt.figure(figsize=(7, 4))
plt.plot(freq/(2*np.pi), lomb_scarg,"x",ms=1, c='blue')
#plt.stem(freq, lomb_scarg, label=r"$\mathrm{P}(N_{\mathrm{signal}})$")
#plt.axvline(1/11, 0, 1, label=r'$T\,=\,11\,$a', color='black')
plt.xlim(0, freq_max)
plt.xlabel(r'Frequency $f\,/\,\mathrm{a}$')
plt.ylabel(r"$\mathrm{P}(2\pi f)$")
plt.legend(loc='best')
plt.tight_layout()
#plt.yscale('log')
#plt.xlim(0, 1)

plt.savefig('plots/lomb_scarg.pdf')
plt.clf()

#d)
# Resampling function
def resampled_fuction(x, x_orig, y_orig):
    interp = interp1d(x_orig, y_orig)
    return interp(x)

gridded_x = np.linspace(2000,2008.9999620210326, len(M_scarg_2))
tofou = resampled_fuction(gridded_x, M_scarg_2,T_scarg_2)

A_signal_gridded = rfft(tofou)
frequencies = rfftfreq(np.size(gridded_x),1/365)


plt.plot(frequencies*2*np.pi, np.abs(A_signal_gridded), 'r-', label='Gridded data')
plt.xlabel(r"x/$\pi$")
plt.ylabel("y")
plt.legend(loc='upper right', framealpha=0.95)
#plot_amplitude(frequencies*2*np.pi, np.abs(A_signal_gridded), 'r-', label='Gridded data', xlim=(0, 5), ylim=(0, 30))
plt.tight_layout()
plt.savefig('plots/furie.pdf')
plt.clf()
