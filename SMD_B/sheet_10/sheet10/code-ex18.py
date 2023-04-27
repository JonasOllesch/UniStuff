import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lombscargle
from scipy.interpolate import interp1d
from scipy.fft import rfftfreq, rfft,irfft


#0)
column_names = ["Date", "Time", "Measurement", "Temperature"]

tempdata = pd.read_csv("temperatures_dortmund.csv", names=column_names, sep=",", skiprows=1)

#b)
Measurement = tempdata["Measurement"].to_numpy()
Temperature = tempdata["Temperature"].to_numpy()

#we kill everything that is broken
mask = np.where(Measurement < 2009)
M_scarg_1 = Measurement[mask] 
T_scarg_1 = Temperature[mask]
mask2 = np.where(~np.isnan(T_scarg_1))
M_scarg_2 = M_scarg_1[mask2]
T_scarg_2 = T_scarg_1[mask2]
freq_max = 400

freq = np.linspace(0.001, freq_max*2*np.pi, 10000)
lomb_scarg = lombscargle(np.array(M_scarg_2), np.array(T_scarg_2), freq, normalize=True)
plt.plot(freq/(2*np.pi), lomb_scarg,"x",ms=1, c='blue')
plt.xlim(0, freq_max)
plt.xlabel('frequency')
plt.ylabel("amplitude")
plt.legend(loc='best')
plt.tight_layout()
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
frequencies = rfftfreq(np.size(gridded_x),1)


plt.plot((frequencies*2*np.pi), np.abs(A_signal_gridded), 'r-', label='frequencies')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plots/furie.pdf')
plt.clf()


#e)

f_two_max = np.zeros(2)
A_signal_gridded_tmp = np.copy(A_signal_gridded)

index_max=np.argmax(abs(A_signal_gridded_tmp))
f_two_max[0]= A_signal_gridded_tmp[index_max]
A_signal_gridded_tmp[index_max] = 0
index_max=np.argmax(abs(A_signal_gridded_tmp))
f_two_max[1]= A_signal_gridded_tmp[index_max]


signal_two_max = irfft(f_two_max,len(M_scarg_2))


plt.plot(M_scarg_2, T_scarg_2, label="original data",c='blue')
plt.plot(gridded_x, signal_two_max, label="inverse Fourier trafo",color='red')
plt.xlabel("Measurement")
plt.ylabel("Temperature")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("plots/e.pdf")
plt.clf()