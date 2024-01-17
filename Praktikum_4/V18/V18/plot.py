import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants
from scipy.signal import find_peaks

#from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)
#from multiprocessing  import Process
class Messreihe:
    def __init__(self, name, Gesamt_Signale, Messzeit):
        self.name = name
        self.Gesamt_Signale = Gesamt_Signale
        self.Messzeit = Messzeit
        self.Daten  = 0
        self.Peaks = None
        self.Peaks_widths = None
#        self.Effektive_Signale = 0

        
HCaesium = np.genfromtxt('Messdaten/Caesium.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HCobalt = np.genfromtxt('Messdaten/Cobalt-60.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HEuropium = np.genfromtxt('Messdaten/Europium.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HUranophan = np.genfromtxt('Messdaten/Uranophan.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HHintergrund = np.genfromtxt('Messdaten/Hintergrund.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 


print(f'Summe der Caesium-Signale: {np.sum(HCaesium)}')
print(f'Summe der Cobalt60-Signale: {np.sum(HCobalt)}')
print(f'Summe der Europium-Signale: {np.sum(HEuropium)}')
print(f'Summe der Uranophan-Signale: {np.sum(HUranophan)}')
print(f'Summe der Hintergrund-Signale: {np.sum(HHintergrund)}')

Bin = np.arange(0, len(HCaesium))

plt.scatter(Bin, HHintergrund, label ="Hintergrund", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Hintergrund.pdf')
plt.clf()

HHintergrund[8000:] = 0
HCaesium[8000:] = 0
HCobalt[8000:] = 0
HEuropium[8000:] = 0
HUranophan[8000:] = 0

Caesium = Messreihe("Caesium", np.sum(HCaesium), 2706)
Cobalt = Messreihe("Cobalt", np.sum(HCobalt), 4324)
Europium = Messreihe("Europium", np.sum(HEuropium), 2836)
Uranophan = Messreihe("Uranophan", np.sum(HUranophan), 2737)
Hintergrund = Messreihe("Hintergrund", np.sum(HHintergrund), 86960)


#Globalen Hintergrund abziehen



Caesium.Daten = HCaesium - HHintergrund*Caesium.Messzeit/Hintergrund.Messzeit 
Cobalt.Daten = HCobalt - HHintergrund*Cobalt.Messzeit/Hintergrund.Messzeit 
Europium.Daten = HEuropium - HHintergrund*Europium.Messzeit/Hintergrund.Messzeit 
Uranophan.Daten = HUranophan - HHintergrund*Uranophan.Messzeit/Hintergrund.Messzeit

#Energiekalibrierung

#Europium.Peaks, _ = find_peaks(Europium.Daten,  width=8, rel_height=0.34, height=10, prominence=2)
Europium.Peaks, _ = find_peaks(np.log(Europium.Daten+1), prominence=1.6, width=8, rel_height=0.34, height=3)

print(f'Peaks Position {Europium.Peaks}')
print(f'Unterstrich:{_}')

plt.scatter(Bin, Europium.Daten, label ="Europium", c = 'midnightblue',marker='x', s = 5)
print(f'Peaks bei Europium: {Europium.Peaks}')
plt.scatter(Europium.Peaks, Europium.Daten[Europium.Peaks], label = "Peaks", c = "firebrick", marker='x', s = 10)

plt.ylim(bottom=1)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Europium.pdf')
plt.clf()


#Europium line Content

Europium.Peaks_widths = _['width_heights']

for i in range(0,len(Europium.Peaks)):
    UnterGrenze = Europium.Peaks[i] -int(Europium.Peaks_widths[i]*3)
    ObereGrenze = Europium.Peaks[i] +int(Europium.Peaks_widths[i]*3) 
    plt.scatter(Bin[UnterGrenze :ObereGrenze], Europium.Daten[UnterGrenze: ObereGrenze], label = f"{i}. Peak", c = "midnightblue", marker='x', s = 10)
    plt.xlabel("Kanal")
    plt.ylabel("Signale")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'build/Europium_Peak{i}.pdf')
    plt.clf()




Bin = np.arange(0, len(HCaesium))
#plt.scatter(Bin, HHintergrund, label ="Hintergrund", c = 'midnightblue',marker='x', s = 5)
#plt.yscale('log')
#plt.xlabel("Kanal")
#plt.ylabel("Signale")
#plt.grid(linestyle = ":")
#plt.tight_layout()
#plt.legend()
#plt.savefig('build/Hintergrund.pdf')
#plt.clf()

plt.scatter(Bin, Caesium.Daten, label ="Caesium", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Caesium.pdf')
plt.clf()


plt.scatter(Bin, Cobalt.Daten, label ="Cobalt", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Cobalt60.pdf')
plt.clf()






plt.scatter(Bin, Uranophan.Daten, label ="Uranophan", c = 'midnightblue',marker='x', s = 5)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Uranophan.pdf')
plt.clf()


