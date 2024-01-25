import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.integrate import quad

def linReg(x,a,b):
    return a*x +b


def berechne_Raumwinkel(a, r):
    return 2*np.pi*(1 -a/np.sqrt(a**2 + r**2))

def berechne_Detektoreffizienz(Linieninhalt, Aktivität, Messzeit, Raumwinkel, Emissionswahrscheinlichkeit):
    return 4*np.pi*Linieninhalt/(Aktivität * Emissionswahrscheinlichkeit * Messzeit * Raumwinkel)


def berechne_Detektoreffizienz_Regression(Energie, alpha, beta):
    return alpha*Energie**beta

def Gaus(x, a, mu, sigma):
    return a/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2) )

def uGaus(x, a, mu, sigma):
    return a/(sigma * unp.sqrt(2 * np.pi)) *unp.exp( - (x - mu)**2 / (2 * sigma**2) )


def berechne_Comptonkante(Photopeak_Energie):
    c = 1#in natürlichen Einheiten
    m_e = 0.51099895*1000
    epsilon = Photopeak_Energie/(c**2*m_e) 
    return Photopeak_Energie*(2*epsilon)/(1+2*epsilon)

def berechne_Rückstreupeak(Photopeak_Energie):
    c = 1
    m_e = 0.51099895*1000
    epsilon = Photopeak_Energie/(c**2*m_e) 
    return Photopeak_Energie/(1+2*epsilon)


def ComptonVerlauf(Energie, a):
    Photopeak_Energie = 661.6553
    t = Energie/Photopeak_Energie#
    c = 1
    m_e = 0.51099895*1000
    epsilon = Photopeak_Energie/(c**2*m_e) 
    ErsterTerm = 2
    ZweiterTerm = t**2/(epsilon**2 * (1-t)**2)
    DritterTerm = t/(1-t) * (t - 2/epsilon)
    return a*(ErsterTerm + ZweiterTerm + DritterTerm)



def IntComptonVerlauf(Energie):
    a = 3.755
    Photopeak_Energie = 661.6553
    t = Energie/Photopeak_Energie#
    c = 1
    m_e = 0.51099895*1000
    epsilon = Photopeak_Energie/(c**2*m_e) 
    ErsterTerm = 2
    ZweiterTerm = t**2/(epsilon**2 * (1-t)**2)
    DritterTerm = t/(1-t) * (t - 2/epsilon)
    return a*(ErsterTerm + ZweiterTerm + DritterTerm)

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
        self.LineContentN = None
        self.LineContentU = None
        self.LineContent = None
        self.PeakEnergie = None
        self.PeakEffizienz = None
        self.Zerfallskonstante = None
        self.Emissionswahrscheinlichkeit = None

        self.Photopeak_Energie_Theorie = None
        self.Photopeak_Energie_Experiment = None
#        self.Comptonkante = None
#        self.Rückstreupeak  = None

#        self.Effektive_Signale = 0

        
HCaesium = np.genfromtxt('Messdaten/Caesium.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HCobalt = np.genfromtxt('Messdaten/Cobalt-60.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HEuropium = np.genfromtxt('Messdaten/Europium.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HUranophan = np.genfromtxt('Messdaten/Uranophan.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 
HHintergrund = np.genfromtxt('Messdaten/Hintergrund.Spe', skip_header = 12, skip_footer = 15, encoding = 'unicode-escape') 


print(f'Summe der Caesium-Signale mit Hintergrund: {np.sum(HCaesium)}')
print(f'Summe der Cobalt60-Signale mit Hintergrund: {np.sum(HCobalt)}')
print(f'Summe der Europium-Signale mit Hintergrund: {np.sum(HEuropium)}')
print(f'Summe der Uranophan-Signale mit Hintergrund: {np.sum(HUranophan)}')
print(f'Summe der Hintergrund-Signale mit Hintergrund: {np.sum(HHintergrund)}')

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
print(np.sum(Uranophan))

#Globalen Hintergrund abziehen


Caesium.Daten = HCaesium - HHintergrund*Caesium.Messzeit/Hintergrund.Messzeit 
Cobalt.Daten = HCobalt - HHintergrund*Cobalt.Messzeit/Hintergrund.Messzeit 
Europium.Daten = HEuropium - HHintergrund*Europium.Messzeit/Hintergrund.Messzeit 
Uranophan.Daten = HUranophan - HHintergrund*Uranophan.Messzeit/Hintergrund.Messzeit

#Energiekalibrierung

#Europium.Peaks, _ = find_peaks(Europium.Daten,  width=8, rel_height=0.34, height=10, prominence=2)
Europium.Peaks, _ = find_peaks(np.log(Europium.Daten+1), prominence=1.6, width=8, rel_height=0.34, height=3)

print(f'Europium Peaks Position {Europium.Peaks}')
print(f'Europium Unterstrich:{_}')

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

Europium.Peaks_widths = _['widths']
Europium.LineContentN = np.zeros(len(Europium.Peaks_widths))
Europium.LineContentU = np.zeros(len(Europium.Peaks_widths))

for i in range(0,len(Europium.Peaks)):
    LinRegOffset = 10


    UnterGrenzeP = Europium.Peaks[i] -int(Europium.Peaks_widths[i])
    ObereGrenzeP = Europium.Peaks[i] +int(Europium.Peaks_widths[i])
    
    UnterGrenzeF = UnterGrenzeP -LinRegOffset
    ObereGrenzeF = ObereGrenzeP +LinRegOffset 



    FitDataX = np.append(Bin[UnterGrenzeF :UnterGrenzeP], Bin[ObereGrenzeP :ObereGrenzeF])
    FitDataY = np.append(Europium.Daten[UnterGrenzeF :UnterGrenzeP], Europium.Daten[ObereGrenzeP :ObereGrenzeF])
    

    para, pcov = curve_fit(linReg, FitDataX, FitDataY)
    HintergrundF = linReg(FitDataX, *para)
    HintergrundP = linReg(Bin[UnterGrenzeP :ObereGrenzeP], *para)

#    print(HintergrundP)
#    print(Europium.Daten[UnterGrenzeP: ObereGrenzeP])
#    print(Bin[UnterGrenzeP :ObereGrenzeP])
#

#    plt.scatter(Bin[UnterGrenzeP :ObereGrenzeP], Europium.Daten[UnterGrenzeP: ObereGrenzeP], label = f"{i}. Peak", c = "midnightblue", marker='x', s = 10)
#    plt.scatter(FitDataX, FitDataY, label = f"{i}. Linfit?", c = "red", marker='x', s = 10)
#    plt.plot(FitDataX, HintergrundF, label='Hintergrund', color='r')
##    plt.yscale('log')
#    plt.xlabel("Kanal")
#    plt.ylabel("Signale")
#    plt.grid(linestyle = ":")
#    plt.tight_layout()
#    plt.legend()
#    plt.savefig(f'build/Europium_Peak{i}mit_Hintergrund.pdf')
#    plt.clf()

    Europium.LineContentN[i] = np.sum(np.abs(Europium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP))
    Europium.LineContentU[i] = np.sum(np.sqrt(np.abs(Europium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP)))


#    plt.scatter(Bin[UnterGrenzeP :ObereGrenzeP], Europium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP, label = f"{i}. Peak", c = "midnightblue", marker='x', s = 10)
#    plt.scatter(FitDataX, FitDataY-HintergrundF, label = f"{i}. Linfit?", c = "red", marker='x', s = 10)
#
#
##    plt.yscale('log')
#    plt.xlabel("Kanal")
#    plt.ylabel("Signale")
#    plt.grid(linestyle = ":")
#    plt.tight_layout()
#    plt.legend()
#    plt.savefig(f'build/Europium_Peak{i}ohne_Hintergrund.pdf')
#    plt.clf()
    
Europium.LineContent = unp.uarray(Europium.LineContentN , Europium.LineContentU)
print(f'Europium LineContent {Europium.LineContent}')
print(f'Summe der Europiums ohne Hintergrund {sum(Europium.Daten)}')

#Lineare Regression, um die Energie der anderen Kanäle zu bestimmen
Europium.PeakEnergie = np.array([121.7817, 244.6974, 344.2785, 367.7891, 411.1165, 443.965, 778.9045, 867.380, 964.079, 1085.837, 1112.076, 1408.013])
Europium.Emissionswahrscheinlichkeit = np.array([ufloat(28.41,13), ufloat(7.55,4), ufloat(26.59,12), ufloat(0.862,5), ufloat(2.238,10), ufloat(2.80,2), ufloat(12.97,6), ufloat(4.243,23), ufloat(14.50,6), ufloat(10.13,6), ufloat(13.41,6), ufloat(20.85,8)])/100




popt_EK, pcov_EK = curve_fit(linReg, Europium.Peaks[1:], Europium.PeakEnergie)
para_EK = correlated_values(popt_EK, pcov_EK)
print(f'Die Parameter der linearen Regression zwischen Energie und Kanälen {para_EK}')


x = np.linspace(0, 8000, 2)
y = linReg(x, *unp.nominal_values(para_EK))

plt.plot(x, y, label = "Lineare Regression", color = 'midnightblue')
plt.scatter(Europium.Peaks[1:], Europium.PeakEnergie, label ="Energie der Peaks", c = 'firebrick',marker='x', s = 10)


plt.xlabel("Kanal")
plt.ylabel(r'$E \mathbin{/} \unit{\kilo\eV}$')
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/EnergieKanäle.pdf')
plt.clf()

del x
del y


#Berechnung der Detektoreffizienz 
#Berechnung des Raumwinkels

Raumwinkel = berechne_Raumwinkel(0.086, 0.0225) 
print(f'Der gegebene Raumwinkel {Raumwinkel}')

A_0 = ufloat(4130, 60)
Europium.Zerfallskonstante = ufloat(1.6244e-9, 0.0019e-9)

A_t = A_0 * unp.exp(-Europium.Zerfallskonstante* 754963200)
print(f'Die Aktivität am Messtag {repr(A_t)}')
Europium.PeakEffizienz = berechne_Detektoreffizienz(Linieninhalt=Europium.LineContent[1:], Aktivität=A_t, Messzeit=Europium.Messzeit, Raumwinkel=Raumwinkel, Emissionswahrscheinlichkeit=Europium.Emissionswahrscheinlichkeit)
print(f'Die Detektoreffizienz an den Peaks in Prozent {Europium.PeakEffizienz*100}')


popt_QE, pcov_QE = curve_fit(berechne_Detektoreffizienz_Regression, unp.nominal_values(Europium.PeakEnergie), unp.nominal_values(Europium.PeakEffizienz))
para_QE = correlated_values(popt_QE, pcov_QE)
print(f'Die Parameter der exponentiellen Regression zwischen Effizienz und Energie {para_QE}')
x = np.linspace(50, 1500, 2000)
y = berechne_Detektoreffizienz_Regression(x, *unp.nominal_values(para_QE))

plt.plot(x,y, label = 'Regression', color = 'midnightblue')
plt.scatter(unp.nominal_values(Europium.PeakEnergie), unp.nominal_values(Europium.PeakEffizienz), label ="Detektoreffizenz", c = 'firebrick', marker='x', s = 8)
plt.xlabel(r"$E \mathbin{/} \unit{\kilo\eV}$")
plt.ylabel(r"$Q$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/EffizienzKanal.pdf')
plt.clf()

del x
del y

#Untersuchung es monochromatisches Gammastrahlers 
Bin = np.arange(0, len(HCaesium))

Caesium.Peaks, _ = find_peaks(np.log(Caesium.Daten+1), prominence=1.6, width=8, rel_height=0.34, height=3)

print(f'Caesium Peaks Position {Caesium.Peaks}')
print(f'Caesium Unterstrich:{_}')



plt.scatter(Bin, Caesium.Daten, label ="Caesium", c = 'midnightblue',marker='x', s = 5)
plt.scatter(Caesium.Peaks, Caesium.Daten[Caesium.Peaks], label = "Peaks", c = "firebrick", marker='x', s = 10)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.ylim(bottom=1)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Caesium.pdf')
plt.clf()

#Untersuchung des Photopeaks
print(f'Energie des Photopeaks {repr(linReg(Caesium.Peaks[-1],*para_EK))}')


Caesium.Peaks_widths = _['widths']
Caesium.LineContentN = np.zeros(len(Caesium.Peaks_widths))
Caesium.LineContentU = np.zeros(len(Caesium.Peaks_widths))




Caesium.Peaks_widths = _['widths']
Caesium.LineContentN = np.zeros(len(Caesium.Peaks_widths))
Caesium.LineContentU = np.zeros(len(Caesium.Peaks_widths))


LinRegOffset = 10
UnterGrenzeP = Caesium.Peaks[-1] -int(Caesium.Peaks_widths[-1])
ObereGrenzeP = Caesium.Peaks[-1] +int(Caesium.Peaks_widths[-1])

UnterGrenzeF = UnterGrenzeP -LinRegOffset
ObereGrenzeF = ObereGrenzeP +LinRegOffset 



FitDataX = np.append(Bin[UnterGrenzeF :UnterGrenzeP], Bin[ObereGrenzeP :ObereGrenzeF])
FitDataY = np.append(Caesium.Daten[UnterGrenzeF :UnterGrenzeP], Caesium.Daten[ObereGrenzeP :ObereGrenzeF])

para, pcov = curve_fit(linReg, FitDataX, FitDataY)
HintergrundF = linReg(FitDataX, *para)
HintergrundP = linReg(Bin[UnterGrenzeP :ObereGrenzeP], *para)
Caesium.LineContentN[-1] = np.sum(np.abs(Caesium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP))
Caesium.LineContentU[-1] = np.sum(np.sqrt(np.abs(Caesium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP)))
    
Caesium.LineContent = unp.uarray(Caesium.LineContentN , Caesium.LineContentU)
print(f'Line content vom Caesium {Caesium.LineContent}')

Caesium_Photopeak_Bins_in_Energie = unp.nominal_values(linReg(Bin[UnterGrenzeP :ObereGrenzeP], *para_EK))
Caesium_Photopeak_Signal_ohne_Hintergrund = Caesium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP
Caesium_Photopeak_Hintergrund_Bin_in_Energie = unp.nominal_values(linReg(FitDataX, *para_EK))

popt_GA, pcov_GA = curve_fit(Gaus,unp.nominal_values(Caesium_Photopeak_Bins_in_Energie), unp.nominal_values(Caesium_Photopeak_Signal_ohne_Hintergrund), absolute_sigma=True, p0=[2000, 661.5, 1])
para_GA = correlated_values(popt_GA, pcov_GA)
print(f'Parameter des Gaußfits {para_GA}')

GausFitx = unp.nominal_values(np.linspace(np.min(Caesium_Photopeak_Bins_in_Energie), np.max(Caesium_Photopeak_Bins_in_Energie), 10000))
GausFity = Gaus(GausFitx, *unp.nominal_values(para_GA))
#Bestimmung der halben Höhe und der zehntel Höhe


tmp = np.array([np.argmax(unp.nominal_values(GausFity))])
Caesium.Photopeak_Energie_Experiment = ufloat(GausFitx[np.argmax(GausFity)], unp.std_devs(uGaus(GausFitx[np.argmax(GausFity)], *para_GA)))

print(tmp)
Caesium_FWHM, Hh ,Hl, Hr = peak_widths(unp.nominal_values(GausFity), tmp, rel_height=0.5)
Caesium_FWZM, Zh ,Zl, Zr = peak_widths(unp.nominal_values(GausFity), tmp, rel_height=0.9)
Caesium_FWHM_Rand = np.array([int(Hl), int(Hr)])
Caesium_FWZM_Rand = np.array([int(Zl), int(Zr)])


plt.plot(GausFitx[Caesium_FWHM_Rand], GausFity[Caesium_FWHM_Rand], label ='FWHM', color = 'lightseagreen')
plt.plot(GausFitx[Caesium_FWZM_Rand], GausFity[Caesium_FWZM_Rand], label ='FWZM', color = 'darkcyan')
print(f'Caesium FWHM {GausFitx[Caesium_FWHM_Rand][1] -GausFitx[Caesium_FWHM_Rand][0]}')
print(f'Caesium FWZM {GausFitx[Caesium_FWZM_Rand][1] -GausFitx[Caesium_FWZM_Rand][0]}')


plt.plot(GausFitx, unp.nominal_values(GausFity), label ='Gaußfit', color = 'forestgreen')
plt.scatter(Caesium_Photopeak_Bins_in_Energie, Caesium.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP, label = f"Photopeak", c = "midnightblue", marker='x', s = 10)
plt.scatter(Caesium_Photopeak_Hintergrund_Bin_in_Energie, FitDataY-HintergrundF, label = f"Hintergrund", c = "firebrick", marker='x', s = 10)
plt.xlabel(r"$E \mathbin{/} \unit{\kilo\eV}$")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig(f'build/Caesium_Photopeak.pdf')
plt.clf()

#Theoretische Berechung vom Rückstreupeak und Comptonkante

Caesium.Photopeak_Energie_Theorie = ufloat(661.6553,0.030)
print(f'Theoretische Photopeak Cs137 {Caesium.Photopeak_Energie_Theorie}')
print(f'Die theoretische Comptonkante von Cs137 {berechne_Comptonkante(Caesium.Photopeak_Energie_Theorie)} in keV')
print(f'Der theoretische Rückstreupeak von Cs137 {berechne_Rückstreupeak(Caesium.Photopeak_Energie_Theorie)} in keV')


print(f'Experiemnteller Photopeak Cs137 {repr(Caesium.Photopeak_Energie_Experiment)}')
print(f'Die Experimentelle Comptonkante von Cs137 {repr(berechne_Comptonkante(Caesium.Photopeak_Energie_Experiment))} in keV')
print(f'Der Experimentelle Rückstreupeak von Cs137 {repr(berechne_Rückstreupeak(Caesium.Photopeak_Energie_Experiment))} in keV')


#Das ComptonKontinumum

BinE = unp.nominal_values(linReg(Bin,*para_EK))
UnterGrenzeComptonFitB = 1100 #Bin
ObereGrenzeComptonFitB = 2550 #Bin
UnterGrenzeComptonFitE = unp.nominal_values(linReg(1100 ,*para_EK))
ObereGrenzeComptonFitE = unp.nominal_values(linReg(2550 ,*para_EK))
ComptonkonFitx = np.linspace(0, 510 ,1000)
print(f'UnterGrenzeComptonFitE: {UnterGrenzeComptonFitE}')
print(f'ObereGrenzeComptonFitE: {ObereGrenzeComptonFitE}')

plt.scatter(BinE[:3000], Caesium.Daten[:3000], label ="Caesium", c = 'midnightblue',marker='x', s = 5)
plt.scatter(BinE[1100:2550], Caesium.Daten[1100:2550], c = 'firebrick',marker='x', s = 5)

popt_CK, pcov_CK = curve_fit(ComptonVerlauf, BinE[1100:2550], Caesium.Daten[1100:2550])
para_CK = correlated_values(popt_CK, pcov_CK)
print(f'Die Parameter des Comptonkontinuumsfit {para_CK}')
ComptonkonFity = ComptonVerlauf(ComptonkonFitx, unp.nominal_values(*para_CK))

#Berechnung des Linieninhalt des Comptonkontinuums
print(f'Linieninhalt des Caesium-Comptonkontinuums {quad(IntComptonVerlauf, 0, ObereGrenzeComptonFitE)}')

#
plt.plot(ComptonkonFitx, ComptonkonFity, color = 'firebrick', label='Fit')


plt.yscale('log')
plt.xlim(right = 655, left = 0)
#plt.xlim(left = 0)

plt.ylim(bottom=1)
plt.xlabel(r"$E \mathbin{/} \unit{\kilo\eV}$")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/CaesiumE.pdf')
plt.clf()





#Bestimmung der Aktivität von Co-60
Cobalt.Peaks, _ = find_peaks(np.log(Cobalt.Daten+1), prominence=1.6, width=8, rel_height=0.34, height=3)

print(f'Cobalt Peaks Position {Cobalt.Peaks}')
print(f'Cobalt Unterstrich:{_}')


Cobalt.Peaks_widths = _['widths']
Cobalt.LineContentN = np.zeros(len(Cobalt.Peaks_widths))
Cobalt.LineContentU = np.zeros(len(Cobalt.Peaks_widths))

for i in range(0,len(Cobalt.Peaks)):
    LinRegOffset = 10

    UnterGrenzeP = Cobalt.Peaks[i] -int(Cobalt.Peaks_widths[i])
    ObereGrenzeP = Cobalt.Peaks[i] +int(Cobalt.Peaks_widths[i])
    
    UnterGrenzeF = UnterGrenzeP -LinRegOffset
    ObereGrenzeF = ObereGrenzeP +LinRegOffset 


    FitDataX = np.append(Bin[UnterGrenzeF :UnterGrenzeP], Bin[ObereGrenzeP :ObereGrenzeF])
    FitDataY = np.append(Cobalt.Daten[UnterGrenzeF :UnterGrenzeP], Cobalt.Daten[ObereGrenzeP :ObereGrenzeF])
    

    para, pcov = curve_fit(linReg, FitDataX, FitDataY)
    HintergrundF = linReg(FitDataX, *para)
    HintergrundP = linReg(Bin[UnterGrenzeP :ObereGrenzeP], *para)


    Cobalt.LineContentN[i] = np.sum(np.abs(Cobalt.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP))
    Cobalt.LineContentU[i] = np.sum(np.sqrt(np.abs(Cobalt.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP)))

    
Cobalt.LineContent = unp.uarray(Cobalt.LineContentN , Cobalt.LineContentU)
print(f'Cobalt LineContent {Cobalt.LineContent}')
print(f'Summe der Cobalts ohne Hintergrund {sum(Cobalt.Daten)}')

Cobalt.PeakEnergie = linReg(Cobalt.Peaks, *para_EK)
Cobalt.Emissionswahrscheinlichkeit = np.array([ufloat(0.999826, 0.000006), ufloat(0.9985, 0.0003)])
print(f'Die PeakEnergie von Cobalt {Cobalt.PeakEnergie}')

Cobalt.PeakEffizienz = berechne_Detektoreffizienz_Regression(Cobalt.PeakEnergie, *para_QE)
print(f'Aktivität von Cobalt60 {4*np.pi*Cobalt.LineContent/(Cobalt.PeakEffizienz * Cobalt.Emissionswahrscheinlichkeit * Cobalt.Messzeit * Raumwinkel)}')
CobaltAktivität = 4*np.pi*Cobalt.LineContent/(Cobalt.PeakEffizienz * Cobalt.Emissionswahrscheinlichkeit * Cobalt.Messzeit * Raumwinkel)


plt.scatter(Bin, Cobalt.Daten, label ="Cobalt", c = 'midnightblue',marker='x', s = 5)
plt.scatter(Cobalt.Peaks, Cobalt.Daten[Cobalt.Peaks], label = "Peaks", c = "firebrick", marker='x', s = 10)

plt.ylim(bottom = 1)
plt.yscale('log')
plt.xlabel("Kanal")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Cobalt60.pdf')
plt.clf()



Uranophan.Peaks, _ = find_peaks(np.log(Uranophan.Daten+1), prominence=1.3, width=5, rel_height=0.34, height=3)
Uranophan.PeakEnergie = linReg(Uranophan.Peaks, *para_EK)


print(f'Uranophan Peaks Position {Uranophan.Peaks}')
print(f'Uranophan Unterstrich:{_}')

for i in range(0,len(Uranophan.Peaks)):
    print(f'{i}, {Uranophan.Peaks[i]}, {Uranophan.PeakEnergie[i]}')







#Uranophan
Uranophan.Peaks_widths = _['widths']
Uranophan.LineContentN = np.zeros(len(Uranophan.Peaks_widths))
Uranophan.LineContentU = np.zeros(len(Uranophan.Peaks_widths))

for i in range(0,len(Uranophan.Peaks)):
    LinRegOffset = 10

    UnterGrenzeP = Uranophan.Peaks[i] -int(Uranophan.Peaks_widths[i])
    ObereGrenzeP = Uranophan.Peaks[i] +int(Uranophan.Peaks_widths[i])
    
    UnterGrenzeF = UnterGrenzeP -LinRegOffset
    ObereGrenzeF = ObereGrenzeP +LinRegOffset 

    FitDataX = np.append(Bin[UnterGrenzeF :UnterGrenzeP], Bin[ObereGrenzeP :ObereGrenzeF])
    FitDataY = np.append(Uranophan.Daten[UnterGrenzeF :UnterGrenzeP], Uranophan.Daten[ObereGrenzeP :ObereGrenzeF])
    
    para, pcov = curve_fit(linReg, FitDataX, FitDataY)
    HintergrundF = linReg(FitDataX, *para)
    HintergrundP = linReg(Bin[UnterGrenzeP :ObereGrenzeP], *para)

    Uranophan.LineContentN[i] = np.sum(np.abs(Uranophan.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP))
    Uranophan.LineContentU[i] = np.sum(np.sqrt(np.abs(Uranophan.Daten[UnterGrenzeP: ObereGrenzeP]-HintergrundP)))

    
Uranophan.LineContent = unp.uarray(Uranophan.LineContentN , Uranophan.LineContentU)
print(f'Uranophan LineContent {Uranophan.LineContent}')


#Uranophan.PeakEnergie = linReg(Uranophan.Peaks, *para_EK)
Uranophan.Emissionswahrscheinlichkeit = np.array([ufloat(np.nan,np.nan ), ufloat(4.33, 0.29), ufloat(1.78,0.19), ufloat(7.268,0.022), ufloat(18.414,0.036), ufloat(35.60,0.07 ), ufloat(45.49,0.19), ufloat(1.530,0.007 ), ufloat(np.nan,np.nan ), ufloat(np.nan,np.nan ), ufloat(4.892,0.016), ufloat(1.064,0.013), ufloat(2.5,0.3 ), ufloat(3.10,0.01 ), ufloat(np.nan,np.nan ), ufloat(14.91,0.03 ), ufloat(1.635,0.007 ), ufloat(np.nan,np.nan ), ufloat(5.831,0.014), ufloat(1.435,0.06 ), ufloat(np.nan,np.nan ), ufloat(3.968,0.011), ufloat(np.nan,np.nan ), ufloat(2.389,0.008 ), ufloat(2.128,0.010), ufloat(np.nan,np.nan ), ufloat(np.nan,np.nan )])/100
#print(f'Die PeakEnergie von Uranophan {Uranophan.PeakEnergie}')
print(f'Die Emsissionswahrschinlichkeiten von Uranophan {Uranophan.Emissionswahrscheinlichkeit}')
Uranophan.PeakEffizienz = berechne_Detektoreffizienz_Regression(Uranophan.PeakEnergie, *para_QE)
#print(f'Aktivität von Uranophan {}')
print(f'Die Detektoreffizienz an den Uranophanpeaks{Uranophan.PeakEffizienz}') 
Aktivität = 4*np.pi*Uranophan.LineContent/(Uranophan.PeakEffizienz * Uranophan.Emissionswahrscheinlichkeit * Uranophan.Messzeit * Raumwinkel)
print(f'Die Aktivität von Uranophan {Aktivität}')

Sonstiges = np.array([0,8,9,14,17,20,22,25,26])
Th = np.array([1])
Ra = np.array([2])
Pb = np.array([3,4,5,11])
Bi = np.array([6,7,10,11,13,15,16,18,19,21,23,24])
Pa = np.array([12])
plt.scatter(BinE, Uranophan.Daten, label ="Uranophan", c = 'midnightblue',marker='x', s = 5)



plt.scatter(unp.nominal_values(Uranophan.PeakEnergie[Sonstiges]), unp.nominal_values(Uranophan.Daten[Uranophan.Peaks[Sonstiges]]), label = "Sonstige", c = "firebrick", marker='x', s = 10)
plt.scatter(unp.nominal_values(Uranophan.PeakEnergie[Th]), unp.nominal_values(Uranophan.Daten[Uranophan.Peaks[Th]]), label = "Th-234", c = "sienna", marker='x', s = 10)
plt.scatter(unp.nominal_values(Uranophan.PeakEnergie[Ra]), unp.nominal_values(Uranophan.Daten[Uranophan.Peaks[Ra]]), label = "Ra-226", c = "yellow", marker='x', s = 10)
plt.scatter(unp.nominal_values(Uranophan.PeakEnergie[Pb]), unp.nominal_values(Uranophan.Daten[Uranophan.Peaks[Pb]]), label = "Pb-214", c = "forestgreen", marker='x', s = 10)
plt.scatter(unp.nominal_values(Uranophan.PeakEnergie[Bi]), unp.nominal_values(Uranophan.Daten[Uranophan.Peaks[Bi]]), label = "Bi-214", c = "cyan", marker='x', s = 10)
plt.scatter(unp.nominal_values(Uranophan.PeakEnergie[Pa]), unp.nominal_values(Uranophan.Daten[Uranophan.Peaks[Pa]]), label = "Pa-234", c = "purple", marker='x', s = 10)

plt.yscale('log')
plt.ylim(bottom = 1)
plt.xlabel(r"$E \mathbin{/} \unit{\kilo\eV}$")
plt.ylabel("Signale")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Uranophan.pdf')
plt.clf()

print(Aktivität[Th])

print(f'die Aktivitäten von Th {np.mean(unp.nominal_values(Aktivität[Th]))} \pm {(unp.std_devs(Aktivität[Th]))}')
print(f'die Aktivitäten von Ra {np.mean(unp.nominal_values(Aktivität[Ra]))} \pm {(unp.std_devs(Aktivität[Ra]))}')
print(f'die Aktivitäten von Pb {np.mean(unp.nominal_values(Aktivität[Pb]))} \pm {np.std(unp.nominal_values(Aktivität[Pb]))}')
print(f'die Aktivitäten von Bi {np.mean(unp.nominal_values(Aktivität[Bi]))} \pm {np.std(unp.nominal_values(Aktivität[Bi]))}')
print(f'die Aktivitäten von Pa {np.mean(unp.nominal_values(Aktivität[Pa]))} \pm {(unp.std_devs(Aktivität[Pa]))}')




