import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.stats import stats

output = ("build/Auswertung")   
my_file = open(output + '.txt', "w") 

def writeW(Wert,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    try:
        for i in Wert:
            my_file.write(str(i))
            my_file.write('\n')
    except:
        my_file.write(str(Wert))
        my_file.write('\n')

    return 0

def wavelength(deltad,z): # deltad ist Spiegelverschiebung um den Abstand d, z ist Zahl der Maxima
    lmda = 2 * deltad/z
    return lmda 

def deltarefrac(b,z,lmda): # b ist Dicke der Messzelle, z wie oben, lmda Literaturwert für Laserwellenlänge
    deltan = (z * lmda)/(2 * b)
    return deltan

def refrac(b,z,lmda,T,T0,p,p0,pstrich):
    deltap = p - pstrich
    n = 1 + deltarefrac(b,z,lmda) * T/T0 * p0/deltap
    return n



deltad, z_wave = np.genfromtxt('messungen/messung1.txt', unpack = True)
pstrich, z_refrac = np.genfromtxt('messungen/messung2.txt', unpack = True)

untersetz = 1/5.017
b = 50 * 10**(-3) # Dicke der Druckzelle in m
p0 = 1.0132 # Atmosphärendruck in bar
T0 = 273.15 # Temperatur in Kelvin bei 0 °C
T  = 298.15 # Raumtemperatur 25 °C
n_luftlit = 1.000292
# pstrich  = 0.666612 # Druck in bar, umgerechnet von mmHg
p = p0 # Nulldruck in Druckzelle ist einfach nur Atmospärendruck
laserlmda = 635 * 10**(-9) # eigentliche Wellenlänge des Lasers

# a)

deltad = deltad * 10**(-3) * untersetz
#print(deltad)

lmda = wavelength(deltad=deltad,z=z_wave)
#print(lmda)

lmdamean = ufloat(np.mean(lmda), stats.sem(lmda)) # stats.sem macht Standardabweichung
#print(lmdamean)

# b)

pstrich = pstrich * 0.001333224 # Umrechnung in bar

#print(pstrich)

n_luft = refrac(b,z_refrac,laserlmda,T,T0,p,p0,pstrich)

#print(n_luft)

n_luftmean = ufloat(np.mean(n_luft), stats.sem(n_luft))

#print(n_luftmean)

np.savetxt('build/wavelength.txt', np.column_stack([deltad * 10**(5), z_wave, lmda * 10**9]), fmt='%.2f', delimiter = '  &  ', header= 'Verschiebung d (mit Untersetzungsfaktor in 10mm), z, Wellenlänge in nm')
np.savetxt('build/refractionindex.txt', np.column_stack([pstrich, z_refrac,n_luft]), fmt='%.6f', delimiter = '          &           ', header = 'Druckdifferenz in bar, z, Brechungsindex')

writeW(lmdamean*10**9, 'Gemittelte Wellenlänge in nm mit Abweichung:')
writeW(abs((laserlmda - lmdamean)/lmdamean), 'Abweichung der berechneten Laserwellenlänge von der Sollwellenlänge lambda = 635 nm:')
writeW(n_luftmean, 'Gemittelter Brechungsindex mit Abweichung:')
writeW(abs((n_luftlit - n_luftmean)/n_luftlit), 'Abweichung des berechneten Brechungsindex vom Literaturwert n = 1.000292:')
