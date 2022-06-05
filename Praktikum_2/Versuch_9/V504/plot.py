import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

# Die Kennlinien bitte in 'kennlinien.pdf' speichern, sonst muss der Code in auswertung.tex noch geändert werden '^^
# Den Fit (Aufgabenteil b)) für Messreihe 5 als 'linregraumladung.pdf' speichern (oder halt in der Auswertung ändern :D)
# Der Fit zum Anlaufstromgebiet sollte irgendwie 'reganlaufstrom.pdf' heißen :D

output = ("Auswertung")   
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