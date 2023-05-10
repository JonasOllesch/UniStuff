import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import pandas as pd

from scipy import interpolate

output = ("build/Auswertung")   
my_file = open(output + '.txt', "w") 
def writeW(Wert,Beschreibung):
    my_file.write(str(Beschreibung))
    my_file.write('\n')
    try:
        for i in Wert:
            my_file.write(repr(i))
            my_file.write('\n')
    except:
        my_file.write( repr(Wert))
        my_file.write('\n')
    return 0
#plt.rcParams['text.usetex'] = True

def pdf(x,N,lam,y):
    return (N*np.exp(-lam*x)) + y

Channel_Kalibrierung = np.genfromtxt('Messdaten/Channel_Kalibrierung.txt',encoding='unicode-escape')
def pol1(a,b,x):
    return a*x +b 

def exponential(x,N,lam,y):
    return (N*np.exp(-lam*x))+y


popt_channel, pcov_cannel = curve_fit(pol1,Channel_Kalibrierung[:,0],Channel_Kalibrierung[:,1])
#print(popt_channel)
#print(pcov_cannel)
#print(correlated_values(popt_channel, pcov_cannel))

x = np.linspace(0,450,10)
y = pol1(popt_channel[1],popt_channel[0],x)

plt.scatter(Channel_Kalibrierung[:,0],Channel_Kalibrierung[:,1],color = 'blue', marker='x',label='Impulsabstände')
#plt.plot(x,y,label="linearer Fit: (" + str(round(popt_channel[1],2)) + r" \pm \," + str(round(np.sqrt(pcov_cannel[1][1]),2)) + ") "+r" \unit{\second} "+ r"$ \cdot x \,$" + r"$ + ($"+str(round(popt_channel[0],2))+ r" \pm \," +str(round(np.sqrt(pcov_cannel[0][0]),2)) + ")",color="red")
plt.plot(x,y,label="linearer Fit",color="red")
plt.ylabel(r'$\text{Zeit} \, \backslash \, \unit{\micro\second}')
plt.xlabel("Kanalnummer")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Channel_Kalibrierung')
plt.clf()

#das funktioniert leider noch nicht ...
#df = pd.DataFrame({'Course': array[:, 0], 'Fee': array[:, 1], 'Discount': array[:, 2]}
#np.flip(Channel_Kalibrierung,axis=1)
#df = pd.DataFrame({'Kanal':Channel_Kalibrierung[:,0],"Zeit":Channel_Kalibrierung[:,1]})
#my_file = open("build/tab_MCA" + '.tex', "w") 
#my_file.write(df.to_latex(index=False,
#                          caption="Einordnung der Impulsabstände in die Kanäle des MCA",
#                          label="tab:KaliMCA_tab",column_format ="c c"))
#df.rename(columns={"Kanal":"Kanal","Zeit":r"$Zeit \mathbin{/} \unit{\micro\second}$"})
#with open("build/tab_MCA.tex", 'w') as tf:
#     tf.write(df
#                        .rename(columns={"t":r"$Zeit \\mathbin{/} \\unit{\\micro\\second}$"})
#                        .to_latex(index=False,
#                        caption="Einordnung der Impulsabstände in die Kanäle des MCA",
#                        label="tab:KaliMCA_tab",
#                        column_format ="c c"))


Messung = np.genfromtxt('Messdaten/v01.Spe',skip_header=12,skip_footer=15,encoding='unicode-escape')        
gefüllte_Kanäle = Messung[3:]
Channel= np.arange(len(gefüllte_Kanäle))
#writeW(np.sum(Messung),"Anzahl der gemessenen Myonen")
#print("der channel Geraden: " ,popt_channel)
Zeit = (popt_channel[1]*Channel+popt_channel[0])

popt_zeit, pcov_zeit = curve_fit(exponential,Channel,gefüllte_Kanäle,p0=[39,0.014566,0.9])
x_test = np.linspace(0,400,1000)
y_test = pdf(x_test,36.24,0.01256,1.217)

plt.plot(x_test,y_test,color='red')
plt.fill_between(Channel,gefüllte_Kanäle,step="pre",alpha=0.6,label="Histogram der Myonenzerfälle")
plt.ylabel("Zählung")
plt.xlabel("Kanalnummer")
plt.xlim(0,len(gefüllte_Kanäle))
plt.ylim(0,60)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Myonenzerfall_Kanalnummer.pdf')
plt.clf()

popt_zeit, pcov_zeit = curve_fit(exponential,Zeit[:],gefüllte_Kanäle[:],p0=[39,0.014566,0.9])
x_zeit = np.linspace(0,11,1000)
y_zeit = pdf(x_zeit,*popt_zeit)
Zeit_parameter = correlated_values(popt_zeit,pcov_zeit)

#print("Zeit_parameter", Zeit_parameter)
#writeW(Zeit_parameter,"die Parameter aus dem Zeit fit")
#writeW(1/Zeit_parameter[1]," Mittlere Lebensdauer in mu s")

plt.errorbar(Zeit,gefüllte_Kanäle,yerr=np.sqrt(gefüllte_Kanäle),elinewidth=0.2,capthick=1,markersize=2,color = 'green', fmt='x')
plt.plot(x_zeit,y_zeit,color='red',label="gefittete Funktion")
plt.fill_between(Zeit,gefüllte_Kanäle,step="pre",alpha=1,label="Histogram der Myonenzerfälle")
plt.ylabel("Zählung")
plt.xlabel(r"$t \mathbin{/} \unit{\micro\second}$")
plt.xlim(0,10)
plt.ylim(0,60)
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Myonenzerfall_Zeit.pdf')
plt.clf()


#Bestimmung der Auflösungszeit
KK10 = np.genfromtxt('Messdaten/Kalibrierung_Koinzidenz_10.txt',encoding='unicode-escape')
KK15 = np.genfromtxt('Messdaten/Kalibrierung_Koinzidenz_15.txt',encoding='unicode-escape')
KK20 = np.genfromtxt('Messdaten/Kalibrierung_Koinzidenz_20.txt',encoding='unicode-escape')
KK10u = unp.uarray(KK10[:,2],np.sqrt(KK10[:,2]))
KK15u = unp.uarray(KK15[:,2],np.sqrt(KK15[:,2]))
KK20u = unp.uarray(KK20[:,2],np.sqrt(KK20[:,2]))
#auf 1 Sekunde rechnen
KK10u = KK10u[:]/KK10[:,1]
KK15u = KK15u[:]/KK15[:,1]
KK20u = KK20u[:]/KK20[:,1]

#writeW(KK10u,"KK10u")
#writeW(KK10,"KK10")


writeW(KK10u,"KK10u")
writeW(KK10,"KK10")

writeW(KK15u,"KK15u")
writeW(KK15,"KK15")
writeW(KK20u,"KK20u")
writeW(KK20,"KK20")


#x = np.arange(0, 1000)
#f = np.arange(0, 1000)
#g = np.sin(np.arange(0, 10, 0.01) * 2) * 1000
#
#plt.plot(x, f, '-',color='blue')
#plt.plot(x, g, '-',color='red')
#
#idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
#plt.plot(x[idx], f[idx], color='green')
#plt.show()


def poly(x,koef):
    return np.polyval(koef,x)

x_kali_koin = np.linspace(-32,32,100000)
plateu_10 = unp.nominal_values(KK10u[KK10u > 18])
plateu_mean_10 = np.mean(plateu_10)

pol4_10 = np.polyfit(KK10[:,0],unp.nominal_values(KK10u[:]),deg = 8)
y_kali_koin_10 = np.polyval(pol4_10,x_kali_koin)

idx = np.argwhere(np.diff(np.sign(plateu_mean_10/2 - y_kali_koin_10))).flatten()
print("Halbwertsbreite 10 in ns", np.max(x_kali_koin[idx]) - np.min(x_kali_koin[idx]))

plt.plot(x_kali_koin,y_kali_koin_10,color = 'red',label="Polynom 8. Grades")
plt.plot(x_kali_koin[idx], y_kali_koin_10[idx], color='green',label="Halbwertsbreite")
plt.errorbar(KK10[:,0],unp.nominal_values(KK10u[:]),yerr=unp.std_devs(KK10u[:]),color = 'blue', fmt='x',label="Messwerte")
plt.ylabel(r"$\text{Zählung} \mathbin{/} \unit{\second}$")
plt.xlabel(r"$\text{Verzögerungszeit} in \unit{\nano\second}$")
plt.grid(linestyle = ":")
plt.ylim(0,30)
plt.tight_layout()
plt.legend()
plt.savefig('build/Kalibrierung_Koinzidenz_10')
plt.clf()


pol4_15 = np.polyfit(KK15[:,0],unp.nominal_values(KK15u[:]),deg = 8)
y_kali_koin_15 = np.polyval(pol4_15,x_kali_koin)

plt.plot(x_kali_koin,y_kali_koin_15,color = 'red',label = "Polynom 8. Grades" )
plateu_15 = unp.nominal_values(KK15u[KK15u > 21])
plateu_mean_15 = np.mean(plateu_15)
idx = np.argwhere(np.diff(np.sign(plateu_mean_15/2 - y_kali_koin_15))).flatten()
print("Halbwertsbreite 15 in ns", np.max(x_kali_koin[idx]) - np.min(x_kali_koin[idx]))


plt.plot(x_kali_koin[idx], y_kali_koin_15[idx], color='green',label='Halbwertsbreite')


plt.errorbar(KK15[:,0],unp.nominal_values(KK15u[:]),yerr=unp.std_devs(KK15u[:]),color = 'blue', fmt='x',label="Messwerte")
plt.ylabel(r"$\text{Zählung} \mathbin{/} \unit{\second}$")
plt.xlabel(r"$\text{Verzögerungszeit} in \unit{\nano\second}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Kalibrierung_Koinzidenz_15')
plt.clf()


plateu_20 = unp.nominal_values(KK20u[KK20u > 19.4])
plateu_mean_20 = np.mean(plateu_20)
pol4_20 = np.polyfit(KK20[:,0],unp.nominal_values(KK20u[:]),deg = 8)
y_kali_koin_20 = np.polyval(pol4_20,x_kali_koin)
idx = np.argwhere(np.diff(np.sign(plateu_mean_20/2 - y_kali_koin_20))).flatten()


print("Halbwertsbreite 20 in ns", np.max(x_kali_koin[idx]) - np.min(x_kali_koin[idx]))
plt.plot(x_kali_koin[idx], y_kali_koin_20[idx], color='green',label='Halbwertsbreite')
plt.plot(x_kali_koin,y_kali_koin_20,color = 'red',label="Polynom 8. Grades")
plt.errorbar(KK20[:,0],unp.nominal_values(KK20u[:]),yerr=unp.std_devs(KK20u[:]),color = 'blue', fmt='x',label="Messwerte")
plt.ylabel(r"$\text{Zählung} \mathbin{/} \unit{\second}$")
plt.xlabel(r"$\text{Verzögerungszeit} in \unit{\nano\second}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Kalibrierung_Koinzidenz_20')
plt.clf()

writeW(plateu_mean_10,"plateu_mean_10")
writeW(np.std(plateu_10),"plateu_std_10")

writeW(plateu_mean_15,"plateu_mean_15")
writeW(np.std(plateu_15),"plateu_std_15")

writeW(plateu_mean_20,"plateu_mean_20")
writeW(np.std(plateu_20),"plateu_std_20")
