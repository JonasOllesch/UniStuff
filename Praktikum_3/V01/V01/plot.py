import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)


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

def pol0(a,x):
    return a 


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


x_left_10  = np.linspace(-18,-1,5)
x_mid_10   = np.linspace(-1,5,5) 
x_right_10 = np.linspace(5,18,5)

popt_x_left_10,pcov_x_left_10 = curve_fit(pol1,KK10[3:8,0],unp.nominal_values(KK10u[3:8]),sigma=unp.std_devs(KK10u[3:8]))
para_x_left_10 = correlated_values(popt_x_left_10,pcov_x_left_10)
y_left_10 = pol1(x_left_10,*para_x_left_10)

mean_10 = np.mean(unp.nominal_values(KK10u[8:14]))

popt_x_right_10,pcov_x_right_10 = curve_fit(pol1,KK10[14:17,0],unp.nominal_values(KK10u[14:17]),sigma=unp.std_devs(KK10u[14:17])) 
para_x_right_10 = correlated_values(popt_x_right_10,pcov_x_right_10)
y_right_10 = pol1(x_right_10,*para_x_right_10)

mm6 = plt.plot(x_left_10, unp.nominal_values(y_left_10), color='#ff7f0e')
mm7 = plt.hlines(mean_10,xmin=-1,xmax=5,color='#2ca02c')
mm8 = plt.plot(x_right_10, unp.nominal_values(y_right_10), color='#d62728')
mm9 = plt.hlines(mean_10/2,xmax=unp.nominal_values((mean_10/2-para_x_left_10[0])/para_x_left_10[1]),xmin=unp.nominal_values((mean_10/2-para_x_right_10[0])/para_x_right_10[1]),color='#2ca02c',linestyle='dashed')

#print(popt_x_left_10)
#print(popt_x_right_10)
#print(mean_10/2)
#print((mean_10/2-popt_x_left_10[1])/popt_x_left_10[0])
#print((mean_10/2-popt_x_right_10[1])/popt_x_right_10[0])

mm1 = plt.errorbar(KK10[:3, 0], unp.nominal_values(KK10u[:3]), yerr=unp.std_devs(KK10u[:3]), fmt='x')
mm2 = plt.errorbar(KK10[3:8, 0], unp.nominal_values(KK10u[3:8]), yerr=unp.std_devs(KK10u[3:8]), fmt='x')
mm3 = plt.errorbar(KK10[8:14, 0], unp.nominal_values(KK10u[8:14]), yerr=unp.std_devs(KK10u[8:14]), fmt='x')
mm4 = plt.errorbar(KK10[14:17, 0], unp.nominal_values(KK10u[14:17]), yerr=unp.std_devs(KK10u[14:17]), fmt='x')
mm5 = plt.errorbar(KK10[17:, 0], unp.nominal_values(KK10u[17:]), yerr=unp.std_devs(KK10u[17:]), fmt='x')

handles = [(mm1, mm2, mm3, mm4, mm5), (mm6[0], mm7, mm8[0]),(mm9)]
labels = ['Messwerte', 'lin. Fit','FWHM']

plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')

plt.ylabel(r"$\text{Zählung} \mathbin{/} \unit{\second}$")
plt.xlabel(r"$\text{Verzögerungszeit} \, \text{in} \, \unit{\nano\second}$")
plt.grid(linestyle = ":")
plt.ylim(0,30)
plt.tight_layout()
plt.savefig('build/Kalibrierung_Koinzidenz_10')
plt.clf()


x_left_15 = np.linspace(-18,-2,5)
x_mid_15 = np.linspace(-4,10,5)
x_right_15 = np.linspace(10,21,5)

popt_x_left_15,pcov_x_left_15 = curve_fit(pol1,KK15[3:8,0],unp.nominal_values(KK15u[3:8]),sigma=unp.std_devs(KK15u[3:8]))
para_x_left_15 = correlated_values(popt_x_left_15,pcov_x_left_15)
y_left_15 = pol1(x_left_15,*para_x_left_15)

mean_15 = np.mean(unp.nominal_values(KK15u[8:15]))

popt_x_right_15,pcov_x_right_15 = curve_fit(pol1,KK15[15:18,0],unp.nominal_values(KK15u[15:18]),sigma=unp.std_devs(KK15u[15:18])) 
para_x_right_15 = correlated_values(popt_x_right_15,pcov_x_right_15)
y_right_15 = pol1(x_right_15,*para_x_right_15)

mm6 = plt.plot(x_left_15, unp.nominal_values(y_left_15), color='#ff7f0e')
mm7 = plt.hlines(mean_15,xmin=-4,xmax=10,color='#2ca02c')
mm8 = plt.plot(x_right_15, unp.nominal_values(y_right_15), color='#d62728')
mm9 = plt.hlines(mean_15/2,xmax=unp.nominal_values((mean_15/2-para_x_left_15[0])/para_x_left_15[1]),xmin=unp.nominal_values((mean_15/2-para_x_right_15[0])/para_x_right_15[1]),color='#2ca02c',linestyle='dashed')


mm1 = plt.errorbar(KK15[:3,0]   ,unp.nominal_values(KK15u[:3])      ,yerr=unp.std_devs(KK15u[:3])   ,fmt='x')
mm2 = plt.errorbar(KK15[3:8,0]  ,unp.nominal_values(KK15u[3:8])     ,yerr=unp.std_devs(KK15u[3:8])  ,fmt='x')
mm3 = plt.errorbar(KK15[8:15,0] ,unp.nominal_values(KK15u[8:15])    ,yerr=unp.std_devs(KK15u[8:15]) ,fmt='x')
mm4 = plt.errorbar(KK15[15:18,0],unp.nominal_values(KK15u[15:18])   ,yerr=unp.std_devs(KK15u[15:18]),fmt='x')
mm5 = plt.errorbar(KK15[18:,0]  ,unp.nominal_values(KK15u[18:])     ,yerr=unp.std_devs(KK15u[18:])  ,fmt='x')

handles = [(mm1, mm2, mm3, mm4, mm5), (mm6[0], mm7, mm8[0]),mm9]
labels = ['Messwerte', 'lin. Fit','FWHM']
plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')


plt.ylim(0,30)
plt.ylabel(r"$\text{Zählung} \mathbin{/} \unit{\second}$")
plt.xlabel(r"$\text{Verzögerungszeit} \, \text{in} \, \unit{\nano\second}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.savefig('build/Kalibrierung_Koinzidenz_15')
plt.clf()

x_left_20 = np.linspace(-28,-18,5)
x_mid_20 = np.linspace(-18,12,5)
x_right_20 = np.linspace(12,25,5)

popt_x_left_20,pcov_x_left_20 = curve_fit(pol1,KK20[1:3,0],unp.nominal_values(KK20u[1:3]),sigma=unp.std_devs(KK20u[1:3]))
para_x_left_20 = correlated_values(popt_x_left_20,pcov_x_left_20)
y_left_20 = pol1(x_left_20,*para_x_left_20)

mean_20 = np.mean(unp.nominal_values(KK20u[3:16]))

popt_x_right_20,pcov_x_right_20 = curve_fit(pol1,KK20[16:19,0],unp.nominal_values(KK20u[16:19]),sigma=unp.std_devs(KK20u[16:19])) 
para_x_right_20 = correlated_values(popt_x_right_20,pcov_x_right_20)
y_right_20 = pol1(x_right_20,*para_x_right_20)

mm6 = plt.plot(x_left_20, unp.nominal_values(y_left_20), color='#ff7f0e')
mm7 = plt.hlines(mean_20,xmin=-18,xmax=12,color='#2ca02c')
mm8 = plt.plot(x_right_20, unp.nominal_values(y_right_20), color='#d62728')
mm9 = plt.hlines(mean_20/2,xmax=unp.nominal_values((mean_20/2-para_x_left_20[0])/para_x_left_20[1]),xmin=unp.nominal_values((mean_20/2-para_x_right_20[0])/para_x_right_20[1]),color='#2ca02c',linestyle='dashed')

print("pcov_x_left_20", pcov_x_left_20)
print("para_x_left_20", para_x_left_20)
print("para_x_right_20", para_x_right_20)
plt.errorbar(KK20[:1,0],unp.nominal_values(KK20u[:1]),yerr=unp.std_devs(KK20u[:1])              ,fmt='x')
plt.errorbar(KK20[1:3,0],unp.nominal_values(KK20u[1:3]),yerr=unp.std_devs(KK20u[1:3])           ,fmt='x')
plt.errorbar(KK20[3:16,0],unp.nominal_values(KK20u[3:16]),yerr=unp.std_devs(KK20u[3:16])        ,fmt='x')
plt.errorbar(KK20[16:19,0],unp.nominal_values(KK20u[16:19]),yerr=unp.std_devs(KK20u[16:19])     ,fmt='x')
plt.errorbar(KK20[19:,0],unp.nominal_values(KK20u[19:]),yerr=unp.std_devs(KK20u[19:])           ,fmt='x')

handles = [(mm1, mm2, mm3, mm4, mm5), (mm6[0], mm7, mm8[0]),mm9]
labels = ['Messwerte', 'lin. Fit','FWHM']
plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')


plt.ylim(0,25)
plt.ylabel(r"$\text{Zählung} \mathbin{/} \unit{\second}$")
plt.xlabel(r"$\text{Verzögerungszeit} \, \text{in} \, \unit{\nano\second}$")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.savefig('build/Kalibrierung_Koinzidenz_20')
plt.clf()


print("mean_10: ", mean_10)
print("mean_15: ", mean_15)
print("mean_20: ", mean_20)

print("FWHM_10 : ",  repr((mean_10/2-para_x_left_10[0])/para_x_left_10[1] - (mean_10/2-para_x_right_10[0])/para_x_right_10[1]))
print("FWHM_15 : ",  repr((mean_15/2-para_x_left_15[0])/para_x_left_15[1] - (mean_15/2-para_x_right_15[0])/para_x_right_15[1]))
print("FWHM_20 : ",  repr((mean_20/2-para_x_left_20[0])/para_x_left_20[1] - (mean_20/2-para_x_right_20[0])/para_x_right_20[1]))