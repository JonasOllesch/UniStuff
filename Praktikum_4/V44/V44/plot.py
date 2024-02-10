import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import peak_widths
import scipy.constants as constants

from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp

from multiprocessing  import Process

import math

from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
from iminuit.cost import LeastSquares


def Gaus(x, a, mu, sigma, b):
    return a/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2) ) + b


def uGaus(x, a, mu, sigma, b):
    return a/(sigma * unp.sqrt(2 * np.pi)) * unp.exp( - (x - mu)**2 / (2 * sigma**2) ) + b


def Geometriefactor_berechnen(SampleWidth, Incidence_angle, BeamWidth, Geometrie_angle):

    if Incidence_angle == 0:
        return 1


    if Incidence_angle < Geometrie_angle:


        return SampleWidth*np.sin(Incidence_angle*np.pi/180)/BeamWidth
    
    if Incidence_angle > Geometrie_angle:
        return 1
    return 1

Detektorscann = np.genfromtxt('Messdaten/GaussScan.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape') 
ZScann1 = np.genfromtxt('Messdaten/Z2raw.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
RockingCurve = np.genfromtxt('Messdaten/RockingCurve1raw.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
Omega2Theta = np.genfromtxt('Messdaten/Omega2Theta.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')
Diffus = np.genfromtxt('Messdaten/Diffus.UXD', skip_header = 56, skip_footer = 0, encoding = 'unicode-escape')

#Detektorscan
#Fit der Gauskurve
x_fit_Dek = np.linspace(-0.5,0.5, 10000)

popt, pcov = curve_fit(Gaus, Detektorscann[:,0], Detektorscann[:,1], absolute_sigma = True ,p0=[1.5e05, 0, 0.05,  200])

para_Detektorscan = correlated_values(popt, pcov)


#Brechnung von FWHM
y_fit_Dek = Gaus(x_fit_Dek, *popt)


Gaus_Peak_idx = np.array([np.argmax(unp.nominal_values(y_fit_Dek))])

Gaus_Peak_FWHM_in_idx, tmp, tmp1, tmp2 = peak_widths(unp.nominal_values(y_fit_Dek), Gaus_Peak_idx, rel_height=0.5)
Gaus_Peak_FWHM = x_fit_Dek[int(Gaus_Peak_FWHM_in_idx)] - x_fit_Dek[0]



print(f'Die Parameter des Gaus Detektorscan a, mu, sigma, b {para_Detektorscan}')
print(f'Die FWHM des Detektorscan {Gaus_Peak_FWHM} in Grad')
I_max = uGaus(x_fit_Dek[Gaus_Peak_idx], *para_Detektorscan)
print(f'Das Maximum des Fits: {I_max}')
print(f'Gaus Detektorscan sigma {para_Detektorscan[2]}')
print(f'Gaus Detektorscan sigma nv{unp.nominal_values(para_Detektorscan[2])}')
print(f'Gaus Detektorscan sigma std {unp.std_devs(para_Detektorscan[2])}')

print(f'FWHM with formular {2*np.sqrt(2*np.log(2))*para_Detektorscan[2]}')


#der Z-Scan
lineOffset = 56
SampleStart = 87
SampleEnd = 97
Z_Scan_width = ZScann1[SampleEnd-lineOffset, 0] - ZScann1[SampleStart-lineOffset, 0]
Z_Scan_width = ufloat(Z_Scan_width/1000, 0.02/1000)

print(f'Die Breite des Beams in z-scan {ZScann1[SampleEnd-lineOffset, 0] - ZScann1[SampleStart-lineOffset, 0]} +- 0.02 / mm ')
BeamWidth = ufloat(ZScann1[SampleEnd-lineOffset, 0] - ZScann1[SampleStart-lineOffset, 0], 0.02)

#Rockingscan
RockingCurve_Left = 69 - lineOffset
RockingCurve_Right = 95 - lineOffset 


#print(RockingCurve[RockingCurve_Right, 0] - RockingCurve[RockingCurve_Left, 0])



Geometrie_angle_in_Grad = ufloat((RockingCurve[RockingCurve_Right, 0] - RockingCurve[RockingCurve_Left, 0])/2, 0.02)
print(f'Der Geometriewinkel beträgt {Geometrie_angle_in_Grad} grad')
Geometrie_angle = Geometrie_angle_in_Grad*np.pi/180
print(f'Der Geometriewinkel {Geometrie_angle}')




#Berechnung des theoretischen Geometriefakors
SampleWidth = 20 #in mm
EffectivBeamWidth = unp.arcsin(BeamWidth/SampleWidth)

print(f'Geometry angle theo. {repr(EffectivBeamWidth)}')
print(f'Geometry angle theo. {repr(EffectivBeamWidth*180/np.pi)} in degree')


print(f'Geometriefaktor exp.  : {SampleWidth/BeamWidth*unp.sin(Geometrie_angle*np.pi/180)}')
print(f'Geometriefaktor Theo. : {SampleWidth/BeamWidth*unp.sin(EffectivBeamWidth*np.pi/180)}')


#Die Reflectivity berechnen und Sachenmachen
Reflectivity_y = Omega2Theta[:, 1]/(5 * I_max)
Reflectivity_x = Omega2Theta[: ,0]
Reflectivity_y_tmp = np.copy(Reflectivity_y)


GeometryFactors = np.copy(Reflectivity_x)
print(f'sample width {SampleWidth}')
print(f'BeamWidth {BeamWidth}')
for i in range(0, len(Reflectivity_x)):
    GeometryFactors[i] = unp.nominal_values(Geometriefactor_berechnen(SampleWidth/1000, Reflectivity_x[i], Z_Scan_width, Geometrie_angle_in_Grad))

R_Diffus = Diffus[0:,1]/(5*unp.nominal_values(I_max))
R_Omega2Theta = Omega2Theta[:,1]/(5*unp.nominal_values(I_max))
R_without_Background = (Omega2Theta[:,1] - Diffus[:,1])/(5*unp.nominal_values(I_max))
Omega2Theta_angle = np.copy(Omega2Theta[:,0])

R_without_Background_corrected = np.copy(R_without_Background)
for i in range(0, len(R_without_Background)):
    R_without_Background_corrected[i] = R_without_Background[i] / (unp.nominal_values(Geometriefactor_berechnen(SampleWidth/1000, Omega2Theta_angle[i], Z_Scan_width,  Geometrie_angle_in_Grad)))



"""

def fit_Parratt_Algorithmus2(angle, layer_thickness, delta_poli, delta_Si, sigma_poli, sigma_Si):

    beta_poli=  delta_poli/200
    beta_si=  delta_Si/40
    Wellenlänge = 1.54e-10
    k = 2*np.pi / Wellenlänge
#    layer_thickness = 8.8e-8
#    layer_thickness = 5e-6
    x_Air_arr = np.zeros(len(angle))
#    print(sigma_poli)
    n_Air = 1
    n_poly = 1 - delta_poli - 1j*beta_poli
    n_Si = 1 - delta_Si - 1j*beta_si

    for i in range(0 , len(angle)):

        k_Air =     k * np.sqrt((n_Air**2 -np.cos(angle[i])**2))
        k_poly =    k * np.sqrt((n_poly**2 -np.cos(angle[i])**2))
        k_Si =      k * np.sqrt((n_Si**2 -np.cos(angle[i])**2))

        r_Air_poly = (k_Air - k_poly)/ (k_Air + k_poly)* np.exp(-2*k_Air*k_poly*sigma_poli**2)
        r_poly_Si = (k_poly - k_Si)  / (k_poly + k_Si) * np.exp(-2*k_poly*k_Si*sigma_Si**2)

        x_poly = np.exp(-2j * k_poly * layer_thickness) * r_poly_Si
        x_Air = (r_Air_poly + x_poly)/(1 + r_Air_poly  *x_poly)

        x_Air_arr[i] = (np.abs(x_Air))**2

#    return x_Air_arr, err_func(x_Air_arr)
    return x_Air_arr




#print(Omega2Theta_angle)
Best_combi = np.array([0.849e-7 ,0.970e-6, 0.693e-5, 5.12e-10, 0.790e-9])

#cost2 = LeastSquares(x = np.deg2rad(Omega2Theta_angle[15:]), y = R_without_Background_corrected[15:], yerror=np.ones(len(Reflectivity_y[15:])), model= fit_Parratt_Algorithmus2)
cost2 = UnbinnedNLL(np.append(Omega2Theta_angle[15:180], R_without_Background_corrected[15:180]), fit_Parratt_Algorithmus2)

m2 = Minuit(cost2, *Best_combi)

m2.limits['layer_thickness']  = (85.03e-9   ,85.03e-9)
m2.limits['delta_poli']       = (0.970e-6   ,0.970e-6)
m2.limits['delta_Si']         = (0.693e-5   ,0.693e-5)
m2.limits['sigma_poli']       = (5.12e-10   ,5.12e-10)
m2.limits['sigma_Si']         = (0.790e-9   ,0.790e-9)
print(m2.migrad())  # find minimum


Best_combi2 = m2.values
Reflectivity_y_Parratt_nach_Suche2 = fit_Parratt_Algorithmus2(np.deg2rad(Omega2Theta_angle), *Best_combi2)
print(Best_combi2)

plt.plot(Omega2Theta_angle, unp.nominal_values(Reflectivity_y_Parratt_nach_Suche2), label = "Parrattfit 2", color = "orange")
plt.scatter(Omega2Theta_angle, unp.nominal_values(R_without_Background_corrected), label = "Reflectivity", c = "midnightblue", marker='.', s = 1)


plt.yscale('log')
plt.xlabel(" alpha \ degree")
plt.ylabel("Reflectivity")
plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/Reflectivity_Parratt2.pdf')
plt.clf()

"""



Reflectivity_y_tmp = Reflectivity_y_tmp*1/unp.nominal_values(GeometryFactors)

#Schichtdicke berechnen

Wellenlänge = 1.54e-10
Ozi_Minima_idx = np.array([118, 126, 136, 145, 155, 166, 176, 187, 197, 207, 218, 228, 239]) - lineOffset

Ozi_Minima = Reflectivity_x[Ozi_Minima_idx]

delta_alpha = np.zeros(len(Ozi_Minima)-1)
for i in range(0, len(delta_alpha)):
    delta_alpha[i] = np.deg2rad(Ozi_Minima[i+1] - Ozi_Minima[i])
layer_thickness = Wellenlänge/(2*ufloat(np.mean(delta_alpha), np.std(delta_alpha)))


print(f'Die Dicke einer Schicht in Meter {layer_thickness}')



#der Parratt-Algorithmus 

def Parratt_Algorithmus(angle, delta_poli, delta_Si, sigma_poli, sigma_Si,beta_poli, beta_si):

    Wellenlänge = 1.54e-10
    k = 2*np.pi / Wellenlänge
    layer_thickness = 8.8e-8
    x_Air_arr = np.zeros(len(angle))
    n_Air = 1
    n_poly = 1 - delta_poli - 1j*beta_poli
    n_Si = 1 - delta_Si - 1j*beta_si

    for i in range(0 , len(angle)):

        k_Air =     k * np.sqrt((n_Air**2 -np.cos(angle[i])**2))
        k_poly =    k * np.sqrt((n_poly**2 -np.cos(angle[i])**2))
        k_Si =      k * np.sqrt((n_Si**2 -np.cos(angle[i])**2))

        r_Air_poly = (k_Air - k_poly)/ (k_Air + k_poly)* np.exp(-2*k_Air*k_poly*sigma_poli**2)
        r_poly_Si = (k_poly - k_Si)  / (k_poly + k_Si) * np.exp(-2*k_poly*k_Si*sigma_Si**2)

        x_poly = np.exp(-2j * k_poly * layer_thickness) * r_poly_Si
        x_Air = (r_Air_poly + x_poly)/(1 + r_Air_poly  *x_poly)

        x_Air_arr[i] = (np.abs(x_Air))**2

    return x_Air_arr


def fit_Parratt_Algorithmus(angle, layer_thickness, delta_poli, delta_Si, sigma_poli, sigma_Si):

    beta_poli=  delta_poli/200
    beta_si=  delta_Si/40
    Wellenlänge = 1.54e-10
    k = 2*np.pi / Wellenlänge
#    layer_thickness = 8.8e-8
#    layer_thickness = 5e-6
    x_Air_arr = np.zeros(len(angle))
#    print(sigma_poli)
    n_Air = 1
    n_poly = 1 - delta_poli - 1j*beta_poli
    n_Si = 1 - delta_Si - 1j*beta_si

    for i in range(0 , len(angle)):

        k_Air =     k * np.sqrt((n_Air**2 -np.cos(angle[i])**2))
        k_poly =    k * np.sqrt((n_poly**2 -np.cos(angle[i])**2))
        k_Si =      k * np.sqrt((n_Si**2 -np.cos(angle[i])**2))

        r_Air_poly = (k_Air - k_poly)/ (k_Air + k_poly)* np.exp(-2*k_Air*k_poly*sigma_poli**2)
        r_poly_Si = (k_poly - k_Si)  / (k_poly + k_Si) * np.exp(-2*k_poly*k_Si*sigma_Si**2)

        x_poly = np.exp(-2j * k_poly * layer_thickness) * r_poly_Si
        x_Air = (r_Air_poly + x_poly)/(1 + r_Air_poly  *x_poly)

        x_Air_arr[i] = (np.abs(x_Air))**2

#    return x_Air_arr, err_func(x_Air_arr)
    return x_Air_arr





#print(np.deg2rad(Reflectivity_x))
Reflectivity_x_rad = np.deg2rad(Reflectivity_x)
Reflectivity_y = unp.nominal_values(Reflectivity_y)


Best_combi = np.array([6e-7, 6e-6, 5.5e-10, 6.45e-10, 6e-7/40 ,6e-6/200])

Reflectivity_y_Parratt_vor_Suche = Parratt_Algorithmus(Reflectivity_x_rad, *Best_combi)




#Best_combi = np.array([6e-7, 6e-6, 5.5e-10, 6.45e-10, 6e-7/40 ,6e-6/200])
#Best_combi = np.array([8.8e-10,6.029982544217072e-07, 5.879718970707921e-06, 5.527500513536649e-10, 6.482225492181412e-10, 3.014991272108536e-10 ,1.4699297426769803e-08])
Best_combi = np.array([0.856e-7 ,0.62e-6, 0.752e-5, 0.512e-9, 0.620e-9])



#cost = LeastSquares(x=Reflectivity_x_rad, y=unp.nominal_values(Reflectivity_y_tmp), yerror=np.ones(len(Reflectivity_y)), model= fit_Parratt_Algorithmus)
cost = UnbinnedNLL(np.append(Reflectivity_x_rad[10:], unp.nominal_values(Reflectivity_y_tmp[10:])), fit_Parratt_Algorithmus)

m = Minuit(cost, *Best_combi)

m.limits['layer_thickness']  = (1e-8    ,1e-7)
m.limits['delta_poli']       = (1e-7    ,1e-6)
m.limits['delta_Si']         = (1e-6    ,1e-5)
m.limits['sigma_poli']       = (1e-10   ,1e-9)
m.limits['sigma_Si']         = (1e-10   ,1e-9)
#m.limits['beta_poli']   = (1e-10,1e-6)
#m.limits['beta_si']     = (1e-10,1e-6)
#print(m.migrad())  # find minimum


Best_combi = m.values

print(Best_combi)
Best_combi = np.array([0.849e-7 ,0.970e-6, 0.693e-5, 0.617e-9, 0.790e-9])
#para, pcov = curve_fit(fit_Parratt_Algorithmus, Reflectivity_x_rad, unp.nominal_values(Reflectivity_y_tmp), p0 = [0.856e-7 ,1.1e-6, 0.752e-5, 0.348e-9, 0.802e-9], maxfev = 5000, method='trf')
#Best_combi = para
#print(Reflectivity_x_rad)

Reflectivity_y_Parratt_nach_Suche = fit_Parratt_Algorithmus(Reflectivity_x_rad, *Best_combi)


#print(para)

def plotte_Z1Scan(ZScann1, SampleStart, SampleEnd):
    lineOffset = 56
    plt.scatter(ZScann1[:,0], ZScann1[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)

    plt.scatter(ZScann1[SampleStart-lineOffset, 0], ZScann1[SampleStart-lineOffset ,1], c = "firebrick", marker='x', s = 10)
    plt.scatter(ZScann1[SampleEnd-lineOffset, 0], ZScann1[SampleEnd-lineOffset ,1], c = "firebrick", marker='x', s = 10)

    plt.vlines(ZScann1[SampleStart-lineOffset, 0], 0, 1.25e6, colors='firebrick', label = "Beam boundaries", linestyles='dashed')
    plt.vlines(ZScann1[SampleEnd-lineOffset, 0], 0, 1.25e6, colors='firebrick', linestyles='dashed')

    plt.xlabel(r"$z \mathbin{/} \unit{\milli\meter}$")
    plt.ylabel("Intensity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.savefig('build/Z1Scann.pdf')
    plt.clf()

def plotte_Detektorscan(Detektorscann, x_fit_Dek, y_fit_Dek, Gaus_Peak_idx, Gaus_Peak_FWHM_in_idx):
    tmp_x = np.array([(Gaus_Peak_idx-Gaus_Peak_FWHM_in_idx/2), (Gaus_Peak_idx+Gaus_Peak_FWHM_in_idx/2)])
    tmp_x = tmp_x.astype(int)
    tmp_y = y_fit_Dek[tmp_x]
    
    plt.plot(x_fit_Dek[tmp_x], tmp_y, label = 'FWHM', color = 'black')
    plt.plot(x_fit_Dek, y_fit_Dek, color = 'firebrick' , label = 'Gaussfit')

    plt.scatter(Detektorscann[:,0], Detektorscann[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Intensity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Detectorscan.pdf')
    plt.clf()

from matplotlib.ticker import FuncFormatter
def format_ticks(x, pos):
    if x == 0:
        return '0'
    else:
        return f'{int(x/1000)}k'

def plotte_RockingCurve(RockingCurve, RockingCurve_Left, RockingCurve_Right):
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))

    plt.scatter(RockingCurve[:,0], RockingCurve[:,1], label = "Data", c = "midnightblue", marker='x', s = 10)

    plt.scatter(RockingCurve[RockingCurve_Left, 0], RockingCurve[RockingCurve_Left, 1], label = "Geometrie angle", c = "firebrick", marker='x', s = 10)
    plt.scatter(RockingCurve[RockingCurve_Right, 0], RockingCurve[RockingCurve_Right, 1], c = "firebrick", marker='x', s = 10)

    plt.vlines(RockingCurve[RockingCurve_Left, 0], -1e4, 480e3, colors='firebrick', linestyles='dashed')
    plt.vlines(RockingCurve[RockingCurve_Right, 0], -1e4, 480e3, colors='firebrick', linestyles='dashed')

    plt.ylim(-1e4,490e3)
    plt.xlim(-1, 1)
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Intensity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Rockingcurve.pdf')
    plt.clf()

def plotte_Omega2Theta(Omega2Theta, Diffus):

    plt.scatter(Omega2Theta[:,0], Omega2Theta[:,1], label = "Compact Reflectivity", c = "midnightblue", marker='.', s = 1)
    plt.scatter(Diffus[:,0], Diffus[:,1], label = "Diffuse Reflectivity", c = "cornflowerblue", marker='.', s = 1)

#    plt.plot(Omega2Theta[:,0], Omega2Theta[:,1], label = "Data", c = "midnightblue")
    plt.yscale('log')
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Intensity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Omega2Theta.pdf')
    plt.clf()


def plotte_Reflectivity(Omega2Theta_angle, R_Diffus, R_Omega2Theta, R_without_Background):

    plt.scatter(Omega2Theta_angle, unp.nominal_values(R_Diffus), label = "Refectivity of background scan", c = "firebrick", marker='.', s = 1)
    plt.scatter(Omega2Theta_angle, unp.nominal_values(R_without_Background), label = "Refectivity without background", c = "darkorange", marker='.', s = 1)
    plt.scatter(Omega2Theta_angle, unp.nominal_values(R_Omega2Theta), label = r"Refectivity of $\Omega\mathbin{/} 2\theta$", c = "midnightblue", marker='.', s = 1)


    plt.yscale('log')
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Reflectivity.pdf')
    plt.clf()


def plotte_Reflectivity_corrected(Omega2Theta_angle, R_without_Background, R_without_Background_corrected):

    plt.scatter(Omega2Theta_angle, unp.nominal_values(R_without_Background), label = "Refectivity without background", c = "darkorange", marker='.', s = 1)
    plt.scatter(Omega2Theta_angle, unp.nominal_values(R_without_Background_corrected), label = "Refectivity corrected", c = "teal", marker='.', s = 1)


    plt.yscale('log')
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Reflectivity_corrected.pdf')
    plt.clf()



def plotte_Parratt_Algorithmus(Omega2Theta_angle, R_without_Background_corrected,  Reflectivity_y_Parratt_nach_Suche):
    #print(Reflectivity_y_Parratt)
#    plt.scatter(unp.nominal_values(Reflectivity_x), unp.nominal_values(Reflectivity_y_Parratt_vor_Suche), label = "Parrattfit vor", c = "firebrick", marker='.', s = 1)

    plt.scatter(unp.nominal_values(Omega2Theta_angle), unp.nominal_values(R_without_Background_corrected), label = "Reflectivity", c = "midnightblue", marker='.', s = 1)
    plt.scatter(unp.nominal_values(Omega2Theta_angle), unp.nominal_values(Reflectivity_y_Parratt_nach_Suche), label = "Parratt-fit", c = "orange", marker='.', s = 1)
    plt.plot(Omega2Theta_angle[Omega2Theta_angle > 1*0.213], (0.213/(2*Omega2Theta_angle[Omega2Theta_angle>1*0.213]))**4, label="Ideal Fresnel Reflectivity",color='teal')

    tmp = np.copy(Omega2Theta)
    tmp[:] = np.max((0.213/(2*Omega2Theta_angle[Omega2Theta_angle>1*0.213]))**4)
    plt.plot(Omega2Theta_angle[Omega2Theta_angle < 1*0.213], tmp[Omega2Theta_angle < 1*0.213] ,color='teal')


    plt.yscale('log')
    plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
    plt.ylabel("Reflectivity")
    plt.grid(linestyle = ":")
    plt.tight_layout()
    plt.legend()
    plt.savefig('build/Reflectivity_Parratt.pdf')
    plt.clf()


Processe = []
Processe.append(Process(target=plotte_Detektorscan, args=([Detektorscann, x_fit_Dek, y_fit_Dek, Gaus_Peak_idx, Gaus_Peak_FWHM_in_idx])))
Processe.append(Process(target=plotte_Z1Scan, args=([ZScann1, SampleStart, SampleEnd])))
Processe.append(Process(target=plotte_RockingCurve, args=([RockingCurve, RockingCurve_Left, RockingCurve_Right])))
Processe.append(Process(target=plotte_Omega2Theta, args=([Omega2Theta, Diffus])))
Processe.append(Process(target=plotte_Reflectivity, args=([Omega2Theta_angle, R_Diffus, R_Omega2Theta, R_without_Background])))
Processe.append(Process(target=plotte_Parratt_Algorithmus, args=([Reflectivity_x, Reflectivity_y_tmp, Reflectivity_y_Parratt_nach_Suche])))
Processe.append(Process(target=plotte_Reflectivity_corrected, args=([Omega2Theta_angle, R_without_Background, R_without_Background_corrected])))

for p in Processe:
    p.start()

for p in Processe:
    p.join()








#Best_combi = np.array([8.8e-10, 6.029982544217072e-07, 5.879718970707921e-06, 5.527500513536649e-10, 6.482225492181412e-10, 3.014991272108536e-10 ,1.4699297426769803e-08])
Best_combi = np.array([8.8e-10, 6.029982544217072e-07, 5.879718970707921e-06, 5.527500513536649e-10, 6.482225492181412e-10])


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Generate some data
x = Reflectivity_x_rad
y = unp.nominal_values(fit_Parratt_Algorithmus(Reflectivity_x_rad, *Best_combi))

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.5)  # Adjust layout to make room for slider

# Plot the data
line, = ax.plot(x, y)

# Create a slider widget
ax_layer = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_delta_poli = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_delta_Si = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_sigma_poli = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_sigma_Si = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#ax_beta_poli = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#ax_beta_si = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_layer = Slider(ax_layer, 'layer', 8.8e-9, 8.8e-7, valinit=1.0)
slider_delta_poli = Slider(ax_delta_poli, 'delta_poli', 6.029982544217072e-08, 6.029982544217072e-06, valinit=1.0)
slider_delta_Si = Slider(ax_delta_Si, 'delta_Si', 5.879718970707921e-07, 5.879718970707921e-05, valinit=1.0)
slider_sigma_poli = Slider(ax_sigma_poli, 'sigma_poli', 5.527500513536649e-11, 5.527500513536649e-9, valinit=1.0)
slider_sigma_Si = Slider(ax_sigma_Si, 'sigma_Si', 6.482225492181412e-11, 6.482225492181412e-9, valinit=1.0)
#slider_beta_poli = Slider(ax_beta_poli, 'beta_poli', 3.014991272108536e-11,  3.014991272108536e-9, valinit=1.0)
#slider_beta_si = Slider(ax_beta_si, 'beta_si', 1.4699297426769803e-09, 1.4699297426769803e-07, valinit=1.0)


ax.scatter(Reflectivity_x_rad, unp.nominal_values(Reflectivity_y_tmp), color='firebrick', alpha=0.5)

# Define an update function
def update(val):
    # Get the slider value
    layer = slider_layer.val
    delta_poli = slider_delta_poli.val
    delta_Si = slider_delta_Si.val
    sigma_poli = slider_sigma_poli.val
    sigma_Si = slider_sigma_Si.val
#    beta_poli = slider_beta_poli.val
#    beta_si = slider_beta_si.val

    # Update the data
#    y_new = fit_Parratt_Algorithmus(Reflectivity_x_rad, layer, delta_poli, delta_Si, sigma_poli, sigma_Si, beta_poli, beta_si)
    y_new = fit_Parratt_Algorithmus(Reflectivity_x_rad, layer, delta_poli, delta_Si, sigma_poli, sigma_Si)
   
    line.set_ydata(y_new)
    
    # Redraw the plot
    fig.canvas.draw_idle()


# Register the update function with the slider
slider_layer.on_changed(update)
slider_delta_poli.on_changed(update)
slider_delta_Si.on_changed(update)
slider_sigma_poli.on_changed(update)
slider_sigma_Si.on_changed(update)
#slider_beta_poli.on_changed(update)
#slider_beta_si.on_changed(update)

ax.grid(True)

# Set y-axis scale to logarithmic
ax.set_yscale('log')
plt.show()