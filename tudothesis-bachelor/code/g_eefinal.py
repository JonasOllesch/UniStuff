import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

def g_lim(mphi): # limit on g_ij (MeV)
    return 0.83*10**(-8) / mphi

## g_ee mit allen möglichen Phasen

def g_ee_g1_pha0_0(g_ee, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2*np.cos(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2)*np.sin(theta_sun)**2*np.cos(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua)/m1**2) * np.sin(theta_13)**2
    return g_ee/(denom1 + denom2 + denom3)

def g_ee_g1_pha90_0(g_ee, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2*np.cos(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2)*np.sin(theta_sun)**2*np.cos(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua)/m1**2) * np.sin(theta_13)**2
    return g_ee/(denom1 - denom2 + denom3)

def g_ee_g1_pha90_90(g_ee, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2*np.cos(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2)*np.sin(theta_sun)**2*np.cos(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua)/m1**2) * np.sin(theta_13)**2
    return g_ee/(denom1 - denom2 - denom3)

def g_ee_g1_pha0_90(g_ee, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2*np.cos(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2)*np.sin(theta_sun)**2*np.cos(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua)/m1**2) * np.sin(theta_13)**2
    return g_ee/(denom1 + denom2 - denom3)


## g_emu mit allen möglichen Phasen

def g_emu_g1_pha0(g_emu, m1, theta_sun, theta_13, delm_sunsqua):
    denom1 = np.sin(2*theta_sun) * np.cos(theta_13) * (np.sqrt(1 + delm_sunsqua / m1**2) - 1)
    return 2 * g_emu / denom1

def g_emu_g1_pha90(g_emu, m1, theta_sun, theta_13, delm_sunsqua):
    denom1 = np.sin(2*theta_sun) * np.cos(theta_13) * (-np.sqrt(1 + delm_sunsqua / m1**2) - 1)
    return 2 * g_emu / denom1


## g_mumu mit allen möglichen Phasen

def g_mumu_g1_pha0(g_mumu, m1, theta_sun, delm_sunsqua):
    denom1 = np.sin(theta_sun)**2
    denom2 = np.sqrt(1 + delm_sunsqua / m1**2) * np.cos(theta_sun)**2
    return g_mumu / (denom1 + denom2)

def g_mumu_g1_pha90(g_mumu, m1, theta_sun, delm_sunsqua):
    denom1 = np.sin(theta_sun)**2
    denom2 = np.sqrt(1 + delm_sunsqua / m1**2) * np.cos(theta_sun)**2
    return g_mumu / (denom1 - denom2)


## g_tautau mit allen möglichen Phasen

def g_tautau_g1_pha0_0(g_tautau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2 * np.sin(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua / m1**2) * np.sin(theta_sun)**2 * np.sin(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.cos(theta_13)**2
    return g_tautau / (denom1 + denom2 + denom3)


def g_tautau_g1_pha90_0(g_tautau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2 * np.sin(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua / m1**2) * np.sin(theta_sun)**2 * np.sin(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.cos(theta_13)**2
    return g_tautau / (denom1 - denom2 + denom3)


def g_tautau_g1_pha90_90(g_tautau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2 * np.sin(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua / m1**2) * np.sin(theta_sun)**2 * np.sin(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.cos(theta_13)**2
    return g_tautau / (denom1 - denom2 - denom3)

def g_tautau_g1_pha0_90(g_tautau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = np.cos(theta_sun)**2 * np.sin(theta_13)**2
    denom2 = np.sqrt(1 + delm_sunsqua / m1**2) * np.sin(theta_sun)**2 * np.sin(theta_13)**2
    denom3 = np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.cos(theta_13)**2
    return g_tautau / (denom1 + denom2 - denom3)


## g_etau mit allen möglichen Phasen

def g_etau_g1_pha0_0(g_etau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = - 1/2 * np.cos(theta_sun)**2 * np.sin(2*theta_sun)
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2) * np.sin(theta_sun)**2 * np.cos(theta_13)**2
    denom3 = 1/2 * np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.sin(2*theta_13)
    return g_etau / (denom1 + denom2 + denom3)


def g_etau_g1_pha90_0(g_etau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = - 1/2 * np.cos(theta_sun)**2 * np.sin(2*theta_sun)
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2) * np.sin(theta_sun)**2 * np.cos(theta_13)**2
    denom3 = 1/2 * np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.sin(2*theta_13)
    return g_etau / (denom1 - denom2 + denom3)


def g_etau_g1_pha90_90(g_etau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = - 1/2 * np.cos(theta_sun)**2 * np.sin(2*theta_sun)
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2) * np.sin(theta_sun)**2 * np.cos(theta_13)**2
    denom3 = 1/2 * np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.sin(2*theta_13)
    return g_etau / (denom1 - denom2 - denom3)


def g_etau_g1_pha0_90(g_etau, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua):
    denom1 = - 1/2 * np.cos(theta_sun)**2 * np.sin(2*theta_sun)
    denom2 = np.sqrt(1 + delm_sunsqua/m1**2) * np.sin(theta_sun)**2 * np.cos(theta_13)**2
    denom3 = 1/2 * np.sqrt(1 + (delm_atmsqua + delm_sunsqua) / m1**2) * np.sin(2*theta_13)
    return g_etau / (denom1 + denom2 - denom3)


## g_mutau

def g_mutau_g1_pha0(g_mutau, m1, theta_sun, theta_13, delm_sunsqua):
    denom1 = np.sin(2*theta_sun) * np.sin(theta_13) * (1 + np.sqrt(1 + delm_sunsqua / m1**2))
    return g_mutau / denom1


def g_mutau_g1_pha90(g_mutau, m1, theta_sun, theta_13, delm_sunsqua):
    denom1 = np.sin(2*theta_sun) * np.sin(theta_13) * (1 - np.sqrt(1 + delm_sunsqua / m1**2))
    return g_mutau / denom1


delm_sunsqua = 7.53 * 10**(-5)                   # new value from the PDG                                        
delm_atmsqua = 2.453 * 10**(-3)                  # new value from the PDG

theta_sun = np.arcsin(np.sqrt(0.307))
theta_13  = np.arcsin(np.sqrt(2.2*10**(-2))) 

m1 = np.logspace(start = -9, stop = np.log10(0.8), num = 5 * 10**6, base = 10.0)      

# upper limit on m1

m1_lim = 0.8
g1_col = np.logspace(start = -16, stop = -3, num = 1000, base = 10.0)

g_upper = g_lim(100)


# g_ee plots with different phases as indicated by the numbers behind _pha

plt.plot(g_ee_g1_pha0_0(g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='blue', label = r'bounds on $g_{ee}$ with $\delta_1=\delta_2=0$')
plt.plot(g_ee_g1_pha90_0(g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='red', label = r'bounds on $g_{ee}$ with $\delta_1=\frac{\pi}{2}, \delta_2=0$')

plt.plot(g_ee_g1_pha90_90(g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='purple', label = r'bounds on $g_{ee}$ with $\delta_1=\delta_2=\frac{\pi}{2}$')
plt.plot(g_ee_g1_pha0_90(g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='green', label = r'bounds on $g_{ee}$ with $\delta_1=0, \delta_2 = \frac{\pi}{2}$')


plt.plot(g_ee_g1_pha0_0(-g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='blue', linestyle = 'dashed', label = r'bounds on $g_{ee}$ with $\delta_1=\delta_2=0$')
plt.plot(g_ee_g1_pha90_0(-g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='red', linestyle = 'dashed', label = r'bounds on $g_{ee}$ with $\delta_1=\frac{\pi}{2}, \delta_2=0$')

plt.plot(g_ee_g1_pha90_90(-g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='purple',linestyle = 'dashed', label = r'bounds on $g_{ee}$ with $\delta_1=\delta_2=\frac{\pi}{2}$')
plt.plot(g_ee_g1_pha0_90(-g_upper, m1, theta_sun, theta_13, delm_sunsqua, delm_atmsqua), m1, color='green',linestyle = 'dashed', label = r'bounds on $g_{ee}$ with $\delta_1=0, \delta_2 = \frac{\pi}{2}$')

plt.fill_between(g1_col, np.full_like(g1_col, m1_lim), np.full_like(g1_col, 1), color='red', alpha=0.5, linewidth=0)


# labels and limits

plt.xlabel(r'$g_1$')
plt.xlim(10**(-13), 10**(-4))
plt.xscale('log')

plt.ylabel(r'$m_1 \mathbin{/} \mathrm{eV}$')
plt.ylim(10**(-6), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/g_eefinal.pdf')
plt.clf()