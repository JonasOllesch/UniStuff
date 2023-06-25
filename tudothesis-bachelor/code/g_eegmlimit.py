import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


## Functions for m1:
# m_1 in dependence of g_ee and g_1 (with phase 0)

def g_ee_m1_pha0(g_ee, g1, delm_sunsqua, sinsqua_sun):
    denom_1 = (g_ee - g1 * (1 - sinsqua_sun)) / (sinsqua_sun * g1) #* np.exp(2j * delta)
    return (delm_sunsqua / (denom_1**2 - 1))**(1/2)

# m_1 in dependence of g_emu and g_1 (with phase 0)

def g_emu_m1_pha0(g_emu, g1, delm_sunsqua, theta_sun):
    denom_1 = (2*g_emu / (g1 * np.sin(2 * theta_sun)) + 1)
    return np.sqrt(delm_sunsqua/(denom_1**2 - 1))


# m_1 in dependence of g_mumu and g_1 (with phase 0)

def g_mumu_m1_pha0(g_mumu, g1, delm_sunsqua, theta_sun):
    denom_1 = (g_mumu - g1 * np.sin(theta_sun)**2) / (np.cos(theta_sun)**2 * g1) #* np.exp(2j * delta)
    return (delm_sunsqua / (denom_1**2 - 1))**(1/2)


# m_1 in dependence of g_tautau and g_1 (with phase 0)

def g_tautau_m1_pha0(g_tautau, g1, delm_sunsqua, delm_atmsqua):
    denom_1 = g_tautau/g1 #* np.exp(2j * delta)
    return ((delm_sunsqua + delm_atmsqua) / (denom_1**2 - 1))**(1/2)


## Functions to calculate the g1-cutoffs from limits on m1 and/or intersections
#  g1-cutoff for g_ee

def g1_g_ee_cutoff_pha0(g_ee, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.cos(theta_sun)**2 + np.sqrt(1 + delm_sunsqua/m1_lim**2) * np.sin(theta_sun)**2
    return g_ee/denom_1

def g1_g_ee_cutoff_pha90(g_ee, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.cos(theta_sun)**2 - np.sqrt(1 + delm_sunsqua/m1_lim**2) * np.sin(theta_sun)**2
    return g_ee/denom_1


# g1-cutoff for g_emu intersection

def g1_g_emu_cutoff(g_low, g_upper, theta_sun):
    denom_1 = (g_low - g_upper) * np.sin(2 * theta_sun)
    return (g_upper**2 - g_low**2) / denom_1


# g1-cutoff for g_mumu from m1 limit

def g1_g_mumu_cutoff_pha0(g_mumu, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.sin(theta_sun)**2 + np.sqrt(1 + delm_sunsqua / m1_lim**2) * np.cos(theta_sun)**2
    return g_mumu/denom_1

def g1_g_mumu_cutoff_pha90(g_mumu, m1_lim, delm_sunsqua, theta_sun):
    denom_1 = np.sin(theta_sun)**2 - np.sqrt(1 + delm_sunsqua / m1_lim**2) * np.cos(theta_sun)**2
    return g_mumu/denom_1


# g1-cutoff for g_tautau from m1 limit

def g1_g_tautau_cutoff(g_tautau, m1_lim, delm_sunsqua, delm_atmsqua):
    denom_1 = 1 + (delm_sunsqua + delm_atmsqua) / m1_lim**2
    return g_tautau / np.sqrt(denom_1)


# g_ee g_emu intersection

def g_ee_g_emu_intersect(g_ee, g_emu, theta_sun):
    enum_1  = g_ee / np.sin(theta_sun)**2 - 2 * g_emu / np.sin(2*theta_sun)
    denom_1 = 1 + np.cos(theta_sun)**2 / np.sin(theta_sun)**2
    return enum_1 / denom_1

def g_ee_g_emu_intersect_comp_pos(g_ee, g_emu, theta_sun):
    p = 1/(1 - 2*np.cos(theta_sun)**2) * (2*g_emu / np.sin(2*theta_sun) * np.sin(theta_sun)**4 + g_ee * np.cos(theta_sun)**2)
    q = 1/(1 - 2*np.cos(theta_sun)**2) * (4*g_emu**2 / np.sin(2*theta_sun)**2 * np.sin(theta_sun)**4 - g_ee**2)
    return - p + np.sqrt(p**2 -q)

def g_ee_g_emu_intersect_comp_neg(g_ee, g_emu, theta_sun):
    p = 1/(1 - 2*np.cos(theta_sun)**2) * (2*g_emu / np.sin(2*theta_sun) * np.sin(theta_sun)**4 + g_ee * np.cos(theta_sun)**2)
    q = 1/(1 - 2*np.cos(theta_sun)**2) * (4*g_emu**2 / np.sin(2*theta_sun)**2 * np.sin(theta_sun)**4 - g_ee**2)
    return - p - np.sqrt(p**2 -q)


# g1-cutoff for g_ee intersection

def g1_g_ee_cutoff(g_low, g_upper, theta_sun):
    denom_1 = 2 * (g_upper - g_low) * np.cos(theta_sun)**2
    return (g_upper**2 - g_low**2) / denom_1



delm_sunsqua = 7.53 * 10**(-5)                   # new value from the PDG                                        
delm_atmsqua = 2.453 * 10**(-5)                  # new value from the PDG

delta1 = 0                                       # CP-violating phase, zero for now
delta2 = np.pi/2


theta_sun = np.arcsin(np.sqrt(0.307))/2
sinsqua_sun = np.sin(theta_sun)**2   

g1 = np.logspace(start = -12, stop = -3, num = 2 * 10**6, base = 10.0)      

# upper limit on m1

m1_lim = 0.8

# lower bounds:
g_ee_low     = 3 * 10**(-7)
g_emu_low    = 3 * 10**(-7)
g_mumu_low   = 3 * 10**(-7)
g_tautau_low = 3 * 10**(-7)

g_ee_neglow     = -3 * 10**(-7)
g_emu_neglow    = -3 * 10**(-7)
g_mumu_neglow   = -3 * 10**(-7)
g_tautau_neglow = -3 * 10**(-7)
# upper bounds:
g_ee_upper     = 2 * 10**(-5)               # updated upper bounds on g_ee
g_emu_upper    = 2 * 10**(-5)
g_mumu_upper   = 2 * 10**(-5)
g_tautau_upper = 2 * 10**(-5)

g_ee_negupper     = -2 * 10**(-5)
g_emu_negupper    = -2 * 10**(-5)
g_mumu_negupper   = -2 * 10**(-5)
g_tautau_negupper = -2 * 10**(-5)


# print(g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun))

print('g_ee cutoff with phase 0: ',g1_g_ee_cutoff_pha0(g_ee_low, m1_lim, delm_sunsqua, theta_sun), g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun))
print('g_ee cutoff with phase 90: ',g1_g_ee_cutoff_pha90(g_ee_low, m1_lim, delm_sunsqua, theta_sun), g1_g_ee_cutoff_pha90(g_ee_upper, m1_lim, delm_sunsqua, theta_sun))

### Intersections between different bounds on g_ee

g_ee_cutoff_neglower = g1_g_ee_cutoff(g_ee_neglow, g_ee_upper, theta_sun)
print('Intersection between negative lower bound and upper bound on g_ee: ', g_ee_cutoff_neglower)

g_ee_cutoff_lower = g1_g_ee_cutoff(g_ee_low, g_ee_upper, theta_sun)
print('Intersection between positive lower bound and upper bound on g_ee: ', g_ee_cutoff_lower)

print('Asymptotischer Konvergenzwert m1: ', g_ee_m1_pha0(g_ee_low, 10**(-3), delm_sunsqua, sinsqua_sun))


# limited logspaces from cutoffs

### for g1_ee
num = 2 * 10**6  # Number of elements in the logspace array

# lower phase 0 cutoff
g1_ee_low_pha0 = np.logspace(-12, np.log10(g1_g_ee_cutoff_pha0(g_ee_low, m1_lim, delm_sunsqua, theta_sun)), num)

# lower phase 90 cutoff
g1_ee_low_pha90 = np.logspace(np.log10(g1_g_ee_cutoff_pha90(g_ee_low, m1_lim, delm_sunsqua, theta_sun)), -3, num)


# upper phase 0 cutoff
g1_ee_upper_pha0 = np.logspace(-12, np.log10(g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), num)

# upper phase 90 cutoff
g1_ee_upper_pha90 = np.logspace(np.log10(g1_g_ee_cutoff_pha90(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), -3, num)

# g_ee lower bound with phase 0

plt.plot(g1_ee_low_pha0, g_ee_m1_pha0(g_ee_low, g1_ee_low_pha0, delm_sunsqua, sinsqua_sun), color='blue', label=r'positive bounds on $g_{ee}$ for phase $\delta = 0$')
plt.plot(g1_ee_low_pha90, g_ee_m1_pha0(g_ee_low, g1_ee_low_pha90, delm_sunsqua, sinsqua_sun), color='blue', linestyle = 'dashed', label=r'positive bounds on $g_{ee}$ for phase $\delta = \frac{\pi}{2}$')

plt.plot(g1, g_ee_m1_pha0(g_ee_neglow, g1, delm_sunsqua, sinsqua_sun), color='teal',label=r'negative bounds on $g_{ee}$')

# g_ee upper bound

plt.plot(g1_ee_upper_pha0, g_ee_m1_pha0(g_ee_upper, g1_ee_upper_pha0, delm_sunsqua, sinsqua_sun), color='blue')
plt.plot(g1_ee_upper_pha90, g_ee_m1_pha0(g_ee_upper, g1_ee_upper_pha90, delm_sunsqua, sinsqua_sun), color='blue', linestyle = 'dashed',)
plt.plot(g1, g_ee_m1_pha0(g_ee_negupper, g1, delm_sunsqua, sinsqua_sun), color='teal')


# m1 limit from KATRIN
g1_col = np.logspace(-12, -3, 1000)
plt.fill_between(g1_col, np.full_like(g1_col, m1_lim), np.full_like(g1_col, 1), color='red', alpha=0.5, linewidth=0)

# fill in
num1 = 100
g1_fillin1 = np.logspace(-12, np.log10(g_ee_cutoff_neglower), num1)
g1_fillintest = np.logspace(np.log10(g_ee_cutoff_neglower), np.log10(g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), num1)
g1_fillin2 = np.logspace(np.log10(g1_g_ee_cutoff_pha90(g_ee_low, m1_lim, delm_sunsqua, theta_sun)), np.log10(g_ee_cutoff_lower), num1)
g1_fillin3 = np.logspace(np.log10(g_ee_cutoff_lower), np.log10(g1_g_ee_cutoff_pha0(g_ee_upper, m1_lim, delm_sunsqua, theta_sun)), num1)
plt.fill_between(g1_fillin1, g_ee_m1_pha0(g_ee_neglow, g1_fillin1, delm_sunsqua, sinsqua_sun), g_ee_m1_pha0(g_ee_upper, g1_fillin1, delm_sunsqua, sinsqua_sun), color='blue', alpha=0.5, linewidth=0)
plt.fill_between(g1_fillin2, g_ee_m1_pha0(g_ee_low, g1_fillin2, delm_sunsqua, sinsqua_sun), np.full_like(g1_fillin2, m1_lim),  color='blue', alpha=0.5, linewidth=0)
plt.fill_between(g1_fillin3, g_ee_m1_pha0(g_ee_upper, g1_fillin3, delm_sunsqua, sinsqua_sun), np.full_like(g1_fillin3, m1_lim), color='blue', alpha=0.5, linewidth=0)
plt.xlabel(r'$g_1$')
plt.xlim(10**(-11), 10**(-3))
plt.xscale('log')

plt.ylabel(r'$m_1 \mathbin{/} \mathrm{eV}$')
plt.ylim(10**(-6), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/m1g1g_ee.pdf')
plt.clf()