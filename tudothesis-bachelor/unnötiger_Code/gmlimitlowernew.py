import numpy as np
import matplotlib.pyplot as plt

# m_1 in dependence of g_ee and g_1 (with phase 0)

def g_ee_m1_pha0(g_ee, g1, delm_sunsqua, sinsqua_sun):
    denom_1 = (g_ee - g1 * (1- sinsqua_sun)) / (sinsqua_sun * g1) #* np.exp(2j * delta)
    return (delm_sunsqua / (denom_1**2 - 1))**(1/2)

# def g_ee_m1_pha90(g_ee, g1, delm_sunsqua, sinsqua_sun):
#     denom_1 = -1 * ((g_ee - g1 * (1- sinsqua_sun)) / (sinsqua_sun * g1)) #* np.exp(2j * delta)
#     return (delm_sunsqua / (denom_1**2 - 1))**(1/2)


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



delm_sunsqua = 7.53 * 10**(-5)                   # new value from the PDG                                        
delm_atmsqua = 2.453 * 10**(-5)                  # new value from the PDG

delta1 = 0                                       # CP-violating phase, zero for now
delta2 = np.pi/2


theta_sun = np.arcsin(np.sqrt(0.546))/2
sinsqua_sun = np.sin(theta_sun)**2                                       # sun_sin = np.sin(theta_sun)**2

g1 = np.logspace(start = -12, stop = -3, num = 2 * 10**6, base = 10.0)       # 20 * 10**6 entries are enough to make both plots grow to m1 = 1 :) Yeah not anymore, going to need a bit more than that


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
g_ee_upper     = 2 * 10**(-6)                               # updated upper bounds on g_ee
g_emu_upper    = 2 * 10**(-5)
g_mumu_upper   = 2 * 10**(-5)
g_tautau_upper = 2 * 10**(-5)

g_ee_negupper     = -2 * 10**(-6)
g_emu_negupper    = -2 * 10**(-5)
g_mumu_negupper   = -2 * 10**(-5)
g_tautau_negupper = -2 * 10**(-5)


print(g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun))

# g_ee lower bound with phase 0

plt.plot(g1, g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun), color='blue',label=r'bounds on $g_{ee}$ from $g_{ij} = 3 \cdot 10^{-7}$')
plt.plot(g1, g_ee_m1_pha0(g_ee_neglow, g1, delm_sunsqua, sinsqua_sun), color='teal', label=r'bounds on $g_{ee}$ from $g_{ij} = - 3 \cdot 10^{-7}$')

# g_eμ lower bound

plt.plot(g1, g_emu_m1_pha0(g_emu_low, g1, delm_sunsqua, theta_sun), color='orange', linestyle='dashed', label=r'bounds on $g_{e \mu^\prime}$ from $g_{ij} =  3 \cdot 10^{-7}$') ### Der ist gut!!
plt.plot(g1, g_emu_m1_pha0(g_emu_neglow, g1, delm_sunsqua, theta_sun), color='yellow', linestyle='dashed', label=r'bounds on $g_{e \mu^\prime}$ from $g_{ij} =  - 3 \cdot 10^{-7}$')

# g_μμ lower bound

#plt.plot(g1, g_mumu_m1_pha0(g_mumu_low, g1, delm_sunsqua, theta_sun), color='purple', linestyle='dotted', label=r'bounds on $g_{\mu^\prime \mu^\prime}$ from $g_{ij} = 3 \cdot 10^{-7}$')
plt.plot(g1, g_mumu_m1_pha0(g_mumu_neglow, g1, delm_sunsqua, theta_sun), color='pink', linestyle='dotted', label=r'bounds on $g_{\mu^\prime \mu^\prime}$ from $g_{ij} = - 3 \cdot 10^{-7}$')

# g_ττ lower bound

plt.plot(g1, g_tautau_m1_pha0(g_tautau_low, g1, delm_sunsqua, delm_atmsqua), color='red', linestyle='dashdot', label=r'bounds on $g_{\tau^\prime \tau^\prime}$')

# g_ττ upper bound

plt.plot(g1, g_tautau_m1_pha0(g_tautau_upper, g1, delm_sunsqua, delm_atmsqua), color='red', linestyle='dashdot', label=r'bounds on $g_{\tau^\prime \tau^\prime}$ from $g_{ij} = 2 \cdot 10^{-5}$')

plt.xlabel(r'$g_1$')
plt.xlim(10**(-11), 10**(-4))
plt.xticks(np.arange(10**(-11), 10**(-3)+10**(-10), step = 10**(-1)))
plt.xscale('log')

plt.ylabel(r'$m_1$')
plt.ylim(10**(-7), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/g_1m_1lowernew.pdf')
plt.clf()