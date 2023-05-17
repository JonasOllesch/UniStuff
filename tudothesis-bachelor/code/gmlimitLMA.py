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



delm_sunsqua = 1 * 10**(-5)                                        # solar mass^2, still need to look up value
delm_atmsqua = 3 * 10**(-3)                                       # atmospheric mass^2, still need to look up value

delta1 = 0                                       # CP-violating phase, zero for now
delta2 = np.pi/2


theta_sun = np.arcsin(np.sqrt(0.6))/2
sinsqua_sun = np.sin(theta_sun)**2                                       # sun_sin = np.sin(theta_sun)**2

g1 = np.linspace(1*10**(-15), 10**(-3), 2 * 10**6)       # 20 * 10**6 entries are enough to make both plots grow to m1 = 1 :) Yeah not anymore, going to need a bit more than that


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
g_ee_upper     = 2 * 10**(-5)
g_emu_upper    = 2 * 10**(-5)
g_mumu_upper   = 2 * 10**(-5)
g_tautau_upper = 2 * 10**(-5)

g_ee_negupper     = -2 * 10**(-5)
g_emu_negupper    = -2 * 10**(-5)
g_mumu_negupper   = -2 * 10**(-5)
g_tautau_negupper = -2 * 10**(-5)


print(g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun))

# g_ee lower bound with phase 0

plt.plot(g1, g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun), color='blue',label=r'bounds on $g_{ee}$')
plt.plot(g1, g_ee_m1_pha0(g_ee_neglow, g1, delm_sunsqua, sinsqua_sun), color='teal')

# g_ee lower bound with phase pi/2

#plt.plot(g1, g_ee_m1_pha90(g_ee_low, g1, delm_sunsqua, sinsqua_sun), color='yellow',label=r'lower bound on $g_{ee}$')  # It seems like I don't need this

# g_ee upper bound

plt.plot(g1, g_ee_m1_pha0(g_ee_upper, g1, delm_sunsqua, sinsqua_sun), color='blue')
plt.plot(g1, g_ee_m1_pha0(g_ee_negupper, g1, delm_sunsqua, sinsqua_sun), color='teal')



# g_eμ lower bound

plt.plot(g1, g_emu_m1_pha0(g_emu_low, g1, delm_sunsqua, theta_sun), color='green', linestyle='dashed', label=r'bounds on $g_{e \mu^\prime}$') ### Der ist gut!!
plt.plot(g1, g_emu_m1_pha0(g_emu_neglow, g1, delm_sunsqua, theta_sun), color='yellow', linestyle='dashed')

# g_eμ upper bound
plt.plot(g1, g_emu_m1_pha0(g_emu_upper, g1, delm_sunsqua, theta_sun), color='green', linestyle='dashed')
plt.plot(g1, g_emu_m1_pha0(g_emu_negupper, g1, delm_sunsqua, theta_sun), color='yellow', linestyle='dashed') ### Der auch, der macht den tollen Knick!


# g_μμ lower bound

plt.plot(g1, g_mumu_m1_pha0(g_mumu_low, g1, delm_sunsqua, theta_sun), color='red', linestyle='dotted', label=r'bounds on $g_{\mu^\prime \mu^\prime}$')
plt.plot(g1, g_mumu_m1_pha0(g_mumu_neglow, g1, delm_sunsqua, theta_sun), color='pink', linestyle='dotted')

# g_μμ upper bound

plt.plot(g1, g_mumu_m1_pha0(g_mumu_upper, g1, delm_sunsqua, theta_sun), color='red', linestyle='dotted')
plt.plot(g1, g_mumu_m1_pha0(g_mumu_negupper, g1, delm_sunsqua, theta_sun), color='pink', linestyle='dotted')

# g_ττ lower bound

plt.plot(g1, g_tautau_m1_pha0(g_tautau_low, g1, delm_sunsqua, delm_atmsqua), color='purple', linestyle='dashdot', label=r'bounds on $g_{\tau^\prime \tau^\prime}$')

# g_ττ upper bound

plt.plot(g1, g_tautau_m1_pha0(g_tautau_upper, g1, delm_sunsqua, delm_atmsqua), color='purple', linestyle='dashdot')

plt.xlabel(r'$g_1$')
plt.xlim(10**(-11), 10**(-3))
plt.xscale('log')

plt.ylabel(r'$m_1$')
plt.ylim(10**(-6), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/g_1m_1LMA.pdf')
plt.clf()