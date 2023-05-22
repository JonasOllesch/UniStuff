import numpy as np
import matplotlib.pyplot as plt

# m_1 in dependence of g_ee and g_1 (with phase 0)

def g_ee_m1_pha0(g_ee, g1, delm_sunsqua, sinsqua_sun):
    denom_1 = (g_ee - g1 * (1- sinsqua_sun)) / (sinsqua_sun * g1) #* np.exp(2j * delta)
    return (delm_sunsqua / (denom_1**2 - 1))**(1/2)

delm_sunsqua = 1 * 10**(-5)                                        # solar mass^2, still need to look up value
delm_atmsqua = 3 * 10**(-3)                                       # atmospheric mass^2, still need to look up value

delta1 = 0                                       # CP-violating phase, zero for now
delta2 = np.pi/2


theta_sun = np.arcsin(np.sqrt(0.6))/2
sinsqua_sun = np.sin(theta_sun)**2                                       # sun_sin = np.sin(theta_sun)**2

#g1 = np.linspace(1*10**(-10), 10**(-3), 1 * 10**6)       # 20 * 10**6 entries are enough to make both plots grow to m1 = 1 :) Yeah not anymore, going to need a bit more than that
g1 = np.logspace(start = -10, stop = -3, num = 2 * 10**6, base = 10.0)

# lower bounds:
g_ee_low    =  3 * 10**(-7)
g_ee_neglow = -3 * 10**(-7)

# upper bounds:
g_ee_upper    =  2 * 10**(-5)
g_ee_negupper = -2 * 10**(-5)

# positive and negative lower bound 

plt.plot(g1, g_ee_m1_pha0(g_ee_low, g1, delm_sunsqua, sinsqua_sun), color='blue',label=r'bounds on $g_{ee}$')
plt.plot(g1, g_ee_m1_pha0(g_ee_neglow, g1, delm_sunsqua, sinsqua_sun), color='blue', linestyle='dashed')

# g_ee lower bound with phase pi/2

#plt.plot(g1, g_ee_m1_pha90(g_ee_low, g1, delm_sunsqua, sinsqua_sun), color='yellow',label=r'lower bound on $g_{ee}$')  # It seems like I don't need this

# g_ee upper bound (pos & neg)

plt.plot(g1, g_ee_m1_pha0(g_ee_upper, g1, delm_sunsqua, sinsqua_sun), color='blue')
plt.plot(g1, g_ee_m1_pha0(g_ee_negupper, g1, delm_sunsqua, sinsqua_sun), color='blue', linestyle='dashed')


plt.xlabel(r'$g_1$')
plt.xlim(10**(-10), 10**(-3))
plt.xscale('log')

plt.ylabel(r'$m_1$')
plt.ylim(10**(-6), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/g_eeg_1m_1test.pdf')
plt.clf()