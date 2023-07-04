import numpy as np
import matplotlib.pyplot as plt

def g_ee_pha0(g1, theta_sun, m1, delm_sunsqua):
    g2 = g1 * np.sqrt(1 + delm_sunsqua / m1**2)
    return g1 * np.cos(theta_sun)**2 + g2 * np.sin(theta_sun)**2

def g_ee_pha90(g1, theta_sun, m1, delm_sunsqua):
    g2 = g1 * np.sqrt(1 + delm_sunsqua / m1**2)
    return g1 * np.cos(theta_sun)**2 - g2 * np.sin(theta_sun)**2

delm_sunsqua = 1 * 10**(-5)                                        # solar mass^2, still need to look up value
delm_atmsqua = 3 * 10**(-3)                                       # atmospheric mass^2, still need to look up value

delta1 = 0                                       # CP-violating phase, zero for now
delta2 = np.pi/2


theta_sun = np.arcsin(np.sqrt(0.6))/2
sinsqua_sun = np.sin(theta_sun)**2                                       # sun_sin = np.sin(theta_sun)**2


g1 = np.linspace(1*10**(-10), 1, 2 * 10**6)
m1 = np.linspace(1*10**(-10), 1, 2 * 10**6)

#g1,m1 = np.meshgrid(g1,m1)
#g_eepha0 = g_ee_pha0(g1, theta_sun, m1, delm_sunsqua)
#g_eepha90 = g_ee_pha90(g1, theta_sun, m1, delm_sunsqua)
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(g1, m1, g_eepha0, 100, c=g_eepha0, cmap='viridis')
#ax.contour3D(g1, m1, g_eepha90, 100, c=g_eepha90, cmap='viridis')
#ax.set_xlabel(r'$g_1$')
#ax.set_ylabel(r'$m_1$')
#ax.set_zlabel(r'$g_{ee}$')
#ax.view_init(90, 45)
#ax.xscale('log')
#ax.yscale('log')

plt.plot(g1, g_ee_pha0(g1, theta_sun, m1, delm_sunsqua), label=r'$g_{ee}$ with phase $\delta = 0$')
plt.plot(g1, g_ee_pha90(g1, theta_sun, m1, delm_sunsqua), label=r'$g_{ee}$ with $\delta = \frac{\pi}{2}$')
plt.xlabel(r'$g_1$')
#plt.xlim(10**(-10), 10**(-3))
plt.xscale('log')

plt.ylabel(r'$g_{ee}$')
#plt.ylim(10**(-6), 1)
plt.yscale('log')

plt.grid(linestyle = ":")
plt.tight_layout()
plt.legend()
plt.savefig('build/g_ee.pdf')
plt.clf()