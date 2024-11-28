import numpy as np
import matplotlib.pyplot as plt

def omega(k, coupling_1, coupling_2, mass, a): # assume equal masses
    root = np.sqrt(coupling_1**2 + coupling_2**2 + 2*coupling_1*coupling_2*np.cos(a*k))
    omega_plus = np.sqrt((coupling_1 + coupling_2 + root)/mass)
    omega_minus = np.sqrt((coupling_1 + coupling_2 - root)/mass)
    return omega_plus, omega_minus

a = 1 # lattice constant
K = 1
mass = 1
k = np.linspace(0, 4*np.pi/a + 1e-10, 10000) # k values


### CASE (i)

coupling_1_1 = K
coupling_1_2 = K/2

plt.plot(k, omega(k, coupling_1_1, coupling_1_2, mass, a)[0], label=r'$\omega_+$', color ='midnightblue')
plt.plot(k, omega(k, coupling_1_1, coupling_1_2, mass, a)[1], label=r'$\omega_-$', color = 'firebrick')
plt.xticks([np.pi/a, 2*np.pi/a, 3*np.pi], [r'$-\pi/a$', '0', r'$+\pi/a$'])
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega(k)$')
plt.legend(loc='lower right')
plt.grid()
plt.title(r'$\omega(k)$ for $K_1 = K, K_2 = K/2$ with mass $m = 1$')
plt.savefig('omega_plots_case_1.pdf')
plt.clf()


### CASE (ii)

coupling_2_1 = K
coupling_2_2 = K

plt.plot(k, omega(k, coupling_2_1, coupling_2_2, mass, a)[0], label=r'$\omega_+$', color ='midnightblue')
plt.plot(k, omega(k, coupling_2_1, coupling_2_2, mass, a)[1], label=r'$\omega_-$', color = 'firebrick')
plt.xticks([np.pi/a, 2*np.pi/a, 3*np.pi], [r'$-\pi/a$', '0', r'$+\pi/a$'])
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega$(k)')
plt.legend(loc='lower right')
plt.grid()
plt.title(r'$\omega(k)$ for $K_1 = K, K_2 = K$ with mass $m = 1$')
plt.savefig('omega_plots_case_2.pdf')
plt.clf()