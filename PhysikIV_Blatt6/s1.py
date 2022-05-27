import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0,100,1000000)
a_B = 5.294 * 10**(-11)
#a_p = 4
def s1(Z):
    f = 1/np.sqrt(np.pi)*(Z/a_B)**(3/2) * np.exp(-Z*r/a_B)
    return f
g = s1(1)
f = 1/np.sqrt(np.pi) * np.exp(-r)
plt.plot(r, f)
#plt.plot(r,g)
plt.xlim(0,20)
#plt.ylim(0,0.5)
plt.xlabel(r'$\dfrac{r}{a_B}$')
plt.ylabel(r'$\Psi (r)$') 
plt.tight_layout()
plt.grid()
plt.savefig('Graph_a.pdf')
plt.clf()


p_1 = 4 * r**2 *np.exp(-2*r)

plt.plot(r, p_1)
#plt.plot(r,g)
plt.xlim(0,20)
#plt.ylim(0,0.5)
plt.xlabel(r'$\dfrac{r}{a_B}$')
plt.ylabel(r'$P(r)$') 
plt.tight_layout()
plt.grid()
plt.savefig('Graph_c.pdf')
plt.clf()


p_2 = 1/4 * r**2 *(1/2*r - 1)**2 *np.exp(-2*r)

plt.plot(r, p_2)
#plt.plot(r,g)
plt.xlim(0,20)
#plt.ylim(0,0.5)
plt.xlabel(r'$\dfrac{r}{a_B}$')
plt.ylabel(r'$P(r)$') 
plt.tight_layout()
plt.grid()
plt.savefig('Graph_d.pdf')
plt.clf()