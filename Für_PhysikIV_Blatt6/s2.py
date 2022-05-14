import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0,100,1000000)
a_B = 5.294 * 10**(-11)
#a_p = 4
def s1(Z):
    f = 1/np.sqrt(np.pi)*(Z/a_B)**(3/2) * (1 - r/2) * np.exp(-Z*r/a_B)
    return f

f = 1/np.sqrt(np.pi) * (1 - r/2) * np.exp(-r)
plt.plot(r, f)
plt.xlim(0,20)
plt.ylim(-0.1, 0.7)
plt.xlabel(r'$\dfrac{r}{a_B}$')
plt.ylabel(r'$\Psi (r)$') #Hier gibt es keine Einheit mehr, weil das nur ein Intensitätsverhältnis ist
plt.tight_layout()
plt.grid()
plt.savefig('Graph_b.pdf')
plt.clf()