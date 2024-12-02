import matplotlib.pyplot as plt
import numpy as np

def dirichlet_kernel(L, q):
    return np.sin((L + 1/2)*q) / (np.sin(q/2))

q = np.linspace(-30*np.pi, 30*np.pi, 100000)
a = 1
L_large = 500

plt.scatter(q, dirichlet_kernel(L_large, q*a), label=r'$\Delta(q)$ für $L=500$', s=1)
plt.xlabel(r'$q$')
plt.ylabel(r'$\Delta(q)$')
plt.title(r'Dirichlet-Kernel $\Delta(q)$ für großes $L$.')
plt.legend()

plt.savefig('dirichlet_kernel.pdf')
plt.clf()

## Zoom in for small q

q = np.linspace(-np.pi, np.pi, 10000)
a = 1   

plt.scatter(q, dirichlet_kernel(L_large, q*a), label=r'$\Delta(q)$ für $L=500$', s=1)
plt.xlim(-0.1, 0.1)
plt.xlabel(r'$q$')
plt.ylabel(r'$\Delta(q)$')
plt.title(r'Dirichlet-Kernel $\Delta(q)$ für großes $L$ im Bereich $q \in [-0.1, 0.1]$.')
plt.legend()

plt.savefig('dirichlet_kernel_zoom.pdf')
plt.clf()