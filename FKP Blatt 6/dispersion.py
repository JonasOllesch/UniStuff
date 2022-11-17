import numpy as np
import matplotlib.pyplot as plt

a  = 1
m  = 1
M  = 1
D1 = 10
D2 = 5
D3 = 5

k = np.linspace(-np.pi/a,np.pi/a,10000)

alpha = -(2*(m*D3 + M*D2)*(np.sin(k*a/2))**2 
        + D1 * (m + M))

beta = (D1*(np.sin(k*a/2))**2 + 
        (2*D1 * (D2 + D3) + 4 * D2*D3 * (np.sin(k*a/2))**2)) * (np.sin(k*a/2))**2

print('alpha: ', alpha)
print('beta: ', beta)

omega_plus = np.sqrt(-alpha/(2*m*M) + np.sqrt((alpha/(2*m*M))**2 - beta/(m*M)))
omega_minus = np.sqrt(-alpha/(2*m*M) - np.sqrt((alpha/(2*m*M))**2 - beta/(m*M)))
plt.plot(k, omega_plus, label=r'$\omega_+(k)$')
plt.plot(k, omega_minus, label=r'$\omega_-(k)$')
plt.title(r'$\omega(k)$')
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a],[r'$-\frac{\pi}{a}$',r'$-\frac{\pi}{2a}$', r'$0$',r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$'])
plt.xlim(-np.pi/a,np.pi/a)
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega(k)$')
plt.ylim(0,8)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('dispersion.pdf')