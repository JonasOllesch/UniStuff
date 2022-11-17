import numpy as np
import matplotlib.pyplot as plt

a = 1
c1 = 5
x = np.linspace(-np.pi/a,np.pi/a,10000)

y = 2 * np.sqrt(c1*(2*(np.sin(x*a/2))**2 + (np.sin(x*a))**2))
y_approx = 2*np.sqrt(c1*(2*(x*a/2)**2 + (x*a)**2))

plt.plot(x, y, label=r'$\omega(k)$')
plt.plot(x,y_approx, label=r'$\omega(k)$ in linear approximation for $k \rightarrow 0$')
plt.title(r'$\omega(k)$, here for $a=1, m=1, c_2=1$')
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a],[r'$-\frac{\pi}{a}$',r'$-\frac{\pi}{2a}$', r'$0$',r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$'])
plt.xlim(-np.pi/a,np.pi/a)
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega(k)$')
plt.ylim(0,8)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('dispersion.pdf')