import numpy as np
import matplotlib.pyplot as plt

a  = 1
m  = 1
M  = 1
D1 = 8
D2 = 1
D3 = 1

d1 = 1
d2 = 1
d3 = 1

k = np.linspace(-np.pi/a,np.pi/a,10000)

def calc_alpha(a,m,M,D1,D2,D3):
        alpha = -2/(m*M)*(2*(D2*M + D3*m)*(np.sin(k*a/2))**2 + D1 * (m + M))
        return alpha

def calc_beta(a,m,M,D1,D2,D3): 
        beta = 8/(m*M) * (2*D2*D3*(np.sin(k*a/2))**2 + D1*(D2 + D3 + 1/2*D1))*(np.sin(k*a/2)**2)
        return beta
alpha1 = calc_alpha(a,m,M,D1,D2,D3)
beta1 = calc_beta(a,m,M,D1,D2,D3)
#print('alpha: ', alpha1)
#print('beta: ', beta)

omega_plus1 =  np.sqrt(-alpha1/2 + np.sqrt((-alpha1/2)**2 - beta1))
omega_minus1 = np.sqrt(-alpha1/2 - np.sqrt((-alpha1/2)**2 - beta1))
plt.plot(k, omega_plus1, label=r'$\omega_+(k)$')
plt.plot(k, omega_minus1, label=r'$\omega_-(k)$')
plt.title(r'$\omega(k)$')
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a],[r'$-\frac{\pi}{a}$',r'$-\frac{\pi}{2a}$', r'$0$',r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$'])
plt.xlim(-np.pi/a,np.pi/a)
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega(k)$')
plt.title(r'Dispersionsrelation $\omega(k)$ für $a = m = M = 1, D_1 = 8, D_2 = D_3 = 1$.')
#plt.ylim(0,8)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('dispersion1.pdf')
plt.clf()

alpha2 = calc_alpha(a,m,M,d1,d2,d3)
beta2 = calc_beta(a,m,M,d1,d2,d3)

omega_plus2 =  np.sqrt(-alpha2/2 + np.sqrt((-alpha2/2)**2 - beta2))
omega_minus2 = np.sqrt(-alpha2/2 - np.sqrt((-alpha2/2)**2 - beta2))
plt.plot(k, omega_plus2, label=r'$\omega_+(k)$')
plt.plot(k, omega_minus2, label=r'$\omega_-(k)$')
plt.title(r'$\omega(k)$')
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a],[r'$-\frac{\pi}{a}$',r'$-\frac{\pi}{2a}$', r'$0$',r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$'])
plt.xlim(-np.pi/a,np.pi/a)
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega(k)$')
plt.title(r'Dispersionsrelation $\omega(k)$ für $a = m = M = 1, D_1 = D_2 = D_3 = 1$.')
#plt.ylim(0,8)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('dispersion2.pdf')
plt.clf()