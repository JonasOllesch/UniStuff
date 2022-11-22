import numpy as np
import matplotlib.pyplot as plt
from numpy import math
from scipy.optimize import minimize
from scipy.optimize import newton

def random_poisson(la,number):
    return np.random.poisson(lam=la,size=number)


def n_log_likelihood(lam):
    return -np.log(np.float64(math.factorial(8)*np.math.factorial(9)*np.math.factorial(13)))-30*np.log(lam)+3*lam


n_log_likelihood_x_min = minimize(n_log_likelihood, x0=10).fun


def difffunc1h2(x):
    return abs(abs(n_log_likelihood(x)) - abs(n_log_likelihood_x_min + 1/2))

def difffunc2(x):
    return abs(abs(n_log_likelihood(x)) - abs(n_log_likelihood_x_min + 2))

def difffunc9h2(x):
    return abs(abs(n_log_likelihood(x)) - abs(n_log_likelihood_x_min + 9/2))

x = np.linspace(1,20,100)
y = n_log_likelihood(x)

plt.plot(x,y,c='blue',label=r"$-\log{\mathcal{L}}$")
plt.legend()
plt.grid()
#plt.tight_layout()
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$-\ln{ \left( \mathcal{L} \right) }$")
plt.savefig("n_log_likelihood.pdf")
plt.clf()


#plt.tight_layout()
plt.plot(x,y,c='blue',label=r"$-\log{\mathcal{L}}$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$-\ln{ \left( \mathcal{L} \right) }$")
S1 = newton(difffunc1h2, x0=[9,12])
S2 = newton(difffunc2, x0=[8,13])
S3 = newton(difffunc9h2, x0=[7,14])

plt.scatter(S1, n_log_likelihood(S1),color='r',label=r"$\sigma \,$Intervall" )
plt.scatter(S2, n_log_likelihood(S2),color='r')
plt.scatter(S3, n_log_likelihood(S3),color='r')

plt.legend()
plt.grid()
plt.savefig("n_log_likelihood_mit_simga.pdf")
plt.clf()

def taylor2o(lam):
    return -np.log(np.float64(math.factorial(8)*np.math.factorial(9)*np.math.factorial(13)))-30*np.log(10)+3*10+ 30/200*(lam-10)**2

def difftaylor1h2(lam):
    return -np.log(np.float64(math.factorial(8)*np.math.factorial(9)*np.math.factorial(13)))-30*np.log(10)+3*10+ 30/200*(lam-10)**2 -n_log_likelihood_x_min -1/2
def difftaylor2(lam):
    return -np.log(np.float64(math.factorial(8)*np.math.factorial(9)*np.math.factorial(13)))-30*np.log(10)+3*10+ 30/200*(lam-10)**2 -n_log_likelihood_x_min -2
def difftaylor9h2(lam):
    return -np.log(np.float64(math.factorial(8)*np.math.factorial(9)*np.math.factorial(13)))-30*np.log(10)+3*10+ 30/200*(lam-10)**2 -n_log_likelihood_x_min -9/2




S1t = newton(difftaylor1h2,x0=[9,12])
S2t = newton(difftaylor2,x0=[8,13])
S3t = newton(difftaylor9h2,x0=[7,14])
print("")
print(S1t)
print(S2t)
print(S3t)
print("")
y_taylor = taylor2o(x)
plt.plot(x,y_taylor,c='green',label=r"$T_{-\log{\left(\mathcal{L}\right)};10}$")
plt.plot(x,y,c='blue',label=r"$-\log{\mathcal{L}}$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$-\ln{ \left( \mathcal{L} \right) }$")
S1 = newton(difffunc1h2, x0=[9,12])
S2 = newton(difffunc2, x0=[8,13])
S3 = newton(difffunc9h2, x0=[7,14])


plt.scatter(S1, n_log_likelihood(S1),color='r',label=r" $\sigma \,$Intervall" )
plt.scatter(S2, n_log_likelihood(S2),color='r')
plt.scatter(S3, n_log_likelihood(S3),color='r')

plt.scatter(S1t, taylor2o(S1t),color='#0ABAB5',label=r" $\sigma_{T} \,$Intervall" )
plt.scatter(S2t, taylor2o(S2t),color='#0ABAB5')
plt.scatter(S3t, taylor2o(S3t),color='#0ABAB5')



plt.legend()
plt.grid()
plt.savefig("n_log_likelihood_mit_simga_u_taylor.pdf")
plt.clf()
