import numpy as np
import matplotlib.pyplot as plt
from numpy import math
from scipy.optimize import minimize

def random_poisson(la,number):
    return np.random.poisson(lam=la,size=number)


def n_log_likelihood(lam):
    return -np.log(np.float64(math.factorial(8)*np.math.factorial(9)*np.math.factorial(13)))-30*np.log(lam)+3*lam

def n_log_likelihood_m_mlh_p1d2(lam):
    return abs(-30*np.log(lam)+3*lam +30*np.log(30)-30 -1/2)

minimum_nllh=n_log_likelihood(10)


#print(minimize(n_log_likelihood_m_mlh_p1d2,10))
#print(minimize(n_log_likelihood_m_mlh_p2,10))
#print(minimize(n_log_likelihood_m_mlh_p9d2,10))

#x = np.linspace(0,20,50)
#y = n_log_likelihood_m_mlh_p1d2(x)
#print(y)
#plt.plot(x,y)
#plt.savefig("test.pdf")

print(minimize(n_log_likelihood_m_mlh_p1d2,x0=10))