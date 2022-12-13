import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial

# params[0] holds m, params[1] holds n as values for y = mx +n

def calc_lin_likelihood(params):
    return (7*params[1] + 28 * params[0] 
    - (data[0]*np.log(params[0] + params[1]) + 
    data[1]*np.log(2*params[0] + params[1]) 
    + data[2]*np.log(3*params[0] + params[1])
    + data[3]*np.log(4*params[0] + params[1]) 
    + data[4]*np.log(5*params[0] + params[1]) 
    + data[5]*np.log(6*params[0] + params[1]) 
    + data[6]*np.log(7*params[0] + params[1])))
    
def calc_likelihood(lamb):
    return (7*lamb - np.sum(data)*np.log(lamb))

def calc_new_lin_likelihood(params):
    return (8*params[1] + 42 * params[0] 
    - (new_data[0]*np.log(params[0] + params[1]) + 
    new_data[1]*np.log(2*params[0] + params[1]) 
    + new_data[2]*np.log(3*params[0] + params[1])
    + new_data[3]*np.log(4*params[0] + params[1]) 
    + new_data[4]*np.log(5*params[0] + params[1]) 
    + new_data[5]*np.log(6*params[0] + params[1]) 
    + new_data[6]*np.log(7*params[0] + params[1])
    + new_data[7]*np.log(14*params[0] + params[1])))

data     = np.array([4135, 4202, 4203, 4218, 4227, 4231, 4310])
new_data = np.array([4135, 4202, 4203, 4218, 4227, 4231, 4310,4402])

lamb = np.mean(data)
new_lamb = np.mean(new_data)


minmin = minimize(calc_lin_likelihood, x0 = [30.0,4000.0])

print('minmin: ', minmin)

signifi = calc_lin_likelihood(minmin.x)/calc_likelihood(lamb)

print('Significance: ', signifi)
print('Test Statistic: ', -2*np.log(signifi))


new_minmin = minimize(calc_new_lin_likelihood, x0 = [30.0,4000.0])

print('new_minmin: ', new_minmin)

new_signifi = calc_new_lin_likelihood(new_minmin.x)/calc_likelihood(new_lamb)

print('New Significance: ', new_signifi)
print('New Test Statistic: ', -2*np.log(new_signifi))