import pandas as pnd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
#a
ex_a_dataframe = pnd.read_csv('ex_a.csv')
ex_a = ex_a_dataframe.to_numpy()
ex_a_x = ex_a[:, 0]
ex_a_y = ex_a[:, 1]

ex_a_fit = Polynomial.fit(ex_a_x, ex_a_y, 6)

print('fitting_polynomial is ', ex_a_fit)
x_a_fit, y_a_fit = ex_a_fit.linspace()
plt.plot(x_a_fit, y_a_fit, 'g')
plt.scatter(ex_a_x, ex_a_y)
plt.savefig
#b)
A = np.zeros((7,7))
print(ex_a)
A[0,:]= ex_a[:,0]







#c
ex_c_dataframe = pnd.read_csv('ex_c.csv')
ex_c = ex_c_dataframe.to_numpy()
ex_c_x = ex_c[:, 0]
ex_c_y_all = (ex_c[:, 1:51])
ex_c_y = np.mean(ex_c_y_all, axis=1)
#print(ex_c_y)

#print(np.size(ex_c_x), np.size(ex_c_y))
plt.scatter(ex_c_x, ex_c_y)
ex_c_fit = Polynomial.fit(ex_c_x, ex_c_y, 6)
x_c_fit, y_c_fit = ex_c_fit.linspace()
plt.plot(x_c_fit, y_c_fit, 'g')
plt.scatter(ex_c_x, ex_c_y)



