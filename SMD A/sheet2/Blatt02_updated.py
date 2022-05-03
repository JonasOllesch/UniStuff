

#den Median finden

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
import scipy as si
from scipy.integrate import quad

def func_g(x):
   return np.sqrt(2/np.pi)*x**2*np.exp((-x**2)/(2*a**2))/a**3

def anpassen(x):
      return abs(quad(func_g,0,x)[0]-1/2)
a= 20


xplot = np.linspace(0,a*5,100)
yplot = func_g(xplot)
plt.xlabel(r'$v\, in \, 1/s $')
plt.ylabel(r'$  p(v)  $')

plt.plot(xplot,yplot)
plt.grid()
plt.show()
quad(func_g,0,a*10)

print(fmin(anpassen,1), " Median")
print(fmin(anpassen,1) /(np.sqrt(2)*a), "  Median in x *sqrt(2)*a")


#full width at half maximum
def optimierung_bei_ymax_halbe_func_g(x):
   return abs(-abs(np.sqrt(2/np.pi)*x**2*np.exp((-x**2)/(2*a**2))/a**3)+y_max/2)


def func_g(x):
   return np.sqrt(2/np.pi)*x**2*np.exp((-x**2)/(2*a**2))/a**3
   
a= 20

xplot = np.linspace(0,a*5,100)
yplot = func_g(xplot)
plt.xlabel(r'$v\, in \, 1/s $')
plt.ylabel(r'$  p(v)  $')
x_max = np.sqrt(2)*a
y_max = func_g(x_max)
plt.plot(xplot,yplot)
plt.grid()


print(x_max, "x_max")
print(y_max, "y_max")

untere_grenze = fmin(optimierung_bei_ymax_halbe_func_g,a)
obere_grenze  = fmin(optimierung_bei_ymax_halbe_func_g,2*a)
v_fwhm = obere_grenze-untere_grenze

print(untere_grenze,"untere_grenze")
print(untere_grenze/(np.sqrt(2)*a),"untere_grenze in x*sqrt(2)a")
print(obere_grenze,"obere_grenze")
print(obere_grenze/(a*np.sqrt(2)),"obere_grenze in x*sqrt(2)*a")

print(v_fwhm,"v_fwhm")
print(v_fwhm/(np.sqrt(2)*a),"v_fwhm in x *a")
plt.scatter(untere_grenze,func_g(untere_grenze) ,c='r', label="Grenze")
plt.scatter(obere_grenze,func_g(obere_grenze),c="r")

plt.show()