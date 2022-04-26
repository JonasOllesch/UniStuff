import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

#Aufgabe2

gamma= 50/0.511#GeV
beta = np.sqrt(1-gamma**(-2))
Ex=50

def f(Theta):
    return ((5-np.cos(2*Theta))/(1-np.cos(2*Theta)+1/gamma**2*(1+np.cos(2*Theta))))
    
def g(Theta):
    return (2+np.sin(Theta)**2)/(1-beta**2*np.cos(Theta)**2)

def h(Theta):
    return gamma**2*(3+np.cos(2*Theta))/(gamma**2*(1-np.cos(2*Theta))+(1+np.cos(2*Theta)))


#Theta = np.linspace(np.pi/2-0.0005,np.pi/2+0.0005,1000,dtype="float32")
def conditionnumber2(Theta):
    return abs(Theta*(2*np.sin(Theta)*np.cos(Theta)*(3*beta**2-1))/((1-beta**2*np.cos(Theta)**2)*(2+np.sin(Theta)**2)))



Theta=np.linspace(0,np.pi,10000,dtype="float32")
plt.plot(Theta,conditionnumber2(Theta),label='orginal')
#plt.plot(Theta,f(Theta),label='links')
#plt.plot(Theta,g(Theta)-f(Theta),label='differenz')
#plt.plot(Theta,h(Theta),label='rechts')
plt.legend(loc='best')
plt.show()

