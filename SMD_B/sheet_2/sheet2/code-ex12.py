import numpy as np
from uncertainties import unumpy
from uncertainties import ufloat

def calc_A_0 (a1,a2):
    return  a1/(unumpy.cos(unumpy.arctan(a2/a1)))

def calc_delta(a1,a2):
    return (-1)*unumpy.arctan(a2/a1)

Data = np.array(np.genfromtxt('Daten.txt'))
A = np.zeros(shape=(12,2))
A[:,0]= np.cos(Data[:,0]*(2*np.pi/360))
A[:,1]= np.sin(Data[:,0]*(2*np.pi/360))

print('This is A:', A)

A_T= A.transpose()
tmp1 = np.matmul(A_T,A)
tmp2 = np.linalg.inv(tmp1)
tmp3 = np.matmul(tmp2,A_T)
a = np.matmul(tmp3,Data[:,1])
print( "this is a -->",a)
Var_a = 0.011**2*tmp2
print(Var_a)
Err_a1 = np.sqrt(Var_a[0][0])
Err_a2 = np.sqrt(Var_a[1][1])
a1 = ufloat(a[0],Err_a1)
a2 = ufloat(a[1],Err_a2)
print("a1" , repr(a1))
print("a2" , repr(a2))
A_0 = calc_A_0(a1, a2)
delta = calc_delta(a1, a2)
print("A_0",    repr(A_0))
print("delta",  repr(delta*360/(2*np.pi)))