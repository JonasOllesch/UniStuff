import numpy as np

def f(x):
    return x**3 - 3/2*x**2 - 2*x + 1  

def x_k(a_k,b_k):
    x_k = a_k - (b_k-a_k)/(f(b_k) - f(a_k)) * f(a_k)
    return x_k

def difquo(a_k,b_k):
    ret = (f(b_k)-f(a_k))/(b_k-a_k)
    return ret

x_0 = 1
a_0 = 1
b_0 = 0

x_1 = x_k(a_0,b_0)
print('x_1 =', x_1)

a_1 = a_0
b_1 = x_1

x_2 = x_k(a_1,b_1)

print('x_2 =', x_2)
print(f(x_2)*f(b_1))

print('f(x_1) =', f(x_1))
print('f(x_2) =', f(x_2))

a_2 = x_2
b_2 = b_1

print('Diffquo 0', difquo(a_0,b_0))
print('Diffquo 1', difquo(a_1,b_1))
print('Diffquo 2', difquo(a_2,b_2))