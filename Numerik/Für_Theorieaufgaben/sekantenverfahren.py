import numpy as np

def f(x):
    return x**3 - 3/2*x**2 - 2*x + 1  

def invdifquo(x_0,x_1):
    #f_0 = x_0**3 - 3/2*x_0**2 - 2*x_0 + 1
    #f_1 = x_1**3 - 3/2*x_1**2 - 2*x_1 + 1
    #x_2 = x_1 - (x_1 - x_0)/(f_1 - f_0) * f_1
    ret = (x_1 - x_0)/(f(x_1) - f(x_0)) * f(x_1)
    return ret

def difquo(x_0,x_1):
    ret = (f(x_1) - f(x_0))/(x_1 - x_0)
    return ret



def sek(x_0,x_1):
    x_2 = x_1 - invdifquo(x_0,x_1)
    return x_2
x_0 = 1
x_1 = 0
x_2 = sek(x_0,x_1)


x_3 = sek(x_1,x_2)


x_4 = sek(x_2,x_3)


print('f(x_0) = ',f(x_0))
print('f(x_1) =',f(x_1))
print('f(x_2) =',f(x_2))
print('f(x_3) =',f(x_3))

print('difquo(x_1) = ', difquo(x_0,x_1))
print('difquo(x_2) = ', difquo(x_1,x_2))
print('difquo(x_3) =', difquo(x_2,x_3))

print('x_2 =', x_2)
print('x_3 =', x_3)
print('x_4 =', x_4)