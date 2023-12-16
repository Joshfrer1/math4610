import numpy as np
import scipy.special as sp

def sin_f(x):
    return np.sin(x)

def sin_fdx(x):
    return np.cos(x)

def central_diff_quotient_5(x, h):
    return (sin_f(x + h) - sin_f(x - h))/(2*h)

def q_9_f(x):
    return sp.erf(x)

def q_9_fdx(x):
    return (2/np.sqrt(np.pi))*np.exp(-x**2)

def central_diff_quotient_9(x, h):
    return (q_9_f(x + h) - q_9_f(x - h))/(2 * h)

def q_12_f(x):
    return (1/np.tan(x))

def q_12_fdx(x):
    return -1 - ((1 + np.cos(2*x))/(1 - np.cos(2*x)))

def central_diff_quotient_12(x, h):
    return (q_12_f(x + h) - q_12_f(x - h))/(2 * h)

def q_6_f(x):
    return (x - 1)/(x + 1)

def q_6_fdx(x):
    return 2/((x + 1)**2)

def diff_quotient(x, h):
    return (q_6_f(x + h) - q_6_f(x))/h

def error(fdx, dq):
    return abs(fdx - dq)

def q_6_error(a):
    h = .5
    for i in range(0, 9):
        e = error(q_6_fdx(a), diff_quotient(a, h))
        h /= 2
        print(e)

def q_9_error(a):
    h = .5
    for i in range(0, 9):
        e = error(q_9_fdx(a), central_diff_quotient_9(a, h))
        h /= 2
        print(e)
        
def q_5_error(a):
    h = .5
    for i in range(0, 52):
        e = error(sin_fdx(a), central_diff_quotient_5(a, h))
        h /= 2
        print(e)

# print(sin_fdx(np.pi))
# q_5_error(1.0)
# q_6_error(1.0)
# q_6_error(2.0)
# q_6_error(3.0)
# q_9_error(0.0)
# q_9_error(1.0)
# q_12(1.570796)
print(central_diff_quotient_12(-100, .00000000001))
