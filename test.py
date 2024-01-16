import numpy as np
import numpy.linalg as la
import math as m


n = 20
t = 20
num_it = 1

a = np.array([n+t] + [n,0,t]*num_it)
b = np.array([n+t] + [0,n,t]*num_it)
a = a / la.norm(a)
b = b / la.norm(b)

c = np.array([n+t,0] + [n,0,0,t]*num_it)
d = np.array([n-3+t,3] + [0,n-3,3,t]*num_it)
c = c / la.norm(c)
d = d / la.norm(d)

print(la.norm(a-b))
print(la.norm(c-d))

print(la.norm(a-b) > la.norm(c-d))

c = 20
print(m.sqrt(2*c**2) >  m.sqrt(3*3**2 + c**2 + (c-3)**2))