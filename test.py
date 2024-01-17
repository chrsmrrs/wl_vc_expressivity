import numpy as np
import numpy.linalg as la
import math as m


c = 50000000000000000
t = 10

a = np.array([2*c] + [0, c, c, 0]*t)
b = np.array([2*c] + [0, c, 0, c]*t)

a = a/la.norm(a)
b = b/la.norm(b)

print(la.norm(a-b))

a = np.array([2*c,0] + [c, c, 0, 0]*t)
b = np.array([2*c-3,3] + [c, 0, c-3, 3]*t)

a = a/la.norm(a)
b = b/la.norm(b)

print(la.norm(a-b))