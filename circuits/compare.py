
import numpy as np

a = np.loadtxt("a.txt",dtype=complex)
b = np.loadtxt("a1.txt",dtype=complex)
p = a.real
q = a.imag
c = b.real
d = b.imag
if(np.allclose(p,c) and np.allclose(q,d)):
     print('Your result is right.')
else:
     print('Your result is wrong')     



