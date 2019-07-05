
import numpy as np

a = np.loadtxt("a.txt",dtype=complex)
b = np.loadtxt("a1.txt",dtype=complex)
p = a.real
q = a.imag
c = b.real
d = b.imag
if(np.allclose(p,c) and np.allclose(q,d)):   # Best way to compare two imaginary numbers with some tolerance value
     print('Your result is right.')
else:
     print('Your result is wrong')     



