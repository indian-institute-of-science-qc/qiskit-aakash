import numpy as np 
import random
from random import randint
noq = 4
l = ['u1','u2','u3','cx','ccx']
for i in range(random.randint(20,1000)):
    x = random.randint(0,4)
    if x == 0:
        print('qc.{}({},q[{}])'.format((l[x]), round(random.uniform(0, 2*np.pi),5), 
        (random.randint(0, noq-1))))
    elif x == 1:
        print('qc.{}({},{},q[{}])'.format(l[x], round(random.uniform(0, 2*np.pi),5), 
        round(random.uniform(0, 2*np.pi),5), random.randint(0, noq-1)))
    elif x == 2:
        print('qc.{}({},{},{},q[{}])'.format(l[x], round(random.uniform(0, np.pi),5), 
        round(random.uniform(0, 2*np.pi),5), round(random.uniform(0, 2*np.pi),5), random.randint(0, noq-1)))
    elif x == 3:
        p, q = random.sample(range(noq),2)
        print('qc.{}(q[{}],q[{}])'.format(l[x], p, q))
        
    elif x == 4:
        a, b, c = random.sample(range(noq),3)
        print('qc.{}(q[{}],q[{}],q[{}])'.format(l[x], a, b, c))



            


