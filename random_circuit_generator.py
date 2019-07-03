import numpy as np 
import random
from random import randint
noq = 4
l = ['u1','u2','u3','cx','ccx']
for i in range(random.randint(1,1000)):
    x = random.randint(0,4)
    if x == 0:
        print('qc.{}({},q[{}])'.format((l[x]),(random.randint(1,100)),(random.randint(0,noq-1))))
    elif x == 1:
        print('qc.{}({},{},q[{}])'.format((l[x]),(random.randint(1,100)),(random.randint(1,100)),(random.randint(0,noq-1))))
    elif x == 2:
         print('qc.{}({},{},{},q[{}])'.format(l[x],random.randint(1,100),random.randint(1,100),random.randint(1,100),random.randint(0,noq-1)))
    elif x == 3:
        print('qc.{}(q[{}],q[{}])'.format(l[x],random.randint(0,noq-1),random.randint(0,noq-1)))
        
    elif x == 4:
        print('qc.{}(q[{}],q[{}],q[{}])'.format(l[x],random.randint(0,noq-1),random.randint(0,noq-1),random.randint(0,noq-1)))



            


