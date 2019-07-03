import numpy as np
import subprocess


subprocess.call("python ./dm_code.py".split())
subprocess.call("python ./qasm_code.py".split())

dm = np.loadtxt("a.txt", dtype=np.complex128)
qasm = np.loadtxt("a2.txt", dtype=np.complex128)


if np.allclose(dm, qasm):
    print("All good\n")
else:
    print("\n\n\n\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nThe results are not same.")
