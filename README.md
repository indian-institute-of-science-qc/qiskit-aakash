<<<<<<< HEAD
# Qiskit dm_simulator User Guide
***
The details about the implementation of the density matrix simulator is given in the `arxiv` paper [1908.05154](https://arxiv.org/abs/1908.05154).
## Installation
> **Optional :** We advise you to use a virtual environment to install the files. Virtual environment can be created using `conda`.  
>
> ```bash
> conda create -y -n QiskitAakash python=3
> ```
> See [this](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) for instruction to install `conda` into your system. You may also give a different name to the environment. In that case, replace QiskitAakash with the name of your choice in the above and below instructions
>
> You can activate/deactivate the virtual environment.
> ```bash
> conda activate QiskitAakash
> conda deactivate
> ```
> Once you have activated the virtual environment follow the instructions below.

Installing from source requires that you have the Rust compiler on your system. To install the Rust compiler the recommended path is to use rustup, which is a cross-platform Rust installer. To use rustup you can go to:

https://rustup.rs/

which will provide instructions for how to install rust on your platform. Besides rustup there are other [installation methods](https://forge.rust-lang.org/infra/other-installation-methods.html) available too.

Once the Rust compiler is installed, you are ready to install Qiskit Aakash.
1. Clone the qiskit-aakash repo and enter it.
```bash
git clone https://github.com/indian-institute-of-science-qc/qiskit-aakash.git
cd qiskit-aakash
```
2. If you want to run tests or linting checks, install the developer requirements.
```bash
pip install -r requirements-dev.txt
```
3. Install qiskit-aakash.
```bash
pip install .
```
If you want to install in editable mode, use 
```bash
pip install -e .
```

> If you want to use it in [`Google Colab`](https://colab.research.google.com/) (easier and convenient but only works online)then the same commands will work
> ```
> !git clone https://github.com/indian-institute-of-science-qc/qiskit-aakash.git && python3 -m pip install qiskit-aakash/
> ```

The code for the new back-end `dm_simulator` can be found in [`dm_simulator.py`](qiskit/providers/basicaer/dm_simulator.py).
This back-end also uses some functionalities from [`basicaertools.py`](qiskit/providers/basicaer/basicaertools.py).

## Example
Once installed, files can be changed and run in python (for instructions to use the qiskit-terra part of the software, please visit [here](https://github.com/Qiskit/qiskit-terra)). For example,
```bash
python3
```
```python
from qiskit import QuantumCircuit,BasicAer,execute
qc = QuantumCircuit(2)
# Gates
qc.x(1)
qc.cx(0,1)
# execution
backend = BasicAer.get_backend('dm_simulator')
run = execute(qc,backend)
result = run.result()
print(result['results'][0]['data']['densitymatrix'])
```
It would output the resultant `densitymatrix` as,
```python
[[0 0 0 0]
[0 1 0 0]
[0 0 0 0]
[0 0 0 0]]
```
There are some `jupyter` notebooks in the repository which provide detailed examples about how to use this simulator.
Those can be viewed in [`Github`](dm_simulator_user_guide/user_guide.ipynb). But the easiest way to interact with them is by using [`Binder Image`](https://mybinder.org/v2/gh/indian-institute-of-science-qc/qiskit-aakash/master?filepath=.%2Fdm_simulator_user_guide%2Fuser_guide.ipynb).