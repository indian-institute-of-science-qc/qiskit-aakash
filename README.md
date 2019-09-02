# Qiskit dm_simulator User Guide
***
[This](https://arxiv.org/abs/1908.05154) is the link to arxiv document 'A Software Simulator for Noisy Quantum Circuits'
## Installation
It is advised to use virtual environment to install the files. Virtual environment can be created using anaconda. See [this](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to install conda into your system.  
Once you have conda in your system you can use it to create virtual environment.
```bash
conda create -n name_of_the_env python=3
```
You can activate/deactivate the virtual enviroment
```bash
conda activate name_of_the_env
conda deactivate
```
Once you have activated your virtual environment go to the folder where you kept the cloned files.  
To install the folder type in the terminal
```bash
pip install .
```
## Examples and Tutorials
Once installed, files can be changed and run in python. As an example,
```bash
python
```
```python
>>> from qiskit import QuantumCircuit,BasicAer,execute
>>> qc = QuantumCircuit(2)
>>> # Gates
>>> qc.x(1)
>>> qc.cx(0,1)
>>> # execution
>>> backend = BasicAer.get_backend('dm_simulator')
>>> run = execute(qc,backend)
>>> result = run.result()
>>> print(result['results'][0]['data']['densitymatrix'])
```
It would output the resultant densitymatrix as,
```python
[[0 0 0 0]
[0 1 0 0]
[0 0 0 0]
[0 0 0 0]]
```
The [jupyter notebook](dm_simulator_user_guide/user_guide.ipynb) provide detail examples about how to use this simulator.
