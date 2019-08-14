# Qiskit dm_simulator User Guide
***
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
In this version the code for the new backend dm_simulator can be found in [dm_simulator.py](qiskit/providers/basicaer/dm_simulator.py). This backend also uses some functionalities from [basicaertools.py](qiskit/providers/basicaer/basicaertools.py).

## Example
Once installed, files can be changed and run in python. As an example,
```bash
python3
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
 There are some jupyter notebook added which provide detail examples about how to use this simulator.
 Those can be viewed in [Github](dm_simulator_user_guide/user_guide.ipynb). But the easiest way to interact with them is by using [Binder Image](https://mybinder.org/v2/gh/indian-institute-of-science-qc/qiskit-aakash/master?filepath=.%2Fdm_simulator_user_guide%2Fuser_guide.ipynb)
