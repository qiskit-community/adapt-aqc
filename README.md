
# Adaptive approximate quantum compiling (ADAPT-AQC)

An open-source implementation of ADAPT-AQC [1], an approximate quantum compiling (AQC) and Matrix Product state (MPS) preparation algorithm.
As supposed to assuming any particular ansatz structure, ADAPT-AQC adaptively builds
an ansatz, adding a new two-qubit unitary every iteration.

ADAPT-AQC is the successor to ISL [2], using much of the same core code, routine and optimiser. The most significant difference
however is its use of MPS simulators. This allows it to compile circuits at 50+ qubits, as well as directly prepare MPSs.

[1] https://arxiv.org/abs/2503.09683 \
[2] https://github.com/abhishekagarwal2301/isl

## Installation

This repository can be easily installed using `pip`. You have two options:

Use a stable version based on the last commit to `master`
```
pip install git+ssh://git@github.com/bjader/adapt-aqc.git
```

Use an editable local version (after already cloning this repository)
```
pip install -e PATH_TO_LOCAL_CLONE
```

## Contributing

To make changes to ADAPT-AQC, first clone the repository.
Then navigate to your local copy, create a Python environment and install the required dependencies

```
pip install .
```

You can check your development environment is ready by successfully running the scripts in `/examples/`.

## Minimal examples

### Compiling with statevector simulator
A circuit can be compiled and the result accessed with only 3 lines if using the 
default settings.

```python
from adaptaqc.compilers import AdaptCompiler
from qiskit import QuantumCircuit

# Setup the circuit
qc = QuantumCircuit(3)
qc.rx(1.23, 0)
qc.cx(0, 1)
qc.ry(2.5, 1)
qc.rx(-1.6, 2)
qc.ccx(2, 1, 0)

# Compile
compiler = AdaptCompiler(qc)
result = compiler.compile()
compiled_circuit = result.circuit

# See the compiled output
print(compiled_circuit)
```

### Compiling matrix product states

Circuits beyond the size accessible to statevector simulators can be compiled via their 
representation as matrix product states. To give a very simple example where most the qubits are 
left in the $|0\rangle$ state.

```python
from qiskit import QuantumCircuit

from adaptaqc.backends.aer_mps_backend import AerMPSBackend
from adaptaqc.compilers import AdaptCompiler

n = 50
qc = QuantumCircuit(n)
qc.h(0)
qc.cx(0, 1)
qc.h(2)
qc.cx(2, 3)
qc.h(range(4, n))

# Create compiler with the default MPS simulator, which has very minimal truncation.
adapt_compiler = AdaptCompiler(qc, backend=AerMPSBackend())

result = adapt_compiler.compile()
print(f"Overlap between circuits is {result.overlap}")
```

### Specifying additional configuration

For more advanced examples, please see `examples/advanced_mps_example.py` and 
`advanced_sv_example.py`.
For a full overview of the different configuration options, on top of the documentation, see
`docs/running_options_explained.md`.

## Citing usage

We respectfully ask any publication, project or whitepaper using ADAPT-AQC to cite the following 
work:

[Jaderberg, B., Pennington, G., Marshall, K.V., Anderson, L.W., Agarwal, A., Lindoy, L.P., Rungger, I., Mensa, S. and Crain, J., 2025. Variational preparation of normal matrix product states on quantum computers. arXiv preprint arXiv:2503.09683](https://arxiv.org/abs/2503.09683)