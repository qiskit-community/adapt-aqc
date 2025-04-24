import logging

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from adaptaqc.compilers import AdaptCompiler, AdaptConfig
from adaptaqc.utils.entanglement_measures import EM_TOMOGRAPHY_CONCURRENCE

logging.basicConfig()
logger = logging.getLogger("adaptaqc")
logger.setLevel(logging.INFO)

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
print(f'{"-" * 10} ORIGINAL CIRCUIT {"-" * 10}')
print(qc)
print(f'{"-" * 10} RECOMPILED CIRCUIT {"-" * 10}')
print(compiled_circuit)

qc = random_circuit(5, 5, seed=1)

for i, (instr, _, _) in enumerate(qc.data):
    if instr.name == "id":
        qc.data.__delitem__(i)

# Compile
config = AdaptConfig(sufficient_cost=1e-2)
compiler = AdaptCompiler(
    qc, entanglement_measure=EM_TOMOGRAPHY_CONCURRENCE, adapt_config=config
)
result = compiler.compile()
print(result)
compiled_circuit = result.circuit

# See the original circuit
print(f'{"-" * 10} ORIGINAL CIRCUIT {"-" * 10}')
print(qc)

# See the compiled solution
print(f'{"-" * 10} RECOMPILED CIRCUIT {"-" * 10}')
print(compiled_circuit)

# Transpile the original circuits to the common basis set
qc_in_basis_gates = transpile(
    qc, basis_gates=["u1", "u2", "u3", "cx"], optimization_level=3
)
print(qc_in_basis_gates.count_ops())
print(qc_in_basis_gates.depth())

# Compare with compiled circuit
print(compiled_circuit.count_ops())
print(compiled_circuit.depth())
