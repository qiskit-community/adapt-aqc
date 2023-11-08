from qiskit.circuit.random import random_circuit

from isl.recompilers import ISLRecompiler, ISLConfig
from qiskit import QuantumCircuit, transpile

from isl.utils.entanglement_measures import EM_TOMOGRAPHY_CONCURRENCE

# Setup the circuit
qc = QuantumCircuit(3)
qc.rx(1.23, 0)
qc.cx(0, 1)
qc.ry(2.5, 1)
qc.rx(-1.6, 2)
qc.ccx(2, 1, 0)

# Recompile
recompiler = ISLRecompiler(qc)
result = recompiler.recompile()
recompiled_circuit = result['circuit']

# See the recompiled output
print(f'{"-" * 10} ORIGINAL CIRCUIT {"-" * 10}')
print(qc)
print(f'{"-" * 10} RECOMPILED CIRCUIT {"-" * 10}')
print(recompiled_circuit)

qc = random_circuit(5, 5, seed=1)

for i, (instr, _, _) in enumerate(qc.data):
    if instr.name == "id":
        qc.data.__delitem__(i)

# Recompile
config = ISLConfig(sufficient_cost=1e-2)
recompiler = ISLRecompiler(qc, entanglement_measure=EM_TOMOGRAPHY_CONCURRENCE, isl_config=config)
result = recompiler.recompile()
print(result)
recompiled_circuit = result['circuit']

# See the original circuit
print(f'{"-" * 10} ORIGINAL CIRCUIT {"-" * 10}')
print(qc)

# See the recompiled solution
print(f'{"-" * 10} RECOMPILED CIRCUIT {"-" * 10}')
print(recompiled_circuit)

# Transpile the original circuits to the common basis set
qc_in_basis_gates = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
print(qc_in_basis_gates.count_ops())
print(qc_in_basis_gates.depth())

# Compare with recompiled circuit
print(recompiled_circuit.count_ops())
print(recompiled_circuit.depth())
