"""
Example script for running ADAPT-AQC recompilation using more advanced options
"""

import logging

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from adaptaqc.compilers import AdaptCompiler, AdaptConfig

logging.basicConfig()
logger = logging.getLogger("adaptaqc")
logger.setLevel(logging.INFO)

n = 4
state_prep_circuit = QuantumCircuit(n)
state_prep_circuit.h(range(n))

# Create a random circuit starting with a layer of hadamard gates
qc = state_prep_circuit.compose(random_circuit(n, 16, 2, seed=0))

config = AdaptConfig(
    # We expect the solution to take longer to converge, so decrease the threshold for exiting
    # early.
    cost_improvement_tol=1e-5,
    # Run Rotosolve only every 10th layer to reduce computational cost.
    rotosolve_frequency=10,
    # Choose Rotosolve to modify only the last 10 layers.
    max_layers_to_modify=10,
    # Setting this value > 0 prioritises not using the same qubit pairs too often.
    reuse_exponent=1,
    # Increase the amount the cost needs to decrease by to terminate Rotosolve. This stops spending
    # too much time fine-tuning the angles.
    rotosolve_tol=1e-2,
)

# Since we know the solution starts with Hadamards, we can pass this information into ADAPT-AQC
starting_circuit = state_prep_circuit

adapt_compiler = AdaptCompiler(
    target=qc,
    adapt_config=config,
    starting_circuit=starting_circuit,
    initial_single_qubit_layer=True,
)

result = adapt_compiler.compile()
approx_circuit = result.circuit
print(f"Overlap between circuits is {result.overlap}")

# Transpile the original circuits to the common basis set with maximum Qiskit optimization
qc_in_basis_gates = transpile(
    qc, basis_gates=["ry", "rz", "rx", "u3", "cx"], optimization_level=3
)
print("Original circuit gates:", qc_in_basis_gates.count_ops())
print("Original circuit depth:", qc_in_basis_gates.depth())

# Compare with compiled circuit
print("Compiled circuit gates:", approx_circuit.count_ops())
print("Compiled circuit depth:", approx_circuit.depth())
