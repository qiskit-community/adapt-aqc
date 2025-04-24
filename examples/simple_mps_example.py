"""
Example script for running ADAPT-AQC recompilation using a Matrix Product State (MPS) backend.
MPS is an alternative quantum state representation to state-vector, and is better suited to handle
large, low-entanglement states.
"""

import logging

from qiskit import QuantumCircuit

from adaptaqc.backends.aer_mps_backend import AerMPSBackend
from adaptaqc.compilers import AdaptCompiler

logging.basicConfig()
logger = logging.getLogger("adaptaqc")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------------
# Very simple MPS example
# Create a large circuit where only some qubits are entangled
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
