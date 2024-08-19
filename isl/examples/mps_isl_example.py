"""
Example script for running ISL recompilation using a Matrix Product State (MPS) backend.
MPS is an alternative quantum state representation to state-vector, and is better suited to handle
large, low-entanglement states.
"""

import logging

from aqc_research.mps_operations import rand_mps_vec
from qiskit import QuantumCircuit

from isl.backends.aer_mps_backend import mps_sim_with_args, AerMPSBackend
from isl.recompilers import ISLRecompiler
from isl.backends.python_default_backends import MPS_SIM

logging.basicConfig()
logger = logging.getLogger('isl')
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------------
# Basic MPS example
# Create a large circuit where only some qubits are entangled
n = 50
qc = QuantumCircuit(n)
qc.h(0)
qc.cx(0, 1)
qc.h(2)
qc.cx(2, 3)
qc.h(range(4, n))

# Create compiler with the default MPS simulator, which has very minimal truncation.
isl_recompiler = ISLRecompiler(qc, backend=MPS_SIM, initial_single_qubit_layer=True)

result = isl_recompiler.recompile()
print(f"Overlap between circuits is {result.overlap}")

# --------------------------------------------------------------------------------
# Extra MPS features
# When using the MPS simulator, it is also possible to
# (a) pass in an MPS as a target.
# (b) Set the Aer truncation threshold manually.

# Create an instance of Qiskit's MPS simulator with a specified truncation threshold
qiskit_mps_sim = mps_sim_with_args(mps_truncation_threshold=1e-8)

# Create the corresponding AQC backend
backend = AerMPSBackend(simulator=qiskit_mps_sim)

# Create a target MPS
mps = rand_mps_vec(4)

# Set the compilation target to be a MPS rather than a circuit
isl_recompiler = ISLRecompiler(target=mps, backend=backend)

result = isl_recompiler.recompile()
print(f"Overlap between circuits is {result.overlap}")
