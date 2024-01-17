import os
print(os.environ.get("PYTHONPATH"))
import logging

import isl.utils.circuit_operations as co
from isl.recompilers import ISLRecompiler

logging.basicConfig()
logger = logging.getLogger('isl')
logger.setLevel(logging.DEBUG)

# Create circuit creating a random initial state
qc = co.create_random_initial_state_circuit(4)

isl_recompiler = ISLRecompiler(qc)

result = isl_recompiler.recompile()
approx_circuit = result["circuit"]
print(f"Overlap between circuits is {result['overlap']}")
print(f'{"-" * 32}')
print(f'{"-" * 10}OLD  CIRCUIT{"-" * 10}')
print(f'{"-" * 32}')
print(qc)
print(f'{"-" * 32}')
print(f'{"-" * 10}ISL  CIRCUIT{"-" * 10}')
print(f'{"-" * 32}')
print(approx_circuit)
