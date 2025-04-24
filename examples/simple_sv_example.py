import logging

import adaptaqc.utils.circuit_operations as co
from adaptaqc.compilers import AdaptCompiler

logging.basicConfig()
logger = logging.getLogger("adaptaqc")
logger.setLevel(logging.INFO)

# Create circuit creating a random initial state
qc = co.create_random_initial_state_circuit(4)

adapt_compiler = AdaptCompiler(qc)

result = adapt_compiler.compile()
approx_circuit = result.circuit
print(f"Overlap between circuits is {result.overlap}")
print(f'{"-" * 32}')
print(f'{"-" * 10}OLD  CIRCUIT{"-" * 10}')
print(f'{"-" * 32}')
print(qc)
print(f'{"-" * 32}')
print(f'{"-" * 10}ADAPT-AQC  CIRCUIT{"-" * 10}')
print(f'{"-" * 32}')
print(approx_circuit)
