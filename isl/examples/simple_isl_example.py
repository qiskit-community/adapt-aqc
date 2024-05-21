import logging
import numpy as np
import isl.utils.circuit_operations as co
from isl.recompilers import ISLRecompiler

logging.basicConfig()
logger = logging.getLogger('isl')
logger.setLevel(logging.INFO)

# Create circuit creating a random initial state
qc = co.create_random_initial_state_circuit(4)

isl_recompiler = ISLRecompiler(qc)

# Using checkpoint
intermediate_recompiler, checkpoint = isl_recompiler.recompile(save_point=4)
print("checkpoint")
np.save("intermediate_recompiler.npy", np.array([intermediate_recompiler, checkpoint], dtype=object))

loaded_recompiler, start_point = np.load("intermediate_recompiler.npy", allow_pickle=True)
result = loaded_recompiler.recompile(start_point=start_point)

# Not using checkpoint
original_recompiler = ISLRecompiler(qc)
original_result = original_recompiler.recompile()

# Comparison
print(result["global_cost_history"])
print(original_result["global_cost_history"])
print(co.calculate_overlap_between_circuits(result["circuit"], original_result["circuit"]))
