import logging
import numpy as np
import isl.utils.circuit_operations as co
from isl.recompilers import ISLRecompiler, ISLConfig

logging.basicConfig()
logger = logging.getLogger('isl')
logger.setLevel(logging.INFO)

# Create circuit creating a random initial state
qc = co.create_random_initial_state_circuit(4)

config = ISLConfig(rotosolve_frequency=5, max_layers_to_modify=5)
isl_recompiler = ISLRecompiler(qc, isl_config=config, backend=co.MPS_SIM)

# Recompile, save recompiler object every 4 layers
result = isl_recompiler.recompile(save_frequency=4)

# Load recompiler after layer 4 and recompile from there
recompiler_4 = np.load("recompiler_after_layer_4.npy", allow_pickle=True)[0]
result_4 = recompiler_4.recompile()


# Comparison
print(result["time_taken"])
print(result_4["time_taken"])
print(co.calculate_overlap_between_circuits(result["circuit"], result_4["circuit"]))
