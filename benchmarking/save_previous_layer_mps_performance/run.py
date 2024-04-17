import logging
import time
import numpy as np

import isl.utils.circuit_operations as co
from isl.recompilers import ISLRecompiler, ISLConfig

num_qubits = [3,4,5,6,7]
times_saving = []
times_without_saving = []

for n in num_qubits:
    # Create circuit creating a random initial state
    qc = co.create_random_initial_state_circuit(n)

    print(f"recompiling_{n}_qubits_saving_mps")
    # Save previous layer MPS
    config_1 = ISLConfig(cost_improvement_num_layers=1e3, rotosolve_frequency=np.inf)
    recompiler_1 = ISLRecompiler(qc, backend=co.MPS_SIM, isl_config=config_1)
    result_1 = recompiler_1.recompile()
    times_saving.append(result_1['time_taken'])

    print(f"recompiling_{n}_qubits_without_saving_mps")
    # Don't save previous layer MPS
    config_2 = ISLConfig(cost_improvement_num_layers=1e3, rotosolve_frequency=1e5)
    recompiler_2 = ISLRecompiler(qc, backend=co.MPS_SIM, isl_config=config_2)
    result_2 = recompiler_2.recompile()
    times_without_saving.append(result_2['time_taken'])

np.save(f"./results/results_{min(num_qubits)}_to_{max(num_qubits)}_qubits_{str(time.time_ns())[:-9]}", np.array([num_qubits, times_saving, times_without_saving]))