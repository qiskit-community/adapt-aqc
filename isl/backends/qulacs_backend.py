import logging
import os

import numpy as np

from isl.backends.aqc_backend import AQCBackend
from isl.utils.circuit_operations.circuit_operations_qulacs import run_on_qulacs_noiseless

logger = logging.getLogger(__name__)


class QulacsBackend(AQCBackend):
    def __init__(self):
        self.simulator = None

    def evaluate_global_cost(self, compiler):
        sv = self.evaluate_circuit(compiler)
        cost = 1 - (np.absolute(sv[0])) ** 2
        return cost

    def evaluate_local_cost(self, compiler):
        e_vals = self.measure_qubit_expectation_values(compiler)
        cost = 0.5 * (1 - np.mean(e_vals))
        return cost

    def evaluate_circuit(self, compiler):
        circuit = compiler.full_circuit
        sv = run_on_qulacs_noiseless(circuit, False)
        if (
            "noise_model" in compiler.execute_kwargs
            and compiler.execute_kwargs["noise_model"] is not None
        ):
            raise ValueError(f"Noisy emulations on qulacs are not supported yet")

        return sv

    def measure_qubit_expectation_values(self, compiler):
        sv = self.evaluate_circuit(compiler)
        expectation_values = []
        n_qubits = sv.num_qubits
        for i in range(n_qubits):
            if i >= n_qubits:
                raise ValueError("qubit_index outside of register range")
            [p0, p1] = sv.probabilities([i])
            exp_val = p0 - p1
            expectation_values.append(exp_val)
        return expectation_values