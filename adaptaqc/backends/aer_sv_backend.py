import os

import numpy as np
from qiskit_aer import Aer

from adaptaqc.backends.aqc_backend import AQCBackend


class AerSVBackend(AQCBackend):
    def __init__(self, simulator=Aer.get_backend("statevector_simulator")):
        self.simulator = simulator

    def evaluate_global_cost(self, compiler):
        if compiler.soften_global_cost:
            raise NotImplementedError(
                "soften_global_cost is currently only implemented for AerMPSBackend"
            )
        sv = self.evaluate_circuit(compiler)
        cost = 1 - (np.absolute(sv[0])) ** 2
        return cost

    def evaluate_local_cost(self, compiler):
        e_vals = self.measure_qubit_expectation_values(compiler)
        cost = 0.5 * (1 - np.mean(e_vals))
        return cost

    def evaluate_circuit(self, compiler):
        # Don't parallelise shots if ADAPT-AQC is already being run in parallel
        already_in_parallel = os.environ["QISKIT_IN_PARALLEL"] == "TRUE"
        backend_options = {} if already_in_parallel else compiler.backend_options

        job = self.simulator.run(
            compiler.full_circuit, **backend_options, **compiler.execute_kwargs
        )

        result = job.result()
        return result.get_statevector()

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
