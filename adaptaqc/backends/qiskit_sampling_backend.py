# (C) Copyright IBM 2025. 
# 
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

import numpy as np
from qiskit_aer import Aer
from qiskit_aer.backends.aerbackend import AerBackend

from adaptaqc.backends.aqc_backend import AQCBackend


class QiskitSamplingBackend(AQCBackend):
    def __init__(self, simulator=Aer.get_backend("qasm_simulator")):
        self.simulator = simulator

    def evaluate_global_cost(self, compiler):
        if compiler.soften_global_cost:
            raise NotImplementedError(
                "soften_global_cost is currently only implemented for AerMPSBackend"
            )
        counts = self.evaluate_circuit(compiler)
        total_qubits = (
            2 * compiler.total_num_qubits
            if compiler.general_initial_state
            else compiler.total_num_qubits
        )
        all_zero_string = "".join(str(int(e)) for e in np.zeros(total_qubits))
        total_shots = sum([each_count for _, each_count in counts.items()])
        # '00...00' might not be present in counts if no shot resulted in
        # the ground state
        if all_zero_string in counts:
            overlap = counts[all_zero_string] / total_shots
        else:
            overlap = 0
        cost = 1 - overlap
        return cost

    def evaluate_local_cost(self, compiler):
        qubit_costs = np.zeros(compiler.total_num_qubits)
        for i in range(compiler.total_num_qubits):
            if compiler.general_initial_state:
                compiler.full_circuit.measure(i, 0)
                compiler.full_circuit.measure(i + compiler.total_num_qubits, 1)
                counts = self.evaluate_circuit(compiler)
                del compiler.full_circuit.data[-1]
                del compiler.full_circuit.data[-1]
                total_shots = sum([each_count for _, each_count in counts.items()])
                # '00...00' might not be present in counts if no shot
                # resulted in the ground state
                if "00" in counts:
                    overlap = counts["00"] / total_shots
                else:
                    overlap = 0
                qubit_costs[i] = 1 - overlap
            else:
                compiler.full_circuit.measure(i, 0)
                counts = self.evaluate_circuit(compiler)
                del compiler.full_circuit.data[-1]
                total_shots = sum([each_count for _, each_count in counts.items()])
                # '00...00' might not be present in counts if no shot
                # resulted in the ground state
                if "0" in counts:
                    overlap = counts["0"] / total_shots
                else:
                    overlap = 0
                qubit_costs[i] = 1 - overlap
        cost = np.mean(qubit_costs)
        return cost

    def evaluate_circuit(self, compiler):
        # Don't parallelise shots if ADAPT-AQC is already being run in parallel
        already_in_parallel = os.environ["QISKIT_IN_PARALLEL"] == "TRUE"
        backend_options = None if already_in_parallel else compiler.backend_options

        if backend_options is None or not isinstance(self.simulator, AerBackend):
            backend_options = {}
        job = self.simulator.run(
            compiler.full_circuit, **backend_options, **compiler.execute_kwargs
        )
        result = job.result()
        return result.get_counts()

    def measure_qubit_expectation_values(self, compiler):
        counts = self.evaluate_circuit(compiler)
        n_qubits = len(list(counts)[0])

        expectation_values = []
        for i in range(n_qubits):
            if i >= n_qubits:
                raise ValueError("qubit_index outside of register range")
            reverse_index = n_qubits - (i + 1)
            exp_val = 0
            total_counts = 0
            for bitstring in list(counts):
                exp_val += (1 if bitstring[reverse_index] == "0" else -1) * counts[
                    bitstring
                ]
                total_counts += counts[bitstring]
            expectation_values.append(exp_val / total_counts)
        return expectation_values
