import numpy as np
from aqc_research.mps_operations import mps_dot, mps_expectation

from isl.backends.aqc_backend import AQCBackend
from isl.utils.circuit_operations import extract_inner_circuit
from isl.utils.cuquantum_functions import (
    mps_from_circuit_and_starting_mps,
    cu_mps_to_aer_mps, DEFAULT_CU_ALGORITHM,
)


class CuQuantumBackend(AQCBackend):
    def __init__(self, cu_algorithm=None):
        if cu_algorithm is None:
            cu_algorithm = DEFAULT_CU_ALGORITHM
        self.cu_algorithm = cu_algorithm

    def evaluate_global_cost(self, compiler):
        circ_mps = self.evaluate_circuit(compiler)
        cost = (
            1
            - np.absolute(
                mps_dot(circ_mps, compiler.zero_mps, already_preprocessed=True)
            )
            ** 2
        )
        return cost

    def evaluate_local_cost(self, compiler):
        e_vals = self.measure_qubit_expectation_values(compiler)
        return 0.5 * (1 - np.mean(e_vals))

    def evaluate_circuit(self, compiler):
        if not compiler.compiling_finished:
            # Contract all gates after the most recent cache
            first_gate_index_not_cached = compiler.next_gate_to_cache_index
            gates_to_contract = extract_inner_circuit(
                compiler.full_circuit,
                (first_gate_index_not_cached, len(compiler.full_circuit)),
            )
            circ_mps = mps_from_circuit_and_starting_mps(
                gates_to_contract, compiler.cu_cached_mps, self.cu_algorithm
            )
        else:
            ansatz_circ = extract_inner_circuit(
                compiler.full_circuit, compiler.ansatz_range()
            )
            circ_mps = mps_from_circuit_and_starting_mps(
                ansatz_circ, compiler.cu_target_mps, self.cu_algorithm
            )
        return cu_mps_to_aer_mps(circ_mps)

    def measure_qubit_expectation_values(self, compiler):
        mps = self.evaluate_circuit(compiler)
        return [
            (mps_expectation(mps, "Z", i, already_preprocessed=True))
            for i in range(len(mps))
        ]
