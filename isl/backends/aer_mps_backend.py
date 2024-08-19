import logging

import numpy as np
from aqc_research.mps_operations import mps_from_circuit, mps_dot, mps_expectation
from qiskit_aer import AerSimulator

from isl.backends.aqc_backend import AQCBackend

logger = logging.getLogger(__name__)


def mps_sim_with_args(mps_truncation_threshold=1e-16, max_chi=None, mps_log_data=False):
    """
    :param mps_truncation_threshold: truncation threshold to use in AerSimulator
    :param max_chi: maximum bond dimension to use in AerSimulator
    :param mps_log_data: same as corresponding argument in AerSimulator. Setting to true will
    massively reduce performance and should only be done for debugging

    :return: instance of AerSimulator using MPS method and parameters specified above
    """
    logger.info(f"Using Aer MPS Simulator with truncation {mps_truncation_threshold}")
    return AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=mps_truncation_threshold,
        matrix_product_state_max_bond_dimension=max_chi,
        mps_log_data=mps_log_data,
    )


class AerMPSBackend(AQCBackend):
    def __init__(self, simulator=mps_sim_with_args()):
        self.simulator = simulator

    def evaluate_global_cost(self, compiler):
        circ_mps = self.evaluate_circuit(compiler)
        return (
            1
            - np.absolute(
                mps_dot(circ_mps, compiler.zero_mps, already_preprocessed=True)
            )
            ** 2
        )

    def evaluate_local_cost(self, compiler):
        evals = self.measure_qubit_expectation_values(compiler)
        return 0.5 * (1 - np.mean(evals))

    def evaluate_circuit(self, compiler):
        circ = compiler.full_circuit.copy()
        return mps_from_circuit(circ, return_preprocessed=True, sim=self.simulator)

    def measure_qubit_expectation_values(self, compiler):
        mps = self.evaluate_circuit(compiler)
        expectation_values = [
            (mps_expectation(mps, "Z", i, already_preprocessed=True))
            for i in range(compiler.full_circuit.num_qubits)
        ]
        return expectation_values
