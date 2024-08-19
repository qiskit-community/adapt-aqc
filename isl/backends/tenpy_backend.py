import logging

import numpy as np
from aqc_research.mps_operations import mps_dot, _preprocess_mps, mps_expectation

from isl.backends.aqc_backend import AQCBackend
from isl.utils.circuit_operations import extract_inner_circuit
from isl.utils.utilityfunctions import tenpy_to_qiskit_mps
class TenpyBackend(AQCBackend):
    def __init__(self, simulator=None, cut_off=1e-13):
        from qiskit_tenpy_converter.simulation.simulator import Simulator
        self.simulator = Simulator() if simulator is None else simulator
        self.tenpy_cut_off = cut_off

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
        ansatz_circ = extract_inner_circuit(
            compiler.full_circuit, compiler.ansatz_range()
        )
        circ_mps = tenpy_to_qiskit_mps(
            self.simulator.simulate(
                ansatz_circ,
                cut_off=self.tenpy_cut_off,
                starting_state=compiler.tenpy_target_mps.copy(),
            )
        )

        return _preprocess_mps(circ_mps)

    def measure_qubit_expectation_values(self, compiler):
        mps = self.evaluate_circuit(compiler)
        return [
            (mps_expectation(mps, "Z", i, already_preprocessed=True))
            for i in range(len(mps))
        ]
