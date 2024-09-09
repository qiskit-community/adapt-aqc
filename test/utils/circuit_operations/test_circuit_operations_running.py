from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from aqc_research.mps_operations import mps_dot, mps_from_circuit

import isl.backends.aer_mps_backend
import isl.backends.python_default_backends
import isl.utils.constants as vconstants
import isl.utils.circuit_operations as co


class TestCircuitOperationsRunning(TestCase):
    def test_given_mps_sims_with_different_trunc_when_mps_from_circuit_then_different_mps(
        self,
    ):
        sim1 = isl.backends.aer_mps_backend.mps_sim_with_args()
        sim2 = isl.backends.aer_mps_backend.mps_sim_with_args(mps_truncation_threshold=1e-1)

        qc = co.create_random_circuit(10, 20)

        mps1 = mps_from_circuit(qc.copy(), sim=sim1)
        mps2 = mps_from_circuit(qc.copy(), sim=sim2)

        self.assertLess(np.abs(mps_dot(mps1, mps2)) ** 2, 0.99)

    def test_given_mps_sims_with_different_chi_when_mps_from_circuit_then_different_mps(
        self,
    ):
        sim1 = isl.backends.aer_mps_backend.mps_sim_with_args()
        sim2 = isl.backends.aer_mps_backend.mps_sim_with_args(max_chi=2)

        qc = co.create_random_circuit(10, 20)

        mps1 = mps_from_circuit(qc.copy(), sim=sim1)
        mps2 = mps_from_circuit(qc.copy(), sim=sim2)

        self.assertLess(np.abs(mps_dot(mps1, mps2)) ** 2, 0.99)

    def test_run_circuit_without_transpilation(self):
        """
        run_circuit_without_transpilation has 3 possible return paths:
        1. if is_statevector_backend(backend)==True and return_statevector==True
        2. if is_statevector_backend(backend)==True and return_statevector==False
        3. if none of the above
        """
        qc = QuantumCircuit(5)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        # 1.
        data = co.run_circuit_without_transpilation(
            qc.copy(), backend=isl.backends.python_default_backends.SV_SIM, return_statevector=True
        )
        sv = data.data
        self.assertAlmostEqual(sv[0], 1 / np.sqrt(2))
        self.assertAlmostEqual(sv[-1], 1 / np.sqrt(2))

        # 2.
        counts = co.run_circuit_without_transpilation(
            qc.copy(), backend=isl.backends.python_default_backends.SV_SIM
        )
        self.assertAlmostEqual(counts["00000"] / sum(counts.values()), 0.5)
        self.assertAlmostEqual(counts["11111"] / sum(counts.values()), 0.5)

        # 3. (must add measurement gates, otherwise no counts)
        qc.measure_all()
        counts = co.run_circuit_without_transpilation(qc.copy())
        sigma = 1/np.sqrt(1024)
        self.assertAlmostEqual(counts["00000"] / sum(counts.values()), 0.5, delta=5*sigma)
        self.assertAlmostEqual(counts["11111"] / sum(counts.values()), 0.5, delta=5*sigma)

    def test_run_circuit_with_transpilation(self):
        """
        test the same return paths as run_circuit_without_transpilation:
        1. if is_statevector_backend(backend)==True and return_statevector==True
        2. if is_statevector_backend(backend)==True and return_statevector==False
        3. if none of the above


        NOTE this transpiles the input circuit (returned as a new circuit),
        and then calls run_circuit_without_transpilation. However transpile,
        and hence unroll_to_basis_gates, does not modify the circuit in place,
        so run_circuit_with_transpilation does not modify the input circuit.
        TODO: double check this is the intended functionality.
        """
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.s(0)
        qc.sdg(0)
        for i in range(4):
            qc.cx(i, i + 1)

        # 1.
        data = co.run_circuit_with_transpilation(
            qc.copy(), backend=isl.backends.python_default_backends.SV_SIM, return_statevector=True
        )
        sv = data.data
        self.assertAlmostEqual(abs(sv[0]) ** 2, 0.5)
        self.assertAlmostEqual(abs(sv[-1]) ** 2, 0.5)

        # 2.
        counts = co.run_circuit_with_transpilation(qc.copy(), backend=isl.backends.python_default_backends.SV_SIM)
        self.assertAlmostEqual(counts["00000"] / sum(counts.values()), 0.5)
        self.assertAlmostEqual(counts["11111"] / sum(counts.values()), 0.5)

        # 3. (must add measurement gates, otherwise no counts)
        qc.measure_all()
        counts = co.run_circuit_with_transpilation(qc.copy())
        self.assertAlmostEqual(counts["00000"] / sum(counts.values()), 0.5, 1)
        self.assertAlmostEqual(counts["11111"] / sum(counts.values()), 0.5, 1)
