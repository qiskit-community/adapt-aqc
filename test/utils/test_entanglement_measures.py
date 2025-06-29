from unittest import TestCase
from unittest.mock import patch

import aqc_research.mps_operations as mpsops
import numpy as np
import qiskit.quantum_info
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, DensityMatrix
from qiskit_aer import Aer

import adaptaqc.backends.python_default_backends
import adaptaqc.utils.circuit_operations as co
import adaptaqc.utils.entanglement_measures as em
from adaptaqc.backends.python_default_backends import SV_SIM, MPS_SIM, QASM_SIM
from adaptaqc.compilers import AdaptCompiler
from adaptaqc.utils.entanglement_measures import (
    perform_quantum_tomography,
    EM_TOMOGRAPHY_CONCURRENCE,
    EM_TOMOGRAPHY_NEGATIVITY,
    EM_TOMOGRAPHY_EOF,
)


class TestEntanglementMeasures(TestCase):
    def test_quantum_tomography(self):
        qc = co.create_random_initial_state_circuit(3)
        dm = perform_quantum_tomography(qc, 0, 1, Aer.get_backend("qasm_simulator"))
        assert isinstance(dm, np.ndarray)

    def test_tomography_entanglement_measures(self):
        qc = co.create_random_initial_state_circuit(3)
        for backend in [
            QASM_SIM,
            SV_SIM,
        ]:
            for method in [
                em.EM_TOMOGRAPHY_CONCURRENCE,
                em.EM_TOMOGRAPHY_EOF,
                em.EM_TOMOGRAPHY_NEGATIVITY,
            ]:
                em.calculate_entanglement_measure(method, qc, 0, 1, backend)

    def test_observable_min_concurrence(self):
        qc = co.create_random_initial_state_circuit(3)
        em.measure_concurrence_lower_bound(qc, 0, 1, QASM_SIM)

    def test_when_calculating_concurrence_then_matches_qiskit(self):
        rho = DensityMatrix(random_statevector(4, seed=0)).data
        self.assertAlmostEqual(
            qiskit.quantum_info.concurrence(rho), em.concurrence(rho)
        )

    @patch("aqc_research.mps_operations.partial_trace")
    def test_given_mps_backend_when_calculate_em_measure_then_mps_operations_partial_trace_called(
        self, mock_mps_partial_trace
    ):
        mock_mps_partial_trace.return_value = np.ones((4, 4))

        backend = MPS_SIM
        qc = co.create_random_initial_state_circuit(3)
        qc_mps = mpsops.mps_from_circuit(qc, print_log_data=False)
        em.calculate_entanglement_measure(
            em.EM_TOMOGRAPHY_CONCURRENCE, qc, 0, 1, backend, mps=qc_mps
        )

        mock_mps_partial_trace.assert_called_once_with(
            qc_mps, [0, 1], already_preprocessed=True
        )

    @patch("adaptaqc.backends.aer_mps_backend.mps_from_circuit")
    def test_given_mps_backend_when_get_all_qubit_pair_em_measure_then_mps_from_circuit_called_exactly_once(
        self, mock_mps_from_circuit
    ):
        from numpy import array

        qc = QuantumCircuit(2)
        circ_mps = (
            [
                (array([[1.0 + 0.0j]]), array([[0.0 + 0.0j]])),
                (array([[1.0 + 0.0j]]), array([[0.0 + 0.0j]])),
            ],
            [array([1.0])],
        )
        circ_mps = mpsops._preprocess_mps(circ_mps)
        mock_mps_from_circuit.return_value = circ_mps
        compiler = AdaptCompiler(
            qc, backend=adaptaqc.backends.python_default_backends.MPS_SIM
        )
        compiler.circ_mps = circ_mps
        compiler._get_all_qubit_pair_entanglement_measures()
        mock_mps_from_circuit.assert_called_once()

    def test_given_random_state_when_backend_mps_or_statevector_then_ent_measures_equal(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)

        entanglement_measures = [
            EM_TOMOGRAPHY_CONCURRENCE,
            EM_TOMOGRAPHY_NEGATIVITY,
            EM_TOMOGRAPHY_EOF,
        ]

        for i in entanglement_measures:
            sv_compiler = AdaptCompiler(qc, entanglement_measure=i, backend=SV_SIM)
            mps_compiler = AdaptCompiler(qc, entanglement_measure=i, backend=MPS_SIM)

            np.testing.assert_allclose(
                sv_compiler._get_all_qubit_pair_entanglement_measures(),
                mps_compiler._get_all_qubit_pair_entanglement_measures(),
                atol=1e-06,
            )
