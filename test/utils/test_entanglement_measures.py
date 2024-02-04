import random
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from qiskit import Aer

import isl.utils.circuit_operations as co
import isl.utils.entanglement_measures as em
from isl.utils.circuit_operations import MPS_SIM
from isl.utils.entanglement_measures import perform_quantum_tomography


class TestEntanglementMeasures(TestCase):
    def test_quantum_tomography(self):

        qc = co.create_random_initial_state_circuit(3)
        dm = perform_quantum_tomography(qc, 0, 1, Aer.get_backend("qasm_simulator"))
        assert isinstance(dm, np.ndarray)

    def test_tomography_entanglement_measures(self):

        qc = co.create_random_initial_state_circuit(3)
        for backend in [
            Aer.get_backend("qasm_simulator"),
            Aer.get_backend("statevector_simulator"),
        ]:
            for method in [
                em.EM_TOMOGRAPHY_CONCURRENCE,
                em.EM_TOMOGRAPHY_EOF,
                em.EM_TOMOGRAPHY_NEGATIVITY,
            ]:
                em.calculate_entanglement_measure(method, qc, 0, 1, backend)

    def test_observable_min_concurrence(self):
        qc = co.create_random_initial_state_circuit(3)
        em.measure_concurrence_lower_bound(qc, 0, 1, Aer.get_backend("qasm_simulator"))

    @patch('isl.utils.entanglement_measures.mps_partial_trace')
    def test_given_mps_backend_when_calculate_em_measure_then_mps_partial_trace_called(self, mock_mps_partial_trace):

        mock_mps_partial_trace.return_value = np.ones((4, 4))

        backend = MPS_SIM
        qc = co.create_random_initial_state_circuit(3)
        em.calculate_entanglement_measure(em.EM_TOMOGRAPHY_CONCURRENCE, qc, 0, 1, backend)

        mock_mps_partial_trace.assert_called_once_with(qc, [0, 1])

    def test_given_random_state_when_partial_trace_with_mps_or_sampling_then_rdms_equal(self):

        qc = co.create_random_initial_state_circuit(3)
        qubits = random.sample(range(3), 2)
        mps_rho = em.mps_partial_trace(qc, qubits)
        sampling_rho = perform_quantum_tomography(qc, qubits[0], qubits[1], Aer.get_backend("qasm_simulator"),
                                                  execute_kwargs={"shots": int(1e6)})

        np.testing.assert_array_almost_equal(mps_rho, sampling_rho, decimal=2)

    def test_given_random_state_when_partial_trace_with_mps_or_sampling_then_ent_measures_equal(self):

        qc = co.create_random_initial_state_circuit(3)
        qubits = random.sample(range(3), 2)
        mps_rho = em.mps_partial_trace(qc, qubits)
        sampling_rho = perform_quantum_tomography(qc, qubits[0], qubits[1], Aer.get_backend("qasm_simulator"),
                                                  execute_kwargs={"shots": int(1e6)})

        self.assertAlmostEqual(em.concurrence(mps_rho), em.concurrence(sampling_rho), delta=1e-2)
        self.assertAlmostEqual(em.negativity(mps_rho), em.negativity(sampling_rho), delta=1e-2)
        self.assertAlmostEqual(em.eof(mps_rho), em.eof(sampling_rho), delta=1e-2)
