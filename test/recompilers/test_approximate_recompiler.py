from unittest import TestCase
from unittest.mock import patch

from qiskit import QuantumCircuit

from isl.recompilers.approximate_recompiler import ApproximateRecompiler
from isl.utils.circuit_operations import MPS_SIM, SV_SIM, QASM_SIM


@patch.multiple(ApproximateRecompiler, __abstractmethods__=set())
class TestApproximateRecompiler(TestCase):

    def setUp(self) -> None:
        self.qc = QuantumCircuit(1)

    def test_when_init_with_mps_backend_then_mps_backend_flag_true(self):
        self.assertTrue(ApproximateRecompiler(self.qc, MPS_SIM).is_mps_backend)

    def test_when_init_with_sv_backend_then_sv_backend_flag_true(self):
        self.assertTrue(ApproximateRecompiler(self.qc, SV_SIM).is_statevector_backend)

    def test_local_measurements_not_supported_for_qasm_and_mps(self):
        for backend in [QASM_SIM, MPS_SIM]:
            with self.assertRaises(NotImplementedError):
                ApproximateRecompiler(self.qc, local_measurements_only=True, backend=backend)

    def test_when_evaluate_cost_with_mps_backend_then_non_sampling_method_called(self):
        compiler = ApproximateRecompiler(QuantumCircuit(1), MPS_SIM)
        with patch.object(compiler, '_evaluate_cost_mps') as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()
