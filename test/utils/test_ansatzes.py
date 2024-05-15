from unittest import TestCase

from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co
from isl import ISLRecompiler
from isl.recompilers import ISLConfig
from isl.utils.ansatzes import u4, identity_resolvable, fully_dressed_cnot, thinly_dressed_cnot
from isl.utils.circuit_operations import MPS_SIM, SV_SIM
from isl.utils.constants import DEFAULT_SUFFICIENT_COST


class TestAnsatzes(TestCase):

    def setUp(self):
        self.ansatz_list = [u4, thinly_dressed_cnot, fully_dressed_cnot, identity_resolvable]

    def test_given_custom_ansatz_when_compile(self):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=1)
                qc = co.unroll_to_basis_gates(qc)
                isl_recompiler = ISLRecompiler(qc,
                                               custom_layer_2q_gate=ansatz())

                result = isl_recompiler.recompile()

                approx_circuit = result["circuit"]

                overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
                assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_given_custom_ansatz_and_all_options_when_compile(self):

        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=1)
                qc = co.unroll_to_basis_gates(qc)

                start_qc = QuantumCircuit(3)
                start_qc.h(range(3))

                isl_recompiler = ISLRecompiler(qc,
                                               custom_layer_2q_gate=ansatz(),
                                               starting_circuit=start_qc,
                                               initial_single_qubit_layer=True,
                                               backend=SV_SIM)

                result = isl_recompiler.recompile()

                approx_circuit = result["circuit"]

                overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
                assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_given_custom_ansatz_when_add_layer_then_parameters_change(self):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=1)
                qc = co.unroll_to_basis_gates(qc)
                isl_recompiler = ISLRecompiler(qc, custom_layer_2q_gate=ansatz())

                isl_recompiler._add_layer(0)

                last_layer_instructions = isl_recompiler.full_circuit[-len(ansatz()):]
                for i, instruction in enumerate(last_layer_instructions):
                    if instruction.operation.name != "cx":
                        if ansatz is u4 and i == 17:
                            pass  # For this seed, final gate of u4 has optimal angle 0
                        else:
                            self.assertNotEqual(instruction.operation.params[0], 0.)

    def test_given_custom_ansatz_and_mps_backend_when_add_layer_then_layers_cached(self):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3)
                qc = co.unroll_to_basis_gates(qc)
                isl_recompiler = ISLRecompiler(qc,
                                               custom_layer_2q_gate=ansatz(),
                                               backend=MPS_SIM,
                                               isl_config=ISLConfig(max_layers_to_modify=2))

                isl_recompiler._add_layer(0)
                self.assertEqual(len(isl_recompiler.full_circuit), 1 + len(ansatz()))
                isl_recompiler._add_layer(1)
                self.assertEqual(len(isl_recompiler.full_circuit), 1 + len(ansatz()))

    def test_given_custom_ansatz_when_add_layer_then_gate_types_as_expected(self):
        # Expected cnot indices for ansatzes: [u4, thinly_dressed_cnot,
        # fully_dressed_cnot, identity_resolvable]
        # E.g. for a thinly-dressed cnot, gates 0, 1, 3, 4 are rotation gates,
        # and gate 2 is a cnot, so expected_cnots[1] = [2]
        expected_cnots = [[6, 9, 11], [2], [6], [2, 5]]
        for ansatz, expected_cnots in zip(self.ansatz_list, expected_cnots):
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=2)
                qc = co.unroll_to_basis_gates(qc)
                isl_recompiler = ISLRecompiler(qc,
                                               custom_layer_2q_gate=ansatz())

                isl_recompiler._add_layer(0)
                isl_recompiler._add_layer(1)

                last_layer_instructions = isl_recompiler.full_circuit[-len(ansatz()):]

                for index in expected_cnots:
                    self.assertEqual(last_layer_instructions[index].operation.name, "cx")

    def test_given_custom_thinly_dressed_when_compile_same_as_default_behaviour(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        default_isl_recompiler = ISLRecompiler(qc)
        default_result = default_isl_recompiler.recompile()

        custom_isl_recompiler = ISLRecompiler(qc,
                                              custom_layer_2q_gate=thinly_dressed_cnot())
        custom_result = custom_isl_recompiler.recompile()

        self.assertEqual(default_result.get("overlap"), custom_result.get("overlap"))
