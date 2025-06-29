from unittest import TestCase

from aqc_research.model_sp_lhs.trotter.trotter import trotter_circuit
from qiskit import QuantumCircuit

import adaptaqc.utils.circuit_operations as co
from adaptaqc.backends.python_default_backends import SV_SIM, MPS_SIM
from adaptaqc.compilers import AdaptConfig
from adaptaqc.utils.ansatzes import (
    u4,
    identity_resolvable,
    fully_dressed_cnot,
    thinly_dressed_cnot,
    heisenberg,
)
from adaptaqc.utils.constants import DEFAULT_SUFFICIENT_COST
from adaptaqc.compilers.adapt.adapt_compiler import AdaptCompiler


class TestAnsatzes(TestCase):
    def setUp(self):
        self.ansatz_list = [
            u4,
            thinly_dressed_cnot,
            fully_dressed_cnot,
            identity_resolvable,
            heisenberg,
        ]

    def test_given_custom_ansatz_when_compile(self):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=1)
                qc = co.unroll_to_basis_gates(qc)
                adapt_compiler = AdaptCompiler(qc, custom_layer_2q_gate=ansatz())

                result = adapt_compiler.compile()

                approx_circuit = result.circuit

                overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
                assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_given_custom_ansatz_and_all_options_when_compile(self):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=1)
                qc = co.unroll_to_basis_gates(qc)

                start_qc = QuantumCircuit(3)
                start_qc.h(range(3))

                adapt_compiler = AdaptCompiler(
                    qc,
                    custom_layer_2q_gate=ansatz(),
                    starting_circuit=start_qc,
                    initial_single_qubit_layer=True,
                    backend=SV_SIM,
                )

                result = adapt_compiler.compile()

                approx_circuit = result.circuit

                overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
                assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_given_custom_ansatz_when_add_layer_then_parameters_change(self):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3, seed=0)
                qc = co.unroll_to_basis_gates(qc)
                adapt_compiler = AdaptCompiler(qc, custom_layer_2q_gate=ansatz())

                adapt_compiler._add_layer(0)

                last_layer_instructions = adapt_compiler.full_circuit[-len(ansatz()):]
                for i, instruction in enumerate(last_layer_instructions):
                    if instruction.operation.name != "cx":
                        if ansatz is u4 and i in [14, 17]:
                            # For U(4) ansatz, the final gates on each qubit can have optimal angle zero. Which one
                            # depends on numpy version (presumably rounding differences at ~machine precision)
                            pass
                        else:
                            self.assertNotEqual(instruction.operation.params[0], 0.0)

    def test_given_custom_ansatz_and_mps_backend_when_add_layer_then_layers_cached(
        self,
    ):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3)
                qc = co.unroll_to_basis_gates(qc)
                adapt_compiler = AdaptCompiler(
                    qc,
                    custom_layer_2q_gate=ansatz(),
                    backend=MPS_SIM,
                    adapt_config=AdaptConfig(max_layers_to_modify=2),
                )

                adapt_compiler._add_layer(0)
                self.assertEqual(len(adapt_compiler.full_circuit), 1 + len(ansatz()))
                adapt_compiler._add_layer(1)
                self.assertEqual(len(adapt_compiler.full_circuit), 1 + len(ansatz()))

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
                adapt_compiler = AdaptCompiler(qc, custom_layer_2q_gate=ansatz())

                adapt_compiler._add_layer(0)
                adapt_compiler._add_layer(1)

                last_layer_instructions = adapt_compiler.full_circuit[-len(ansatz()):]

                for index in expected_cnots:
                    self.assertEqual(
                        last_layer_instructions[index].operation.name, "cx"
                    )

    def test_given_custom_thinly_dressed_when_compile_same_as_default_behaviour(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        default_adapt_compiler = AdaptCompiler(qc)
        default_result = default_adapt_compiler.compile()

        custom_adapt_compiler = AdaptCompiler(
            qc, custom_layer_2q_gate=thinly_dressed_cnot()
        )
        custom_result = custom_adapt_compiler.compile()

        self.assertEqual(default_result.overlap, custom_result.overlap)

    def test_given_use_rotoselect_false_when_add_layer_then_rotation_axes_unchanged(
        self,
    ):
        for ansatz in self.ansatz_list:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3)
                qc = co.unroll_to_basis_gates(qc)
                adapt_compiler = AdaptCompiler(
                    qc, custom_layer_2q_gate=ansatz(), use_rotoselect=False
                )

                adapt_compiler._add_layer(0)
                adapt_compiler._add_layer(1)

                last_layer_gates = adapt_compiler.full_circuit.data[-len(ansatz()):]

                for i, gate in enumerate(last_layer_gates):
                    self.assertEqual(gate[0].name, ansatz().data[i][0].name)

    def test_given_u4_or_fully_dressed_when_compile_without_rotoselect_then_works(
        self,
    ):
        for ansatz in [u4, fully_dressed_cnot]:
            with self.subTest(ansatz):
                qc = co.create_random_initial_state_circuit(3)
                qc = co.unroll_to_basis_gates(qc)
                adapt_compiler = AdaptCompiler(
                    qc, custom_layer_2q_gate=ansatz(), use_rotoselect=False
                )
                result = adapt_compiler.compile()
                overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
                self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_xxx_state_and_heisenberg_ansatz_when_compile_without_rotoselect_then_works(
        self,
    ):
        qc = QuantumCircuit(4)
        qc.x([0, 2])
        qc = trotter_circuit(
            qc, dt=0.1, delta=1.0, num_trotter_steps=2, second_order=False
        )
        adapt_compiler = AdaptCompiler(
            qc, custom_layer_2q_gate=heisenberg(), use_rotoselect=False
        )
        result = adapt_compiler.compile()
        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)
