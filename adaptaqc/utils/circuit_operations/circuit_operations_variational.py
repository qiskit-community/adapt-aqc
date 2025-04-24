import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import U3Gate, U1Gate

from adaptaqc.utils.circuit_operations.circuit_operations_basic import (
    is_supported_1q_gate,
)
from adaptaqc.utils.utilityfunctions import normalized_angles


def find_angles_in_circuit(circuit, gate_range=None):
    """
    Find the angles of rotation gates in the circuit. Ordering is relative
    to position of gate in circuit.data.
    Ignores gates that have the label FIXED_GATE_LABEL
    :param circuit: QuantumCircuit
    :param gate_range: The search space for the angles (full circuit if None)
    :return: Angles in circuit (list)
    """
    angles = []
    if gate_range is None:
        gate_range = (0, len(circuit.data))
    angle_index = 0
    for gate_index in range(*gate_range):
        gate = circuit.data[gate_index].operation
        if is_supported_1q_gate(gate):
            # Normalize angle to between -pi and pi
            angles += [normalized_angles(gate.params[0])]
            angle_index += 1
    return angles


def update_angles_in_circuit(circuit: QuantumCircuit, angles, gate_range=None):
    """
    Changes the angle of all rotation gates in the circuit except those with
    label = FIXED_GATE_LABEL
    :param circuit: Circuit to modify
    :param angles: New angles (list/np.ndarray)
    :param gate_range: The range of gates in which the 1q gates are located
    for the angles (full circuit if None)
    """
    if gate_range is None:
        gate_range = (0, len(circuit.data))
    angle_index = 0
    for gate_index in range(*gate_range):
        circ_instr = circuit.data[gate_index]
        gate = circ_instr.operation
        if is_supported_1q_gate(gate):
            gate.params[0] = angles[angle_index]
            angle_index += 1
            circuit.data[gate_index] = circ_instr


def create_variational_circuit(circuit: QuantumCircuit):
    new_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for circ_instr in circuit.data:
        gate = circ_instr.operation
        if isinstance(gate, U1Gate):
            gate.label = "rz"
        elif isinstance(gate, U3Gate):
            if gate.params[1] == gate.params[2] and gate.params[2] == 0:
                gate.label = "ry"
            elif np.isclose(-0.5 * np.pi, gate.params[1]) and np.isclose(
                0.5 * np.pi, gate.params[2]
            ):
                gate.label = "rx"
        new_circ.data.append(
            CircuitInstruction(
                operation=gate, qubits=circ_instr.qubits, clbits=circ_instr.clbits
            )
        )
    return new_circ
