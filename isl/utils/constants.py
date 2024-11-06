"""Contains constants"""
from typing import List, Optional, Tuple
import numpy as np

# Type of MPS data as it outputted by Qiskit.
QiskitMPS = Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]

ALG_ROTOSOLVE = "rotosolve"
ALG_ROTOSELECT = "rotoselect"
ALG_NLOPT = "nlopt"
ALG_SCIPY = "scipy"
ALG_PYBOBYQA = "pybobyqa"

FIXED_GATE_LABEL = "fixed_gate"

CMAP_FULL = "CMAP_FULL"
CMAP_LINEAR = "CMAP_LINEAR"
CMAP_LADDER = "CMAP_LADDER"

DEFAULT_SUFFICIENT_COST = 1e-2

def generate_coupling_map(num_qubits, map_kind, both_dir=False, loop=False):
    if map_kind == CMAP_FULL:
        return coupling_map_fully_entangled(num_qubits, both_dir)
    elif map_kind == CMAP_LINEAR:
        return coupling_map_linear(num_qubits, both_dir, loop)
    elif map_kind == CMAP_LADDER:
        return coupling_map_ladder(num_qubits, both_dir, loop)
    else:
        raise ValueError(f"Invalid coupling map type {map_kind}")


def coupling_map_fully_entangled(num_qubits, both_dir=False):
    """
    Coupling map with all qubits connected to each other
    :param num_qubits: Number of qubits
    :param both_dir: If true, map will include gates with control and target
    swapped
    :return: [(control(int),target(int))]
    """
    c_map = []
    for i in range(1, num_qubits):
        for j in range(num_qubits - i):
            c_map.append((j, j + i))
    if both_dir:
        c_map_rev = [(target, source) for (source, target) in c_map]
        c_map += c_map_rev
    return c_map


def coupling_map_linear(num_qubits, both_dir=False, loop=False):
    """
    Coupling map with qubits connected to adjacent qubits
    :param num_qubits: Number of qubits
    :param both_dir: If true, map will include gates with control and target
    swapped
    :param loop: If true, the first qubit will be connected to the last
    qubit as well
    :return: [(control(int),target(int))]
    """
    c_map = []
    for j in range(num_qubits - 1):
        c_map.append((j, j + 1))
    if loop:
        c_map.append((num_qubits - 1, 0))
    if both_dir:
        c_map_rev = [(target, source) for (source, target) in c_map]
        c_map += c_map_rev
    return c_map


def coupling_map_ladder(num_qubits, both_dir=False, loop=False):
    """
    Low depth coupling map with qubits connected to adjacent qubits
    :param num_qubits: Number of qubits
    :param both_dir: If true, map will include gates with control and target
    swapped
    :param loop: If true, the first qubit will be connected to the last
    qubit as well
    :return: [(control(int),target(int))]
    """
    c_map = []
    j = 0
    while j + 1 <= num_qubits - 1:
        c_map.append((j, j + 1))
        j += 2
    j = 1
    if loop and num_qubits % 2 == 1:
        c_map.append((num_qubits - 1, 0))
    while j + 1 <= num_qubits - 1:
        c_map.append((j, j + 1))
        j += 2
    if loop and num_qubits % 2 == 0:
        c_map.append((num_qubits - 1, 0))
    if both_dir:
        c_map_rev = [(target, source) for (source, target) in c_map]
        c_map += c_map_rev
    return c_map


def convert_cmap_to_qiskit_format(c_map):
    """
    Convert a list of tuples to a list of lists that qiskit expects for transpiling with a c_map.
    :param c_map: List of tuples [(int, int)]
    :return: List of lists [[int, int]]
    """
    return [list(pair) for pair in c_map]


def get_initial_layout(circuit):
    """
    Extracts initial layout of a circuit.

    :param circuit: The original circuit to determine the layout for.
    :return: Dictionary for initial_layout in the form {logical_qubit: physical_qubit}
    """
    # map logical qubits to their indices in the circuit
    initial_layout = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    return initial_layout
