"""Contains functions"""

import copy
import functools
from collections.abc import Iterable
from typing import Union, Dict, List, Tuple

import aqc_research.mps_operations as mpsop
import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.result import Counts
from qiskit_aer.backends.compatibility import Statevector
from tenpy import SpinHalfSite, SpinSite
from tenpy.networks.mps import MPS

from adaptaqc.backends.aer_sv_backend import AerSVBackend
from adaptaqc.utils.circuit_operations import SUPPORTED_1Q_GATES


# ------------------Trigonometric functions------------------ #


def minimum_of_sinusoidal(value_0, value_pi_by_2, value_minus_pi_by_2):
    """
    Find the minimum of a sinusoidal function with period 2*pi and of the
    form f(x) = a*sin(x+b)+c
    :param value_0: f(0)
    :param value_pi_by_2: f(pi/2)
    :param value_minus_pi_by_2: f(-pi/2)
    :return: (x_min, f(x_min))
    """
    theta_min = -(np.pi / 2) - np.arctan2(
        2 * value_0 - value_pi_by_2 - value_minus_pi_by_2,
        value_pi_by_2 - value_minus_pi_by_2,
    )

    theta_min = normalized_angles(theta_min)

    intercept_c = 0.5 * (value_pi_by_2 + value_minus_pi_by_2)
    value_pi = (value_pi_by_2 + value_minus_pi_by_2) - value_0
    amplitude_a = 0.5 * (
        ((value_0 - value_pi) ** 2 + (value_pi_by_2 - value_minus_pi_by_2) ** 2) ** 0.5
    )
    value_theta_min = intercept_c - amplitude_a

    return theta_min, value_theta_min


def amplitude_of_sinusoidal(value_0, value_pi_by_2, value_minus_pi_by_2):
    """
    Find the amplitude of a sinusoidal function with period 2*pi and of the
    form f(x) = a*sin(x+b)+c
    :param value_0: f(0)
    :param value_pi_by_2: f(pi/2)
    :param value_minus_pi_by_2: f(-pi/2)
    :return: Amplitude
    """

    value_pi = (value_pi_by_2 + value_minus_pi_by_2) - value_0
    amplitude_a = 0.5 * (
        ((value_0 - value_pi) ** 2 + (value_pi_by_2 - value_minus_pi_by_2) ** 2) ** 0.5
    )

    return amplitude_a


def derivative_of_sinusoidal(theta, value_0, value_pi_by_2, value_minus_pi_by_2):
    """
    Find the derivative of a sinusoidal function with period 2*pi and of the
    form f(x) = a*sin(x+b)+c at x=theta
    :param theta: Angle at which derivative is to be evaluated
    :param value_0: f(0)
    :param value_pi_by_2: f(pi/2)
    :param value_minus_pi_by_2: f(-pi/2)
    :return: df(x)/dx at x=theta
    """
    value_pi = (value_pi_by_2 + value_minus_pi_by_2) - value_0
    amplitude_a = 0.5 * (
        ((value_0 - value_pi) ** 2 + (value_pi_by_2 - value_minus_pi_by_2) ** 2) ** 0.5
    )
    phase_b = np.arctan2(value_0 - value_pi, value_pi_by_2 - value_minus_pi_by_2)

    derivative = amplitude_a * np.cos(theta + phase_b)
    return derivative


def normalized_angles(angles):
    """
    Normalize angle(s) to between -pi, pi by adding/subtracting multiples of
    2pi
    :param angles: float or Iterable(float)
    :return: float or Iterable(float)
    """
    single = not isinstance(angles, Iterable)
    if single:
        angles = [angles]
    new_angles = []
    for angle in angles:
        while (angle > np.pi) or (angle < -np.pi):
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < np.pi:
                angle += 2 * np.pi
        new_angles += [angle]
    return new_angles[0] if single else new_angles


# ------------------Misc. functions------------------ #


def is_statevector_backend(backend):
    """
    Check if backend is a statevector simulator backed
    :param backend: Simulator backend
    :return: Boolean
    """
    if isinstance(backend, AerSVBackend):
        return True
    return False


def counts_data_from_statevector(
    statevector,
    num_shots=2**40,
):
    """
    Get counts data from statevector by multiplying amplitude squares with num_shots.
    Note: Doesn't guarantee total number of shots in returned counts data will be num_shots.
    Warning: Doesn't work well if num_shots << number of non-zero elements in statevector
    :param statevector: Statevector (list/array)
    :return: Counts data (e.g. {'00':13, '10':7}) with bitstrings ordered
        with decreasing qubit number
    """
    num_qubits = int(np.log2(len(statevector)))
    counts = {}
    probs = np.absolute(statevector) ** 2
    bit_str_array = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]
    counts = dict(zip(bit_str_array, np.asarray(probs * num_shots, int)))
    # counts = dict(zip(*np.unique(np.random.choice(bit_str_array, num_shots,p=probs),return_counts=True)))
    return counts


def statevector_from_counts_data(counts):
    """
    Get statevector from counts (works only for real, positive states)
    :param: Counts data (e.g. {'00':13, '10':7})
    :return statevector: Statevector (list/array)
    """
    num_qubits = len(list(counts.keys())[0])
    sv = np.zeros(2**num_qubits)
    for i in range(2**num_qubits):
        bitstr = bin(i)[2:].zfill(num_qubits)
        if bitstr in counts:
            sv[i] = counts[bitstr] ** 0.5
    sv /= np.linalg.norm(sv)
    return sv


def expectation_value_of_qubits(data: Union[Counts, Dict, Statevector]):
    """
    Expectation value of qubits (in computational basis)
    :param counts: Counts data (e.g. {'00':13, '10':7})
    :return: [expectation_value(float)]
    """
    data = Statevector(data) if isinstance(data, np.ndarray) else data

    num_qubits = (
        data.num_qubits if isinstance(data, Statevector) else len(list(data)[0])
    )

    expectation_values = []
    for i in range(num_qubits):
        expectation_values.append(_expectation_value_of_qubit(i, data, num_qubits))
    return expectation_values


def expectation_value_of_qubits_mps(circuit: QuantumCircuit, sim=None):
    """
    Expectation value of qubits (in computational basis) using mps
    :param circuit: Circuit corresponding to state
    :param sim: MPS AerSimulator instance. If none, will use default in AQC Research.
    :return: [expectation_value(float)]
    """
    # Get mps from circuit
    circ = circuit.copy()
    mps = mpsop.mps_from_circuit(circ, return_preprocessed=True, sim=sim)

    num_qubits = circuit.num_qubits

    expectation_values = [
        (mpsop.mps_expectation(mps, "Z", i, already_preprocessed=True))
        for i in range(num_qubits)
    ]
    return expectation_values


def _expectation_value_of_qubit(
    qubit_index, data: Union[Counts, Statevector], num_qubits
):
    """
    Expectation value of qubit (in computational basis) at given index
    :param qubit_index: Index of qubit (int)
    :param data: Counts data (e.g. {'00':13, '10':7}) or Statevector
    :return: [expectation_value(float)]
    """
    if qubit_index >= num_qubits:
        raise ValueError("qubit_index outside of register range")

    reverse_index = num_qubits - (qubit_index + 1)

    if type(data) is Statevector:
        [p0, p1] = data.probabilities([qubit_index])
        exp_val = p0 - p1
        return exp_val

    else:
        exp_val = 0
        total_counts = 0
        for bitstring in list(data):
            exp_val += (1 if bitstring[reverse_index] == "0" else -1) * data[bitstring]
            total_counts += data[bitstring]
        return exp_val / total_counts


def expectation_value_of_pauli_observable(counts, pauli):
    """
    Copied from measure_pauli_z in qiskit.aqua.operators.common

    Args:
        counts (dict): a dictionary of the form counts = {'00000': 10} ({
            str: int})
        pauli (Pauli): a Pauli object
    Returns:
        float: Expected value of paulis given data
    """
    observable = 0.0
    num_shots = sum(counts.values())
    p_z_or_x = np.logical_or(pauli.z, pauli.x)
    for key, value in counts.items():
        bitstr = np.asarray(list(key))[::-1].astype(bool)
        sign = (
            -1.0
            if functools.reduce(np.logical_xor, np.logical_and(bitstr, p_z_or_x))
            else 1.0
        )
        observable += sign * value
    observable /= num_shots
    return observable


def remove_permutations_from_coupling_map(coupling_map):
    seen = set()
    unique_list = []
    for pair in coupling_map:
        if tuple(sorted(pair)) not in seen:
            seen.add(tuple(sorted(pair)))
            unique_list.append(pair)
    return unique_list


def has_stopped_improving(cost_history, rel_tol=1e-2):
    try:
        poly_fit_res = np.polyfit(list(range(len(cost_history))), cost_history, 1)
        grad = poly_fit_res[0] / np.absolute(np.mean(cost_history))
        return grad > -1 * rel_tol
    except np.linalg.LinAlgError:
        return False


def multi_qubit_gate_depth(qc: QuantumCircuit) -> int:
    """
    Return the multi-qubit gate depth.

    When the circuit has been transpiled for IBM Quantum hardware
    this will be equivalent to the CNOT depth.
    """
    return qc.depth(filter_function=lambda instr: len(instr.qubits) > 1)


def tenpy_to_qiskit_mps(tenpy_mps):
    num_sites = tenpy_mps.L
    tenpy_mps.canonical_form()

    # Check convention of basis states
    flip = check_flipped_basis_states(tenpy_mps)

    gam = [0] * num_sites
    lam = [0] * (num_sites - 1)
    permutation = None
    for n in range(num_sites):
        # Get the tenpy "B" tensor for site n, with indices in Qiskit MPS order (p, L, R)
        g_n = tenpy_mps.get_B(n, form="G").itranspose(["p", "vL", "vR"]).to_ndarray()
        if permutation is not None:
            g_n[:] = g_n[
                :, permutation, :
            ]  # permute left index in the same way the left singlular values were permuted
        if n < num_sites - 1:
            l_n = tenpy_mps.get_SR(n)  # Get singular values to the right of tensor n
            permutation = np.argsort(l_n)[::-1]
            l_n = np.sort(l_n)[::-1]  # Sort singular values in descending order
            lam[n] = l_n
            if permutation is not None:
                g_n[:] = g_n[
                    :, :, permutation
                ]  # permute right index in the same way the right singular values were permuted

        # Split physical dimension into two parts of a tuple
        if flip[n]:
            gam[n] = (g_n[1], g_n[0])
        else:
            gam[n] = (g_n[0], g_n[1])

    qiskit_mps = (gam, lam)

    return copy.deepcopy(qiskit_mps)


def tenpy_chi_1_mps_to_circuit(mps: MPS) -> QuantumCircuit:
    if not np.allclose(mps.chi, 1):
        raise Exception("MPS must have bond dimension 1 for all bonds.")

    flip = check_flipped_basis_states(mps)

    qc = QuantumCircuit(mps.L)
    for i in range(mps.L):
        # 2 x 1 x 1 array representing the state of site i
        array = mps.get_B(i, form="B").itranspose(["p", "vL", "vR"]).to_ndarray()
        # Extract the length-2 vector, with the correct basis-ordering
        if flip[i]:
            vec = array[::-1, 0, 0]
        else:
            vec = array[:, 0, 0]

        # Make unitary with column 0 corresponding to the state of site i
        U = np.zeros((2, 2), dtype=array.dtype)
        U[:, 0] = vec
        U[0, 1] = np.conj(U[1, 0])
        U[1, 1] = -np.conj(U[0, 0])
        qc.unitary(U, i)

    qc = transpile(qc, basis_gates=["rx", "ry", "rz"])
    return qc


def qiskit_to_tenpy_mps(qiskit_mps, return_form: str = "SpinSite") -> MPS:
    """
    Converts a Qiskit MPS to a Tenpy MPS.

    Args:
        qiskit_mps: The Qiskit MPS.
        return_form: The type of site to use for the Tenpy MPS.
    Returns:
        tenpy_mps: The Tenpy MPS
    """
    # If not preprocessed, preprocess MPS
    if isinstance(qiskit_mps[0], List):
        qiskit_mps = mpsop._preprocess_mps(qiskit_mps)

    N = len(qiskit_mps)

    if return_form == "SpinSite":
        sites = [SpinSite(conserve=None)] * N
        # Flip basis state ordering for SpinSite
        qiskit_mps = [tensor[::-1, :, :] for tensor in qiskit_mps]
    elif return_form == "SpinHalfSite":
        sites = [SpinHalfSite(conserve=None)] * N
    else:
        raise ValueError(
            f"Invalid return_form: {return_form}. Must be SpinSite or SpinHalfSite"
        )

    tenpy_mps = MPS.from_Bflat(sites, qiskit_mps, SVs=None)

    return tenpy_mps


def find_rotation_indices(qc: QuantumCircuit, indices: List[int]) -> List[int]:
    """
    Given a QuantumCircuit and a list of indices, returns a list containing the subset of indices
    corresponding to rotation gates in the circuit
    """
    rotation_indices = []
    for index in indices:
        if qc.data[index].operation.name in SUPPORTED_1Q_GATES:
            rotation_indices.append(index)

    return rotation_indices


def get_distinct_items_and_degeneracies(items: List) -> Tuple[List, List[int]]:
    """
    Given a list of items, return a list containing the distinct items, along with their
    degeneracies (number of repetitions).

    Args:
        items: List of items.
    Returns:
        distinct_items: List of distinct items.
        degeneracies: List of degeneracies.
    """
    distinct_items = []
    degeneracies = []
    for i in range(len(items)):
        item = items[i]
        distinct = True
        for j in range(len(distinct_items)):
            if item == distinct_items[j]:
                degeneracies[j] += 1
                distinct = False
                break
        if distinct:
            distinct_items.append(item)
            degeneracies.append(1)

    return (distinct_items, degeneracies)


def check_flipped_basis_states(mps: MPS) -> List[bool]:
    """
    Given a Tenpy MPS, generate a list where the ith element is False(True) if the ith site of the
    MPS is(isn't) ordering the basis states with the same convention as Qiskit.

    Args:
        mps: The Tenpy MPS.
    Returns:
        flipped_basis_states: The list of basis conventions.
    """

    flipped_basis_states = [None] * mps.L

    for i in range(mps.L):
        sz_matrix = mps.sites[i].get_op("Sz").to_ndarray()
        if np.array_equal(sz_matrix, [[0.5, 0], [0, -0.5]]):
            flipped_basis_states[i] = False
        elif np.array_equal(sz_matrix, [[-0.5, 0], [0, 0.5]]):
            flipped_basis_states[i] = True
        else:
            raise ValueError(f"Invalid Tenpy convention for site {i}")

    return flipped_basis_states


def tenpy_mps_to_statevector(mps: MPS) -> np.ndarray:
    """
    Convert a Tenpy MPS to a little-endian statevector

    Args:
        mps: The MPS.
    Returns:
        sv: The statevector.
    """

    # Get the 2 x 2 x ... tensor representing the state
    sv = mps.get_theta(0, mps.L).to_ndarray().reshape([2] * mps.L)

    # Flip the basis ordering for any sites using the opposite convention to Qiskit
    flip = check_flipped_basis_states(mps)
    for i in range(mps.L):
        if flip[i]:
            sv = np.flip(sv, axis=i)
        else:
            continue

    # Convert from big-endian to little-endian ordering
    sv = np.transpose(sv, axes=range(mps.L)[::-1])

    # Convert to 2^N dimensional vector
    sv = sv.flatten()

    return sv
