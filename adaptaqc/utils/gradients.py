from typing import List, Tuple

import numpy as np
from aqc_research.mps_operations import mps_from_circuit, mps_dot
from qiskit import QuantumCircuit

from adaptaqc.backends.aer_mps_backend import AerMPSBackend
from adaptaqc.backends.python_default_backends import MPS_SIM
from adaptaqc.utils.circuit_operations import remove_unnecessary_2q_gates_from_circuit
from adaptaqc.utils.utilityfunctions import get_distinct_items_and_degeneracies


def general_grad_of_pairs(
    circuit: QuantumCircuit,
    inverse_zero_ansatz: QuantumCircuit,
    generators: List[QuantumCircuit],
    degeneracies: List[int],
    coupling_map: List[Tuple],
    starting_circuit=None,
    backend: AerMPSBackend = MPS_SIM,
):
    """
    For an ansatz of the form U(θ) = U_N(θ_N) * ... * U_1(θ_1), parameterised by θ = (θ_1, ..., θ_N),
    and with U_k(θ_k) = exp(-i * (θ_k / 2) * A_k), this function:
    1. Calculates the cost-gradient with respect to each θ_k at θ=0. The gradient is given by:
        dC/d(θ_k)|θ=0 = -imag(<s|G_k|ψ><ψ|U†(0)|s>) = g_k
        where:
        • |s> is the state obtained by acting with the starting_circuit on |0>
        • U†(0) is the inverse of the ansatz evaluated at θ=0
        • G_k = U_N(0) * ... * U_(k+1)(0) * A_k * U_(k-1)(0) * ... * U_1(0) I.e. the ansatz
          evaluated at θ=0 BUT with U_k replaced by its generator A_k
    2. Calculates the Euclidean norm of the gradients: g = sqrt(g_1 ** 2 + ... + g_N ** 2)
    3. Returns a list of the gradient g for each pair in the coupling map

    Args:
        circuit (QuantumCircuit): a circuit representing |ψ>
        inverse_zero_ansatz (QuantumCircuit): a circuit representing U†(0)
        generators (List[QuantumCircuit]): a list of quantum circuits representing (G_k)†
        degeneracies (List[int]): a list of the degeneracies of the generators
        coupling_map (List[Tuple]): the list of all pairs of qubits for which to calculate the gradient
        starting_circuit (QuantumCircuit): a circuit representing |s>
        backend (AerSimulator): Aer MPS simulator used to generate relevant states
    Returns:
        gradients (List): List of gradients g for each pair
    """
    gradients = []
    ansatz_resolves_to_id = inverse_zero_ansatz == QuantumCircuit(2)

    # Get MPS of |ψ>
    circ_mps = mps_from_circuit(
        circuit.copy(), return_preprocessed=True, sim=backend.simulator
    )

    # Get the starting circuit
    if starting_circuit is not None:
        starting_circuit = starting_circuit
    else:
        starting_circuit = QuantumCircuit(circuit.num_qubits)

    # Only calculate <ψ|U†(0)|s> = <ψ|s> once if ansatz resolves to identity
    if ansatz_resolves_to_id:
        # Find |s>
        starting_circuit_mps = mps_from_circuit(
            starting_circuit.copy(), return_preprocessed=True, sim=backend.simulator
        )
        # Find <ψ|s>
        zero_ansatz_overlap = mps_dot(
            circ_mps, starting_circuit_mps, already_preprocessed=True
        )

    for control, target in coupling_map:
        # Calculate <ψ|U†(0)|s> for each pair if ansatz does not resolve to identity
        if not ansatz_resolves_to_id:
            # Find U†(0)|s>
            ansatz_on_starting_circuit = starting_circuit.compose(
                inverse_zero_ansatz, [control, target]
            )
            ansatz_on_starting_circuit_mps = mps_from_circuit(
                ansatz_on_starting_circuit,
                return_preprocessed=True,
                sim=backend.simulator,
            )
            # Find <ψ|U†(0)|s>
            zero_ansatz_overlap = mps_dot(
                circ_mps, ansatz_on_starting_circuit_mps, already_preprocessed=True
            )

        gradient = 0
        for i, generator in enumerate(generators):
            # Find (G_k)†|s>
            generator_on_starting_circuit = starting_circuit.compose(
                generator, [control, target]
            )
            generator_on_starting_circuit_mps = mps_from_circuit(
                generator_on_starting_circuit,
                return_preprocessed=True,
                sim=backend.simulator,
            )
            # Find <s|G_k|ψ>, computed as the dot product of (G_k)†|s> and |ψ>
            generator_overlap = mps_dot(
                generator_on_starting_circuit_mps, circ_mps, already_preprocessed=True
            )

            generator_gradient = -1 * np.imag(generator_overlap * zero_ansatz_overlap)

            # Add contribution to gradient from generator, accounting for degeneracy
            gradient += (generator_gradient**2) * degeneracies[i]

        # Calculate the Euclidean norm of the gradients for each generator
        grad_norm = np.sqrt(gradient)

        gradients.append(grad_norm)

    return gradients


def get_generators_and_degeneracies(
    ansatz: QuantumCircuit, rotoselect: bool = False, inverse: bool = False
):
    """
    For an ansatz of the form U(θ) = U_N(θ_N) * ... * U_1(θ_1), parameterised by θ = (θ_1, ..., θ_N),
    and with U_k(θ_k) = exp(-i * (θ_k / 2) * A_k), this function finds the generators of the ansatz:

    G_k = U_N(0) * ... * U_(k+1)(0) * A_k * U_(k-1)(0) * ... * U_1(0) I.e. the ansatz evaluated at
    θ=0 BUT with U_k replaced by its generator A_k.

    If rotoselect=True, for every rotation gate in the ansatz, return all three generators as if the
    rotation gate was Rx, Ry, or Rz.

    Args:
        ansatz (QuantumCircuit): a circuit representing the ansatz U
        rotoselect (bool): set to True to return the x, y, z generators for each rotation gate, set
        to False to only return the specific generator for the gate.
        inverse (bool): set to True to return the inverse of the generators
    Returns:
        generator_circuits (List[QuantumCircuit]): List of generators G_k (or their inverses), one
        for each parameterised gate if rotoselect=False, three if rotoselect=True.
        degeneracies (List[int]): List of degeneracies of generators.
    """
    parameterised_gates = ["rx", "ry", "rz"]
    generator_circuits = []
    for i, circ_instr in enumerate(ansatz):
        if circ_instr.operation.name in parameterised_gates:
            if rotoselect:
                # Get all Rx, Ry, Rz generators
                for op in parameterised_gates:
                    generator = get_generator(ansatz, i, op)
                    generator_circuits.append(
                        generator.inverse() if inverse else generator
                    )
            else:
                # Get the generator for the specific gate
                generator = get_generator(ansatz, i, circ_instr.operation.name)
                generator_circuits.append(generator.inverse() if inverse else generator)

    distinct_generators, degeneracies = get_distinct_items_and_degeneracies(
        generator_circuits
    )

    return (distinct_generators, degeneracies)


def get_generator(ansatz: QuantumCircuit, index: int, op: str):
    """
    Given an ansatz consisting of only rx, ry, rz and cx gates, this function replaces the gate at
    index=index with the generator of op, removes all other rotation gates, and removes consecutive
    cx gates that would resolve to the identity.

    Example:
    index = 4, op = 'ry', ansatz:
         ┌───────┐     ┌───────┐     ┌───────┐
    q_0: ┤ Rx(0) ├──■──┤ Rx(0) ├──■──┤ Rx(0) ├
         ├───────┤┌─┴─┐├───────┤┌─┴─┐├───────┤
    q_1: ┤ Rx(0) ├┤ X ├┤ Rx(0) ├┤ X ├┤ Rx(0) ├
         └───────┘└───┘└───────┘└───┘└───────┘

    will return:
    q_0: ──■─────────■──
         ┌─┴─┐┌───┐┌─┴─┐
    q_1: ┤ X ├┤ Y ├┤ X ├
         └───┘└───┘└───┘

    Args:
        ansatz (QuantumCircuit): a circuit representing the ansatz
        index: the index of the operator to be replaced
        op: the operator, one of rx, ry or rz, the generator of which will replace the gate at
        index=index
    Returns:
        generator (QuantumCircuit): The generator
    """
    supported_ops = ["rx", "ry", "rz"]
    if op not in supported_ops:
        raise ValueError("op must be one of rx, ry or rz")

    generator = QuantumCircuit(2)
    for i, circ_instr in enumerate(ansatz):
        operation = circ_instr.operation
        qubits = circ_instr.qubits
        if operation.name not in ["rx", "ry", "rz", "cx"]:
            raise ValueError("Circuit must only contain rx, ry, rz and cx gates")
        if i == index:
            if op == "rx":
                generator.x(qubits[0])
            if op == "ry":
                generator.y(qubits[0])
            if op == "rz":
                generator.z(qubits[0])
        if operation.name == "cx":
            generator.cx(qubits[0], qubits[1])

    # remove consecutive cx gates which resolve to the identity
    remove_unnecessary_2q_gates_from_circuit(generator)

    return generator
