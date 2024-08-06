from typing import List, Tuple
import aqc_research.mps_operations as mpsop
import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def xx_grad_of_pairs(circuit: QuantumCircuit, coupling_map: List[Tuple], sim=None):
    """
    Returns the magnitude of the cost-gradient of each qubit pair in the coupling map w.r.t Rxx(θ) at θ=0
    The gradient takes the form:
    dC/dθ|θ=0 = -imag(<0|XX|ψ><ψ|0>)
    where XX acts on the qubit pair, and |ψ> is the state prepared by circuit

    Args:
        circuit (QuantumCircuit): a circuit representing |ψ>
        coupling_map (List[Tuple]): the list of all pairs of qubits for which to calculate the gradient
        sim (AerSimulator): Aer MPS simulator used to generate relevant states
    Returns:
        gradients (List): List of gradients w.r.t. Rxx(θ) at θ=0 for each pair
    """
    gradients = []
    circ_mps = mpsop.mps_from_circuit(circuit.copy(), return_preprocessed=True, sim=sim)
    zero_mps = mpsop.mps_from_circuit(
        QuantumCircuit(circuit.num_qubits), return_preprocessed=True, sim=sim
    )
    # Find <ψ|0>
    zero_overlap = mpsop.mps_dot(circ_mps, zero_mps, already_preprocessed=True)
    for control, target in coupling_map:
        circ = circuit.copy()
        circ.x(control)
        circ.x(target)
        xx_mps = mpsop.mps_from_circuit(circ, return_preprocessed=True, sim=sim)
        # Find <0|XX|ψ>
        xx_overlap = mpsop.mps_dot(zero_mps, xx_mps, already_preprocessed=True)
        gradient = -1 * np.imag(xx_overlap * zero_overlap)
        gradients.append(abs(gradient))

    return gradients


def general_grad_of_pairs(
    circuit: QuantumCircuit,
    inverse_zero_ansatz: QuantumCircuit,
    generators: List[QuantumCircuit],
    coupling_map: List[Tuple],
    starting_circuit=None,
    sim=None,
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
        coupling_map (List[Tuple]): the list of all pairs of qubits for which to calculate the gradient
        starting_circuit (QuantumCircuit): a circuit representing |s>
        sim (AerSimulator): Aer MPS simulator used to generate relevant states
    Returns:
        gradients (List): List of gradients g for each pair
    """
    gradients = []

    # Get MPS of |ψ>
    circ_mps = mpsop.mps_from_circuit(circuit.copy(), return_preprocessed=True, sim=sim)

    # Get the starting circuit
    if starting_circuit is not None:
        starting_circuit = starting_circuit
    else:
        starting_circuit = QuantumCircuit(circuit.num_qubits)

    # TODO optimise for when ansatz resolves to identity
    for control, target in coupling_map:
        # Find U†(0)|s>
        ansatz_on_starting_circuit = starting_circuit.compose(
            inverse_zero_ansatz, [control, target]
        )
        ansatz_on_starting_circuit_mps = mpsop.mps_from_circuit(
            ansatz_on_starting_circuit, return_preprocessed=True, sim=sim
        )
        # Find <ψ|U†(0)|s>
        zero_ansatz_overlap = mpsop.mps_dot(
            circ_mps, ansatz_on_starting_circuit_mps, already_preprocessed=True
        )

        generator_gradients = []
        for generator in generators:
            # Find (G_k)†|s>
            generator_on_starting_circuit = starting_circuit.compose(
                generator, [control, target]
            )
            generator_on_starting_circuit_mps = mpsop.mps_from_circuit(
                generator_on_starting_circuit, return_preprocessed=True, sim=sim
            )
            # Find <s|G_k|ψ>, computed as the dot product of (G_k)†|s> and |ψ>
            generator_overlap = mpsop.mps_dot(
                generator_on_starting_circuit_mps, circ_mps, already_preprocessed=True
            )

            generator_gradient = -1 * np.imag(generator_overlap * zero_ansatz_overlap)
            generator_gradients.append(generator_gradient)

        # Calculate the Euclidean norm of the gradients for each generator
        grad_norm = np.sqrt(
            sum([generator_gradient**2 for generator_gradient in generator_gradients])
        )
        gradients.append(grad_norm)

    return gradients


def get_generators(
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
    """
    parameterised_gates = ["rx", "ry", "rz"]
    generator_circuits = []
    for i, (operation, qubits, clbits) in enumerate(ansatz):
        if operation.name in parameterised_gates:
            if rotoselect:
                # Get all Rx, Ry, Rz generators
                for op in parameterised_gates:
                    generator = get_generator(ansatz, i, op)
                    generator_circuits.append(
                        generator.inverse() if inverse else generator
                    )
            else:
                # Get the generator for the specific gate
                generator = get_generator(ansatz, i, operation.name)
                generator_circuits.append(generator.inverse() if inverse else generator)

    return generator_circuits


def get_generator(ansatz: QuantumCircuit, index: int, op: str):
    """
    Given an ansatz consisting of only rx, ry, rz and cx gates, this function replaces the gate at
    index=index with the generator of op, and then removes all rotation gates.

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
    for i, (operation, qubits, clbits) in enumerate(ansatz):
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

    # transpile to remove e.g. rotation gates of angle 0 or consecutive cx's
    generator = transpile(generator, basis_gates=["x", "y", "z", "cx"])

    return generator
