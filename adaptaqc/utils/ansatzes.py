# (C) Copyright IBM 2025. 
# 
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit


def u4():
    """
    U(4) ansatz from Fig. 6 of
    Vatan, Farrokh, and Colin Williams. "Optimal quantum circuits for general two-qubit gates."
    Physical Review A 69.3 (2004): 032315.
    """
    qc = QuantumCircuit(2)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    qc.cx(1, 0)
    qc.rz(0, 0)
    qc.ry(0, 1)
    qc.cx(0, 1)
    qc.ry(0, 1)
    qc.cx(1, 0)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    return qc


def thinly_dressed_cnot():
    qc = QuantumCircuit(2)
    qc.rx(0, 0)
    qc.rx(0, 1)
    qc.cx(0, 1)
    qc.rx(0, 0)
    qc.rx(0, 1)
    return qc


def fully_dressed_cnot():
    qc = QuantumCircuit(2)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    qc.cx(0, 1)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    return qc


def identity_resolvable():
    qc = QuantumCircuit(2)
    qc.rx(0, 0)
    qc.rx(0, 1)
    qc.cx(0, 1)
    qc.rx(0, 0)
    qc.rx(0, 1)
    qc.cx(0, 1)
    qc.rx(0, 0)
    qc.rx(0, 1)
    return qc


def heisenberg():
    """
    Based on fig 2. from N. Robertson et al. "Approximate Quantum Compiling for Quantum Simulation: A Tensor Network
    based approach" arxiv:2301.08609, which gives circuit representing two site evolution operator
    e^(iαXX + iβYY + iγZZ) corresponding to XYZ Heisenberg model with no field. Here, we additionally allow for the Rz
    gates applied (at the end) to the first qubit and (at the start) to the second qubit to be trainable, to mimic
    additional (learnable) evolution under an external field (effectively a first order trotter expansion).
    """
    qc = QuantumCircuit(2)
    qc.rz(0.0, 1)
    qc.cx(1, 0)
    qc.rz(0.0, 0)
    qc.ry(0.0, 1)
    qc.cx(0, 1)
    qc.ry(0.0, 1)
    qc.cx(1, 0)
    qc.rz(0.0, 0)
    return qc
