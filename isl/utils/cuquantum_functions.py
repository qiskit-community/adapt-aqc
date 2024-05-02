"""Contains functions for cuquantum backend operations"""
import logging
from copy import copy

from isl.utils.circuit_operations import DEFAULT_CU_ALGORITHM

logger = logging.getLogger(__name__)

try: 
    import cupy as cp
    from cuquantum import contract
    from cuquantum import CircuitToEinsum, cutensornet as cutn
    from cuquantum.cutensornet.experimental import contract_decompose
except ModuleNotFoundError as e:
    logger.debug(e)

def _mps_site_right_swap(
    mps_tensors, 
    i, 
    algorithm=None, 
    options=None
):
    """
    Perform the swap operation between the ith and i+1th MPS tensors.
    """
    # contraction followed by QR decomposition
    a, _, b = contract_decompose('ipj,jqk->iqj,jpk', *mps_tensors[i:i+2], algorithm=algorithm, options=options)
    mps_tensors[i:i+2] = (a, b)
    return mps_tensors

def _apply_gate(
    mps_tensors, 
    gate, 
    qubits, 
    algorithm=None, 
    options=None
):
    """
    Apply the gate operand to the MPS tensors in-place.
    
    Args:
        mps_tensors: A list of rank-3 ndarray-like tensor objects. 
            The indices of the ith tensor are expected to be the bonding index to the i-1 tensor, 
            the physical mode, and then the bonding index to the i+1th tensor.
        gate: A ndarray-like tensor object representing the gate operand. 
            The modes of the gate is expected to be output qubits followed by input qubits, e.g, 
            ``A, B, a, b`` where ``a, b`` denotes the inputs and ``A, B`` denotes the outputs. 
        qubits: A sequence of integers denoting the qubits that the gate is applied onto.
        algorithm: The contract and decompose algorithm to use for gate application. 
            Can be either a `dict` or a `ContractDecomposeAlgorithm`.
        options: Specify the contract and decompose options. 
    
    Returns:
        The updated MPS tensors.
    """
    
    n_qubits = len(qubits)
    if n_qubits == 1:
        # single-qubit gate
        i = qubits[0]
        mps_tensors[i] = contract('ipj,qp->iqj', mps_tensors[i], gate, options=options) # in-place update
    elif n_qubits == 2:
        # two-qubit gate
        i, j = qubits
        if i > j:
            # swap qubits order
            return _apply_gate(mps_tensors, gate.transpose(1,0,3,2), (j, i), algorithm=algorithm, options=options)
        elif i+1 == j:
            # two adjacent qubits
            a, _, b = contract_decompose('ipj,jqk,rspq->irj,jsk', *mps_tensors[i:i+2], gate, algorithm=algorithm, options=options)
            mps_tensors[i:i+2] = (a, b) # in-place update
        else:
            # non-adjacent two-qubit gate
            # step 1: swap i with i+1
            _mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
            # step 2: apply gate to (i+1, j) pair. This amounts to a recursive swap until the two qubits are adjacent
            _apply_gate(mps_tensors, gate, (i+1, j), algorithm=algorithm, options=options) 
            # step 3: swap back i and i+1
            _mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
    else:
        raise NotImplementedError("Only one- and two-qubit gates supported")
    return mps_tensors

def _get_initial_mps(num_qubits, dtype='complex128'):
    """
    Generate the MPS with an initial state of |00...00> 
    """
    state_tensor = cp.asarray([1, 0], dtype=dtype).reshape(1,2,1)
    mps_tensors = [state_tensor] * num_qubits
    return mps_tensors

def contract_mps_tensors_with_circuit(circuit, mps_tensors, algorithm=DEFAULT_CU_ALGORITHM):
    converter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)
    gates = converter.gates
    gate_map = dict(zip(converter.qubits, range(circuit.num_qubits)))
    handle = cutn.create()
    for (gate, qubits) in gates:
        qubits = [gate_map[q] for q in qubits]
        _apply_gate(mps_tensors, gate, qubits, algorithm=algorithm, options={'handle': handle})
    cutn.destroy(handle)

def cu_mps_to_aer_mps(mps_tensors):
    return [cp.asnumpy(cp.swapaxes(e, 0, 1)) for e in mps_tensors]

def mps_from_circuit_and_starting_mps(circuit, starting_mps, algorithm=DEFAULT_CU_ALGORITHM):
    # Copy the starting mps in case it's not intended to be modified in place
    mps_tensors = copy(starting_mps)
    contract_mps_tensors_with_circuit(circuit, mps_tensors, algorithm)
    return mps_tensors


def mps_from_circuit(circuit, algorithm=DEFAULT_CU_ALGORITHM):
    # Start with initial MPS |000>
    mps_tensors = _get_initial_mps(circuit.num_qubits, dtype='complex128')
    contract_mps_tensors_with_circuit(circuit, mps_tensors, algorithm)
    # Return in the form to be used later by CuQuantum
    return mps_tensors


