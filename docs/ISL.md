This document has been adapted from Chapter 2 of:

> [Solving optimisation problems on near-term quantum computers](https://ora.ox.ac.uk/objects/uuid:26ac5fa7-b323-426c-9d5a-423e16992a78) \
> 2022, (Doctoral dissertation, University of Oxford) \
> Ben Jaderberg

## Motivation

Given a computation which we wish to run on a quantum computer, it is vital that we implement it in
as few quantum logic gates as possible. Suppose that our quantum algorithm requires the action of a
unitary operation $\hat{U}$ on the qubits, which can be implemented through a decomposition into $m$
logical gate operations. Quantum (circuit) compilation is the process of trying to improve an
existing circuit by replacing it with a circuit that generates the same unitary but takes less time
to run. Generally, without knowledge of specific gate times, quantum compiling can be reformulated
as trying to find a circuit which implements $\hat{U}$ in $o < m$ gate operations. Solutions to this
range from duplicate gate cancellation to two-qubit block re-synthesis involving the KAK
decomposition. Significantly, since the fidelity of the circuit evaluated on noisy quantum hardware
decreases exponentially with the number of gates, the compiled implementation with fewer gates
generates less noise. As such, the process of quantum compiling can also be viewed as an error
mitigation technique.

Despite the success of quantum compiling methods, there remains key limitations in its applicability
to quantum computing in the near term. Notably, the unitaries that we would like to implement are
often too deep to be evaluated on current hardware, even in their theoretically minimal form after
compiling. This motivates another consideration. Suppose that the evaluation of the minimal
implementation of $\hat{U}$ on a noisy quantum computer produces an errored final state
$|{\tilde{\psi}_f}\rangle$ with an overlap to the true final state $|{\psi_f}\rangle$ of
$|\langle{\tilde{\psi}_f}|{\psi_f}\rangle| = 0.6$. In this case, is it better to instead evaluate an
alternative circuit, which in the noiseless case produces the state $|{\phi_f}\rangle$ which is only
approximately correct such that $|\langle{\phi_f}|{\psi_f}\rangle| < 1$, yet is implemented in so
few gates that its evaluation on noisy hardware produces a state $|{\tilde{\phi}_f}\rangle$ with
higher overlap to the true solution $|\langle{\tilde{\phi}_f}|{\psi_f}\rangle| > 0.6$? In other
words, when our computation is in the presence of heavy noise, do we really need to evaluate the
exact decomposition of $\hat{U}$?

This is the question posed by approximate quantum compiling (AQC). Here, rather than find an exact
alternative implementation of $\hat{U}$, the goal is to find a shallower quantum circuit $\hat{V}$
which has approximately the same action on some initial state $\hat{U}|\psi\rangle \approx
\hat{V}|\psi\rangle$. Reformulating this as $\langle\psi|\hat{V}^\dagger \hat{U}|\psi\rangle \approx
1$, we notice that the problem can be viewed as finding the set of gates $\hat{V^\dagger}$ that
inverts the action of $\hat{U}$ as measured by the overlap between the initial and final states, to
within some desired accuracy. Once the inverse is found, each gate that makes up $\hat{V^\dagger}$
can be inverted individually to produce our desired approximately equivalent circuit $\hat{V}$. This
process of trying to find the inverse can itself be recognised as an energy minimisation problem,
leading to the popular approach to solve it as a variational quantum algorithm. However, like many
variational algorithms, the best solution often requires finding the optimal ansatz through a
potentially lengthy trial-and-error process. Furthermore, what constitutes the best ansatz may vary
dramatically for different inputs, such that training multiple ansatzes for every circuit to be
compiled dramatically increases the complexity of general AQC. Attempts to use a single
problem-agnostic ansatz for multiple inputs would require the ansatz to have a random structure,
leading to trainability issues imposed by barren plateaus. These limitations combined are what we
consider to be the ansatz problem for AQC.

The idea behind incremental structural learning (ISL) begins with the consideration that for all
quantum algorithms, the convention is for the qubits to begin in the initial state $\ket{\psi_{0}} =
\ket{0}^{\otimes n}$. Here the problem can then be expressed as finding the set of gates that maps
the original circuit back to $\ket{\psi_{0}}$. We reiterate here that the goal of AQC is to find a
compiled solution for only one initial quantum state, satisfying $\hat{U}|\psi_{0}\rangle \approx
\hat{V}|\psi_{0}\rangle$ and not the more general condition $\hat{U} \approx \hat{V}$. One notable
feature of our choice of initial state, and thus target final state of compiling, is a product
state, with no pairwise entanglement. Thus instead of using a random ansatz for $\hat{V^\dagger}$,
our solution should be physically inspired, specifically, one that works to disentangle the state
$\hat{U}\ket{\psi_{0}}$. This represents the structural component of the ISL algorithm.

The second novel component of ISL is the incremental construction of the ansatz. Instead of using an
ansatz with a fixed layout, we build its structure layer-by-layer, optimising and evaluating the
cost function for each layer added. This is also referred to as an adaptive ansatz, which has proven
very successful when applied to variationally finding ground states in ADAPT-VQE. This approach not
only offers a greater range of potential solutions, but also reduces the likelihood of a redundantly
deep ansatz wasting resources or an inadequately shallow ansatz failing to find the optimal
solution.

## ISL Routine

### Adding a layer

We describe the process of ISL through the routine of adding a single layer. Suppose that after the
application of $n-1$ layers, we have our current best guess of the inverse $\hat{V}^{\dagger} =
\hat{V}^{\dagger}\_{n-1}...\hat{V}^{\dagger}_{1}$ and evaluate the cost function. Here the cost is
specifically

$$C = 1 - \left|\bra{\psi_{0}}\hat{V}^{\dagger}\hat{U}\ket{\psi_{0}}\right|^2$$

If the cost is greater than the required threshold $C > C_{t}$, we continue the compilation and add
the $n^{\mathrm{th}}$ layer. In ISL, the structure of every layer is a thinly-dressed CNOT gate.
This choice of two-qubit block is based on the regular dressed CNOT gate, used successfully in
previous works as an ansatz layer specifically for the problem of variational quantum compiling.
Here we describe our layer as _thinly_ dressed because the single-qubit gates are restricted to a
parameterised rotation along one axis, as opposed to the combination of three parameterised
rotations along all three axes in the original scheme. Interestingly, one layer of the (thinly)
dressed CNOT gate does not contain the 15 independent parameters required to span the special
unitary group SU(4) which contains all two-qubit interactions. However, this is a feature shared
amongst the overwhelming majority of ansatzes used in VQAs and has not been shown to hinder
practical performance. Moreover, the non-universality of our ansatz layer has the pragmatic benefit
of requiring fewer gates to implement, making it more efficient to implement on real quantum
computers.

Since each ansatz layer acts only on two qubits, if our compilation target $\hat{U}$ spans more than
two qubits we must first decide which qubits the thinly dressed CNOT gate should apply across. To
solve this, first we evaluate an entanglement measure $E$ between each pair of qubits, such as
entanglement of formation, concurrence or negativity. This can be calculated from a state vector /
tensor network when using emulated quantum hardware or directly on the quantum circuit. Since our
goal is to reduce the entanglement of the system to zero, we choose our layer to apply to the qubit
pair with the largest $E$.

It is also possible that all qubit pairs have $E=0$. For example, the maximally entangled state
$\ket{GHZ} = \frac{1}{\sqrt2}(\ket{000} + \ket{111})$ or the non-superposition state $\ket{\psi} =
\ket{011}$ does not have any pairwise local entanglement and will result in $E=0$ for all qubit
pairs. This typically happens towards the end of compilation rather than as the initial output of
the target circuit, after several disentangling ansatz layers have already been applied. In this
case, pairwise entanglement is not a good metric to base which qubits to apply the next layer on.
Instead, recall that ISL converges when a set of gates are found which inverts the target circuit to
produce the $\ket{0}^{\otimes n}$ state. Thus, we can impose that convergence requires both $E=0$
and that the single-qubit expectation value of each qubit in the $\hat{\sigma}\_{z}$ basis equals 1,
since $\bra{0}\hat{\sigma}_{z}\ket{0} = 1$. Practically then, when faced with a scenario where all
qubit pairs have $E=0$, the ISL algorithm measures each qubit and then applies a thinly-dressed CNOT
layer to the two qubits with the lowest and second lowest expectation values, since these are the
furthest from the solution. In the example of $\ket{\psi} = \ket{q_3\:q_2\:q_1} = \ket{011}$, a
single layer would be added to the first and second qubits, which would be optimised to produce the
net effect of a bit flip on both.

One constraint that we impose on the choice of the control and target is that it must not be the
same as the control and target for the previous layer. This is because in general, adding layers to
different choices of control and target qubits allows us to explore a greater region of the
available Hilbert space. This also avoids creating circuits with large depth but small numbers of
gates. Hence, if the qubit pair with the highest $E$ is the same as in the previous layer, we choose
different qubits with next largest value of $E$ instead, working our way through the remaining
possible pairs.

Once we have chosen the control and target qubits, we add the thinly-dressed CNOT layer
$\hat{V}^{\dagger}_{n}$ with initial rotations $\theta=0$ about the Z axis.

### Optimising the layer

After the layer $\hat{V}^{\dagger}_{n}$ is added, the axes and angles of rotation of the
single-qubit gates are optimised using the Rotoselect
algorithm (https://quantum-journal.org/papers/q-2021-01-28-391/), with respect to minimising the
cost. This procedure works by fixing three of the gates and varying the rotation axes and angle for
the remaining one. Importantly, adjusting the angle of rotation for the single unfixed gate will
cause a sinusoidal change in the cost function, displaying a clear minima. This is then repeated
over the remaining 3 gates, concluding the first cycle of Rotoselect. As the changing of later gates
in the layer will affect the loss landscape of the first gates, we repeatedly cycle over all 4
rotation gates until a termination criteria is reached. Here we define the termination criteria to
be when the reductions in the cost function between cycles is less than a user defined amount, or
when an upper-limit of 1000 cycles is reached.

Once the single-qubit gates of this particular layer have been optimised, we can then choose to
optimise the larger blocks of the ansatz $\hat{V^{\dagger}}$ using the Rotosolve
algorithm (https://quantum-journal.org/papers/q-2021-01-28-391/). This is a similar procedure to
Rotoselect, but optimises only the rotation gate angles rather than the bases. How often this
Rotosolve is called and the number of blocks to optimise is set by the user.

### Finishing the layer

Once the layer is optimised, we perform simple non-approximate compilation of the new best guess
$\hat{V^{\dagger}}$. This includes e.g., merging of adjacent rotation gates in the same basis,
removing two-qubit gate blocks that form an identity.

After this we take one final evaluation of the cost function. If the cost is now below the threshold
$C < C_t$, the ISL algorithm is terminated and we recursively invert all of the gates in the ansatz
to return the compiled solution $\hat{V}$. If not, then the next layer is added and the steps
outlined above are repeated.