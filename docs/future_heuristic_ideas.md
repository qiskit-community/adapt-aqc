This document is to detail some ideas for potential future features.

# Local minima

## random_cost_noise

**What is it:** \
Every time the cost function is evaluated, some noise value $\epsilon$ is added. The distribution
that $\epsilon$ is drawn from could be uniform, Gaussian or another and perhaps user defined. Likely
the magnitude of $\epsilon$ would need to be adjusted depending on the number of qubits.

**What problem it aims to solve:** \
Adding noise is thought to help avoid getting stuck in local minima. This is a technique applied
in DMRG (see https://itensor.org/support/2529/details-of-how-dmrg-works), and also explored e.g., in
deep learning (https://arxiv.org/abs/1511.06807).

## add_local_minima_to_cost

**What is it:** \
If we have identified that ISL is stuck in a local minima, we could save the MPS $|LM\rangle$ at
this point. Then, we could explicitly add to the cost function a new term that would be the overlap
of the trial solution to this MPS. So the cost would then be

$$C = 1 - |\langle 0 | V^\dagger U|0\rangle|^2 + |\langle LM| V^\dagger U|0\rangle|^2$$$

Since the cost is being minimised, this encourages ISL to minimise the overlap between the current
solution $V^\dagger U|0\rangle$ and the state that was in a local minima $|LM\rangle$.

**What problem it aims to solve:** \
This is another technique borrowed from DMRG, which we learnt about in a seminar (reference needed
for what this is called in DMRG literature). The idea is once we have identified a local minima, we
can repel the optimisation away from this area of the cost landscape.

# Performance

## optimiser == "gradient_descent"

**What is it:** \
Gradient descent is the most popular optimisation algorithm in classical and quantum ML alike. At
the moment, ISL does not use gradient descent and instead uses non-gradient based sequential
optimisation in the form of the Rotosolve/Rotoselect algorithms.

**What problem it aims to solve:** \
Gradient descent was originally not used due to fears over encountering barren plateaus. However,
when running ISL fully classically, barren plateaus should not be an issue as we can access
observables with exponential precision. The benefit of gradient descent would be a potentially
large performance improvement, as the gradients of each parameter can be calculated independently
of one another. Note, however, that this improvement would be mostly dependent on calculating
gradients using backpropagation (for simulated circuits), as opposed to parameter-shift.

## parallel_rotosolve

**What is it:** \
Given $P$ parameterised gates, Rotosolve cycles through them one-by-one finding the optimal angle
in the case that all others are fixed. Since the optimal gate angles are dependent on one another,
this cycle is repeated until the cost function converges (i.e., does not change by a defined amount
between two cycles). However, if we assume independence of the parameters, we could optimise each
gate in parallel. This could be done with no downside if the gates truly are independent (e.g., not
in a light-cone of each other) or as an approximation with the hope that the optimal angles in
parallel are not too far from those in sequence.

**What problem it aims to solve:** \
With enough computational threads, parallelising Rotosolve could lead to a performance
improvement.