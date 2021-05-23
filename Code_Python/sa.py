"""
Metaheuristic Minimization Using Population-Based Simulated Annealing.

Copyright (c) 2021 Gabriele Gilardi

"""

import numpy as np


def SA(func, LB, UB, nPop=10, epochs=100, nMove=20, T0=0.1, alphaT=0.99,
       sigma0=0.1, alphaS=1.0, prob=0.5, IntVar=None, normalize=False,
       args=None):
    """
    func            Function to minimize
    LB              Lower boundaries of the search space
    UB              Upper boundaries of the search space
    nPop            Number of agents (population)
    epochs          Number of iterations
    nMove           Number of neighbours of a state evaluated at each epoch
    T0              Initial temperature
    alphaT          Temperature reduction rate
    sigma0          Initial standard deviation used to search the neighboroud
                    of a state (given as a fraction of the search space)
    alphaS          Sigma (std) reduction rate
    prob            Probability the dimension of a state is changed
    IntVar          List of indexes specifying which variable should be treated
                    as integer
    normalize       Specifies if the search space should be normalized (to
                    improve convergency)
    args            Tuple containing any parameter that needs to be passed to
                    the function

    Dimensions:
    (nVar, )        LB, UB, LB_orig, UB_orig, sigma
    (nPop, nVar)    agent_pos, neigh_pos, agent_pos_orig, neigh_pos_orig, flips
    (nPop)          agent_cost, neigh_cost
    (0-nVar, )      IntVar
    """
    nVar = len(LB)

    F = np.zeros(epochs)
    neigh_pos = np.zeros((nPop, nVar))      # Neighbour state
    neigh_cost = np.zeros(nPop)             # Neighbour cost

    # Normalize search space
    if (normalize):
        LB_orig = LB.copy()
        UB_orig = UB.copy()
        LB = np.zeros(nVar)
        UB = np.ones(nVar)

    T = T0                          # Temperature
    sigma = sigma0 * (UB - LB)      # Standard deviation of each dimension

    # Define (if any) which variables are treated as integers (indexes are in
    # the range 1 to nVar)
    if (IntVar is None):
        nIntVar = 0
    else:
        IntVar = np.asarray(IntVar, dtype=int) - 1
        nIntVar = len(IntVar)

    # Initial position of each agent
    agent_pos = LB + np.random.rand(nPop, nVar) * (UB - LB)

    # Correct for any integer variable
    if (nIntVar > 0):
        agent_pos[:, IntVar] = np.round(agent_pos[:, IntVar])

    # Initial cost of each agent
    if (normalize):
        agent_pos_orig = LB_orig + agent_pos * (UB_orig - LB_orig)
        agent_cost = func(agent_pos_orig, args)
    else:
        agent_cost = func(agent_pos, args)

    # Initial (overall) best position/cost
    idx = np.argmin(agent_cost)
    best_pos = agent_pos[idx, :]
    best_cost = agent_cost[idx]

    # Main loop (temperature is fixed)
    for epoch in range(epochs):

        # Sub-loop (search the neighboroud of a state)
        for move in range(nMove):

            # Randomly decide in which dimension to search
            flips = (np.random.rand(nPop, nVar) <= prob)

            # Create each agent's neighbour
            neigh_pos = agent_pos + np.random.randn(nPop, nVar) * sigma
            neigh_pos = np.where(flips == 1, neigh_pos, agent_pos)

            # Correct for any integer variable
            if (nIntVar > 0):
                neigh_pos[:, IntVar] = np.round(neigh_pos[:, IntVar])

            # Impose position boundaries
            neigh_pos = np.fmin(np.fmax(neigh_pos, LB), UB)
            if (nIntVar > 0):
                neigh_pos[:, IntVar] = np.fmax(neigh_pos[:, IntVar],
                                               np.ceil(LB[IntVar]))
                neigh_pos[:, IntVar] = np.fmin(neigh_pos[:, IntVar],
                                               np.floor(UB[IntVar]))

            # Evaluate the cost of each agent's neighbour
            if (normalize):
                neigh_pos_orig = LB_orig + neigh_pos * (UB_orig - LB_orig)
                neigh_cost = func(neigh_pos_orig, args)
            else:
                neigh_cost = func(neigh_pos, args)

            # Decide if each agent will change its state
            for i in range(nPop):

                # Swap states if the neighbour state is better ...
                if (neigh_cost[i] <= agent_cost[i]):
                    agent_pos[i, :] = neigh_pos[i, :]
                    agent_cost[i] = neigh_cost[i]

                # ... or decide probabilistically
                else:

                    # Acceptance probability
                    delta = (neigh_cost[i] - agent_cost[i]) / agent_cost[i]
                    p = np.exp(- delta / T)

                    # Randomly swap states
                    if (np.random.rand() <= p):
                        agent_pos[i, :] = neigh_pos[i, :]
                        agent_cost[i] = neigh_cost[i]

            # Update the (overall) best position/cost
            idx = np.argmin(agent_cost)
            if (agent_cost[idx] < best_cost):
                best_cost = agent_cost[idx]
                best_pos = agent_pos[idx, :]

        # Save the (min) function value for this epoch
        F[epoch] = best_cost

        # Cooling scheduling
        T = alphaT * T

        # Random neighboroud search schedule
        sigma = alphaS * sigma

    # De-normalize
    if (normalize):
        best_pos = LB_orig + best_pos * (UB_orig - LB_orig)
        sigma = sigma * (UB_orig - LB_orig)

    # Return info about the solution
    info = (F, T, sigma)

    return best_pos, info
