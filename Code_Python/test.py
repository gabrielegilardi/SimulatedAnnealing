"""
Metaheuristic Optimization Using Population-Based Simulated Annealing.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written and tested in Python 3.8.8.
- Variables can be real, integer, or mixed real/integer.
- Variables can be constrained to a specific interval or value setting the
  lower and the upper boundaries.
- Neighboroud search is performed using a normal distribution along randomly
  chosen dimensions.
- A nonlinear decreasing schedule is used for the temperature and for the
  standard deviation in the neighboroud search.
- Search space can be normalized to improve convergency.
- To improve the execution speed the algorithm has been designed removing
  almost all loops on the agents.
- An arbitrary number of parameters can be passed (in a tuple) to the function
  to minimize.
- Usage: python test.py <example>.

Main Parameters
---------------
example
    Name of the example to run (Parabola, Alpine, Tripod, and Ackley.)
func
    Function to minimize. The position of all agents is passed to the function
    at the same time.
LB, UB
    Lower and upper boundaries of the search space.
nPop >=1, epochs >= 1
    Number of agents (population) and number of iterations.
T0 > 0
    Initial temperature.
0 < alphaT < 1
    Temperature reduction rate.
0 < alphaS <= 1
    Standard deviation (neighboroud search) reduction rate.
nMove > 0
    Number of neighbours of a state evaluated at each epoch.
0 < prob < 1
    Probability the dimension of a state is changed.
sigma0 > 0
    Initial standard deviation used to search the neighboroud of a state.
IntVar
    List of indexes specifying which variable should be treated as integer.
    If all variables are real set <IntVar> = <None>. Indexes are specified
    in the range 1 to <nVar>.
normalize = True, False
    Specifies if the search space should be normalized. If <True>, parameter
    <sigma0> is applied to the normalized search space.
args
    Tuple containing any parameter that needs to be passed to the function. If
    no parameters are passed set <args> = <None>.
nVar
    Number of variables.
Xsol
    Solution to the minimization point. Set <Xsol> = <None> if not known.

Examples
--------
There are four examples: Parabola, Alpine, Tripod, and Ackley.

- Parabola, Alpine, and Ackley can have an arbitrary number of dimensions,
  while Tripod has only two dimensions.

- Parabola, Tripod, and Ackley are examples where parameters (respectively,
  array X0, scalars kx and ky, and array X0) are passed using args.

- The global minimum for Parabola and Ackley is at X0; the global minimum for
  Alpine is at zero; the global minimum for Tripod is at [0,-ky] with local
  minimum at [-kx,+ky] and [+kx,+ky].

References
----------
- Simulated annealing @ https://en.wikipedia.org/wiki/Simulated_annealing
- Kirkpatrick et al., 1983, "Optimization by Simulated Annealing", JSTOR
  @ https://www.jstor.org/stable/1690046
- Jamil and Yang, 2013, "A Literature Survey of Benchmark Functions For Global
  Optimization Problems", arXiv @ https://arxiv.org/abs/1308.4008
"""

import sys
import numpy as np
from sa import SA

# Read example to run
if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

# Parabola: F(X) = sum((X - X0)^2), Xmin = X0

if (example == 'Parabola'):

    def Parabola(X, args):
        X0 = args
        F = ((X - X0) ** 2).sum(axis=1)
        return F

    # Problem parameters
    nVar = 20
    X0 = np.ones(nVar) * 1.2345        # Args

    # SA parameters
    func = Parabola
    UB = np.ones(nVar) * 20.0
    LB = - UB
    nPop = 20
    epochs = 1000
    T0 = 0.1
    alphaT = 0.99
    alphaS = 0.98
    nMove = 100
    prob = 0.2
    sigma0 = 0.1
    IntVar = None
    normalize = False
    args = (X0)

    # Solution
    Xsol = X0

# Ackley: F(X)= 20 + exp(1) - exp(sum(cos(2*pi*X))/n)
#               - 20*exp(-0.2*sqrt(sum(X^2)/n))
# Xmin = X0
elif (example == 'Ackley'):

    def Ackley(X, args):
        X0 = args
        n = float(X.shape[1])
        F = 20.0 + np.exp(1.0) \
            - np.exp((np.cos(2.0 * np.pi * (X - X0))).sum(axis=1) / n) \
            - 20.0 * np.exp(-0.2 * np.sqrt(((X - X0) ** 2).sum(axis=1) / n))
        return F

    # Function parameters
    nVar = 10
    X0 = np.ones(nVar) * 1.6            # args

    # SA parameters
    func = Ackley
    UB = np.ones(nVar) * 30.0
    LB = - UB
    nPop = 50
    epochs = 1000
    T0 = 0.1
    alphaT = 0.99
    alphaS = 0.99
    nMove = 100
    prob = 0.3
    sigma0 = 0.1
    IntVar = None
    normalize = False
    args = (X0)

    # Solution
    Xsol = X0

# Tripod:
# F(x,y)= p(y)*(1 + p(x)) + abs(x + kx*p(y)*(1 - 2*p(x)))
#         + abs(y + ky*(1 - 2*p(y)))
# p(x) = 1 if x >= 0, p(x) = 0 if x < 0; p(y) = 1 if y >= 0, p(y) = 0 if y < 0
# Global minimum at [0,-ky], local minimum at [-kx,ky] and [kx,ky]; kx, ky > 0
elif (example == 'Tripod'):

    def Tripod(X, args):
        x = X[:, 0]
        y = X[:, 1]
        kx = args[0]
        ky = args[1]
        px = (x >= 0.0)
        py = (y >= 0.0)
        F = py * (1.0 + px) + np.abs(x + kx * py * (1.0 - 2.0 * px)) \
            + np.abs(y + ky * (1.0 - 2.0 * py))
        return F

    # Function parameters
    nVar = 2                # The equation works only with two dimensions
    kx = 20.0               # Args
    ky = 40.0

    # SA parameters
    func = Tripod
    UB = np.ones(nVar) * 100.0
    LB = - UB
    nPop = 10
    epochs = 1000
    T0 = 0.1
    alphaT = 0.99
    alphaS = 0.98
    nMove = 50
    prob = 0.5
    sigma0 = 0.1
    IntVar = None
    normalize = False
    args = (kx, ky)

    # Solution
    Xsol = np.array([0.0, -ky])

# Alpine: F(X) = sum(abs(X*sin(X) + 0.1*X)), Xmin = 0
elif (example == 'Alpine'):

    def Alpine(X, args):
        F = np.abs(X * np.sin(X) + 0.1 * X).sum(axis=1)
        return F

    # Function parameters
    nVar = 10

    # SA parameters
    func = Alpine
    UB = np.ones(nVar) * 10.0
    LB = - UB
    nPop = 20
    epochs = 500
    T0 = 1.0
    alphaT = 0.99
    alphaS = 1.0
    nMove = 50
    prob = 0.3
    sigma0 = 0.2
    IntVar = None
    normalize = False
    args = None

    # Solution
    Xsol = np.zeros(nVar)

else:
    print("Function not found")
    sys.exit(1)

np.random.seed(1294404794)          # Seed random generator

# Solve
X, info = SA(func, LB, UB, nPop=nPop, epochs=epochs, nMove=nMove, T0=T0,
             alphaT=alphaT, sigma0=sigma0, alphaS=alphaS, prob=prob,
             IntVar=IntVar, normalize=normalize, args=args)
F, T_final, sigma_final = info

# Results
print("Function: ", example)
print("\nBest position:")
print(X)
print("\nCost: ", F[-1])
print("Final T: ", T_final)
print("Final sigma (avr): ", sigma_final.mean())
if (Xsol is not None):
    error = np.linalg.norm(Xsol - X)
    print("Error: {0:e}".format(error))
