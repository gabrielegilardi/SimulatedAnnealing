/*
Metaheuristic Optimization Using Population-Based Simulated Annealing.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written in C++ using the Eigen library (ver. 3.3.9) and
  tested using g++ 8.1.0 (MinGW-W64).
- Variables can be real, integer, or mixed real/integer.
- Variables can be constrained to a specific interval or value setting the
  lower and the upper boundaries.
- Neighboroud search is performed using a normal distribution along randomly
  chosen dimensions.
- A nonlinear decreasing schedule is used for the temperature and for the
  standard deviation in the neighboroud search.
- Search space can be normalized to improve convergency.
- An arbitrary number of parameters can be passed (in a tuple) to the function
  to minimize.
- Solver parameters and results are passed using structures.
- Usage: test_Eigen.exe <example>.

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
    If all variables are real set <IntVar> = <NULL>. Indexes are specified
    in the range 0 to <nVar-1>. It cannot be used when the search space is
    normalized.
normalize = true, false
    Specifies if the search space should be normalized. If <true>, parameter
    <sigma0> is applied to the normalized search space. 
args
    Tuple containing any parameter that needs to be passed to the function. If
    no parameters are passed set <args> = <NULL>.
nVar
    Number of variables.
nIntVar >= 0
    Number of integer variables.
X0
    Global minimum point (used only to compare with the numerical solution).
seed
    Seeding value for the random number generator.

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
- Eigen template library for linear algebra @ https://eigen.tuxfamily.org/
*/

#include <random>
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

/* Structure used to pass the parameters (with default values) */
struct Parameters {
    int nPop = 20;
    int epochs = 1000;
    int nMove = 100;
    double T0 = 0.1;
    double alphaT = 0.99;
    double sigma0 = 0.1;
    double alphaS = 0.98;
    double prob = 0.5;
    bool normalize = false;
    ArrayXi IntVar;
    ArrayXXd args;
    int seed = 1234567890;
};

/* Structure used to return the results */
struct Results {
    double best_cost;
    ArrayXXd best_pos;
    ArrayXd F;
    double T;
    ArrayXXd sigma;
};


/* Simulated annealing function prototype */
Results sa(ArrayXd (*func)(ArrayXXd, ArrayXXd), ArrayXXd LB, ArrayXXd UB,
           Parameters p);


// Parabola: F(X) = sum((X - X0)^2)
// Xmin = X0
ArrayXd Parabola(ArrayXXd X, ArrayXXd args)
{
    int nPop;
    ArrayXd f;
    ArrayXXd dX;

    nPop = X.rows();
    dX = X - args.replicate(nPop, 1);
    f = (dX * dX).rowwise().sum();

    return f;
}


// Ackley: F(X)= + 20 + exp(1) - exp(sum(cos(2*pi*(X-X0))/n)
//               - 20*exp(-0.2*sqrt(sum((X-X0)^2)/n))
// Xmin = X0
ArrayXd Ackley(ArrayXXd X, ArrayXXd args)
{
    const double pi = 3.14159265358979323846;
    int nPop, nVar;
    ArrayXd f;
    ArrayXXd dX;

    nPop = X.rows();
    nVar = X.cols();
    dX = X - args.replicate(nPop, 1);
    f = + 20.0 + exp(1.0)
        - exp((cos(2.0 * pi * dX)).rowwise().sum() / nVar)
        - 20.0 * exp(-0.2 * sqrt((dX * dX).rowwise().sum() / nVar));

    return f;
}


// Tripod:
// F(x,y)= p(y)*(1 + p(x)) + abs(x + kx*p(y)*(1 - 2*p(x)))
//         + abs(y + ky*(1 - 2*p(y)))
// p(x) = 1 if x >= 0, p(x) = 0 if x < 0; p(y) = 1 if y >= 0, p(y) = 0 if y < 0
// Global minimum at [0,-ky], local minimum at [-kx,ky] and [kx,ky]; kx, ky > 0
ArrayXd Tripod(ArrayXXd X, ArrayXXd args)
{
    double kx = args(0, 0), ky = args(0, 1);
    ArrayXd f, x, y, px, py;

    x = X.col(0);
    y = X.col(1);
    px = (x >= 0.0).cast<double>();
    py = (y >= 0.0).cast<double>();

    f = py * (1.0 + px) + abs(x + kx * py * (1.0 - 2.0 * px)) +
        abs(y + ky * (1.0 - 2.0 * py));

    return f;
}


// Alpine: F(X) = sum(abs(X*sin(X) + 0.1*X))
// Xmin = 0
ArrayXd Alpine(ArrayXXd X, ArrayXXd args)
{
    ArrayXd f;

    f = (abs(X * sin(X) + 0.1 * X)).rowwise().sum();

    return f;
}


/* Main function */
int main(int argc, char **argv) 
{
    Parameters p;
    Results res;
    string example;
    int nVar;
    double sum, err;

    /*Eigen declarations*/
    ArrayXXd UB, LB, X0;
    ArrayXd (*func)(ArrayXXd, ArrayXXd);

    /* Read example to run */
    if (argc != 2) {
        printf("\nUsage: test_Eigen <example>\n");
        exit(EXIT_FAILURE);
    }
    example = argv[1];

    // Parabola: F(X) = sum((X - X0)^2)
    // Xmin = X0
    if (example == "Parabola") {
        func = Parabola;
        nVar = 20;

        UB.setConstant(1, nVar, 500.0);     // Upper and lower boundaries
        LB = -UB;

        X0.setZero(1, nVar);                // Global minimum
        for (int i=0; i<nVar; i++) {
            X0(0, i) = 1.1 * double(i);
        }

        p.args = X0;                        // Arguments
    }

    // Ackley: F(X)= + 20 + exp(1) - exp(sum(cos(2*pi*(X-X0))/n)
    //               - 20*exp(-0.2*sqrt(sum((X-X0)^2)/n))
    // Xmin = X0
    else if (example == "Ackley") {
        func = Ackley;
        nVar = 10;
        p.nPop = 50;

        UB.setConstant(1, nVar, 50.0);      // Upper and lower boundaries
        LB = -UB;

        X0.setConstant(1, nVar, 1.6789);    // Global minimum

        p.args = X0;                        // Arguments
    }

    // Tripod:
    // F(x,y)= p(y)*(1 + p(x)) + abs(x + kx*p(y)*(1 - 2*p(x)))
    //         + abs(y + ky*(1 - 2*p(y)))
    // p(x) = 1 if x >= 0, p(x) = 0 if x < 0; p(y) = 1 if y >= 0, p(y) = 0 if y < 0
    // Global minimum at [0,-ky], local minimum at [-kx,ky] and [kx,ky]; kx, ky > 0
    else if (example == "Tripod") {
        func = Tripod;
        nVar = 2;               // The equation works only with two dimensions
        double kx = 20.0;
        double ky = 40.0;

        UB.setConstant(1, nVar, 100.0);     // Upper and lower boundaries
        LB = -UB;

        X0.setZero(1, nVar);                // Global minimum
        X0(0, 1) = -ky;

        p.args.setZero(1, nVar);            // Arguments
        p.args(0, 0) = kx;
        p.args(0, 1) = ky;
    }

    // Alpine: F(X) = sum(abs(X*sin(X) + 0.1*X))
    // Xmin = 0
    // Note: the solution is VERY sensitive to the parameter values and the
    //       random generated numbers
    else if (example == "Alpine") {
        func = Alpine;
        nVar = 10;
        p.sigma0 = 0.2;
        p.alphaS = 1.0;

        UB.setConstant(1, nVar, 10.0);      // Upper and lower boundaries
        LB = -UB;

        X0.setZero(1, nVar);                // Global minimum
    }

    else {
        printf("\nFunction not found.\n");
        exit(EXIT_FAILURE);
    }

    /* Solve */
    res = sa(func, LB, UB, p);

    /* Print results */
    printf("\nBest position:");
    for (int j=0; j<nVar; j++) {
        printf("\n %g", res.best_pos(0, j));
    }
    printf("\n\nCost: %g", res.best_cost);
    printf("\nFinal T: %g", res.T);
    sum = res.sigma.sum();
    printf("\nFinal sigma (avr): %g", sum / double(nVar));
    err = ((res.best_pos - X0) * (res.best_pos - X0)).sum();
    printf("\nError: %g\n", sqrt(err));

    return 0;
}
