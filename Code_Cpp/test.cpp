/*
Metaheuristic Optimization Using Population-Based Simulated Annealing.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written in plain vanilla C++ and tested using g++ 8.1.0
  (MinGW-W64).
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
- Usage: test.exe <example>.

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
nIntVar >= 0
    Number of integer variables.
normalize = true, false
    Specifies if the search space should be normalized. If <true>, parameter
    <sigma0> is applied to the normalized search space. 
args
    Tuple containing any parameter that needs to be passed to the function. If
    no parameters are passed set <args> = <NULL>.
nVar
    Number of variables.
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
*/

#include <random>

using namespace std;

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
    int *IntVar = NULL;
    int nIntVar = 0;
    double *args = NULL;
    int seed = 1234567890;
};

/* Structure used to return the results */
struct Results {
    double best_cost;
    double *best_pos;
    double *F;
    double T;
    double *sigma;
};

/* Simulated annealing function prototype */
Results sa(double (*func)(double[], int, double[]), double LB[], double UB[],
           int nVar, Parameters p);

// Parabola: F(X) = sum((X - X0)^2)
// Xmin = X0
double Parabola(double X[], int nVar, double args[])
{
    double f = 0.0, dX;

    for (int j=0; j<nVar; j++) {
        dX = X[j] - args[j];
        f += dX * dX;
    }

    return f;
}


// Ackley: F(X)= + 20 + exp(1) - exp(sum(cos(2*pi*(X-X0))/n)
//               - 20*exp(-0.2*sqrt(sum((X-X0)^2)/n))
// Xmin = X0
double Ackley(double X[], int nVar, double args[])
{
    double f, dX, sum1 = 0.0, sum2 = 0.0;
    const double pi = 3.14159265358979323846;
    
    for (int j=0; j<nVar; j++) {
        dX = X[j] - args[j];
        sum1 += cos(2.0 * pi * dX);
        sum2 += dX * dX;
    }

    f = 20.0 + exp(1.0) - exp(sum1 / double(nVar)) -
        20.0 * exp(-0.2 * sqrt(sum2 / double(nVar)));

    return f;
}


// Tripod:
// F(x,y)= p(y)*(1 + p(x)) + abs(x + kx*p(y)*(1 - 2*p(x)))
//         + abs(y + ky*(1 - 2*p(y)))
// p(x) = 1 if x >= 0, p(x) = 0 if x < 0; p(y) = 1 if y >= 0, p(y) = 0 if y < 0
// Global minimum at [0,-ky], local minimum at [-kx,ky] and [kx,ky]; kx, ky > 0
double Tripod(double X[], int nVar, double args[])
{
    double x = X[0], y = X[1], kx = args[0], ky = args[1], px, py, f;

    px = (x >= 0.0) ? 1.0 : 0.0;
    px = (y >= 0.0) ? 1.0 : 0.0;

    f = py * (1.0 + px) + abs(x + kx * py * (1.0 - 2.0 * px)) +
        abs(y + ky * (1.0 - 2.0 * py));

    return f;
}


// Alpine: F(X) = sum(abs(X*sin(X) + 0.1*X))
// Xmin = 0
double Alpine(double X[], int nVar, double args[])
{
    double f = 0.0;

    for (int j=0; j<nVar; j++) {
        f += abs(X[j] * sin(X[j]) + 0.1 * X[j]);
    }

    return f;
}


/* Main function */
int main(int argc, char **argv) 
{
    Parameters p;
    Results res;
    string example;
    int nVar;
    double sum = 0.0, err = 0.0;
    double *args, *UB, *LB, *X0;
    double (*func)(double[], int, double[]);

    /* Read example to run */
    if (argc != 2) {
        printf("\nUsage: sa <example>\n");
        exit(EXIT_FAILURE);
    }
    example = argv[1];

    // Parabola: F(X) = sum((X - X0)^2)
    // Xmin = X0
    if (example == "Parabola") {
        func = Parabola;
        nVar = 20;

        UB = new double [nVar];             // Upper and lower boundaries
        LB = new double [nVar];
        for (int i=0; i<nVar; i++) {
            UB[i] = +500.0;
            LB[i] = -500.0;
        }

        X0 = new double [nVar];             // Global minimum
        for (int i=0; i<nVar; i++) {
            X0[i] = 1.1 * double(i);
        }

        p.args = new double [nVar];         // Arguments
        for (int i=0; i<nVar; i++) {
            p.args[i] = X0[i];
        }
    }

    // Ackley: F(X)= + 20 + exp(1) - exp(sum(cos(2*pi*(X-X0))/n)
    //               - 20*exp(-0.2*sqrt(sum((X-X0)^2)/n))
    // Xmin = X0
    else if (example == "Ackley") {
        func = Ackley;
        nVar = 10;
        p.nPop = 50;

        UB = new double [nVar];             // Upper and lower boundaries
        LB = new double [nVar];
        for (int i=0; i<nVar; i++) {
            UB[i] = +50.0;
            LB[i] = -50.0;
        }

        X0 = new double [nVar];             // Global minimum
        for (int i=0; i<nVar; i++) {
            X0[i] = 1.6789;
        }

        p.args = new double [nVar];         // Arguments
        for (int i=0; i<nVar; i++) {
            p.args[i] = X0[i];
        }
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

        UB = new double [nVar];             // Upper and lower boundaries
        LB = new double [nVar];
        for (int i=0; i<nVar; i++) {
            UB[i] = +100.0;
            LB[i] = -100.0;
        }

        X0 = new double [nVar];             // Global minimum
        X0[0] = 0.0;
        X0[1] = -ky;

        p.args = new double [nVar];         // Arguments
        p.args[0] = kx;
        p.args[1] = ky;
    }

    // Alpine: F(X) = sum(abs(X*sin(X) + 0.1*X))
    // Xmin = 0
    // Note: the solution is VERY sensitive to the parameter values and the
    //       random generated numbers
    else if (example == "Alpine") {
        func = Alpine;
        nVar = 10;
        p.sigma0 = 0.4;
        p.alphaS = 1.0;

        UB = new double [nVar];             // Upper and lower boundaries
        LB = new double [nVar];
        for (int i=0; i<nVar; i++) {
            UB[i] = +10.0;
            LB[i] = -10.0;
        }

        X0 = new double [nVar];             // Global minimum
        for (int i=0; i<nVar; i++) {
            X0[i] = 0.0;
        }
    }

    else {
        printf("\nFunction not found.\n");
        exit(EXIT_FAILURE);
    }

    /* Solve */
    res = sa(func, LB, UB, nVar, p);

    /* Print results */
    printf("\nBest position:");
    for (int j=0; j<nVar; j++) {
        printf("\n %g", res.best_pos[j]);
    }
    printf("\n\nCost: %g", res.best_cost);
    printf("\nFinal T: %g", res.T);
    for (int j=0; j<nVar; j++) {
        sum += res.sigma[j];
        err += (res.best_pos[j] - X0[j]) * (res.best_pos[j] - X0[j]);
    }
    printf("\nFinal sigma (avr): %g", sum / double(nVar));
    printf("\nError: %g\n", sqrt(err));

    return 0;
}
