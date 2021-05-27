/*
Metaheuristic Optimization Using Population-Based Simulated Annealing.

Copyright (c) 2021 Gabriele Gilardi


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
    alphaS          Standard deviation reduction rate
    prob            Probability the dimension of a state is changed
    IntVar          List of indexes specifying which variable should be treated
                    as integer
    normalize       Specifies if the search space should be normalized
    args            Tuple containing any parameter that needs to be passed to
                    the function

    Dimensions:
    (nVar)          LB, UB, LB_orig, UB_orig, sigma, best_pos
    (nPop, nVar)    agent_pos, neigh_pos, agent_pos_orig, neigh_pos_orig
    (nPop)          agent_cost, neigh_cost
    (epochs)        F
    (0-nVar)        IntVar
*/


#include <random>

using namespace std;

/* Structure used to pass the parameters */
struct Parameters {
    int nPop;
    int epochs;
    int nMove;
    double T0;
    double alphaT;
    double sigma0;
    double alphaS;
    double prob;
    bool normalize;
    int *IntVar;
    int nIntVar;
    double *args;
    int seed;
};

/* Structure used to return the results */
struct Results {
    double best_cost;
    double *best_pos;
    double *F;
    double T;
    double *sigma;
};

/* Returns the index corresponding to the minimum value of an array */
int fmin(double A[], int n)
{
    int idx = 0;
    double fmin = A[0];

    for (int i=1; i<n; i++) {
        if (A[i] < fmin) {
            idx = i;
            fmin = A[i];
        }
    }

    return idx;
}


/* Minimiza a function using simulated annealing */
Results sa(double (*func)(double[], int, double[]), double LB[], double UB[],
           int nVar, Parameters p)
{
    int idx;
    double T, prob_swap, delta, best_cost;
    double *agent_cost, *neigh_cost, *best_pos, *sigma, *F, *LB_orig, *UB_orig;
    double **agent_pos, **neigh_pos, **agent_pos_orig, **neigh_pos_orig;
    Results res;

    /* Random generator and probability distributions */
    mt19937 generator(p.seed);
    uniform_real_distribution<double> unif(0.0, 1.0);
    normal_distribution<double> norm(0.0, 1.0);

    /* Create arrays */
	F = new double [p.epochs];              // Best cost for each epoch
	sigma = new double [nVar];              // Standard deviation
	best_pos = new double [nVar];           // Best position

	agent_pos = new double *[p.nPop];       // Agent's & neighbour's position
	neigh_pos = new double *[p.nPop];
	for (int i=0; i<p.nPop; i++) {
		agent_pos[i] = new double[nVar];
		neigh_pos[i] = new double[nVar];
    }

	agent_cost = new double [p.nPop];       // Agent's & neighbour's cost
	neigh_cost = new double [p.nPop];

    if (p.normalize) {                      // Original quantities
        LB_orig = new double [nVar];
        UB_orig = new double [nVar];
    	agent_pos_orig = new double *[p.nPop];
        neigh_pos_orig = new double *[p.nPop];
        for (int i=0; i<p.nPop; i++) {
    		agent_pos_orig[i] = new double[nVar];
            neigh_pos_orig[i] = new double[nVar];
        }
    }

    /* Normalize search space */
    if (p.normalize) {
        for (int j=0; j<nVar; j++) {
            LB_orig[j] = LB[j];
            UB_orig[j] = UB[j];
            LB[j] = 0.0;
            UB[j] = 1.0;
        }
    }

    /* Initial temperature and standard deviation */
    T = p.T0;
    for (int j=0; j<nVar; j++) {
        sigma[j] = p.sigma0 * (UB[j] - LB[j]);
    }

    /* Initial position of each agent */
	for (int i=0; i<p.nPop; i++) {
        for (int j=0; j<nVar; j++) {
            agent_pos[i][j] = LB[j] + unif(generator) * (UB[j] - LB[j]);
        }
	}

    /* Correct for any integer variable */
    for (int j=0; j<p.nIntVar; j++) {
        idx = p.IntVar[j];
        for (int i=0; i<p.nPop; i++) {
            agent_pos[i][idx] = round(agent_pos[i][idx]);
        }
    }

    /* Initial cost of each agent */
    if (p.normalize) {
        for (int i=0; i<p.nPop; i++) {
            for (int j=0; j<nVar; j++) {
                agent_pos_orig[i][j] = LB_orig[j] + agent_pos[i][j] *
                                       (UB_orig[j] - LB_orig[j]);
            }
            agent_cost[i] = func(agent_pos_orig[i], nVar, p.args);
        }
    }
    else {
        for (int i=0; i<p.nPop; i++) {
            agent_cost[i] = func(agent_pos[i], nVar, p.args);
        }
    }

    /* Initial (overall) best position/cost */
    idx = fmin(agent_cost, p.nPop);
    best_cost = agent_cost[idx];
	for (int j=0; j<nVar; j++) {
        best_pos[j] = agent_pos[idx][j];
    }

    /* Main loop (T = const) */
    for (int epoch=0; epoch<p.epochs; epoch++) {

        /* Sub-loop (search the neighboroud of a state) */
        for (int move=0; move<p.nMove; move++) {

            /* Create the agent's neighbours */
            for (int i=0; i<p.nPop; i++) {
                for (int j=0; j<nVar; j++) {
                    if (unif(generator) <= p.prob) {
                        neigh_pos[i][j] = agent_pos[i][j] +
                                          norm(generator) * sigma[j];
                    }
                    else {
                        neigh_pos[i][j] = agent_pos[i][j];
                    }
                }
            }

            /* Correct for any integer variable */
            for (int j=0; j<p.nIntVar; j++) {
                idx = p.IntVar[j];
                for (int i=0; i<p.nPop; i++) {
                    neigh_pos[i][idx] = round(neigh_pos[i][idx]);
                }
            }

            /* Impose position boundaries */
            for (int i=0; i<p.nPop; i++) {
                for (int j=0; j<nVar; j++) {
                    neigh_pos[i][j] = max(neigh_pos[i][j], LB[j]);
                    neigh_pos[i][j] = min(neigh_pos[i][j], UB[j]);
                }
            }

            /* Impose position boundaries on any integer variable */
            for (int j=0; j<p.nIntVar; j++) {
                idx = p.IntVar[j];
                for (int i=0; i<p.nPop; i++) {
                    neigh_pos[i][idx] = max(neigh_pos[i][idx], ceil(LB[idx]));
                    neigh_pos[i][idx] = min(neigh_pos[i][idx], floor(UB[idx]));
                }
            }

            /* Evaluate the cost of each agent's neighbour */
            if (p.normalize) {
                for (int i=0; i<p.nPop; i++) {
                    for (int j=0; j<nVar; j++) {
                        neigh_pos_orig[i][j] = LB_orig[j] + neigh_pos[i][j] *
                                               (UB_orig[j] - LB_orig[j]);
                    }
                    neigh_cost[i] = func(neigh_pos_orig[i], nVar, p.args);
                }
            }
            else {
                for (int i=0; i<p.nPop; i++) {
                    neigh_cost[i] = func(neigh_pos[i], nVar, p.args);
                }
            }

            /* Decide if each agent will change its state */
            for (int i=0; i<p.nPop; i++) {

                /* Swap states if the neighbour state is better ... */
                if (neigh_cost[i] <= agent_cost[i]) {
                    agent_cost[i] = neigh_cost[i];
                    for (int j=0; j<nVar; j++) {
                        agent_pos[i][j] = neigh_pos[i][j];
                    }
                }

                /* ... or decide probabilistically */
                else {
                    delta = (neigh_cost[i] - agent_cost[i]) / agent_cost[i];
                    prob_swap = exp(-delta / T);

                    /* Randomly swap states */
                    if (unif(generator) <= prob_swap) {
                        agent_cost[i] = neigh_cost[i];
                        for (int j=0; j<nVar; j++) {
                            agent_pos[i][j] = neigh_pos[i][j];
                        }
                    }
                }  

                /* Update the (overall) best position/cost */
                idx = fmin(agent_cost, p.nPop);
                best_cost = agent_cost[idx];
                for (int j=0; j<nVar; j++) {
                    best_pos[j] = agent_pos[idx][j];
                }
            }
        }

        /* Save the best cost for this epoch */
        F[epoch] = best_cost;

        /* Cooling scheduling */
        T = p.alphaT * T;
 
        /* Random neighboroud search schedule */
        for (int j=0; j<nVar; j++) {
            sigma[j] = p.alphaS * sigma[j];
        }
    }

    /* De-normalize */
    if (p.normalize) {
        for (int j=0; j<nVar; j++) {
            best_pos[j] = LB_orig[j] + best_pos[j] * (UB_orig[j] - LB_orig[j]);
            sigma[j] = sigma[j] * (UB_orig[j] - LB_orig[j]);
        }
    }

    /* Copy solution */
    res.best_cost = best_cost;
    res.T = T;
  	res.best_pos = new double [nVar];
  	res.sigma = new double [nVar];
    for (int j=0; j<nVar; j++) {
        res.best_pos[j] = best_pos[j];
        res.sigma[j] = sigma[j];
    }
  	res.F = new double [p.epochs];
    for (int epoch; epoch<p.epochs; epoch++) {
        res.F[epoch] = F[epoch];
    }

	return res;
}
