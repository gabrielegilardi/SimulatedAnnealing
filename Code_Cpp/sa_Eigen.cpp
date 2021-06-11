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
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

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
    ArrayXi IntVar;
    ArrayXXd args;
    int seed;
};

/* Structure used to return the results */
struct Results {
    double best_cost;
    ArrayXXd best_pos;
    ArrayXd F;
    double T;
    ArrayXXd sigma;
};


/* Minimize a function using simulated annealing */
Results sa(ArrayXd (*func)(ArrayXXd, ArrayXXd), ArrayXXd LB, ArrayXXd UB,
           Parameters p)
{
    int nVar, nIntVar, idx;
    double T, prob_swap, delta, best_cost;
    Results res;

    /*Eigen declarations*/
    Index r_min, c_min;
    ArrayXd agent_cost, neigh_cost, F;
    ArrayXXd LBe, UBe, LBe_orig, UBe_orig, sigma, agent_pos, neigh_pos,
             agent_pos_orig, neigh_pos_orig, best_pos, rn, flips;
    
    /* Random generator and probability distributions */
    mt19937 generator(p.seed);
    uniform_real_distribution<double> unif(0.0, 1.0);
    normal_distribution<double> norm(0.0, 1.0);


    nVar = LB.size();
    nIntVar = p.IntVar.size();

    /* Create boundaries for all agents*/
    LBe = LB.replicate(p.nPop, 1);
    UBe = UB.replicate(p.nPop, 1);

    /* Normalize search space */
    if (p.normalize) {
        LBe_orig = LBe;
        UBe_orig = UBe;
        LBe.setZero(p.nPop, nVar);
        UBe.setOnes(p.nPop, nVar);
    }

    T = p.T0;                           // Temperature
    sigma = p.sigma0 * (UBe - LBe);     // Standard deviation of each dimension

    /* Initial position of each agent */
    rn.setZero(p.nPop, nVar);
    for (int i=0; i<p.nPop; i++) {
        for (int j=0; j<nVar; j++) {
            rn(i, j) = unif(generator);
        }
    }
    agent_pos = LBe + rn * (UBe - LBe);

    /* Correct for any integer variable */
    for (int j=0; j<nIntVar; j++) {
        idx = p.IntVar(j);
        agent_pos.col(idx) = round(agent_pos.col(idx));
    }

    /* Initial cost of each agent */
    if (p.normalize) {
        agent_pos_orig = LBe_orig + agent_pos * (UBe_orig - LBe_orig);
        agent_cost = func(agent_pos_orig, p.args);
    }
    else {
        agent_cost = func(agent_pos, p.args);
    }

    /* Initial (overall) best position/cost */
    best_cost = agent_cost.minCoeff(&r_min, &c_min);
    best_pos = agent_pos.row(r_min);

    /* Main loop (T = const) */
    neigh_pos.setZero(p.nPop, nVar);
    F.setZero(p.epochs);
    for (int epoch=0; epoch<p.epochs; epoch++) {

        /* Sub-loop (search the neighboroud of a state) */
        for (int move=0; move<p.nMove; move++) {

            /* Randomly decide in which dimension to search */
            for (int i=0; i<p.nPop; i++) {
                for (int j=0; j<nVar; j++) {
                    rn(i, j) = unif(generator);
                }
            }
            flips = (rn <= p.prob).cast<double>();

            /* Create each agent's neighbours */
            for (int i=0; i<p.nPop; i++) {
                for (int j=0; j<nVar; j++) {
                    rn(i, j) = norm(generator);
                }
            }
            neigh_pos = agent_pos + flips * rn * sigma;

            /* Correct for any integer variable */
            for (int j=0; j<nIntVar; j++) {
                idx = p.IntVar(j);
                neigh_pos.col(idx) = round(neigh_pos.col(idx));
            }

            /* Impose position boundaries */
            neigh_pos = neigh_pos.max(LBe);
            neigh_pos = neigh_pos.min(UBe);

            for (int j=0; j<nIntVar; j++) {
                idx = p.IntVar(j);
                neigh_pos.col(idx) = neigh_pos.col(idx).max(ceil(LBe.col(idx)));
                neigh_pos.col(idx) = neigh_pos.col(idx).min(floor(UBe.col(idx)));
            }

            /* Evaluate the cost of each agent's neighbour */
            if (p.normalize) {
                neigh_pos_orig = LBe_orig + neigh_pos * (UBe_orig - LBe_orig);
                neigh_cost = func(neigh_pos_orig, p.args);
            }
            else {
                neigh_cost = func(neigh_pos, p.args);
            }

            /* Decide if each agent will change its state */
            for (int i=0; i<p.nPop; i++) {

                /* Swap states if the neighbour state is better ... */
                if (neigh_cost(i) <= agent_cost(i)) {
                    agent_cost(i) = neigh_cost(i);
                    agent_pos.row(i) = neigh_pos.row(i);
                }

                /* ... or decide probabilistically */
                else {

                    /* Acceptance probability */
                    delta = (neigh_cost(i) - agent_cost(i)) / agent_cost(i);
                    prob_swap = exp(-delta / T);

                    /* Randomly swap states */
                    if (unif(generator) <= prob_swap) {
                        agent_cost(i) = neigh_cost(i);
                        agent_pos.row(i) = neigh_pos.row(i);
                    }
                }  

                /* Update the (overall) best position/cost */
                best_cost = agent_cost.minCoeff(&r_min, &c_min);
                best_pos = agent_pos.row(r_min);
            }
        }

        /* Save the best cost for this epoch */
        F(epoch) = best_cost;

        /* Cooling scheduling */
        T = p.alphaT * T;
 
        /* Random neighboroud search schedule */
        sigma = p.alphaS * sigma;
    }

    /* De-normalize */
    if (p.normalize) {
        best_pos = LB + best_pos * (UB - LB);
        sigma = sigma * (UBe_orig - LBe_orig);
    }

    /* Copy solution */
    res.best_cost = best_cost;
    res.T = T;
    res.best_pos = best_pos;
    res.sigma = sigma.row(0);
    res.F = F;

    return res;
}
