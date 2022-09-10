#ifndef SEIRODE_HPP
#define SEIRODE_HPP

#include <iostream>
#include <vector>

namespace sir
{

double constexpr TIME_STEP = 0.01; // days

struct State
{
    double S;
    double I;
    double R;

    // Susceptibles change rate
    double dS_dt(double alpha, int N) const;

    // Infected change rate
    double dI_dt(double alpha, double beta, int N) const;

    // Recovered change rate
    double dR_dt(double beta) const;
};

class SIR
{
  private:
    int N;        // Total number of individuals
    int t;        // Duration time of the simulation(days)
    State S_0;    // Initial state
    double alpha; // Governs the lag between infectious contact and showing symptoms
    double beta;  // Number of people an infective person infects each day
    double gamma; // Probability to recover or die (accounts for two parameters)
    double mu;
    double eps;

  public:
    // Constructor
    SIR(int population, int time, State initial_state, double alpha, double beta);

    // The default constructor should not be used
    SIR() = delete;

  private:
    // Returns newly calculated State using Euler method(1th order numerical approximation)
    State EulerSolver(const State& current_state);

    // Returns newly calculated State using Runge Kutta method(4th order numerical approximation)
    State RungeKuttaSolver(const State& current_state);

  public:
    // Performs the actual simulation:determines the various states by solving the ode system
    // Over the setted time interval either with Euler method (method == 0) or with RK4 (method == 1)
    void evolve(std::vector<State>& states, bool method = 1);
};

} // namespace seir

#endif
