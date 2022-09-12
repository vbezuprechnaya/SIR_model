#include <iostream>
#include "sir.hpp"
#include "GaussNewton.h"

int main() {
    sir::State init{762, 1, 0};
    int time = 5;
    sir::SIR Model(763, time, init, 0.00218, 0.4485);
    std::vector<sir::State> states = {};
    Model.evolve(states);


    double s = states[states.size() - 1].S + states[states.size() - 1].I + states[states.size() - 1].R;
    double appr_alpha = 0.003, appr_beta = 0.4485;
    GaussNewton Optimizer(states, init, appr_alpha, appr_beta);
    Optimizer.Approximate(100);
    return 0;
}
