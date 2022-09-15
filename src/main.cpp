#include <iostream>
#include "sir.hpp"
#include "GaussNewton.h"

int main() {
    sir::State init{762, 1, 0};
    int time = 5;
    sir::SIR Model(763, time, init, 0.03218, 0.4485);
    std::vector<sir::State> states = {};
    Model.evolve(states);

    double ans_a = 0, ans_b = 0;
    for (int i = 0; i < states.size() - 1; i++){
        ans_a += (states[i].S - states[i + 1].S) / (states[i].S * states[i].I * 0.01);
        ans_b += (states[i + 1].R - states[i].R) / (states[i].I * 0.01);

    }
    std::cout << ans_a / (states.size() - 1) << " " << ans_b / (states.size() - 1);


//    double s = states[states.size() - 1].S + states[states.size() - 1].I + states[states.size() - 1].R;
//    double appr_alpha = 0.003, appr_beta = 0.4485;
//    GaussNewton Optimizer(states, init, appr_alpha, appr_beta);
//    Optimizer.Approximate(100);
    return 0;
}
