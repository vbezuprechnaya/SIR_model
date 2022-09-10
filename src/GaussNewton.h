#include "Matrix.h"
#include "sir.hpp"
#include "math.h"
#ifndef SIR_GAUSSNEWTON_H
#define SIR_GAUSSNEWTON_H


class GaussNewton {
public:
    GaussNewton(std::vector<sir::State> data, sir::State init, double alpha, double beta);
    std::pair<double, double> GetParameters();
    void Approximate(int cnt_iter);
private:
    Matrix CalculateJacobian();
    //Matrix CalculateDelta();
    double appr_alpha;
    double appr_beta;
    sir::State init;
    std::vector<sir::State> data;
    std::vector<sir::State> actual_predict;
};


#endif //SIR_GAUSSNEWTON_H
