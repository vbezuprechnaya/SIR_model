#include "GaussNewton.h"

double LossFunc(sir::State predict, sir::State labels){
    double result = 0;
    result += (labels.S - predict.S) * (labels.S - predict.S);
    result += (labels.I - predict.I) * (labels.I - predict.I);
    result += (labels.R - predict.R) * (labels.R - predict.R);
    return result / 3;
}

Matrix LossFuncAll(std::vector<sir::State> ans, std::vector<sir::State> predict){
    Matrix result(ans.size() * 3, 1);
    for (int i = 0; i < ans.size(); i++){
        result(i, 0) = ans[i].S - predict[i].S;
    }

    for (int i = 0; i < ans.size(); i++){
        result(i + ans.size(), 0) = ans[i].I - predict[i].I;
    }

    for (int i = 0; i < ans.size(); i++){
        result(i + 2 * ans.size(), 0) = ans[i].R - predict[i].R;
    }
    return result;
}


GaussNewton::GaussNewton(std::vector<sir::State> data, sir::State init, double alpha, double beta) {
    this->appr_alpha = alpha;
    this->appr_beta = beta;
    this->data = data;
    this->init = init;
}

std::pair<double, double> GaussNewton::GetParameters(){
    return std::make_pair(this->appr_alpha, this->appr_beta);
}


void GaussNewton::Approximate(int cnt_iter){
    int step = 0;
    Matrix Params(2, 1);
    Params(0, 0) = appr_alpha, Params(1, 0) = appr_beta;

    while (step < cnt_iter){
        step++;
        for (int i = 1; i < this->data.size(); i++){
            Matrix J  = this->CalculateJacobian();
            Matrix Delta = ((J.T() * J).Inv() * J.T()) * LossFuncAll(this->data, this->actual_predict) * (-1);
            Params(0, 0) = Params(0, 0) + Delta(0, 0);
            std::cout << Params(0, 0) << "  " << Params(1, 0) << "\n";
        }
    }
}

Matrix GaussNewton::CalculateJacobian() {
    Matrix J(this->data.size() * 3, 2);
    int population = (int)(this->data[0].S + this->data[0].I + this->data[0].R);
    double step = 10e-3;
    int time = (int)(std::floor(this->data.size() * sir::TIME_STEP) + 1);
    sir::SIR Model(population, time, this->init,
                   this->appr_alpha, this->appr_beta);
    std::vector<sir::State> appr_states = {};
    Model.evolve(appr_states);
    this->actual_predict = appr_states;
    sir::SIR ChangeAlphaModel(population, time, this->init,
                   this->appr_alpha + step, this->appr_beta);
    std::vector<sir::State> change_alpha_states = {};
    ChangeAlphaModel.evolve(change_alpha_states);

    sir::SIR ChangeBetaModel(population, time, this->init,
                              this->appr_alpha, this->appr_beta + step);
    std::vector<sir::State> change_beta_states = {};
    ChangeBetaModel.evolve(change_beta_states);

    for (int i = 0; i < this->data.size(); i++){
        J(i, 0) = -(change_alpha_states[i].S - appr_states[i].S) / step;
        J(i, 1) = -(change_beta_states[i].S - appr_states[i].S) / step;
    }

    for (int i = 0; i < this->data.size(); i++){
        J(i + this->data.size(), 0) = -(change_alpha_states[i].I - appr_states[i].I) / step;
        J(i + this->data.size(), 1) = -(change_beta_states[i].I - appr_states[i].I) / step;
    }

    for (int i = 0; i < this->data.size(); i++){
        J(i + 2 * this->data.size(), 0) = -(change_alpha_states[i].R - appr_states[i].R) / step;
        J(i + 2 * this->data.size(), 1) = -(change_beta_states[i].R - appr_states[i].R) / step;
    }

    return J;
}
