#include "Matrix.h"

Matrix::Matrix(){

}


Matrix::Matrix(std::vector<std::vector<double>> data) {
    this->data = data;
}

Matrix::Matrix(int n, int m) {
    this->data = std::vector<std::vector<double>>(n, std::vector<double>(m, 0));
}

Matrix::Matrix(std::vector<double> diag){
    this->data = std::vector<std::vector<double>>(diag.size(), std::vector<double>(diag.size(), 0));
    for(int i = 0; i < diag.size(); i++){
        data[i][i] = diag[i];
    }
}

Matrix::Matrix(const Matrix & other){
    this->data = std::vector<std::vector<double>>(other.data.size(), std::vector<double>(other.data[0].size(), 0));
    for(int i = 0; i < other.data.size(); i++){
        for (int j = 0; j < other.data[0].size(); j++){
            this->data[i][j] = other.data[i][j];
        }
    }
}

Matrix & Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        this->data = std::vector<std::vector<double>>(other.data.size(), std::vector<double>(other.data[0].size(), 0));
        for(int i = 0; i < other.data.size(); i++){
            for (int j = 0; j < other.data[0].size(); j++){
                this->data[i][j] = other.data[i][j];
            }
        }
    }
    return *this;
}

Matrix::Matrix(Matrix &&other) {
    this->data = other.data;
}

Matrix& Matrix::operator=(Matrix &&other) {
    if (this == &other) return *this;
    this->data = other.data;
    return *this;
}

Matrix Matrix::operator * (const Matrix& obj){
    // check matrix size
    if (data[0].size() != obj.data.size()){
        throw std::logic_error("Bad size!");
    }
    Matrix res(this->data.size(), obj.data[0].size());

    for (int i = 0; i < this->data.size(); i++){
        for (int j = 0; j < obj.data[0].size(); j++){
            for (int k = 0; k < this->data[0].size(); k++){
                res(i, j) += this->data[i][k] * obj.data[k][j];
            }
        }
    }

    return res;
}

double Matrix::det(){
    if (data.size() != data[0].size()){
        throw std::logic_error("Bad size!");
    }

    if ((data.size() == 1) & (data[0].size() == 1)){
        return data[0][0];
    }

    if ((data.size() == 2) & (data[0].size() == 2)){
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    }

    double d = 0;
    for (int k = 0; k < data.size(); k++){
        Matrix m(data.size() - 1, data.size() - 1);
        for (int i = 0; i < data.size(); i++){
            for (int j = 1; j < data.size(); j++){
                if (i == k ){
                    continue;
                }
                else{
                    m(i - (int)(i > k), j - 1) = data[i][j];
                }
            }
        }
        d += std::pow(-1, k + 2) * data[k][0] * m.det();
    }
    return d;
}

std::pair<int, int> Matrix::Size(){
    return std::make_pair(this->data.size(), this->data[0].size());
}

double Matrix::L2(){
    double res = 0;
    for (int i = 0; i < this->Size().first; i++){
        for (int j = 0; j < this->Size().second; j++){
            res += this->data[i][j] * this->data[i][j];
        }
    }
    return sqrt(res);
}

double Matrix::L1(){
    double res = 0;
    for (int i = 0; i < this->Size().first; i++){
        for (int j = 0; j < this->Size().second; j++){
            res += fabs(this->data[i][j]);
        }
    }
    return res;
}

Matrix Matrix::operator * (double k){
    Matrix res = (*this);
    for (int i = 0; i < this->Size().first; i++){
        for (int j = 0; j < this->Size().second; j++){
            res(i, j) = k * res(i, j);
        }
    }
    return res;
}

Matrix Matrix::T(){
    Matrix res(this->data[0].size(), this->data.size());
    for (int i = 0; i < this->data.size(); i++){
        for (int j = 0; j < this->data[0].size(); j++){
            res(j, i) = this->data[i][j];
        }
    }
    return res;
}


Matrix Matrix::operator + (const Matrix& obj){
    Matrix res(data.size(), data[0].size());
    for (int i = 0; i < data.size(); i++){
        for (int j = 0; j < data[0].size(); j++){
            res(i, j) = (*this)(i, j) + obj(i, j);
        }
    }
    return res;
}

double& Matrix::operator()(int i, int j){
    return this->data[i][j];
}

const double& Matrix::operator()(int i, int j) const{
    return this->data[i][j];
}

Matrix Matrix::Cofactor() {
    if(this->Size().first != this->Size().second) {
        throw std::runtime_error("Matrix is not quadratic");
    }

    Matrix solution(this->Size().first, this->Size().first);
    Matrix subVect(this->Size().first - 1, this->Size().first - 1);

    for(std::size_t i = 0; i < this->Size().first; i++) {
        for(std::size_t j = 0; j < this->Size().first; j++) {

            int p = 0;
            for(size_t x = 0; x < this->Size().first; x++) {
                if(x == i) {
                    continue;
                }
                int q = 0;

                for(size_t y = 0; y < this->Size().first; y++) {
                    if(y == j) {
                        continue;
                    }

                    subVect(p, q) = (*this)(x, y);
                    q++;
                }
                p++;
            }
            solution(i, j) = pow(-1, i + j) * subVect.det();
        }
    }
    return solution;
}



Matrix Matrix::Inv() {
    if(this->det() == 0) {
        throw std::runtime_error("Determinant is 0");
    }

    double d = 1.0 / this->det();
    Matrix solution(this->Size().first, this->Size().second);

    for(int i = 0; i < this->Size().first; i++) {
        for(int j = 0; j < this->Size().second; j++) {
            solution(i, j) = (*this)(i, j);
        }
    }

    solution = solution.Cofactor().T();

    for(int i = 0; i < this->Size().first; i++) {
        for(int j = 0; j < this->Size().second; j++) {
            solution(i, j) *= d;
        }
    }

    return solution;
}

Matrix Matrix::invertLT()
{
    int N = this->Size().first;
    Matrix M( N, N);

    for ( int i = 0; i < N; i++ )
    {
        if ( (*this)(i, i) == 0.0 )
        {
            throw std::logic_error("Singular matrix!");
            std::cout<<"Singular matrix\n";
            return M;

        }
        M(i, i) = 1.0 / (*this)(i, i);
        for ( int j = 0; j < i; j++ )
        {
            for ( int k = j; k < i; k++ ) M(i, j) += (*this)(i, k) * M(k, j);
            M(i, j) = -M(i, j) / (*this)(i, i);
        }
    }
    return M;
}