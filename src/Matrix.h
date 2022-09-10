
#ifndef SVD_MATRIX_H
#define SVD_MATRIX_H
#include <vector>
#include <iostream>
#include <cmath>
#include <exception>
#include <stdexcept>

class Matrix {
public:
    Matrix();
    Matrix(int n, int m);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other);
    Matrix(std::vector<std::vector<double>> data);
    Matrix(std::vector<double> diag);
    Matrix operator + (const Matrix& obj);
    Matrix operator * (const Matrix& obj);
    Matrix operator * (double k);
    double det();
    Matrix T();
    std::pair<int, int> Size();
    double L2();
    double L1();
    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;
    Matrix Inv();
    Matrix Cofactor();
    Matrix invertLT();
private:
    std::vector<std::vector<double>> data;
};


#endif
