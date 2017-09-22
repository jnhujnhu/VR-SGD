#include "svm.hpp"
#include "utils.hpp"
#include <math.h>
#include <string.h>

extern size_t MAX_DIM;

svm::svm(size_t params_no, double* params, int regular) {
    m_params = new double[params_no];
    for(size_t i = 0; i < params_no; i ++)
        m_params[i] = params[i];
    m_regularizer = regular;
    m_weights = new double[MAX_DIM];
}

svm::svm(double param, int regular) {
    m_regularizer = regular;
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double svm::zero_component_oracle_dense(double* X, double* Y, size_t N, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < N; i ++) {
        double innr_xw = 0;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            innr_xw += weights[j] * X[i * MAX_DIM + j];
        }
        double slack = 1 - Y[i] * innr_xw;
        if(slack > 0) {
            _F += slack / (double) N;
        }
    }
    return _F;
}

double svm::zero_component_oracle_sparse(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < N; i ++) {
        double innr_xw = 0;
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            innr_xw += weights[Ir[j]] * X[j];
        }
        double slack = 1 - Y[i] * innr_xw;
        if(slack > 0) {
            _F += slack / (double) N;
        }
    }
    return _F;
}

double svm::first_component_oracle_core_dense(double* X, double* Y, size_t N, int given_index, double* weights) const {
    //Sub Gradient For SVM
    if(weights == NULL) weights = m_weights;
    double innr_xw = 0;
    for(size_t j = 0; j < MAX_DIM; j ++) {
        innr_xw += weights[j] * X[given_index * MAX_DIM + j];
    }
    if(Y[given_index] * innr_xw < 1)
        return -Y[given_index];
    else
        return 0.0;
}

double svm::first_component_oracle_core_sparse(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, int given_index, double* weights) const {
    //Sub Gradient For SVM
    if(weights == NULL) weights = m_weights;
    double innr_xw = 0;
    for(size_t j = Jc[given_index]; j < Jc[given_index + 1]; j ++) {
        innr_xw += weights[Ir[j]] * X[j];
    }
    if(Y[given_index] * innr_xw < 1)
        return -Y[given_index];
    else
        return 0.0;
}

int svm::classify(double* sample) const{
    return 1;
}
