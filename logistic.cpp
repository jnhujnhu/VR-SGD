#include "logistic.hpp"
#include "utils.hpp"
#include <math.h>
#include <string.h>

extern size_t MAX_DIM;

logistic::logistic(size_t params_no, double* params, int regular) {
    m_params = new double[params_no];
    for(size_t i = 0; i < params_no; i ++)
        m_params[i] = params[i];
    m_regularizer = regular;
    m_weights = new double[MAX_DIM];
}

logistic::logistic(double param, int _regularizer) {
    m_regularizer = _regularizer;
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}

double logistic::zero_component_oracle_dense(double* X, double* Y, size_t N, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < N; i ++) {
        double innr_yxw = 0.0;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            innr_yxw += weights[j] * X[i * MAX_DIM + j];
        }
        innr_yxw *= -Y[i];
        _F += log(1.0 + exp(innr_yxw));
    }
    return _F / (double) N;
}

double logistic::zero_component_oracle_sparse(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(size_t i = 0; i < N; i ++) {
        double innr_yxw = 0.0;
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            innr_yxw += weights[Ir[j]] * X[j];
        }
        innr_yxw *= -Y[i];
        _F += log(1.0 + exp(innr_yxw));
    }
    return _F / (double) N;
}

double logistic::first_component_oracle_core_dense(double* X, double* Y, size_t N, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double sigmoid = 0.0;
    for(size_t j = 0; j < MAX_DIM; j ++) {
        sigmoid += weights[j] * X[given_index * MAX_DIM + j];
    }
    sigmoid = exp(sigmoid * - Y[given_index]);
    sigmoid = -Y[given_index] * sigmoid /  (1 + sigmoid);
    return sigmoid;
}

double logistic::first_component_oracle_core_sparse(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double sigmoid = 0.0;
    for(size_t j = Jc[given_index]; j < Jc[given_index + 1]; j ++) {
        sigmoid += weights[Ir[j]] * X[j];
    }
    sigmoid = exp(sigmoid * - Y[given_index]);
    sigmoid = -Y[given_index] * sigmoid /  (1 + sigmoid);
    return sigmoid;
}

int logistic::classify(double* sample) const{
    return 1;
}
