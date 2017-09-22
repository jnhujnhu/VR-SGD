#include "blackbox.hpp"
#include "regularizer.hpp"
#include <string.h>

blackbox::~blackbox() {
    delete[] m_weights;
    delete[] m_params;
}

double blackbox::zero_oracle_dense(double* X, double* Y, size_t N, double* weights) const {
    return zero_component_oracle_dense(X, Y, N, weights) + zero_regularizer_oracle(weights);
}

double blackbox::zero_oracle_sparse(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, double* weights) const {
    return zero_component_oracle_sparse(X, Y, Jc, Ir, N, weights) + zero_regularizer_oracle(weights);
}

double blackbox::zero_regularizer_oracle(double* weights) const {
    if(weights == NULL) weights = m_weights;
    return regularizer::zero_oracle(m_regularizer, m_params, weights);
}

void blackbox::first_regularizer_oracle(double* _pR, double* weights) const {
    if(weights == NULL) weights = m_weights;
    regularizer::first_oracle(m_regularizer, _pR, m_params, weights);
}

void blackbox::set_init_weights(double* init_weights) {
    update_model(init_weights);
}

double* blackbox::get_model() const {
    return m_weights;
}

int blackbox::get_regularizer() const {
    return m_regularizer;
}

double* blackbox::get_params() const {
    return m_params;
}

void blackbox::update_model(double* new_weights) {
    for(size_t i = 0; i < MAX_DIM; i ++)
        m_weights[i] = new_weights[i];
}
