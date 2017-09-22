#ifndef REGULARIZER_HPP
#define REGULARIZER_HPP
#include <stdio.h>

namespace regularizer {
    const int NONE = -1;
    const int L1 = 0;
    const int L2 = 1;
    const int ELASTIC_NET = 2;
    double zero_oracle(int _regular, double* lambda, double* weight);
    void first_oracle(int _regular, double* _pR, double* lambda, double* weight);
    double L1_proximal_loop(double& _prox, double param, size_t times, double additional_constant
        , bool is_averaged);
    double EN_proximal_loop(double& _prox, double P, double Q, size_t times, double additional_constant
        , bool is_averaged);
    double L1_single_step(double& X, double P, double C, bool is_averaged);
    double EN_single_step(double& X, double P, double Q, double C, bool is_averaged);
    double proximal_operator(int _regular, double& _prox, double step_size, double* lambda
            , size_t times, bool is_averaged = false, double additional_constant = 0.0);
    double proximal_operator(int _regular, double& _prox, double step_size, double* lambda);
}
#endif
