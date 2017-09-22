#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "regularizer.hpp"
#include "utils.hpp"

extern size_t MAX_DIM;

double regularizer::zero_oracle(int _regular, double* lambda, double* weight) {
    assert (weight != NULL);
    switch(_regular) {
        case regularizer::L1: {
            return lambda[1] * comp_l1_norm(weight);
            break;
        }
        case regularizer::L2: {
            double l2_norm = comp_l2_norm(weight);
            return lambda[0] * 0.5 * l2_norm * l2_norm;
            break;
        }
        case regularizer::ELASTIC_NET: {
            double l2_norm = comp_l2_norm(weight);
            double l2_part = lambda[0] * 0.5 * l2_norm * l2_norm;
            double l1_part = lambda[1] * comp_l1_norm(weight);
            return l1_part + l2_part;
            break;
        }
        default:
            return 0;
    }
}

void regularizer::first_oracle(int _regular, double* _pR, double* lambda, double* weight) {
    assert (weight != NULL);
    memset(_pR, 0, MAX_DIM * sizeof(double));
    switch(_regular) {
        case regularizer::L1: {
            //Not Available
            break;
        }
        case regularizer::L2: {
            for(size_t i = 0; i < MAX_DIM; i ++) {
                _pR[i] = lambda[0] * weight[i];
            }
            break;
        }
        case regularizer::ELASTIC_NET: {
            //Not Available
            break;
        }
        default:
            break;
    }
}

double regularizer::L1_proximal_loop(double& _prox, double param, size_t times, double additional_constant,
        bool is_averaged) {
    double lazy_average = 0.0;
    for(size_t i = 0; i < times; i ++) {
        _prox += additional_constant;
        if(_prox > param)
            _prox -= param;
        else if(_prox < -param)
            _prox += param;
        else
            _prox = 0;
        if(is_averaged)
            lazy_average += _prox;
    }
    return lazy_average;
}

double regularizer::L1_single_step(double& X, double P, double C, bool is_averaged) {
    double lazy_average = 0.0;
    X += C;
    if(X > P)
        X -= P;
    else if(X < -P)
        X += P;
    else
        X = 0;
    if(is_averaged)
        lazy_average = X;
    return lazy_average;
}

double regularizer::EN_proximal_loop(double& X, double P, double Q, size_t times, double C, bool is_averaged) {
    double lazy_average = 0;
    for(size_t i = 0; i < times; i ++) {
        X += C;
        if(X > P)
            X = Q * (X - P);
        else if(X < -P)
            X = Q * (X + P);
        else
            X = 0;
        if(is_averaged)
            lazy_average += X;
    }
    return lazy_average;
}

double regularizer::EN_single_step(double& X, double P, double Q, double C, bool is_averaged) {
    double lazy_average = 0.0;
    X += C;
    if(X > P)
        X = Q * (X - P);
    else if(X < -P)
        X = Q * (X + P);
    else
        X = 0;
    if(is_averaged)
        lazy_average = X;
    return lazy_average;
}

double regularizer::proximal_operator(int _regular, double& _prox, double step_size
    , double* lambda) {
    switch(_regular) {
        case regularizer::L1: {
            double param = step_size * lambda[1];
            if(_prox > param)
                _prox -= param;
            else if(_prox < -param)
                _prox += param;
            else
                _prox = 0;
            return _prox;
        }
        case regularizer::L2: {
            _prox = _prox / (1 + step_size * lambda[0]);
            return _prox;
        }
        case regularizer::ELASTIC_NET: {
            double param_1 = step_size * lambda[1];
            double param_2 = 1.0 / (1.0 + step_size * lambda[0]);
            if(_prox > param_1)
                _prox = param_2 * (_prox - param_1);
            else if(_prox < - param_1)
                _prox = param_2 * (_prox + param_1);
            else
                _prox = 0;
            return _prox;
            break;
        }
        default:
            return 0.0;
            break;
    }
}

// Lazy(Lagged) Update
double regularizer::proximal_operator(int _regular, double& _prox, double step_size
    , double* lambda, size_t times, bool is_averaged, double C) {
    double lazy_average = 0.0;
    switch(_regular) {
        case regularizer::L1: {
            // New DnC Method
            double P = step_size * lambda[1];
            double X = _prox;
            size_t K = times;
            if(C >= P || C <= -P) {
                bool flag = false;
                // Dual Case
                if(C < -P) {
                    flag = true;
                    C = -C;
                    X = -_prox;
                }
                while(X < P - C && K > 0) {
                    double thres = ceil((-P - C - X) / (P + C));
                    if(K <= thres) {
                        if(is_averaged)
                            lazy_average = K * X + (P + C) * (1 + K) * K / 2.0;
                        _prox = X + K * (P + C);
                        if(flag) {
                            _prox = -_prox;
                            lazy_average = -lazy_average;
                        }
                        return lazy_average;
                    }
                    else if(thres > 0.0){
                        if(is_averaged)
                            lazy_average += thres * X + (P + C) * (1 + thres) * thres / 2.0;
                        X += thres * (P + C);
                        K -= thres;
                    }
                    lazy_average += L1_single_step(X, P, C, is_averaged);
                    K --;
                }
                if(K == 0) {
                    _prox = X;
                    if(flag) {
                        _prox = -_prox;
                        lazy_average = -lazy_average;
                    }
                    return lazy_average;
                }
                _prox = X + K * (C - P);
                if(is_averaged)
                    lazy_average += K * X + (C - P) * (1 + K) * K / 2.0;
                if(flag) {
                    lazy_average = -lazy_average;
                    _prox = -_prox;
                }
                return lazy_average;
            }
            else {
                double thres_1 = max(ceil((P - C - X) / (C - P)), 0.0);
                double thres_2 = max(ceil((-P - C - X) / (C + P)), 0.0);
                if(thres_2 == 0 && thres_1 == 0) {
                    _prox = 0;
                    return 0;
                }
                else if(K > thres_1 && K > thres_2) {
                    _prox = 0;
                    if(thres_1 != 0.0 && is_averaged)
                        lazy_average = thres_1 * X + (C - P) * (1 + thres_1) * thres_1 / 2.0;
                    else if(is_averaged)
                        lazy_average = thres_2 * X + (P + C) * (1 + thres_2) * thres_2 / 2.0;
                }
                else {
                    if(X > 0) {
                        if(is_averaged)
                            lazy_average = K * X + (C - P) * (1 + K) * K / 2.0;
                        _prox = X + K * (C - P);
                    }
                    else {
                        if(is_averaged)
                            lazy_average = K * X + (P + C) * (1 + K) * K / 2.0;
                        _prox = X + K * (P + C);
                    }
                }
                return lazy_average;
            }
            break;
        }
        case regularizer::L2: {
            if(times == 1) {
                _prox = (_prox + C) / (1 + step_size * lambda[0]);
                return _prox;
            }
            double param_1 = step_size * lambda[0];
            double param_2 = pow((double) 1.0 / (1 + param_1), (double) times);
            double param_3 = C / param_1;
            if(is_averaged)
                lazy_average = (_prox - param_3) * (1 - param_2) / param_1 + param_3 * times;
            _prox = _prox * param_2 + param_3 * (1 - param_2);
            return lazy_average;
            break;
        }
        case regularizer::ELASTIC_NET: {
            // New DnC Method
            double P = step_size * lambda[1];
            double Q = 1.0 / (1.0 + step_size * lambda[0]);
            double X = _prox;
            size_t K = times;
            if(C >= P || C <= -P) {
                bool flag = false;
                // Dual Case
                if(C < -P) {
                    flag = true;
                    C = -C;
                    X = -_prox;
                }
                double ratio_PCQ = (P + C) * Q / (1 - Q);
                double ratio_NPCQ = (P - C) * Q / (1 - Q);
                while(X < P - C && K > 0) {
                    double thres = ceil(log((P + C + ratio_PCQ) / (ratio_PCQ - X)) / log(Q));
                    if(K <= thres) {
                        double pow_QK = pow((double) Q, (double) K);
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_QK, K) * (X - ratio_PCQ)
                                         + ratio_PCQ * K;
                        _prox = pow_QK * X + ratio_PCQ * (1 - pow_QK);
                        if(flag) {
                            _prox = -_prox;
                            lazy_average = -lazy_average;
                        }
                        return lazy_average;
                    }
                    else if(thres > 0.0){
                        double pow_Qtrs = pow((double) Q, (double) thres);
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_Qtrs, thres) * (X - ratio_PCQ)
                                         + ratio_PCQ * thres;
                        X = pow_Qtrs * X + ratio_PCQ * (1 - pow_Qtrs);
                        K -= thres;
                    }
                    lazy_average += EN_single_step(X, P, Q, C, is_averaged);
                    K --;
                }
                if(K == 0) {
                    _prox = X;
                    if(flag) {
                        _prox = -_prox;
                        lazy_average = -lazy_average;
                    }
                    return lazy_average;
                }
                double pow_QK = pow((double) Q, (double) K);
                _prox = pow_QK * X - ratio_NPCQ * (1 - pow_QK);
                if(is_averaged)
                    lazy_average += equal_ratio(Q, pow_QK, K) * (X + ratio_NPCQ)
                                         - ratio_NPCQ * K;
                if(flag) {
                    lazy_average = -lazy_average;
                    _prox = -_prox;
                }
                return lazy_average;
            }
            else {
                double ratio_PCQ = (P + C) * Q / (1 - Q);
                double ratio_NPCQ = (P - C) * Q / (1 - Q);
                double thres_1 = max(ceil(log((P - C + ratio_NPCQ) / (ratio_NPCQ + X)) / log(Q)), 0.0); // P-C
                double thres_2 = max(ceil(log((P + C + ratio_PCQ) / (ratio_PCQ - X)) / log(Q)), 0.0); // -P-C
                if(thres_2 == 0 && thres_1 == 0) {
                    _prox = 0;
                    return 0;
                }
                else if(K > thres_1 && K > thres_2) {
                    _prox = 0;
                    double pow_Qtrs1 = pow((double) Q, (double) thres_1);
                    double pow_Qtrs2 = pow((double) Q, (double) thres_2);
                    if(thres_1 != 0.0 && is_averaged)
                        lazy_average = equal_ratio(Q, pow_Qtrs1, thres_1) * (X + ratio_NPCQ)
                                         - ratio_NPCQ * thres_1;
                    else if(is_averaged)
                        lazy_average = equal_ratio(Q, pow_Qtrs2, thres_2) * (X - ratio_PCQ)
                                         + ratio_PCQ * thres_2;
                }
                else {
                    double pow_QK = pow((double) Q, (double) K);
                    if(X > 0) {
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_QK, K) * (X + ratio_NPCQ)
                                         - ratio_NPCQ * K;
                        _prox = pow_QK * X - ratio_NPCQ * (1 - pow_QK);
                    }
                    else {
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_QK, K) * (X - ratio_PCQ)
                                         + ratio_PCQ * K;
                        _prox = pow_QK * X + ratio_PCQ * (1 - pow_QK);
                    }
                }
                return lazy_average;
            }
            break;
        }
        default:
            return 0.0;
            break;
    }
}
