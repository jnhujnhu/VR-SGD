#include "grad_desc_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

double* grad_desc_sparse::GD(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_result) {
    double* stored_F = NULL;
    size_t passes = iteration_no;
    if(is_store_result)
        stored_F = new double[passes];
    double* new_weights = new double[MAX_DIM];
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        for(size_t j = 0; j < N; j ++) {
            double full_grad_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core / (double) N;
            }
        }
        double* _pR = new double[MAX_DIM];
        model->first_regularizer_oracle(_pR);
        for(size_t j = 0; j < MAX_DIM; j ++) {
            full_grad[j] += _pR[j];
            new_weights[j] = (model->get_model())[j] - step_size * (full_grad[j]);
        }
        delete[] _pR;
        model->update_model(new_weights);
        //For Matlab
        if(is_store_result) {
            stored_F[i] = model->zero_oracle_sparse(X, Y, Jc, Ir, N);
        }
        delete[] full_grad;
    }
    delete[] new_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

double* grad_desc_sparse::SGD(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    double* stored_F = NULL;
    size_t passes = (size_t) floor((double) iteration_no / N);
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // lazy updates extra array.
    int* last_seen = new int[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
    // For Matlab
    if(is_store_result) {
        stored_F = new double[passes + 1];
        stored_F[0] = model->zero_oracle_sparse(X, Y, Jc, Ir, N);
    }

    double* sub_grad = new double[MAX_DIM];
    double* new_weights = new double[MAX_DIM];
    copy_vec(new_weights, model->get_model());
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, new_weights);
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // lazy update.
            if((int)i > last_seen[index] + 1) {
                regularizer::proximal_operator(regular, new_weights[index], step_size, lambda
                    , i - (last_seen[index] + 1), false);
            }
            new_weights[index] -= step_size * core * X[j];
            regularizer::proximal_operator(regular, new_weights[index], step_size, lambda);
            last_seen[index] = i;
        }
        // For Matlab
        if(is_store_result) {
            if(!((i + 1) % N)) {
                for(size_t j = 0; j < MAX_DIM; j ++) {
                    if((int)i > last_seen[j]) {
                        regularizer::proximal_operator(regular, new_weights[j], step_size, lambda
                            , i - last_seen[j], false);
                        last_seen[j] = i;
                    }
                }
                stored_F[(size_t) floor((double) i / N) + 1]
                    = model->zero_oracle_sparse(X, Y, Jc, Ir, N, new_weights);
            }
        }
    }
    // lazy update aggragate
    for(size_t i = 0; i < MAX_DIM; i ++) {
        if((int)iteration_no > last_seen[i] + 1) {
            regularizer::proximal_operator(regular, new_weights[i], step_size, lambda
                , iteration_no - (last_seen[i] + 1), false);
        }
    }
    model->update_model(new_weights);
    delete[] sub_grad;
    delete[] new_weights;
    delete[] last_seen;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_sparse::SAGA(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_result) {
        // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // For Matlab
    if(is_store_result) {
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        // Extra Pass for Create Gradient Table
        stored_F->push_back((*stored_F)[0]);
    }
    double* new_weights = new double[MAX_DIM];
    double* grad_core_table = new double[N];
    double* aver_grad = new double[MAX_DIM];
    int* last_seen = new int[MAX_DIM];
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    for(size_t j = 0; j < MAX_DIM; j ++) last_seen[j] = -1;

    copy_vec(new_weights, model->get_model());
    // Init Gradient Core Table
    for(size_t i = 0; i < N; i ++) {
        grad_core_table[i] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, i);
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++)
            aver_grad[Ir[j]] += grad_core_table[i] * X[j] / N;
    }

    size_t skip_pass = 0;
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, new_weights);
        double past_grad_core = grad_core_table[rand_samp];
        grad_core_table[rand_samp] = core;
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // lazy update or lagged update in Schmidt et al.[2016]
            if((int) i > last_seen[index] + 1) {
                regularizer::proximal_operator(regular, new_weights[index], step_size
                        , lambda, i - (last_seen[index] + 1), false, -step_size * aver_grad[index]);
            }
            // Update Weight
            new_weights[index] -= step_size * ((core - past_grad_core)* X[j] + aver_grad[index]);
            // Update Gradient Table Average
            aver_grad[index] -= (past_grad_core - core) * X[j] / N;
            regularizer::proximal_operator(regular, new_weights[index], step_size, lambda);
            last_seen[index] = i;
        }
        // For Matlab
        if(is_store_result) {
            if(!(i % N)) {
                skip_pass ++;
                if(skip_pass == 3) {
                    // Force Lazy aggragate for function value evaluate
                    for(size_t j = 0; j < MAX_DIM; j ++) {
                        if((int)i > last_seen[j]) {
                            regularizer::proximal_operator(regular, new_weights[j], step_size, lambda
                                , i - last_seen[j], false, -step_size * aver_grad[j]);
                            last_seen[j] = i;
                        }
                    }
                    stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N, new_weights));
                    skip_pass = 0;
                }
            }
        }
    }
    // lazy update aggragate
    for(size_t i = 0; i < MAX_DIM; i ++) {
        if((int)iteration_no > last_seen[i] + 1) {
            regularizer::proximal_operator(regular, new_weights[i], step_size, lambda
                , iteration_no - (last_seen[i] + 1), false, -step_size * aver_grad[i]);
        }
    }
    model->update_model(new_weights);
    delete[] new_weights;
    delete[] grad_core_table;
    delete[] aver_grad;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_sparse::Prox_SVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        // lazy update extra params.
        int* last_seen = new int[MAX_DIM];
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
            }
        }
        switch(Mode) {
            case SVRG_LAST_LAST:
            case SVRG_AVER_LAST:
                break;
            case SVRG_AVER_AVER:
                copy_vec(inner_weights, model->get_model());
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_sparse(X, Y
                        , Jc, Ir, N, rand_samp, inner_weights);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                double val = X[k];
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    switch(Mode) {
                        case SVRG_LAST_LAST:
                            regularizer::proximal_operator(regular, inner_weights[index], step_size
                                , lambda, j - (last_seen[index] + 1), false, -step_size * full_grad[index]);
                            break;
                        case SVRG_AVER_LAST:
                        case SVRG_AVER_AVER:
                            aver_weights[index] += regularizer::proximal_operator(regular, inner_weights[index]
                                , step_size, lambda, j - (last_seen[index] + 1), true, -step_size * full_grad[index]) / inner_m;
                            break;
                        default:
                            throw std::string("500 Internal Error.");
                            break;
                    }
                }
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                aver_weights[index] += regularizer::proximal_operator(regular, inner_weights[index]
                    , step_size, lambda) / inner_m;
                last_seen[index] = j;
            }
            total_iterations ++;
        }
        // lazy update aggragate
        switch(Mode) {
            case SVRG_LAST_LAST:
                for(size_t j = 0; j < MAX_DIM; j ++) {
                    if(inner_m > last_seen[j] + 1) {
                        regularizer::proximal_operator(regular, inner_weights[j], step_size, lambda
                            , inner_m - (last_seen[j] + 1), false, -step_size * full_grad[j]);
                    }
                }
                model->update_model(inner_weights);
                break;
            case SVRG_AVER_LAST:
            case SVRG_AVER_AVER:
                for(size_t j = 0; j < MAX_DIM; j ++) {
                    if(inner_m > last_seen[j] + 1) {
                        aver_weights[j] += regularizer::proximal_operator(regular, inner_weights[j], step_size
                            , lambda, inner_m - (last_seen[j] + 1), true, -step_size * full_grad[j]) / inner_m;
                    }
                }
                model->update_model(aver_weights);
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        delete[] last_seen;
        delete[] aver_weights;
        delete[] full_grad_core;
    }
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// w = A*w + Const
double lazy_update_SVRG(double& w, double Const, double A, size_t times, bool is_averaged = true) {
    if(times == 1) {
        w = A * w + Const;
        return w;
    }
    double pow_A = pow((double)A, (double)times);
    double T1 = equal_ratio(A, pow_A, times);
    double lazy_average = 0.0;
    if(is_averaged) {
        double T2 = Const / (1 - A);
        lazy_average = T1 * w + T2 * times - T1 * T2;
    }
    w = pow_A * w + Const * T1 / A;
    return lazy_average;
}

// Only Applicable for L2 regularizer
std::vector<double>* grad_desc_sparse::SVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L
    , double step_size, bool is_store_result) {
        // Random Generator
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_int_distribution<int> distribution(0, N - 1);
        std::vector<double>* stored_F = new std::vector<double>;
        double* inner_weights = new double[MAX_DIM];
        double* full_grad = new double[MAX_DIM];
        double* lambda = model->get_params();
        //FIXME: Epoch Size(SVRG / SVRG++)
        double m0 = (double) N * 2.0;
        size_t total_iterations = 0;
        copy_vec(inner_weights, model->get_model());
        // Init Weight Evaluate
        if(is_store_result)
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        // OUTTER_LOOP
        for(size_t i = 0 ; i < iteration_no; i ++) {
            double* full_grad_core = new double[N];
            // Average Iterates
            double* aver_weights = new double[MAX_DIM];
            // lazy update extra params.
            int* last_seen = new int[MAX_DIM];
            //FIXME: SVRG / SVRG++
            double inner_m = m0;//pow(2, i + 1) * m0;
            memset(aver_weights, 0, MAX_DIM * sizeof(double));
            memset(full_grad, 0, MAX_DIM * sizeof(double));
            for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
            // Full Gradient
            for(size_t j = 0; j < N; j ++) {
                full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
                for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                    full_grad[Ir[k]] += (X[k] * full_grad_core[j]) / (double) N;
                }
            }
            switch(Mode) {
                case SVRG_LAST_LAST:
                case SVRG_AVER_LAST:
                    break;
                case SVRG_AVER_AVER:
                    copy_vec(inner_weights, model->get_model());
                    break;
                default:
                    throw std::string("500 Internal Error.");
                    break;
            }
            // INNER_LOOP
            for(size_t j = 0; j < inner_m ; j ++) {
                int rand_samp = distribution(generator);
                double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N
                    , rand_samp, inner_weights);
                for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                    size_t index = Ir[k];
                    double val = X[k];
                    // lazy update
                    if((int)j > last_seen[index] + 1) {
                        switch(Mode) {
                            case SVRG_LAST_LAST:
                                lazy_update_SVRG(inner_weights[index], -step_size * full_grad[index]
                                    , 1 - step_size * lambda[0], j - (last_seen[index] + 1), false);
                                break;
                            case SVRG_AVER_LAST:
                            case SVRG_AVER_AVER:
                                aver_weights[index] += lazy_update_SVRG(inner_weights[index]
                                    , -step_size * full_grad[index], 1 - step_size * lambda[0]
                                    , j - (last_seen[index] + 1)) / inner_m;
                                break;
                            default:
                                throw std::string("500 Internal Error.");
                                break;
                        }
                    }
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + inner_weights[index]* lambda[0] + full_grad[index];
                    inner_weights[index] -= step_size * vr_sub_grad;
                    aver_weights[index] += inner_weights[index] / inner_m;
                    last_seen[index] = j;
                }
                total_iterations ++;
            }
            // lazy update aggragate
            switch(Mode) {
                case SVRG_LAST_LAST:
                    for(size_t j = 0; j < MAX_DIM; j ++) {
                        if(inner_m > last_seen[j] + 1) {
                            lazy_update_SVRG(inner_weights[j], -step_size * full_grad[j]
                                , 1 - step_size * lambda[0], inner_m - (last_seen[j] + 1), false);
                        }
                    }
                    model->update_model(inner_weights);
                    break;
                case SVRG_AVER_LAST:
                case SVRG_AVER_AVER:
                    for(size_t j = 0; j < MAX_DIM; j ++) {
                        if(inner_m > last_seen[j] + 1) {
                            aver_weights[j] += lazy_update_SVRG(inner_weights[j]
                                , -step_size * full_grad[j], 1 - step_size * lambda[0]
                                , inner_m - (last_seen[j] + 1)) / inner_m;
                        }
                    }
                    model->update_model(aver_weights);
                    break;
                default:
                    throw std::string("500 Internal Error.");
                    break;
            }
            // For Matlab (per m/n passes)
            if(is_store_result) {
                stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
            }
            delete[] last_seen;
            delete[] aver_weights;
            delete[] full_grad_core;
        }
        delete[] full_grad;
        delete[] inner_weights;
        if(is_store_result)
            return stored_F;
        return NULL;
}

////// For Katyusha With Update Option I //////
double L2_Katyusha_Y_lazy_proximal(double& _y0, double _z0, double tau_1, double tau_2
    , double lambda, double step_size_y, double alpha, double _outterx, double _F
    , size_t times, int start_iter, double compos_factor, double compos_base, double* compos_pow) {
    double lazy_average = 0.0;
    // Constants
    double prox_y = 1.0 / (1.0 + step_size_y * lambda);
    double ETA = (1.0 - tau_1 - tau_2) * prox_y;
    double M = tau_1 * prox_y * (_z0 + _F / lambda);
    double A = 1.0 / (1.0 + alpha * lambda);
    double constant = prox_y * (tau_2 * _outterx - _F * (step_size_y + tau_1 / lambda));
    double MAETA = M / (A - ETA), CONSTETA = constant / (1.0 - ETA);
    // Powers
    double pow_eta = pow((double) ETA, (double) times);
    double pow_A = pow((double) A, (double) times);

    if(start_iter == -1)
        lazy_average = 1.0 / (compos_base * compos_factor);
    else
        lazy_average = compos_pow[start_iter] / compos_base;
    lazy_average *= equal_ratio(ETA * compos_factor, pow_eta * compos_pow[times], times)* (_y0 - MAETA - CONSTETA)
                 + equal_ratio(A * compos_factor, pow_A * compos_pow[times], times)* MAETA
                 + equal_ratio(compos_factor, compos_pow[times], times) * CONSTETA;

    _y0 = pow_eta * (_y0 - MAETA - CONSTETA) + MAETA * pow_A + CONSTETA;
    return lazy_average;
}

////// For Katyusha With Update Option I //////
double Naive_Katyusha_lazy_proximal(double& _y0, double& _z0, int regular, double tau_1, double tau_2
    , double* lambda, double step_size_y, double alpha, double _outterx, double _F
    , size_t times, int start_iter, double compos_factor, double compos_base, double* compos_pow) {
    double lazy_average = 0.0;
    for(size_t i = 0; i < times; i ++) {
        double temp_y = tau_1 * _z0 + tau_2 * _outterx + (1 - tau_1 - tau_2) * _y0 - step_size_y * _F;
        _y0 = regularizer::proximal_operator(regular, temp_y, step_size_y, lambda);
        lazy_average += _y0 * (compos_pow[i + start_iter + 1] / compos_base);
        double temp_z = _z0 - alpha * _F;
        _z0 = regularizer::proximal_operator(regular, temp_z, alpha, lambda);
    }
    return lazy_average;
}

///// For Katyusha With Update Option II //////
double Naive_Katyusha_lazy_proximal2(double& _y0, double& _z0, int regular, double tau_1, double tau_2
    , double* lambda, double alpha, double _outterx, double _F, size_t times, int start_iter
    , double compos_factor, double compos_base, double m, double* compos_pow) {
    double lazy_average = 0.0;
    for(size_t i = 0; i < times; i ++) {
        double temp_z = _z0 - alpha * _F;
        _z0 = regularizer::proximal_operator(regular, temp_z, alpha, lambda);
        _y0 = tau_1 * _z0 + tau_2 * _outterx + (1 - tau_1 - tau_2) * _y0;
        switch(regular) {
            case regularizer::L2:
            case regularizer::ELASTIC_NET: // Strongly Convex Case
                lazy_average += _y0 * (compos_pow[i + start_iter + 1] / compos_base);
                break;
            case regularizer::L1: // Non-Strongly Convex Case
                lazy_average += _y0 / (double) m;
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }

    }
    return lazy_average;
}

///// For Katyusha With Update Option II //////
double Katyusha_lazy_proximal(double& _y, double& _z, int _regular, double* lambda
        , size_t times, double tau_1, double tau_2, double alpha, double _outterx
        , double _F, int start_iter, double compos_factor, double compos_base
        , double m, double* compos_pow) {
    double lazy_average_z = 0.0;
    double weighted_lazy_average_z = 0.0;
    double lazy_average_y = 0.0;
    double C = -alpha * _F;
    double S = 1 - tau_1 - tau_2;
    double pow_S = pow(S, (double) times);
    double ER_S = equal_ratio(S, pow_S, (double) times);
    // Compute Sigma{z}--lazy_average_z   Sigma{Sz}--weighted_lazy_average_z
    switch(_regular) {
        case regularizer::L1: {
            // Compute Y-related terms
            double coeff = 1.0 / (double) m;
            double XS = tau_2 * _outterx / (1.0 - S);
            lazy_average_y = coeff * (ER_S * (_y - XS) + XS * (double)times);
            double z_term_prefix = coeff * tau_1 / (1.0 - S);
            // New DnC Method
            double P = alpha * lambda[1];
            double X = _z;
            size_t K = times;
            if(C >= P || C <= -P) {
                bool flag = false;
                // Dual Case
                if(C < -P) {
                    flag = true;
                    C = -C;
                    X = -_z;
                }
                while(X < P - C && K > 0) {
                    double thres = ceil((-P - C - X) / (P + C));
                    if(K <= thres) {
                        lazy_average_z = K * X + (P + C) * (1 + K) * K / 2.0;
                        weighted_lazy_average_z = X * ER_S + (P + C) * (ER_S * S / (S - 1.0)
                                        - K * S / (S - 1.0));
                        _z = X + K * (P + C);
                        if(flag) {
                            _z = -_z;
                            lazy_average_z = -lazy_average_z;
                            weighted_lazy_average_z = -weighted_lazy_average_z;
                        }
                        // Special Case: 1 - tau_1 - tau_2 = 0
                        if(S == 0)
                            _y = tau_1 * _z + tau_2 * _outterx;
                        else
                            _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;
                        return lazy_average_y + z_term_prefix * (lazy_average_z - weighted_lazy_average_z);
                    }
                    else if(thres > 0.0){
                        double ER_St = equal_ratio(S, pow(S, thres), thres);
                        lazy_average_z += thres * X + (P + C) * (1 + thres) * thres / 2.0;
                        weighted_lazy_average_z = (X * ER_St + (P + C) * (ER_St * S / (S - 1.0)
                                        - thres * S / (S - 1.0))) * pow(S, (double) times - thres);
                        X += thres * (P + C);
                        K -= thres;
                    }
                    lazy_average_z += regularizer::L1_single_step(X, P, C, true);
                    weighted_lazy_average_z += pow(S, (double) K) * X;
                    K --;
                }
                if(K == 0) {
                    _z = X;
                    if(flag) {
                        _z = -_z;
                        lazy_average_z = -lazy_average_z;
                        weighted_lazy_average_z = -weighted_lazy_average_z;
                    }
                    // Special Case: 1 - tau_1 - tau_2 = 0
                    if(S == 0)
                        _y = tau_1 * _z + tau_2 * _outterx;
                    else
                        _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;
                    return lazy_average_y + z_term_prefix * (lazy_average_z - weighted_lazy_average_z);
                }
                _z = X + K * (C - P);
                lazy_average_z += K * X + (C - P) * (1 + K) * K / 2.0;
                double ER_S2 = equal_ratio(S, pow(S, (double) K), (double) K);
                weighted_lazy_average_z = X * ER_S2 + (C - P) * (ER_S2 * S / (S - 1.0)
                                        - K * S / (S - 1.0));
                if(flag) {
                    lazy_average_z = -lazy_average_z;
                    weighted_lazy_average_z = -weighted_lazy_average_z;
                    _z = -_z;
                }
                // Special Case: 1 - tau_1 - tau_2 = 0
                if(S == 0)
                    _y = tau_1 * _z + tau_2 * _outterx;
                else
                    _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * (1.0 - pow_S) / (1.0 - S);

                return lazy_average_y + z_term_prefix * (lazy_average_z - weighted_lazy_average_z);
            }
            else {
                double thres_1 = max(ceil((P - C - X) / (C - P)), 0.0);
                double thres_2 = max(ceil((-P - C - X) / (C + P)), 0.0);
                if(thres_2 == 0 && thres_1 == 0) {
                    _z = 0;
                    if(S == 0)
                        _y = tau_2 * _outterx;
                    else
                        _y = pow_S * _y + tau_2 * _outterx * ER_S / S;
                    return lazy_average_y;
                }
                else if(K > thres_1 && K > thres_2) {
                    _z = 0;
                    if(thres_1 != 0.0) {
                        double ER_St = equal_ratio(S, pow(S, thres_1), thres_1);
                        lazy_average_z = thres_1 * X + (C - P) * (1 + thres_1) * thres_1 / 2.0;
                        weighted_lazy_average_z = (X * ER_St + (C - P) * (ER_St * S / (S - 1.0)
                                        - thres_1 * S / (S - 1.0))) * pow(S, (double) times - thres_1);
                    }
                    else {
                        double ER_St = equal_ratio(S, pow(S, thres_2), thres_2);
                        lazy_average_z = thres_2 * X + (P + C) * (1 + thres_2) * thres_2 / 2.0;
                        weighted_lazy_average_z = (X * ER_St + (P + C) * (ER_St * S / (S - 1.0)
                                        - thres_2 * S / (S - 1.0))) * pow(S, (double) times - thres_2);
                    }
                }
                else {
                    if(X > 0) {
                        lazy_average_z = K * X + (C - P) * (1 + K) * K / 2.0;
                        weighted_lazy_average_z = X * ER_S + (C - P) * (ER_S * S / (S - 1.0)
                                        - K * S / (S - 1.0));
                        _z = X + K * (C - P);
                    }
                    else {
                        lazy_average_z = K * X + (P + C) * (1 + K) * K / 2.0;
                        weighted_lazy_average_z = X * ER_S + (P + C) * (ER_S * S / (S - 1.0)
                                        - K * S / (S - 1.0));
                        _z = X + K * (P + C);
                    }
                }
                // Special Case: 1 - tau_1 - tau_2 = 0
                if(S == 0)
                    _y = tau_1 * _z + tau_2 * _outterx;
                else
                    _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;

                return lazy_average_y + z_term_prefix * (lazy_average_z - weighted_lazy_average_z);
            }
            break;
        }
        case regularizer::L2: {
            double coeff;
            if(start_iter == -1)
                coeff = 1.0 / (compos_base * compos_factor);
            else
                coeff = compos_pow[start_iter] / compos_base;
            if(times == 1) {
                _z = (_z + C) / (1 + alpha * lambda[0]);
                _y = S * _y + tau_1 * _z + tau_2 * _outterx;
                return coeff * compos_factor * _y;
            }
            // Compute Y-related terms
            double XS = tau_2 * _outterx / (1.0 - S);
            lazy_average_y = coeff * (equal_ratio(compos_factor * S, pow_S * compos_pow[times], times)
                                * (_y - XS) + equal_ratio(compos_factor, compos_pow[times], times)
                                * XS);

            // Compute Z-related terms
            double A = 1.0 / (1.0 + alpha * lambda[0]);
            double CA = C * A / (1.0 - A);
            double pow_A = pow(A, (double) times);

            lazy_average_z = equal_ratio(compos_factor * A, pow_A * compos_pow[times], times)
                        * (_z - CA) + equal_ratio(compos_factor, compos_pow[times], times) * CA;
            weighted_lazy_average_z = pow_A * A * equal_ratio(S / A, pow_S / pow_A, times)
                        * (_z - CA) +  equal_ratio(S, pow_S, times) * CA;
            _z = pow_A * (_z - CA) + CA;
            // Special Case: 1 - tau_1 - tau_2 = 0
            if(S == 0)
                _y = tau_1 * _z + tau_2 * _outterx;
            else
                _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;
            return lazy_average_y + coeff * tau_1 / (1.0 - compos_factor * S)
                    * (lazy_average_z - compos_pow[times] * compos_factor * weighted_lazy_average_z);
            break;
        }
        case regularizer::ELASTIC_NET: {
            double coeff;
            if(start_iter == -1)
                coeff = 1.0 / (compos_base * compos_factor);
            else
                coeff = compos_pow[start_iter] / compos_base;
            // Compute Y-related terms
            double XS = tau_2 * _outterx / (1.0 - S);
            double z_term_prefix = coeff * tau_1 / (1.0 - compos_factor * S);
            lazy_average_y = coeff * (equal_ratio(compos_factor * S, pow_S * compos_pow[times], times)
                                * (_y - XS) + equal_ratio(compos_factor, compos_pow[times], times)
                                * XS);
            // New DnC Method
            double P = alpha * lambda[1];
            double Q = 1.0 / (1.0 + alpha * lambda[0]);
            double X = _z;
            size_t K = times;
            if(C >= P || C <= -P) {
                bool flag = false;
                // Dual Case
                if(C < -P) {
                    flag = true;
                    C = -C;
                    X = -_z;
                }
                double ratio_PCQ = (P + C) * Q / (1 - Q);
                double ratio_NPCQ = (P - C) * Q / (1 - Q);
                while(X < P - C && K > 0) {
                    double thres = ceil(log((P + C + ratio_PCQ) / (ratio_PCQ - X)) / log(Q));
                    if(K <= thres) {
                        double pow_QK = pow((double) Q, (double) K);
                        lazy_average_z = equal_ratio(compos_factor * Q, pow_QK * compos_pow[K], K)
                                    * (X - ratio_PCQ) + ratio_PCQ * equal_ratio(compos_factor, compos_pow[K], K);
                        weighted_lazy_average_z = pow_QK * Q * equal_ratio(S / Q, pow_S / pow_QK, K)
                                    * (X - ratio_PCQ) + ratio_PCQ * ER_S;
                        _z = pow_QK * X + ratio_PCQ * (1 - pow_QK);
                        if(flag) {
                            _z = -_z;
                            lazy_average_z = -lazy_average_z;
                            weighted_lazy_average_z = -weighted_lazy_average_z;
                        }

                        // Special Case: 1 - tau_1 - tau_2 = 0
                        if(S == 0)
                            _y = tau_1 * _z + tau_2 * _outterx;
                        else
                            _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;
                        return lazy_average_y + z_term_prefix * (lazy_average_z
                                    - compos_pow[K] * compos_factor * weighted_lazy_average_z);
                    }
                    else if(thres > 0.0){
                        double pow_Qtrs = pow((double) Q, (double) thres);
                        lazy_average_z = equal_ratio(compos_factor * Q, pow_Qtrs * compos_pow[(int)thres], thres)
                                     * (X - ratio_PCQ) + ratio_PCQ * equal_ratio(compos_factor, compos_pow[(int)thres], thres);
                        weighted_lazy_average_z = pow_Qtrs * Q * equal_ratio(S / Q, pow(S, thres) / pow_Qtrs, thres)
                                     * (X - ratio_PCQ) + ratio_PCQ * equal_ratio(S, pow(S, thres), thres);
                        weighted_lazy_average_z *= pow(S, (double) times - thres);
                        X = pow_Qtrs * X + ratio_PCQ * (1 - pow_Qtrs);
                        K -= thres;
                    }
                    lazy_average_z += compos_pow[K] * regularizer::EN_single_step(X, P, Q, C, true);
                    weighted_lazy_average_z += pow(S, (double) K) * X;
                    K --;
                }
                if(K == 0) {
                    _z = X;
                    if(flag) {
                        _z = -_z;
                        lazy_average_z = -lazy_average_z;
                        weighted_lazy_average_z = -weighted_lazy_average_z;
                    }
                    // Special Case: 1 - tau_1 - tau_2 = 0
                    if(S == 0)
                        _y = tau_1 * _z + tau_2 * _outterx;
                    else
                        _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;

                    return lazy_average_y + z_term_prefix * (lazy_average_z
                                - compos_pow[K] * compos_factor * weighted_lazy_average_z);
                }
                double pow_QK = pow((double) Q, (double) K);
                double pow_S2 = pow(S, (double)K);
                _z = pow_QK * X - ratio_NPCQ * (1 - pow_QK);
                lazy_average_z = equal_ratio(compos_factor * Q, pow_QK * compos_pow[K], K)
                            * (X + ratio_NPCQ) - ratio_NPCQ * equal_ratio(compos_factor, compos_pow[K], K);
                weighted_lazy_average_z = pow_QK * Q * equal_ratio(S / Q, pow_S2 / pow_QK, K)
                            * (X + ratio_NPCQ) - ratio_NPCQ * equal_ratio(S, pow_S2, K);
                if(flag) {
                    lazy_average_z = -lazy_average_z;
                    weighted_lazy_average_z = -weighted_lazy_average_z;
                    _z = -_z;
                }
                // Special Case: 1 - tau_1 - tau_2 = 0
                if(S == 0)
                    _y = tau_1 * _z + tau_2 * _outterx;
                else
                    _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;
                return lazy_average_y + z_term_prefix * (lazy_average_z
                            - compos_pow[K] * compos_factor * weighted_lazy_average_z);
            }
            else {
                double ratio_PCQ = (P + C) * Q / (1 - Q);
                double ratio_NPCQ = (P - C) * Q / (1 - Q);
                double thres_1 = max(ceil(log((P - C + ratio_NPCQ) / (ratio_NPCQ + X)) / log(Q)), 0.0); // P-C
                double thres_2 = max(ceil(log((P + C + ratio_PCQ) / (ratio_PCQ - X)) / log(Q)), 0.0); // -P-C
                if(thres_2 == 0 && thres_1 == 0) {
                    _z = 0;
                    if(S == 0)
                        _y = tau_2 * _outterx;
                    else
                        _y = pow_S * _y + tau_2 * _outterx * ER_S / S;
                    return lazy_average_y;
                }
                else if(K > thres_1 && K > thres_2) {
                    _z = 0;
                    double pow_Qtrs1 = pow((double) Q, (double) thres_1);
                    double pow_Qtrs2 = pow((double) Q, (double) thres_2);
                    if(thres_1 != 0.0) {
                        double pow_S1 = pow(S, thres_1);
                        lazy_average_z = equal_ratio(compos_factor * Q, pow_Qtrs1 * compos_pow[(int)thres_1], thres_1)
                                 * (X + ratio_NPCQ) - ratio_NPCQ * equal_ratio(compos_factor, compos_pow[(int)thres_1], thres_1);
                        weighted_lazy_average_z = pow_Qtrs1 * Q * equal_ratio(S / Q, pow_S1 / pow_Qtrs1, thres_1)
                                 * (X + ratio_NPCQ) - ratio_NPCQ * equal_ratio(S, pow_S1, thres_1);
                        weighted_lazy_average_z *= pow(S, (double) times - thres_1);
                    }
                    else {
                        double pow_S2 = pow(S, thres_2);
                        lazy_average_z = equal_ratio(compos_factor * Q, pow_Qtrs2 * compos_pow[(int)thres_2], thres_2)
                                 * (X - ratio_PCQ) + ratio_PCQ * equal_ratio(compos_factor, compos_pow[(int)thres_2], thres_2);
                        weighted_lazy_average_z = pow_Qtrs2 * Q * equal_ratio(S / Q, pow_S2 / pow_Qtrs2, thres_2)
                                 * (X - ratio_PCQ) + ratio_PCQ * equal_ratio(S, pow_S2, thres_2);
                        weighted_lazy_average_z *= pow(S, (double) times - thres_2);
                    }
                }
                else {
                    double pow_QK = pow((double) Q, (double) K);
                    if(X > 0) {
                        lazy_average_z = equal_ratio(compos_factor * Q, pow_QK * compos_pow[K], K)
                                    * (X + ratio_NPCQ) - ratio_NPCQ * equal_ratio(compos_factor, compos_pow[K], K);
                        weighted_lazy_average_z = pow_QK * Q * equal_ratio(S / Q, pow_S / pow_QK, K)
                                    * (X + ratio_NPCQ) - ratio_NPCQ * ER_S;
                        _z = pow_QK * X - ratio_NPCQ * (1 - pow_QK);
                    }
                    else {
                        lazy_average_z = equal_ratio(compos_factor * Q, pow_QK * compos_pow[K], K)
                                    * (X - ratio_PCQ) + ratio_PCQ * equal_ratio(compos_factor, compos_pow[K], K);
                        weighted_lazy_average_z = pow_QK * Q * equal_ratio(S / Q, pow_S / pow_QK, K)
                                    * (X - ratio_PCQ) + ratio_PCQ * ER_S;
                        _z = pow_QK * X + ratio_PCQ * (1 - pow_QK);
                    }
                }
                // Special Case: 1 - tau_1 - tau_2 = 0
                if(S == 0)
                    _y = tau_1 * _z + tau_2 * _outterx;
                else
                    _y = pow_S * _y + tau_1 * weighted_lazy_average_z / S + tau_2 * _outterx * ER_S / S;
                return lazy_average_y + z_term_prefix * (lazy_average_z
                            - compos_pow[K] * compos_factor * weighted_lazy_average_z);
            }
            break;
        }
        default:
            return 0.0;
            break;
    }
}

std::vector<double>* grad_desc_sparse::Katyusha(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double sigma
    , double step_size, bool is_store_result) {
    // Random Generator
    std::vector<double>* stored_F = new std::vector<double>;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    size_t m = 2.0 * N;
    size_t total_iterations = 0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    double tau_2 = 0.5, tau_1 = 0.5;
    if(sqrt(sigma * m / (3.0 * L)) < 0.5) tau_1 = sqrt(sigma * m / (3.0 * L));
    double alpha = 1.0 / (tau_1 * 3.0 * L);
    ///// For Katyusha With Update Option I //////
    double step_size_y = 1.0 / (3.0 * L);
    double compos_factor = 1.0 + alpha * sigma;
    double compos_base = (pow((double)compos_factor, (double)m) - 1.0) / (alpha * sigma);
    double* compos_pow = new double[m + 1];
    for(size_t i = 0; i <= m; i ++)
        compos_pow[i] = pow((double)compos_factor, (double)i);
    double* y = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_weights = (model->get_model());
        double* aver_weights = new double[MAX_DIM];
        // lazy update extra param
        double* last_seen = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
        switch(regular) {
            case regularizer::L2:
            case regularizer::ELASTIC_NET: // Strongly Convex Case
                break;
            case regularizer::L1: // Non-Strongly Convex Case
                tau_1 = 2.0 / ((double) i + 4.0);
                alpha = 1.0 / (tau_1 * 3.0 * L);
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
            }
        }
        // 0th Inner Iteration
        for(size_t k = 0; k < MAX_DIM; k ++)
            inner_weights[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                             + (1 - tau_1 - tau_2) * y[k];
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, inner_weights);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                double val = X[k];
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    ////// For Katyusha With Update Option I //////
                    // if(regular == regularizer::L2) {
                    //     aver_weights[index] += L2_Katyusha_Y_lazy_proximal(y[index], z[index]
                    //         , tau_1, tau_2, lambda[0], step_size_y, alpha, outter_weights[index]
                    //         , full_grad[index], j - (last_seen[index] + 1), last_seen[index]
                    //         , compos_factor, compos_base, compos_pow);
                    //     regularizer::proximal_operator(regular, z[index], alpha, lambda, j - (last_seen[index] + 1), false
                    //          , -alpha * full_grad[index]);
                    // }
                    // else
                    //     aver_weights[index] += Naive_Katyusha_lazy_proximal(y[index], z[index], regular
                    //         , tau_1, tau_2, lambda, step_size_y, alpha, outter_weights[index]
                    //         , full_grad[index], j - (last_seen[index] + 1), last_seen[index]
                    //         , compos_factor, compos_base, compos_pow);

                    ////// For Katyusha With Update Option II //////
                    // aver_weights[index] += Naive_Katyusha_lazy_proximal2(y[index], z[index], regular
                    //         , tau_1, tau_2, lambda, alpha, outter_weights[index], full_grad[index]
                    //         , j - (last_seen[index] + 1), last_seen[index]
                    //         , compos_factor, compos_base, m, compos_pow);
                    aver_weights[index] += Katyusha_lazy_proximal(y[index], z[index], regular
                            , lambda, j - (last_seen[index] + 1), tau_1, tau_2, alpha
                            , outter_weights[index], full_grad[index], last_seen[index]
                            , compos_factor, compos_base, m, compos_pow);
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
                }
                double katyusha_grad = full_grad[index] + val * (inner_core - full_grad_core[rand_samp]);
                double prev_z = z[index];
                z[index] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[index], alpha, lambda);

                ////// For Katyusha With Update Option I //////
                // y[index] = inner_weights[index] - step_size_y * katyusha_grad;
                // regularizer::proximal_operator(regular, y[index], step_size_y, lambda);

                ////// For Katyusha With Update Option II //////
                y[index] = inner_weights[index] + tau_1 * (z[index] - prev_z);
                switch(regular) {
                    case regularizer::L2:
                    case regularizer::ELASTIC_NET: // Strongly Convex Case
                        aver_weights[index] += compos_pow[j] / compos_base * y[index];
                        break;
                    case regularizer::L1: // Non-Strongly Convex Case
                        aver_weights[index] += y[index] / m;
                        break;
                    default:
                        throw std::string("500 Internal Error.");
                        break;
                }
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
                last_seen[index] = j;
            }
            total_iterations ++;
        }
        // lazy update aggragate
        for(size_t j = 0; j < MAX_DIM; j ++) {
            if(m > last_seen[j] + 1) {
                ////// For Katyusha With Update Option I //////
                // if(regular == regularizer::L2) {
                //     aver_weights[j] += L2_Katyusha_Y_lazy_proximal(y[j], z[j]
                //         , tau_1, tau_2, lambda[0], step_size_y, alpha, outter_weights[j]
                //         , full_grad[j], m - (last_seen[j] + 1), last_seen[j]
                //         , compos_factor, compos_base, compos_pow);
                //     regularizer::proximal_operator(regular, z[j], alpha, lambda, m - (last_seen[j] + 1), false
                //          , -alpha * full_grad[j]);
                // }
                // else
                //     aver_weights[j] += Naive_Katyusha_lazy_proximal(y[j], z[j], regular
                //         , tau_1, tau_2, lambda, step_size_y, alpha, outter_weights[j]
                //         , full_grad[j], m - (last_seen[j] + 1), last_seen[j]
                //         , compos_factor, compos_base, compos_pow);

                ////// For Katyusha With Update Option II //////
                // aver_weights[j] += Naive_Katyusha_lazy_proximal2(y[j], z[j], regular
                //             , tau_1, tau_2, lambda, alpha, outter_weights[j]
                //             , full_grad[j], m - (last_seen[j] + 1), last_seen[j]
                //             , compos_factor, compos_base, m, compos_pow);
                aver_weights[j] += Katyusha_lazy_proximal(y[j], z[j], regular
                            , lambda, m - (last_seen[j] + 1), tau_1, tau_2, alpha
                            , outter_weights[j], full_grad[j], last_seen[j]
                            , compos_factor, compos_base, m, compos_pow);
                inner_weights[j] = tau_1 * z[j] + tau_2 * outter_weights[j]
                                 + (1 - tau_1 - tau_2) * y[j];
            }
        }
        model->update_model(aver_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        delete[] last_seen;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
    }
    delete[] y;
    delete[] z;
    delete[] inner_weights;
    delete[] full_grad;
    delete[] compos_pow;
    if(is_store_result)
        return stored_F;
    return NULL;
}
