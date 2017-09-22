#include <iostream>
#include "mex.h"
#include "grad_desc_sparse.hpp"
#include "grad_desc_dense.hpp"
#include "svm.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"
#include <string.h>

size_t MAX_DIM;
const size_t MAX_PARAM_STR_LEN = 15;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        double *X = mxGetPr(prhs[0]);
        double *Y = mxGetPr(prhs[1]);
        MAX_DIM = mxGetM(prhs[0]);
        size_t N = mxGetN(prhs[0]);
        bool is_sparse = mxGetLogicals(prhs[10])[0];
        size_t* Jc;
        size_t* Ir;
        if(is_sparse) {
            Jc = mxGetJc(prhs[0]);
            Ir = mxGetIr(prhs[0]);
        }
        double *init_weight = mxGetPr(prhs[5]);
        double lambda1 = mxGetScalar(prhs[6]);
        double lambda2 = mxGetScalar(prhs[13]);
        double L = mxGetScalar(prhs[7]);
        double sigma = mxGetScalar(prhs[12]);
        double step_size = mxGetScalar(prhs[8]);
        // For SVRG
        int Mode = (int) mxGetScalar(prhs[11]);
        size_t iteration_no = (size_t) mxGetScalar(prhs[9]);
        bool is_store_result = false;
        if(nlhs == 1)
            is_store_result = true;

        int regularizer;
        char* _regul = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[4], _regul, MAX_PARAM_STR_LEN);
        if(strcmp(_regul, "L2") == 0) {
            regularizer = regularizer::L2;
        }
        else if(strcmp(_regul, "L1") == 0) {
            regularizer = regularizer::L1;
        }
        else if(strcmp(_regul, "elastic_net") == 0) {
            regularizer = regularizer::ELASTIC_NET;
        }
        else mexErrMsgTxt("400 Unrecognized regularizer.");
        delete[] _regul;

        blackbox* model;
        char* _model = new char[MAX_PARAM_STR_LEN];
        double lambdas[2] = {lambda1, lambda2};
        mxGetString(prhs[3], _model, MAX_PARAM_STR_LEN);
        if(strcmp(_model, "logistic") == 0) {
            model = new logistic(2, lambdas, regularizer);
        }
        else if(strcmp(_model, "least_square") == 0) {
            model = new least_square(2, lambdas, regularizer);
        }
        else if(strcmp(_model, "svm") == 0) {
            model = new svm(2, lambdas, regularizer);
        }
        else mexErrMsgTxt("400 Unrecognized model.");
        delete[] _model;
        model->set_init_weights(init_weight);

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[2], _algo, MAX_PARAM_STR_LEN);
        double* stored_F;
        std::vector<double>* vec_stored_F;
        size_t len_stored_F;
        if(strcmp(_algo, "GD") == 0) {
            if(regularizer == regularizer::L1)
                mexErrMsgTxt("405 GD not applicable to L1 regularizer.");
            if(is_sparse)
                stored_F = grad_desc_sparse::GD(X, Y, Jc, Ir, N, model, iteration_no, L, step_size,
                    is_store_result);
            else
                stored_F = grad_desc_dense::GD(X, Y, N, model, iteration_no, L, step_size,
                    is_store_result);
            len_stored_F = iteration_no;
        }
        else if(strcmp(_algo, "SGD") == 0) {
            if(is_sparse)
                stored_F = grad_desc_sparse::SGD(X, Y, Jc, Ir, N, model, iteration_no, L, step_size,
                    is_store_result);
            else
                stored_F = grad_desc_dense::SGD(X, Y, N, model, iteration_no, L, step_size,
                    is_store_result);
            len_stored_F = (size_t) floor((double) iteration_no / N) + 1;
        }
        else if(strcmp(_algo, "SAGA") == 0) {
            if(is_sparse)
                vec_stored_F = grad_desc_sparse::SAGA(X, Y, Jc, Ir, N, model, iteration_no, L, step_size,
                    is_store_result);
            else
                vec_stored_F = grad_desc_dense::SAGA(X, Y, N, model, iteration_no, L, step_size,
                    is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else if(strcmp(_algo, "Prox_SVRG") == 0) {
            if(is_sparse)
                vec_stored_F = grad_desc_sparse::Prox_SVRG(X, Y, Jc, Ir, N, model, iteration_no, Mode, L, step_size,
                    is_store_result);
            else
                vec_stored_F = grad_desc_dense::Prox_SVRG(X, Y, N, model, iteration_no, Mode, L, step_size,
                    is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else if(strcmp(_algo, "SVRG") == 0) {
            if(regularizer != regularizer::L2)
                mexErrMsgTxt("405 SVRG not applicable to non-differentiable regularizer.");
            if(is_sparse)
                vec_stored_F = grad_desc_sparse::SVRG(X, Y, Jc, Ir, N, model, iteration_no, Mode, L, step_size,
                    is_store_result);
            else
                vec_stored_F = grad_desc_dense::SVRG(X, Y, N, model, iteration_no, Mode, L, step_size,
                    is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else if(strcmp(_algo, "Katyusha") == 0) {
            if(is_sparse)
                vec_stored_F = grad_desc_sparse::Katyusha(X, Y, Jc, Ir, N, model, iteration_no, L, sigma, step_size,
                    is_store_result);
            else
                vec_stored_F = grad_desc_dense::Katyusha(X, Y, N, model, iteration_no, L, sigma, step_size,
                    is_store_result);
            stored_F = &(*vec_stored_F)[0];
            len_stored_F = vec_stored_F->size();
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");
        delete[] _algo;

        if(is_store_result) {
            plhs[0] = mxCreateDoubleMatrix(len_stored_F, 1, mxREAL);
        	double* res_stored_F = mxGetPr(plhs[0]);
            for(size_t i = 0; i < len_stored_F; i ++)
                res_stored_F[i] = stored_F[i];
        }
        delete[] stored_F;
        delete model;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
