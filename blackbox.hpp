#ifndef BLACK_BOX_H
#define BLACK_BOX_H
#include <random>

extern size_t MAX_DIM;
class blackbox {
public:
    virtual ~blackbox();
    virtual int classify(double* sample) const = 0;

    virtual double zero_oracle_dense(double* X, double* Y, size_t N, double* weights = NULL) const;
    virtual double zero_oracle_sparse(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, double* weights = NULL) const;
    virtual double zero_component_oracle_dense(double* X, double* Y, size_t N, double* weights = NULL) const = 0;
    virtual double zero_component_oracle_sparse(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, double* weights = NULL) const = 0;
    virtual double zero_regularizer_oracle(double* weights = NULL) const;

    virtual double first_component_oracle_core_dense(double* X, double* Y, size_t N, int given_index, double* weights = NULL) const = 0;
    virtual double first_component_oracle_core_sparse(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, int given_index, double* weights = NULL) const = 0;
    virtual void first_regularizer_oracle(double* _pR, double* weights = NULL) const;

    void set_init_weights(double* init_weights);
    double* get_model() const;
    double* get_params() const;
    int get_regularizer() const;
    void update_model(double* new_weights);
protected:
    int m_regularizer;
    double* m_weights;
    double* m_params;
};


#endif
