#ifndef UTILS_HPP
#define UTILS_HPP
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

extern size_t MAX_DIM;
inline double comp_l2_norm(std::vector<double>* vec) {
    double res = 0.0;
    for(std::vector<double>::iterator iter = (*vec).begin(); iter != (*vec).end()
    ; iter ++) {
        res += (*iter) * (*iter);
    }
    return sqrt(res);
}

inline double comp_l2_norm(double* vec) {
    double res = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++){
        res += vec[i] * vec[i];
    }
    return sqrt(res);
}

inline double comp_l1_norm(double* vec) {
    double res = 0.0;
    for(size_t i = 0; i < MAX_DIM; i ++){
        res += std::abs(vec[i]);
    }
    return res;
}

inline double equal_ratio(double p, double pow_term, double times) {
    if(p == 1.0) // Degenerate Case.
        return p * times;
    else
        return p * (1 - pow_term) / (1 - p);
}

inline void copy_vec(double* vec_to, double* vec_from) {
    for(size_t i = 0; i < MAX_DIM; i ++)
        vec_to[i] = vec_from[i];
}

inline double max(double a, double b) {
    return a > b ? a : b;
}

inline double min(double a, double b) {
    return a < b ? a : b;
}


inline std::vector<std::string> split(const std::string &s, const std::string &separator) {
    std::vector<std::string> result;
    std::string temp_s = s;
    size_t found = 0;
    while(1) {
        found = temp_s.find(separator);
        result.push_back(temp_s.substr(0, found));
        if(found >= temp_s.size() - separator.size())
            break;
        temp_s = temp_s.substr(found + separator.size());
    }
    return result;
}

// inline constexpr unsigned int _hash(const char* str, int h = 0) {
//     return !str[h] ? 5381 : (_hash(str, h+1) * 33) ^ str[h];
// }

#endif
