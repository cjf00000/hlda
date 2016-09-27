//
// Created by jianfei on 8/29/16.
//

#ifndef FAST_HLDA2_UTILS_H
#define FAST_HLDA2_UTILS_H

#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>

extern std::uniform_real_distribution<double> u01;

template<class TIterator, class TGenerator>
int DiscreteSample(TIterator begin, TIterator end, TGenerator &generator) {
    if (begin == end)
        throw std::runtime_error("Incorrect range for DiscreteSample");

    double prob_sum = 0;
    for (auto it = begin; it != end; it++) prob_sum += *it;

    double u = u01(generator) * prob_sum;
    for (auto it = begin; it != end; it++) {
        u -= *it;
        if (u <= 0)
            return it - begin;
    }
    return (end - begin) - 1;
};

template<class TIterator>
void Softmax(TIterator begin, TIterator end) {
    double maximum = *std::max_element(begin, end);
    double sum = 0;
    for (auto it = begin; it != end; it++) {
        *it = exp(*it - maximum);
        sum += *it;
    }
    double inv_sum = 1. / sum;
    for (auto it = begin; it != end; it++)
        *it *= inv_sum;
}

// lgamma(start+len) - lgamma(start)
extern double LogGammaDifference(double start, int len);

extern double LogSum(double log_a, double log_b);

template<class T>
class beta_distribution {
public:
    beta_distribution(T alpha, T beta) :
            gam1(alpha), gam2(beta) {}

    template<class TGenerator>
    T operator()(TGenerator &generator) {
        T a = gam1(generator);
        T b = gam2(generator);
        return a / (a + b);
    }

private:
    std::gamma_distribution<T> gam1, gam2;
};

template<class T>
class dirichlet_distribution {
public:
    dirichlet_distribution(std::vector<T> &prob) {
        gammas.resize(prob.size());
        for (size_t i = 0; i < prob.size(); i++)
            gammas[i] = std::gamma_distribution<T>(prob[i]);
    }

    template<class TGenerator>
    std::vector<T> operator()(TGenerator &generator) {
        std::vector<T> result(gammas.size());
        T sum = 0;
        for (size_t n = 0; n < gammas.size(); n++)
            sum += result[n] = gammas[n](generator);
        for (auto &r: result)
            r /= sum;
        return std::move(result);
    }

private:
    std::vector<std::gamma_distribution<T>> gammas;
};

#endif //FAST_HLDA2_UTILS_H
