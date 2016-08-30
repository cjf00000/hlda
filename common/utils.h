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


#endif //FAST_HLDA2_UTILS_H
