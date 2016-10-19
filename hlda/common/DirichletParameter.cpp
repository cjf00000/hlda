//
// Created by jianfei on 8/30/16.
//

#include "DirichletParameter.h"

void DirichletParameter::Set(std::vector<TProb> &baseMeasure, TProb concentration) {
    TProb sum = std::accumulate(baseMeasure.begin(), baseMeasure.end(), (TProb) 0);
    TProb inv_sum = 1. / sum;
    TProb factor = inv_sum * concentration;
    alpha.resize(baseMeasure.size());
    for (size_t i = 0; i < baseMeasure.size(); i++)
        alpha[i] = baseMeasure[i] * factor;
}