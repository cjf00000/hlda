//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_DIRICHLETPARAMETER_H
#define HLDA_DIRICHLETPARAMETER_H

#include <vector>
#include <algorithm>
#include "types.h"

class DirichletParameter {
public:
    DirichletParameter(int N, TProb concentration) : alpha(N, concentration / N) {}

    void Set(std::vector<TProb> &baseMeasure, TProb concentration);

    double Concentration() { return std::accumulate(alpha.begin(), alpha.end(), (TProb) 0); }

    TProb &operator()(int index) { return alpha[index]; }

private:
    std::vector<TProb> alpha;
};


#endif //HLDA_DIRICHLETPARAMETER_H
