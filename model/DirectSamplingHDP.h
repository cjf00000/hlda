//
// Created by jianfei on 9/2/16.
//

#ifndef HLDA_DIRECTSAMPLINGHDP_H
#define HLDA_DIRECTSAMPLINGHDP_H

#include <vector>
#include "types.h"
#include "Matrix.h"


class DirectSamplingHDP {
    struct Document {

    };

private:
    Matrix<TCount> cwk;
    std::vector<TCount> ck;

    std::vector<TProb> beta;

};


#endif //HLDA_DIRECTSAMPLINGHDP_H
