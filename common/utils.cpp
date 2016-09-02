//
// Created by jianfei on 8/29/16.
//

#include "utils.h"
#include <iostream>

std::uniform_real_distribution<double> u01;

double LogGammaDifference(double start, int len) {
    double result = 0;
    for (int i = 0; i < len; i++)
        result += log(start + i);
    return result;
}