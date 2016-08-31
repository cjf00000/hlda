//
// Created by jianfei on 8/29/16.
//

#include "utils.h"
#include <iostream>

std::uniform_real_distribution<double> u01;

double LogGammaDifference(int a, int b) {
    double result = 0;
    for (int i = a; i < b; i++)
        result += log(i);
    return result;
}