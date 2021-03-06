//
// Created by jianfei on 16-11-1.
//

#include "utils.h"

std::uniform_real_distribution<double> u01;

double LogGammaDifference(double start, int len) {
    double result = 0;
    for (int i = 0; i < len; i++)
        result += log(start + i);
    return result;
}

extern float LogSum(float log_a, float log_b) {
    if (log_a > log_b) std::swap(log_a, log_b);
    return log_b + logf(expf(log_a - log_b) + 1);
}

int bsr(int x)
{
    return _bit_scan_reverse(x);
}
