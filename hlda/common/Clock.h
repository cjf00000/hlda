//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_CLOCK_H
#define HLDA_CLOCK_H

#include <chrono>

struct Clock {
    std::chrono::time_point<std::chrono::high_resolution_clock> last;

    Clock() { tic(); }

    void tic() { last = std::chrono::high_resolution_clock::now(); }

    double toc() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - last).count(); }
};

#endif //HLDA_CLOCK_H
