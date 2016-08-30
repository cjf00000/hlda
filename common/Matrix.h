//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_MATRIX_H
#define HLDA_MATRIX_H

#include <vector>
#include <algorithm>
#include <memory.h>

// An matrix with variable number of rows
template<class T>
class Matrix {
public:
    Matrix(int R, int C) : R(R), C(C) {}

    Matrix(int C) : R(0), C(C) {}

    void Resize(int new_R) {
        if (new_R > R) {
            std::vector<T> old_data = std::move(data);
            while (R < new_R) R = R * 2 + 1;
            data.resize(R * C);
            std::copy(old_data.begin(), old_data.end(), data.begin());
        }
    }

    T &operator()(int r, int c) {
        return data[r * C + c];
    }

    T *RowPtr(int r) {
        return &data[r * C];
    }

    T *Data() {
        return data.data();
    }

    void Clear() {
        memset(data.data(), 0, sizeof(T) * R * C);
    }

private:
    int R;
    int C;
    std::vector<T> data;
};


#endif //HLDA_MATRIX_H
