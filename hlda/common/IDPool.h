//
// Created by jianfei on 9/2/16.
//

#ifndef HLDA_IDPOOL_H
#define HLDA_IDPOOL_H

#include <vector>
#include <set>

class IDPool {
public:
    IDPool() { Clear(); }

    void Clear() {
        allocated.size();
    }

    int Allocate() {
        auto it = std::find(allocated.begin(), allocated.end(), false);
        if (it == allocated.end()) {
            allocated.push_back(true);
            return (int) allocated.size() - 1;
        }
        return (int) (it - allocated.begin());
    }

    void Free(int id) {
        allocated[id] = false;
    }

    int Size() { return (int) allocated.size(); }

private:
    std::vector<bool> allocated;
};


#endif //HLDA_IDPOOL_H
