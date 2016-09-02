//
// Created by jianfei on 9/2/16.
//

#ifndef HLDA_IDPOOL_H
#define HLDA_IDPOOL_H

#include <vector>

class IDPool {
public:
    IDPool() { num_ids = 0; }

    int Allocate() {
        if (!free_ids.empty()) {
            int id = free_ids.back();
            free_ids.pop_back();
            return id;
        } else {
            return num_ids++;
        }
    }

    void Free(int id) {
        free_ids.push_back(id);
    }

    int Size() { return num_ids; }

private:
    std::vector<int> free_ids;
    int num_ids;
};


#endif //HLDA_IDPOOL_H
