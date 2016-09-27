//
// Created by jianfei on 9/2/16.
//

#ifndef HLDA_IDPOOL_H
#define HLDA_IDPOOL_H

#include <vector>
#include <set>

class IDPool {
public:
    IDPool() { num_ids = 0; }

    int Allocate() {
        int id;
        if (!free_ids.empty()) {
            id = free_ids.back();
            free_ids.pop_back();

        } else {
            id = num_ids++;
        }
        allocated_ids.insert(id);
        return id;
    }

    void Free(int id) {
        allocated_ids.erase(id);
        free_ids.push_back(id);
    }

    int Size() { return num_ids; }

    bool Has(int id) {
        return allocated_ids.find(id) != allocated_ids.end();
    }

private:
    std::vector<int> free_ids;
    std::set<int> allocated_ids;
    int num_ids;
};


#endif //HLDA_IDPOOL_H
