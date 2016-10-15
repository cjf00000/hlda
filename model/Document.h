//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_DOCUMENT_H
#define HLDA_DOCUMENT_H

#include <vector>
#include "types.h"
#include "Tree.h"

struct Document {
    Path c;
    std::vector<TTopic> z;
    std::vector<TWord> w;

    std::vector<TProb> theta;

    std::vector<TWord> reordered_w;
    std::vector<TLen> offsets;

    std::vector<TTopic> GetIDs();

    bool initialized;

    void PartitionWByZ(int L);

    void Check();

    TWord *BeginLevel(int l) { return reordered_w.data() + offsets[l]; }

    TWord *EndLevel(int l) { return reordered_w.data() + offsets[l + 1]; }
};

#endif //HLDA_DOCUMENT_H
