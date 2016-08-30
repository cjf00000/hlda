//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_BASEHLDA_H
#define HLDA_BASEHLDA_H

#include <vector>
#include <string>
#include "Matrix.h"
#include "Tree.h"
#include "xorshift.h"
#include "types.h"

class Corpus;

struct Document {
    Path c;
    std::vector<TTopic> z;
    std::vector<TWord> w;

    std::vector<TProb> theta;

    std::vector<TWord> reordered_w;
    std::vector<TLen> offsets;

    std::vector<TTopic> GetIDs();

    void PartitionWByZ(int L);

    TWord *BeginLevel(int l) { return reordered_w.data() + offsets[l]; }

    TWord *EndLevel(int l) { return reordered_w.data() + offsets[l + 1]; }
};

class BaseHLDA {
public:
    BaseHLDA(Corpus &corpus, int L,
             TProb alpha, TProb beta, TProb gamma,
             int num_iters);

    virtual void Initialize();

    void Visualize(std::string fileName);

    std::string TopWords(int id);

protected:
    void UpdateCount();

    Tree tree;
    Corpus &corpus;
    int L;
    TProb alpha, beta, gamma;
    int num_iters;
    xorshift generator;

    std::vector<Document> docs;
    Matrix<TProb> phi;
    Matrix<TProb> log_phi;
    Matrix<TCount> count;
};


#endif //HLDA_BASEHLDA_H
