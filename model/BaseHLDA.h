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
#include "Document.h"
#include "DirichletParameter.h"

class Corpus;

class BaseHLDA {

public:
    BaseHLDA(Corpus &corpus, int L,
             TProb alpha, TProb beta, TProb gamma,
             int num_iters);

    virtual void Initialize();

    void Visualize(std::string fileName, int threshold = -1);

    std::string TopWords(int id);

protected:
    void UpdateCount(size_t end = (size_t) -1);

    // Required by SampleC
    virtual void DFSSample(Document &doc);

    virtual TProb WordScore(Document &doc, int l, int topic) = 0;

    Tree tree;
    Corpus &corpus;
    int L;
    TProb alpha;
    DirichletParameter beta;
    TProb gamma;
    int num_iters;
    xorshift generator;

    std::vector<Document> docs;
    Matrix<TProb> phi;
    Matrix<TProb> log_phi;
    Matrix<TCount> count;
};


#endif //HLDA_BASEHLDA_H
