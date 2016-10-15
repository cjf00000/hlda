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
             std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<TProb> gamma,
             int num_iters, int mc_samples);

    virtual void Initialize();

    void Visualize(std::string fileName, int threshold = -1);

    std::string TopWords(int id);

protected:
    void UpdateCount(size_t end = (size_t) -1);

    // Required by SampleC
    virtual void DFSSample(Document &doc);

    virtual TProb WordScore(Document &doc, int l, int topic, Tree::Node *node) = 0;

    virtual void InitializeTreeWeight();

    Tree tree;
    Corpus &corpus;
    int L;
    std::vector<TProb> alpha;
    double alpha_bar;
    std::vector<TProb> beta;        // Beta for each layer
    std::vector<TProb> gamma;
    int num_iters, mc_samples;
    xorshift generator;

    std::vector<Document> docs;
    Matrix<TProb> phi;
    Matrix<TProb> log_phi;
    Matrix<TCount> count;
};


#endif //HLDA_BASEHLDA_H
