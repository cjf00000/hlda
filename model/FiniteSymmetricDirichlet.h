//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_FINITESYMMETRICDIRICHLET_H
#define HLDA_FINITESYMMETRICDIRICHLET_H

#include "BaseHLDA.h"

// Finite Symmetric Dirichlet approximation
class FiniteSymmetricDirichlet : public BaseHLDA {
public:
    FiniteSymmetricDirichlet(Corpus &corpus, int L,
                             TProb alpha, TProb beta, TProb gamma,
                             int branching_factor, int num_iters);

    void Initialize() override;

    void Estimate();

private:
    void SampleZ(Document &doc);

    void SamplePi();

    void SamplePhi();

    void SampleTheta(Document &doc);

    void AddVirtualTree(Tree::Node *node);

    void InitializeTreeWeight() override;

    TProb WordScore(Document &doc, int l, int topic) override;

    double Perplexity();

    int branching_factor;
};


#endif //HLDA_FINITESYMMETRICDIRICHLET_H
