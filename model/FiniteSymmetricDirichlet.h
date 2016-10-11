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
                             TProb alpha, TProb beta, std::vector<TProb> gamma,
                             int branching_factor, int num_iters);

    void Initialize() override;

    void Estimate();

    void LayerwiseInitialize(FiniteSymmetricDirichlet &model);

private:
    void SampleC(bool clear_doc_count,
                 size_t d_start = (size_t) -1,
                 size_t d_end = (size_t) -1);

    void SampleCCollapseZ(bool clear_doc_count,
                  size_t d_start = (size_t) -1,
                  size_t d_end = (size_t) -1);

    void SampleZ(Document &doc);

    void SamplePi();

    void SamplePhi();

    void SampleTheta(Document &doc);

    void AddVirtualTree(Tree::Node *node);

    void InitializeTreeWeight() override;

    TProb WordScore(Document &doc, int l, int topic, Tree::Node *node) override;

    double Perplexity();

    void ProgressivelyOnlineInitialize();

    void InitializeBeta();

    int branching_factor;
};


#endif //HLDA_FINITESYMMETRICDIRICHLET_H
