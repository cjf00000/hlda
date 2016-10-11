//
// Created by jianfei on 9/19/16.
//

#ifndef HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
#define HLDA_PARTIALLYCOLLAPSEDSAMPLING_H

#include "CollapsedSampling.h"

class PartiallyCollapsedSampling : public CollapsedSampling {
public:
    PartiallyCollapsedSampling(Corpus &corpus, int L,
                               TProb alpha, TProb beta, std::vector<TProb> gamma, int num_iters);

    void Initialize() override;

    void Estimate() override;

private:
    void SampleC(Document &doc, bool decrease_count, bool increase_count) override;

    void SampleZ(Document &doc, bool decrease_count, bool increase_count) override;

    void SamplePhi();

    TProb WordScore(Document &doc, int l, int topic, Tree::Node *node) override;
};


#endif //HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
