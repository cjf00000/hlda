//
// Created by jianfei on 9/19/16.
//

#ifndef HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
#define HLDA_PARTIALLYCOLLAPSEDSAMPLING_H

#include "CollapsedSampling.h"

class PartiallyCollapsedSampling : public CollapsedSampling {
public:
    PartiallyCollapsedSampling(Corpus &corpus, int L,
                               std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<TProb> gamma,
                               int num_iters, int mc_samples, int mc_iters, size_t minibatch_size,
                               int topic_limit, int threshold);

    void Initialize() override;

    virtual void Estimate() override;

private:
    void SampleZ(Document &doc, bool decrease_count, bool increase_count) override;

    virtual void SamplePhi();

    size_t minibatch_size;
    int threshold;
};


#endif //HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
