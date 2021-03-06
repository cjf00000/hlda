//
// Created by jianfei on 16-10-20.
//

#ifndef HLDA_INSTANTIATEDWEIGHTSAMPLING_H
#define HLDA_INSTANTIATEDWEIGHTSAMPLING_H

#include "PartiallyCollapsedSampling.h"

class InstantiatedWeightSampling : public PartiallyCollapsedSampling {
public:
    InstantiatedWeightSampling(Corpus &corpus, int L,
                               std::vector<TProb> alpha, std::vector<TProb> beta,
                               std::vector<TProb> gamma,
                               int num_iters, int mc_samples, int mc_iters,
                               size_t minibatch_size, int topic_limit,
                               int threshold, int branching_factor, bool sample_phi);

    void Estimate() override;

private:
    virtual void SamplePhi() override;

    virtual void InitializeTreeWeight() override;

    virtual std::vector<TProb>
    WordScore(Document &doc, int l, int num_instantiated, int num_collapsed) override;

    int branching_factor;

    std::vector<int> num_nontrivial_nodes;
};


#endif //HLDA_INSTANTIATEDWEIGHTSAMPLING_H
