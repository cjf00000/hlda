//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_COLLAPSEDSAMPLING_H
#define HLDA_COLLAPSEDSAMPLING_H

#include "BaseHLDA.h"

class CollapsedSampling : public BaseHLDA {
public:
    CollapsedSampling(Corpus &corpus, int L,
                      TProb alpha, TProb beta, TProb gamma, int num_iters);

    void Initialize() override;

    void ProgressivelyOnlineInitialize();

    virtual void Estimate();

protected:
    virtual void SampleZ(Document &doc, bool decrease_count);

    virtual void SampleC(Document &doc, bool decrease_count);

    void DFSSample(Document &doc) override;

    double Perplexity();

    void Check();

    void UpdateDocCount(Document &doc, int delta);

    virtual TProb WordScore(Document &doc, int l, int topic, Tree::Node *node);

    std::vector<TCount> ck;
};


#endif //HLDA_COLLAPSEDSAMPLING_H
