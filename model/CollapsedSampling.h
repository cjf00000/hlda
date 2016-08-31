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

    void Estimate();

private:
    void SampleZ(Document &doc);

    void SampleC();

    void InitializeTreeWeight();

    void DFSSample(Document &doc) override;

    double Perplexity();

    void Check();

    TProb WordScore(Document &doc, int l, int topic);

    void UpdateDocCount(Document &doc, int delta);

    std::vector<TCount> ck;
};


#endif //HLDA_COLLAPSEDSAMPLING_H
